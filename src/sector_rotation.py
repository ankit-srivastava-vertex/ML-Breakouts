"""Self-contained sector / theme rotation pipeline (Angel One backed).

Purpose:
  Build the daily sector-rotation snapshot (per-sector returns, breadth,
  RS vs Nifty, momentum score, rank, rank deltas) and the sector
  leaderboard. Outputs feed both human chart-review (top sectors get
  prioritized) and the ML feature pipeline
  (`symbol_rotation_features` joined into `make_features`).

How it works (everything lives in this single file):
  1. fetch_fundamentals(symbols)   → sector mapping (yfinance cache)
  2. download_ohlcv(symbols)       → daily OHLCV (Angel One via
                                     legacy_scanner/data_provider.py)
  3. download_benchmark()          → ^NSEI Nifty 50 close (Angel One)
  4. build_sector_indices()        → equal-weight per-sector index
  5. compute_rotation()            → returns / RS / breadth / rank /
                                     rank deltas / mom_score
  6. leaderboard()                 → pretty-print top / bottom / rotating-in
  7. run_full_pipeline()           → orchestrates 1–6 with refresh flags

Why a self-contained module?
  Avoids circular imports with `src.sectors` / `src.fundamentals` /
  `src.yf_ingestion` so the rotation pipeline can be invoked from
  notebooks, tests, and `scripts/08_sector_rotation.py` independently.
  The sector-index algorithm is byte-equivalent to `src.sectors`
  (verified by `tests/test_sector_rotation_match.py`).

Data sources:
  * Angel One SmartAPI — OHLCV (stocks + ^NSEI)
  * yfinance         — fundamentals / sector mapping (Angel doesn't
                       expose sector metadata)
  * On-disk caches reused if present:
      data/fundamentals.parquet, data/ohlcv/*.parquet,
      data/benchmark_NIFTY.parquet, data/sector_indices.parquet

Outputs:
  Returns a wide `rot` DataFrame (index = sector, ~16 cols).
  Persisted by the script wrapper to:
    Output/sector_rotation_<YYYYMMDD>.csv  (dated archive)
    Output/sector_rotation.parquet         (always-latest, fixed name)

How to run:
  Programmatically:
      from src.sector_rotation import run_full_pipeline, leaderboard
      rot = run_full_pipeline()             # force-refresh by default
      leaderboard(rot, n=5)
  CLI driver: `python scripts/08_sector_rotation.py --no-refresh`

Notes:
  * `symbol_rotation_features(symbol, rotation_snapshot)` returns a flat
    dict (`rot_sector_*` keys) for joining into ML feature rows.
  * Refreshes are idempotent on the disk cache; pass
    `force_refresh_ohlcv=False` to reuse parquets.
  * `compute_rotation()` accepts an optional `sym_to_sec` map so the
    same engine can be reused for any group definition (this is what
    `src/theme_rotation.py` exploits for custom thematic baskets read
    from `src/index_constituents.json`).
"""

from __future__ import annotations
import sys
import time
import datetime as dt
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "Output"
OHLCV_DIR = DATA_DIR / "ohlcv"
FUND_PATH = DATA_DIR / "fundamentals.parquet"
BENCH_PATH = DATA_DIR / "benchmark_NIFTY.parquet"
SECTOR_INDEX_PATH = DATA_DIR / "sector_indices.parquet"
ROTATION_PATH = OUTPUT_DIR / "sector_rotation.parquet"

for _d in (DATA_DIR, OHLCV_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── Angel-One-backed OHLCV provider ──────────────────────────────────────
# data_provider lives in legacy_scanner/ and chains Angel→jugaad→yfinance.
_LEGACY_DIR = ROOT / "legacy_scanner"
if str(_LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(_LEGACY_DIR))

from data_provider import download as _provider_download  # noqa: E402


# ─── yfinance ticker helpers ──────────────────────────────────────────────

def _to_yf(symbol: str) -> str:
    return f"{symbol.upper()}.NS"


# ─── 1. Fundamentals (sector mapping) ─────────────────────────────────────

FUND_FIELDS = [
    "sector", "industry", "marketCap", "trailingPE", "forwardPE",
    "priceToBook", "returnOnEquity", "debtToEquity",
    "profitMargins", "operatingMargins",
    "revenueGrowth", "earningsGrowth",
    "beta", "dividendYield", "freeCashflow",
    "totalRevenue", "totalDebt", "totalCash", "enterpriseValue",
    "enterpriseToRevenue", "enterpriseToEbitda",
]


def _fetch_fund_one(symbol: str, max_retries: int = 3) -> Optional[dict]:
    import yfinance as yf
    for attempt in range(max_retries):
        try:
            info = yf.Ticker(_to_yf(symbol)).info
            if not info or "symbol" not in info:
                return None
            row = {"symbol": symbol}
            for f in FUND_FIELDS:
                row[f] = info.get(f)
            return row
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "too many" in msg or "429" in msg:
                time.sleep(2.0 * (attempt + 1))
                continue
            return None
    return None


def fetch_fundamentals(symbols: Iterable[str], workers: int = 4,
                       verbose: bool = True) -> pd.DataFrame:
    """Fetch sector + fundamentals from yfinance for a list of symbols."""
    syms = sorted(set(s.upper() for s in symbols))
    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_fund_one, s): s for s in syms}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                rows.append(r)
            done += 1
            if verbose and done % 100 == 0:
                print(f"  fundamentals: {done}/{len(syms)}, ok={len(rows)}, "
                      f"elapsed={time.time() - t0:.0f}s")
    df = pd.DataFrame(rows)
    numeric_fields = [f for f in FUND_FIELDS if f not in ("sector", "industry")]
    for c in numeric_fields:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if verbose:
        print(f"  fundamentals: {len(df)}/{len(syms)} ok in "
              f"{time.time() - t0:.0f}s")
    return df


def load_fundamentals() -> Optional[pd.DataFrame]:
    """Load cached fundamentals; returns None if absent."""
    if not FUND_PATH.exists():
        return None
    return pd.read_parquet(FUND_PATH)


def symbol_to_sector_map(fund: Optional[pd.DataFrame] = None) -> dict[str, str]:
    if fund is None:
        fund = load_fundamentals()
    if fund is None:
        return {}
    return dict(zip(fund["symbol"], fund["sector"].fillna("Unknown")))


# ─── 2. OHLCV (Angel One via data_provider) ───────────────────────────────

def _write_symbol_parquet(symbol: str, df: pd.DataFrame) -> bool:
    sub = df.reindex(columns=["Open", "High", "Low", "Close", "Volume"]).dropna()
    if sub.empty:
        return False
    sub.index = pd.to_datetime(sub.index).normalize()
    sub.index.name = "DATE"
    sub.to_parquet(OHLCV_DIR / f"{symbol.upper()}.parquet")
    return True


def download_ohlcv(symbols: Iterable[str],
                   start: dt.date,
                   end: Optional[dt.date] = None,
                   batch_size: int = 50,
                   chunk_days: int = 1800,
                   pause_between_batches: float = 0.0,
                   angel_only: bool = True,
                   verbose: bool = True) -> int:
    """Bulk-download daily OHLCV via Angel One (with jugaad/yfinance fallback)
    and write per-symbol parquet to data/ohlcv/.

    Angel One's historical-candle endpoint caps each request at ~2 000 days.
    For longer windows (e.g. 2018→today is ~3 050 days) we split the date
    range into `chunk_days`-sized windows and concatenate per ticker before
    writing the parquet.

    When `angel_only=True` (default), we set data_provider.ANGEL_ONLY=True
    for the duration of this call so symbols Angel can't resolve are simply
    skipped (no slow yfinance fallback). Symbols that don't exist in
    Angel's master are reported as "missing" rather than being silently
    backfilled by another provider.
    """
    end = end or dt.date.today()
    syms = sorted(set(s.upper() for s in symbols))

    # Build date-range chunks (≤ chunk_days each)
    chunks: list[tuple[dt.date, dt.date]] = []
    cur = start
    while cur <= end:
        ce = min(end, cur + dt.timedelta(days=chunk_days - 1))
        chunks.append((cur, ce))
        cur = ce + dt.timedelta(days=1)
    if verbose:
        print(f"  Angel One bulk download: {len(syms)} symbols, "
              f"{start} → {end}, batch={batch_size}, chunks={len(chunks)}, "
              f"angel_only={angel_only}")

    # Toggle data_provider's fallback chain
    import data_provider as _dp
    _prev_angel_only = _dp.ANGEL_ONLY
    if angel_only:
        _dp.ANGEL_ONLY = True

    n_ok, n_total = 0, len(syms)
    t0 = time.time()
    try:
        for batch_idx in range(0, n_total, batch_size):
            batch = syms[batch_idx:batch_idx + batch_size]
            yf_tickers = [_to_yf(s) for s in batch]
            per_ticker_parts: dict[str, list[pd.DataFrame]] = {s: [] for s in batch}
            for cs, ce in chunks:
                try:
                    df = _provider_download(
                        yf_tickers, start=cs, end=ce, progress=False,
                        threads=False, group_by="column",
                    )
                except Exception as e:
                    if verbose:
                        print(f"    chunk err [{cs}→{ce}]: {e}")
                    continue
                if df is None or df.empty:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    for s, yt in zip(batch, yf_tickers):
                        try:
                            sub = df.xs(yt, axis=1, level=1)
                        except KeyError:
                            continue
                        if sub is not None and not sub.empty:
                            per_ticker_parts[s].append(sub)
                else:
                    # single-ticker shape (only when batch has 1 element)
                    per_ticker_parts[batch[0]].append(df)
            # Stitch chunks per ticker and persist
            for s, parts in per_ticker_parts.items():
                parts = [p for p in parts if p is not None and not p.empty]
                if not parts:
                    continue
                merged = pd.concat(parts).sort_index()
                merged = merged[~merged.index.duplicated(keep="last")]
                if _write_symbol_parquet(s, merged):
                    n_ok += 1
            if verbose and (batch_idx // batch_size) % 5 == 0:
                done = min(batch_idx + batch_size, n_total)
                print(f"    {done}/{n_total} done, ok={n_ok}, "
                      f"elapsed={time.time() - t0:.0f}s")
            if pause_between_batches:
                time.sleep(pause_between_batches)
    finally:
        _dp.ANGEL_ONLY = _prev_angel_only
    if verbose:
        print(f"  OHLCV: {n_ok}/{n_total} OK in {time.time() - t0:.0f}s")
    return n_ok


def load_symbol(symbol: str) -> Optional[pd.DataFrame]:
    """Load cached OHLCV for one symbol."""
    p = OHLCV_DIR / f"{symbol.upper()}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


# ─── 3. Benchmark (Nifty 50 ^NSEI) — Angel One ────────────────────────────

def download_benchmark(start: dt.date = dt.date(2018, 1, 1),
                       end: Optional[dt.date] = None) -> pd.DataFrame:
    """Download ^NSEI Nifty 50 daily close via Angel One (data_provider
    fallback chain) and cache to data/benchmark_NIFTY.parquet."""
    end = end or dt.date.today()
    df = _provider_download("^NSEI", start=start, end=end, progress=False)
    if df is None or df.empty:
        raise RuntimeError(
            "Failed to fetch ^NSEI (Angel/jugaad/yfinance all returned empty)"
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reindex(columns=["Open", "High", "Low", "Close", "Volume"]).dropna()
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "DATE"
    df.to_parquet(BENCH_PATH)
    return df


def load_benchmark() -> Optional[pd.Series]:
    """Return Nifty Close series; None if cache missing."""
    if not BENCH_PATH.exists():
        return None
    b = pd.read_parquet(BENCH_PATH)["Close"]
    b.index = pd.to_datetime(b.index).normalize()
    return b


# ─── 4. Sector indices (equal-weight) ─────────────────────────────────────

def build_sector_indices(min_members: int = 10,
                         save: bool = True) -> pd.DataFrame:
    """Equal-weight cumulative-return index per sector.

    Identical algorithm to src.sectors.build_sector_indices() so output
    matches data/sector_indices.parquet byte-for-byte given the same
    inputs.

    Returns wide DataFrame: index=DATE, columns=sector, values=index level (base 100).
    """
    fund = load_fundamentals()
    if fund is None or fund.empty:
        raise RuntimeError(
            "No fundamentals cache. Call fetch_fundamentals(symbols) first, "
            "or run scripts/06_fetch_fundamentals.py."
        )
    by_sector = fund.dropna(subset=["sector"]).groupby("sector")["symbol"].apply(list)

    out: dict[str, pd.Series] = {}
    for sector, syms in by_sector.items():
        if len(syms) < min_members:
            continue
        rets = []
        for s in syms:
            df = load_symbol(s)
            if df is None or len(df) < 250:
                continue
            r = df["Close"].pct_change()
            rets.append(r)
        if len(rets) < min_members:
            continue
        ret_df = pd.concat(rets, axis=1).fillna(0.0)
        idx = (1.0 + ret_df.mean(axis=1)).cumprod() * 100
        out[sector] = idx

    if not out:
        raise RuntimeError("No sector indices built — OHLCV cache empty?")
    wide = pd.concat(out, axis=1)
    wide.index = pd.to_datetime(wide.index).normalize()
    if save:
        SECTOR_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        wide.to_parquet(SECTOR_INDEX_PATH)
        print(f"  built {wide.shape[1]} sector indices, "
              f"{wide.index.min().date()} → {wide.index.max().date()}")
    return wide


def load_sector_indices() -> Optional[pd.DataFrame]:
    if not SECTOR_INDEX_PATH.exists():
        return None
    return pd.read_parquet(SECTOR_INDEX_PATH)


# ─── 5. Rotation analytics ────────────────────────────────────────────────

# Composite-momentum weights (sum to 1.0); 3m gets the most.
MOMENTUM_WEIGHTS = {"r_1m": 0.15, "r_3m": 0.40, "r_6m": 0.30, "r_12m": 0.15}


def _rotation_core(asof: pd.Timestamp,
                   sector_idx: pd.DataFrame,
                   bench: Optional[pd.Series]) -> pd.DataFrame:
    """Compute returns, RS vs bench, momentum z-score, rank.
    Used both for the asof snapshot and historical comparison points."""
    sec_idx = sector_idx.sort_index().loc[:asof]
    if sec_idx.empty:
        return pd.DataFrame()
    rows = []
    for sector in sec_idx.columns:
        s = sec_idx[sector].dropna()
        if len(s) < 260:
            continue
        row = {"sector": sector}
        for d, lbl in [(20, "1m"), (60, "3m"), (130, "6m"), (260, "12m")]:
            row[f"r_{lbl}"] = float((s.iloc[-1] / s.iloc[-d - 1] - 1) * 100) \
                if len(s) > d else np.nan
        s50 = s.rolling(50).mean()
        s200 = s.rolling(200).mean()
        row["above_50dma"] = float(s.iloc[-1] > s50.iloc[-1]) \
            if not pd.isna(s50.iloc[-1]) else np.nan
        row["above_200dma"] = float(s.iloc[-1] > s200.iloc[-1]) \
            if not pd.isna(s200.iloc[-1]) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows).set_index("sector")
    if bench is not None and not bench.empty:
        b = bench.loc[:asof]
        for d, lbl in [(20, "1m"), (60, "3m"), (130, "6m"), (260, "12m")]:
            br = float((b.iloc[-1] / b.iloc[-d - 1] - 1) * 100) \
                if len(b) > d else np.nan
            df[f"rs_{lbl}"] = df[f"r_{lbl}"] - br
    else:
        for lbl in ["1m", "3m", "6m", "12m"]:
            df[f"rs_{lbl}"] = df[f"r_{lbl}"]
    z = pd.DataFrame(index=df.index)
    for lbl in ["1m", "3m", "6m", "12m"]:
        v = df[f"rs_{lbl}"]
        z[f"rs_{lbl}"] = (v - v.mean()) / v.std() if v.std() > 0 else 0.0
    df["mom_score"] = sum(MOMENTUM_WEIGHTS[f"r_{lbl}"] * z[f"rs_{lbl}"].fillna(0.0)
                          for lbl in ["1m", "3m", "6m", "12m"])
    df["rank"] = (df["mom_score"]
                  .rank(ascending=False, method="min")
                  .fillna(-1)
                  .astype(int))
    return df


def compute_rotation(asof: Optional[pd.Timestamp] = None,
                     sector_idx: Optional[pd.DataFrame] = None,
                     bench: Optional[pd.Series] = None,
                     compute_breadth: bool = True,
                     sym_to_sec: Optional[dict] = None) -> pd.DataFrame:
    """Full rotation snapshot at `asof` (default = latest).

    Returns one row per sector with returns, RS, mom_score, rank,
    rank deltas (5d/20d) and member breadth above 50/200 DMA.

    `sym_to_sec` lets callers (e.g. theme rotation) supply their own
    {symbol -> group-name} map for breadth computation. When None, falls
    back to the yfinance fundamentals sector map.
    """
    if sector_idx is None:
        sector_idx = load_sector_indices()
    if sector_idx is None or sector_idx.empty:
        raise RuntimeError("No sector_indices.parquet — run build_sector_indices().")
    if bench is None:
        bench = load_benchmark()

    sector_idx = sector_idx.sort_index()
    if asof is None:
        asof = sector_idx.index.max()
    df = _rotation_core(asof, sector_idx, bench)

    # Rank delta vs 5d / 20d ago
    for lookback, lbl in [(5, "5d"), (20, "20d")]:
        try:
            past = _rotation_core(
                asof - pd.Timedelta(days=lookback * 2),  # weekend buffer
                sector_idx, bench,
            )
            # Positive delta = rank improved (e.g. 5 → 2 yields +3)
            df[f"rank_chg_{lbl}"] = past["rank"].reindex(df.index) - df["rank"]
        except Exception:
            df[f"rank_chg_{lbl}"] = np.nan

    if compute_breadth:
        if sym_to_sec is None:
            sym_to_sec = symbol_to_sector_map()
        sec_to_syms: dict[str, list[str]] = {}
        for sym, sec in sym_to_sec.items():
            sec_to_syms.setdefault(sec, []).append(sym)
        breadth_50, breadth_200 = {}, {}
        for sector in df.index:
            members = sec_to_syms.get(sector, [])
            n_50 = n_200 = total = 0
            for sym in members:
                sdf = load_symbol(sym)
                if sdf is None or len(sdf) < 200:
                    continue
                sdf = sdf.sort_index().loc[:asof]
                if len(sdf) < 200:
                    continue
                c = sdf["Close"]
                ma50 = c.rolling(50).mean().iloc[-1]
                ma200 = c.rolling(200).mean().iloc[-1]
                last = c.iloc[-1]
                total += 1
                if last > ma50:
                    n_50 += 1
                if last > ma200:
                    n_200 += 1
            if total >= 5:
                breadth_50[sector] = n_50 / total * 100.0
                breadth_200[sector] = n_200 / total * 100.0
            else:
                breadth_50[sector] = np.nan
                breadth_200[sector] = np.nan
        df["breadth_above_50dma"] = pd.Series(breadth_50)
        df["breadth_above_200dma"] = pd.Series(breadth_200)

    return df.sort_values("rank")


def leaderboard(rot: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    cols = ["rank", "mom_score", "rs_1m", "rs_3m", "rs_6m",
            "rank_chg_5d", "rank_chg_20d",
            "breadth_above_50dma", "breadth_above_200dma"]
    cols = [c for c in cols if c in rot.columns]
    top = rot.head(n)[cols].round(2)
    bot = rot.tail(n)[cols].round(2)
    print(f"\n  ── TOP {n} (LEADERS) ──")
    print(top.to_string())
    print(f"\n  ── BOTTOM {n} (LAGGARDS) ──")
    print(bot.to_string())
    if "rank_chg_5d" in rot.columns:
        rotating_in = rot.nlargest(min(3, len(rot)), "rank_chg_5d")
        print("\n  ── ROTATING IN (rank improving fastest, 5d) ──")
        print(rotating_in[["rank", "rank_chg_5d", "rs_1m", "rs_3m"]].round(2).to_string())
    return top


# ─── 6. Per-symbol rotation features (for ML feature join) ────────────────

def symbol_rotation_features(symbol: str,
                             rotation_snapshot: pd.DataFrame,
                             sym_to_sector: Optional[dict] = None) -> dict:
    """Flat feature dict for one symbol's setup row, joinable into make_features()."""
    if sym_to_sector is None:
        sym_to_sector = symbol_to_sector_map()
    sector = sym_to_sector.get(symbol)
    blank = {
        "rot_sector_rank": np.nan, "rot_sector_mom_score": np.nan,
        "rot_sector_rank_chg_5d": np.nan, "rot_sector_rank_chg_20d": np.nan,
        "rot_sector_breadth_50": np.nan, "rot_sector_breadth_200": np.nan,
        "rot_sector_is_top3": 0.0, "rot_sector_is_bottom3": 0.0,
        "rot_sector_rotating_in": 0.0,
    }
    if sector is None or sector not in rotation_snapshot.index:
        return blank
    row = rotation_snapshot.loc[sector]
    n = len(rotation_snapshot)
    return {
        "rot_sector_rank": float(row["rank"]),
        "rot_sector_mom_score": float(row.get("mom_score", np.nan)),
        "rot_sector_rank_chg_5d": float(row.get("rank_chg_5d", np.nan)),
        "rot_sector_rank_chg_20d": float(row.get("rank_chg_20d", np.nan)),
        "rot_sector_breadth_50": float(row.get("breadth_above_50dma", np.nan)),
        "rot_sector_breadth_200": float(row.get("breadth_above_200dma", np.nan)),
        "rot_sector_is_top3": float(row["rank"] <= 3),
        "rot_sector_is_bottom3": float(row["rank"] >= n - 2),
        "rot_sector_rotating_in": float((row.get("rank_chg_5d", 0) or 0) >= 2),
    }


# ─── 7. End-to-end orchestrator ───────────────────────────────────────────

def run_full_pipeline(symbols: Optional[Iterable[str]] = None,
                      rebuild_indices: bool = True,
                      force_refresh_ohlcv: bool = True,
                      force_refresh_benchmark: bool = True,
                      ohlcv_start: dt.date = dt.date(2018, 1, 1),
                      asof: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """End-to-end: data → indices → rotation snapshot.

    - Fundamentals: re-uses data/fundamentals.parquet if present (yfinance only;
      cheap to keep — sector mapping changes rarely).
    - OHLCV: when `force_refresh_ohlcv=True` (default), every symbol's parquet
      is re-downloaded from Angel One on every run, overwriting the previous
      file. Otherwise only missing symbols are fetched.
    - Benchmark: when `force_refresh_benchmark=True` (default), ^NSEI is
      re-pulled from Angel One every run.
    - If `rebuild_indices=True`, regenerate sector_indices.parquet from the
      now-Angel-sourced OHLCV cache.

    Returns the rotation DataFrame.
    """
    # ─── Fundamentals (yfinance; sector mapping only) ────────────────────
    fund = load_fundamentals()
    if fund is None:
        if symbols is None:
            raise RuntimeError(
                "No fundamentals cache and no `symbols` provided to fetch."
            )
        print("  [1/4] Fetching fundamentals from yfinance ...")
        fund = fetch_fundamentals(symbols)
        FUND_PATH.parent.mkdir(parents=True, exist_ok=True)
        fund.to_parquet(FUND_PATH)
    else:
        print(f"  [1/4] Using cached fundamentals ({len(fund)} rows)")

    # ─── OHLCV (Angel One; force-refresh by default) ─────────────────────
    needed = sorted(set(s.upper() for s in (symbols or fund["symbol"].tolist())))
    if force_refresh_ohlcv:
        print(f"  [2/4] OHLCV: force-refreshing ALL {len(needed)} symbols "
              f"from Angel One (overwrites existing parquets) ...")
        download_ohlcv(needed, start=ohlcv_start)
    else:
        have = {p.stem.upper() for p in OHLCV_DIR.glob("*.parquet")}
        missing = sorted(set(needed) - have)
        if missing:
            print(f"  [2/4] OHLCV: {len(missing)} symbols missing — "
                  f"downloading from Angel One ...")
            download_ohlcv(missing, start=ohlcv_start)
        else:
            print(f"  [2/4] OHLCV cache complete ({len(have)} symbols, "
                  f"force_refresh=False)")

    # ─── Benchmark (Angel One ^NSEI) ─────────────────────────────────────
    if force_refresh_benchmark or not BENCH_PATH.exists():
        print("  [3/4] Downloading ^NSEI benchmark from Angel One ...")
        download_benchmark(start=ohlcv_start)
    else:
        print("  [3/4] Using cached ^NSEI benchmark (force_refresh=False)")

    # ─── Sector indices ──────────────────────────────────────────────────
    if rebuild_indices or not SECTOR_INDEX_PATH.exists():
        print("  [4/4] Building sector indices from refreshed OHLCV ...")
        build_sector_indices()
    else:
        print("  [4/4] Using cached sector indices")

    return compute_rotation(asof=asof)
