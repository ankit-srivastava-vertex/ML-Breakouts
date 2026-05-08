"""Fundamentals + sector/industry cache (yfinance-backed).

Purpose:
  Provide a single parquet cache of per-symbol fundamental snapshots that
  the ML feature builder, sector-index builder and sector-rotation
  pipeline all rely on for sector mapping and `fund_*` features.

How it works:
  * `fetch_all(symbols, workers)` — concurrent `yfinance.Ticker(<sym>.NS).info`
    calls (ThreadPoolExecutor). Each symbol's failure is caught and skipped.
  * `add_cross_sectional_ranks(df)` — appends within-sector and global
    percentile ranks (`fund_*_rank_sector`, `fund_*_rank_all`).
  * `save(df)` / `load()` — read/write `data/fundamentals.parquet`.

Fields captured per symbol:
  sector, industry, marketCap, trailingPE, forwardPE, priceToBook,
  returnOnEquity, debtToEquity, profitMargins, operatingMargins,
  revenueGrowth, earningsGrowth, beta, dividendYield, freeCashflow,
  totalRevenue, totalDebt, totalCash, enterpriseValue,
  enterpriseToRevenue, enterpriseToEbitda.

Data sources:
  yfinance `Ticker(<NSE_SYMBOL>.NS).info`  (the only fundamentals
  source — Angel One SmartAPI does NOT expose fundamentals).

Outputs:
  data/fundamentals.parquet  (one row per symbol, ~22 cols + ranks).

How to run:
  Refreshed by `scripts/06_fetch_fundamentals.py` (quarterly cadence).
  Used by:
    src/sectors.py            (sector → symbol map)
    src/sector_rotation.py    (sector mapping & rotation features)
    src/features.py           (fund_* feature block)
    scripts/02_build_training_set.py / scripts/05_daily_scan.py

Notes:
  * yfinance .info is flaky — expect ~5% of symbols to silently miss
    individual fields. We tolerate NaNs in downstream features.
  * Refresh cadence: quarterly (sector mapping changes rarely).
"""

from __future__ import annotations
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import pandas as pd

from .paths import DATA_DIR
from .yf_ingestion import _to_yf

FUND_PATH = DATA_DIR / "fundamentals.parquet"

FIELDS = [
    "sector", "industry", "marketCap", "trailingPE", "forwardPE",
    "priceToBook", "returnOnEquity", "debtToEquity",
    "profitMargins", "operatingMargins",
    "revenueGrowth", "earningsGrowth",
    "beta", "dividendYield", "freeCashflow",
    "totalRevenue", "totalDebt", "totalCash", "enterpriseValue",
    "enterpriseToRevenue", "enterpriseToEbitda",
]


def _fetch_one(symbol: str, max_retries: int = 3) -> dict | None:
    import yfinance as yf
    for attempt in range(max_retries):
        try:
            info = yf.Ticker(_to_yf(symbol)).info
            if not info or "symbol" not in info:
                return None
            row = {"symbol": symbol}
            for f in FIELDS:
                row[f] = info.get(f)
            return row
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "too many" in msg or "429" in msg:
                time.sleep(2.0 * (attempt + 1))
                continue
            return None
    return None


def fetch_all(symbols: Iterable[str], workers: int = 4,
              verbose: bool = True,
              save_every: int = 200) -> pd.DataFrame:
    """Fetch fundamentals; saves a partial parquet every `save_every`
    successes so a rate-limit-induced exit doesn't lose work."""
    syms = sorted(set(s.upper() for s in symbols))
    rows = []
    t0 = time.time()
    last_save = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, s): s for s in syms}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                rows.append(r)
            done += 1
            if verbose and done % 100 == 0:
                print(f"  {done}/{len(syms)} fetched, ok={len(rows)}, "
                      f"elapsed={time.time() - t0:.0f}s")
            if len(rows) - last_save >= save_every:
                _save_partial(rows)
                last_save = len(rows)
    df = pd.DataFrame(rows)
    # Coerce all numeric fields (yfinance occasionally returns "Infinity"
    # strings or other junk that breaks parquet)
    numeric_fields = [f for f in FIELDS if f not in ("sector", "industry")]
    for c in numeric_fields:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if verbose:
        print(f"  fundamentals: {len(df)}/{len(syms)} ok in "
              f"{time.time() - t0:.0f}s")
    return df


def save(df: pd.DataFrame, path: Path = FUND_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _save_partial(rows: list) -> None:
    """Periodic checkpoint of in-progress fetch."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    numeric_fields = [f for f in FIELDS if f not in ("sector", "industry")]
    for c in numeric_fields:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    save(df)


def load(path: Path = FUND_PATH) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


# ─── derived features ───────────────────────────────────────────────────────

def add_cross_sectional_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Within each sector, compute percentile rank for each numeric metric.

    These ranks are stable across regimes (unlike absolute values which
    drift with the market) — much better features for ML.
    """
    out = df.copy()
    rank_cols = [
        "marketCap", "trailingPE", "priceToBook", "returnOnEquity",
        "debtToEquity", "profitMargins", "operatingMargins",
        "revenueGrowth", "earningsGrowth",
    ]
    for c in rank_cols:
        if c not in out.columns:
            continue
        # Coerce to numeric (yfinance sometimes returns strings or None)
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[f"{c}_rank_sector"] = out.groupby("sector")[c].rank(pct=True)
        out[f"{c}_rank_all"] = out[c].rank(pct=True)
    return out
