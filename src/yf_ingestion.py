"""OHLCV downloader for NSE-EQ symbols (Angel One backed).

Purpose:
  Maintain `data/ohlcv/<SYMBOL>.parquet` — the canonical per-symbol
  daily OHLCV cache used by every downstream stage of the pipeline.

How it works:
  * `download_history(symbols, start, end, batch_size, pause, angel_only)`
      Bulk historical fetch. Chunks the date window at 1800 days (Angel
      One's max per request), batches symbols, accumulates per-ticker,
      and writes one parquet per symbol with columns
      [Open, High, Low, Close, Volume] and DatetimeIndex named DATE.
  * `update_history(symbols, lookback_days, batch_size, angel_only)`
      Incremental tail fetch (no chunking) — appended/merged with the
      cached history. Used by `scripts/05_daily_scan.py`.
  * `_provider_fetch_batch(...)` calls
      `legacy_scanner/data_provider.download(...)` which prefers Angel
      One SmartAPI and falls back to jugaad / yfinance only when
      `ANGEL_ONLY=False`. Tickers are passed in `<SYM>.NS` form and the
      MultiIndex is split via `df.xs(yt, axis=1, level=1)`.
  * `load_symbol(sym)` returns the cached parquet (sorted, index name
      normalised). `_to_yf` / `_from_yf` translate between local NSE
      symbols and `yfinance` `<SYM>.NS` form.

Data sources:
  * Angel One SmartAPI (primary)
  * jugaad-data, yfinance (fallback chain when `angel_only=False`)

Outputs:
  data/ohlcv/<SYMBOL>.parquet  (overwritten on refresh, merged on update)

How to run:
  Import-only — driven by:
    scripts/01_build_dataset.py    bulk historical bootstrap
    scripts/05_daily_scan.py       daily tail-update before scanning
    src.sector_rotation            also calls `data_provider.download` directly

Why we still keep yfinance-style helpers:
  `src.fundamentals` needs `yf.Ticker(<sym>.NS).info` for sector/mcap data
  (Angel One does NOT expose fundamentals). Keeping `_to_yf` here is the
  cheapest way to share that ticker-formatting code.
"""

from __future__ import annotations
import sys
import time
import warnings
import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .paths import OHLCV_DIR

warnings.filterwarnings("ignore")

# ─── Wire in the Angel-One-backed provider ────────────────────────────────
# data_provider.download(...) returns a yfinance-shaped DataFrame
# (MultiIndex columns: level-0=field, level-1=ticker for multi-ticker;
#  flat OHLCV for single-ticker). Same shape as yf.download(group_by="ticker").
_LEGACY_DIR = Path(__file__).resolve().parents[1] / "legacy_scanner"
if str(_LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(_LEGACY_DIR))
try:
    from data_provider import download as _provider_download  # type: ignore
    import data_provider as _dp                                # type: ignore
except Exception as _e:
    _provider_download = None
    _dp = None
    warnings.warn(f"data_provider import failed: {_e!r}; falling back to yfinance")


def _to_yf(symbol: str) -> str:
    """NSE symbol → Yahoo ticker (still used by fundamentals.py)."""
    return f"{symbol.upper()}.NS"


def _from_yf(yf_ticker: str) -> str:
    return yf_ticker.upper().replace(".NS", "").replace(".BO", "")


# Angel-One historical-candle endpoint caps a single request at ~2000 days.
# 8y * 252 trading days ≈ 2016 — too close to the cap. Chunk every 1800
# calendar days to be safe.
_ANGEL_CHUNK_DAYS = 1800


def _date_chunks(start: datetime.date, end: datetime.date,
                 max_days: int = _ANGEL_CHUNK_DAYS):
    """Yield (chunk_start, chunk_end) pairs covering [start, end]."""
    cur = start
    while cur <= end:
        nxt = min(cur + datetime.timedelta(days=max_days - 1), end)
        yield cur, nxt
        cur = nxt + datetime.timedelta(days=1)


def _write_symbol_parquet(symbol: str, df: pd.DataFrame) -> bool:
    """Coerce to canonical schema + write parquet. Returns True on success."""
    if df is None or df.empty:
        return False
    sub = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if sub.empty:
        return False
    sub.index = pd.to_datetime(sub.index).tz_localize(None) if (
        getattr(sub.index, "tz", None) is not None
    ) else pd.to_datetime(sub.index)
    sub.index = sub.index.normalize()
    sub.index.name = "DATE"
    sub = sub[~sub.index.duplicated(keep="last")].sort_index()
    sub.to_parquet(OHLCV_DIR / f"{symbol.upper()}.parquet")
    return True


def _provider_fetch_batch(syms: list[str],
                          start: datetime.date,
                          end: datetime.date) -> dict[str, pd.DataFrame]:
    """Call data_provider.download for a batch and split MultiIndex →
    {symbol: df}. Provider expects `.NS`-suffixed tickers and returns a
    yfinance-shaped MultiIndex with level-0=field, level-1=ticker.
    Handles single-symbol flat-frame case too.
    """
    if _provider_download is None:
        return {}
    yf_tickers = [_to_yf(s) for s in syms]
    df = _provider_download(
        yf_tickers,
        start=start.isoformat(),
        end=(end + datetime.timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=False,
        threads=False,
        group_by="column",
    )
    out: dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out
    if isinstance(df.columns, pd.MultiIndex):
        for s, yt in zip(syms, yf_tickers):
            try:
                sub = df.xs(yt, axis=1, level=1)
            except KeyError:
                continue
            if sub is not None and not sub.empty:
                out[s] = sub
    else:
        # single-ticker → flat frame
        out[syms[0]] = df
    return out


def download_history(symbols: Iterable[str],
                     start: datetime.date,
                     end: datetime.date | None = None,
                     batch_size: int = 50,
                     pause_between_batches: float = 0.5,
                     angel_only: bool = True) -> int:
    """Bulk-download daily OHLCV for symbols, write per-symbol parquet.

    Returns number of symbols successfully written. Routes through Angel
    One via legacy_scanner/data_provider; chunks date windows above
    Angel's ~2000-day single-request cap.
    """
    end = end or datetime.date.today()
    syms = sorted(set(s.upper() for s in symbols))
    n_total = len(syms)
    print(f"  Angel One bulk download: {n_total} symbols, "
          f"{start} → {end}, batch={batch_size}, "
          f"chunks={sum(1 for _ in _date_chunks(start, end))}, "
          f"angel_only={angel_only}")

    prev_angel_only = getattr(_dp, "ANGEL_ONLY", False) if _dp else None
    if _dp is not None and angel_only:
        _dp.ANGEL_ONLY = True

    n_ok = 0
    t0 = time.time()
    try:
        for batch_idx in range(0, n_total, batch_size):
            batch = syms[batch_idx:batch_idx + batch_size]
            per_ticker_parts: dict[str, list[pd.DataFrame]] = {s: [] for s in batch}
            for c_start, c_end in _date_chunks(start, end):
                try:
                    chunk_map = _provider_fetch_batch(batch, c_start, c_end)
                except Exception as e:
                    print(f"    chunk err {c_start}→{c_end}: {e}")
                    continue
                for s, sdf in chunk_map.items():
                    if sdf is not None and not sdf.empty:
                        per_ticker_parts[s].append(sdf)

            for s, parts in per_ticker_parts.items():
                if not parts:
                    continue
                merged = pd.concat(parts).sort_index()
                merged = merged[~merged.index.duplicated(keep="last")]
                if _write_symbol_parquet(s, merged):
                    n_ok += 1

            done = min(batch_idx + batch_size, n_total)
            elapsed = time.time() - t0
            print(f"    {done}/{n_total} done, ok={n_ok}, "
                  f"elapsed={elapsed:.0f}s")
            time.sleep(pause_between_batches)
    finally:
        if _dp is not None and prev_angel_only is not None:
            _dp.ANGEL_ONLY = prev_angel_only

    print(f"  OHLCV download complete: {n_ok}/{n_total} OK "
          f"in {time.time() - t0:.0f}s")
    return n_ok


def update_history(symbols: Iterable[str], lookback_days: int = 10,
                   batch_size: int = 50,
                   angel_only: bool = True) -> int:
    """Incremental tail update for daily refresh. Reads cached parquet,
    fetches only new bars via Angel One, appends and writes back.

    Tail windows are always small (≤ lookback_days), so no chunking needed.
    """
    end = datetime.date.today()
    syms = sorted(set(s.upper() for s in symbols))
    n_updated = 0

    prev_angel_only = getattr(_dp, "ANGEL_ONLY", False) if _dp else None
    if _dp is not None and angel_only:
        _dp.ANGEL_ONLY = True

    try:
        for batch_idx in range(0, len(syms), batch_size):
            batch = syms[batch_idx:batch_idx + batch_size]
            # Find earliest needed start across this batch
            starts: list[datetime.date] = []
            for s in batch:
                p = OHLCV_DIR / f"{s}.parquet"
                if p.exists():
                    last = pd.read_parquet(p).index.max().date()
                    starts.append(last + datetime.timedelta(days=1))
                else:
                    starts.append(end - datetime.timedelta(days=lookback_days))
            start = min(starts)
            if start > end:
                continue

            try:
                fetched = _provider_fetch_batch(batch, start, end)
            except Exception:
                continue

            for s, new in fetched.items():
                if new is None or new.empty:
                    continue
                cols = ["Open", "High", "Low", "Close", "Volume"]
                new = new[cols].dropna()
                if new.empty:
                    continue
                new.index = pd.to_datetime(new.index)
                if getattr(new.index, "tz", None) is not None:
                    new.index = new.index.tz_localize(None)
                new.index = new.index.normalize()
                new.index.name = "DATE"
                p = OHLCV_DIR / f"{s}.parquet"
                if p.exists():
                    old = pd.read_parquet(p)
                    merged = pd.concat([old, new])
                    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                else:
                    merged = new.sort_index()
                merged.to_parquet(p)
                n_updated += 1
            time.sleep(0.3)
    finally:
        if _dp is not None and prev_angel_only is not None:
            _dp.ANGEL_ONLY = prev_angel_only

    return n_updated


def load_symbol(symbol: str) -> Optional[pd.DataFrame]:
    p = OHLCV_DIR / f"{symbol.upper()}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)

