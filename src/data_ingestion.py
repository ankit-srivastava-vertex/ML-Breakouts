"""Data-ingestion helpers (post-bhavcopy era).

Purpose:
  Provide the two "non-OHLCV" data utilities every pipeline step needs:
    1. derive the active NSE-EQ universe, and
    2. fetch the Nifty 50 (^NSEI) benchmark close series.

How it works:
  * `derive_universe_from_ohlcv_cache(min_trading_days, recent_window_days,
     min_recent_days)` — walks `data/ohlcv/*.parquet`, reads each file's
     index, applies length / recency filters and returns sorted symbols.
  * `derive_universe_from_bhavcopies` — back-compat alias for callers
     written before the bhavcopy code path was removed.
  * `list_cached_symbols()` — fast `glob` over OHLCV parquets.
  * `fetch_benchmark(start, end)` — single-ticker yfinance call for
     ^NSEI; safe (rate-limit free) and writes to
     `data/benchmark_NIFTY.parquet`.

Data sources:
  * data/ohlcv/*.parquet                  (universe derivation)
  * yfinance ^NSEI                         (benchmark fetch only)

Outputs:
  * list[str] universes (returned)
  * data/benchmark_NIFTY.parquet           (single Close column, DatetimeIndex)

How to run:
  Import-only.
      from src.data_ingestion import derive_universe_from_ohlcv_cache,
                                     fetch_benchmark
      syms = derive_universe_from_ohlcv_cache(60, 30, 15)
      bench = fetch_benchmark()

History:
  Originally wrapped `archives.nseindia.com` bhavcopy CSVs for both
  the universe and a quick OHLCV bootstrap. WAF blocking + the move to
  Angel One in May 2026 made that path obsolete; ~190 lines of legacy
  bhavcopy code were dropped. The function name
  `derive_universe_from_bhavcopies` is preserved as an alias only.
"""

from __future__ import annotations
import datetime
import warnings

import pandas as pd

from .paths import OHLCV_DIR

warnings.filterwarnings("ignore")


# ─── Universe derivation (from the cached OHLCV parquets) ──────────────────

def derive_universe_from_ohlcv_cache(
    min_trading_days: int = 200,
    recent_window_days: int = 90,
    min_recent_days: int = 30,
) -> list[str]:
    """Derive NSE-EQ universe from the existing per-symbol OHLCV cache.

    A symbol is kept if:
      - its parquet has >= `min_trading_days` total bars (excludes very
        new IPOs)
      - it has >= `min_recent_days` bars in the last
        `recent_window_days` calendar days (excludes delisted /
        suspended names)
    """
    files = sorted(OHLCV_DIR.glob("*.parquet"))
    if not files:
        return []

    today = pd.Timestamp(datetime.date.today())
    cutoff = today - pd.Timedelta(days=recent_window_days)

    keep: list[str] = []
    for p in files:
        try:
            df = pd.read_parquet(p, columns=["Close"])
        except Exception:
            continue
        if len(df) < min_trading_days:
            continue
        recent = df.index >= cutoff
        if int(recent.sum()) < min_recent_days:
            continue
        keep.append(p.stem.upper())
    return sorted(keep)


# Back-compat alias (older callers may still import the old name)
derive_universe_from_bhavcopies = derive_universe_from_ohlcv_cache


def list_cached_symbols() -> list[str]:
    return sorted(p.stem for p in OHLCV_DIR.glob("*.parquet"))


# ─── Benchmark ──────────────────────────────────────────────────────────────

def fetch_benchmark(start: datetime.date,
                    end: datetime.date | None = None) -> pd.Series:
    """Nifty 50 close history via yfinance (single ticker, no WAF risk)."""
    import yfinance as yf
    end = end or datetime.date.today()
    df = yf.download("^NSEI", start=start.isoformat(),
                     end=(end + datetime.timedelta(days=1)).isoformat(),
                     progress=False, auto_adjust=False, threads=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].dropna()
