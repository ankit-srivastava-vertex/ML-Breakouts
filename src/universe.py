"""Universe builder — thin wrapper around the OHLCV-cache derivation.

Purpose:
  Return the list of "active" NSE-EQ symbols that should participate in
  training-set construction and daily scans.

How it works:
  Delegates to `src.data_ingestion.derive_universe_from_ohlcv_cache`,
  which inspects every parquet in `data/ohlcv/` and keeps symbols that:
    - have at least `min_trading_days` total bars on file, AND
    - have at least `min_recent_days` bars in the last
      `recent_window_days` (filters delisted / inactive tickers).

Why not NSE EQUITY_L.csv or bhavcopies?
  Both endpoints (www.nseindia.com and archives.nseindia.com) are now
  Akamai-WAF-blocked for retail IPs. The OHLCV cache itself is the
  authoritative "universe" because it is populated by Angel One.

Data sources:
  data/ohlcv/*.parquet   (file system)

Outputs:
  list[str]  — sorted upper-case symbols.

How to run:
  Import-only.
      from src.universe import build_universe
      syms = build_universe()              # defaults: 200 / 90 / 30
"""

from __future__ import annotations
from .data_ingestion import derive_universe_from_ohlcv_cache


def build_universe(min_trading_days: int = 200,
                   recent_window_days: int = 90,
                   min_recent_days: int = 30) -> list[str]:
    return derive_universe_from_ohlcv_cache(
        min_trading_days=min_trading_days,
        recent_window_days=recent_window_days,
        min_recent_days=min_recent_days,
    )
