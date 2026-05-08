"""Equal-weight sector indices + symbol→sector mapping.

Purpose:
  Build the cached `data/sector_indices.parquet` (one column per yfinance
  sector, daily index level) used by feature engineering and by the
  sector-rotation report, and expose the `symbol → sector` lookup that
  the daily scan and rotation cross-reference rely on.

How it works:
  * `build_sector_indices(min_members)` —
      1. Loads `data/fundamentals.parquet` (sector mapping).
      2. For each sector with >= `min_members` symbols, normalises each
         member's Close to 100 at first observation, then averages across
         members per day to produce an equal-weight cumulative index.
      3. Writes the wide parquet (index = date, columns = sector names).
  * `load_sector_indices()` — read the parquet (returns None if absent).
  * `symbol_to_sector_map()` — dict[symbol, sector] from fundamentals.

Data sources:
  * data/fundamentals.parquet  (sector field per symbol)
  * data/ohlcv/<SYM>.parquet   (closing prices for each sector member)

Outputs:
  data/sector_indices.parquet  (one column per sector, daily index level).

How to run:
  Indirectly via `scripts/06_fetch_fundamentals.py`, which calls
  `build_sector_indices(min_members=10)` after the fundamentals refresh.
  Or import-and-call:
      from src.sectors import build_sector_indices
      build_sector_indices()

Notes:
  * `src.sector_rotation` contains a self-contained re-implementation
    of the same algorithm (used by the rotation pipeline) so the two
    parquets are byte-identical when run on the same OHLCV snapshot.
    `tests/test_sector_rotation_match.py` verifies that equality.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import DATA_DIR, OHLCV_DIR
from .fundamentals import load as load_fund
from .yf_ingestion import load_symbol


SECTOR_INDEX_PATH = DATA_DIR / "sector_indices.parquet"


def build_sector_indices(min_members: int = 10) -> pd.DataFrame:
    """Equal-weight cumulative-return index per sector.

    Returns wide DataFrame: index=date, columns=sector_name, values=index level.
    """
    fund = load_fund()
    if fund is None or fund.empty:
        raise RuntimeError("No fundamentals — run scripts/06_fetch_fundamentals.py first.")
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
        # equal-weight daily mean return → cumulative index
        idx = (1.0 + ret_df.mean(axis=1)).cumprod() * 100
        out[sector] = idx

    if not out:
        raise RuntimeError("No sector indices built.")
    wide = pd.concat(out, axis=1)
    wide.index = pd.to_datetime(wide.index).normalize()
    SECTOR_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(SECTOR_INDEX_PATH)
    print(f"  built {wide.shape[1]} sector indices, "
          f"{wide.index.min().date()} → {wide.index.max().date()}")
    return wide


def load_sector_indices() -> pd.DataFrame | None:
    if not SECTOR_INDEX_PATH.exists():
        return None
    return pd.read_parquet(SECTOR_INDEX_PATH)


def symbol_to_sector_map() -> dict[str, str]:
    fund = load_fund()
    if fund is None:
        return {}
    return dict(zip(fund["symbol"], fund["sector"].fillna("Unknown")))
