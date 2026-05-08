"""Phase 6 — refresh fundamentals cache + rebuild sector indices.

Purpose:
  Populate `data/fundamentals.parquet` with the latest per-symbol
  yfinance `.info` snapshot, then regenerate `data/sector_indices.parquet`
  off the refreshed sector mapping. These two artefacts are read by
  `src/sectors.py`, `src/sector_rotation.py`, `src/features.py` and the
  daily scan / training-set builder.

How it works:
  1. Determine universe from `data/ohlcv/*.parquet` (or `--max-symbols N`).
  2. `src.fundamentals.fetch_all(symbols, workers)` — concurrent
     `yfinance.Ticker(<sym>.NS).info` calls; tolerant of per-symbol
     failures.
  3. `src.fundamentals.add_cross_sectional_ranks(df)` — add
     `fund_*_rank_sector` and `fund_*_rank_all` percentile columns.
  4. Persist `data/fundamentals.parquet`.
  5. `src.sectors.build_sector_indices(min_members=10)` — build
     equal-weight sector indices and write `data/sector_indices.parquet`.

Data sources:
  yfinance .info  (the only source for sector / fundamentals data;
                   Angel One SmartAPI does NOT expose them)
  data/ohlcv/*.parquet  (closes used to build the sector indices)

Outputs:
  data/fundamentals.parquet      ~22 raw fields + cross-sectional ranks
  data/sector_indices.parquet    one column per sector, daily index level

How to run:
  python scripts/06_fetch_fundamentals.py
  python scripts/06_fetch_fundamentals.py --workers 8
  python scripts/06_fetch_fundamentals.py --skip-fetch       # only rebuild idx
  python scripts/06_fetch_fundamentals.py --resume           # incremental
  python scripts/06_fetch_fundamentals.py --max-symbols 50   # smoke test

Cadence:
  Quarterly (sector mappings change rarely). Piggy-back on the same
  schedule as the routine retrain (#02 + #03).

Notes:
  * yfinance .info is flaky — expect ~5% NaNs per refresh; downstream
    feature code tolerates them.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import OHLCV_DIR
from src.fundamentals import (
    fetch_all, save, add_cross_sectional_ranks, FUND_PATH, load,
)
from src.sectors import build_sector_indices
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--skip-fetch", action="store_true",
                    help="reuse existing fundamentals.parquet")
    ap.add_argument("--resume", action="store_true",
                    help="skip symbols already in fundamentals.parquet")
    args = ap.parse_args()

    syms = sorted(p.stem for p in OHLCV_DIR.glob("*.parquet"))
    if args.max_symbols:
        syms = syms[:args.max_symbols]

    if not args.skip_fetch:
        existing = load() if args.resume else None
        if existing is not None and len(existing):
            already = set(existing["symbol"])
            todo = [s for s in syms if s not in already]
            print(f"Resuming: {len(already)} cached, {len(todo)} todo")
        else:
            existing = None
            todo = syms
            print(f"Fetching fundamentals for {len(todo)} symbols ...")

        new = fetch_all(todo, workers=args.workers)

        if existing is not None and len(new):
            df = pd.concat([existing, new], ignore_index=True)
            df = df.drop_duplicates(subset=["symbol"], keep="last")
        else:
            df = new if len(new) else (existing if existing is not None else new)

        save(df)
        if len(df):
            df = add_cross_sectional_ranks(df)
            save(df)
            print(f"  ✓ saved → {FUND_PATH}")
            print(f"  rows:       {len(df)}")
            print(f"  sectors:    {df['sector'].nunique()}")
            print(f"  industries: {df['industry'].nunique()}")
            print(f"  with mcap:  {df['marketCap'].notna().sum()}")
        else:
            print("  Empty fundamentals — Yahoo rate-limited. "
                  "Wait an hour and retry with --resume.")
            return
    else:
        print("(skipping fundamentals fetch)")

    print("\nBuilding sector indices ...")
    build_sector_indices(min_members=10)


if __name__ == "__main__":
    main()
