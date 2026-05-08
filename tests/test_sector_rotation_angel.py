"""Smoke test: sector_rotation routes OHLCV through Angel (data_provider).

Purpose:
  Verify (post-migration) that `src.sector_rotation` actually fetches
  OHLCV via `legacy_scanner/data_provider.py` (Angel One primary) and
  not directly via yfinance, and that a tiny live fetch + the full
  pipeline still work on cached data.

How it works:
  Step 1: Import `data_provider` and assert
          `sr._provider_download is data_provider.download` so we know
          the rotation module is wired to the right entry point.
  Step 2: Issue a live 7-day OHLCV fetch via `data_provider.download`
          for a tiny ticker basket and print the head.
  Step 3: Optionally run `sr.run_full_pipeline(force_refresh_ohlcv=False)`
          and print the leaderboard to confirm end-to-end works.

Data sources:
  Live: Angel One SmartAPI (requires valid creds in legacy_scanner config).
  Cached: existing data/ohlcv/, data/fundamentals.parquet, ...

Outputs:
  Console only.

How to run:
  python tests/test_sector_rotation_angel.py

Notes:
  * Not a pytest test (no assertions). Smoke / wiring check only.
  * Requires a working Angel One session for the live fetch step.
"""
import sys
import datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src import sector_rotation as sr

print("=" * 70)
print("STEP 1: Verify data_provider import path is wired in")
print("=" * 70)
import data_provider
print(f"  data_provider module:    {data_provider.__file__}")
print(f"  _provider_download id:   {sr._provider_download is data_provider.download}")
print(f"  Angel available:         {data_provider._angel_available()}")

print()
print("=" * 70)
print("STEP 2: Try a tiny live fetch via the new download_ohlcv path")
print("        (1 symbol, 7 days). Provider chain: Angel \u2192 jugaad \u2192 yfinance")
print("=" * 70)
end = dt.date.today()
start = end - dt.timedelta(days=10)
test_sym = "RELIANCE"
# Use a tmp dir so we don't clobber the real cache
import tempfile, os
tmp = Path(tempfile.mkdtemp())
orig = sr.OHLCV_DIR
sr.OHLCV_DIR = tmp
try:
    n_ok = sr.download_ohlcv([test_sym], start=start, end=end, verbose=True)
    print(f"  ok={n_ok}")
    p = tmp / f"{test_sym}.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        print(f"  rows: {len(df)}, last date: {df.index.max().date()}")
        print(df.tail(3).round(2))
    else:
        print("  ! No parquet written")
finally:
    sr.OHLCV_DIR = orig
    import shutil; shutil.rmtree(tmp, ignore_errors=True)

print()
print("=" * 70)
print("STEP 3: Run full rotation pipeline on existing cache (no fetch)")
print("=" * 70)
rot = sr.run_full_pipeline(rebuild_indices=False)
sr.leaderboard(rot, n=5)
