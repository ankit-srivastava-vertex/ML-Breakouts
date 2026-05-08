"""Diagnostic: rebuild sector indices via the new self-contained module
and compare to a backup byte-for-byte.

Purpose:
  Confirm that the equal-weight sector-index code copied into
  `src/sector_rotation.py` is byte-equivalent to the canonical
  implementation in `src/sectors.py`, so the two parquets cannot drift.

How it works:
  Step 1: Call `src.sector_rotation.build_sector_indices()` to rebuild
          `data/sector_indices.parquet`.
  Step 2: Diff against `data/sector_indices.parquet.bak` on the common
          date range; assert columns / index / max-abs-delta agree.
  Step 3: Run `run_full_pipeline()` end-to-end as a smoke check and
          print `leaderboard()`.

Data sources:
  data/sector_indices.parquet.bak   (must exist; previously-trusted run)
  data/ohlcv/, data/fundamentals.parquet, data/benchmark_NIFTY.parquet

Outputs:
  Console only.

How to run:
  Make a backup first:
      cp data/sector_indices.parquet data/sector_indices.parquet.bak
  Then:
      python tests/test_sector_rotation_match.py

Notes:
  * Not a pytest test (no assertions). Smoke / equivalence check.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from src.sector_rotation import (
    run_full_pipeline, leaderboard, build_sector_indices, load_sector_indices,
)

print("=" * 70)
print("STEP 1: Rebuild sector_indices.parquet via new self-contained module")
print("=" * 70)
new_idx = build_sector_indices()

print()
print("=" * 70)
print("STEP 2: Compare to backup (old src/sectors.py output)")
print("=" * 70)
old_idx = pd.read_parquet("data/sector_indices.parquet.bak")
new_idx_loaded = load_sector_indices()

print(f"  old shape : {old_idx.shape}")
print(f"  new shape : {new_idx_loaded.shape}")
old_cols = sorted(map(str, old_idx.columns))
new_cols = sorted(map(str, new_idx_loaded.columns))
print(f"  cols match : {old_cols == new_cols}")
print(f"  index match: {old_idx.index.equals(new_idx_loaded.index)}")

common_cols = old_idx.columns.intersection(new_idx_loaded.columns)
old_sub = old_idx[common_cols].sort_index()
new_sub = new_idx_loaded[common_cols].sort_index()
old_sub, new_sub = old_sub.align(new_sub, join="inner")
diff = (old_sub - new_sub).abs()
print(f"  max abs diff   : {diff.values.max():.2e}")
print(f"  exactly equal  : {old_sub.equals(new_sub)}")
print(f"  np.allclose 0  : {np.allclose(old_sub.values, new_sub.values, atol=1e-9, equal_nan=True)}")

print()
print("=" * 70)
print("STEP 3: Compute rotation + leaderboard via run_full_pipeline()")
print("=" * 70)
rot = run_full_pipeline(rebuild_indices=False)
leaderboard(rot, n=5)
