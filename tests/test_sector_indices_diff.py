"""Diagnostic: diff old vs new `sector_indices.parquet` on common date range.

Purpose:
  When `data/sector_indices.parquet` is rebuilt (e.g. after a fundamentals
  refresh, after sector-mapping changes, or after switching the OHLCV
  source), confirm that the recomputed indices agree with the previous
  snapshot on the overlapping date range and only differ at the new tail.

How it works:
  Loads `data/sector_indices.parquet.bak` and `data/sector_indices.parquet`,
  clips the new run to the old run's index, and prints summary diffs
  (column equality, index equality, max abs delta).

Data sources:
  data/sector_indices.parquet.bak   (previous snapshot — must exist)
  data/sector_indices.parquet       (current snapshot)

Outputs:
  Console only.

How to run:
  Make a backup BEFORE rebuilding:
      cp data/sector_indices.parquet data/sector_indices.parquet.bak
  Then rebuild via `scripts/06_fetch_fundamentals.py` (or
  `src.sectors.build_sector_indices()`), then run:
      python tests/test_sector_indices_diff.py

Notes:
  * Not a pytest test — a one-shot diagnostic.
"""
import pandas as pd
import numpy as np

old = pd.read_parquet("data/sector_indices.parquet.bak").sort_index()
new = pd.read_parquet("data/sector_indices.parquet").sort_index()
print(f"old: {old.index.min().date()} -> {old.index.max().date()}  rows={len(old)}")
print(f"new: {new.index.min().date()} -> {new.index.max().date()}  rows={len(new)}")

# The new run includes 5 extra trading days (newer OHLCV bars).
# Compare on old's date range only.
new_clip = new.loc[old.index]
print(f"new clipped rows : {len(new_clip)}")
print(f"index equals     : {old.index.equals(new_clip.index)}")
print(f"cols equals      : {list(old.columns) == list(new_clip.columns)}")

diff = (old - new_clip).abs()
print(f"max abs diff     : {np.nanmax(diff.values):.2e}")
print(f"exactly equal    : {old.equals(new_clip)}")
print(f"np.allclose 1e-12: {np.allclose(old.values, new_clip.values, atol=1e-12, equal_nan=True)}")

print()
print("Side-by-side on last common date:")
last = old.index.max()
side = pd.concat(
    [old.loc[last].rename("old"), new_clip.loc[last].rename("new")], axis=1
)
side["diff"] = (side["old"] - side["new"]).abs()
print(side.round(8).to_string())
