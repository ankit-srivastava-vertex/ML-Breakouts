"""Diagnostic: compare Angel-sourced sector rotation against the yfinance backup.

Purpose:
  After the May 2026 migration from yfinance to Angel One SmartAPI for
  OHLCV, sanity-check that the new sector-rotation snapshot agrees
  closely with the prior yfinance-era snapshot. Used once during the
  migration and kept as a regression diagnostic.

How it works:
  Loads the latest sector-rotation parquet (preferring `Output/`, then
  legacy `data/`) and the saved `data/sector_rotation.parquet.yf_bak`,
  joins on sector, and prints side-by-side numeric diffs (rank, RS,
  breadth, momentum) plus rank-correlation.

Data sources:
  Output/sector_rotation.parquet            (Angel-sourced; current)
  data/sector_rotation.parquet (fallback)   (legacy location)
  data/sector_rotation.parquet.yf_bak       (yfinance-era reference)

Outputs:
  Console only.

How to run:
  python tests/compare_angel_vs_yf.py

Notes:
  * Not a pytest test — it's a one-shot diagnostic. Safe to delete
    once the .yf_bak is no longer trusted as a baseline.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
OUT = ROOT / "Output"

# Look in Output/ first, fall back to legacy data/ location.
_new_p = OUT / "sector_rotation.parquet"
if not _new_p.exists():
    _new_p = D / "sector_rotation.parquet"
new = pd.read_parquet(_new_p)
old = pd.read_parquet(D / "sector_rotation.parquet.yf_bak")

cmp = pd.DataFrame({
    "rank_yf":  old["rank"],
    "rank_ang": new["rank"],
    "mom_yf":   old["mom_score"].round(2),
    "mom_ang":  new["mom_score"].round(2),
}).sort_values("rank_ang")
cmp["rank_delta"] = cmp["rank_ang"] - cmp["rank_yf"]
print("\n== LEADERBOARD: Angel vs yfinance ==")
print(cmp.to_string())

sn = pd.read_parquet(D / "sector_indices.parquet").sort_index()
so = pd.read_parquet(D / "sector_indices.parquet.yf_bak").sort_index()
common_sec = [c for c in sn.columns if c in so.columns]
common_dt = sn.index.intersection(so.index)
print(f"\n== Sector indices: {len(common_sec)} sectors, "
      f"{len(common_dt)} common dates ==")
last_dt = common_dt[-1]
last = pd.DataFrame({
    "yf_last":  so.loc[last_dt, common_sec].round(3),
    "ang_last": sn.loc[last_dt, common_sec].round(3),
})
last["pct_diff"] = ((last["ang_last"] / last["yf_last"] - 1) * 100).round(2)
print(f"  (as of {last_dt.date()})")
print(last.to_string())

bn = pd.read_parquet(D / "benchmark_NIFTY.parquet").sort_index()
bo = pd.read_parquet(D / "benchmark_NIFTY.parquet.yf_bak").sort_index()
print("\n== ^NSEI benchmark ==")
print(f"  yf:    rows={len(bo):4d}  "
      f"range={bo.index.min().date()} -> {bo.index.max().date()}  "
      f"last_close={bo['Close'].iloc[-1]:.2f}")
print(f"  angel: rows={len(bn):4d}  "
      f"range={bn.index.min().date()} -> {bn.index.max().date()}  "
      f"last_close={bn['Close'].iloc[-1]:.2f}")

try:
    from scipy.stats import spearmanr
    joined = old[["rank"]].join(new[["rank"]], lsuffix="_yf", rsuffix="_ang",
                                how="inner")
    rho, _ = spearmanr(joined["rank_yf"], joined["rank_ang"])
    print(f"\nSpearman rank correlation (yf vs Angel leaderboard): "
          f"rho = {rho:.3f}")
except Exception as e:
    print(f"\n(Spearman skipped: {e})")
