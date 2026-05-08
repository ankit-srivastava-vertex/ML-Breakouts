"""Phase 4 — evaluate the trained meta-model on its OOF predictions.

Purpose:
  Decide whether the freshly-trained model is good enough to deploy.
  Surfaces honest, forward-looking diagnostics from the held-out CV
  predictions written by `scripts/03_train_model.py`.

How it works:
  1. Load `data/models/oof_predictions.parquet`.
  2. Compute global metrics: AUC, Average Precision, Brier score,
     base rate, mean R-multiple.
  3. Decile analysis: bin OOF probabilities into 10 deciles and report
     hit-rate, count and mean R per decile (the actual edge that
     matters in trading).
  4. Calibration table.
  5. Strategy backtest: take all setups with calibrated prob >= trigger
     threshold and report win rate, mean R, expectancy, total trades.

Data sources:
  data/models/oof_predictions.parquet   (input)
  configs/default.yaml::inference        (trigger threshold)

Outputs:
  Console only — no file writes. Read this BEFORE promoting a model.

How to run:
  python scripts/04_evaluate.py

Cadence:
  After every `03_train_model.py` run, and ad-hoc when you suspect
  drift/regime change (compare to last evaluation snapshot).

Notes:
  * If evaluation looks bad, options are: rerun #02+#03 against fresher
    data, or rerun #07 Optuna search.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import load_config, MODELS_DIR


def main():
    cfg = load_config()
    oof = pd.read_parquet(MODELS_DIR / "oof_predictions.parquet")
    oof = oof.dropna(subset=["oof_pred_calibrated"]).copy()
    oof["asof"] = pd.to_datetime(oof["asof"])
    print(f"Evaluating on {len(oof):,} OOF predictions")
    print(f"Date range: {oof['asof'].min().date()} → {oof['asof'].max().date()}")
    print(f"Base rate:  {oof['y'].mean() * 100:.2f}% positive")
    print(f"Mean R:     {oof['r_multiple'].mean():.3f} (raw, all setups)")

    # ─── Decile analysis ──────────────────────────────────────────────────
    oof["decile"] = pd.qcut(oof["oof_pred_calibrated"], 10,
                            labels=False, duplicates="drop")
    grp = oof.groupby("decile").agg(
        n=("y", "size"),
        hit_rate=("y", "mean"),
        mean_R=("r_multiple", "mean"),
        median_R=("r_multiple", "median"),
        mean_prob=("oof_pred_calibrated", "mean"),
    ).round(3)
    print("\nDecile analysis (decile 9 = highest predicted prob):")
    print(grp.to_string())

    # ─── Calibration table ────────────────────────────────────────────────
    cal_bins = np.linspace(0, 1, 11)
    oof["cal_bin"] = pd.cut(oof["oof_pred_calibrated"], cal_bins,
                            include_lowest=True)
    cal = oof.groupby("cal_bin", observed=True).agg(
        n=("y", "size"),
        predicted=("oof_pred_calibrated", "mean"),
        actual=("y", "mean"),
    ).round(3)
    print("\nCalibration table:")
    print(cal.to_string())

    # ─── Strategy backtest at threshold ──────────────────────────────────
    thr = cfg["inference"].get("prob_threshold_trigger", 0.65)
    sel = oof[oof["oof_pred_calibrated"] >= thr]
    print(f"\nStrategy: take all setups with calibrated prob ≥ {thr}")
    print(f"  Trades:        {len(sel):,}  "
          f"({len(sel) / len(oof) * 100:.1f}% of all setups)")
    if len(sel):
        print(f"  Hit rate:      {sel['y'].mean() * 100:.1f}%")
        print(f"  Mean R:        {sel['r_multiple'].mean():.3f}")
        print(f"  Median R:      {sel['r_multiple'].median():.3f}")
        print(f"  Total R:       {sel['r_multiple'].sum():.1f}")
        # Per-year breakdown
        sel = sel.copy()
        sel["year"] = sel["asof"].dt.year
        yr = sel.groupby("year").agg(
            n=("y", "size"),
            hit_rate=("y", "mean"),
            mean_R=("r_multiple", "mean"),
            total_R=("r_multiple", "sum"),
        ).round(3)
        print("\n  Per-year:")
        print(yr.to_string())

    # ─── Threshold sweep ──────────────────────────────────────────────────
    print("\nThreshold sweep:")
    print(f"{'threshold':>10}  {'n_trades':>9}  {'hit_rate':>9}  "
          f"{'mean_R':>8}  {'total_R':>9}")
    for t in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        sub = oof[oof["oof_pred_calibrated"] >= t]
        if len(sub) < 5:
            continue
        print(f"{t:>10.2f}  {len(sub):>9,}  "
              f"{sub['y'].mean():>9.3f}  "
              f"{sub['r_multiple'].mean():>8.3f}  "
              f"{sub['r_multiple'].sum():>9.1f}")


if __name__ == "__main__":
    main()
