"""Phase 3 — train the LightGBM meta-model + isotonic calibrator.

Purpose:
  Fit the model that turns rule-detected breakout setups into a
  calibrated win-probability used by the daily scan.

How it works:
  Delegates the heavy lifting to `src.model.train(...)`:
    1. Load `data/training/setups.parquet`, sort by `asof`.
    2. Reserve the last `cv.holdout_months` as a strict forward holdout
       (NEVER seen during training/CV) and report honest forward metrics.
    3. Run cross-validation (default = WalkForwardCV, expanding window;
       PurgedKFold available via `cfg.cv.scheme`).
    4. Generate OOF predictions, fit isotonic calibrator on a held-out
       tail.
    5. Re-fit a final LightGBM on all training data (CV-best n_estimators).
    6. Optionally fit a stacker / regime sub-model.
    7. Persist all artefacts under `data/models/`.

v2 (May 2026):
  * Holdout: `cv.holdout_months` are NEVER seen during training/CV.
  * Walk-forward CV is the default; PurgedKFold available via cfg.cv.scheme.

Data sources:
  data/training/setups.parquet         (default input)
  configs/default.yaml::lightgbm,cv    (hyperparameters)
  data/models/best_hparams.yaml        (optional Optuna params)

Outputs (all in data/models/):
  lgbm.txt, calibrator.pkl, features.json,
  oof_predictions.parquet, cv_metrics.csv,
  feature_importance.csv, (optional) stacker_xgb.pkl, regime_model.pkl

How to run:
  python scripts/03_train_model.py
  python scripts/03_train_model.py --no-holdout            # train on all data
  python scripts/03_train_model.py --data path/to/other.parquet

Cadence:
  Quarterly routine refresh; or after #07 Optuna search finds new params;
  or after `src/features.py` is modified.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import load_config, TRAINING_DIR
from src.model import train, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(TRAINING_DIR / "setups.parquet"))
    ap.add_argument("--no-holdout", action="store_true",
                    help="Train on all data (skip honest forward holdout)")
    args = ap.parse_args()

    cfg = load_config()
    df = pd.read_parquet(args.data)
    df["asof"] = pd.to_datetime(df["asof"])
    df = df.sort_values("asof").reset_index(drop=True)
    print(f"Loaded {len(df):,} setups from {args.data}")
    print(f"  Date range: {df['asof'].min().date()} → {df['asof'].max().date()}")

    holdout_months = cfg.get("cv", {}).get("holdout_months", 12)
    if not args.no_holdout and holdout_months > 0:
        cutoff = df["asof"].max() - pd.DateOffset(months=holdout_months)
        train_df = df[df["asof"] <= cutoff].copy().reset_index(drop=True)
        holdout_df = df[df["asof"] > cutoff].copy().reset_index(drop=True)
        print(f"\n  Holdout split: cutoff={cutoff.date()}, "
              f"train={len(train_df):,}, holdout={len(holdout_df):,}")
        print(f"  Holdout base rate: {holdout_df['y'].mean()*100:.1f}%")
    else:
        train_df = df
        holdout_df = None
        print("\n  No holdout — training on all data")

    print("\n========== TRAINING ==========")
    train(train_df, cfg)

    if holdout_df is not None and len(holdout_df) > 0:
        print("\n========== HOLDOUT EVAL (NEVER SEEN BY MODEL) ==========")
        from src.model import _holdout_eval
        _holdout_eval(holdout_df, cfg)


if __name__ == "__main__":
    main()
