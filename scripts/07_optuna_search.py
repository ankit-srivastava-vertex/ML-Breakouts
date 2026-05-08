"""Phase 7 — Optuna hyperparameter search for the LightGBM meta-model.

Purpose:
  Search LightGBM hyperparameters with an honest, leakage-free objective
  and persist the winning set as `data/models/best_hparams.yaml` so the
  next run of `scripts/03_train_model.py` picks them up automatically.

How it works:
  1. Load `data/training/setups.parquet`.
  2. Compute sample-uniqueness weights from triple-barrier `t1`.
  3. PurgedKFold (with embargo) splits respecting label overlap.
  4. Optuna trial samples LightGBM params (num_leaves, min_data_in_leaf,
     lambda_l1/l2, feature_fraction, bagging_*, learning_rate, ...).
  5. Per-fold fit → OOF predictions → mean precision@top-decile across
     folds is the objective (correlates with strategy P&L far better
     than AUC for this imbalanced, threshold-driven problem).
  6. After N trials, dump the best params to YAML.

Data sources:
  data/training/setups.parquet     (default --data)
  configs/default.yaml::cv,labeling (CV scheme + label horizon)

Outputs:
  data/models/best_hparams.yaml    (consumed by scripts/03_train_model.py)
  optuna study DB (in-memory by default; pass `--storage` to persist)

How to run:
  python scripts/07_optuna_search.py --trials 100
  python scripts/07_optuna_search.py --trials 200 --timeout 7200
  python scripts/07_optuna_search.py --data path/to/other.parquet

Cadence:
  Yearly, or whenever a meaningful new feature group is added in
  `src/features.py`.

Notes:
  * Long-running (hours). Run on a workstation, not a laptop.
  * After this completes, ALWAYS run `scripts/03_train_model.py`
    followed by `scripts/04_evaluate.py` to validate the new params.
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import load_config, TRAINING_DIR, MODELS_DIR
from src.cv import PurgedKFold, sample_uniqueness_weights
from src.model import feature_list


def objective(trial, df, feats, t1, weights, cfg):
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "seed": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0, log=True),
    }
    if trial.suggest_categorical("extra_trees", [True, False]):
        params["extra_trees"] = True
    n_rounds = trial.suggest_int("num_boost_round", 200, 3000, log=True)

    # Apply monotone constraints (same as training) so HP search optimizes
    # the *deployed* model, not an unconstrained one
    from src.model import _monotone_vector
    mono = _monotone_vector(feats, cfg)
    if any(v != 0 for v in mono):
        params["monotone_constraints"] = mono
        params["monotone_constraints_method"] = "advanced"

    X = df[feats].astype(float)
    y = df["y"].astype(int).values

    pkf = PurgedKFold(
        n_splits=cfg["cv"]["n_splits"],
        embargo_days=cfg["cv"]["embargo_days"],
        label_horizon_days=cfg["labeling"]["time_barrier_days"],
    )

    fold_scores = []
    for tr_idx, te_idx in pkf.split(t1):
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue
        d_tr = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx],
                           weight=weights[tr_idx])
        d_te = lgb.Dataset(X.iloc[te_idx], label=y[te_idx],
                           weight=weights[te_idx], reference=d_tr)
        booster = lgb.train(
            params, d_tr, num_boost_round=n_rounds,
            valid_sets=[d_te],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                       lgb.log_evaluation(0)],
        )
        p = booster.predict(X.iloc[te_idx],
                            num_iteration=booster.best_iteration)
        # Blended objective: precision@top10% (strategy P&L proxy) +
        # 0.5 * AP (overall ranking quality), minus 0.25 * std penalty
        n_top = max(10, len(p) // 10)
        order = np.argsort(p)[::-1]
        prec_top = y[te_idx][order[:n_top]].mean()
        ap = average_precision_score(y[te_idx], p)
        fold_scores.append(float(prec_top + 0.5 * ap))

    if not fold_scores:
        return 0.0
    return float(np.mean(fold_scores) - 0.25 * np.std(fold_scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(TRAINING_DIR / "setups.parquet"))
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=3600,
                    help="seconds before stopping search")
    ap.add_argument("--subsample", type=float, default=0.4,
                    help="fraction of training set to use during HP search "
                         "(stratified by year)")
    args = ap.parse_args()

    import optuna

    cfg = load_config()
    df = pd.read_parquet(args.data).dropna(subset=["y"]).copy()
    df["asof"] = pd.to_datetime(df["asof"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values("asof").reset_index(drop=True)

    if 0 < args.subsample < 1.0:
        df["year"] = df["asof"].dt.year
        df = (df.groupby("year", group_keys=False)
                .apply(lambda x: x.sample(frac=args.subsample, random_state=42))
                .sort_values("asof").reset_index(drop=True))
        df = df.drop(columns=["year"])
        print(f"Subsampled to {len(df):,} rows ({args.subsample * 100:.0f}%)")

    feats = feature_list(df)
    t1 = pd.Series(df["exit_date"].values, index=df["asof"].values)
    weights = sample_uniqueness_weights(t1)

    print(f"Optuna search: {args.n_trials} trials, "
          f"timeout {args.timeout}s, {len(feats)} features")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    t0 = time.time()
    study.optimize(
        lambda t: objective(t, df, feats, t1, weights, cfg),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=False,
    )

    print(f"\n=== Best ===")
    print(f"  score (prec@top10 + 0.5*AP - 0.25std): {study.best_value:.4f}")
    print(f"  trials run: {len(study.trials)}, "
          f"elapsed: {time.time() - t0:.0f}s")
    print(f"  params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Save to YAML for easy re-use
    out = MODELS_DIR / "best_hparams.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    with open(out, "w") as f:
        yaml.safe_dump({
            "best_score": float(study.best_value),
            "n_trials": len(study.trials),
            "params": study.best_params,
        }, f)
    print(f"\n✓ Saved best params → {out}")
    print("\nTo train final model with these params, edit "
          "configs/default.yaml model.params block, then run:")
    print("    python3 scripts/03_train_model.py")


if __name__ == "__main__":
    main()
