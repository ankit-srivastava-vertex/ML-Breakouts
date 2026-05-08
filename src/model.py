"""LightGBM meta-model trainer + isotonic calibrator (+ optional stacker).

Purpose:
  Learn `P(setup hits +2 ATR before -1 ATR within time_days | features)`.
  Outputs a calibrated probability that the daily scan thresholds against
  watchlist / trigger cutoffs.

How it works:
  1. Load the labeled training parquet (output of
     `scripts/02_build_training_set.py`).
  2. Drop META_COLS (identity + label cols) to derive the feature list.
  3. Compute sample-uniqueness weights from `t1` (label-end dates).
  4. CV split (default = WalkForwardCV with `cv.holdout_months` reserved;
     PurgedKFold available via `cfg.cv.scheme`). For each fold:
       - fit LightGBM on train, predict OOF on test.
  5. Fit isotonic calibrator on the held-out tail of OOF predictions.
  6. Re-fit a final LightGBM on ALL training data with the
     CV-best `n_estimators`.
  7. (Optional) Train an XGBoost-DART stacker over OOF predictions +
     a regime sub-model, blending into a single probability at predict-time.
  8. Persist all artefacts under `data/models/`.

Data sources:
  data/training/setups.parquet         (input, ~24k rows × 119 cols)
  configs/default.yaml::lightgbm,cv    (hyperparameters)

Outputs (all under `data/models/`):
  lgbm.txt                  primary booster
  calibrator.pkl            isotonic calibrator
  features.json             ordered feature list (locked for inference)
  oof_predictions.parquet   per-row CV OOF + final probabilities
  cv_metrics.csv            per-fold AUC / AP / Brier / decile lift
  feature_importance.csv    LightGBM gain importance
  best_hparams.yaml         (optional) Optuna best params
  stacker_xgb.pkl           (optional) DART stacker
  regime_model.pkl          (optional) regime sub-model

How to run:
  Driven by `scripts/03_train_model.py` (and `scripts/07_optuna_search.py`
  to populate `best_hparams.yaml`). Inference uses `predict(feature_row)`.

      from src.model import train, predict
      out = predict(make_features(df, setup, bench, sector_idx, fund_row))
      # out: {prob_raw, prob_calibrated, prob_blend, ...}

Notes:
  * `META_COLS` is the canonical list of non-feature columns. Editing
    it changes which columns are treated as features — keep in sync
    with `scripts/02_build_training_set.py`.
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .paths import MODELS_DIR
from .cv import PurgedKFold, WalkForwardCV, sample_uniqueness_weights


META_COLS = {
    "symbol", "asof", "exit_date", "label_raw", "y", "r_multiple",
    "days_held", "R", "close_at_setup",
}


# ─── Monotone constraints (domain priors) ───────────────────────────────────
# +1 = expected monotonically increasing in P(breakout success)
# -1 = expected monotonically decreasing
# Features not listed → 0 (no constraint).
DEFAULT_MONOTONE = {
    # higher RS / momentum / accumulation should not hurt
    "rs_mansfield": 1, "rs_slope_20": 1, "rs_3m": 1, "rs_6m": 1,
    "obv_slope_20": 1, "obv_slope_50": 1, "obv_accel": 1,
    "adx_14": 1, "ema200_slope_20": 1,
    "wk_adx_14": 1, "wk_roc_12": 1, "wk_roc_26": 1,
    "sec_above_50dma": 1, "sec_above_200dma": 1,
    "bench_above_200dma": 1, "ema_full_stack": 1,
    "coiled_spring_score": 1, "is_52w_high": 1,
    "wyckoff_spring": 1, "pocket_pivot": 1,
    "nr7": 1, "inside_bar": 1, "close_touches_R": 1, "high_pierces_R": 1,
    # farther from a 52w high → less explosive
    "dist_52w_high_pct": -1, "wk_dist_52w_high_pct": -1,
    "sec_dist_52w_high_pct": -1,
    # higher debt / extreme valuation → worse
    "fund_debtToEquity": -1,
    # wider/looser base, more failed attempts → worse
    "base_depth_width_ratio": -1, "base_tightness_30d": -1,
}


def _monotone_vector(feats: list[str], cfg: dict) -> list[int]:
    """Return monotone constraints vector aligned to feats list."""
    if not cfg.get("model", {}).get("monotone_constraints", True):
        return [0] * len(feats)
    overrides = cfg.get("model", {}).get("monotone_overrides", {}) or {}
    table = {**DEFAULT_MONOTONE, **overrides}
    return [int(table.get(f, 0)) for f in feats]


def feature_list(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in META_COLS]
    # Drop any object/string cols
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def train(df: pd.DataFrame, cfg: dict, save: bool = True) -> dict:
    """Train LightGBM meta-model and return artifacts dict."""
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 brier_score_loss)

    df = df.dropna(subset=["y"]).copy()
    df["asof"] = pd.to_datetime(df["asof"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values("asof").reset_index(drop=True)

    feats = feature_list(df)
    X = df[feats].astype(float)
    y = df["y"].astype(int).values
    t1 = pd.Series(df["exit_date"].values, index=df["asof"].values)

    print(f"Training rows: {len(df):,}")
    print(f"Features:      {len(feats)}")
    print(f"Base rate:     {y.mean() * 100:.1f}% positive")

    weights = sample_uniqueness_weights(t1)
    print(f"Sample weights: mean={weights.mean():.3f}, "
          f"min={weights.min():.3f}, max={weights.max():.3f}")

    # Purged k-fold CV  (or WalkForwardCV via cfg.cv.scheme)
    n_splits = cfg["cv"]["n_splits"]
    embargo = cfg["cv"]["embargo_days"]
    horizon = cfg["labeling"]["time_barrier_days"]
    cv_scheme = cfg["cv"].get("scheme", "walkforward")
    if cv_scheme == "walkforward":
        pkf = WalkForwardCV(
            train_min_days=cfg["cv"].get("train_min_days", 365 * 3),
            test_days=cfg["cv"].get("test_days", 180),
            step_days=cfg["cv"].get("step_days", 90),
            embargo_days=embargo,
        )
        print(f"CV scheme: walk-forward (train_min={cfg['cv'].get('train_min_days', 365*3)}d, "
              f"test={cfg['cv'].get('test_days', 180)}d, step={cfg['cv'].get('step_days', 90)}d)")
    else:
        pkf = PurgedKFold(n_splits=n_splits, embargo_days=embargo,
                          label_horizon_days=horizon)
        print(f"CV scheme: purged-{n_splits}-fold")

    mcfg = cfg["model"]
    p = mcfg.get("params", {})
    params = {
        "objective": p.get("objective", "binary"),
        "metric": p.get("metric", "binary_logloss"),
        "learning_rate": p.get("learning_rate", 0.03),
        "num_leaves": p.get("num_leaves", 63),
        "min_data_in_leaf": p.get("min_data_in_leaf", 50),
        "feature_fraction": p.get("feature_fraction", 0.9),
        "bagging_fraction": p.get("bagging_fraction", 0.85),
        "bagging_freq": p.get("bagging_freq", 5),
        "lambda_l1": p.get("lambda_l1", 0.0),
        "lambda_l2": p.get("lambda_l2", 1.0),
        "min_gain_to_split": p.get("min_gain_to_split", 0.0),
        "verbosity": -1,
        "seed": 42,
    }
    if "scale_pos_weight" in p:
        params["scale_pos_weight"] = float(p["scale_pos_weight"])
    if p.get("extra_trees", False):
        params["extra_trees"] = True
    # Monotone constraints (huge overfitting reducer when domain priors hold)
    mono = _monotone_vector(feats, cfg)
    if any(v != 0 for v in mono):
        params["monotone_constraints"] = mono
        params["monotone_constraints_method"] = p.get(
            "monotone_constraints_method", "advanced")
        n_active = sum(1 for v in mono if v != 0)
        print(f"Monotone constraints: {n_active}/{len(mono)} features constrained")
    n_rounds = mcfg.get("num_boost_round", 2000)

    fold_metrics = []
    oof_pred = np.full(len(df), np.nan)
    splits = list(pkf.split(t1))
    n_folds = len(splits)
    for fold_i, (tr_idx, te_idx) in enumerate(splits, start=1):
        print(f"\n  Fold {fold_i}/{n_folds}: "
              f"train={len(tr_idx):,} test={len(te_idx):,}")
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue
        d_tr = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx],
                           weight=weights[tr_idx])
        d_te = lgb.Dataset(X.iloc[te_idx], label=y[te_idx],
                           weight=weights[te_idx], reference=d_tr)
        booster = lgb.train(
            params, d_tr, num_boost_round=n_rounds,
            valid_sets=[d_te],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False),
                       lgb.log_evaluation(0)],
        )
        p = booster.predict(X.iloc[te_idx],
                            num_iteration=booster.best_iteration)
        oof_pred[te_idx] = p

        auc = roc_auc_score(y[te_idx], p)
        ap = average_precision_score(y[te_idx], p)
        brier = brier_score_loss(y[te_idx], p)
        base = y[te_idx].mean()
        # Precision at top decile
        n_top = max(1, len(p) // 10)
        order = np.argsort(p)[::-1]
        prec_top = y[te_idx][order[:n_top]].mean()
        fold_metrics.append({
            "fold": fold_i, "auc": auc, "ap": ap, "brier": brier,
            "base_rate": base, "prec_top10": prec_top,
            "best_iter": booster.best_iteration,
        })
        print(f"    AUC={auc:.4f}  AP={ap:.4f}  "
              f"Brier={brier:.4f}  base={base:.3f}  "
              f"prec@top10%={prec_top:.3f}  best_iter={booster.best_iteration}")

    fm = pd.DataFrame(fold_metrics)
    print("\n  Cross-validated metrics (mean ± std):")
    for c in ["auc", "ap", "brier", "prec_top10"]:
        print(f"    {c:14s}  {fm[c].mean():.4f}  ±{fm[c].std():.4f}")

    # ─── Final fit on all data (use median best_iter from CV) ─────────────
    best_iter = int(fm["best_iter"].median())
    print(f"\n  Final fit: {best_iter} rounds on full data ...")
    d_all = lgb.Dataset(X, label=y, weight=weights)
    final_booster = lgb.train(
        params, d_all, num_boost_round=best_iter,
        callbacks=[lgb.log_evaluation(0)],
    )

    # ─── Isotonic calibration on out-of-fold predictions ──────────────────
    valid_oof = ~np.isnan(oof_pred)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_pred[valid_oof], y[valid_oof])
    print("  Isotonic calibrator fit on OOF predictions")

    # ─── Feature importance ───────────────────────────────────────────────
    imp = pd.DataFrame({
        "feature": feats,
        "gain": final_booster.feature_importance(importance_type="gain"),
        "split": final_booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    print("\n  Top 20 features by gain:")
    print(imp.head(20).to_string(index=False))

    # ─── Stacking ensemble (LGBM + XGBoost-DART + CatBoost → LR meta) ────
    stack_artifacts = None
    if mcfg.get("stacking", {}).get("enabled", False):
        stack_artifacts = _train_stack(
            X, y, weights, t1, pkf, params, n_rounds, mcfg, feats)

    # ─── Regime-conditioned models (split by bench_above_200dma) ─────────
    regime_artifacts = None
    if mcfg.get("regime_conditioned", {}).get("enabled", False):
        regime_artifacts = _train_regime_models(
            df, feats, y, weights, t1, pkf, params, n_rounds, mcfg)

    # ─── R-multiple regression head (for Kelly sizing) ───────────────────
    rmult_artifacts = None
    if mcfg.get("rmultiple_regressor", {}).get("enabled", False):
        rmult_artifacts = _train_rmultiple(
            X, df["r_multiple"].astype(float).values,
            weights, pkf, t1, params, n_rounds, feats)

    artifacts = {
        "booster": final_booster,
        "calibrator": iso,
        "features": feats,
        "cv_metrics": fm,
        "feature_importance": imp,
        "params": params,
        "best_iter": best_iter,
        "stacking": stack_artifacts,
        "regime": regime_artifacts,
        "rmultiple": rmult_artifacts,
    }

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        final_booster.save_model(str(MODELS_DIR / "lgbm.txt"))
        with open(MODELS_DIR / "calibrator.pkl", "wb") as f:
            pickle.dump(iso, f)
        with open(MODELS_DIR / "features.json", "w") as f:
            json.dump(feats, f)
        fm.to_csv(MODELS_DIR / "cv_metrics.csv", index=False)
        imp.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
        # Save OOF for downstream evaluation
        oof_df = df[["symbol", "asof", "y", "r_multiple"]].copy()
        oof_df["oof_pred"] = oof_pred
        oof_df["oof_pred_calibrated"] = np.where(
            valid_oof, iso.transform(np.where(valid_oof, oof_pred, 0.5)), np.nan)
        oof_df.to_parquet(MODELS_DIR / "oof_predictions.parquet")
        # Stacking artifacts
        if stack_artifacts:
            with open(MODELS_DIR / "stacker.pkl", "wb") as f:
                pickle.dump(stack_artifacts, f)
            print(f"✓ Saved stacker → {MODELS_DIR / 'stacker.pkl'}")
        # Regime artifacts
        if regime_artifacts:
            for k, b in regime_artifacts.get("boosters", {}).items():
                b.save_model(str(MODELS_DIR / f"lgbm_regime_{k}.txt"))
            with open(MODELS_DIR / "regime_calibrators.pkl", "wb") as f:
                pickle.dump(regime_artifacts.get("calibrators", {}), f)
            print(f"✓ Saved regime models → {MODELS_DIR}")
        # R-multiple regressor
        if rmult_artifacts:
            rmult_artifacts["booster"].save_model(
                str(MODELS_DIR / "rmult_regressor.txt"))
            print(f"✓ Saved R-multiple regressor → {MODELS_DIR / 'rmult_regressor.txt'}")
        print(f"\n✓ Saved artifacts → {MODELS_DIR}")

    return artifacts


# ─── Stacking helper ────────────────────────────────────────────────────────

def _train_stack(X, y, weights, t1, pkf, lgbm_params, n_rounds, mcfg, feats):
    """Train LGBM + XGBoost-DART + CatBoost as base learners; LR meta-learner."""
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    print("\n=== Stacking ensemble (LGBM + XGB-DART + CatBoost → LR) ===")
    # XGBoost / CatBoost reject ±inf. LightGBM tolerates them.
    # Sanitize a copy used only for non-LGBM learners.
    X_safe = X.replace([np.inf, -np.inf], np.nan)
    n = len(y)
    oof = {"lgbm": np.full(n, np.nan), "xgb": np.full(n, np.nan),
           "cat": np.full(n, np.nan)}

    # XGBoost / CatBoost params (sane defaults; tunable via cfg)
    scfg = mcfg.get("stacking", {})
    use_xgb = scfg.get("use_xgboost", True)
    use_cat = scfg.get("use_catboost", True)

    try:
        import xgboost as xgb
    except ImportError:
        print("  xgboost not installed → skipping XGB-DART base learner")
        use_xgb = False
    try:
        import catboost as cb
    except ImportError:
        print("  catboost not installed → skipping CatBoost base learner")
        use_cat = False

    fold_idx = list(pkf.split(t1))
    for fi, (tr, te) in enumerate(fold_idx, 1):
        if len(tr) == 0 or len(te) == 0:
            continue
        # LGBM
        d_tr = lgb.Dataset(X.iloc[tr], label=y[tr], weight=weights[tr])
        d_te = lgb.Dataset(X.iloc[te], label=y[te], weight=weights[te],
                           reference=d_tr)
        b = lgb.train(lgbm_params, d_tr, num_boost_round=n_rounds,
                      valid_sets=[d_te],
                      callbacks=[lgb.early_stopping(100, verbose=False),
                                 lgb.log_evaluation(0)])
        oof["lgbm"][te] = b.predict(X.iloc[te], num_iteration=b.best_iteration)

        if use_xgb:
            xb = xgb.XGBClassifier(
                booster="gbtree", n_estimators=400, learning_rate=0.05,
                max_depth=5, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                eval_metric="logloss", n_jobs=-1, random_state=42,
                tree_method="hist")
            xb.fit(X_safe.iloc[tr], y[tr], sample_weight=weights[tr])
            oof["xgb"][te] = xb.predict_proba(X_safe.iloc[te])[:, 1]

        if use_cat:
            cm = cb.CatBoostClassifier(
                iterations=300, learning_rate=0.06, depth=5,
                l2_leaf_reg=3.0, bagging_temperature=0.5,
                random_seed=42, loss_function="Logloss",
                verbose=False, allow_writing_files=False,
                thread_count=-1)
            cm.fit(X_safe.iloc[tr], y[tr], sample_weight=weights[tr])
            oof["cat"][te] = cm.predict_proba(X_safe.iloc[te])[:, 1]

        print(f"  stack fold {fi}/{len(fold_idx)}: done")

    # Build meta-features (only base learners that produced OOF)
    base_cols = [k for k, v in oof.items() if np.isfinite(v).any()]
    valid = np.all([np.isfinite(oof[k]) for k in base_cols], axis=0)
    meta_X = np.vstack([oof[k][valid] for k in base_cols]).T
    meta_y = y[valid]

    meta = LogisticRegression(C=1.0, max_iter=1000)
    meta.fit(meta_X, meta_y)

    # Calibrate stack
    stack_oof = meta.predict_proba(meta_X)[:, 1]
    iso_stack = IsotonicRegression(out_of_bounds="clip")
    iso_stack.fit(stack_oof, meta_y)

    print(f"  Stack AUC (OOF) = {roc_auc_score(meta_y, stack_oof):.4f}  "
          f"AP = {average_precision_score(meta_y, stack_oof):.4f}")

    # Refit base learners on full data
    base_models = {}
    d_all = lgb.Dataset(X, label=y, weight=weights)
    base_models["lgbm"] = lgb.train(lgbm_params, d_all,
                                    num_boost_round=int(n_rounds))
    if use_xgb:
        xb = xgb.XGBClassifier(
            booster="gbtree", n_estimators=400, learning_rate=0.05,
            max_depth=5, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", n_jobs=-1, random_state=42,
            tree_method="hist")
        xb.fit(X_safe, y, sample_weight=weights)
        base_models["xgb"] = xb
    if use_cat:
        cm = cb.CatBoostClassifier(
            iterations=300, learning_rate=0.06, depth=5,
            l2_leaf_reg=3.0, bagging_temperature=0.5,
            random_seed=42, loss_function="Logloss",
            verbose=False, allow_writing_files=False,
            thread_count=-1)
        cm.fit(X_safe, y, sample_weight=weights)
        base_models["cat"] = cm

    return {
        "base_models": base_models,
        "base_cols": base_cols,
        "meta": meta,
        "calibrator": iso_stack,
        "features": feats,
    }


# ─── Regime-conditioned models ─────────────────────────────────────────────

def _train_regime_models(df, feats, y, weights, t1, pkf, params, n_rounds, mcfg):
    """Train separate boosters for bull regime (bench_above_200dma=1) vs
    bear regime (=0). At inference we route by the same flag."""
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    print("\n=== Regime-conditioned models ===")
    if "bench_above_200dma" not in df.columns:
        print("  bench_above_200dma not in features — skipping regime split")
        return None

    regime = df["bench_above_200dma"].fillna(1).astype(int).values
    boosters: dict = {}
    calibrators: dict = {}
    X_full = df[feats].astype(float)

    for label, mask_val in [("bull", 1), ("bear", 0)]:
        mask = regime == mask_val
        if mask.sum() < 200:
            print(f"  {label}: only {mask.sum()} rows — skipping")
            continue
        Xr, yr, wr = X_full[mask], y[mask], weights[mask]
        # Local CV for early stopping (single split: last 20% by date)
        n = len(Xr)
        cut = int(n * 0.8)
        d_tr = lgb.Dataset(Xr.iloc[:cut], label=yr[:cut], weight=wr[:cut])
        d_te = lgb.Dataset(Xr.iloc[cut:], label=yr[cut:], weight=wr[cut:],
                           reference=d_tr)
        b = lgb.train(params, d_tr, num_boost_round=n_rounds,
                      valid_sets=[d_te],
                      callbacks=[lgb.early_stopping(100, verbose=False),
                                 lgb.log_evaluation(0)])
        # Refit on all regime data with median-best iter * 1.0
        b_full = lgb.train(params,
                           lgb.Dataset(Xr, label=yr, weight=wr),
                           num_boost_round=b.best_iteration)
        # Calibrate on the held-out 20%
        oof_p = b.predict(Xr.iloc[cut:], num_iteration=b.best_iteration)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_p, yr[cut:])
        boosters[label] = b_full
        calibrators[label] = iso
        print(f"  {label}: trained on {mask.sum()} rows, "
              f"best_iter={b.best_iteration}")

    return {"boosters": boosters, "calibrators": calibrators}


# ─── R-multiple regression head ────────────────────────────────────────────

def _train_rmultiple(X, r, weights, pkf, t1, params, n_rounds, feats):
    """Predict expected R-multiple — used for Kelly sizing at inference."""
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error

    print("\n=== R-multiple regressor (for Kelly sizing) ===")
    # Use Huber-like (regression_l1) — robust to outlier R values
    reg_params = {**params, "objective": "regression_l1",
                  "metric": "mae"}
    reg_params.pop("scale_pos_weight", None)
    # Monotone constraints not allowed with regression_l1
    reg_params.pop("monotone_constraints", None)
    reg_params.pop("monotone_constraints_method", None)

    # CV with median best_iter
    best_iters = []
    fold_idx = list(pkf.split(t1))
    for fi, (tr, te) in enumerate(fold_idx, 1):
        if len(tr) == 0 or len(te) == 0:
            continue
        # Mask each side independently for finite r
        tr_ok = np.isfinite(r[tr])
        te_ok = np.isfinite(r[te])
        if tr_ok.sum() < 50 or te_ok.sum() < 50:
            continue
        tr = tr[tr_ok]
        te = te[te_ok]
        d_tr = lgb.Dataset(X.iloc[tr], label=r[tr], weight=weights[tr])
        d_te = lgb.Dataset(X.iloc[te], label=r[te], weight=weights[te],
                           reference=d_tr)
        b = lgb.train(reg_params, d_tr, num_boost_round=n_rounds,
                      valid_sets=[d_te],
                      callbacks=[lgb.early_stopping(100, verbose=False),
                                 lgb.log_evaluation(0)])
        best_iters.append(b.best_iteration)
        p = b.predict(X.iloc[te], num_iteration=b.best_iteration)
        mae = mean_absolute_error(r[te], p)
        print(f"  fold {fi}: MAE={mae:.3f}, best_iter={b.best_iteration}")

    bi = int(np.median(best_iters)) if best_iters else int(n_rounds * 0.5)
    final = lgb.train(reg_params, lgb.Dataset(X, label=r, weight=weights),
                      num_boost_round=bi)
    return {"booster": final, "features": feats}


# ─── Holdout evaluation ───────────────────────────────────────────────────

def _holdout_eval(holdout_df: pd.DataFrame, cfg: dict) -> dict:
    """Evaluate the just-saved model on the never-seen holdout set.

    Reports honest forward-looking metrics: AUC, AP, prec@top-K, decile
    hit rate, R-multiple per decile, threshold sweep.
    """
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 brier_score_loss)

    holdout_df = holdout_df.dropna(subset=["y"]).copy()
    feats = feature_list(holdout_df)
    X = holdout_df[feats].astype(float)
    y = holdout_df["y"].astype(int).values
    r = holdout_df["r_multiple"].astype(float).values

    probs = []
    for _, row in X.iterrows():
        out = predict(row.to_dict())
        probs.append(out.get("prob_blend", out.get("prob_calibrated", 0.5)))
    probs = np.array(probs)

    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    ap = average_precision_score(y, probs)
    brier = brier_score_loss(y, probs)
    base = float(y.mean())

    print(f"\n  Holdout: n={len(y):,}, base rate={base*100:.1f}%")
    print(f"  AUC={auc:.4f}  AP={ap:.4f}  Brier={brier:.4f}")

    # Decile analysis on holdout
    order = np.argsort(probs)[::-1]
    deciles = np.array_split(order, 10)
    print("\n  Decile (0=top, 9=bottom):")
    print(f"  {'dec':>3} {'n':>5} {'mean_prob':>9} {'hit_rate':>8} "
          f"{'mean_R':>7} {'total_R':>8}")
    for di, idx in enumerate(deciles):
        if len(idx) == 0:
            continue
        print(f"  {di:>3} {len(idx):>5} {probs[idx].mean():>9.3f} "
              f"{y[idx].mean():>8.3f} {r[idx].mean():>7.3f} {r[idx].sum():>8.1f}")

    # Threshold sweep — what user really cares about
    print("\n  Threshold sweep:")
    print(f"  {'thr':>5} {'n_trades':>9} {'hit_rate':>8} {'mean_R':>7} {'total_R':>8}")
    for thr in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
        sel = probs >= thr
        n = int(sel.sum())
        if n == 0:
            print(f"  {thr:>5.2f} {0:>9d} {'-':>8} {'-':>7} {'-':>8}")
            continue
        hit = y[sel].mean()
        mr = r[sel].mean()
        tr = r[sel].sum()
        print(f"  {thr:>5.2f} {n:>9d} {hit:>8.3f} {mr:>7.3f} {tr:>8.1f}")

    # Top-K precision
    print("\n  Top-K precision (by predicted prob):")
    for k in [10, 25, 50, 100]:
        if k > len(probs):
            continue
        top_k = order[:k]
        print(f"   top-{k:>3}: hit_rate={y[top_k].mean():.3f}  "
              f"mean_R={r[top_k].mean():.3f}  total_R={r[top_k].sum():.1f}")

    return {"auc": auc, "ap": ap, "brier": brier, "base_rate": base}


# ─── Inference ─────────────────────────────────────────────────────────────

def predict(features_row: dict, model_dir: Path = MODELS_DIR) -> dict:
    """Load model(s) and predict for one new setup.

    Returns dict with keys:
      prob_raw, prob_calibrated  — primary LGBM (always present)
      prob_stack                 — stacked LGBM+XGB+CAT (if stacker.pkl exists)
      prob_regime                — regime-conditioned (if regime models exist)
      prob_blend                 — weighted blend of all available
      r_multiple_pred            — expected R (if regressor exists)
      kelly_fraction             — Kelly fraction (if r_multiple_pred set)
    """
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(model_dir / "lgbm.txt"))
    with open(model_dir / "calibrator.pkl", "rb") as f:
        iso = pickle.load(f)
    feats = json.load(open(model_dir / "features.json"))
    X = pd.DataFrame([{c: features_row.get(c, np.nan) for c in feats}]).astype(float)
    raw = float(booster.predict(X)[0])
    cal = float(iso.transform([raw])[0])
    out = {"prob_raw": raw, "prob_calibrated": cal}

    probs_for_blend = [cal]

    # Stacking
    stacker_p = model_dir / "stacker.pkl"
    if stacker_p.exists():
        with open(stacker_p, "rb") as f:
            st = pickle.load(f)
        # Sanitize ±inf for non-LGBM learners
        X_safe = X.replace([np.inf, -np.inf], np.nan)
        meta_inputs = []
        for k in st["base_cols"]:
            m = st["base_models"][k]
            if k == "lgbm":
                p_b = float(m.predict(X)[0])
            else:
                p_b = float(m.predict_proba(X_safe)[0, 1])
            meta_inputs.append(p_b)
        import numpy as _np
        stack_raw = float(st["meta"].predict_proba(_np.array(meta_inputs).reshape(1, -1))[0, 1])
        stack_cal = float(st["calibrator"].transform([stack_raw])[0])
        out["prob_stack"] = stack_cal
        probs_for_blend.append(stack_cal)

    # Regime model
    bull_p = model_dir / "lgbm_regime_bull.txt"
    bear_p = model_dir / "lgbm_regime_bear.txt"
    cal_p = model_dir / "regime_calibrators.pkl"
    if bull_p.exists() and bear_p.exists() and cal_p.exists():
        regime_flag = features_row.get("bench_above_200dma", 1)
        if pd.isna(regime_flag) or regime_flag is None:
            regime_flag = 1
        regime_key = "bull" if int(regime_flag) == 1 else "bear"
        b_r = lgb.Booster(model_file=str(model_dir / f"lgbm_regime_{regime_key}.txt"))
        with open(cal_p, "rb") as f:
            cals = pickle.load(f)
        raw_r = float(b_r.predict(X)[0])
        cal_r = float(cals[regime_key].transform([raw_r])[0]) \
            if regime_key in cals else raw_r
        out["prob_regime"] = cal_r
        probs_for_blend.append(cal_r)

    # Blend (simple mean of all calibrated probabilities)
    out["prob_blend"] = float(np.mean(probs_for_blend))

    # R-multiple regression
    rmult_p = model_dir / "rmult_regressor.txt"
    if rmult_p.exists():
        rb = lgb.Booster(model_file=str(rmult_p))
        rmult = float(rb.predict(X)[0])
        out["r_multiple_pred"] = rmult
        # Kelly fraction for asymmetric bet:
        # Expected payoff: prob_blend * upper_R + (1-prob_blend) * (-1)
        # We use rmult as estimated b (gain/loss ratio if win).
        # f* = (p*b - q) / b   where b = max(rmult, eps), q = 1-p
        b_kelly = max(abs(rmult), 0.5)
        p_eff = out["prob_blend"]
        kelly = (p_eff * b_kelly - (1 - p_eff)) / b_kelly
        out["kelly_fraction"] = float(max(0.0, min(kelly, 0.25)))  # cap 25%

    return out
