"""Phase 5 — daily production scan (PRIMARY → META → RANK → chain).

Purpose:
  End-of-day script that produces today's actionable watchlist and
  trigger list, then chains the sector-rotation pipeline so a single
  command yields all daily artefacts.

How it works:
  1. Tail-update OHLCV cache (Angel One) for every cached symbol unless
     `--no-update` is passed.
  2. For each symbol:
       a. Load `data/ohlcv/<SYM>.parquet`.
       b. Detect a horizontal-resistance setup
          (`src.setup_detector.detect_resistance`) using
          `configs/default.yaml::setup` thresholds.
       c. If a setup is found, build the ~110-feature row
          (`src.features.make_features` with bench / sector_idx /
           fundamentals).
       d. Score with `src.model.predict(...)` (LightGBM + isotonic +
          optional stacker / regime blend).
  3. Drop rows below the watchlist threshold; rank remaining setups by
     blended probability and compute risk plan (entry / stop / target /
     R:R) per setup.
  4. Save scan + trigger CSVs.
  5. Chain `scripts/08_sector_rotation.py` (cadence row #7) unless
     `--no-rotation` is passed. The chained run produces the combined
     `MLRotation_*.xlsx` workbook with all 5 sheets.

Data sources:
  Angel One SmartAPI (OHLCV tail-update)
  data/ohlcv/*.parquet,  data/models/*,  data/benchmark_NIFTY.parquet,
  data/sector_indices.parquet,  data/fundamentals.parquet
  configs/default.yaml

Outputs (under Output/):
  scan_<YYYYMMDD>.csv      full ranked watchlist (prob >= watchlist thr)
  triggers_<YYYYMMDD>.csv  high-conviction subset (prob >= trigger thr)
  + chained `08_sector_rotation.py` artefacts (sector_rotation_*.csv,
    sector_rotation.parquet, watchlist_in_leaders_*.csv,
    MLRotation_<YYYYMMDD>_<HHMMSS>.xlsx)

How to run:
  python scripts/05_daily_scan.py                        # production daily run
  python scripts/05_daily_scan.py --no-update            # use cached OHLCV
  python scripts/05_daily_scan.py --max-symbols 100      # smoke test
  python scripts/05_daily_scan.py --no-rotation          # skip 08 chain
  python scripts/05_daily_scan.py --rotation-no-refresh  # 08 reuses cache

Cadence:
  Daily, automatic (cron / launchd) after market close.
"""

import sys
import json
import pickle
import datetime
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import load_config, OHLCV_DIR, MODELS_DIR, DATA_DIR, OUTPUT_DIR
from src.yf_ingestion import load_symbol, update_history
from src.setup_detector import detect_resistance
from src.features import make_features
from src.labeling import atr_at
from src.fundamentals import load as load_fund
from src.sectors import load_sector_indices, symbol_to_sector_map
from src.model import predict as model_predict


SCANS_DIR = OUTPUT_DIR  # back-compat alias


def _risk_plan(df, setup, atr20, lower_atr=1.0, upper_atr=2.0):
    last = float(df["Close"].iloc[-1])
    R = setup["R"]
    base_low = float(df.loc[setup["base_start"]:]["Low"].min())
    swing_low = float(df["Low"].iloc[-20:].min())
    stop = max(swing_low * 0.99, last - lower_atr * atr20)
    height = R - base_low
    target = R + height
    risk = last - stop
    reward = target - last
    return {
        "entry": round(last, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "risk_pct": round(risk / last * 100, 2) if last else None,
        "reward_pct": round(reward / last * 100, 2) if last else None,
        "rr": round(reward / risk, 2) if risk > 0 else None,
    }


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-update", action="store_true",
                    help="Skip yfinance tail-update (use cached data as-is)")
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--no-rotation", action="store_true",
                    help="Skip the sector-rotation chain step at the end "
                         "(scripts/08_sector_rotation.py).")
    ap.add_argument("--rotation-no-refresh", action="store_true",
                    help="When chaining 08_sector_rotation, pass --no-refresh "
                         "to reuse cached OHLCV/benchmark.")
    args = ap.parse_args()

    cfg = load_config()
    syms = sorted(p.stem for p in OHLCV_DIR.glob("*.parquet"))
    if args.max_symbols:
        syms = syms[:args.max_symbols]
    print(f"Universe: {len(syms)} symbols")

    if not args.no_update:
        print("Updating OHLCV cache (yfinance tail-fetch) ...")
        n_up = update_history(syms, lookback_days=10)
        print(f"  updated {n_up} symbols")

    # Load benchmark
    bench_p = DATA_DIR / "benchmark_NIFTY.parquet"
    bench = pd.read_parquet(bench_p)["Close"] if bench_p.exists() else None

    # Sector + fundamentals (optional)
    sector_idx_df = load_sector_indices()
    sym_to_sector = symbol_to_sector_map()
    fund = load_fund()
    fund_map = fund.set_index("symbol").to_dict(orient="index") if fund is not None and len(fund) else {}

    # Load model
    booster = lgb.Booster(model_file=str(MODELS_DIR / "lgbm.txt"))
    with open(MODELS_DIR / "calibrator.pkl", "rb") as f:
        iso = pickle.load(f)
    feats_list = json.load(open(MODELS_DIR / "features.json"))
    print(f"Model loaded: {len(feats_list)} features, "
          f"best_iter={booster.best_iteration}")

    setup_cfg = cfg["setup"]
    lb_cfg = cfg["labeling"]
    thr_watch = cfg["inference"].get("prob_threshold_watchlist", 0.50)
    thr_trig = cfg["inference"].get("prob_threshold_trigger", 0.65)

    rows = []
    for i, sym in enumerate(syms, start=1):
        df = load_symbol(sym)
        if df is None or len(df) < 250:
            continue
        df = df.sort_index()
        asof = df.index[-1]
        bsub = bench.loc[:asof] if bench is not None else None
        try:
            setup = detect_resistance(
                df,
                res_lookback=setup_cfg.get("lookback_days", 600),
                base_min_days=setup_cfg.get("base_min_days", 30),
                res_band_pct=setup_cfg.get("res_band_pct", 0.035),
                proximity_max_pct=setup_cfg.get("proximity_max_pct", 0.05),
                min_touches=setup_cfg.get("min_touches", 2),
                enforce_quality_gates=setup_cfg.get("enforce_quality_gates", True),
                max_atr_pct=setup_cfg.get("max_atr_pct", 6.0),
                max_rvol_dryup_5d=setup_cfg.get("max_rvol_dryup_5d", 1.20),
                min_rs_3m_pct=setup_cfg.get("min_rs_3m_pct", 0.0),
                max_dist_52w_pct=setup_cfg.get("max_dist_52w_pct", 12.0),
                min_dollar_vol_cr=setup_cfg.get("min_dollar_vol_cr", 0.5),
                bench=bsub,
            )
        except Exception:
            continue
        if setup is None:
            continue
        try:
            sec_name = sym_to_sector.get(sym)
            sec_sub = (sector_idx_df[sec_name].loc[:asof]
                       if sector_idx_df is not None and sec_name
                       and sec_name in sector_idx_df.columns else None)
            f = make_features(df, setup, bsub,
                              sector_idx=sec_sub,
                              fund_row=fund_map.get(sym))
        except Exception:
            continue

        X = pd.DataFrame([{c: f.get(c, np.nan) for c in feats_list}]).astype(float)
        # Use unified predictor: returns prob_calibrated (LGBM), and if
        # available, prob_stack, prob_regime, prob_blend, r_multiple_pred,
        # kelly_fraction.
        try:
            pr = model_predict(f)
        except Exception:
            raw = float(booster.predict(X)[0])
            pr = {
                "prob_raw": raw,
                "prob_calibrated": float(iso.transform([raw])[0]),
                "prob_blend": float(iso.transform([raw])[0]),
            }

        # Use blended probability for thresholding (best signal available)
        cal = pr.get("prob_blend", pr["prob_calibrated"])
        raw = pr["prob_raw"]

        if cal < thr_watch:
            continue

        a20 = atr_at(df, asof, n=lb_cfg.get("atr_period", 20)) or 0.0
        risk = _risk_plan(df, setup, a20,
                          lower_atr=lb_cfg.get("lower_barrier_atr", 1.0),
                          upper_atr=lb_cfg.get("upper_barrier_atr", 2.0))

        rows.append({
            "symbol": sym,
            "asof": asof.date().isoformat(),
            "close": round(float(df["Close"].iloc[-1]), 2),
            "resistance": round(setup["R"], 2),
            "distance_pct": round(setup["distance_pct"] * 100, 2),
            "base_days": setup["base_len_days"],
            "touches": setup["touches"],
            "is_52w_high": setup["is_52w_high"],
            "prob_raw": round(raw, 4),
            "prob": round(cal, 4),
            "prob_lgbm": round(pr["prob_calibrated"], 4),
            "prob_stack": round(pr.get("prob_stack", float("nan")), 4)
                if "prob_stack" in pr else None,
            "prob_regime": round(pr.get("prob_regime", float("nan")), 4)
                if "prob_regime" in pr else None,
            "r_mult_pred": round(pr.get("r_multiple_pred", float("nan")), 3)
                if "r_multiple_pred" in pr else None,
            "kelly_frac": round(pr.get("kelly_fraction", float("nan")), 4)
                if "kelly_fraction" in pr else None,
            **risk,
            "atr_20": round(a20, 2),
            "rvol_20": round(f.get("rvol_20", np.nan), 2),
            "rs_3m": round(f.get("rs_3m", np.nan), 2),
            "adx_14": round(f.get("adx_14", np.nan), 1),
            "ttm_squeeze": int(f.get("ttm_squeeze", 0) or 0),
            "pocket_pivot": int(f.get("pocket_pivot", 0) or 0),
            "wyckoff_spring": int(f.get("wyckoff_spring", 0) or 0),
            "coiled_score": round(f.get("coiled_spring_score", np.nan) or np.nan, 1),
        })

        if i % 200 == 0:
            print(f"  {i}/{len(syms)} scanned, {len(rows)} setups so far")

    if not rows:
        print("No setups today.")
    else:
        out = pd.DataFrame(rows).sort_values("prob", ascending=False)
        SCANS_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.date.today().strftime("%Y%m%d")
        full = SCANS_DIR / f"scan_{today}.csv"
        trig = SCANS_DIR / f"triggers_{today}.csv"
        out.to_csv(full, index=False)
        out[out["prob"] >= thr_trig].to_csv(trig, index=False)

        print(f"\n✓ Watchlist: {len(out):,} setups → {full}")
        print(f"  Triggers (prob ≥ {thr_trig}): {(out['prob'] >= thr_trig).sum()} → {trig}")
        print("\nTop 20:")
        cols = ["symbol", "close", "resistance", "distance_pct",
                "prob", "rr", "rvol_20", "rs_3m"]
        if "kelly_frac" in out.columns and out["kelly_frac"].notna().any():
            cols += ["kelly_frac", "r_mult_pred"]
        print(out.head(20)[cols].to_string(index=False))

    # ─── Chain: invoke sector-rotation pipeline (cadence row #7) ──────
    if not args.no_rotation:
        print("\n" + "=" * 70)
        print("Chaining → scripts/08_sector_rotation.py")
        print("=" * 70)
        try:
            sr_argv: list = []
            if args.rotation_no_refresh:
                sr_argv.append("--no-refresh")
            # Import lazily so a missing optional dep doesn't break the
            # primary scan run.
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "sector_rotation_runner",
                Path(__file__).resolve().parent / "08_sector_rotation.py",
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main(sr_argv)
        except SystemExit:
            # argparse in the chained script may call sys.exit on --help etc.
            pass
        except Exception as e:
            print(f"  ! Sector-rotation chain failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
