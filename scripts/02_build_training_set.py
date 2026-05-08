"""Phase 2 — build the labeled training corpus by replaying history.

Purpose:
  Produce the (features, label) parquet that the ML trainer
  (`scripts/03_train_model.py`) consumes.

How it works:
  For every symbol in the universe:
    1. Walk forward bar-by-bar through `data/ohlcv/<SYM>.parquet`.
    2. At each bar, run the rule-based primary detector
       (`src.setup_detector.detect_resistance`) on data <= that bar.
    3. If a setup is found AND no open setup is already tracked for
       the same symbol within `cooldown_days`, emit one training row
       containing:
         - identity:  symbol, asof, exit_date
         - features:  ~110 cols from `src.features.make_features`
                      (incl. sector RS + fundamentals + regime)
         - label:     triple-barrier outcome computed from FUTURE bars
                      (`src.labeling.triple_barrier_label`).
    4. Drop rows where forward history is insufficient to label.
  Multiprocessing via ProcessPoolExecutor (one symbol per worker).

Data sources (all on disk):
  data/ohlcv/*.parquet            OHLCV (Angel One)
  data/benchmark_NIFTY.parquet    benchmark for RS / regime features
  data/fundamentals.parquet       sector mapping + fund_* features
  data/sector_indices.parquet     sector RS features
  configs/default.yaml::setup,labeling   detector / barrier params

Outputs:
  data/training/setups.parquet    (default ~24k rows × 119 cols, ~14 MB)

How to run:
  python scripts/02_build_training_set.py
  python scripts/02_build_training_set.py --workers 8
  python scripts/02_build_training_set.py --out data/training/setups.parquet

Cadence:
  Quarterly (after #01 refresh) or whenever `src/features.py` /
  `src/setup_detector.py` change.

Notes:
  * v2 hardened gates dropped raw-setup count from ~242k → ~24k.
  * `WARMUP_BARS = 250` ensures at least ~1 year of history before the
    first scan; symbols with less data are skipped.
  * Idempotent: re-runs overwrite the parquet.
"""

import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import load_config, OHLCV_DIR, TRAINING_DIR, DATA_DIR
from src.yf_ingestion import load_symbol
from src.data_ingestion import fetch_benchmark
from src.setup_detector import detect_resistance
from src.features import make_features
from src.labeling import triple_barrier_label, to_binary
from src.fundamentals import load as load_fund
from src.sectors import load_sector_indices, symbol_to_sector_map

WARMUP_BARS = 250          # need at least 1y of data before scanning
SCAN_STRIDE_DAYS = 1       # check every trading day
COOLDOWN_DAYS = 15         # don't record same symbol twice within this


def _scan_symbol(args) -> list[dict]:
    """Worker: scan one symbol across history, return list of training rows."""
    symbol, bench_df, sector_idx_df, sym_to_sector, fund_map, cfg = args
    df = load_symbol(symbol)
    if df is None or len(df) < WARMUP_BARS + 60:
        return []

    df = df.sort_index()
    bench = bench_df["Close"] if bench_df is not None else None
    sec_name = sym_to_sector.get(symbol) if sym_to_sector else None
    sector_idx = (sector_idx_df[sec_name] if (sector_idx_df is not None
                  and sec_name and sec_name in sector_idx_df.columns) else None)
    fund_row = fund_map.get(symbol) if fund_map else None

    lb_cfg = cfg["labeling"]
    setup_cfg = cfg["setup"]

    rows = []
    last_recorded: pd.Timestamp | None = None
    # Walk forward; skip warmup
    asof_dates = df.index[WARMUP_BARS::SCAN_STRIDE_DAYS]

    for asof in asof_dates:
        if last_recorded is not None and (asof - last_recorded).days < COOLDOWN_DAYS:
            continue
        sub = df.loc[:asof]
        bench_sub = bench.loc[:asof] if bench is not None else None
        try:
            setup = detect_resistance(
                sub,
                res_lookback=setup_cfg.get("lookback_days", 600),
                base_min_days=setup_cfg.get("base_min_days", 30),
                res_band_pct=setup_cfg.get("res_band_pct", 0.035),
                proximity_max_pct=setup_cfg.get("proximity_max_pct", 0.05),
                min_touches=setup_cfg.get("min_touches", 2),
                # v2 quality gates
                enforce_quality_gates=setup_cfg.get("enforce_quality_gates", True),
                max_atr_pct=setup_cfg.get("max_atr_pct", 6.0),
                max_rvol_dryup_5d=setup_cfg.get("max_rvol_dryup_5d", 1.20),
                min_rs_3m_pct=setup_cfg.get("min_rs_3m_pct", 0.0),
                max_dist_52w_pct=setup_cfg.get("max_dist_52w_pct", 12.0),
                min_dollar_vol_cr=setup_cfg.get("min_dollar_vol_cr", 0.5),
                bench=bench_sub,
            )
        except Exception:
            continue
        if setup is None:
            continue

        # Label using FUTURE bars
        try:
            lab = triple_barrier_label(
                df, asof,
                upper_atr=lb_cfg.get("upper_barrier_atr", 3.0),
                lower_atr=lb_cfg.get("lower_barrier_atr", 1.0),
                time_days=lb_cfg.get("time_barrier_days", 10),
                atr_period=lb_cfg.get("atr_period", 20),
                resistance=setup["R"],
                require_close_above_R=lb_cfg.get("require_close_above_R", True),
                confirm_within_bars=5,
            )
        except Exception:
            continue
        if lab.get("label") is None:
            continue

        # Features (no look-ahead; uses sub only)
        try:
            sec_sub = sector_idx.loc[:asof] if sector_idx is not None else None
            feats = make_features(sub, setup, bench_sub,
                                  sector_idx=sec_sub, fund_row=fund_row)
        except Exception:
            continue

        row = {
            "symbol": symbol,
            "asof": asof,
            "exit_date": lab["exit_date"],
            "label_raw": lab["label"],
            "y": to_binary(lab["label"]),
            "r_multiple": lab["r_multiple"],
            "days_held": lab["days_held"],
            "R": setup["R"],
            "close_at_setup": float(sub["Close"].iloc[-1]),
            **feats,
        }
        rows.append(row)
        last_recorded = asof

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-symbols", type=int, default=None,
                    help="Limit (for smoke testing).")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(TRAINING_DIR / "setups.parquet"))
    args = ap.parse_args()

    cfg = load_config()
    syms = sorted(p.stem for p in OHLCV_DIR.glob("*.parquet"))
    if args.max_symbols:
        syms = syms[:args.max_symbols]
    print(f"Scanning {len(syms)} symbols across history ...")

    print("Loading benchmark ...")
    cache_p = OHLCV_DIR.parent / "benchmark_NIFTY.parquet"
    if cache_p.exists():
        bench = pd.read_parquet(cache_p)["Close"]
    else:
        bench = fetch_benchmark(
            pd.Timestamp(cfg["data"]["history_start"]).date(),
            pd.Timestamp.today().date(),
        )
        if bench is not None and not bench.empty:
            bench.to_frame("Close").to_parquet(cache_p)
    bench_df = bench.to_frame("Close") if isinstance(bench, pd.Series) else bench
    print(f"  benchmark rows: {0 if bench_df is None else len(bench_df)}")

    # ─── Sector indices + fundamentals (optional, NaN-safe) ──────────────
    sector_idx_df = load_sector_indices()
    sym_to_sector = symbol_to_sector_map()
    fund = load_fund()
    if sector_idx_df is not None:
        print(f"  sector indices: {sector_idx_df.shape[1]} sectors, "
              f"{len(sector_idx_df)} days")
    else:
        print("  sector indices: NONE (run scripts/06_fetch_fundamentals.py)")
    if fund is not None and len(fund):
        fund_map = fund.set_index("symbol").to_dict(orient="index")
        print(f"  fundamentals:   {len(fund)} symbols")
    else:
        fund_map = {}
        print("  fundamentals:   NONE (run scripts/06_fetch_fundamentals.py)")

    t0 = time.time()
    all_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_scan_symbol,
                          (s, bench_df, sector_idx_df, sym_to_sector,
                           fund_map, cfg)): s for s in syms}
        done = 0
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                rows = fut.result()
            except Exception as e:
                rows = []
                print(f"  {sym}: ERROR {e}")
            all_rows.extend(rows)
            done += 1
            if done % 50 == 0 or done == len(syms):
                print(f"  {done}/{len(syms)} symbols, "
                      f"setups so far: {len(all_rows)}, "
                      f"elapsed: {time.time() - t0:.0f}s")

    if not all_rows:
        print("NO SETUPS FOUND. Check thresholds.")
        return

    df_out = pd.DataFrame(all_rows)
    df_out = df_out.sort_values("asof").reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(args.out)

    print(f"\n✓ Wrote {len(df_out):,} setups → {args.out}")
    print(f"  Date range: {df_out['asof'].min().date()} → {df_out['asof'].max().date()}")
    print(f"  Symbols:    {df_out['symbol'].nunique()}")
    print(f"  Base rate:  {df_out['y'].mean() * 100:.1f}% positive")
    print(f"  Mean R:     {df_out['r_multiple'].mean():.2f}")
    print(f"  Median R:   {df_out['r_multiple'].median():.2f}")
    print(f"  Hit rate (raw label): "
          f"+1: {(df_out['label_raw'] == 1).mean() * 100:.1f}%, "
          f"-1: {(df_out['label_raw'] == -1).mean() * 100:.1f}%, "
          f" 0: {(df_out['label_raw'] == 0).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
