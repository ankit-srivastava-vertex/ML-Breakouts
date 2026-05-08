"""Phase 8 — daily sector / theme rotation report + combined workbook.

Purpose:
  Produce the daily sector-rotation snapshot (per-sector returns, RS,
  breadth, momentum score, rank deltas), print the leader board,
  optionally cross-reference against today's ML watchlist, and combine
  every daily artefact into a single multi-sheet xlsx for review.

How it works:
  1. `src.sector_rotation.run_full_pipeline(...)` orchestrates:
       - Fundamentals (yfinance, cached)
       - OHLCV refresh (Angel One; force-refresh by default)
       - ^NSEI benchmark refresh (Angel One)
       - Build sector indices
       - Compute rotation snapshot at `--asof` (default = latest).
  2. Print top / bottom / rotating-in leaderboard.
  3. Save dated CSV + always-latest parquet snapshot.
  4. If today's `Output/scan_<YYYYMMDD>.csv` exists, join the ML
     watchlist against the top-N leading sectors and write
     `Output/watchlist_in_leaders_<YYYYMMDD>.csv`.
  5. Build `Output/MLRotation_<YYYYMMDD>_<HHMMSS>.xlsx` with up to 5 sheets:
       Scans / Triggers / Watchlist_in_Leaders /
       Sector_Rotation_Report (top + bottom view) / Sector_Rotation
     (sheets are skipped gracefully when their source CSV is absent).

Data sources:
  Angel One SmartAPI (OHLCV + ^NSEI), yfinance (fundamentals).
  Reuses on-disk caches under `data/`.

Outputs (under Output/):
  sector_rotation_<YYYYMMDD>.csv       dated archive of the rotation table
  sector_rotation.parquet              always-latest snapshot (fixed name)
  watchlist_in_leaders_<YYYYMMDD>.csv  ML setups in top-N sectors (if scan exists)
  MLRotation_<YYYYMMDD>_<HHMMSS>.xlsx  combined multi-sheet workbook

How to run:
  python scripts/08_sector_rotation.py                       # full refresh
  python scripts/08_sector_rotation.py --no-refresh          # use cached OHLCV
  python scripts/08_sector_rotation.py --asof 2026-04-30     # backfill a day
  python scripts/08_sector_rotation.py --top 7               # bigger leaderboard
  python scripts/08_sector_rotation.py --no-combined         # skip MLRotation xlsx

Programmatic invocation:
  Called automatically at the end of `scripts/05_daily_scan.py` (cadence
  row #7) via importlib so a single command runs the full daily pipeline.

Cadence:
  Daily, automatic (chained after the daily scan).
"""

from __future__ import annotations

import sys
import argparse
import datetime as _dt
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import DATA_DIR, OUTPUT_DIR
from src.sector_rotation import (
    compute_rotation, leaderboard, run_full_pipeline,
    ROTATION_PATH, symbol_to_sector_map,
)


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df if len(df) else None
    except Exception as e:
        print(f"  ! Could not read {path.name}: {e}")
        return None


def _build_combined_workbook(asof: pd.Timestamp,
                             rot: pd.DataFrame,
                             top_n: int,
                             in_top_df: pd.DataFrame | None) -> Path:
    """Combine all daily artefacts into a single multi-sheet xlsx."""
    asof_str = asof.strftime("%Y%m%d")
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"MLRotation_{ts}.xlsx"

    scan_df = _safe_read_csv(OUTPUT_DIR / f"scan_{asof_str}.csv")
    trig_df = _safe_read_csv(OUTPUT_DIR / f"triggers_{asof_str}.csv")

    if in_top_df is None or in_top_df.empty:
        in_top_df = _safe_read_csv(
            OUTPUT_DIR / f"watchlist_in_leaders_{asof_str}.csv"
        )

    # Leaderboard sheet: top-N + bottom-N view
    lb_cols = [c for c in ["rank", "mom_score", "rs_1m", "rs_3m", "rs_6m",
                           "rank_chg_5d", "rank_chg_20d",
                           "breadth_above_50dma", "breadth_above_200dma"]
               if c in rot.columns]
    top = rot.head(top_n)[lb_cols].round(2).copy()
    top.insert(0, "section", "TOP")
    bot = rot.tail(top_n)[lb_cols].round(2).copy()
    bot.insert(0, "section", "BOTTOM")
    leaderboard_df = pd.concat([top, bot])

    sheets: list[tuple[str, pd.DataFrame]] = []
    if scan_df is not None:
        sheets.append(("Scans", scan_df))
    if trig_df is not None:
        sheets.append(("Triggers", trig_df))
    if in_top_df is not None and len(in_top_df):
        sheets.append(("Watchlist_in_Leaders", in_top_df))
    sheets.append(("Sector_Rotation_Report", leaderboard_df.reset_index()))
    sheets.append(("Sector_Rotation", rot.reset_index()))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        for name, df in sheets:
            df.to_excel(xw, sheet_name=name[:31], index=False)
    print(f"\n✓ Combined workbook → {out_path}  "
          f"({len(sheets)} sheets)")
    return out_path


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None,
                    help="YYYY-MM-DD (default = latest)")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--no-refresh", action="store_true",
                    help="Skip OHLCV/benchmark re-download; reuse cached parquets.")
    ap.add_argument("--no-combined", action="store_true",
                    help="Skip writing the combined MLRotation_*.xlsx workbook.")
    args = ap.parse_args(argv)

    asof = pd.Timestamp(args.asof) if args.asof else None

    print("Running sector-rotation pipeline ...")
    rot = run_full_pipeline(
        asof=asof,
        force_refresh_ohlcv=not args.no_refresh,
        force_refresh_benchmark=not args.no_refresh,
        rebuild_indices=True,
    )

    if asof is None:
        asof = pd.Timestamp.today().normalize()
    print(f"\n  asof: {asof.date()},  sectors: {len(rot)}")
    leaderboard(rot, n=args.top)

    # Save snapshot
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / f"sector_rotation_{asof.strftime('%Y%m%d')}.csv"
    rot.to_csv(out_csv)
    rot.to_parquet(ROTATION_PATH)
    print(f"\n✓ Saved snapshot → {out_csv}")
    print(f"✓ Saved latest   → {ROTATION_PATH}")

    # ─── Cross-reference with today's daily-scan watchlist ────────────
    in_top: pd.DataFrame | None = None
    scan_p = OUTPUT_DIR / f"scan_{asof.strftime('%Y%m%d')}.csv"
    if scan_p.exists():
        scan = pd.read_csv(scan_p)
        sym_to_sector = symbol_to_sector_map()
        scan["sector"] = scan["symbol"].map(sym_to_sector)
        top_sectors = rot.head(args.top).index.tolist()
        in_top = scan[scan["sector"].isin(top_sectors)].copy()
        if len(in_top):
            in_top = in_top.sort_values("prob", ascending=False)
            print(f"\n  ── WATCHLIST in TOP-{args.top} SECTORS "
                  f"(highest-conviction setups in leading themes) ──")
            cols = [c for c in ["symbol", "sector", "close", "resistance",
                                "distance_pct", "prob", "rs_3m", "rvol_20"]
                    if c in in_top.columns]
            print(in_top[cols].head(20).to_string(index=False))
            joined_p = OUTPUT_DIR / f"watchlist_in_leaders_{asof.strftime('%Y%m%d')}.csv"
            in_top.to_csv(joined_p, index=False)
            print(f"\n✓ Saved → {joined_p}")
        else:
            print(f"\n  No watchlist setups in top-{args.top} sectors today.")
    else:
        print(f"\n  (No scan_{asof.strftime('%Y%m%d')}.csv found — "
              f"run scripts/05_daily_scan.py first to cross-reference.)")

    # ─── Combined multi-sheet workbook ────────────────────────────────
    if not args.no_combined:
        try:
            _build_combined_workbook(asof, rot, args.top, in_top)
        except Exception as e:
            print(f"  ! Could not build combined workbook: {e}")


if __name__ == "__main__":
    main()

