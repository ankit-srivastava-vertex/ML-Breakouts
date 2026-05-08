"""Phase 8 — daily sector + custom-theme rotation report + combined workbook.

Purpose:
  Produce TWO daily rotation snapshots side-by-side:
    (a) GICS sector rotation  — yfinance sector mapping (12 sectors)
    (b) Custom theme rotation — user-defined baskets in
        src/index_constituents.json (e.g. Wires&Cables, Forgings,
        Aerospace&Defense ...)
  For each, print the leader board, cross-reference today's ML watchlist
  against the top-N leaders, and merge every artefact into a single
  multi-sheet xlsx for review.

How it works:
  1. `src.sector_rotation.run_full_pipeline(...)` orchestrates:
       - Fundamentals (yfinance, cached)
       - OHLCV refresh for the FULL universe (Angel One; force-refresh
         by default)
       - ^NSEI benchmark refresh (Angel One)
       - Build sector indices (equal-weight)
       - Compute sector rotation snapshot at `--asof` (default = latest).
  2. Print sector top / bottom / rotating-in leaderboard.
  3. Save dated CSV + always-latest parquet snapshot for sectors.
  4. If today's `Output/scan_<YYYYMMDD>.csv` exists, intersect the ML
     watchlist with the top-N leading sectors and write
     `Output/watchlist_in_leaders_<YYYYMMDD>.csv`.
  5. `src.theme_rotation.run_full_pipeline(...)` then runs the same
     algorithm on the CUSTOM themes:
       - Loads themes from src/index_constituents.json.
       - Backfills any missing constituents into data/ohlcv/ via Angel
         One (force_refresh_ohlcv=False so we don't re-pull the world).
       - Builds equal-weight theme indices (data/theme_indices.parquet).
       - Computes theme rotation snapshot at `--asof`.
  6. Print theme leaderboard + save theme CSV / parquet.
  7. Intersect the ML watchlist with the top-N THEMES and write
     `Output/watchlist_in_theme_leaders_<YYYYMMDD>.csv`.
  8. Build `Output/MLRotation_<YYYYMMDD>_<HHMMSS>.xlsx` with up to 8 sheets:
       Scans / Triggers /
       Watchlist_in_Leaders        (sectors intersection)
       Watchlist_in_Theme_Leaders  (themes intersection)
       Sector_Rotation_Report  (top + bottom sectors)
       Sector_Rotation         (full sector table)
       Theme_Rotation_Report   (top + bottom themes)
       Theme_Rotation          (full theme table)
     Sheets are skipped gracefully when their source is absent.

Data sources:
  Angel One SmartAPI (OHLCV + ^NSEI), yfinance (fundamentals).
  src/index_constituents.json (custom themes).
  Reuses on-disk caches under `data/`.

Outputs (all under Output/):
  sector_rotation_<YYYYMMDD>.csv          dated sector rotation table
  sector_rotation.parquet                 always-latest sector snapshot
  watchlist_in_leaders_<YYYYMMDD>.csv     ML setups in top-N sectors
  theme_rotation_<YYYYMMDD>.csv           dated theme rotation table
  theme_rotation.parquet                  always-latest theme snapshot
  watchlist_in_theme_leaders_<YYYYMMDD>.csv  ML setups in top-N themes
  MLRotation_<YYYYMMDD>_<HHMMSS>.xlsx     combined multi-sheet workbook
And under data/:
  theme_indices.parquet                   wide DF: index=DATE, cols=theme

How to run:
  python scripts/08_sector_rotation.py                       # full refresh
  python scripts/08_sector_rotation.py --no-refresh          # use cached OHLCV
  python scripts/08_sector_rotation.py --asof 2026-04-30     # backfill a day
  python scripts/08_sector_rotation.py --top 7               # bigger leaderboard
  python scripts/08_sector_rotation.py --no-themes           # skip theme pipeline
  python scripts/08_sector_rotation.py --no-combined         # skip MLRotation xlsx

Programmatic invocation:
  Called automatically at the end of `scripts/05_daily_scan.py` via
  importlib so a single command runs the full daily pipeline (scan +
  sectors + themes + workbook).

Cadence:
  Daily, automatic (chained after the daily scan).

Notes:
  * Sector and theme pipelines are independent: a failure in the theme
    pipeline (e.g. Angel One outage, malformed JSON) is caught and the
    sector workbook is still written.
  * Theme breadth (`breadth_above_*dma`) may show NaN for tiny baskets
    (<5 members with 200d history) — by design, mirrors sector logic.
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
from src import theme_rotation as tr


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
                             in_top_df: pd.DataFrame | None,
                             theme_rot: pd.DataFrame | None = None,
                             in_top_theme_df: pd.DataFrame | None = None) -> Path:
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
    if in_top_theme_df is None or (in_top_theme_df is not None and in_top_theme_df.empty):
        in_top_theme_df = _safe_read_csv(
            OUTPUT_DIR / f"watchlist_in_theme_leaders_{asof_str}.csv"
        )

    lb_cols_all = ["rank", "mom_score", "rs_1m", "rs_3m", "rs_6m",
                   "rank_chg_5d", "rank_chg_20d",
                   "breadth_above_50dma", "breadth_above_200dma"]

    def _leader_view(rdf: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in lb_cols_all if c in rdf.columns]
        top = rdf.head(top_n)[cols].round(2).copy()
        top.insert(0, "section", "TOP")
        bot = rdf.tail(top_n)[cols].round(2).copy()
        bot.insert(0, "section", "BOTTOM")
        return pd.concat([top, bot])

    sector_leader_df = _leader_view(rot)

    sheets: list[tuple[str, pd.DataFrame]] = []
    if scan_df is not None:
        sheets.append(("Scans", scan_df))
    if trig_df is not None:
        sheets.append(("Triggers", trig_df))
    if in_top_df is not None and len(in_top_df):
        sheets.append(("Watchlist_in_Leaders", in_top_df))
    if in_top_theme_df is not None and len(in_top_theme_df):
        sheets.append(("Watchlist_in_Theme_Leaders", in_top_theme_df))
    sheets.append(("Sector_Rotation_Report", sector_leader_df.reset_index()))
    sheets.append(("Sector_Rotation", rot.reset_index()))
    if theme_rot is not None and not theme_rot.empty:
        sheets.append(("Theme_Rotation_Report",
                       _leader_view(theme_rot).reset_index()))
        sheets.append(("Theme_Rotation", theme_rot.reset_index()))

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
    ap.add_argument("--no-themes", action="store_true",
                    help="Skip the parallel custom-theme rotation pipeline.")
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

    # ─── Custom-theme rotation pipeline (parallel to GICS sectors) ────
    theme_rot: pd.DataFrame | None = None
    in_top_theme: pd.DataFrame | None = None
    if not args.no_themes:
        try:
            print("\n" + "=" * 70)
            print("Running custom-theme rotation pipeline ...")
            print("=" * 70)
            theme_rot = tr.run_full_pipeline(
                asof=asof,
                rebuild_indices=True,
                force_refresh_ohlcv=False,  # OHLCV already refreshed by sectors
            )
            print(f"\n  asof: {asof.date()},  themes: {len(theme_rot)}")
            tr.leaderboard(theme_rot, n=args.top)

            t_csv = OUTPUT_DIR / f"theme_rotation_{asof.strftime('%Y%m%d')}.csv"
            theme_rot.to_csv(t_csv)
            theme_rot.to_parquet(tr.THEME_ROTATION_PATH)
            print(f"\n✓ Saved snapshot → {t_csv}")
            print(f"✓ Saved latest   → {tr.THEME_ROTATION_PATH}")

            # Cross-reference scan with top-N themes
            if scan_p.exists():
                scan = pd.read_csv(scan_p)
                sym_to_theme = tr.symbol_to_theme_map()
                scan["theme"] = scan["symbol"].str.upper().map(sym_to_theme)
                top_themes = theme_rot.head(args.top).index.tolist()
                in_top_theme = scan[scan["theme"].isin(top_themes)].copy()
                if len(in_top_theme):
                    in_top_theme = in_top_theme.sort_values("prob", ascending=False)
                    print(f"\n  ── WATCHLIST in TOP-{args.top} THEMES "
                          f"(highest-conviction setups in leading custom themes) ──")
                    cols = [c for c in ["symbol", "theme", "close", "resistance",
                                        "distance_pct", "prob", "rs_3m", "rvol_20"]
                            if c in in_top_theme.columns]
                    print(in_top_theme[cols].head(20).to_string(index=False))
                    j_p = OUTPUT_DIR / f"watchlist_in_theme_leaders_{asof.strftime('%Y%m%d')}.csv"
                    in_top_theme.to_csv(j_p, index=False)
                    print(f"\n✓ Saved → {j_p}")
                else:
                    print(f"\n  No watchlist setups in top-{args.top} themes today.")
        except Exception as e:
            print(f"  ! Theme rotation failed: {e}")
            theme_rot = None
            in_top_theme = None

    # ─── Combined multi-sheet workbook ────────────────────────────────
    if not args.no_combined:
        try:
            _build_combined_workbook(asof, rot, args.top, in_top,
                                     theme_rot=theme_rot,
                                     in_top_theme_df=in_top_theme)
        except Exception as e:
            print(f"  ! Could not build combined workbook: {e}")


if __name__ == "__main__":
    main()

