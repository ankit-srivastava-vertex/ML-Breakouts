"""Phase 1 — build / refresh the per-symbol OHLCV cache (Angel One).

Purpose:
  Bootstrap or refresh `data/ohlcv/<SYMBOL>.parquet` for the entire
  active NSE-EQ universe so every later stage of the ML pipeline (and
  the daily scanner / sector rotation) has consistent local data.

How it works:
  [1/2] Derive the active universe from the existing OHLCV cache via
        `src.data_ingestion.derive_universe_from_ohlcv_cache(...)`.
        On a fresh checkout the cache is empty — in that case seed it
        once by passing an explicit `--symbols` list (or by piggy-backing
        on the daily-scan cron run).
  [2/2] Bulk download / refresh full history per symbol via
        `src.yf_ingestion.download_history(...)` which routes through
        `legacy_scanner/data_provider.py` (Angel One SmartAPI primary;
        jugaad / yfinance fallback only when ANGEL_ONLY=False).

Data sources:
  Angel One SmartAPI (primary), jugaad-data / yfinance (fallback).
  Legacy NSE bhavcopy archives are WAF-blocked and no longer used.

Outputs:
  data/ohlcv/<SYMBOL>.parquet  (Open / High / Low / Close / Volume)

How to run:
  python scripts/01_build_dataset.py --start 2018-01-01
  python scripts/01_build_dataset.py --smoke              # tiny sanity run

Cadence:
  Quarterly (refresh as part of routine retrain). Daily updates are
  handled incrementally by `scripts/05_daily_scan.py`.

Notes:
  * The previous `--workers` / `--seed-bhavcopy-days` flags were
    removed when the bhavcopy code path was deleted.
  * Re-runs are idempotent and overwrite existing parquets.
"""

import sys
import argparse
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import load_config
from src.data_ingestion import derive_universe_from_ohlcv_cache
from src.yf_ingestion import download_history, load_symbol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None,
                    help="YYYY-MM-DD (default: config.data.history_start)")
    ap.add_argument("--end", default=None,
                    help="YYYY-MM-DD (default: today)")
    ap.add_argument("--batch-size", type=int, default=50,
                    help="Angel One batch size (tickers per request)")
    ap.add_argument("--smoke", action="store_true",
                    help="quick run: 10 symbols, 6 months")
    args = ap.parse_args()

    cfg = load_config()
    start = datetime.date.fromisoformat(
        args.start or cfg["data"]["history_start"])
    end = datetime.date.fromisoformat(
        args.end or datetime.date.today().isoformat())

    print("=" * 70)
    print(f"  Phase 1 — Historical OHLCV build  {start} → {end}")
    print("=" * 70)

    # Step 1: derive universe from existing OHLCV cache
    print("\n[1/2] Deriving active NSE-EQ universe from OHLCV cache ...")
    syms = derive_universe_from_ohlcv_cache(
        min_trading_days=60, recent_window_days=30, min_recent_days=15,
    )
    print(f"  Universe size: {len(syms)}")
    if not syms:
        print("  OHLCV cache is empty. Seed it first via the daily-scan "
              "pipeline (scripts/05_daily_scan.py) or supply symbols "
              "explicitly. Aborting.")
        return

    if args.smoke:
        syms = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN",
                "ICICIBANK", "ITC", "LT", "BHARTIARTL", "AXISBANK"]
        start = end - datetime.timedelta(days=180)
        print(f"  SMOKE: {len(syms)} symbols, {start} → {end}")

    # Step 2: bulk download via Angel One
    print(f"\n[2/2] Angel One bulk download (batch={args.batch_size}) ...")
    n_ok = download_history(syms, start=start, end=end,
                            batch_size=args.batch_size)
    print(f"\n  Cached {n_ok}/{len(syms)} symbols")

    # Spot-check liquid names
    print()
    for s in ["RELIANCE", "TCS", "INFY", "HDFCBANK"]:
        df = load_symbol(s)
        if df is None or df.empty:
            print(f"  {s}: NOT CACHED")
        else:
            print(f"  {s}: {len(df)} rows, "
                  f"{df.index.min().date()} → {df.index.max().date()}, "
                  f"last close = {df['Close'].iloc[-1]:.2f}")


if __name__ == "__main__":
    main()
