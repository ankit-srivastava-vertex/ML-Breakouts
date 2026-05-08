"""
upperband_analyze.py — Daily NSE Upper-Circuit deep-dive pipeline.
==================================================================

Workflow (run once per trading day):

  1. Pull today's price-band-hitter list directly from NSE:
        https://www.nseindia.com/api/live-analysis-price-band-hitter
     (use ``--csv path.csv`` to fall back to a downloaded CSV instead).
     A snapshot of the JSON is saved to
     ``upper_band/analysis/<DATE>/nse_raw.json`` for replay.
  2. For every symbol, pull ~1 year of daily OHLCV via Angel One SmartAPI
     (uses ``legacy_scanner/angel_client.py``).
  3. Render a daily candle + volume chart with overlays (20/50/200 SMA,
     52-week high/low, today's circuit close) and save to::
         upper_band/charts/<DATE>/<SYMBOL>.png
  4. Compute a structured per-symbol technical feature row
     (gap, body, range, vol surge, RSI(14), ATR(14), distance from
     52w high/low, EMA stack, Bollinger position, base-length,
     pre-move pattern flags, etc.) and save to::
         upper_band/analysis/<DATE>/per_symbol.csv
  5. Aggregate cohort-level statistics into a Markdown summary at::
         upper_band/analysis/<DATE>/summary.md
     and append a one-line digest to::
         upper_band/analysis/_pattern_log.md
     so we can build pattern memory across many sessions.

Usage:
    python upperband_analyze.py                         # live NSE API
    python upperband_analyze.py --csv "Upper Band.csv"  # CSV fallback
    python upperband_analyze.py --side both             # also lower-band
    python upperband_analyze.py --date 2026-05-06       # override as-of
    python upperband_analyze.py --top 50                # only top-N by Value
    python upperband_analyze.py --skip-charts           # features only
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths and Angel client import
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))             # .../upper_band
LEGACY_DIR = os.path.join(os.path.dirname(HERE), "legacy_scanner")
if LEGACY_DIR not in sys.path:
    sys.path.insert(0, LEGACY_DIR)

import angel_client as _ac                                    # noqa: E402
from angel_client import (                                    # noqa: E402
    angel_download,
    _load_scrip_master,
    _ensure_session,
)

CHARTS_ROOT = os.path.join(HERE, "charts")
ANALYSIS_ROOT = os.path.join(HERE, "analysis")
PATTERN_LOG = os.path.join(ANALYSIS_ROOT, "_pattern_log.md")


# ---------------------------------------------------------------------------
# Symbol resolution helpers
# ---------------------------------------------------------------------------
def resolve_symbol(sym: str) -> Optional[str]:
    """Return Angel ticker form (``SYMBOL.NS``) if resolvable, else None.

    Angel's _parse_ticker already falls back to bare-symbol lookup for
    series like SM/ST/BE/BZ/SZ, so passing ``<SYMBOL>.NS`` works for all
    the variants we see in the Upper-Band CSV.
    """
    sym = (sym or "").strip().upper()
    if not sym:
        return None
    if _ac._symbol_index is None:
        _load_scrip_master()
    idx = _ac._symbol_index
    if (("NSE", sym) in idx
            or ("NSE", sym + "-EQ") in idx):
        return sym + ".NS"
    # Try every series suffix we know exists in Upper-Band CSVs
    for suf in ("-BE", "-BZ", "-SM", "-ST", "-SZ", "-BL"):
        if ("NSE", sym + suf) in idx:
            return sym + ".NS"
    return None


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    dn = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1.0 / n, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(),
                    (h - pc).abs(),
                    (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def _consecutive_up_days(close: pd.Series, lookback: int = 10) -> int:
    """How many consecutive green closes immediately before today (inclusive)."""
    chg = close.diff().tail(lookback)
    n = 0
    for v in reversed(chg.tolist()):
        if v is None or np.isnan(v) or v <= 0:
            break
        n += 1
    return n


def _base_length(close: pd.Series, tol: float = 0.05) -> int:
    """Bars since last close > today_close * (1 + tol). Caps at 250."""
    if len(close) < 2:
        return 0
    today = close.iloc[-1]
    threshold = today * (1.0 + tol)
    prev = close.iloc[:-1]
    above = np.where(prev.values > threshold)[0]
    if len(above) == 0:
        return min(len(prev), 250)
    return int(len(prev) - 1 - above[-1])


def compute_features(df: pd.DataFrame, row: dict) -> dict:
    """Build a feature dict from 1-year OHLCV ``df`` and CSV ``row``."""
    f: dict = {"symbol": row["Symbol"],
               "series": row["Series"],
               "csv_pct_chg": row["%chng"],
               "csv_band_pct": row["Price Band %"],
               "csv_value_cr": row["Value (Rs Crores)"]}

    if df.empty or len(df) < 30:
        f["status"] = "insufficient_data"
        f["bars"] = len(df)
        return f

    df = df.copy()
    c = df["Close"]; h = df["High"]; l = df["Low"]; o = df["Open"]; v = df["Volume"]

    df["sma20"] = c.rolling(20).mean()
    df["sma50"] = c.rolling(50).mean()
    df["sma200"] = c.rolling(200).mean()
    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["ema21"] = c.ewm(span=21, adjust=False).mean()
    df["vol_avg20"] = v.rolling(20).mean()
    df["vol_avg50"] = v.rolling(50).mean()
    df["rsi14"] = _rsi(c, 14)
    df["atr14"] = _atr(df, 14)
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std

    last = df.iloc[-1]
    prev = df.iloc[-2]

    range_today = max(last.High - last.Low, 1e-9)
    body = abs(last.Close - last.Open)
    upper_wick = last.High - max(last.Close, last.Open)
    lower_wick = min(last.Close, last.Open) - last.Low

    f.update({
        "status":            "ok",
        "bars":              int(len(df)),
        "last_date":         df.index[-1].strftime("%Y-%m-%d"),
        "close":             round(float(last.Close), 4),
        "open":              round(float(last.Open), 4),
        "high":              round(float(last.High), 4),
        "low":               round(float(last.Low), 4),
        "volume":            int(last.Volume),
        "pct_chg_actual":    round(float((last.Close / prev.Close - 1) * 100), 2),
        "gap_pct":           round(float((last.Open / prev.Close - 1) * 100), 2),
        "body_pct_range":    round(float(body / range_today * 100), 1),
        "upper_wick_pct":    round(float(upper_wick / range_today * 100), 1),
        "lower_wick_pct":    round(float(lower_wick / range_today * 100), 1),
        "vol_x_avg20":       round(float(last.Volume / max(last.vol_avg20, 1)), 2),
        "vol_x_avg50":       round(float(last.Volume / max(last.vol_avg50, 1)), 2),
        "rsi14":             round(float(last.rsi14), 1),
        "atr14_pct":         round(float(last.atr14 / last.Close * 100), 2),
        "above_sma20":       bool(last.Close > last.sma20) if pd.notna(last.sma20) else None,
        "above_sma50":       bool(last.Close > last.sma50) if pd.notna(last.sma50) else None,
        "above_sma200":      bool(last.Close > last.sma200) if pd.notna(last.sma200) else None,
        "ema_stack_bull":    bool(last.ema9 > last.ema21 and last.Close > last.ema9),
        "bb_position":       (round(float((last.Close - last.bb_lower)
                                          / max(last.bb_upper - last.bb_lower, 1e-9)), 2)
                              if pd.notna(last.bb_upper) else None),
        "dist_52w_high_pct": round(float((last.Close / c.tail(252).max() - 1) * 100), 2),
        "dist_52w_low_pct":  round(float((last.Close / c.tail(252).min() - 1) * 100), 2),
        "ret_5d_pct":        round(float((last.Close / c.iloc[-6] - 1) * 100), 2)
                              if len(c) > 6 else None,
        "ret_20d_pct":       round(float((last.Close / c.iloc[-21] - 1) * 100), 2)
                              if len(c) > 21 else None,
        "ret_60d_pct":       round(float((last.Close / c.iloc[-61] - 1) * 100), 2)
                              if len(c) > 61 else None,
        "consec_up_days":    _consecutive_up_days(c, 15),
        "base_len_5pct":     _base_length(c, tol=0.05),
        "is_52w_high":       bool(last.Close >= c.tail(252).max() * 0.999),
        "vol_avg20_lakhs":   round(float(last.vol_avg20 / 1e5), 2),
    })

    # heuristic pattern tags (cheap, additive — refined as we collect data)
    tags = []
    if f["base_len_5pct"] >= 40 and f["vol_x_avg20"] >= 3:
        tags.append("base_breakout")
    if f["consec_up_days"] >= 4:
        tags.append("momentum_run")
    if f["gap_pct"] >= 4:
        tags.append("gap_up")
    if f["lower_wick_pct"] >= 50 and f["body_pct_range"] >= 25:
        tags.append("hammer_reversal")
    if f["body_pct_range"] >= 70 and f["upper_wick_pct"] <= 10:
        tags.append("marubozu_strong")
    if f["is_52w_high"]:
        tags.append("new_52w_high")
    if f["vol_x_avg20"] >= 5 and f["pct_chg_actual"] >= 5:
        tags.append("vol_thrust")
    if f["rsi14"] is not None and f["rsi14"] >= 70:
        tags.append("rsi_overbought_entry")
    if (f["dist_52w_high_pct"] is not None
            and -3 <= f["dist_52w_high_pct"] <= 0
            and f["base_len_5pct"] >= 20):
        tags.append("near_pivot_breakout")
    f["tags"] = ",".join(tags) if tags else "none"
    return f


# ---------------------------------------------------------------------------
# Charting
# ---------------------------------------------------------------------------
def render_chart(df: pd.DataFrame, symbol: str, out_path: str,
                 row: dict) -> bool:
    """Save 1y daily candle + volume chart with overlays."""
    if df.empty or len(df) < 30:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import mplfinance as mpf
    except Exception as e:
        print(f"   chart libs missing: {e}")
        return False

    pdf = df.copy()
    pdf["sma20"] = pdf["Close"].rolling(20).mean()
    pdf["sma50"] = pdf["Close"].rolling(50).mean()
    pdf["sma200"] = pdf["Close"].rolling(200).mean()
    hi52 = pdf["Close"].tail(252).max()
    lo52 = pdf["Close"].tail(252).min()

    addplots = [
        mpf.make_addplot(pdf["sma20"], width=0.8, color="#1f77b4"),
        mpf.make_addplot(pdf["sma50"], width=0.8, color="#ff7f0e"),
        mpf.make_addplot(pdf["sma200"], width=0.8, color="#7f7f7f"),
    ]
    title = (f"{symbol}  |  {row['Series']}  |  Band {row['Price Band %']}%  "
             f"|  +{row['%chng']:.2f}%  |  Val Rs {row['Value (Rs Crores)']:.2f} Cr")
    try:
        fig, _ = mpf.plot(
            pdf,
            type="candle",
            style="yahoo",
            volume=True,
            addplot=addplots,
            figsize=(14, 7),
            title=title,
            tight_layout=True,
            returnfig=True,
            hlines=dict(hlines=[hi52, lo52], colors=["g", "r"],
                        linestyle="--", linewidths=0.6),
            datetime_format="%b'%y",
            xrotation=0,
        )
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return True
    except Exception as e:
        print(f"   chart failed for {symbol}: {e}")
        return False


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------
def _to_num(s):
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        if c.startswith("Value"):
            rename[c] = "Value (Rs Crores)"
        elif c.startswith("Volume"):
            rename[c] = "Volume (Lakhs)"
    df = df.rename(columns=rename)
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df["Series"] = df["Series"].astype(str).str.strip().str.upper()
    for col in ("LTP", "%chng", "Price Band %",
                "Volume (Lakhs)", "Value (Rs Crores)"):
        if col in df.columns:
            df[col] = _to_num(df[col])
    return df


# ---------------------------------------------------------------------------
# NSE live fetch — https://www.nseindia.com/api/live-analysis-price-band-hitter
# ---------------------------------------------------------------------------
NSE_API_URL = "https://www.nseindia.com/api/live-analysis-price-band-hitter"
NSE_REFERER = "https://www.nseindia.com/market-data/price-band-hitter"
NSE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": NSE_REFERER,
    "X-Requested-With": "XMLHttpRequest",
    "Connection": "keep-alive",
}


def fetch_nse_price_band_hitter(timeout: int = 20) -> dict:
    """Fetch raw JSON from NSE's price-band-hitter endpoint.

    NSE blocks plain GETs — we must warm up cookies on the homepage first.
    """
    try:
        import requests
    except ImportError as e:
        raise RuntimeError(
            "requests not installed. Run: pip install requests") from e

    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    # Cookie warm-up
    s.get("https://www.nseindia.com", timeout=timeout)
    s.get(NSE_REFERER, timeout=timeout)
    r = s.get(NSE_API_URL, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _flatten_band_payload(payload: dict, side: str = "upper") -> pd.DataFrame:
    """Convert NSE JSON to the same DataFrame schema used by load_csv().

    The endpoint returns:
        { "upper": {"AllSec": {"data": [...]}, "5":{"data":[...]}, ...},
          "lower": {...}, "timestamp": ... }
    We use the ``AllSec`` bucket — it is the union across all band tiers.
    """
    block = (payload or {}).get(side) or {}
    rows = ((block.get("AllSec") or {}).get("data")) or []
    if not rows:
        # Fall back: union all tier buckets if AllSec is missing.
        rows = []
        for k, v in block.items():
            if k == "AllSec":
                continue
            for r in (v or {}).get("data", []) or []:
                rows.append(r)

    out = []
    for r in rows:
        try:
            out.append({
                "Symbol":             str(r.get("symbol", "")).strip().upper(),
                "Series":             str(r.get("series", "")).strip().upper(),
                "LTP":                pd.to_numeric(r.get("ltp"), errors="coerce"),
                "%chng":              pd.to_numeric(
                                          str(r.get("pChange", "")).strip(),
                                          errors="coerce"),
                "Price Band %":       pd.to_numeric(r.get("priceBand"),
                                                    errors="coerce"),
                "Volume (Lakhs)":     pd.to_numeric(r.get("totalTradedVol"),
                                                    errors="coerce"),
                "Value (Rs Crores)":  pd.to_numeric(r.get("turnover"),
                                                    errors="coerce"),
                "_side":              side,
                "_yearHigh":          pd.to_numeric(r.get("yearHigh"),
                                                    errors="coerce"),
                "_yearLow":           pd.to_numeric(r.get("yearLow"),
                                                    errors="coerce"),
            })
        except Exception:
            continue
    df = pd.DataFrame(out)
    if not df.empty:
        df = df[df["Symbol"] != ""].drop_duplicates("Symbol", keep="first")
    return df


def load_nse_live(side: str, raw_dump_path: Optional[str] = None) -> pd.DataFrame:
    """Pull live NSE price-band-hitter list. ``side`` in {upper, lower, both}."""
    payload = fetch_nse_price_band_hitter()
    if raw_dump_path:
        os.makedirs(os.path.dirname(raw_dump_path), exist_ok=True)
        with open(raw_dump_path, "w") as f:
            json.dump(payload, f)
    if side == "both":
        up = _flatten_band_payload(payload, "upper")
        dn = _flatten_band_payload(payload, "lower")
        return pd.concat([up, dn], ignore_index=True)
    return _flatten_band_payload(payload, side)


# ---------------------------------------------------------------------------
# Cohort summary
# ---------------------------------------------------------------------------
def write_summary(feats: pd.DataFrame, as_of: str, out_path: str) -> None:
    ok = feats[feats["status"] == "ok"].copy()
    n_total = len(feats); n_ok = len(ok)

    def pct(mask):
        return f"{int(mask.sum())}/{n_ok} ({mask.mean()*100:.0f}%)" if n_ok else "0/0"

    lines = [f"# Upper-Band cohort — {as_of}", ""]
    lines.append(f"- Total rows in CSV : **{n_total}**")
    lines.append(f"- With usable history: **{n_ok}**")
    if not n_ok:
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        return

    # Buckets by % change actually delivered today
    bins = [(20, "circuit_20"), (10, "circuit_10"),
            (5, "circuit_5"), (2, "circuit_2"), (-100, "down_or_flat")]
    lines.append("")
    lines.append("## Move-size distribution (actual today)")
    last = -100
    for thr, name in bins:
        mask = (ok["pct_chg_actual"] >= thr) & (ok["pct_chg_actual"] < last)
        if name == "circuit_20":
            mask = ok["pct_chg_actual"] >= 19.5
        last = thr
        lines.append(f"- **{name}** (>= {thr}% to <{last if last<100 else 999}%): "
                     f"{int(mask.sum())}")

    lines += ["", "## Pattern flag prevalence"]
    all_tags = (",".join(ok["tags"].fillna(""))).split(",")
    counts = pd.Series([t for t in all_tags if t and t != "none"]).value_counts()
    for t, n in counts.items():
        lines.append(f"- `{t}` : {n}/{n_ok} ({n/n_ok*100:.0f}%)")

    lines += ["", "## Pre-move structure (medians on usable rows)"]
    med = ok[[
        "vol_x_avg20", "vol_x_avg50", "rsi14", "atr14_pct",
        "dist_52w_high_pct", "dist_52w_low_pct",
        "ret_5d_pct", "ret_20d_pct", "ret_60d_pct",
        "consec_up_days", "base_len_5pct", "gap_pct",
        "body_pct_range", "upper_wick_pct", "lower_wick_pct",
    ]].median(numeric_only=True)
    for k, v in med.items():
        lines.append(f"- {k:22s} median = {v:.2f}")

    lines += ["", "## Trend posture"]
    lines.append(f"- above SMA20  : {pct(ok['above_sma20'].fillna(False))}")
    lines.append(f"- above SMA50  : {pct(ok['above_sma50'].fillna(False))}")
    lines.append(f"- above SMA200 : {pct(ok['above_sma200'].fillna(False))}")
    lines.append(f"- bullish EMA stack (close>EMA9>EMA21): {pct(ok['ema_stack_bull'].fillna(False))}")
    lines.append(f"- new 52w high : {pct(ok['is_52w_high'].fillna(False))}")

    # By circuit band
    lines += ["", "## By price-band tier (count, median vol_x_avg20, median RSI, % at 52w high)"]
    for band, sub in ok.groupby("csv_band_pct"):
        if pd.isna(band):
            continue
        lines.append(
            f"- band {band:.0f}% : n={len(sub)}, "
            f"vol_x={sub['vol_x_avg20'].median():.2f}, "
            f"rsi={sub['rsi14'].median():.1f}, "
            f"at52wH={int(sub['is_52w_high'].fillna(False).sum())}"
        )

    # Top "interesting" rows for manual review
    lines += ["", "## Top 15 by liquidity (Value Cr)"]
    top = ok.sort_values("csv_value_cr", ascending=False).head(15)[
        ["symbol", "series", "csv_band_pct", "csv_pct_chg",
         "vol_x_avg20", "rsi14", "consec_up_days",
         "base_len_5pct", "dist_52w_high_pct", "tags"]]
    lines.append(top.to_markdown(index=False))

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def append_pattern_log(as_of: str, feats: pd.DataFrame, summary_path: str) -> None:
    os.makedirs(os.path.dirname(PATTERN_LOG), exist_ok=True)
    ok = feats[feats["status"] == "ok"]
    digest = (f"- **{as_of}**: n_total={len(feats)}, n_ok={len(ok)}, "
              f"median vol_x_avg20={ok['vol_x_avg20'].median():.2f}, "
              f"median RSI={ok['rsi14'].median():.1f}, "
              f"at52wH={int(ok['is_52w_high'].fillna(False).sum())}, "
              f"momentum_run={int(ok['tags'].str.contains('momentum_run').sum())}, "
              f"vol_thrust={int(ok['tags'].str.contains('vol_thrust').sum())}, "
              f"base_breakout={int(ok['tags'].str.contains('base_breakout').sum())}, "
              f"see [{os.path.basename(os.path.dirname(summary_path))}]"
              f"({os.path.relpath(summary_path, os.path.dirname(PATTERN_LOG))})")
    header_needed = not os.path.exists(PATTERN_LOG)
    with open(PATTERN_LOG, "a") as f:
        if header_needed:
            f.write("# Upper-Band pattern log (auto-appended)\n\n"
                    "Compact daily digest of cohort stats. Use this to spot\n"
                    "long-horizon shifts in pre-move structure.\n\n")
        f.write(digest + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="Optional CSV fallback. If omitted, fetch live from "
                         "NSE price-band-hitter API.")
    ap.add_argument("--side", choices=["upper", "lower", "both"],
                    default="upper",
                    help="Which side of the band to analyse (live API only).")
    ap.add_argument("--date", default=None,
                    help="As-of trading date, YYYY-MM-DD. Default: today.")
    ap.add_argument("--lookback-days", type=int, default=400)
    ap.add_argument("--top", type=int, default=0,
                    help="Process only top-N rows by Value (Rs Crores). 0=all.")
    ap.add_argument("--skip-charts", action="store_true")
    args = ap.parse_args()

    as_of = args.date or dt.date.today().strftime("%Y-%m-%d")
    analysis_dir = os.path.join(ANALYSIS_ROOT, as_of)
    chart_dir = os.path.join(CHARTS_ROOT, as_of)
    os.makedirs(chart_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    if args.csv:
        if not os.path.exists(args.csv):
            print(f"CSV not found: {args.csv}")
            return 2
        print(f"== Upper-Band deep-dive | as_of={as_of} | source=CSV ==")
        df_csv = load_csv(args.csv)
    else:
        print(f"== Upper-Band deep-dive | as_of={as_of} | "
              f"source=NSE API (side={args.side}) ==")
        try:
            df_csv = load_nse_live(
                args.side,
                raw_dump_path=os.path.join(analysis_dir, "nse_raw.json"),
            )
        except Exception as e:
            print(f"NSE live fetch failed: {e}")
            return 3
        if df_csv.empty:
            print("NSE returned 0 rows. Market may be closed or "
                  "endpoint structure changed.")
            return 4

    if args.top > 0:
        df_csv = df_csv.sort_values("Value (Rs Crores)",
                                    ascending=False).head(args.top)
    print(f"  rows in cohort: {len(df_csv)}")

    _load_scrip_master()
    _ensure_session()

    end_date = dt.datetime.strptime(as_of, "%Y-%m-%d").date()
    start_date = end_date - dt.timedelta(days=args.lookback_days)

    feats: list = []
    unresolved: list = []
    t0 = time.time()
    for i, r in enumerate(df_csv.to_dict("records"), 1):
        sym = r["Symbol"]
        ticker = resolve_symbol(sym)
        if not ticker:
            unresolved.append(sym)
            feats.append({"symbol": sym, "series": r["Series"],
                          "status": "unresolved", "tags": "none"})
            continue

        ohlcv = angel_download(ticker, start_date, end_date, interval="1d")
        feat = compute_features(ohlcv, r)
        feats.append(feat)

        if not args.skip_charts:
            out_png = os.path.join(chart_dir, f"{sym}.png")
            render_chart(ohlcv, sym, out_png, r)

        if i % 10 == 0 or i == len(df_csv):
            print(f"  {i}/{len(df_csv)} done ({time.time()-t0:.0f}s, "
                  f"unresolved={len(unresolved)})")

    feats_df = pd.DataFrame(feats)
    csv_out = os.path.join(analysis_dir, "per_symbol.csv")
    feats_df.to_csv(csv_out, index=False)
    print(f"\n  per-symbol features -> {csv_out}")

    summary_path = os.path.join(analysis_dir, "summary.md")
    write_summary(feats_df, as_of, summary_path)
    print(f"  summary             -> {summary_path}")

    append_pattern_log(as_of, feats_df, summary_path)
    print(f"  pattern log         -> {PATTERN_LOG}")

    if unresolved:
        with open(os.path.join(analysis_dir, "unresolved.txt"), "w") as f:
            f.write("\n".join(unresolved))
        print(f"  unresolved symbols ({len(unresolved)}): "
              f"{', '.join(unresolved[:10])}"
              f"{' ...' if len(unresolved) > 10 else ''}")

    print(f"\nDONE in {time.time()-t0:.0f}s. Charts: {chart_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
