"""
Breakout Scanner v1 (Angel One edition) — Pre-Breakout Setup Detector
=====================================================================
Identical to breakout_scanner.py, but historical OHLCV is sourced from
the free Angel One SmartAPI (via angel_client.py) instead of yfinance.
This gives full coverage of NSE main + NSE Emerge (SME) + BSE main + BSE SME.

Prereq:
  - Free Angel One demat account (angelone.in)
  - Free SmartAPI app (smartapi.angelbroking.com) -> ANGEL_API_KEY
  - TOTP enabled in Angel One profile -> ANGEL_TOTP_SECRET (base32)
  - .env file with ANGEL_API_KEY, ANGEL_CLIENT_CODE, ANGEL_PIN,
    ANGEL_TOTP_SECRET
  - python3 -m pip install pyotp python-dotenv
  - First run downloads ~25 MB Angel scrip master (cached weekly)

Usage:
  python breakout_scanner_angel.py                   # full scan
  python breakout_scanner_angel.py --max 30          # quick test
  python breakout_scanner_angel.py --min-score 70    # change cut-off
  python breakout_scanner_angel.py --charts 15       # # of top charts
"""

import os
import sys
import re
import math
import argparse
import datetime
import warnings
import urllib.request
import urllib.parse
import http.cookiejar
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, os.pardir, "Output")
TODAY = datetime.date.today()
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class _Tee:
    """Duplicate writes to both a file and the original stream."""
    def __init__(self, stream, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._file = open(filepath, "w")
        self._stream = stream

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()

# Universe source — multi_pct_down_report.xlsx (all sheets)
PCT_DOWN_REPORT = os.path.join(SCRIPT_DIR, "multi_pct_down_report.xlsx")

NIFTY50_BENCH = "^NSEI"  # Nifty 50 index (handled via Angel INDEX_OVERRIDES)

# ─── Defaults / thresholds (v4.1) ───────────────────────────────────────────
LOOKBACK_DAYS = 252        # v4.1: only consider last ~1 year of daily history
RES_LOOKBACK_DAYS = 252    # v4.1: pivot/resistance search restricted to 1y
MIN_HISTORY_DAYS = 45      # v4.3: min trading days required to evaluate a name
BASE_MIN_DAYS = 20         # pattern matters more than duration
BASE_MAX_DAYS = 180        # v4.1: cap base length at ~180 calendar days
RES_BAND_PCT = 0.050       # touches counted within +/- 5% of resistance
PROXIMITY_MAX_PCT = 0.04   # distance to resistance upper bound = +4%
PROXIMITY_MIN_PCT = -0.05  # distance to resistance lower bound = -5%
MIN_TOUCHES = 2            # detect_resistance qualification floor
HC_MULTITOUCH_MIN = 2      # v4.1: pattern A requires >= 2 touches
MAX_BASE_RANGE_PCT = 0.40  # reject any base wider than 40%
RECENT_R_TEST_LOOKBACK = 50  # 50 sessions for recent resistance test
RS_RISING_LOOKBACK = 50    # v4.4: rising RS-line over last 50 sessions
MIN_AVG_VOL = 0            # v4.3: liquidity filter disabled
WATCHLIST_MIN_SCORE = 50
TRIGGER_MIN_SCORE = 65


# ─── Universe ────────────────────────────────────────────────────────────────

def fetch_universe() -> list:
    """Build ticker universe from multi_pct_down_report.xlsx.

    Reads the 'Yahoo' column from every sheet, deduplicates, and returns
    a sorted list of yfinance-style tickers (e.g. 'RELIANCE.NS', '543745.BO').
    """
    if not os.path.exists(PCT_DOWN_REPORT):
        raise FileNotFoundError(f"Universe file not found: {PCT_DOWN_REPORT}")

    xls = pd.ExcelFile(PCT_DOWN_REPORT)
    universe = set()
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        if "Yahoo" not in df.columns:
            print(f"  WARNING: sheet '{sheet}' has no 'Yahoo' column — skipped")
            continue
        tickers = (
            df["Yahoo"]
            .dropna()
            .astype(str)
            .str.strip()
        )
        tickers = set(tickers[tickers != ""])
        new = tickers - universe
        universe |= tickers
        print(f"  {sheet:<28}: {len(tickers):>5} symbols  (+{len(new)} new)")
    xls.close()

    universe = sorted(universe)
    print(f"  Total universe: {len(universe)} unique tickers\n")
    return universe


# ─── Screener.in universe fetch ──────────────────────────────────────────────

def _screener_login():
    """Log in to screener.in, return authenticated opener or None."""
    load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

    email = os.environ.get("SCREENER_USER", "")
    password = os.environ.get("SCREENER_PASS", "")
    if not email or not password:
        print("  ERROR: SCREENER_USER / SCREENER_PASS not set in .env")
        return None

    ua = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
    login_url = "https://www.screener.in/login/"
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    # GET login page for CSRF
    req = urllib.request.Request(login_url, headers={"User-Agent": ua, "Accept": "text/html"})
    html = opener.open(req, timeout=30).read().decode("utf-8", errors="ignore")
    m = re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"', html)
    if not m:
        print("  ERROR: Could not find CSRF token on screener.in login page")
        return None
    csrf = m.group(1)

    # POST login
    data = urllib.parse.urlencode({
        "csrfmiddlewaretoken": csrf,
        "username": email,
        "password": password,
    }).encode("utf-8")
    req2 = urllib.request.Request(login_url, data=data, headers={
        "User-Agent": ua, "Referer": login_url,
        "Content-Type": "application/x-www-form-urlencoded"})
    resp = opener.open(req2, timeout=30)
    body = resp.read().decode("utf-8", errors="ignore")
    if "Please enter a correct" in body or "Invalid username" in body:
        print("  ERROR: screener.in login failed — invalid credentials")
        return None
    print("  screener.in login OK")
    return opener


def _screener_fetch_names(opener, url: str) -> list:
    """Fetch all pages of a screener.in screen, return list of stock names."""
    ua = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
    names = []
    page = 1
    while True:
        page_url = f"{url.rstrip('/')}/?page={page}" if page > 1 else url
        req = urllib.request.Request(page_url, headers={
            "User-Agent": ua, "Accept": "text/html",
            "Referer": "https://www.screener.in/"})
        html = opener.open(req, timeout=30).read().decode("utf-8", errors="ignore")
        pattern = r'href="/company/([^/]+)/[^"]*"[^>]*>\s*([^<]+?)\s*</a>'
        found = re.findall(pattern, html)
        if not found:
            break
        for sym_slug, name in found:
            names.append((sym_slug.strip().upper(), name.strip()))
        page += 1
        if f"page={page}" not in html and "Next" not in html:
            break
    return names


def fetch_screener_universe(url: str) -> list:
    """Fetch a screener.in screen and resolve names to Angel-compatible tickers.

    Returns a list of yfinance-style tickers (e.g. 'RELIANCE.NS', '543745.BO').
    Resolution strategy:
      1. screener.in URL slugs are usually NSE symbols → try SYM.NS directly
      2. Fall back to Angel scrip master name-match for any unresolved
    """
    opener = _screener_login()
    if opener is None:
        raise SystemExit("Cannot proceed without screener.in login")

    print(f"  Fetching screen: {url}")
    raw = _screener_fetch_names(opener, url)
    if not raw:
        raise SystemExit("No stocks found on screener.in (check URL or visibility)")
    print(f"  Found {len(raw)} stocks on screener.in")

    # screener.in slugs are typically NSE symbols (e.g. RELIANCE, HDFCBANK)
    # Try .NS first; for numeric slugs (BSE scrip codes) try .BO
    tickers = []
    for slug, name in raw:
        if slug.isdigit():
            tickers.append(f"{slug}.BO")
        else:
            tickers.append(f"{slug}.NS")

    tickers = sorted(set(tickers))
    print(f"  Resolved to {len(tickers)} unique tickers")

    # Save to Output/screener_data.xlsx for reference
    out_dir = os.path.join(SCRIPT_DIR, os.pardir, "Output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "screener_data.xlsx")
    df = pd.DataFrame({"Name": [n for _, n in raw], "Ticker": [
        f"{s}.BO" if s.isdigit() else f"{s}.NS" for s, _ in raw]})
    df.to_excel(out_path, index=False, engine="openpyxl")
    print(f"  Reference saved: {out_path}")

    return tickers


# ─── Data ingestion (Angel One SmartAPI) ───────────────────────────────────

def fetch_ohlcv(tickers: list, lookback_days: int = LOOKBACK_DAYS,
                batch_size: int = 100) -> dict:
    """Bulk-download daily OHLCV via Angel One SmartAPI.

    `tickers` are yfinance-style symbols (e.g. 'RELIANCE.NS', '534109.BO').
    The angel_client adapter resolves them to Angel symboltokens and
    returns DataFrames with the same Open/High/Low/Close/Volume columns,
    so downstream scoring code is unchanged.

    `batch_size` is kept for API parity but not used — SmartAPI is
    single-symbol per call, internally rate-limited by angel_client.
    Returns {ticker: DataFrame}.
    """
    from angel_client import angel_download_many
    end = TODAY + datetime.timedelta(days=1)
    start = TODAY - datetime.timedelta(days=int(lookback_days * 1.5))
    print(f"  Downloading OHLCV for {len(tickers)} tickers via Angel One ...")
    raw = angel_download_many(tickers, start, end)
    out = {}
    for tk, df in raw.items():
        if df is None or df.empty or len(df) < BASE_MIN_DAYS + 30:
            continue
        out[tk] = df
    print(f"  Got usable history for {len(out)} tickers "
          f"(of {len(tickers)} requested)")
    return out


def fetch_benchmark(lookback_days: int = LOOKBACK_DAYS) -> pd.Series:
    """Fetch Nifty 50 index close history from Angel One."""
    from angel_client import angel_download
    end = TODAY + datetime.timedelta(days=1)
    start = TODAY - datetime.timedelta(days=int(lookback_days * 1.5))
    df = angel_download("^NSEI", start, end)
    if df.empty:
        return pd.Series(dtype=float)
    return df["Close"].rename("Bench")


# ─── Part 2: Resistance detection ───────────────────────────────────────────

def fractal_pivots(highs: pd.Series, k: int = 3) -> pd.Series:
    """Boolean series: True where high is local max over [-k, +k] window."""
    h = highs.values
    n = len(h)
    out = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        if h[i] == h[i - k:i + k + 1].max() and h[i] >= h[i - 1] and h[i] >= h[i + 1]:
            out[i] = True
    return pd.Series(out, index=highs.index)


def detect_resistance(df: pd.DataFrame) -> Optional[dict]:
    """Find best horizontal resistance the stock is currently approaching.

    Uses long history (RES_LOOKBACK_DAYS) so the level is stable across days.
    Returns dict {R, base_start, touches, distance_pct, base_len_days}.
    """
    if len(df) < BASE_MIN_DAYS + 20:
        return None

    close = df["Close"]
    last_close = float(close.iloc[-1])

    # Use a long window so R doesn't drift day-to-day
    window = df.tail(RES_LOOKBACK_DAYS)
    # Two pivot scales: tight (k=3) and broad (k=8) -- broad gives stable
    # multi-month swing highs the eye picks out.
    piv_mask_tight = fractal_pivots(window["High"], k=3)
    piv_mask_broad = fractal_pivots(window["High"], k=8)
    pivots_tight = window["High"][piv_mask_tight]
    pivots_broad = window["High"][piv_mask_broad]
    pivots = pd.concat([pivots_tight, pivots_broad]).groupby(level=0).max()
    if len(pivots) < MIN_TOUCHES:
        return None

    # Cluster pivots into bands of width = RES_BAND_PCT * level (greedy)
    levels = sorted(pivots.tolist(), reverse=True)
    clusters = []
    for lvl in levels:
        placed = False
        for c in clusters:
            if abs(lvl - c["level"]) / c["level"] <= RES_BAND_PCT:
                c["sum"] += lvl
                c["count"] += 1
                c["level"] = c["sum"] / c["count"]
                placed = True
                break
        if not placed:
            clusters.append({"level": lvl, "sum": lvl, "count": 1})

    # Allow a wider distance window: from 3% above (just broken) to 8% below.
    candidates = []
    for c in clusters:
        R = c["level"]
        dist = (R - last_close) / last_close
        if c["count"] < MIN_TOUCHES:
            continue
        if dist < PROXIMITY_MIN_PCT or dist > PROXIMITY_MAX_PCT:
            continue
        cluster_pivots_idx = [
            ts for ts in pivots.index
            if abs(pivots.loc[ts] - R) / R <= RES_BAND_PCT
        ]
        if len(cluster_pivots_idx) < MIN_TOUCHES:
            continue
        base_start = min(cluster_pivots_idx)
        base_len = (df.index[-1] - base_start).days
        if base_len < BASE_MIN_DAYS:
            continue
        # v4.1: cap base length at BASE_MAX_DAYS (180 calendar days). Older
        # pivots are treated as historical context, not the active base.
        if base_len > BASE_MAX_DAYS:
            recent_pivots = [ts for ts in cluster_pivots_idx
                             if (df.index[-1] - ts).days <= BASE_MAX_DAYS]
            if len(recent_pivots) < MIN_TOUCHES:
                continue
            base_start = min(recent_pivots)
            base_len = (df.index[-1] - base_start).days
        # Score: more touches better, longer base better, closer to 52w high better
        is_52w_high = R >= float(window["High"].max()) * 0.98
        candidates.append({
            "R": R,
            "touches": len(cluster_pivots_idx),
            "base_start": base_start,
            "base_len_days": base_len,
            "distance_pct": dist,
            "touch_dates": cluster_pivots_idx,
            "is_52w_high": is_52w_high,
        })

    if not candidates:
        return None
    # Best = most-tested + longest base first; break ties by proximity.
    # (Dropped is_52w_high primary key — it forced selection of fresh swing
    # highs over the structurally significant horizontal level.)
    candidates.sort(key=lambda c: (
        -c["touches"], -c["base_len_days"], abs(c["distance_pct"])
    ))
    best = candidates[0]
    # Fix B: if there exists ANOTHER candidate ABOVE the chosen one with
    # similar or stronger structural strength (>=80% of best's touches AND
    # base length), prefer the higher one — that is the real ceiling, not a
    # mid-range support. Stops the scanner picking a 55 line when the real
    # box top is 60 with 12+ touches.
    higher = [c for c in candidates
              if c["R"] > best["R"] * 1.02
              and c["touches"] >= max(2, int(best["touches"] * 0.8))
              and c["base_len_days"] >= int(best["base_len_days"] * 0.8)]
    if higher:
        # Pick the highest such ceiling
        higher.sort(key=lambda c: -c["R"])
        return higher[0]
    return best


# ─── Indicators ─────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["Close"].diff().fillna(0))
    return (sign * df["Volume"]).cumsum()


def linreg_slope(y: pd.Series) -> float:
    if y.dropna().size < 5:
        return 0.0
    yy = y.dropna().values
    xx = np.arange(len(yy))
    return float(np.polyfit(xx, yy, 1)[0])


def rs_rising(df: pd.DataFrame, bench: pd.Series,
              lookback: int = RS_RISING_LOOKBACK) -> dict:
    """v4.0: True if the relative-strength line (stock_close / benchmark_close)
    has a positive slope over the last `lookback` sessions. This is the
    classic Mansfield/IBD RS line — independent of absolute RS magnitude;
    it captures whether the stock is OUT-PERFORMING the index right now,
    regardless of the longer-term gap. Returns dict {pass, slope, lookback}.
    """
    if bench is None or len(bench) < lookback or len(df) < lookback:
        return {"pass": False, "slope": 0.0, "lookback": lookback}
    common = df.index.intersection(bench.index)
    if len(common) < lookback:
        return {"pass": False, "slope": 0.0, "lookback": lookback}
    s = df["Close"].reindex(common).tail(lookback).astype(float)
    b = bench.reindex(common).tail(lookback).astype(float)
    if (b <= 0).any() or s.isna().any() or b.isna().any():
        return {"pass": False, "slope": 0.0, "lookback": lookback}
    rs_line = (s / b).values
    # Normalise so slope magnitude is comparable across stocks
    rs_norm = rs_line / rs_line[0] if rs_line[0] > 0 else rs_line
    x = np.arange(len(rs_norm), dtype=float)
    slope = float(np.polyfit(x, rs_norm, 1)[0])
    return {"pass": bool(slope > 0), "slope": slope, "lookback": lookback}




# ─── Part 3: Composite "Coiled Spring" Score ────────────────────────────────

def compute_score(df: pd.DataFrame, res: dict, bench: pd.Series) -> dict:
    """Compute 0-100 composite score. Returns dict of components + total.

    Re-weighted (v2 calibration) after 10-ticker screenshot audit:
      A Base quality      25
      B Volatility contr  10
      C Volume dry-up      5
      D Proximity to R    20
      E Trend             15
      F Relative strength 10
      G 52w high          15
      Total              100
    """
    base_start = res["base_start"]
    base = df.loc[base_start:]
    if len(base) < 20:
        return {"score": 0.0}

    R = res["R"]
    last = df.iloc[-1]
    last_close = float(last["Close"])

    # ── A: Base quality (25) ──
    T = res["base_len_days"]
    Tmax = 120
    base_score = 25.0 * min(T / Tmax, 1.0)
    # touches multiplier: 2=0.75, 3=0.9, 4=1.0, 5+=1.0 (no penalty above 4)
    touches_mult = min(0.75 + 0.075 * (res["touches"] - 2), 1.0)
    if res["touches"] >= 5:
        touches_mult = 1.0
    base_score *= touches_mult
    lows_idx = base["Low"].rolling(11, center=True).min() == base["Low"]
    swing_lows = base["Low"][lows_idx].dropna()
    higher_lows = (linreg_slope(swing_lows) > 0) if len(swing_lows) >= 3 else False
    if higher_lows:
        base_score = min(base_score * 1.15, 25.0)
    # Young-leader / trend-continuation boost: post-IPO or short-base names
    # in a strong uptrend (price > 50dma > 200dma, both rising) get a floor
    # so that a 60-day VCP against a 200dma slope doesn't get penalised
    # purely on length. Lifts base_score to at least 16/25.
    ma50_now = df["Close"].rolling(50).mean().iloc[-1]
    ma200_series = df["Close"].rolling(200).mean()
    ma200_now = ma200_series.iloc[-1] if not pd.isna(ma200_series.iloc[-1]) else 0
    in_uptrend = (last_close > ma50_now > ma200_now > 0
                  and linreg_slope(df["Close"].rolling(50).mean().tail(20)) > 0)
    young_leader = bool(in_uptrend and 30 <= T < 100)
    if young_leader:
        base_score = max(base_score, 16.0)

    # ── B: Volatility Contraction (10) ──
    a_series = atr(df, 14)
    atr_now = float(a_series.iloc[-10:].mean())
    atr_then = float(a_series.loc[base_start:].iloc[:20].mean()) if len(base) >= 20 else atr_now
    vcr = 1.0 - (atr_now / atr_then) if atr_then > 0 else 0.0
    vcr_score = 10.0 * max(min(vcr / 0.30, 1.0), 0.0)

    # ── C: Volume Dry-Up (5) ──
    v50 = float(df["Volume"].rolling(50).mean().iloc[-1])
    v10 = float(df["Volume"].iloc[-10:].mean())
    vdu = 1.0 - (v10 / v50) if v50 > 0 else 0.0
    vdu_score = 5.0 * max(min(vdu / 0.20, 1.0), 0.0)

    # ── D: Proximity (20) — reward being close to or just above R ──
    dist = (R - last_close) / last_close
    if PROXIMITY_MIN_PCT <= dist <= PROXIMITY_MAX_PCT:
        prox_score = 20.0 * max(0.0, 1.0 - abs(dist) / PROXIMITY_MAX_PCT)
    else:
        prox_score = 0.0

    # ── E: Trend (15) ──
    ma50 = df["Close"].rolling(50).mean()
    ma200 = df["Close"].rolling(200).mean()
    trend_score = 0.0
    if last_close > ma50.iloc[-1]:
        trend_score += 5
    if last_close > ma200.iloc[-1]:
        trend_score += 5
    if (linreg_slope(ma50.tail(20)) > 0
            and linreg_slope(ma200.tail(20)) > 0):
        trend_score += 5

    # ── F: Mansfield RS (10) ──
    rs_score = 0.0
    rs_value = 0.0
    if not bench.empty:
        b = bench.reindex(df.index).ffill()
        ratio = (df["Close"] / b).dropna()
        if len(ratio) >= 60:
            sma52w = ratio.rolling(min(252, len(ratio))).mean()
            mans = (ratio / sma52w - 1.0) * 100.0
            rs_value = float(mans.iloc[-1]) if not pd.isna(mans.iloc[-1]) else 0.0
            if rs_value > 0:
                rs_score = 5.0
            if linreg_slope(mans.tail(20)) > 0:
                rs_score += 5.0

    # ── G: 52-week high proximity (15) ──
    hi_52w = float(df["High"].tail(252).max())
    pct_off_high = (hi_52w - last_close) / hi_52w
    if pct_off_high <= 0.15:
        hi_score = 15.0 * (1.0 - pct_off_high / 0.15)
    else:
        hi_score = 0.0

    total = (base_score + vcr_score + vdu_score
             + prox_score + trend_score + rs_score + hi_score)

    return {
        "score": round(total, 2),
        "base_quality": round(base_score, 2),
        "vcr": round(vcr_score, 2),
        "vdu": round(vdu_score, 2),
        "proximity": round(prox_score, 2),
        "trend": round(trend_score, 2),
        "rs": round(rs_score, 2),
        "hi_52w": round(hi_score, 2),
        "vcr_raw": round(vcr, 3),
        "vdu_raw": round(vdu, 3),
        "atr_now": round(atr_now, 3),
        "atr_then": round(atr_then, 3),
        "higher_lows": higher_lows,
        "rs_value": round(rs_value, 3),
        "pct_off_52w_high": round(pct_off_high * 100, 2),
    }


# ─── Part 4: Pocket Pivot ───────────────────────────────────────────────────


# ─── Part 6: Risk architecture ──────────────────────────────────────────────

def risk_plan(df: pd.DataFrame, res: dict) -> dict:
    R = res["R"]
    base = df.loc[res["base_start"]:]
    base_low = float(base["Low"].min())
    last_close = float(df["Close"].iloc[-1])
    swing_lows_recent = base["Low"].tail(20).min()
    stop = float(swing_lows_recent) * 0.99
    height = R - base_low
    target = R + height  # measured move
    risk = last_close - stop
    reward = target - last_close
    rr = round(reward / risk, 2) if risk > 0 else None
    return {
        "entry": round(last_close, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "risk_pct": round(risk / last_close * 100, 2) if last_close else None,
        "reward_pct": round(reward / last_close * 100, 2) if last_close else None,
        "rr": rr,
        "base_low": round(base_low, 2),
        "base_height": round(height, 2),
    }


# ─── Part 7: Output ─────────────────────────────────────────────────────────

def render_chart(symbol: str, df: pd.DataFrame, res: dict, score: dict,
                 risk: dict, flags: dict, out_path: str):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Resistance line
    fig.add_hline(y=res["R"], line_color="#1e88e5", line_width=2,
                  annotation_text=f"R = {res['R']:.2f}", row=1, col=1)
    # Stop / target
    fig.add_hline(y=risk["stop"], line_color="#ef5350", line_dash="dash",
                  annotation_text=f"Stop {risk['stop']:.2f}", row=1, col=1)
    fig.add_hline(y=risk["target"], line_color="#26a69a", line_dash="dash",
                  annotation_text=f"Tgt {risk['target']:.2f}", row=1, col=1)

    # Mark base region
    fig.add_vrect(x0=res["base_start"], x1=df.index[-1],
                  fillcolor="#1e88e5", opacity=0.05, line_width=0, row=1, col=1)

    # Volume + 50d MA
    colors = np.where(df["Close"] >= df["Open"], "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors,
                         name="Volume", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index,
                             y=df["Volume"].rolling(50).mean(),
                             line=dict(color="white", width=1.2),
                             name="Vol 50DMA"), row=2, col=1)

    flags_str = ", ".join([k for k, v in flags.items() if v]) or "—"
    title = (
        f"{symbol} — Score {score['score']:.1f}/100 | "
        f"R={res['R']:.2f} ({res['distance_pct']*100:+.2f}%) | "
        f"Touches={res['touches']} | RR={risk['rr']} | "
        f"Flags: {flags_str}"
    )
    fig.update_layout(
        title=title, template="plotly_dark", height=720,
        xaxis_rangeslider_visible=False, showlegend=False,
    )
    fig.write_html(out_path, include_plotlyjs="cdn")


def _build_summary_sheet():
    """Static reference content written as the first sheet of the Excel.

    Includes:
      1. Run summary placeholder (filled by write_excel from `rows`)
      2. Final pre-breakout audit table (from the 25-ticker calibration set)
      3. Metric legend explaining what each row means
    """
    # 25-ticker calibration audit (pre-breakout, T-1 evaluation)
    audit_rows = [
        ("KRN.NS",         89.80, "YES", "YES",   990.0,  999.54,   0.96, "OK",   3.11,  7,  428, "od",   0.96),
        ("SYRMA.NS",       84.89, "YES", "YES",   915.0,  871.67,  -4.73, "OK",  -2.61, 10,  219, "pp",   0.82),
        ("WELCORP.NS",     75.60, "YES", "YES",   990.0,  950.82,  -3.96, "OK",  -1.14, 10,  304, "-",    0.88),
        ("SAAKSHI.NS",     75.24, "YES", "YES",   196.0,  188.67,  -3.74, "OK",  -3.25, 16,  504, "-",    0.38),
        ("MCX.NS",         74.28, "YES", "YES",  2780.0, 2647.15,  -4.78, "OK",  -4.28,  4,   74, "-",    0.91),
        ("SKYGOLD.NS",     72.25, "YES", "YES",   372.0,  353.10,  -5.08, "near", -4.39, 12,  529, "ppsqod", 0.54),
        ("PRUDENT.NS",     72.09, "YES", "YES",  2760.0, 2752.73,  -0.26, "OK",   0.07, 21,  616, "-",    0.62),
        ("ROLEXRINGS.NS",  70.85, "YES", "YES",   145.0,  138.74,  -4.32, "OK",  -0.53, 12,  392, "ppod", 0.74),
        ("SCI.NS",         70.82, "YES", "YES",   272.0,  280.89,   3.27, "OK",  10.87, 14,  800, "sqod", 0.80),
        ("QPOWER.NS",      67.67, "YES", "YES",  1080.0, 1054.72,  -2.34, "OK",   6.18,  4,  201, "-",    0.95),
        ("AZAD.NS",        66.24, "YES", "YES",  1780.0, 1670.00,  -6.18, "near",-4.09, 27,  623, "pp",   0.52),
        ("PARAS.NS",       65.70, "YES", "YES",   760.0,  724.64,  -4.65, "OK",  -4.00, 11,  631, "pp",   0.33),
        ("ASTRAMICRO.NS",  64.74, "YES", "no",   1050.0, 1045.11,  -0.47, "OK",   1.22, 12,  666, "-",    0.35),
        ("SUDEEPPHRM.NS",  64.48, "YES", "no",    690.0,  683.57,  -0.93, "OK",  -1.33,  8,  138, "-",    0.92),
        ("JAYNECOIND.NS",  64.26, "YES", "no",     83.0,   82.35,  -0.78, "OK",   0.81,  3,  163, "-",    0.67),
        ("ADVAIT.BO",      62.57, "YES", "no",   1880.0, 1941.27,   3.26, "OK",   3.96, 12,  666, "-",    0.46),
        ("RKFORGE.NS",     59.31, "YES", "no",    580.0,  587.24,   1.25, "OK",   4.27, 12,  257, "ppod", 0.96),
        ("KECL.NS",        58.55, "YES", "no",    108.0,  108.97,   0.90, "OK",   4.56,  5,  753, "-",   -0.49),
        ("WEBELSOLAR.NS",  55.58, "YES", "no",     98.5,   98.64,   0.14, "OK",   1.60,  4,  608, "pp",   0.29),
        ("TRITURBINE.NS",  54.10, "YES", "no",    550.0,  545.96,  -0.74, "OK",   5.92, 17,  359, "od",   0.25),
        ("EMMVEE.NS",      44.86, "no",  "no",    237.0,  227.62,  -3.96, "OK",   4.95,  3,   68, "pp",   0.99),
        ("FINBUD.NS",      41.07, "no",  "no",    113.0,  121.23,   7.29, "near",-3.55,  3,   92, "-",    0.27),
        ("PATILAUTOM.NS",  26.08, "no",  "no",    157.0,  159.70,   1.72, "OK",   7.91,  4,   62, "-",    0.97),
        ("SYSTEMATIC.BO",  13.42, "no",  "no",    168.0,  169.67,   0.99, "OK",  13.26,  3,   58, "-",    0.97),
    ]
    audit_df = pd.DataFrame(audit_rows, columns=[
        "symbol", "score", "WL>=50", "TR>=65",
        "expR", "scnR", "R_err%", "R_acc",
        "dist%", "touches", "base_d", "flags", "lvs",
    ])

    legend_rows = [
        ("R found",                "Pivot detector saw a level",         "Sanity check; if low, the geometry engine is broken"),
        ("R err <= 5%",            "R-line accuracy",                    "Is the level we picked the same one you'd draw?"),
        ("WL >= 50  (Watchlist)",  "Setup is forming",                   "Stocks worth monitoring daily"),
        ("TR >= 65  (Trigger) *",  "Setup is ripe",                      "* Stocks worth acting on -- this is what fills your buy list"),
        ("High-conviction",        "TR + all 4 confirm flags",           "Strict swing-trade entries"),
    ]
    legend_df = pd.DataFrame(legend_rows, columns=["Metric", "What it means", "When to look at it"])
    return audit_df, legend_df


def write_excel(rows: list, out_path: str):
    df = pd.DataFrame(rows).sort_values("score", ascending=False)

    # Build run-summary numbers from `rows`
    n_total   = len(df)
    n_wl      = int((df["score"] >= WATCHLIST_MIN_SCORE).sum())
    n_tr      = int((df["score"] >= TRIGGER_MIN_SCORE).sum())
    n_hc      = int(df.get("high_conviction", pd.Series([], dtype=bool)).sum()) if "high_conviction" in df.columns else 0
    run_summary = pd.DataFrame([
        ("Scan date",                   TODAY.strftime("%d-%b-%Y")),
        ("Universe candidates scored",  n_total),
        ("Watchlist  (score >= 50)",    n_wl),
        ("Trigger    (score >= 65)  *", n_tr),
        ("High-conviction (TR + all 5 conditions)", n_hc),
        ("", ""),
        ("Focus on TR >= 65 -- that's your actionable signal rate.", ""),
        ("Everything else is diagnostic.", ""),
    ], columns=["Metric", "Value"])

    audit_df, legend_df = _build_summary_sheet()

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        # ── Sheet 1: Summary ──
        startrow = 0
        run_summary.to_excel(w, sheet_name="Summary", index=False, startrow=startrow)
        startrow += len(run_summary) + 3

        # Section header for legend
        pd.DataFrame([["METRIC LEGEND -- what each row means and when to look at it"]]).to_excel(
            w, sheet_name="Summary", index=False, header=False, startrow=startrow)
        startrow += 2
        legend_df.to_excel(w, sheet_name="Summary", index=False, startrow=startrow)
        startrow += len(legend_df) + 3

        # Section header for audit
        pd.DataFrame([["FINAL PRE-BREAKOUT AUDIT (25-ticker calibration set, evaluated at T-1)"]]).to_excel(
            w, sheet_name="Summary", index=False, header=False, startrow=startrow)
        startrow += 2
        audit_df.to_excel(w, sheet_name="Summary", index=False, startrow=startrow)

        # ── Sheet 2: Watchlist (full sorted by score) ──
        df.to_excel(w, sheet_name="Watchlist", index=False)

        # ── Sheet 3: Triggers (HC first, then by score) ──
        # Include any high-conviction pick even if its score < 65, since HC
        # is the strictest signal and should always appear on the action list.
        if "high_conviction" in df.columns:
            mask = (df["score"] >= TRIGGER_MIN_SCORE) | (df["high_conviction"] == True)  # noqa: E712
        else:
            mask = df["score"] >= TRIGGER_MIN_SCORE
        triggers = df[mask].copy()
        if not triggers.empty:
            if "high_conviction" in triggers.columns:
                triggers = triggers.sort_values(
                    ["high_conviction", "score"],
                    ascending=[False, False],
                )
            else:
                triggers = triggers.sort_values("score", ascending=False)
            triggers.to_excel(w, sheet_name="Triggers", index=False)

        # ── Sheet 4: High Conviction ──
        if "high_conviction" in df.columns:
            hc = df[df["high_conviction"] == True].copy()  # noqa: E712
            if not hc.empty:
                hc = hc.sort_values("score", ascending=False)
                cols_front = [
                    "symbol", "close", "resistance", "distance_pct",
                    "score", "hc_path", "pattern_multi_touch", "pattern_vcp",
                    "pattern_cup_handle", "rs_rising_50d",
                    "rr", "stop", "target",
                ]
                others = [c for c in hc.columns if c not in cols_front]
                hc[cols_front + others].to_excel(
                    w, sheet_name="High Conviction", index=False)
    print(f"  Excel written: {out_path}")



# ─── Hard gates (v4.3, eliminative) ────────────────────────────────────────
# Built from the 11-chart audit (SUBAHOTELS, ZAPPFRESH, ADCOUNTY, FINBUD,
# MSAFE, INDIAMART, KMEW, ALIVUS, PRIMECAB, JTLIND, ROLEXRINGS) + chart
# follow-up (FIEMIND, INDOBORAX, SJS). Each gate eliminates a specific
# chart pathology and is logged in the drop funnel for transparency.

def stage2_uptrend(df: pd.DataFrame) -> dict:
    """Stage-2 transition gate (v4.4, simplified).

    Required: close > 50DMA. The 200-DMA and 52w-low checks were removed
    in v4.4 — they over-filtered recovering names in the universe.
    """
    if len(df) < 60:
        return {"pass": False, "reason": "insufficient_history"}
    c = df["Close"]
    last = float(c.iloc[-1])
    ma50 = c.rolling(50).mean()
    if pd.isna(ma50.iloc[-1]) or last <= ma50.iloc[-1]:
        return {"pass": False, "reason": "below_ma50"}
    return {"pass": True, "reason": "ok"}



def not_extended(df: pd.DataFrame, max_bar_gain: float = 0.05,
                 max_bar_atr_mult: float = 2.5) -> bool:
    """True if entry isn't a vertical chase.
       No |close-to-close| > 5% in last 5 bars AND last TR <= 2.5*ATR(20)."""
    c = df["Close"]
    if len(c) < 25:
        return False
    if (c.pct_change().tail(5).abs() > max_bar_gain).any():
        return False
    a = atr(df, 20)
    if pd.isna(a.iloc[-1]) or a.iloc[-1] <= 0:
        return True
    last = df.iloc[-1]
    return float(last["High"] - last["Low"]) <= float(a.iloc[-1]) * max_bar_atr_mult


def recent_failed_breakout(df: pd.DataFrame, R: float,
                           lookback: int = 15) -> bool:
    """True if any high in last `lookback` bars pierced R*1.03 but stock
    is now < R*0.98 — R just rejected price (ROLEXRINGS / PRIMECAB)."""
    seg = df.tail(lookback)
    if seg.empty:
        return False
    pierced = bool((seg["High"] > R * 1.03).any())
    return pierced and float(df["Close"].iloc[-1]) < R * 0.98


def recent_r_test(df: pd.DataFrame, R: float, band_pct: float = 0.04,
                  lookback: int = RECENT_R_TEST_LOOKBACK) -> dict:
    """At least one bar in last `lookback` whose High is within band_pct
    of R. Kills KMEW-style picks where R was drawn from old pivots and
    the current rally has not yet physically tested the level.
    (Note: an absorption-volume sub-clause was tried in v3.3-strict but
    conflicted with the base dry-up requirement — kept as touch-only.)"""
    if len(df) < 60 or R <= 0:
        return {"pass": False, "reason": "insufficient_history",
                "n_touches_recent": 0}
    seg = df.tail(lookback)
    near = seg[seg["High"] >= R * (1 - band_pct)]
    n = int(len(near))
    if n == 0:
        return {"pass": False, "reason": "no_recent_test",
                "n_touches_recent": 0}
    return {"pass": True, "reason": "ok", "n_touches_recent": n}


def base_metrics(df: pd.DataFrame, base_start, R: float) -> dict:
    """Geometry of the base region.
       range_pct = (base_high-base_low)/R; trailing on last 25 bars."""
    base = df.loc[base_start:]
    if base.empty or R <= 0:
        return {"range_pct": 1.0, "trailing_pct": 1.0, "is_flat": False}
    base_low = float(base["Low"].min())
    base_high = float(base["High"].max())
    range_pct = (base_high - base_low) / R
    trail = base.tail(25)
    if trail.empty:
        trailing_pct = range_pct
    else:
        trailing_pct = (float(trail["High"].max())
                        - float(trail["Low"].min())) / R
    return {"range_pct": float(range_pct),
            "trailing_pct": float(trailing_pct),
            "is_flat": bool(trailing_pct <= 0.10)}


# ─── Pattern detectors (boost score / inform HC rule) ────────────────────

def cup_and_handle(df: pd.DataFrame, base_start, R: float) -> bool:
    """Crude cup & handle (kept for diagnostics, NOT used in HC rule —
       backtest showed 0 wins / 16 trades; detector misfires)."""
    base = df.loc[base_start:]
    if len(base) < 50 or R <= 0:
        return False
    cup = base.iloc[:-5]
    handle = base.tail(15)
    if len(cup) < 40 or len(handle) < 5:
        return False
    cup_high = float(cup["High"].max())
    cup_low = float(cup["Low"].min())
    if cup_high <= 0:
        return False
    cup_depth = (cup_high - cup_low) / cup_high
    if cup_depth < 0.12 or cup_depth > 0.40:
        return False
    third = max(1, len(cup) // 3)
    mid_low = float(cup.iloc[third:2 * third]["Low"].min())
    if mid_low > cup_low * 1.05:
        return False
    handle_depth = (cup_high - float(handle["Low"].min())) / cup_high
    if handle_depth > cup_depth * 0.5:
        return False
    last_close = float(df["Close"].iloc[-1])
    if abs(R - last_close) / last_close > 0.08:
        return False
    return True


def vcp_contractions(df: pd.DataFrame, base_start) -> int:
    """Count VCP-style successive contractions in the base.
       Each pullback < 85% of prior AND < 15% absolute. >=2 is valid VCP."""
    base = df.loc[base_start:]
    if len(base) < 30:
        return 0
    h = base["High"].values
    l = base["Low"].values
    n = len(h)
    k = 5
    pivots = []
    for i in range(k, n - k):
        if h[i] == h[i - k:i + k + 1].max():
            pivots.append((i, float(h[i])))
    if len(pivots) < 2:
        return 0
    pullbacks = []
    for j in range(1, len(pivots)):
        i_prev, p_prev = pivots[j - 1]
        i_cur, _ = pivots[j]
        seg_low = float(l[i_prev:i_cur + 1].min())
        if p_prev > 0:
            pullbacks.append((p_prev - seg_low) / p_prev)
    n_contr = 0
    for j in range(1, len(pullbacks)):
        if pullbacks[j] < pullbacks[j - 1] * 0.85 and pullbacks[j] < 0.15:
            n_contr += 1
    return n_contr




def w_pattern(df: pd.DataFrame, base_start, R: float) -> bool:
    """v4.1: Double-bottom (W) pattern inside the base.

    Geometry:
      - Two swing-lows (L1, L2) within 4% of each other
      - L2 occurs at least 5 bars after L1
      - A middle peak between them >= 5% above the lower low
      - Right side recovering: current close above 0.97 * middle-peak
        OR current close already inside resistance proximity band
    """
    base = df.loc[base_start:]
    if len(base) < 25 or R <= 0:
        return False
    lows = base["Low"].values
    highs = base["High"].values
    n = len(lows)
    k = 4
    pivot_lows = []
    for i in range(k, n - k):
        if lows[i] == lows[i - k:i + k + 1].min():
            pivot_lows.append((i, float(lows[i])))
    if len(pivot_lows) < 2:
        return False
    last_close = float(df["Close"].iloc[-1])
    for a in range(len(pivot_lows) - 1):
        i1, l1 = pivot_lows[a]
        for b in range(a + 1, len(pivot_lows)):
            i2, l2 = pivot_lows[b]
            if i2 - i1 < 5:
                continue
            lo, hi = (l1, l2) if l1 <= l2 else (l2, l1)
            if (hi - lo) / lo > 0.04:
                continue
            mid_high = float(highs[i1 + 1:i2].max()) if i2 > i1 + 1 else 0.0
            if mid_high <= 0:
                continue
            if (mid_high - lo) / lo < 0.05:
                continue
            near_neckline = last_close >= mid_high * 0.97
            dist = (R - last_close) / last_close
            near_R = (PROXIMITY_MIN_PCT <= dist <= PROXIMITY_MAX_PCT)
            if near_neckline or near_R:
                return True
    return False


# ─── Scan driver ─────────────────────────────────────────────────

def scan(symbols: list, ohlcv: dict, bench: pd.Series,
         min_score: float, strict: bool = True) -> tuple:
    """Run per-ticker scan. Returns (rows, drop_counts).

    When strict=True, the v3.3 hard gates are enforced and every drop
    is logged into drop_counts for the funnel report. strict=False
    disables gates (diagnostic v1 funnel)."""
    rows = []
    drops: dict = {}

    def _drop(reason: str):
        drops[reason] = drops.get(reason, 0) + 1

    n = len(symbols)
    for i, sym in enumerate(symbols, 1):
        if sym not in ohlcv:
            _drop("no_data"); continue
        df = ohlcv[sym]
        # v4.0 GATE 0: minimum history (100 trading days)
        if len(df) < MIN_HISTORY_DAYS:
            _drop("insufficient_history"); continue
        if df["Volume"].rolling(50).mean().iloc[-1] < MIN_AVG_VOL:
            _drop("liquidity"); continue

        # ── HARD GATE 1: Stage-2 uptrend (Minervini, MA200-based) ──
        # NOTE v4.0: 50DMA-falling gate REMOVED per user request.
        if strict:
            s2 = stage2_uptrend(df)
            if not s2["pass"]:
                _drop(f"stage2:{s2['reason']}"); continue

        # ── HARD GATE 1d: entry must not be a vertical chase ──
        if strict and not not_extended(df):
            _drop("extended_entry"); continue

        try:
            res = detect_resistance(df)
            if res is None:
                _drop("no_resistance"); continue
            R = res["R"]

            # ── HARD GATE 2: distance to resistance in [-5%, +4%] ──
            if strict and not (PROXIMITY_MIN_PCT <= res["distance_pct"]
                               <= PROXIMITY_MAX_PCT):
                _drop("dist_out_of_band"); continue

            # ── HARD GATE 2a: recent failed breakout ──
            if strict and recent_failed_breakout(df, R):
                _drop("recent_failed_bo"); continue

            # ── HARD GATE 2b: recent R touch (50 sessions) ──
            rrt = recent_r_test(df, R, lookback=RECENT_R_TEST_LOOKBACK)
            if strict and not rrt["pass"]:
                _drop(f"r_test:{rrt['reason']}"); continue

            # ── HARD GATE 3: base width <= 40% (no wider bases) ──
            base_geo = base_metrics(df, res["base_start"], R)
            if strict and base_geo["range_pct"] > MAX_BASE_RANGE_PCT:
                _drop("base_too_wide"); continue

            # ── HARD GATE 4: rising relative strength over last 50 sessions ──
            rs = rs_rising(df, bench, lookback=RS_RISING_LOOKBACK)
            if strict and not rs["pass"]:
                _drop("rs_not_rising_50d"); continue

            # Volume ratio kept as informational column only (no longer a gate)
            v50 = float(df["Volume"].rolling(50).mean().iloc[-1])
            base_window = df["Volume"].iloc[-28:-3]
            v_base = float(base_window.mean()) if len(base_window) else v50
            v_ratio = (v_base / v50) if v50 > 0 else 1.0

            score = compute_score(df, res, bench)
            if score["score"] < min_score:
                _drop("low_score"); continue

            # ── Pattern detection (priority: multi_touch > vcp > W > C&H) ──
            n_vcp = vcp_contractions(df, res["base_start"])
            pattern_multitouch = bool(res["touches"] >= HC_MULTITOUCH_MIN)
            pattern_vcp = bool(n_vcp >= 2)
            pattern_w = bool(w_pattern(df, res["base_start"], R))
            pattern_ch = bool(cup_and_handle(df, res["base_start"], R))
            if pattern_multitouch:
                pattern_label = "multi_touch"
            elif pattern_vcp:
                pattern_label = "vcp"
            elif pattern_w:
                pattern_label = "w_pattern"
            elif pattern_ch:
                pattern_label = "cup_handle"
            else:
                pattern_label = ""
            pattern_ok = bool(pattern_multitouch or pattern_vcp
                              or pattern_w or pattern_ch)

            risk = risk_plan(df, res)
            distance_pct_value = round(res["distance_pct"] * 100, 2)

            # === HIGH-CONVICTION (v4.0) ===
            # All hard gates already enforced above. HC requires that the
            # setup also matches at least one of the three approved patterns.
            high_conviction = pattern_ok
            hc_path = pattern_label  # "multi_touch" | "vcp" | "cup_handle" | ""

            rows.append({
                "symbol": sym,
                "high_conviction": high_conviction,
                "hc_path": hc_path,
                "score": score["score"],
                "close": round(float(df["Close"].iloc[-1]), 2),
                "resistance": round(R, 2),
                "distance_pct": distance_pct_value,
                "touches": res["touches"],
                "base_days": res["base_len_days"],
                "base_range_pct": round(base_geo["range_pct"] * 100, 2),
                "trail25_range_pct": round(base_geo["trailing_pct"] * 100, 2),
                "vol_ratio_base": round(v_ratio, 3),
                "n_touches_recent": rrt.get("n_touches_recent", 0),
                **{k: score[k] for k in
                    ["base_quality", "vcr", "vdu", "proximity", "trend", "rs"]},
                "vcr_raw": score["vcr_raw"],
                "vdu_raw": score["vdu_raw"],
                "higher_lows": score["higher_lows"],
                "rs_rising_50d": rs["pass"],
                "rs_slope_50d": round(rs["slope"], 6),
                "pattern_multi_touch": pattern_multitouch,
                "pattern_vcp": pattern_vcp,
                "pattern_w": pattern_w,
                "pattern_cup_handle": pattern_ch,
                "n_vcp": n_vcp,
                **risk,
            })
        except Exception as e:
            print(f"  [{sym}] error: {e}")
        if i % 100 == 0:
            print(f"  scanned {i}/{n} ...")
    return rows, drops


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Breakout Scanner v4.3")
    p.add_argument("--max", type=int, default=0,
                   help="cap universe size (0 = all)")
    p.add_argument("--min-score", type=float, default=WATCHLIST_MIN_SCORE)
    p.add_argument("--charts", type=int, default=20,
                   help="render top-N charts")
    p.add_argument("--lookback", type=int, default=LOOKBACK_DAYS)
    p.add_argument("--no-strict", action="store_true",
                   help="disable v3.3 hard gates (diagnostic v1 funnel)")
    p.add_argument("--high-conviction", action="store_true",
                   help="only output HC picks (v3.3 calibrated rule)")
    p.add_argument("--symbols-csv", type=str, default="",
                   help="path to CSV with a 'ticker' column to use as universe "
                        "(overrides multi_pct_down_report.xlsx)")
    p.add_argument("--screener-url", type=str, default="",
                   help="screener.in screen URL to fetch universe from "
                        "(overrides multi_pct_down_report.xlsx)")
    p.add_argument("--out-tag", type=str, default="",
                   help="suffix appended to output Excel filenames "
                        "(e.g. 'watchlist' -> breakout_watchlist_watchlist.xlsx)")
    args = p.parse_args()
    strict = not args.no_strict

    # Tee stdout+stderr to Output/logs/ so logs land in the right place.
    log_path = os.path.join(OUTPUT_DIR, "logs",
                            f"logs_breakout_scanner_angel_v35_{TIMESTAMP}.txt")
    _tee_out = _Tee(sys.stdout, log_path)
    _tee_err = _Tee(sys.stderr, log_path)
    sys.stdout = _tee_out
    sys.stderr = _tee_err

    print("=" * 70)
    print(f"  BREAKOUT SCANNER v4.3 — {TODAY.strftime('%d-%b-%Y')}")
    if args.symbols_csv:
        print(f"  Source: {args.symbols_csv} (custom CSV)")
    elif args.screener_url:
        print(f"  Source: screener.in screen")
    else:
        print(f"  Source: {os.path.basename(PCT_DOWN_REPORT)}")
    print(f"  Mode  : {'STRICT (v3.3 hard gates ON)' if strict else 'DIAGNOSTIC (gates OFF)'}")
    if args.high_conviction:
        print("  Filter: HIGH-CONVICTION only (v3.3 rule)")
    print("=" * 70)

    if args.symbols_csv:
        scsv = pd.read_csv(args.symbols_csv)
        if "ticker" not in scsv.columns:
            raise SystemExit(f"--symbols-csv {args.symbols_csv} must have a 'ticker' column")
        tickers = sorted({str(t).strip() for t in scsv["ticker"].dropna() if str(t).strip()})
        print(f"  Custom universe: {len(tickers)} tickers from {args.symbols_csv}")
    elif args.screener_url:
        tickers = fetch_screener_universe(args.screener_url)
    else:
        tickers = fetch_universe()
    if args.max > 0:
        tickers = tickers[:args.max]
        print(f"  Universe capped to {len(tickers)}")

    ohlcv = fetch_ohlcv(tickers, args.lookback)
    bench = fetch_benchmark(args.lookback)

    print("\n  Scanning ...")
    effective_min_score = 0.0 if args.high_conviction else args.min_score
    rows, drops = scan(list(ohlcv.keys()), ohlcv, bench,
                       effective_min_score, strict=strict)

    # Drop funnel
    if drops:
        print("\n  Drop funnel (reason -> count):")
        for reason, cnt in sorted(drops.items(), key=lambda x: -x[1]):
            print(f"    {reason:32s} {cnt:>5d}")

    print(f"\n  Candidates surviving all gates (score >= {effective_min_score}): {len(rows)}")

    if rows:
        n_mt   = sum(1 for r in rows if r["pattern_multi_touch"])
        n_vcp  = sum(1 for r in rows if r["pattern_vcp"])
        n_w    = sum(1 for r in rows if r["pattern_w"])
        n_ch   = sum(1 for r in rows if r["pattern_cup_handle"])
        n_rsr  = sum(1 for r in rows if r["rs_rising_50d"])
        n_d    = sum(1 for r in rows if PROXIMITY_MIN_PCT * 100
                                       <= r["distance_pct"]
                                       <= PROXIMITY_MAX_PCT * 100)
        n_b40  = sum(1 for r in rows if r["base_range_pct"] <= 40.0)
        n_hc   = sum(1 for r in rows if r["high_conviction"])
        n_hc_mt = sum(1 for r in rows if r.get("hc_path") == "multi_touch")
        n_hc_vcp = sum(1 for r in rows if r.get("hc_path") == "vcp")
        n_hc_w = sum(1 for r in rows if r.get("hc_path") == "w_pattern")
        n_hc_ch = sum(1 for r in rows if r.get("hc_path") == "cup_handle")
        print("  HC v4.3 condition pass rates:")
        print(f"    patterns: multi_touch={n_mt}, vcp={n_vcp},"
              f" w_pattern={n_w}, cup_handle={n_ch}")
        print(f"    rs_rising_50d={n_rsr}, dist[-5,+4]={n_d},"
              f" base<=40%={n_b40}")
        print(f"    HIGH-CONVICTION total: {n_hc}  "
              f"(multi_touch={n_hc_mt}, vcp={n_hc_vcp},"
              f" w={n_hc_w}, cup_handle={n_hc_ch})")
        print(f"      Priority order: multi_touch > vcp > w_pattern > cup_handle")
        print(f"      All HC names already passed: stage2 (>MA50),"
              f" not_extended, dist[-5,+4], r_test_50, base<=40%,"
              f" rs_rising_50d")

    if rows:
        excel_full = os.path.join(SCRIPT_DIR, "breakout_watchlist.xlsx")
        if args.out_tag:
            excel_full = os.path.join(SCRIPT_DIR,
                                      f"breakout_watchlist_{args.out_tag}.xlsx")
        write_excel(rows, excel_full)

    if args.high_conviction:
        rows = [r for r in rows if r.get("high_conviction")]
        print(f"  High-conviction picks: {len(rows)}")

    if not rows:
        print("  No candidates found.")
        return

    n_hc = sum(1 for r in rows if r.get("high_conviction"))
    print(f"  Rows in output: {len(rows)} | HC: {n_hc}")

    if args.high_conviction:
        excel_path = os.path.join(SCRIPT_DIR, "breakout_high_conviction.xlsx")
        if args.out_tag:
            excel_path = os.path.join(
                SCRIPT_DIR, f"breakout_high_conviction_{args.out_tag}.xlsx")
        write_excel(rows, excel_path)

    rows_sorted = sorted(rows, key=lambda r: (not r.get("high_conviction"),
                                              -r["score"]))
    charts_dir = os.path.join(SCRIPT_DIR, os.pardir, "Output", "breakout_charts")
    os.makedirs(charts_dir, exist_ok=True)
    print(f"\n  Rendering top {min(args.charts, len(rows_sorted))} charts ...")
    for r in rows_sorted[:args.charts]:
        sym = r["symbol"]
        df = ohlcv[sym]
        res = detect_resistance(df)
        if res is None:
            continue
        score = compute_score(df, res, bench)
        risk = risk_plan(df, res)
        flags = {
            "multi_touch": r.get("pattern_multi_touch", False),
            "vcp": r.get("pattern_vcp", False),
            "w_pattern": r.get("pattern_w", False),
            "cup_handle": r.get("pattern_cup_handle", False),
            "rs_rising_50d": r.get("rs_rising_50d", False),
        }
        prefix = "HC_" if r.get("high_conviction") else ""
        out = os.path.join(charts_dir, f"{prefix}{sym}_breakout.html")
        render_chart(sym, df.tail(args.lookback), res, score, risk, flags, out)
    print(f"  Charts saved to: {charts_dir}")

    print("\n  Top 10 (HC first, then by score):")
    cols = ["symbol", "high_conviction", "hc_path", "score", "close",
            "resistance", "distance_pct", "touches", "base_days",
            "base_range_pct", "rs_rising_50d", "rr"]
    top = pd.DataFrame(rows_sorted[:10])[cols]
    print(top.to_string(index=False))
    print("\nDONE.")


if __name__ == "__main__":
    main()
