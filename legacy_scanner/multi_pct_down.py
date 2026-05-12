"""
Multi-Universe Pct-Down Screener (NSE / NSE-SME / BSE-SME)
==========================================================

GOAL
----
Find stocks that have pulled back a moderate amount (default 2%-21%) from
recent highs, are NOT near the 52-week low (falling-knife guard), have
outperform the
NIFTY 500 over 3 months (relative strength), are still in a long-term
uptrend (above 200-DMA), are forming higher lows (base building), trade
above Rs.45 (no pennies), and (for NSE main board) sit in a sensible
market-cap band. Output is an Excel workbook with one sheet per
(universe x lookback period: 12M).


==========================================================================
DATA SOURCES (single source of truth — change here if endpoints move)
==========================================================================

Universe / metadata sources (HTTP, no auth):
  1. NSE main board listing
       URL : https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv
       Use : seed list of all NSE-listed equity symbols
       Fn  : fetch_nse_equity_universe()
  2. NSE SME (Emerge) listing
       URL : https://nsearchives.nseindia.com/emerge/corporates/content/
             SME_EQUITY_L.csv
       Use : seed list of NSE Emerge SME symbols
       Fn  : fetch_nse_sme_universe()
  3. BSE active equity listing (filtered to SME groups M / MT / MS)
       URL : https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w
             ?Group=&Scripcode=&industry=&segment=Equity&status=Active
       Use : seed list of BSE SME platform symbols
             AND the .NS -> .BO ticker-fallback map for yfinance
       Fns : fetch_bse_sme_universe(), fetch_bse_full_symbol_map()
  4. NSE F&O underlyings
       URL : https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv
       Use : remove F&O-listed names from NSE main universe
       Fn  : load_fno_symbols()

OHLCV (price history) sources, in fallback order:
  PRIMARY  : Angel One SmartAPI (smartapi-python SDK)
             - Auth via .env: ANGEL_API_KEY / ANGEL_CLIENT_CODE /
               ANGEL_PIN / ANGEL_TOTP_SECRET
             - Daily candles via getCandleData()
             - Coverage: NSE main + NSE Emerge + BSE (incl. SME)
             - Rate limit: ~2 req/sec (handled by angel_client)
             - Resolved through `data_provider.download(...)`
  FALLBACK1: jugaad-data (only NSE main board; scrapes nseindia.com)
  FALLBACK2: yfinance (broad coverage, occasionally flaky / empty)
  Routed automatically by `data_provider.py`. yfinance is also still
  used directly here for `_get_market_cap_cr(...)` because Angel does
  NOT expose market cap or shares-outstanding.


==========================================================================
END-TO-END WORKFLOW (with realistic NSE numbers from a recent run)
==========================================================================

Stage 0 - Universe load (free, ~seconds)
  Pull the three universe lists above.
  NSE example  : 2,360 symbols pulled from NSE archives CSV.

Stage 1 - F&O drop  [NSE only; toggled via FILTER_MATRIX.apply_fno]
  Remove names that have futures/options listed.
  NSE example  : 2,360 - 209 = 2,151 remaining.

Stage 2 - Pre-warm Angel One session (single login)
  One-time login BEFORE worker threads start (avoids parallel TOTP
  hits which Angel rate-limits at the auth endpoint).
  Also loads/caches the Angel scrip-master (~25 MB, weekly TTL) once.

Stage 3 - OHLCV pull (Angel primary, jugaad/yf fallback)
  For every surviving ticker, request the last 13 months of daily
  candles. Run with `workers` threads (default 4). Per-thread retries
  + exponential backoff handle transient failures.
  NSE example  : 2,151 fetches, 0 errors / no-data.

Stage 4 - EARLY filters applied immediately after each pull
  (Inside _analyze_one - runs PER TICKER, no extra network cost.)

  4a) Min last close:
        keep only if  last_close >= MIN_LAST_CLOSE  (default Rs.45).
      Drops penny / micro-priced names that are noisy / hard to trade.

  4b) 52-week band:
        keep only if  -MAX_PCT <= (last_close / 52W_high - 1) <= -MIN_PCT
      Default band: -21% to -2%.

  4c) 52-week low buffer (avoid falling knives):
        keep only if  (last_close - 52W_low) / 52W_low > LOW52W_BUFFER_PCT
      Default: drop if last close is within 20% of the 52-week low.
      Ensures the stock has bounced meaningfully off its low.

  4d) Drawdown duration  [CURRENTLY DISABLED]:
        (would keep only if  DD_MIN_DAYS < days_since(5M_high) < DD_MAX_DAYS)
      Default: keep if the 5-month high was hit between 90 and 150
      days ago. Drops stocks that pulled back too recently (< 90d)
      and those stuck in a drawdown too long (>= 150d).

  4e) 1Y runup cap:
        drop if  1Y_return > MAX_1Y_RUNUP_PCT  (default 54%).
      Removes stocks that have already run up too much in the past year.

  4f) Relative Strength vs NIFTY 500 over 3M:
        keep only if  stock_3M_return > NIFTY_500_3M_return
      Requires the stock to be outperforming the broad market over the
      last 3 months. Index return is fetched once at startup from
      Yahoo ticker ^CRSLDX.

  4g) Above 200-DMA:
        keep only if  last_close > mean(Close[-200:])
      Long-term uptrend filter.

  4h) Higher lows (swing-low staircase, last 50 sessions):
        Find the absolute low in the last 50 sessions ("base").
        Detect swing-low pivots (low <= N bars on each side, N=3)
        after the base.  Keep only if at least 2 successive swing
        lows form an ascending sequence above the base.
      Base-building filter: confirms a staircase of rising lows
      rather than just comparing two window minimums.

Stage 5 - Market cap
  5a) Mcap data required  [ALL universes]:
      Drop if yfinance cannot provide market_cap or shares_outstanding.
      Ensures every output row has a valid Mcap (Cr) value.
  5b) Mcap band  [NSE only; toggled via apply_mcap]:
      Compute mcap_cr = shares_outstanding * last_close / 1e7
      via yfinance.Ticker(...).fast_info.
      Keep only if  MCAP_MIN_CR <= mcap_cr <= MCAP_MAX_CR  (350 - 34,000 Cr).
      Done AFTER stages 4a/4b so we make ~60% fewer (slow) yfinance calls.
      NSE example  : 894 - 134 (out of band) - 96 (no mcap) = 664 remaining.

Stage 6 - Per-period band check (12M, hard-coded in PERIODS)
  For each period N in {12} months: compute the high in the last
  N months and check the same -MAX_PCT..-MIN_PCT band against it.
  A stock can pass for some periods and fail others; each period gets
  its own output sheet.

Stage 7 - Excel write
  One workbook with 1 sheet per universe:
    <UNI> 12M


==========================================================================
FILTER MATRIX (per-universe toggles for the universe-wide filters)
==========================================================================
  +-----------+-----------+-----------------+----------------+
  | Universe  | F&O drop  | Mcap 350-34k Cr | Pct down 2-21% |
  +-----------+-----------+-----------------+----------------+
  | NSE       |    Yes    |      Yes        |      Yes       |
  | NSE_SME   |    No     |      No         |      Yes       |
  | BSE_SME   |    No     |      No         |      Yes       |
  +-----------+-----------+-----------------+----------------+
Always-on per-ticker filters (hard-coded in _analyze_one):
  - Last close >= Rs.45
  - 52W band: -21% to -2% from 52-week high
  - 52W low buffer: (Close - 52W_low) / 52W_low > 20%
  - Drawdown duration: DISABLED (value still tracked)
  - RS vs NIFTY 500: stock 3M return > index 3M return
  - Above 200-DMA
  - Higher lows: >=2 ascending swing lows in last 50 sessions
  - 1Y runup cap: drop if 1Y return > 54%
  - Mcap data required: drop if yfinance has no mcap/shares data
Only the F&O drop and the mcap band are matrix-toggled.


==========================================================================
USAGE
==========================================================================
  python multi_pct_down.py
  python multi_pct_down.py --min 5 --max 25
  python multi_pct_down.py --skip bse_sme --workers 4
  python multi_pct_down.py --max-symbols 100   # quick smoke test
  python multi_pct_down.py --workers 2         # gentlest on rate limits
  python multi_pct_down.py -o my_report

OUTPUT
  multi_pct_down.xlsx (next to this script unless --out / -o overrides),
  1 period sheet per universe = 3 sheets total.
"""

import os
import sys
import csv
import io
import json
import time
import argparse
import datetime
import warnings
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

warnings.filterwarnings("ignore")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Make sibling modules (data_provider, angel_client) importable when this
# script is run directly from anywhere.
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
try:
    import data_provider as _dp
except Exception as _e:  # pragma: no cover - import-time diagnostics
    _dp = None
    print("WARN: data_provider import failed (%s); will fall back to "
          "yfinance only." % _e)
TODAY = datetime.date.today()
PERIODS = [(12, "12M")]

# Filters
MCAP_MIN_CR = 350
MCAP_MAX_CR = 34000
MAX_1Y_RUNUP_PCT = 54.0
MIN_LAST_CLOSE = 45.0      # drop sub-Rs.45 names (penny / illiquid)
DMA200_WINDOW = 200        # require last_close > 200-day MA
HL_LOOKBACK = 50           # higher-lows: sessions to scan for base + pivots
HL_SWING_ORDER = 3         # pivot detection: low must be <= N bars each side
HL_MIN_HIGHER_LOWS = 2     # require at least N ascending swing lows after base
LOW52W_BUFFER_PCT = 20.0   # require last_close >= 52W_low * (1 + 20%)
RS_SESSIONS = 50           # RS comparison window in trading sessions
RS_INDEX_TICKER = "^CRSLDX"  # NIFTY 500 (Yahoo)
DD_LOOKBACK_M = 5          # drawdown duration: window to find pivot high
DD_MIN_DAYS = 90           # min days since 5M high (not too fresh)
DD_MAX_DAYS = 150          # max days since 5M high (not stuck too long)
DEFAULT_WORKERS = 4   # lower than before: Yahoo rate-limits aggressive
                      # parallelism and starts returning empty data
RETRY_BACKOFF_S = 1.0 # base backoff seconds between retries

# ---------------------------------------------------------------------------
# Per-universe filter matrix (single source of truth - used by run() and
# echoed at the top of every run for explainability).
#
#   +-----------+-----------+-----------------+----------+----------------+
#   | Universe  | F&O drop  | Mcap 350-34k Cr | 1Y runup | Pct down 2-30% |
#   +-----------+-----------+-----------------+----------+----------------+
#   | NSE       |    Yes    |      Yes        |   Yes    |      Yes       |
#   | NSE_SME   |    No     |      No         |   Yes    |      Yes       |
#   | BSE_SME   |    No     |      No         |   Yes    |      Yes       |
#   +-----------+-----------+-----------------+----------+----------------+
#
# 1Y runup and Pct-down filters are always applied (hard-coded in the
# per-ticker analyzer); the booleans below toggle only F&O removal and the
# market-cap band.
# ---------------------------------------------------------------------------
FILTER_MATRIX = {
    # apply_fno  : drop F&O underlyings
    # apply_mcap : enforce MCAP_MIN_CR..MCAP_MAX_CR band
    # max_retries: Yahoo download retry budget (NSE_SME=1 because Yahoo
    #              does not carry NSE Emerge listings, so retries waste time)
    "NSE":     {"apply_fno": True,  "apply_mcap": True,  "max_retries": 3},
    "NSE_SME": {"apply_fno": False, "apply_mcap": False, "max_retries": 1},
    "BSE_SME": {"apply_fno": False, "apply_mcap": False, "max_retries": 3},
}


def print_filter_matrix():
    print("  Filter matrix:")
    print("  +-----------+----------+----------+----------+----------+")
    print("  | Universe  | F&O drop | Mcap band| 1Y runup | Pct down |")
    print("  +-----------+----------+----------+----------+----------+")
    for uni, cfg in FILTER_MATRIX.items():
        print("  | %-9s |   %-3s    |   %-3s    |   Yes    |   Yes    |" % (
            uni,
            "Yes" if cfg["apply_fno"] else "No",
            "Yes" if cfg["apply_mcap"] else "No",
        ))
    print("  +-----------+----------+----------+----------+----------+")

# --- live data sources ------------------------------------------------------
NSE_EQUITY_URL = (
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
)
NSE_SME_URL = (
    "https://nsearchives.nseindia.com/emerge/corporates/content/"
    "SME_EQUITY_L.csv"
)
BSE_LIST_URL = (
    "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
    "?Group=&Scripcode=&industry=&segment=Equity&status=Active"
)
BSE_SME_GROUPS = {"M", "MT", "MS"}

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _http_get(url, referer=None, timeout=30):
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        **({"Referer": referer} if referer else {}),
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_nse_equity_universe():
    """Live-fetch NSE main board list -> [(yahoo, symbol, name), ...]."""
    print("-> Fetching NSE main board list ...")
    raw = _http_get(NSE_EQUITY_URL, referer="https://www.nseindia.com/")
    text = raw.decode("utf-8", errors="ignore")
    out = []
    r = csv.reader(io.StringIO(text))
    next(r, None)
    for row in r:
        if not row:
            continue
        sym = row[0].strip()
        name = row[1].strip() if len(row) > 1 else sym
        if sym:
            out.append(("%s.NS" % sym, sym, name))
    print("   NSE symbols: %d" % len(out))
    return out


def fetch_nse_sme_universe():
    """Live-fetch NSE SME (Emerge) list -> [(yahoo, symbol, name), ...]."""
    print("-> Fetching NSE SME (Emerge) list ...")
    raw = _http_get(NSE_SME_URL, referer="https://www.nseindia.com/emerge/")
    text = raw.decode("utf-8", errors="ignore")
    out = []
    r = csv.reader(io.StringIO(text))
    next(r, None)
    for row in r:
        if not row:
            continue
        sym = row[0].strip()
        name = row[1].strip() if len(row) > 1 else sym
        if sym:
            out.append(("%s.NS" % sym, sym, name))
    print("   NSE_SME symbols: %d" % len(out))
    return out


def fetch_bse_sme_universe():
    """Live-fetch BSE SME platform list -> [(yahoo, code, name), ...]."""
    print("-> Fetching BSE SME platform list ...")
    raw = _http_get(BSE_LIST_URL, referer="https://www.bseindia.com/")
    data = json.loads(raw)
    out = []
    for r in data:
        if r.get("GROUP") not in BSE_SME_GROUPS:
            continue
        if r.get("Status") != "Active":
            continue
        code = (r.get("SCRIP_CD") or "").strip()
        name = (r.get("Scrip_Name") or r.get("Issuer_Name") or code).strip()
        if code:
            out.append(("%s.BO" % code, code, name))
    print("   BSE_SME symbols: %d" % len(out))
    return out


def fetch_bse_full_symbol_map():
    """Return {scrip_id_upper: '<scripcode>.BO'} for ALL active BSE
    equities (used as a .NS -> .BO fallback when Yahoo doesn't carry the
    NSE listing)."""
    print("-> Fetching BSE full equity list (for NSE->BSE fallback) ...")
    try:
        raw = _http_get(BSE_LIST_URL, referer="https://www.bseindia.com/")
        data = json.loads(raw)
    except Exception as e:
        print("   WARN  Could not fetch BSE list (%s); fallback disabled."
              % e)
        return {}
    mp = {}
    for r in data:
        if r.get("Status") != "Active":
            continue
        sid = (r.get("scrip_id") or "").strip().upper()
        code = (r.get("SCRIP_CD") or "").strip()
        if sid and code:
            mp[sid] = "%s.BO" % code
    print("   BSE active equities indexed: %d" % len(mp))
    return mp


# F&O underlyings list (NSE) -> used to drop F&O names from the universe.
FNO_URL = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"


def load_fno_symbols():
    """Return set of NSE symbols that have F&O contracts."""
    try:
        raw = _http_get(FNO_URL, referer="https://www.nseindia.com/")
    except Exception as e:
        print("   WARN  Could not fetch F&O list (%s); skipping F&O filter."
              % e)
        return set()
    syms = set()
    for ln in raw.decode("utf-8", errors="ignore").splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 2:
            continue
        sym = parts[1].strip().upper()
        if not sym or sym == "SYMBOL" or sym.startswith("NIFTY") or \
                sym.startswith("BANKNIFTY") or sym.startswith("FINNIFTY") or \
                sym.startswith("MIDCPNIFTY"):
            continue
        syms.add(sym)
    return syms


# --- helpers ----------------------------------------------------------------

def _months_ago(n):
    # Accepts int or float (e.g. 4.5). Approximates 1 month = 30.44 days.
    return TODAY - datetime.timedelta(days=int(round(float(n) * 30.44)))


def _find_swing_lows(lows_arr, order=3):
    """Return list of (index, value) for swing-low pivots.
    A swing low at position i satisfies:
      lows_arr[i] <= min(lows_arr[i-order:i])
      lows_arr[i] <= min(lows_arr[i+1:i+order+1])
    """
    swings = []
    for i in range(order, len(lows_arr) - order):
        left_min = min(lows_arr[i - order:i])
        right_min = min(lows_arr[i + 1:i + order + 1])
        if lows_arr[i] <= left_min and lows_arr[i] <= right_min:
            swings.append((i, float(lows_arr[i])))
    return swings


def _get_market_cap_cr(yf, ticker, last_close=None):
    """Yahoo `fast_info.market_cap` is None for most NSE/BSE tickers,
    so we compute mcap = shares * last_close."""
    try:
        fi = yf.Ticker(ticker).fast_info
        try:
            mc = fi.get("market_cap") if hasattr(fi, "get") else \
                getattr(fi, "market_cap", None)
        except Exception:
            mc = None
        if mc:
            return float(mc) / 1e7
        try:
            shares = fi.get("shares") if hasattr(fi, "get") else \
                getattr(fi, "shares", None)
        except Exception:
            shares = None
        if shares and last_close:
            return float(shares) * float(last_close) / 1e7
    except Exception:
        return None
    return None


# --- per-ticker work --------------------------------------------------------

def _yf_download_with_retry(yf, ticker, start_date, max_retries):
    """Download price history with retries + backoff. Returns DataFrame
    or None. Treats empty df as failure (Yahoo's typical rate-limit
    response is 200 OK with an empty body).

    Routes through ``data_provider.download`` so Angel One SmartAPI is
    tried first, with jugaad-data and yfinance as automatic fallbacks.
    Falls back to direct yfinance only if data_provider failed to import.
    """
    for attempt in range(max(1, max_retries)):
        try:
            if _dp is not None:
                df = _dp.download(
                    ticker, start=start_date, progress=False,
                    auto_adjust=False, threads=False,
                )
            else:
                df = yf.download(
                    ticker, start=start_date.isoformat(),
                    progress=False, auto_adjust=False, threads=False,
                )
            if df is not None and not df.empty and "Close" in df.columns:
                return df
        except Exception:
            pass
        # backoff only between retries (not after the last attempt)
        if attempt < max_retries - 1:
            time.sleep(RETRY_BACKOFF_S * (2 ** attempt))
    return None


def _fetch_history(yf, primary_ticker, fallback_ticker, start_date,
                   max_retries):
    """Try primary (.NS) first; if empty, try fallback (.BO).
    Returns (df, ticker_used) or (None, primary)."""
    df = _yf_download_with_retry(yf, primary_ticker, start_date, max_retries)
    if df is not None:
        return df, primary_ticker
    if fallback_ticker and fallback_ticker != primary_ticker:
        df = _yf_download_with_retry(yf, fallback_ticker, start_date,
                                     max_retries)
        if df is not None:
            return df, fallback_ticker
    return None, primary_ticker


def _analyze_one(yf, ticker, symbol, name, start_date,
                 mcap_min, mcap_max, max_1y_runup, min_pct, max_pct,
                 apply_mcap, fallback_ticker=None, max_retries=3,
                 index_ret_3m=None):
    """Return dict for one ticker:
        {'_drop': str|None, 'periods': {label: row|None}}
    """
    df, used = _fetch_history(yf, ticker, fallback_ticker, start_date,
                              max_retries)
    if df is None:
        return {"_drop": "no_data"}
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    closes = df["Close"].dropna()
    highs = df["High"].dropna() if "High" in df.columns else closes
    if closes.empty:
        return {"_drop": "no_close"}
    # Track which ticker actually delivered data (for diagnostics).
    ticker = used

    last_close = float(closes.iloc[-1])
    last_date = closes.index[-1].date()

    # ----- EARLY FILTER 0: minimum last close ----------------------------
    if last_close < MIN_LAST_CLOSE:
        return {"_drop": "price_%.0f" % last_close}

    # ----- EARLY FILTER 1: down 2-30% from 52-week high ------------------
    cutoff_52w = pd.Timestamp(_months_ago(12))
    highs_52w = highs[highs.index >= cutoff_52w]
    if highs_52w.empty:
        return {"_drop": "no_52w_high"}
    hi_52w = float(highs_52w.max())
    if hi_52w <= 0:
        return {"_drop": "bad_52w_high"}
    pct_52w = (last_close - hi_52w) / hi_52w * 100.0
    if not (-max_pct <= pct_52w <= -min_pct):
        return {"_drop": "band52w_%.0f" % pct_52w}

    # ----- EARLY FILTER 1b: 52W low buffer (avoid falling knives) --------
    lows_52w = (df["Low"].dropna() if "Low" in df.columns else closes)
    lows_52w = lows_52w[lows_52w.index >= cutoff_52w]
    if not lows_52w.empty:
        lo_52w = float(lows_52w.min())
        if lo_52w > 0:
            buf_pct = (last_close - lo_52w) / lo_52w * 100.0
            if buf_pct <= LOW52W_BUFFER_PCT:
                return {"_drop": "low52w_%.0f" % buf_pct}

    # ----- EARLY FILTER 1c: drawdown duration (DISABLED) ----------------
    # cutoff_dd = pd.Timestamp(_months_ago(DD_LOOKBACK_M))
    # highs_dd = highs[highs.index >= cutoff_dd]
    # if highs_dd.empty:
    #     return {"_drop": "no_dd_high"}
    # hi_dd_date = highs_dd.idxmax().date()
    # days_since_hi = (last_date - hi_dd_date).days
    # if days_since_hi <= DD_MIN_DAYS or days_since_hi >= DD_MAX_DAYS:
    #     return {"_drop": "dd_%d" % days_since_hi}

    # ----- EARLY FILTER 2: 1Y price change (computed; runup filter off) --
    cutoff_1y = pd.Timestamp(_months_ago(12))
    closes_1y = closes[closes.index >= cutoff_1y]
    first_close = float(closes_1y.iloc[0]) if not closes_1y.empty \
        else float(closes.iloc[0])
    pct_1y = ((last_close - first_close) / first_close * 100.0
              if first_close > 0 else None)
    if pct_1y is not None and pct_1y > max_1y_runup:
        return {"_drop": "runup_%.0f" % pct_1y}

    # ----- EARLY FILTER 2b: RS vs NIFTY 500 over 50 sessions -------------
    if index_ret_3m is not None:
        if len(closes) >= RS_SESSIONS:
            base_rs = float(closes.iloc[-RS_SESSIONS])
            if base_rs > 0:
                ret_rs = (last_close - base_rs) / base_rs * 100.0
                if ret_rs <= index_ret_3m:
                    return {"_drop": "rs_%.0f" % ret_rs}

    # ----- EARLY FILTER 3: above 200-DMA ---------------------------------
    if len(closes) < DMA200_WINDOW:
        return {"_drop": "short_history"}
    dma200 = float(closes.iloc[-DMA200_WINDOW:].mean())
    if dma200 <= 0 or last_close <= dma200:
        return {"_drop": "below_200dma"}

    # ----- EARLY FILTER 4: higher-lows staircase (last 50 sessions) ------
    # Find the base (absolute low in last HL_LOOKBACK sessions), then
    # detect swing-low pivots after the base and require at least
    # HL_MIN_HIGHER_LOWS ascending swing lows.
    lows = df["Low"].dropna() if "Low" in df.columns else closes
    if len(lows) < HL_LOOKBACK:
        return {"_drop": "short_lows"}
    lows_window = lows.iloc[-HL_LOOKBACK:]
    lows_vals = lows_window.values.astype(float).tolist()
    base_pos = int(lows_window.values.argmin())
    base_val = lows_vals[base_pos]
    # Swing-low pivots after the base
    all_swings = _find_swing_lows(lows_vals, order=HL_SWING_ORDER)
    swings_after = [(i, v) for i, v in all_swings if i > base_pos]
    # Build greedy ascending sequence from the base
    higher_lows = []
    prev_val = base_val
    for _i, v in swings_after:
        if v > prev_val:
            higher_lows.append(round(v, 2))
            prev_val = v
    # Tentative: if min of the last SWING_ORDER bars is above the
    # latest confirmed level, count it as an additional higher low.
    tail_low = float(min(lows_vals[-HL_SWING_ORDER:]))
    if tail_low > prev_val:
        higher_lows.append(round(tail_low, 2))
    hl_count = len(higher_lows)
    hl_base = round(base_val, 2)
    if hl_count < HL_MIN_HIGHER_LOWS:
        return {"_drop": "no_higher_lows"}

    # Market cap: band only when apply_mcap=True; no longer drop on missing
    mcap_cr = _get_market_cap_cr(yf, ticker, last_close=last_close)
    if apply_mcap:
        if mcap_cr is None:
            return {"_drop": "no_mcap"}
        if not (mcap_min <= mcap_cr <= mcap_max):
            return {"_drop": "mcap_%.0f" % mcap_cr}

    # Period highs / pct down (only keep rows within band)
    periods = {}
    for months, label in PERIODS:
        cutoff = pd.Timestamp(_months_ago(months))
        window_high = highs[highs.index >= cutoff]
        if window_high.empty:
            periods[label] = None
            continue
        hi = float(window_high.max())
        hi_date = window_high.idxmax().date()
        if hi <= 0:
            periods[label] = None
            continue
        pct = (last_close - hi) / hi * 100.0
        if not (-max_pct <= pct <= -min_pct):
            periods[label] = None
            continue
        periods[label] = {
            "Symbol": symbol,
            "Name": name,
            "Yahoo": ticker,
            "Mcap (Cr)": round(mcap_cr, 1) if mcap_cr is not None else None,
            "1Y %": round(pct_1y, 2) if pct_1y is not None else None,
            "Last Close": round(last_close, 2),
            "Last Date": last_date,
            "%s High" % label: round(hi, 2),
            "%s High Date" % label: hi_date,
            "Pct From High": round(pct, 2),
            "HL Count": hl_count,
            "HL Base": hl_base,
            "HL Values": ", ".join(str(v) for v in higher_lows),
        }
    return {"_drop": None, "periods": periods}


# --- per-universe screen ----------------------------------------------------

def screen_universe(yf, name, tickers, fno_set, mcap_min, mcap_max,
                    max_1y_runup, min_pct, max_pct, workers,
                    apply_fno=True, apply_mcap=True,
                    bse_symbol_map=None, max_retries=3,
                    index_ret_3m=None):
    n_initial = len(tickers)
    print("\n--- %s -------------------------------" % name)
    print("  Initial universe       : %d" % n_initial)

    bse_symbol_map = bse_symbol_map or {}

    def _fallback_for(yahoo_ticker, sym):
        # NSE symbol -> .BO via BSE scrip_id lookup. Only meaningful for
        # .NS tickers; .BO tickers have no useful fallback.
        if not yahoo_ticker.endswith(".NS"):
            return None
        return bse_symbol_map.get(sym.upper())

    # F&O removal (optional)
    if apply_fno and fno_set:
        kept = [t for t in tickers if t[1].upper() not in fno_set]
        print("  After F&O removal      : %d  (-%d)" %
              (len(kept), n_initial - len(kept)))
        tickers = kept
    elif not apply_fno:
        print("  F&O filter             : skipped")

    start_date = _months_ago(13)  # need >= 52 weeks for the early filter
    period_rows = {label: [] for _, label in PERIODS}
    counts = {"pass": 0, "runup": 0, "mcap_drop": 0, "no_mcap": 0,
              "band52w": 0, "price": 0, "below_200dma": 0,
              "no_higher_lows": 0, "short_history": 0, "errors": 0,
              "low52w": 0, "rs": 0, "dd": 0}
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_analyze_one, yf, t, s, nm, start_date,
                      mcap_min, mcap_max, max_1y_runup,
                      min_pct, max_pct, apply_mcap,
                      _fallback_for(t, s), max_retries,
                      index_ret_3m): (t, s, nm)
            for (t, s, nm) in tickers
        }
        for fut in as_completed(futs):
            done += 1
            if done % 200 == 0 or done == len(futs):
                print("    %d/%d (%.1fs)"
                      % (done, len(futs), time.time() - t0))
            try:
                res = fut.result()
            except Exception:
                counts["errors"] += 1
                continue
            if not res:
                counts["errors"] += 1
                continue
            drop = res.get("_drop")
            if drop is None:
                counts["pass"] += 1
                for label, row in (res.get("periods") or {}).items():
                    if row:
                        period_rows[label].append(row)
            elif drop.startswith("band52w"):
                counts["band52w"] += 1
            elif drop.startswith("runup_"):
                counts["runup"] += 1
            elif drop.startswith("price_"):
                counts["price"] += 1
            elif drop == "below_200dma":
                counts["below_200dma"] += 1
            elif drop == "no_higher_lows":
                counts["no_higher_lows"] += 1
            elif drop.startswith("low52w"):
                counts["low52w"] += 1
            elif drop.startswith("rs_"):
                counts["rs"] += 1
            elif drop.startswith("dd_") or drop == "no_dd_high":
                counts["dd"] += 1
            elif drop in ("short_history", "short_lows"):
                counts["short_history"] += 1
            elif drop == "no_mcap":
                counts["no_mcap"] += 1
            elif drop.startswith("mcap_"):
                counts["mcap_drop"] += 1
            else:
                counts["errors"] += 1

    print("  After 52W band %g-%g%%   : -%d dropped"
          % (min_pct, max_pct, counts["band52w"]))
    print("  After 52W low buf >%g%% : -%d dropped"
          % (LOW52W_BUFFER_PCT, counts["low52w"]))
    print("  DD filter              : disabled (-%d would have dropped)"
          % counts["dd"])
    print("  After RS vs NIFTY500   : -%d dropped (idx %dS=%s)"
          % (counts["rs"], RS_SESSIONS,
             ("%.2f%%" % index_ret_3m) if index_ret_3m is not None
             else "n/a"))
    print("  After 1Y runup >%g%%   : -%d dropped"
          % (MAX_1Y_RUNUP_PCT, counts["runup"]))
    print("  After last close >=%g  : -%d dropped"
          % (MIN_LAST_CLOSE, counts["price"]))
    print("  After above 200-DMA    : -%d dropped (-%d short history)"
          % (counts["below_200dma"], counts["short_history"]))
    print("  After higher-lows test : -%d dropped"
          % counts["no_higher_lows"])
    if apply_mcap:
        print("  After mcap %d-%d Cr  : %d kept  (-%d out of band, "
              "-%d no-mcap)" % (mcap_min, mcap_max, counts["pass"],
                                counts["mcap_drop"], counts["no_mcap"]))
    else:
        print("  Mcap band             : skipped")
        print("  No-mcap data          : %d (kept anyway)" % counts["no_mcap"])
    print("  Errors / no-data       : %d" % counts["errors"])

    # Build per-period DataFrames
    period_dfs = {}
    period_syms = {}
    for _, label in PERIODS:
        rows = period_rows[label]
        if not rows:
            period_dfs[label] = pd.DataFrame()
            period_syms[label] = set()
        else:
            df = pd.DataFrame(rows).sort_values("Pct From High")
            period_dfs[label] = df
            period_syms[label] = set(df["Symbol"].tolist())
        print("  %s hits (down %g-%g%%)  : %d"
              % (label, min_pct, max_pct, len(period_dfs[label])))

    return period_dfs, period_syms


# --- common-set sheet builders ----------------------------------------------

def _build_common(period_dfs, period_syms, labels):
    """Return DataFrame of stocks present in ALL given period sets,
    with Pct-from-High columns for each requested period."""
    if not all(lbl in period_syms for lbl in labels):
        return pd.DataFrame()
    common = set.intersection(*[period_syms[lbl] for lbl in labels])
    if not common:
        return pd.DataFrame()

    # Use first label's DF as base; merge pct cols from others
    base_label = labels[0]
    base = period_dfs[base_label]
    base = base[base["Symbol"].isin(common)].copy()
    base = base.rename(columns={
        "Pct From High": "Pct %s" % base_label,
        "%s High" % base_label: "%s High" % base_label,
        "%s High Date" % base_label: "%s High Date" % base_label,
    })
    keep_base = ["Symbol", "Name", "Yahoo", "Mcap (Cr)", "1Y %",
                 "Last Close", "Last Date",
                 "%s High" % base_label, "%s High Date" % base_label,
                 "Pct %s" % base_label]
    base = base[[c for c in keep_base if c in base.columns]]

    for lbl in labels[1:]:
        df = period_dfs[lbl]
        df = df[df["Symbol"].isin(common)][
            ["Symbol", "%s High" % lbl, "%s High Date" % lbl,
             "Pct From High"]
        ].rename(columns={"Pct From High": "Pct %s" % lbl})
        base = base.merge(df, on="Symbol", how="left")

    pct_cols = ["Pct %s" % lbl for lbl in labels]
    base["Worst Pct"] = base[pct_cols].min(axis=1)
    base = base.sort_values("Worst Pct").drop(columns=["Worst Pct"])
    return base


# --- main runner ------------------------------------------------------------

def run(out_dir, skip, min_pct, max_pct, max_symbols, workers,
        output_prefix):
    if min_pct < 0 or max_pct <= min_pct:
        sys.exit("Invalid --min/--max range.")

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Requires: pip install yfinance pandas openpyxl")

    _ = out_dir  # not used for input anymore; kept for output path

    universes = []
    if "nse" not in skip:
        try:
            universes.append(("NSE", fetch_nse_equity_universe()))
        except Exception as e:
            print("  FAIL fetch NSE: %s" % e)
    if "nse_sme" not in skip:
        try:
            universes.append(("NSE_SME", fetch_nse_sme_universe()))
        except Exception as e:
            print("  FAIL fetch NSE_SME: %s" % e)
    if "bse_sme" not in skip:
        try:
            universes.append(("BSE_SME", fetch_bse_sme_universe()))
        except Exception as e:
            print("  FAIL fetch BSE_SME: %s" % e)
    if max_symbols > 0:
        universes = [(n, t[:max_symbols]) for n, t in universes]

    print("=" * 72)
    print("  MULTI-UNIVERSE PCT-DOWN SCREENER")
    print("  Band: %.1f%% - %.1f%% from high  |  Drop 1Y runup > %.0f%%"
          % (min_pct, max_pct, MAX_1Y_RUNUP_PCT))
    print("  Mcap band (when applied): %d - %d Cr"
          % (MCAP_MIN_CR, MCAP_MAX_CR))
    print("=" * 72)
    print_filter_matrix()
    print("=" * 72)

    print("-> Loading F&O underlyings list ...")
    fno_set = load_fno_symbols()
    print("   F&O symbols: %d" % len(fno_set))

    # NIFTY 500 return over RS_SESSIONS trading sessions for RS filter.
    index_ret_3m = None
    try:
        print("-> Fetching NIFTY 500 (%s) for RS baseline (%d sessions) ..."
              % (RS_INDEX_TICKER, RS_SESSIONS))
        idx_df = yf.download(RS_INDEX_TICKER,
                             start=_months_ago(4).isoformat(),
                             progress=False, auto_adjust=False,
                             threads=False)
        if idx_df is not None and not idx_df.empty:
            if isinstance(idx_df.columns, pd.MultiIndex):
                idx_df.columns = idx_df.columns.get_level_values(0)
            idx_closes = idx_df["Close"].dropna()
            if len(idx_closes) >= RS_SESSIONS:
                base = float(idx_closes.iloc[-RS_SESSIONS])
                last = float(idx_closes.iloc[-1])
                if base > 0:
                    index_ret_3m = (last - base) / base * 100.0
                    print("   NIFTY 500 %dS return: %.2f%%"
                          % (RS_SESSIONS, index_ret_3m))
    except Exception as _e:
        print("   WARN  Could not fetch NIFTY 500 (%s); RS filter off."
              % _e)

    # Build NSE-symbol -> .BO fallback map once (covers NSE + NSE_SME).
    bse_symbol_map = fetch_bse_full_symbol_map()

    # Pre-warm the Angel One session BEFORE spawning any worker threads.
    # Without this, every worker thread independently calls
    # SmartConnect.generateSession() at startup, and Angel rate-limits
    # the auth endpoint ("Access denied because of exceeding access
    # rate"), causing all parallel logins to fail.
    try:
        from angel_client import _load_env, _get_credentials, \
            _ensure_session, _load_scrip_master  # noqa: F401
        _load_env()
        if _get_credentials():
            print("-> Pre-warming Angel One session ...")
            _ensure_session()
            _load_scrip_master()
            print("   Angel session ready (single-threaded login).")
        else:
            print("-> Angel One credentials missing; using "
                  "jugaad/yfinance only.")
    except Exception as _e:
        print("-> Angel pre-warm skipped (%s); will rely on fallbacks." % _e)

    all_sheets = {}  # ordered
    for uni_name, tickers in universes:
        cfg = FILTER_MATRIX.get(
            uni_name,
            {"apply_fno": True, "apply_mcap": True, "max_retries": 3})
        try:
            period_dfs, period_syms = screen_universe(
                yf, uni_name, tickers, fno_set,
                MCAP_MIN_CR, MCAP_MAX_CR, MAX_1Y_RUNUP_PCT,
                min_pct, max_pct, workers,
                apply_fno=cfg["apply_fno"],
                apply_mcap=cfg["apply_mcap"],
                bse_symbol_map=bse_symbol_map,
                max_retries=cfg.get("max_retries", 3),
                index_ret_3m=index_ret_3m,
            )
        except Exception as e:
            print("  FAIL %s: %s" % (uni_name, e))
            continue

        for _, label in PERIODS:
            sheet = ("%s %s" % (uni_name, label))[:31]
            all_sheets[sheet] = period_dfs.get(label, pd.DataFrame())

        # common3691 = _build_common(period_dfs, period_syms,
        #                            ["3M", "6M", "12M"])
        # all_sheets[("%s Common 3M+6M+12M" % uni_name)[:31]] = common3691
        # print("  Common 3M+6M+12M       : %d" % len(common3691))

    prefix = output_prefix or os.path.join(out_dir, "multi_pct_down")
    out_xlsx = "%s.xlsx" % prefix

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        wrote = 0
        for sheet, df in all_sheets.items():
            if df is None or df.empty:
                pd.DataFrame({"Note": ["No matches"]}).to_excel(
                    w, sheet_name=sheet, index=False)
            else:
                df.to_excel(w, sheet_name=sheet, index=False)
                wrote += 1

    # --- TradingView watchlist TXT (one combined, deduplicated list) ------
    # Format: one symbol per line as EXCHANGE:SYMBOL
    #   .NS tickers -> NSE:SYMBOL
    #   .BO tickers -> BSE:SCRIPCODE
    tv_symbols = []
    seen = set()
    for _sheet, df in all_sheets.items():
        if df is None or df.empty or "Yahoo" not in df.columns:
            continue
        for yticker in df["Yahoo"].dropna().unique():
            if yticker in seen:
                continue
            seen.add(yticker)
            if yticker.endswith(".NS"):
                tv_symbols.append("NSE:%s" % yticker[:-3])
            elif yticker.endswith(".BO"):
                tv_symbols.append("BSE:%s" % yticker[:-3])
            else:
                tv_symbols.append(yticker)

    out_txt = "%s.txt" % prefix
    with open(out_txt, "w") as f:
        f.write("\n".join(sorted(tv_symbols)))
        if tv_symbols:
            f.write("\n")

    print("\n" + "=" * 72)
    print("  Written: %s  (%d sheets, %d with hits)"
          % (out_xlsx, len(all_sheets), wrote))
    print("  Written: %s  (%d unique symbols, TradingView format)"
          % (out_txt, len(tv_symbols)))
    print("=" * 72)
    return out_xlsx


def main():
    ap = argparse.ArgumentParser(
        description="Multi-universe pct-down screener (NSE / NSE-SME / "
                    "BSE-SME).")
    ap.add_argument("--out", default=SCRIPT_DIR,
                    help="Output directory (default: script dir)")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["nse", "nse_sme", "bse_sme"],
                    help="Universes to skip")
    ap.add_argument("--min", dest="min_pct", type=float, default=2.0,
                    help="Min %% down from high (default: 2)")
    ap.add_argument("--max", dest="max_pct", type=float, default=21.0,
                    help="Max %% down from high (default: 21)")
    ap.add_argument("--max-symbols", type=int, default=0,
                    help="Cap symbols per universe (0 = no cap)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="Parallel download threads (default: %d)" %
                    DEFAULT_WORKERS)
    ap.add_argument("-o", "--output-prefix", default=None,
                    help="Output Excel prefix "
                         "(default: multi_pct_down_<date>)")
    args = ap.parse_args()

    run(
        out_dir=args.out,
        skip=set(args.skip),
        min_pct=args.min_pct,
        max_pct=args.max_pct,
        max_symbols=args.max_symbols,
        workers=args.workers,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()
