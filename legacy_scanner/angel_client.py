"""
angel_client.py — yfinance-shaped adapter for Angel One SmartAPI (SDK)
=====================================================================

Free Indian-market historical OHLCV via the official smartapi-python SDK.

Public surface:
  - angel_download(ticker, start, end, interval="1d") -> pd.DataFrame
        DataFrame[Open,High,Low,Close,Volume] indexed by Timestamp.
  - angel_download_many(tickers, start, end, max_workers=2) -> dict
        Bulk fetch, rate-limit safe (~2 req/sec).
  - get_angel_session() -> (api_key, jwt_token)  (lazy, auto-relogin)
  - refresh_token(force=False) -> bool

.env keys required:
  ANGEL_API_KEY=...
  ANGEL_CLIENT_CODE=...
  ANGEL_PIN=...
  ANGEL_TOTP_SECRET=...

References:
  - https://smartapi.angelbroking.com/docs/Historical
  - https://smartapi.angelbroking.com/docs/User
  - pip install smartapi-python pyotp
"""

import os
import sys
import json
import time
import threading
import datetime
import warnings
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, ".env")

SCRIP_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/"
    "OpenAPIScripMaster.json"
)
SCRIP_MASTER_CACHE = os.path.join(SCRIPT_DIR, ".angel_scrip_master.json")
SCRIP_MASTER_TTL_DAYS = 7

RATE_LIMIT_PER_SEC = 2
_rate_lock = threading.Lock()
_last_call_ts = [0.0] * RATE_LIMIT_PER_SEC

_smart_api = None   # SmartConnect instance (None until logged in)
_api_key_cache = ""  # cached for get_angel_session() return value
_refresh_token_cache = None  # stored from generateSession for renewAccessToken
_master_df: Optional[pd.DataFrame] = None
_symbol_index: Optional[dict] = None  # (exch, symbol_upper) -> token


# ─────────────────────────── env / credentials ─────────────────────────────

def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH, override=True)
        return
    except ImportError:
        pass
    if not os.path.exists(ENV_PATH):
        return
    with open(ENV_PATH) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def _get_credentials():
    keys = {}
    for k in ("ANGEL_API_KEY", "ANGEL_CLIENT_CODE",
              "ANGEL_PIN", "ANGEL_TOTP_SECRET"):
        v = os.environ.get(k, "").strip()
        if not v or v.startswith("your_"):
            return None
        keys[k] = v
    return keys


# ─────────────────────────── SDK login ─────────────────────────────────────

def _sdk_login(creds: dict):
    """Login via official SmartConnect SDK. Returns (SmartConnect, api_key, refresh_token)."""
    try:
        import pyotp
    except ImportError as e:
        raise RuntimeError(
            "pyotp not installed. Run: python3 -m pip install pyotp") from e
    try:
        from SmartApi import SmartConnect
    except ImportError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            "Failed to import SmartApi (missing module: %s). "
            "Run: python3 -m pip install smartapi-python logzero websocket-client"
            % missing) from e

    api_key = creds["ANGEL_API_KEY"]
    obj = SmartConnect(api_key=api_key)
    totp = pyotp.TOTP(creds["ANGEL_TOTP_SECRET"]).now()
    data = obj.generateSession(creds["ANGEL_CLIENT_CODE"],
                               creds["ANGEL_PIN"], totp)
    if not data or not data.get("status"):
        msg = data.get("message", data) if data else "No response"
        raise RuntimeError("Login failed: %s" % msg)
    rt = (data.get("data") or {}).get("refreshToken")
    return obj, api_key, rt


def _try_refresh_access_token() -> bool:
    """Renew JWT via refresh token (no TOTP needed). Returns True on success."""
    global _smart_api, _refresh_token_cache
    if _smart_api is None or not _refresh_token_cache:
        return False
    try:
        new_data = _smart_api.renewAccessToken({
            "refreshToken": _refresh_token_cache,
        })
        if new_data and new_data.get("status"):
            jwt = (new_data.get("data") or {}).get("jwtToken")
            if jwt:
                _smart_api.setAccessToken(jwt)
                _refresh_token_cache = (
                    (new_data.get("data") or {}).get("refreshToken")
                    or _refresh_token_cache
                )
                print("Angel token refreshed (no TOTP).")
                return True
    except Exception as e:
        print("Token refresh failed: %s" % e)
    return False


def refresh_token(force: bool = False) -> bool:
    """Re-establish session. Tries renewAccessToken first, then full TOTP login."""
    global _smart_api, _api_key_cache, _refresh_token_cache
    # Fast path: renew with refresh token (no TOTP)
    if _try_refresh_access_token():
        return True
    # Slow path: full TOTP re-login
    _load_env()
    creds = _get_credentials()
    if not creds:
        if force:
            print("\n" + "=" * 70)
            print("  ANGEL ONE CREDENTIALS MISSING")
            print("=" * 70)
            print("  Required keys in %s :" % ENV_PATH)
            print("    ANGEL_API_KEY=...")
            print("    ANGEL_CLIENT_CODE=...   (e.g. R12345)")
            print("    ANGEL_PIN=...           (4-digit MPIN)")
            print("    ANGEL_TOTP_SECRET=...   (base32 from TOTP setup)")
            print("=" * 70)
            try:
                input("  Press Enter when .env is ready... ")
            except (KeyboardInterrupt, EOFError):
                return False
            _load_env()
            creds = _get_credentials()
        if not creds:
            return False
    try:
        obj, api_key, rt = _sdk_login(creds)
    except Exception as e:
        print("Angel login failed: %s" % e)
        return False
    _smart_api = obj
    _api_key_cache = api_key
    _refresh_token_cache = rt
    return True


def get_angel_session():
    """Return (api_key, jwt_token). Logs in lazily on first call."""
    global _smart_api
    if _smart_api is not None:
        return (_api_key_cache, _smart_api.access_token)
    if not refresh_token(force=True):
        raise RuntimeError("Angel One auth failed; check .env")
    return (_api_key_cache, _smart_api.access_token)


def _ensure_session():
    """Ensure SmartConnect is logged in. Returns the SmartConnect instance."""
    global _smart_api
    if _smart_api is not None:
        return _smart_api
    get_angel_session()
    return _smart_api


# ─────────────────────────── scrip master ──────────────────────────────────

def _master_is_fresh() -> bool:
    if not os.path.exists(SCRIP_MASTER_CACHE):
        return False
    age = time.time() - os.path.getmtime(SCRIP_MASTER_CACHE)
    return age < SCRIP_MASTER_TTL_DAYS * 86400


def _download_scrip_master():
    print("-> Downloading Angel One scrip master (~25 MB, weekly)...")
    req = urllib.request.Request(
        SCRIP_MASTER_URL, headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as r, \
            open(SCRIP_MASTER_CACHE, "wb") as f:
        f.write(r.read())
    print("   Saved to %s" % SCRIP_MASTER_CACHE)


def _load_scrip_master() -> pd.DataFrame:
    global _master_df, _symbol_index
    if _master_df is not None:
        return _master_df
    if not _master_is_fresh():
        _download_scrip_master()
    with open(SCRIP_MASTER_CACHE) as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    if "instrumenttype" in df.columns:
        df = df[df["instrumenttype"].astype(str).isin(["", "AMXIDX"])
                | df["instrumenttype"].isna()]
    keep = [c for c in ("token", "symbol", "name", "exch_seg", "lotsize")
            if c in df.columns]
    df = df[keep].copy()
    _master_df = df.reset_index(drop=True)
    idx = {}
    for r in _master_df.itertuples(index=False):
        try:
            sym_full = str(r.symbol).strip().upper()
            exch = str(r.exch_seg).strip().upper()
            tok = str(r.token).strip()
            name = str(getattr(r, "name", "")).strip().upper()
            if not (sym_full and exch and tok):
                continue
            base = sym_full.split("-", 1)[0]
            idx.setdefault((exch, sym_full), tok)
            idx.setdefault((exch, base), tok)
            if name and name != base:
                idx.setdefault((exch, name), tok)
        except Exception:
            continue
    _symbol_index = idx
    print("   Indexed %d (exch, symbol) -> token pairs" % len(idx))
    return _master_df


# ─────────────────────────── ticker resolution ─────────────────────────────

INDEX_OVERRIDES = {
    "^NSEI":    ("NSE", "99926000", "Nifty 50"),
    "^NSEBANK": ("NSE", "99926009", "Nifty Bank"),
    "^BSESN":   ("BSE", "99919000", "Sensex"),
}


def _parse_ticker(ticker: str):
    if not ticker:
        return None, None
    if ticker in INDEX_OVERRIDES:
        ex, tok, _ = INDEX_OVERRIDES[ticker]
        return ex, tok
    t = ticker.strip()
    if ":" in t:
        prefix, raw = t.split(":", 1)
        prefix = prefix.upper()
        raw = raw.strip().upper()
        _load_scrip_master()
        if prefix == "BSE":
            if raw.isdigit():
                return "BSE", raw
            tok = (_symbol_index.get(("BSE", raw))
                   or _symbol_index.get(("BSE", raw + "-EQ")))
            return ("BSE", tok) if tok else (None, None)
        if prefix == "NSE":
            tok = (_symbol_index.get(("NSE", raw))
                   or _symbol_index.get(("NSE", raw + "-EQ")))
            return ("NSE", tok) if tok else (None, None)
        return None, None
    if t.upper().endswith(".BO"):
        raw = t[:-3].strip()
        if raw.isdigit():
            return "BSE", raw
        _load_scrip_master()
        tok = (_symbol_index.get(("BSE", raw.upper()))
               or _symbol_index.get(("BSE", raw.upper() + "-EQ")))
        return ("BSE", tok) if tok else (None, None)
    if t.upper().endswith(".NS"):
        raw = t[:-3].strip().upper()
        _load_scrip_master()
        tok = (_symbol_index.get(("NSE", raw + "-EQ"))
               or _symbol_index.get(("NSE", raw)))
        return ("NSE", tok) if tok else (None, None)
    _load_scrip_master()
    tok = (_symbol_index.get(("NSE", t.upper() + "-EQ"))
           or _symbol_index.get(("NSE", t.upper())))
    return ("NSE", tok) if tok else (None, None)


# ─────────────────────────── rate limiter ──────────────────────────────────

def _rate_limit_acquire():
    with _rate_lock:
        now = time.time()
        oldest = _last_call_ts[0]
        wait = (oldest + 1.0) - now
        if wait > 0:
            time.sleep(wait)
            now = time.time()
        _last_call_ts.pop(0)
        _last_call_ts.append(now)


# ─────────────────────────── public download API ───────────────────────────

def _to_date_str(d, with_time=True) -> str:
    if isinstance(d, str):
        if with_time and len(d) == 10:
            return d + " 09:15"
        return d
    if isinstance(d, (datetime.date, datetime.datetime)):
        if with_time:
            return d.strftime("%Y-%m-%d") + " 09:15"
        return d.strftime("%Y-%m-%d")
    return str(d)


def _empty_df():
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


_INTERVAL_MAP = {
    "1d":  "ONE_DAY",
    "1h":  "ONE_HOUR",
    "30m": "THIRTY_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "5m":  "FIVE_MINUTE",
    "1m":  "ONE_MINUTE",
}


def _is_auth_error_msg(msg: str) -> bool:
    """Check if an error message indicates an auth/session problem."""
    low = msg.lower()
    return any(s in low for s in (
        "ag8001", "ab1010", "invalid token", "expired",
        "session", "unauthor"
    ))


def angel_download(ticker: str,
                   start,
                   end=None,
                   interval: str = "1d",
                   retries: int = 2) -> pd.DataFrame:
    """Drop-in replacement for `yf.download(ticker, start, end)`.

    Returns DataFrame indexed by Timestamp with columns
    ['Open','High','Low','Close','Volume']. Empty on failure.
    Note: Angel daily candles cap at 2 000 days per request.
    """
    interval_const = _INTERVAL_MAP.get(interval)
    if interval_const is None:
        raise NotImplementedError("interval=%r not supported" % interval)
    end = end or datetime.date.today()
    fromdate = _to_date_str(start)
    todate = _to_date_str(end).replace("09:15", "15:30")

    exch, tok = _parse_ticker(ticker)
    if not tok:
        return _empty_df()

    historicParam = {
        "exchange":    exch,
        "symboltoken": tok,
        "interval":    interval_const,
        "fromdate":    fromdate,
        "todate":      todate,
    }

    for attempt in range(retries + 1):
        _rate_limit_acquire()
        try:
            obj = _ensure_session()
            resp = obj.getCandleData(historicParam)
        except Exception:
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            return _empty_df()

        if resp is None or not isinstance(resp, dict):
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            return _empty_df()

        if not resp.get("status"):
            err_code = str(resp.get("errorcode", "")).upper()
            err_msg = str(resp.get("message", ""))
            # Rate limit — back off and retry
            if err_code == "AB1004" and attempt < retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            # Auth error — refresh token first, then full re-login
            if _is_auth_error_msg(err_code + " " + err_msg) and attempt < retries:
                if _try_refresh_access_token() or refresh_token(force=False):
                    continue
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            return _empty_df()

        data = resp.get("data") or []
        if not data:
            return _empty_df()
        df = pd.DataFrame(
            data, columns=["Date", "Open", "High", "Low", "Close", "Volume"],
        )
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df.set_index("Date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df
    return _empty_df()


def angel_download_many(tickers,
                        start,
                        end=None,
                        max_workers: int = RATE_LIMIT_PER_SEC) -> dict:
    """Bulk fetch. Returns {ticker: DataFrame}, omitting empties."""
    out = {}
    if not tickers:
        return out
    _load_scrip_master()
    _ensure_session()
    print("  Angel bulk fetch: %d tickers (max_workers=%d, ~%.0fs minimum)"
          % (len(tickers), max_workers,
             len(tickers) / RATE_LIMIT_PER_SEC))
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(angel_download, t, start, end): t for t in tickers}
        for fut in as_completed(futs):
            t = futs[fut]
            done += 1
            try:
                df = fut.result()
            except Exception:
                df = _empty_df()
            if df is not None and not df.empty:
                out[t] = df
            if done % 50 == 0 or done == len(futs):
                print("    %d/%d (%.1fs, usable=%d)"
                      % (done, len(futs), time.time() - t0, len(out)))
    return out


# ─────────────────────────── self-test ─────────────────────────────────────

def _selftest():
    print("Angel One client self-test (SDK)")
    print("--------------------------------")
    _load_env()
    creds = _get_credentials()
    print("Credentials present : %s"
          % ("yes" if creds else "NO (fill .env)"))
    if not creds:
        return 1
    try:
        api_key, jwt = get_angel_session()
        print("Login (TOTP)        : OK (jwt len=%d)" % len(jwt))
    except Exception as e:
        print("Login FAILED        : %s" % e)
        return 2
    for t in ("RELIANCE.NS", "TCS.NS", "500325.BO", "^NSEI"):
        ex, tok = _parse_ticker(t)
        print("  resolve %-14s -> %s / %s" % (t, ex, tok))
    end = datetime.date.today()
    start = end - datetime.timedelta(days=40)
    df = angel_download("RELIANCE.NS", start, end)
    print("RELIANCE.NS rows    : %d" % len(df))
    if not df.empty:
        print(df.tail(3))
    return 0 if not df.empty else 4


if __name__ == "__main__":
    sys.exit(_selftest())
