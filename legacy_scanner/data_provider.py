"""
data_provider.py — unified historical OHLCV provider with fallback chain
========================================================================

Single entry point that drop-in replaces `yfinance.download(...)` everywhere
in the workspace, while transparently using Angel One SmartAPI as the
primary source.

Fallback order (per ticker):
    1. Angel One SmartAPI  (free, complete coverage of NSE/BSE incl. SME)
    2. jugaad-data          (NSE only, scrapes nseindia.com)
    3. yfinance             (broad coverage, sometimes flaky)

API:
    download(tickers, start=None, end=None, period=None,
             interval="1d", progress=False, threads=False, **_) -> pd.DataFrame

    - tickers : str  -> flat DataFrame[Open,High,Low,Close,Volume]
    - tickers : list -> MultiIndex DataFrame [(field, ticker)]  (matches
                        yfinance's default group_by="column" shape so
                        `raw["Close"]` and `raw[ticker]` both work)

    All other yfinance kwargs are accepted and ignored (for compatibility).

The fallback chain is short-circuited on first success. Empty / partial
results from a higher-priority source still cause fallback to the next.
"""

import datetime
import warnings
from typing import Optional

import pandas as pd

warnings.filterwarnings("ignore")

_PERIOD_DAYS = {
    "1d": 1, "5d": 5, "1mo": 31, "3mo": 92, "6mo": 183,
    "1y": 366, "2y": 731, "5y": 1827, "10y": 3653,
    "ytd": None, "max": 7300,
}

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# Set True to test Angel One end-to-end with NO fallback to jugaad/yfinance
ANGEL_ONLY = False


# ────────────────────── Angel-availability gate ────────────────────────────
# Skip Angel entirely (no scrip-master download, no input() prompt) when
# the user hasn't configured ANGEL_* creds in .env. This keeps the existing
# scripts working out-of-the-box even before the user opts into Angel.

_angel_available_cache = None  # tri-state: None=unchecked, True/False after


def _angel_available() -> bool:
    global _angel_available_cache
    if _angel_available_cache is not None:
        return _angel_available_cache
    try:
        from angel_client import _load_env, _get_credentials
        _load_env()
        _angel_available_cache = _get_credentials() is not None
    except Exception:
        _angel_available_cache = False
    return _angel_available_cache


# ────────────────────── normalised single-ticker fetchers ──────────────────

def _try_angel(ticker: str, start, end) -> Optional[pd.DataFrame]:
    if not _angel_available():
        return None
    try:
        from angel_client import angel_download
        df = angel_download(ticker, start, end)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    return df.reindex(columns=_OHLCV)


def _try_jugaad(ticker: str, start, end) -> Optional[pd.DataFrame]:
    """jugaad-data only handles NSE equity. Skip silently otherwise."""
    if not ticker.upper().endswith(".NS"):
        return None
    sym = ticker[:-3]
    if sym.startswith("^"):  # indices not supported by jugaad stock_df
        return None
    try:
        from jugaad_data.nse import stock_df
        s = pd.Timestamp(start).date()
        e = pd.Timestamp(end).date() if end else datetime.date.today()
        df = stock_df(symbol=sym, from_date=s, to_date=e, series="EQ")
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = df.rename(columns={
        "OPEN": "Open", "HIGH": "High", "LOW": "Low",
        "CLOSE": "Close", "VOLUME": "Volume", "DATE": "Date",
    })
    if "Date" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.reindex(columns=_OHLCV)


def _try_yfinance(ticker: str, start, end) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        df = yf.download(
            ticker, start=str(start),
            end=str(end) if end else None,
            progress=False, auto_adjust=False, threads=False,
        )
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.reindex(columns=_OHLCV)


def _resolve_period(start, end, period):
    """Translate yf-style (start, end, period) into concrete date objects."""
    end = end or datetime.date.today()
    if isinstance(end, str):
        end = pd.Timestamp(end).date()
    if start is None and period:
        days = _PERIOD_DAYS.get(period.lower(), 366)
        if period.lower() == "ytd":
            start = datetime.date(end.year, 1, 1)
        else:
            start = end - datetime.timedelta(days=days)
    if start is None:
        start = end - datetime.timedelta(days=366)
    if isinstance(start, str):
        start = pd.Timestamp(start).date()
    return start, end


def _fetch_one(ticker: str, start, end) -> pd.DataFrame:
    """Run the fallback chain for a single ticker. Always returns a
    DataFrame (possibly empty) with OHLCV columns and a date-normalised
    DatetimeIndex (midnight UTC) so different providers align cleanly."""
    chain = (_try_angel,) if ANGEL_ONLY else (_try_angel, _try_jugaad, _try_yfinance)
    for fn in chain:
        df = fn(ticker, start, end)
        if df is not None and not df.empty:
            return _normalise_index(df)
    return pd.DataFrame(columns=_OHLCV)


def _normalise_index(df: pd.DataFrame) -> pd.DataFrame:
    """Strip tz + time-of-day so jugaad (18:30 UTC) and yf (00:00 UTC)
    rows align on the same calendar dates."""
    if df is None or df.empty:
        return df
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ────────────────────── public API ─────────────────────────────────────────

def download(tickers, start=None, end=None, period=None,
             interval: str = "1d", progress: bool = False,
             threads: bool = False, group_by: str = "column",
             auto_adjust: bool = False, **_) -> pd.DataFrame:
    """yfinance.download-shaped facade with Angel→jugaad→yf fallback.

    Returns:
      - flat DataFrame[Open,High,Low,Close,Volume] for a single ticker
      - MultiIndex DataFrame columns=(field, ticker) for a list/tuple of
        tickers (matches yfinance default).
    """
    if interval != "1d":
        # Defer non-daily to yfinance directly (Angel supports them but
        # the consumers in this repo only ever ask for daily).
        import yfinance as yf
        return yf.download(
            tickers, start=start, end=end, period=period,
            interval=interval, progress=progress, threads=threads,
            group_by=group_by, auto_adjust=auto_adjust,
        )

    start, end = _resolve_period(start, end, period)

    # Single ticker → flat frame
    if isinstance(tickers, str):
        return _fetch_one(tickers, start, end)

    # Multi: parallelise the Angel pass via angel_client, then patch holes
    # with jugaad/yfinance per remaining ticker. Skip Angel entirely if creds
    # are missing — keeps existing scripts working without .env setup.
    tlist = list(tickers)
    angel_results = {}
    if _angel_available():
        try:
            from angel_client import angel_download_many
            angel_results = angel_download_many(tlist, start, end)
        except Exception:
            angel_results = {}

    per_ticker = {}
    for t in tlist:
        df = angel_results.get(t)
        if not ANGEL_ONLY:
            if df is None or df.empty:
                df = _try_jugaad(t, start, end)
            if df is None or df.empty:
                df = _try_yfinance(t, start, end)
        if df is not None and not df.empty:
            per_ticker[t] = _normalise_index(df.reindex(columns=_OHLCV))

    if not per_ticker:
        return pd.DataFrame()

    # Assemble yf-shape MultiIndex (level 0 = field, level 1 = ticker)
    frames = []
    for t, df in per_ticker.items():
        df2 = df.copy()
        df2.columns = pd.MultiIndex.from_product([_OHLCV, [t]])
        frames.append(df2)
    combined = pd.concat(frames, axis=1).sort_index()
    # Order columns: all Opens, all Highs, ... (matches yfinance default)
    combined = combined.reindex(
        columns=pd.MultiIndex.from_product([_OHLCV, tlist]),
    )
    # Drop ticker columns where Angel/jugaad/yf all returned nothing
    combined = combined.dropna(axis=1, how="all")
    return combined


# ────────────────────── self-test ──────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("data_provider self-test")
    print("-----------------------")
    end = datetime.date.today()
    start = end - datetime.timedelta(days=20)
    df = download("RELIANCE.NS", start, end)
    print("single RELIANCE.NS rows:", len(df))
    if not df.empty:
        print(df.tail(2))
    multi = download(["RELIANCE.NS", "TCS.NS", "INFY.NS"], start, end)
    print("multi shape:", multi.shape, "cols:", list(multi.columns)[:6])
    sys.exit(0 if not df.empty else 1)
