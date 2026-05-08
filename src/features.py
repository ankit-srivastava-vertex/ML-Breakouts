"""Feature engineering for the meta-model.

Purpose:
  Convert one (df, asof, setup_dict, benchmark, sector_idx, fund_row)
  tuple into a flat dict of ~110 numeric features. This is the input
  row to the LightGBM meta-model both during training (one row per
  historical setup) and inference (one row per live setup).

How it works:
  * All features are computed using ONLY data with index <= `asof`
    (strict no-look-ahead). Bench / sector / fundamentals are sliced
    by caller before being passed in.
  * Features are grouped into independent helper functions per group;
    the public `make_features(...)` orchestrates them and merges the
    result into a single dict.

Feature groups (~110 total cols):
  A. Setup geometry          (base_days, touches, dist_pct, R, ...)
  B. Volume                  (VCR, VDU, pocket_pivot, OBV slopes, RVOL)
  C. Volatility              (BB width Z, ATR/Close, TTM squeeze)
  D. Momentum                (RSI14, ROC20/60/120, EMA distances)
  E. Position                (dist from 52w high/low, % of 52w range)
  F. Trend quality           (ADX14, EMA stack, higher-lows)
  G. Relative strength       (vs Nifty 1/3/6m, Mansfield RS, slope)
  H. Microstructure          (HL spread, dollar volume, Amihud)
  I. Regime                  (Nifty above 200dma, Nifty 20d slope)
  J. Calendar                (day-of-week, day-of-month)
  K. Sector RS               (sector_idx-relative momentum)
  L. Fundamentals            (fund_*_rank_sector, fund_log_mcap, ...)

Data sources:
  Pure-function module — takes pandas DataFrames + dicts as input.
  Upstream data origins:
    * df, bench, sector_idx — OHLCV cache (Angel One)
    * fund_row              — data/fundamentals.parquet (yfinance)

Outputs:
  dict[str, float]  — a single feature row keyed by column name.

How to run:
  Import-only.
      from src.features import make_features
      f = make_features(df, setup, bench, sector_idx, fund_row)

Notes:
  * The exact list of feature names produced here MUST match
    `data/models/features.json` for inference to succeed; that JSON
    is written by `scripts/03_train_model.py`.
  * Failures inside any feature group are swallowed and produce NaN
    so a single bad symbol never aborts a daily scan.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ─── small helpers ──────────────────────────────────────────────────────────

def _safe_div(a, b, default=np.nan):
    try:
        if b == 0 or pd.isna(b):
            return default
        return a / b
    except Exception:
        return default


def _slope(y: pd.Series) -> float:
    yy = y.dropna().values
    if len(yy) < 5:
        return 0.0
    xx = np.arange(len(yy))
    return float(np.polyfit(xx, yy, 1)[0])


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di = 100 * plus_dm.rolling(n).mean() / atr
    minus_di = 100 * minus_dm.rolling(n).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(n).mean()


def _obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["Close"].diff().fillna(0))
    return (sign * df["Volume"]).cumsum()


def _pocket_pivot(df: pd.DataFrame, R: float) -> int:
    """1 if last bar is a pocket-pivot up day near resistance, else 0."""
    if len(df) < 12:
        return 0
    last = df.iloc[-1]
    prior10 = df.iloc[-11:-1]
    down = prior10[prior10["Close"] < prior10["Close"].shift(1)]
    if down.empty:
        return 0
    if last["Volume"] <= float(down["Volume"].max()):
        return 0
    if last["Close"] <= last["Open"]:
        return 0
    rng = last["High"] - last["Low"]
    if rng <= 0:
        return 0
    if (last["Close"] - last["Low"]) / rng < 0.5:
        return 0
    if abs(R - last["Close"]) / last["Close"] > 0.05:
        return 0
    return 1


def _ttm_squeeze(df: pd.DataFrame, n: int = 20) -> int:
    if len(df) < n + 5:
        return 0
    c = df["Close"]
    ma = c.rolling(n).mean()
    sd = c.rolling(n).std()
    bb_up, bb_dn = ma + 2.0 * sd, ma - 2.0 * sd
    a = _atr(df, n)
    kc_up, kc_dn = ma + 1.5 * a, ma - 1.5 * a
    return int(bb_up.iloc[-1] < kc_up.iloc[-1]
               and bb_dn.iloc[-1] > kc_dn.iloc[-1])


def _ttm_squeeze_series(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Vector squeeze flag (1 = squeeze on) over the full series."""
    if len(df) < n + 5:
        return pd.Series(dtype=float, index=df.index)
    c = df["Close"]
    ma = c.rolling(n).mean()
    sd = c.rolling(n).std()
    bb_up, bb_dn = ma + 2.0 * sd, ma - 2.0 * sd
    a = _atr(df, n)
    kc_up, kc_dn = ma + 1.5 * a, ma - 1.5 * a
    return ((bb_up < kc_up) & (bb_dn > kc_dn)).astype(float)


def _wyckoff_spring(df: pd.DataFrame, base_start) -> int:
    """1 if a Wyckoff spring (false breakdown then reclaim) occurred in
    the last 15 bars of the base — a textbook accumulation pattern."""
    try:
        base = df.loc[base_start:]
    except Exception:
        return 0
    if len(base) < 20:
        return 0
    base_low = float(base["Low"].iloc[:-5].min())
    recent = base.iloc[-15:]
    wicks = recent[(recent["Low"] < base_low) & (recent["Close"] > base_low)]
    return int(not wicks.empty)


# ─── Fractional differentiation (Lopez de Prado, AFML Ch. 5) ───────────────

def _ffd_weights(d: float, thres: float = 1e-4, max_size: int = 200) -> np.ndarray:
    """Fixed-width fractional-differentiation weights, truncated when |w| < thres."""
    w = [1.0]
    k = 1
    while k < max_size:
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < thres:
            break
        w.append(wk)
        k += 1
    return np.array(w[::-1])


def _frac_diff_last(series: pd.Series, d: float = 0.4,
                    thres: float = 1e-4) -> float:
    """Last fractionally-differentiated value of `series`. Stationary
    transform that retains long-memory (unlike pct_change)."""
    s = series.dropna()
    if len(s) < 30:
        return np.nan
    w = _ffd_weights(d, thres)
    L = len(w)
    if len(s) < L:
        return np.nan
    window = s.iloc[-L:].values
    return float(np.dot(w, window))


# ─── Weekly resample helpers ────────────────────────────────────────────────

def _to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV → weekly (Friday close).
    Keeps standard column names. Drops weeks with no trades."""
    if len(df) < 10:
        return df.iloc[0:0]
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    w = df.resample("W-FRI").agg(agg).dropna(how="any")
    return w


def _coiled_spring_score(f: dict) -> float:
    """Port of legacy `compute_score` — composite 0-100 setup quality.
    Uses already-computed feature dict values. Direct port of the formula
    from legacy_scanner/breakout_scanner.py.
    """
    # A: Base quality (25 pts) — length × touches × higher-lows
    base_days = f.get("base_days", 0.0) or 0.0
    touches = f.get("touches", 0.0) or 0.0
    base_score = 25.0 * min(base_days / 120.0, 1.0)
    # touches multiplier: 2→0.7, 3→0.85, 4→1.0, 5+→1.05
    if touches >= 5: tm = 1.05
    elif touches >= 4: tm = 1.0
    elif touches >= 3: tm = 0.85
    elif touches >= 2: tm = 0.7
    else: tm = 0.5
    base_score *= tm
    if f.get("higher_lows", 0) > 0:
        base_score *= 1.15

    # B: Volatility contraction (10 pts) — vcr in [0, 0.30+]
    vcr = f.get("vcr", 0.0) or 0.0
    vcr_score = 10.0 * max(min(vcr / 0.30, 1.0), 0.0)

    # C: Volume dry-up (5 pts)
    vdu = f.get("vdu", 0.0) or 0.0
    vdu_score = 5.0 * max(min(vdu / 0.20, 1.0), 0.0)

    # D: Proximity (20 pts) — distance to R in [-3%, +8%]
    dist = (f.get("dist_pct", 0.0) or 0.0) / 100.0  # back to fraction
    if -0.03 <= dist <= 0.08:
        prox_score = 20.0 * max(0.0, 1.0 - abs(dist) / 0.08)
    else:
        prox_score = 0.0

    # E: Trend (15 pts) — distance vs EMA50/200
    trend_score = 0.0
    if (f.get("dist_ema50_pct", -1) or -1) > 0: trend_score += 5
    if (f.get("dist_ema200_pct", -1) or -1) > 0: trend_score += 5
    if (f.get("ema200_slope_20", -1) or -1) > 0: trend_score += 5

    # F: Mansfield RS (10 pts)
    rs_m = f.get("rs_mansfield", 0.0) or 0.0
    rs_slope = f.get("rs_slope_20", 0.0) or 0.0
    rs_score = (5.0 if rs_m > 0 else 0.0) + (5.0 if rs_slope > 0 else 0.0)

    # G: 52-week high (15 pts)
    pct_off = (f.get("dist_52w_high_pct", 100.0) or 100.0) / 100.0
    if pct_off <= 0.15:
        hi_score = 15.0 * (1.0 - pct_off / 0.15)
    else:
        hi_score = 0.0

    return float(base_score + vcr_score + vdu_score
                 + prox_score + trend_score + rs_score + hi_score)


# ─── main feature extractor ─────────────────────────────────────────────────

def make_features(df: pd.DataFrame, setup: dict,
                  bench: pd.Series | None = None,
                  sector_idx: pd.Series | None = None,
                  fund_row: dict | None = None) -> dict:
    """Compute the feature dict for one (symbol, asof) setup row.

    df must be the symbol's OHLCV up to and including asof (no future data).
    setup is the output of setup_detector.detect_resistance(df).
    bench is the benchmark close series (Nifty 50), aligned to df.index
    or longer.
    sector_idx (optional): the symbol's sector equal-weight index series.
    fund_row (optional): dict of fundamentals + cross-sectional ranks
        for this symbol (sector, marketCap, PE, PB, ROE, growth, ranks…).

    Returns a flat dict — all numeric (NaN allowed; LightGBM handles them).
    """
    R = setup["R"]
    base_start = setup["base_start"]
    base = df.loc[base_start:]
    last_close = float(df["Close"].iloc[-1])
    f: dict = {}

    # ─── A. Setup geometry ────────────────────────────────────────────────
    f["base_days"] = float(setup["base_len_days"])
    f["touches"] = float(setup["touches"])
    f["dist_pct"] = float(setup["distance_pct"]) * 100
    f["is_52w_high"] = float(setup["is_52w_high"])
    base_low = float(base["Low"].min())
    f["base_height_pct"] = (R - base_low) / R * 100 if R > 0 else 0.0
    f["base_slope"] = _slope(base["Close"])

    # ─── A2. Chart-pattern microstructure (v2) ────────────────────────────
    # Base depth-to-width ratio: tight bases (low depth, long width) = good.
    bw = max(float(setup["base_len_days"]), 1.0)
    f["base_depth_width_ratio"] = f["base_height_pct"] / bw
    # # daily closes that touched within 1% of R during the base
    if R > 0 and len(base) > 0:
        closes_near_R = (base["Close"] >= R * 0.99) & (base["Close"] <= R * 1.01)
        f["close_touches_R"] = float(closes_near_R.sum())
        # # bars where High pierced R
        f["high_pierces_R"] = float((base["High"] > R).sum())
        # Tightness: stddev of Close in last 30 bars / Close
        last30 = df["Close"].iloc[-30:]
        f["base_tightness_30d"] = float(last30.std() / last30.mean() * 100) \
            if last30.mean() > 0 else np.nan
        # Drift: linear slope of last 30 closes (positive = rising into R)
        f["base_drift_slope_30d"] = _slope(last30) / last30.mean() * 100 \
            if last30.mean() > 0 else 0.0
    else:
        f["close_touches_R"] = 0.0
        f["high_pierces_R"] = 0.0
        f["base_tightness_30d"] = np.nan
        f["base_drift_slope_30d"] = 0.0
    # Days since last failed breakout: scan base for bars where High > R*1.005
    # but Close in next 5 bars < R (failed = pierced and rejected).
    days_since_fail = np.nan
    if R > 0 and len(base) >= 6:
        base_h = base["High"].values
        base_c = base["Close"].values
        last_idx = len(base) - 1
        for i in range(last_idx - 5, -1, -1):
            if base_h[i] > R * 1.005:
                # rejected if any close in next 5 bars < R*0.99
                if i + 5 <= last_idx:
                    fwd_c = base_c[i + 1:i + 6]
                    if (fwd_c < R * 0.99).any():
                        days_since_fail = float(last_idx - i)
                        break
    f["days_since_failed_bo"] = days_since_fail if not np.isnan(days_since_fail) else 999.0
    # Gap behavior at last bar
    if len(df) >= 2:
        prev_c = float(df["Close"].iloc[-2])
        today_o = float(df["Open"].iloc[-1])
        f["gap_open_pct"] = (today_o - prev_c) / prev_c * 100 if prev_c > 0 else 0.0
    else:
        f["gap_open_pct"] = 0.0
    # Inside-bar / NR7 flag
    if len(df) >= 8:
        last_range = float(df["High"].iloc[-1] - df["Low"].iloc[-1])
        prev7_max = float((df["High"].iloc[-8:-1] - df["Low"].iloc[-8:-1]).max())
        f["nr7"] = float(last_range < prev7_max)
        # Inside bar: today's H<prev H AND today's L>prev L
        f["inside_bar"] = float(
            df["High"].iloc[-1] < df["High"].iloc[-2]
            and df["Low"].iloc[-1] > df["Low"].iloc[-2])
    else:
        f["nr7"] = 0.0
        f["inside_bar"] = 0.0

    # ─── B. Volume ─────────────────────────────────────────────────────────
    a = _atr(df, 14)
    atr_now = float(a.iloc[-10:].mean())
    atr_then = float(a.loc[base_start:].iloc[:20].mean()) if len(base) >= 20 else atr_now
    f["vcr"] = 1.0 - _safe_div(atr_now, atr_then, 0.0)
    f["atr_ratio"] = _safe_div(atr_now, last_close, np.nan) * 100

    v50 = float(df["Volume"].rolling(50).mean().iloc[-1])
    v10 = float(df["Volume"].iloc[-10:].mean())
    v20 = float(df["Volume"].iloc[-20:].mean())
    f["vdu"] = 1.0 - _safe_div(v10, v50, 0.0)
    f["rvol_5"] = _safe_div(float(df["Volume"].iloc[-5:].mean()), v50, np.nan)
    f["rvol_20"] = _safe_div(v20, v50, np.nan)
    f["pocket_pivot"] = float(_pocket_pivot(df, R))

    obv_s = _obv(df)
    f["obv_slope_20"] = _slope(obv_s.tail(20))
    f["obv_slope_50"] = _slope(obv_s.tail(50))
    # Multi-window OBV for richer accumulation signal (port of legacy scanner)
    f["obv_slope_10"] = _slope(obv_s.tail(10))
    f["obv_slope_100"] = _slope(obv_s.tail(100)) if len(obv_s) >= 100 else 0.0
    # OBV acceleration: slope(20) - slope(50) (rising = recent accumulation)
    f["obv_accel"] = f["obv_slope_20"] - f["obv_slope_50"]
    # OBV vs price divergence: positive if OBV makes new high while price doesn't
    if len(obv_s) >= 50 and len(df) >= 50:
        obv_hi = float(obv_s.tail(50).max())
        obv_now = float(obv_s.iloc[-1])
        px_hi = float(df["Close"].tail(50).max())
        px_now = float(df["Close"].iloc[-1])
        f["obv_divergence"] = float(
            (obv_now >= obv_hi * 0.999) and (px_now < px_hi * 0.99))
    else:
        f["obv_divergence"] = 0.0

    # ─── C. Volatility ─────────────────────────────────────────────────────
    c = df["Close"]
    bb_ma = c.rolling(20).mean()
    bb_sd = c.rolling(20).std()
    bb_width = (4 * bb_sd / bb_ma).iloc[-100:]
    bb_width_now = float(bb_width.iloc[-1])
    f["bb_width"] = bb_width_now * 100
    f["bb_width_z"] = (bb_width_now - bb_width.mean()) / (bb_width.std() + 1e-9)
    f["ttm_squeeze"] = float(_ttm_squeeze(df))
    # Squeeze depth: how many of last 20 bars were in squeeze + bars since
    # last squeeze release (high values = explosive setup just released).
    sq_series = _ttm_squeeze_series(df)
    if len(sq_series) >= 20:
        last20 = sq_series.tail(20)
        f["squeeze_density_20"] = float(last20.mean())
        # bars_in_current_squeeze: trailing run of 1s at the end (capped 50)
        run = 0
        for v in sq_series.iloc[::-1]:
            if v == 1.0:
                run += 1
                if run >= 50:
                    break
            else:
                break
        f["bars_in_squeeze"] = float(run)
        # bars_since_release: trailing run of 0s after a squeeze (capped 30)
        rel = 0
        seen_squeeze = False
        for v in sq_series.iloc[::-1]:
            if v == 0.0 and not seen_squeeze:
                rel += 1
                if rel >= 30:
                    break
            elif v == 1.0:
                seen_squeeze = True
                break
            else:
                break
        f["bars_since_squeeze_release"] = float(rel) if seen_squeeze else 0.0
    else:
        f["squeeze_density_20"] = 0.0
        f["bars_in_squeeze"] = 0.0
        f["bars_since_squeeze_release"] = 0.0

    # Wyckoff spring (false breakdown then reclaim) — accumulation tell
    f["wyckoff_spring"] = float(_wyckoff_spring(df, base_start))

    # Fractional differentiation of close (stationary, retains memory)
    f["frac_diff_close_04"] = _frac_diff_last(c, d=0.4)
    f["frac_diff_close_06"] = _frac_diff_last(c, d=0.6)

    # ─── D. Momentum ──────────────────────────────────────────────────────
    f["rsi_14"] = float(_rsi(c, 14).iloc[-1])
    for k, lbl in [(20, "20"), (60, "60"), (120, "120")]:
        if len(c) > k:
            f[f"roc_{lbl}"] = (c.iloc[-1] / c.iloc[-k - 1] - 1) * 100
        else:
            f[f"roc_{lbl}"] = np.nan

    ema20 = c.ewm(span=20).mean().iloc[-1]
    ema50 = c.ewm(span=50).mean().iloc[-1]
    ema200 = c.ewm(span=200).mean().iloc[-1] if len(c) >= 200 else np.nan
    f["dist_ema20_pct"] = (last_close - ema20) / ema20 * 100
    f["dist_ema50_pct"] = (last_close - ema50) / ema50 * 100
    f["dist_ema200_pct"] = ((last_close - ema200) / ema200 * 100
                            if pd.notna(ema200) else np.nan)

    # ─── E. Position in 52w range ─────────────────────────────────────────
    hi52 = float(df["High"].tail(252).max()) if len(df) >= 252 else float(df["High"].max())
    lo52 = float(df["Low"].tail(252).min()) if len(df) >= 252 else float(df["Low"].min())
    f["dist_52w_high_pct"] = (hi52 - last_close) / hi52 * 100
    f["dist_52w_low_pct"] = (last_close - lo52) / lo52 * 100
    f["pct_of_52w_range"] = ((last_close - lo52) / (hi52 - lo52) * 100
                             if hi52 > lo52 else 50.0)

    # ─── F. Trend quality ─────────────────────────────────────────────────
    f["adx_14"] = float(_adx(df, 14).iloc[-1])
    ema20_s = c.ewm(span=20).mean()
    ema50_s = c.ewm(span=50).mean()
    f["ema_stack"] = float(
        last_close > ema20_s.iloc[-1] > ema50_s.iloc[-1])
    if len(c) >= 200:
        ema200_s = c.ewm(span=200).mean()
        f["ema_full_stack"] = float(
            last_close > ema20_s.iloc[-1] > ema50_s.iloc[-1] > ema200_s.iloc[-1])
        f["ema200_slope_20"] = _slope(ema200_s.tail(20))
    else:
        f["ema_full_stack"] = 0.0
        f["ema200_slope_20"] = 0.0

    # higher-lows in base
    if len(base) >= 30:
        lows_idx = base["Low"].rolling(11, center=True).min() == base["Low"]
        swing_lows = base["Low"][lows_idx].dropna()
        f["higher_lows"] = float(_slope(swing_lows) > 0
                                 if len(swing_lows) >= 3 else False)
    else:
        f["higher_lows"] = 0.0

    # ─── G. Relative strength vs Nifty ────────────────────────────────────
    if bench is not None and not bench.empty:
        b = bench.reindex(df.index).ffill()
        ratio = (df["Close"] / b).dropna()
        if len(ratio) >= 60:
            for k, lbl in [(20, "1m"), (60, "3m"), (120, "6m")]:
                if len(ratio) > k:
                    f[f"rs_{lbl}"] = (ratio.iloc[-1] / ratio.iloc[-k - 1] - 1) * 100
                else:
                    f[f"rs_{lbl}"] = np.nan
            sma52 = ratio.rolling(min(252, len(ratio))).mean()
            mansfield = (ratio / sma52 - 1.0) * 100
            f["rs_mansfield"] = float(mansfield.iloc[-1])
            f["rs_slope_20"] = _slope(mansfield.tail(20))
        else:
            for lbl in ("1m", "3m", "6m"):
                f[f"rs_{lbl}"] = np.nan
            f["rs_mansfield"] = np.nan
            f["rs_slope_20"] = np.nan
    else:
        for lbl in ("1m", "3m", "6m"):
            f[f"rs_{lbl}"] = np.nan
        f["rs_mansfield"] = np.nan
        f["rs_slope_20"] = np.nan

    # ─── H. Microstructure / liquidity ────────────────────────────────────
    f["hl_spread_pct"] = ((df["High"] - df["Low"]) / df["Close"]).iloc[-20:].mean() * 100
    dv = (df["Close"] * df["Volume"])
    f["dollar_vol_50d_cr"] = float(dv.rolling(50).mean().iloc[-1]) / 1e7
    ret = c.pct_change().abs()
    f["amihud_20"] = float(
        (ret.iloc[-20:] / dv.iloc[-20:].replace(0, np.nan)).mean() * 1e7)

    # ─── I. Regime (benchmark) ────────────────────────────────────────────
    # NOTE (v2): Removed bench_roc_60, bench_slope_20 as direct features.
    # In v1 they dominated importance — model learned "market is up" not
    # "stock is breaking out". bench_above_200dma kept ONLY as a regime
    # gate flag; everything else is computed RELATIVE to bench (rs_*) which
    # is regime-invariant.
    if bench is not None and not bench.empty:
        b_aligned = bench.reindex(df.index).ffill()
        if len(b_aligned.dropna()) >= 200:
            b200 = b_aligned.rolling(200).mean()
            f["bench_above_200dma"] = float(b_aligned.iloc[-1] > b200.iloc[-1])
        else:
            f["bench_above_200dma"] = np.nan
    else:
        f["bench_above_200dma"] = np.nan

    # ─── J. Calendar — REMOVED in v2 (was pure leakage / trend artifact) ─

    # ─── K. Sector relative strength ──────────────────────────────────────
    if sector_idx is not None and not sector_idx.empty:
        s_aligned = sector_idx.reindex(df.index).ffill()
        if len(s_aligned.dropna()) >= 60:
            sec_ratio = (df["Close"] / s_aligned).dropna()
            for k, lbl in [(20, "1m"), (60, "3m"), (120, "6m")]:
                if len(sec_ratio) > k:
                    f[f"sec_rs_{lbl}"] = (sec_ratio.iloc[-1] / sec_ratio.iloc[-k - 1] - 1) * 100
                else:
                    f[f"sec_rs_{lbl}"] = np.nan
            f["sec_rs_slope_20"] = _slope(sec_ratio.tail(20))
            # Sector itself vs benchmark (sector momentum)
            if bench is not None and not bench.empty:
                b_a = bench.reindex(df.index).ffill()
                if len(b_a.dropna()) >= 60:
                    sb = (s_aligned / b_a).dropna()
                    if len(sb) > 60:
                        f["sec_vs_bench_3m"] = (sb.iloc[-1] / sb.iloc[-61] - 1) * 100
                    else:
                        f["sec_vs_bench_3m"] = np.nan
                    f["sec_above_50dma"] = float(
                        s_aligned.iloc[-1] > s_aligned.rolling(50).mean().iloc[-1])
                else:
                    f["sec_vs_bench_3m"] = np.nan
                    f["sec_above_50dma"] = np.nan
            else:
                f["sec_vs_bench_3m"] = np.nan
                f["sec_above_50dma"] = np.nan
        else:
            for c in ("sec_rs_1m", "sec_rs_3m", "sec_rs_6m",
                      "sec_rs_slope_20", "sec_vs_bench_3m", "sec_above_50dma"):
                f[c] = np.nan
    else:
        for c in ("sec_rs_1m", "sec_rs_3m", "sec_rs_6m",
                  "sec_rs_slope_20", "sec_vs_bench_3m", "sec_above_50dma"):
            f[c] = np.nan

    # ─── K2. Sector-itself breakout (is the sector breaking out too?) ─────
    if sector_idx is not None and not sector_idx.empty:
        s_aligned = sector_idx.reindex(df.index).ffill().dropna()
        if len(s_aligned) >= 252:
            sec_hi52 = float(s_aligned.tail(252).max())
            sec_now = float(s_aligned.iloc[-1])
            f["sec_dist_52w_high_pct"] = (sec_hi52 - sec_now) / sec_hi52 * 100
            f["sec_at_52w_high"] = float(sec_now >= sec_hi52 * 0.98)
            sec_ma50 = float(s_aligned.rolling(50).mean().iloc[-1])
            f["sec_above_200dma"] = float(
                sec_now > float(s_aligned.rolling(200).mean().iloc[-1]))
            f["sec_breakout_strength"] = ((sec_now / sec_ma50 - 1.0) * 100
                                          if sec_ma50 > 0 else 0.0)
        else:
            for c in ("sec_dist_52w_high_pct", "sec_at_52w_high",
                      "sec_above_200dma", "sec_breakout_strength"):
                f[c] = np.nan
    else:
        for c in ("sec_dist_52w_high_pct", "sec_at_52w_high",
                  "sec_above_200dma", "sec_breakout_strength"):
            f[c] = np.nan

    # ─── M. Weekly timeframe (multi-TF confirmation) ──────────────────────
    wk = _to_weekly(df)
    if len(wk) >= 30:
        wc = wk["Close"]
        f["wk_rsi_14"] = float(_rsi(wc, 14).iloc[-1])
        f["wk_adx_14"] = float(_adx(wk, 14).iloc[-1])
        wema10 = wc.ewm(span=10).mean()
        wema30 = wc.ewm(span=30).mean()
        f["wk_ema_stack"] = float(
            float(wc.iloc[-1]) > float(wema10.iloc[-1]) > float(wema30.iloc[-1]))
        f["wk_roc_4"] = float(wc.iloc[-1] / wc.iloc[-5] - 1) * 100 if len(wc) > 5 else np.nan
        f["wk_roc_12"] = float(wc.iloc[-1] / wc.iloc[-13] - 1) * 100 if len(wc) > 13 else np.nan
        f["wk_roc_26"] = float(wc.iloc[-1] / wc.iloc[-27] - 1) * 100 if len(wc) > 27 else np.nan
        # Weekly base: % off 52-week (52-bar) high
        wk_hi52 = float(wk["High"].tail(52).max()) if len(wk) >= 52 else float(wk["High"].max())
        f["wk_dist_52w_high_pct"] = (wk_hi52 - float(wc.iloc[-1])) / wk_hi52 * 100
        # Weekly volume contraction
        wv50 = float(wk["Volume"].rolling(min(20, len(wk))).mean().iloc[-1])
        wv4 = float(wk["Volume"].iloc[-4:].mean())
        f["wk_vdu"] = 1.0 - _safe_div(wv4, wv50, 0.0)
        f["wk_higher_lows"] = float(_slope(wc.tail(12)) > 0)
        # Weekly squeeze
        f["wk_ttm_squeeze"] = float(_ttm_squeeze(wk, n=20))
    else:
        for c in ("wk_rsi_14", "wk_adx_14", "wk_ema_stack", "wk_roc_4",
                  "wk_roc_12", "wk_roc_26", "wk_dist_52w_high_pct",
                  "wk_vdu", "wk_higher_lows", "wk_ttm_squeeze"):
            f[c] = np.nan

    # ─── N. Composite "Coiled Spring" score (port of legacy scanner) ──────
    # Meta-labelling pattern: feed the rule-based score so the model can
    # learn when to trust it (often a top-5 feature in importance tables).
    try:
        f["coiled_spring_score"] = _coiled_spring_score(f)
    except Exception:
        f["coiled_spring_score"] = np.nan

    # ─── L. Fundamentals (cross-sectional ranks) ──────────────────────────
    fund_fields = [
        "marketCap", "trailingPE", "forwardPE", "priceToBook",
        "returnOnEquity", "debtToEquity", "profitMargins",
        "operatingMargins", "revenueGrowth", "earningsGrowth",
        "beta", "dividendYield",
    ]
    rank_fields = [
        "marketCap_rank_sector", "marketCap_rank_all",
        "trailingPE_rank_sector", "trailingPE_rank_all",
        "priceToBook_rank_sector", "priceToBook_rank_all",
        "returnOnEquity_rank_sector", "returnOnEquity_rank_all",
        "debtToEquity_rank_sector", "debtToEquity_rank_all",
        "profitMargins_rank_sector", "profitMargins_rank_all",
        "revenueGrowth_rank_sector", "revenueGrowth_rank_all",
        "earningsGrowth_rank_sector", "earningsGrowth_rank_all",
    ]
    if fund_row:
        for c in fund_fields + rank_fields:
            v = fund_row.get(c)
            try:
                f[f"fund_{c}"] = float(v) if v is not None and pd.notna(v) else np.nan
            except (TypeError, ValueError):
                f[f"fund_{c}"] = np.nan
        # log-mcap (raw mcap is highly skewed)
        mc = fund_row.get("marketCap")
        try:
            f["fund_log_mcap"] = float(np.log10(mc)) if mc and mc > 0 else np.nan
        except (TypeError, ValueError):
            f["fund_log_mcap"] = np.nan
    else:
        for c in fund_fields + rank_fields:
            f[f"fund_{c}"] = np.nan
        f["fund_log_mcap"] = np.nan

    return f


FEATURE_COLUMNS = None  # populated lazily after first call to make_features


def feature_columns(df: pd.DataFrame, setup: dict,
                    bench: pd.Series | None = None) -> list[str]:
    """Return the canonical feature column order. Run once on a sample to
    initialize."""
    global FEATURE_COLUMNS
    if FEATURE_COLUMNS is None:
        FEATURE_COLUMNS = list(make_features(df, setup, bench).keys())
    return FEATURE_COLUMNS
