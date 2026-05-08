"""Setup detector — the rule-based PRIMARY model.

Purpose:
  Identify a horizontal-resistance breakout setup at any as-of date.
  Used in two places:
    1. Live scanning  — `scripts/05_daily_scan.py` (one call per symbol).
    2. Training-set construction — `scripts/02_build_training_set.py`
       replays history bar-by-bar and emits one row per detected setup.

How it works:
  * Scans the last `RES_LOOKBACK_DAYS` (default 600) of price history
    and clusters horizontal pivots within `RES_BAND_PCT` of one another
    to identify candidate resistance levels (R).
  * Picks the resistance with the most touches (>= MIN_TOUCHES) where
    the current close is within `PROXIMITY_MAX_PCT` of R AND the base
    length is >= `BASE_MIN_DAYS`.
  * v2 (May 2026) hardened quality gates — a setup is emitted ONLY
    when ALL of the following hold (configurable via
    `configs/default.yaml::setup`):
      - cluster has >=2 touches AND base_len >= 30 bars
      - within `proximity_max_pct` of R
      - ATR% (atr/close) <= `max_atr_pct`              (reject vol chop)
      - 5-day RVOL <= `max_rvol_dryup_5d`              (require dry-up)
      - close > 50DMA                                  (uptrend posture)
      - RS vs benchmark over 3m >= `min_rs_3m_pct`
      - distance from 52w high <= `max_dist_52w_pct`
      - dollar volume (50d avg, INR cr) >= `min_dollar_vol_cr`
  * v2 dropped raw-setup count from ~242k → ~24k labeled setups.

Data sources:
  Pure-function. Caller supplies `(df, bench)` (OHLCV + benchmark Close).

Outputs:
  dict | None  — setup descriptor with keys:
    R, base_start, base_end, base_len_days, touches, distance_pct,
    is_52w_high, ... (None when no qualifying setup is present).

How to run:
  Import-only.
      from src.setup_detector import detect_resistance
      setup = detect_resistance(df, bench=bench)

Notes:
  * Disable hardening by passing `enforce_quality_gates=False` (used
    when intentionally building a noisier corpus for analysis).
  * Defaults at the top of this file are NOT the production values —
    `configs/default.yaml::setup` overrides them at every call site.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

# Defaults (overridable via configs/default.yaml)
RES_LOOKBACK_DAYS = 600
BASE_MIN_DAYS = 30
RES_BAND_PCT = 0.035
PROXIMITY_MAX_PCT = 0.05      # was 0.08
MIN_TOUCHES = 2

# v2 quality gates
MAX_ATR_PCT = 6.0             # ATR/Close * 100, last bar
MAX_RVOL_DRYUP_5D = 1.20      # 5d avg vol / 50d avg vol — require contraction
MIN_RS_3M_PCT = 0.0           # stock 3m return - bench 3m return >= this
MAX_DIST_52W_PCT = 12.0       # |R - 52w high| / 52w high * 100
MIN_DOLLAR_VOL_CR = 0.5       # 50d avg INR cr daily turnover


def fractal_pivots(highs: pd.Series, k: int = 3) -> pd.Series:
    """True where highs[i] is the local max over [i-k, i+k]."""
    h = highs.values
    n = len(h)
    out = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        if (h[i] == h[i - k:i + k + 1].max()
                and h[i] >= h[i - 1] and h[i] >= h[i + 1]):
            out[i] = True
    return pd.Series(out, index=highs.index)


def detect_resistance(
    df: pd.DataFrame,
    res_lookback: int = RES_LOOKBACK_DAYS,
    base_min_days: int = BASE_MIN_DAYS,
    res_band_pct: float = RES_BAND_PCT,
    proximity_max_pct: float = PROXIMITY_MAX_PCT,
    min_touches: int = MIN_TOUCHES,
    # v2 quality gates
    max_atr_pct: float = MAX_ATR_PCT,
    max_rvol_dryup_5d: float = MAX_RVOL_DRYUP_5D,
    min_rs_3m_pct: float = MIN_RS_3M_PCT,
    max_dist_52w_pct: float = MAX_DIST_52W_PCT,
    min_dollar_vol_cr: float = MIN_DOLLAR_VOL_CR,
    bench: Optional[pd.Series] = None,
    enforce_quality_gates: bool = True,
) -> Optional[dict]:
    """Find the best horizontal resistance the stock is approaching.

    df must be ordered ascending by date. Returns dict or None.
    """
    if len(df) < base_min_days + 20:
        return None

    last_close = float(df["Close"].iloc[-1])

    # ─── v2 quality gates (cheap, do first) ────────────────────────────
    if enforce_quality_gates:
        # Liquidity: 50d avg INR cr turnover
        if len(df) >= 50:
            dv_cr = float((df["Close"] * df["Volume"]).rolling(50).mean().iloc[-1] / 1e7)
            if dv_cr < min_dollar_vol_cr:
                return None
        # ATR%
        if len(df) >= 21:
            h, l, c = df["High"], df["Low"], df["Close"]
            pc = c.shift(1)
            tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()],
                           axis=1).max(axis=1)
            atr = float(tr.rolling(20).mean().iloc[-1])
            atr_pct = atr / last_close * 100 if last_close > 0 else 999
            if atr_pct > max_atr_pct:
                return None
        # Volume dry-up: 5d / 50d
        if len(df) >= 50:
            v5 = float(df["Volume"].iloc[-5:].mean())
            v50 = float(df["Volume"].iloc[-50:].mean())
            if v50 > 0 and (v5 / v50) > max_rvol_dryup_5d:
                return None
        # Trend posture: close > 50DMA
        if len(df) >= 50:
            ma50 = float(df["Close"].rolling(50).mean().iloc[-1])
            if last_close < ma50:
                return None
        # Relative strength vs benchmark, 3m
        if bench is not None and len(bench.dropna()) > 65:
            b = bench.reindex(df.index).ffill()
            if len(b.dropna()) > 65:
                stk_3m = (df["Close"].iloc[-1] / df["Close"].iloc[-65] - 1) * 100
                bch_3m = (b.iloc[-1] / b.iloc[-65] - 1) * 100
                if (stk_3m - bch_3m) < min_rs_3m_pct:
                    return None

    window = df.tail(res_lookback)

    piv_tight = fractal_pivots(window["High"], k=3)
    piv_broad = fractal_pivots(window["High"], k=8)
    pivots = pd.concat([
        window["High"][piv_tight],
        window["High"][piv_broad],
    ]).groupby(level=0).max()
    if len(pivots) < min_touches:
        return None

    levels = sorted(pivots.tolist(), reverse=True)
    clusters = []
    for lvl in levels:
        placed = False
        for c in clusters:
            if abs(lvl - c["level"]) / c["level"] <= res_band_pct:
                c["sum"] += lvl
                c["count"] += 1
                c["level"] = c["sum"] / c["count"]
                placed = True
                break
        if not placed:
            clusters.append({"level": lvl, "sum": lvl, "count": 1})

    candidates = []
    high_52w = float(window["High"].max())
    for c in clusters:
        R = c["level"]
        dist = (R - last_close) / last_close
        if c["count"] < min_touches:
            continue
        if dist < -0.03 or dist > proximity_max_pct:
            continue
        # 52w-high proximity gate
        if enforce_quality_gates and high_52w > 0:
            dist_from_52w_pct = abs(R - high_52w) / high_52w * 100
            if dist_from_52w_pct > max_dist_52w_pct:
                continue
        cluster_idx = [
            ts for ts in pivots.index
            if abs(pivots.loc[ts] - R) / R <= res_band_pct
        ]
        if len(cluster_idx) < min_touches:
            continue
        base_start = min(cluster_idx)
        base_len = (df.index[-1] - base_start).days
        if base_len < base_min_days:
            continue
        is_52w_high = R >= high_52w * 0.98
        candidates.append({
            "R": R,
            "touches": len(cluster_idx),
            "base_start": base_start,
            "base_len_days": base_len,
            "distance_pct": dist,
            "touch_dates": cluster_idx,
            "is_52w_high": is_52w_high,
        })

    if not candidates:
        return None
    candidates.sort(key=lambda c: (
        not c["is_52w_high"], -c["touches"], abs(c["distance_pct"])
    ))
    return candidates[0]
