"""Triple-barrier labeling (López de Prado, AFML Ch. 3).

Purpose:
  Convert a candidate breakout setup at `asof` into a supervised
  outcome `y` for ML training, using ONLY future OHLCV bars from
  the same symbol.

How it works:
  At entry (next bar after `asof`) we place three barriers:
    * UPPER  = entry + upper_atr_mult * ATR(n)        (take-profit)
    * LOWER  = entry - lower_atr_mult * ATR(n)        (stop-loss)
    * TIME   = entry + time_days bars                 (timeout)
  Whichever barrier the price touches first decides the label:
    +1 upper, -1 lower, 0 time.
  The realised payoff is also captured as an R-multiple
  (`r_multiple = (exit - entry) / (entry - lower)`).

  For meta-modeling we collapse to binary `y = 1 if label == +1 else 0`.

  `atr_at(df, asof, n)` is a strict no-look-ahead Wilder ATR helper.

Data sources:
  * Per-symbol OHLCV DataFrame passed in by caller.
  * Configurable horizons via `configs/default.yaml::labeling`.

Outputs:
  * `triple_barrier_label(...)` -> dict{label, exit_date, r_multiple,
    days_held, ...}
  * `to_binary(label_raw)`      -> int (0 or 1)
  * `atr_at(df, asof, n)`       -> float | None

How to run:
  Import-only.
      from src.labeling import triple_barrier_label, atr_at

Notes:
  * Bars used for the forward outcome are STRICTLY > asof, ensuring no
    look-ahead leakage.
  * If insufficient forward history exists (recent setups), the label
    is None and the row is dropped from the training corpus.
"""

from __future__ import annotations
import datetime
from typing import Optional

import numpy as np
import pandas as pd


def atr_at(df: pd.DataFrame, asof: pd.Timestamp, n: int = 20) -> Optional[float]:
    """ATR(n) computed using only data ≤ asof."""
    sub = df.loc[:asof]
    if len(sub) < n + 1:
        return None
    h, l, c = sub["High"], sub["Low"], sub["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()],
                   axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def triple_barrier_label(
    df: pd.DataFrame,
    entry_date: pd.Timestamp,
    upper_atr: float = 2.0,
    lower_atr: float = 1.0,
    time_days: int = 20,
    atr_period: int = 20,
    resistance: Optional[float] = None,
    require_close_above_R: bool = False,
    confirm_within_bars: int = 5,
) -> dict:
    """Compute triple-barrier outcome for an entry on entry_date.

    Returns dict with keys:
      label (+1/-1/0/None), exit_date, exit_price, r_multiple, days_held
    label=None if not enough forward data exists.

    If require_close_above_R is True and `resistance` is provided, the
    +1 label is only awarded when BOTH the upper barrier is hit AND at
    least one daily close > resistance occurs within `confirm_within_bars`
    of entry. This filters out spike-and-fail breakouts.
    """
    if entry_date not in df.index:
        return {"label": None}
    entry_close = float(df.loc[entry_date, "Close"])
    a = atr_at(df, entry_date, n=atr_period)
    if a is None or a <= 0:
        return {"label": None}

    upper = entry_close + upper_atr * a
    lower = entry_close - lower_atr * a

    # Forward window
    fwd = df.loc[entry_date:].iloc[1:time_days + 1]
    if fwd.empty:
        return {"label": None}

    # Confirmation: did we get a close above resistance within first N bars?
    confirmed = True
    if require_close_above_R and resistance is not None:
        head = fwd.iloc[:confirm_within_bars]
        confirmed = bool((head["Close"] > resistance).any())

    for i, (ts, row) in enumerate(fwd.iterrows(), start=1):
        # Order: assume worst-case stop-first if both pierced same bar
        # (conservative; matches retail execution reality)
        if row["Low"] <= lower:
            return {
                "label": -1, "exit_date": ts, "exit_price": lower,
                "r_multiple": (lower - entry_close) / (lower_atr * a),
                "days_held": i, "confirmed": confirmed,
            }
        if row["High"] >= upper:
            lbl = 1 if confirmed else 0  # spike-and-fail: count as timeout
            return {
                "label": lbl, "exit_date": ts, "exit_price": upper,
                "r_multiple": (upper - entry_close) / (lower_atr * a),
                "days_held": i, "confirmed": confirmed,
            }
    # Time barrier
    last_close = float(fwd["Close"].iloc[-1])
    return {
        "label": 0, "exit_date": fwd.index[-1], "exit_price": last_close,
        "r_multiple": (last_close - entry_close) / (lower_atr * a),
        "days_held": len(fwd), "confirmed": confirmed,
    }


def to_binary(label: int | None) -> int | None:
    """Meta-model target: 1 = profitable breakout, 0 = stop or chop."""
    if label is None:
        return None
    return 1 if label == 1 else 0


# ─── Trend-scanning labels (López de Prado, AFML Ch. 3.5) ──────────────────

def trend_scanning_label(
    df: pd.DataFrame,
    entry_date: pd.Timestamp,
    min_horizon: int = 5,
    max_horizon: int = 30,
    t_threshold: float = 2.0,
) -> dict:
    """Label by the *strongest* statistically-significant trend that
    follows entry_date.

    Fits an OLS regression of close[entry:entry+h] on time for each
    h in [min_horizon, max_horizon] and picks the horizon with the
    largest |t-stat| of the slope. Returns:
      label = +1 if slope>0 and |t|>t_threshold,
              -1 if slope<0 and |t|>t_threshold,
               0 otherwise.
      best_horizon, t_stat, slope_per_day, exit_date, exit_price.

    Cleaner targets than fixed-horizon: each label captures the *real*
    post-entry trend instead of being noised by an arbitrary cut-off.
    """
    if entry_date not in df.index:
        return {"label": None}
    fwd = df.loc[entry_date:].iloc[1:max_horizon + 1]
    if len(fwd) < min_horizon:
        return {"label": None}

    closes = fwd["Close"].values.astype(float)
    if not np.all(np.isfinite(closes)):
        return {"label": None}

    best_t = 0.0
    best_h = min_horizon
    best_slope = 0.0
    for h in range(min_horizon, len(closes) + 1):
        y = closes[:h]
        x = np.arange(h, dtype=float)
        # OLS: y = a + b*x
        x_mean, y_mean = x.mean(), y.mean()
        sxx = ((x - x_mean) ** 2).sum()
        if sxx <= 0:
            continue
        b = ((x - x_mean) * (y - y_mean)).sum() / sxx
        a = y_mean - b * x_mean
        resid = y - (a + b * x)
        rss = (resid ** 2).sum()
        if h <= 2 or rss <= 0:
            continue
        sigma2 = rss / (h - 2)
        se_b = (sigma2 / sxx) ** 0.5
        if se_b <= 0:
            continue
        t = b / se_b
        if abs(t) > abs(best_t):
            best_t = t
            best_h = h
            best_slope = b

    entry_close = float(df.loc[entry_date, "Close"])
    if abs(best_t) < t_threshold:
        label = 0
    elif best_t > 0:
        label = 1
    else:
        label = -1

    exit_date = fwd.index[best_h - 1]
    exit_price = float(fwd["Close"].iloc[best_h - 1])
    return {
        "label": label,
        "exit_date": exit_date,
        "exit_price": exit_price,
        "t_stat": float(best_t),
        "slope_per_day": float(best_slope),
        "best_horizon": int(best_h),
        "r_multiple": (exit_price - entry_close) / max(entry_close * 0.01, 1e-6),
        "days_held": int(best_h),
    }
