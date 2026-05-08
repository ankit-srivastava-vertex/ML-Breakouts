"""Time-aware cross-validation for financial ML.

Purpose:
  Provide cross-validation splitters that respect the temporal /
  label-overlap structure of breakout-style problems, and a
  `sample_uniqueness_weights()` helper that down-weights samples whose
  forward windows overlap many neighbours.

Why not standard k-fold?
  Triple-barrier labels span up to `time_days` future bars, so two
  samples at t and t+5 share label information. Standard k-fold and
  i.i.d. shuffles leak future info into the train fold via this overlap.

What is provided:
  * `PurgedKFold` (López de Prado, AFML Ch. 7)
      - PURGE: drop train samples whose label window overlaps the test set.
      - EMBARGO: drop train samples within `embargo_days` AFTER the test set
        (to prevent serial-correlation leakage).
  * `WalkForwardCV` — expanding-window splits (canonical for production
    backtests; default scheme used by `scripts/03_train_model.py`).
  * `sample_uniqueness_weights(t1)` — Σ 1 / concurrency for each sample,
     fed to LightGBM as `sample_weight`.

Data sources:
  None directly. Operates on pandas Series/DataFrames passed in by callers.

Inputs (all callers):
  * `t1` — Series indexed by entry-date, value = label-exit-date
    (i.e. `asof + label_horizon_days`).

Outputs:
  Generators yielding `(train_idx, test_idx)` numpy arrays.

How to run:
  Import-only.
      from src.cv import PurgedKFold, WalkForwardCV, sample_uniqueness_weights

References:
  Marcos López de Prado, *Advances in Financial Machine Learning*,
  Wiley 2018, Ch. 4 (sample weights) and Ch. 7 (cross-validation).
"""

from __future__ import annotations
from typing import Iterator
import numpy as np
import pandas as pd


class PurgedKFold:
    def __init__(self, n_splits: int = 5, embargo_days: int = 25,
                 label_horizon_days: int = 20):
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.label_horizon_days = label_horizon_days

    def split(self, t1: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """t1: Series indexed by entry-date (asof) where each value is the
        label EXIT date (entry + label_horizon).

        Yields (train_idx, test_idx) as positional integer arrays.
        """
        if not t1.index.is_monotonic_increasing:
            raise ValueError("t1 must be sorted ascending by index.")

        n = len(t1)
        indices = np.arange(n)
        fold_sizes = np.full(self.n_splits, n // self.n_splits)
        fold_sizes[:n % self.n_splits] += 1

        cur = 0
        bounds = []
        for fs in fold_sizes:
            bounds.append((cur, cur + fs))
            cur += fs

        for start, stop in bounds:
            test_idx = indices[start:stop]
            test_start = t1.index[start]
            test_end = t1.index[stop - 1]
            embargo_end = test_end + pd.Timedelta(days=self.embargo_days)

            # Train = everything outside test, with purge + embargo applied
            train_mask = np.ones(n, dtype=bool)
            train_mask[start:stop] = False

            # Purge: any train sample whose label-exit ≥ test_start AND
            # whose entry ≤ test_end overlaps the test fold
            for i in indices:
                if not train_mask[i]:
                    continue
                entry = t1.index[i]
                exit_ = t1.iloc[i]
                if entry <= test_end and exit_ >= test_start:
                    train_mask[i] = False
                # Embargo: drop train samples in window after test
                elif test_end < entry <= embargo_end:
                    train_mask[i] = False

            train_idx = indices[train_mask]
            yield train_idx, test_idx


class WalkForwardCV:
    """Walk-forward CV with growing train window + fixed-size test window.

    Yields (train_idx, test_idx) positional integer arrays. Embargo is
    applied between train end and test start to prevent label-overlap
    leakage. This is the honest forward-looking validation.

    Args:
        train_min_days:  min calendar days of training history before
                         first fold can run (warm-up).
        test_days:       calendar days per test window.
        step_days:       slide step between consecutive folds.
        embargo_days:    embargo after train -> test boundary.
    """

    def __init__(self, train_min_days: int = 365 * 3,
                 test_days: int = 180,
                 step_days: int = 90,
                 embargo_days: int = 15):
        self.train_min_days = train_min_days
        self.test_days = test_days
        self.step_days = step_days
        self.embargo_days = embargo_days

    def split(self, t1: pd.Series):
        if not t1.index.is_monotonic_increasing:
            raise ValueError("t1 must be sorted ascending by index.")
        n = len(t1)
        indices = np.arange(n)
        first = t1.index.min()
        last = t1.index.max()

        cur_test_start = first + pd.Timedelta(days=self.train_min_days)
        while cur_test_start + pd.Timedelta(days=self.test_days) <= last:
            test_end = cur_test_start + pd.Timedelta(days=self.test_days)
            train_end = cur_test_start - pd.Timedelta(days=self.embargo_days)

            tr_mask = (t1.index <= train_end)
            te_mask = (t1.index >= cur_test_start) & (t1.index < test_end)
            tr = indices[tr_mask]
            te = indices[te_mask]
            if len(tr) > 100 and len(te) > 20:
                yield tr, te
            cur_test_start = cur_test_start + pd.Timedelta(days=self.step_days)


def sample_uniqueness_weights(t1: pd.Series) -> np.ndarray:
    """Average uniqueness weights (AFML §4.3).

    Sample weight ∝ 1 / (number of concurrent overlapping labels at that
    time). Reduces effective influence of clustered samples.
    """
    if t1.empty:
        return np.array([])

    all_dates = pd.date_range(t1.index.min(), t1.max(), freq="D")
    concurrency = pd.Series(0, index=all_dates, dtype=float)
    for entry, exit_ in t1.items():
        concurrency.loc[entry:exit_] += 1.0
    concurrency = concurrency.replace(0, 1.0)

    weights = np.zeros(len(t1))
    for i, (entry, exit_) in enumerate(t1.items()):
        span = concurrency.loc[entry:exit_]
        weights[i] = (1.0 / span).mean() if len(span) else 1.0
    # Normalize to mean 1
    weights = weights / weights.mean() if weights.mean() > 0 else weights
    return weights
