"""
PurgedKFold with Embargo — proper cross-validation for financial data.

Implements PurgedKFold as described in Lopez de Prado (ML4T Ch6:04):
- Purges training samples whose label window overlaps with any test sample
- Adds an embargo buffer after test period to prevent information leakage

The key insight: in financial ML, labels are forward returns. A sample at time t
with horizon h uses data from [t, t+h]. If any test date falls in [t, t+h],
sample t must be purged from training because its label leaks test information.

For K-Fold, the train set contains all indices *not* in fold i. Some of those
indices have labels computed with data that extends into the fold-i period.
We remove (purge) those indices from training.

Usage:
    from src.ml.purged_cv import PurgedKFold
    cv = PurgedKFold(n_splits=5, pct_embargo=0.05)
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Iterator


class PurgedKFold(KFold):
    """K-Fold with purging and embargo for financial time series.

    For each test fold, removes training samples whose forward-looking label
    window would overlap with the test period. Additionally adds an embargo
    buffer between test and the next training sample.

    Attributes:
        n_splits: Number of folds.
        pct_embargo: Fraction of test span to embargo (e.g. 0.05).
        embargo_days: Fixed embargo days (overrides pct_embargo if > 0).
        label_span: Forward return horizon (how far labels look ahead).
    """

    def __init__(
        self,
        n_splits: int = 5,
        pct_embargo: float = 0.0,
        embargo_days: int = 0,
        label_span: int = 1,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        if shuffle:
            raise ValueError("PurgedKFold does not support shuffle")
        if pct_embargo < 0 or pct_embargo >= 1:
            raise ValueError("pct_embargo must be in [0, 1)")
        if embargo_days < 0:
            raise ValueError("embargo_days must be >= 0")
        if label_span < 1:
            raise ValueError("label_span must be >= 1")

        super().__init__(n_splits=n_splits, shuffle=False, random_state=random_state)
        self.pct_embargo = pct_embargo
        self.embargo_days = embargo_days
        self.label_span = label_span

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate purged train/test indices per fold.

        Purge window for fold with test range [t1, t2]:
          - Purge train samples i where t1 <= i <= t2 + label_span + embargo
          - Because sample i's label uses data up to i + label_span,
            any i with i + label_span >= t1 leaks into the test period
          - Also embargo t2+1 through t2+embargo to prevent serial correlation

        Args:
            X: Feature data (DataFrame with DatetimeIndex or array).
            y: Labels (only used for length).
            groups: Unused (sklearn compatibility).

        Yields:
            (purged_train_idx, test_idx) numpy arrays.
        """
        n_samples = len(X)

        for _, (base_train_idx, test_idx) in enumerate(super().split(X)):
            t1 = int(np.min(test_idx))
            t2 = int(np.max(test_idx))
            embargo = self._embargo_samples(test_idx, n_samples)

            # Sample i's label uses data in [i, i + label_span].
            # Purge from training any i where i + label_span >= t1 (i.e. i >= t1 - label_span)
            # and also i <= t2 + embargo (samples right after test period due to embargo).
            purge_start = t1 - self.label_span
            purge_end = t2 + embargo

            train_idx = base_train_idx[
                (base_train_idx < purge_start) | (base_train_idx > purge_end)
            ]

            yield train_idx, test_idx

    def _embargo_samples(self, test_idx: np.ndarray, n_samples: int) -> int:
        """Compute embargo count from config.

        Args:
            test_idx: Test fold indices.
            n_samples: Total samples.

        Returns:
            Number of samples to embargo after test period.
        """
        if self.embargo_days > 0:
            return self.embargo_days
        if self.pct_embargo <= 0:
            return 0
        test_span = len(test_idx)
        embargo = int(np.ceil(test_span * self.pct_embargo))
        return max(0, min(embargo, n_samples - int(np.max(test_idx)) - 1))

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return number of splits."""
        return self.n_splits


def no_overlap_check(
    all_indices: np.ndarray,
    train_indices: list[int] | np.ndarray,
    test_indices: list[int] | np.ndarray,
    label_span: int,
) -> bool:
    """Verify no training label window overlaps with test indices.

    For training sample i, label window is [i, i + label_span].
    Returns True if no test index falls in any training sample's label window.

    Args:
        all_indices: All sample indices.
        train_indices: Training fold indices.
        test_indices: Test fold indices.
        label_span: Forward return horizon.

    Returns:
        True = no overlap (pass), False = overlap detected (fail).
    """
    test_set = set(test_indices)
    for i in train_indices:
        for j in range(i, min(i + label_span + 1, len(all_indices))):
            if j in test_set:
                return False
    return True


def generate_purged_splits(
    X: pd.DataFrame,
    n_splits: int = 5,
    pct_embargo: float = 0.05,
    label_span: int = 5,
) -> list[dict]:
    """Generate and summarize purged CV splits.

    Args:
        X: Feature DataFrame.
        n_splits: Number of folds.
        pct_embargo: Embargo fraction.
        label_span: Label computation window.

    Returns:
        List of dicts: fold, n_train, n_test, n_purged, has_overlap.
    """
    cv = PurgedKFold(n_splits=n_splits, pct_embargo=pct_embargo, label_span=label_span)
    all_indices = np.arange(len(X))
    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        n_purged = len(all_indices) - len(train_idx) - len(test_idx)
        results.append(
            {
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_purged": n_purged,
                "has_overlap": not no_overlap_check(all_indices, train_idx, test_idx, label_span),
            }
        )

    return results
