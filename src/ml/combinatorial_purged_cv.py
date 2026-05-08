"""
Combinatorial Purged Cross-Validation (CPCV) for financial data.

Extends the standard PurgedKFold with combinatorial splitting as described
in López de Prado (2018) "Advances in Financial Machine Learning", Chapter 12.

Standard PurgedKFold removes training samples whose label window overlaps with
the test period boundary. CPCV goes further by:

1. Splitting data into N groups (instead of folds).
2. Testing on all combinations of k groups (instead of sequential folds).
3. Purging any training sample whose label window overlaps with ANY test sample
   in the current combination.

The result is C(N, k) backtest paths, providing:
- A distribution of performance statistics (not a single point estimate)
- Better estimate of backtest variance
- Helps identify strategies that are robust across different test periods

Usage:
    from src.ml.combinatorial_purged_cv import CombinatorialPurgedCV

    cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, label_span=5)
    for path_idx, (train_idx, test_idx) in enumerate(cpcv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
"""

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from typing import Iterator
from math import comb


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation for time series.

    Generates backtest paths by testing on all combinations of n_test_groups
    from n_groups total groups, with purging between train and test sets.

    Each sample i has a label window [i, i + label_span]. When a sample's label
    window overlaps with any test sample, it is purged from training for that
    specific backtest path.

    Attributes:
        n_groups: Total number of time-ordered groups.
        n_test_groups: Number of groups per test combination.
        label_span: Forward return horizon (label window size).
        embargo_days: Additional buffer after test period.
        purge_all_overlaps: If True, purge all train samples whose label
            windows overlap with ANY test sample (strict). If False, only
            purge at combinatorial boundaries (matches standard PurgedKFold).
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        label_span: int = 1,
        embargo_days: int = 0,
        pct_embargo: float = 0.0,
        purge_all_overlaps: bool = True,
    ) -> None:
        if n_groups < 3:
            raise ValueError("n_groups must be >= 3")
        if n_test_groups < 1 or n_test_groups >= n_groups:
            raise ValueError(f"n_test_groups must be in [1, {n_groups - 1}]")
        if label_span < 1:
            raise ValueError("label_span must be >= 1")
        if embargo_days < 0:
            raise ValueError("embargo_days must be >= 0")
        if pct_embargo < 0 or pct_embargo >= 1:
            raise ValueError("pct_embargo must be in [0, 1)")

        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.label_span = label_span
        self.embargo_days = embargo_days
        self.pct_embargo = pct_embargo
        self.purge_all_overlaps = purge_all_overlaps

        self._n_paths = comb(n_groups, n_test_groups)
        self._group_boundaries: list[tuple[int, int]] = []
        self._all_indices: np.ndarray | None = None

    @property
    def n_paths(self) -> int:
        """Number of combinatorial backtest paths: C(n_groups, n_test_groups)."""
        return self._n_paths

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, dict]]:
        """Generate purged train/test splits for each combinatorial path.

        Args:
            X: Feature data (DataFrame with DatetimeIndex or array).
            y: Labels (only used for length).
            groups: Unused (sklearn compatibility).

        Yields:
            (train_idx, test_idx, meta_info) per backtest path.
            meta_info includes: path_index, test_groups, n_train, n_test, n_purged.
        """
        n_samples = len(X)
        self._all_indices = np.arange(n_samples)
        self._compute_group_boundaries(n_samples)

        path_idx = 0
        for test_combo in itertools.combinations(range(self.n_groups), self.n_test_groups):
            test_idx = self._indices_for_groups(test_combo)
            train_idx = self._build_purged_train(test_combo, test_idx, n_samples)

            meta = {
                "path_index": path_idx,
                "test_groups": test_combo,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_purged": n_samples - len(train_idx) - len(test_idx),
            }
            yield train_idx, test_idx, meta
            path_idx += 1

    def _compute_group_boundaries(self, n_samples: int) -> None:
        """Partition all samples into n_groups of roughly equal size.

        Args:
            n_samples: Total number of samples.
        """
        group_size = n_samples // self.n_groups
        remainder = n_samples % self.n_groups

        boundaries = []
        start = 0
        for g in range(self.n_groups):
            size = group_size + (1 if g < remainder else 0)
            end = start + size
            boundaries.append((start, end - 1))
            start = end
        self._group_boundaries = boundaries

    def _indices_for_groups(self, group_indices: tuple[int, ...]) -> np.ndarray:
        """Get all sample indices belonging to specified groups.

        Args:
            group_indices: Tuple of group numbers to collect.

        Returns:
            Sorted array of all sample indices in those groups.
        """
        indices = []
        for g in group_indices:
            start, end = self._group_boundaries[g]
            indices.extend(range(start, end + 1))
        return np.array(sorted(indices))

    def _build_purged_train(
        self,
        test_combo: tuple[int, ...],
        test_idx: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Build training set with purging.

        Args:
            test_combo: Tuple of group indices in the test set.
            test_idx: Actual sample indices in the test set.
            n_samples: Total number of samples.

        Returns:
            Purged training indices array.
        """
        test_set = set(test_idx)
        train_indices = []

        for i in range(n_samples):
            if i in test_set:
                continue
            if self._should_purge(i, test_idx, n_samples):
                continue
            train_indices.append(i)

        return np.array(train_indices)

    def _should_purge(
        self,
        sample_idx: int,
        test_idx: np.ndarray,
        n_samples: int,
    ) -> bool:
        """Check if a training sample should be purged.

        A sample is purged if its label window [idx, idx + label_span]
        overlaps with any test index, either directly or through the
        embargo buffer.

        Args:
            sample_idx: Candidate training sample index.
            test_idx: All test sample indices for current path.
            n_samples: Total samples (for boundary checking).

        Returns:
            True if sample should be purged from training.
        """
        label_start = sample_idx
        label_end = sample_idx + self.label_span

        if self.embargo_days > 0:
            label_end += self.embargo_days
        elif self.pct_embargo > 0 and len(test_idx) > 0:
            embargo = int(np.ceil(len(test_idx) * self.pct_embargo))
            label_end += embargo

        if label_start < 0:
            label_start = 0

        test_min = int(np.min(test_idx))
        test_max = int(np.max(test_idx))

        if label_end < test_min:
            return False

        if label_start > test_max:
            return False

        if self.purge_all_overlaps:
            for j in range(sample_idx, min(label_end + 1, n_samples)):
                if j in set(test_idx):
                    return True
            return False
        else:
            return label_end >= test_min

    def get_path_summary(self, X: pd.DataFrame | np.ndarray) -> list[dict]:
        """Generate a summary of all backtest paths.

        Args:
            X: Feature data for sizing.

        Returns:
            List of dicts with per-path statistics.
        """
        summaries = []
        for train_idx, test_idx, meta in self.split(X):
            summaries.append(
                {
                    "path": meta["path_index"],
                    "test_groups": list(meta["test_groups"]),
                    "n_train": meta["n_train"],
                    "n_test": meta["n_test"],
                    "n_purged": meta["n_purged"],
                    "train_pct": meta["n_train"] / len(X),
                    "test_pct": meta["n_test"] / len(X),
                }
            )
        return summaries

    def get_test_coverage(self) -> dict[int, int]:
        """Count how many paths include each group in the test set.

        Returns:
            Dict mapping group index to count of paths where it's in test.
        """
        coverage = {g: 0 for g in range(self.n_groups)}
        for test_combo in itertools.combinations(range(self.n_groups), self.n_test_groups):
            for g in test_combo:
                coverage[g] += 1
        return coverage

    def get_train_test_ratio_stats(self, X: pd.DataFrame | np.ndarray) -> dict:
        """Compute statistics on train/test split sizes across all paths.

        Args:
            X: Feature data.

        Returns:
            Dict with min/max/mean for train, test, and purged counts.
        """
        train_sizes = []
        test_sizes = []
        purged_sizes = []

        for _, _, meta in self.split(X):
            train_sizes.append(meta["n_train"])
            test_sizes.append(meta["n_test"])
            purged_sizes.append(meta["n_purged"])

        return {
            "n_paths": len(train_sizes),
            "train_size": {
                "min": int(np.min(train_sizes)),
                "max": int(np.max(train_sizes)),
                "mean": float(np.mean(train_sizes)),
            },
            "test_size": {
                "min": int(np.min(test_sizes)),
                "max": int(np.max(test_sizes)),
                "mean": float(np.mean(test_sizes)),
            },
            "purged_size": {
                "min": int(np.min(purged_sizes)),
                "max": int(np.max(purged_sizes)),
                "mean": float(np.mean(purged_sizes)),
            },
        }
