"""Unit tests for PurgedKFold with Embargo."""

import numpy as np
import pandas as pd
import pytest

from src.ml.purged_cv import PurgedKFold, generate_purged_splits, no_overlap_check


@pytest.fixture
def sample_df():
    """Sample DataFrame with 100 rows and DatetimeIndex."""
    return pd.DataFrame(
        {"a": np.arange(100)},
        index=pd.date_range("2020-01-01", periods=100, freq="B"),
    )


class TestPurgedKFold:
    """Tests for PurgedKFold."""

    def test_basic_split_count(self, sample_df):
        """Test that correct number of splits is generated."""
        cv = PurgedKFold(n_splits=5)
        splits = list(cv.split(sample_df))
        assert len(splits) == 5

    def test_test_size_per_fold(self, sample_df):
        """Test that test sets are roughly equal size."""
        cv = PurgedKFold(n_splits=5)
        for _, test_idx in cv.split(sample_df):
            assert 18 <= len(test_idx) <= 22

    def test_train_test_disjoint(self, sample_df):
        """Test that train and test indices are disjoint."""
        cv = PurgedKFold(n_splits=5)
        for train_idx, test_idx in cv.split(sample_df):
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_purging_removes_samples(self, sample_df):
        """Test that purging removes samples near test period for middle folds."""
        cv = PurgedKFold(n_splits=5, label_span=5)
        total = len(sample_df)
        at_least_one_purged = False
        for train_idx, test_idx in cv.split(sample_df):
            n_purged = total - len(train_idx) - len(test_idx)
            if n_purged > 0:
                at_least_one_purged = True
        assert at_least_one_purged, "No samples purged in any fold! label_span=5 should purge some"

    def test_embargo_removes_additional_samples(self, sample_df):
        """Test that embargo removes more samples than no embargo."""
        cv_no_embargo = PurgedKFold(n_splits=5, label_span=5, embargo_days=0)
        cv_embargo = PurgedKFold(n_splits=5, label_span=5, embargo_days=10)

        total = len(sample_df)
        purged_no_embargo = []
        purged_embargo = []
        for (train1, test1), (train2, test2) in zip(
            cv_no_embargo.split(sample_df), cv_embargo.split(sample_df)
        ):
            purged_no_embargo.append(total - len(train1) - len(test1))
            purged_embargo.append(total - len(train2) - len(test2))

        assert all(e >= n for e, n in zip(purged_embargo, purged_no_embargo))
        assert any(e > n for e, n in zip(purged_embargo, purged_no_embargo))

    def test_pct_embargo(self, sample_df):
        """Test percentage-based embargo."""
        cv = PurgedKFold(n_splits=5, pct_embargo=0.1, label_span=1)
        splits = list(cv.split(sample_df))
        total = len(sample_df)
        for train_idx, test_idx in splits:
            n_purged = total - len(train_idx) - len(test_idx)
            assert n_purged >= 0

    def test_no_shuffle_supported(self):
        """Test that shuffle=True raises ValueError."""
        with pytest.raises(ValueError, match="does not support shuffle"):
            PurgedKFold(n_splits=5, shuffle=True)

    def test_invalid_pct_embargo(self):
        """Test invalid pct_embargo raises ValueError."""
        with pytest.raises(ValueError, match="pct_embargo"):
            PurgedKFold(pct_embargo=-0.1)
        with pytest.raises(ValueError, match="pct_embargo"):
            PurgedKFold(pct_embargo=1.0)

    def test_invalid_embargo_days(self):
        """Test negative embargo_days raises ValueError."""
        with pytest.raises(ValueError, match="embargo_days"):
            PurgedKFold(embargo_days=-1)

    def test_invalid_label_span(self):
        """Test label_span < 1 raises ValueError."""
        with pytest.raises(ValueError, match="label_span"):
            PurgedKFold(label_span=0)

    def test_train_indices_in_order(self, sample_df):
        """Test that remaining train indices are in temporal order."""
        cv = PurgedKFold(n_splits=5, label_span=3)
        for train_idx, _ in cv.split(sample_df):
            assert all(train_idx[i] < train_idx[i + 1] for i in range(len(train_idx) - 1))

    def test_get_n_splits(self, sample_df):
        """Test get_n_splits returns correct value."""
        cv = PurgedKFold(n_splits=7)
        assert cv.get_n_splits(sample_df) == 7

    def test_numpy_array_input(self):
        """Test that numpy arrays work as input."""
        X = np.arange(50).reshape(50, 1)
        cv = PurgedKFold(n_splits=5, label_span=2)
        splits = list(cv.split(X))
        assert len(splits) == 5

    def test_combined_purge_and_embargo(self, sample_df):
        """Test combined label_span purging and embargo_days."""
        cv = PurgedKFold(n_splits=3, label_span=5, embargo_days=3)
        total = len(sample_df)
        purged_counts = []
        for train_idx, test_idx in cv.split(sample_df):
            n_purged = total - len(train_idx) - len(test_idx)
            purged_counts.append(n_purged)
        # At least one fold should purge samples (middle folds have both before and after)
        assert any(n > 0 for n in purged_counts)

    def test_all_samples_covered(self, sample_df):
        """Test that all samples appear in at least one test fold."""
        cv = PurgedKFold(n_splits=5)
        covered = set()
        for _, test_idx in cv.split(sample_df):
            covered.update(test_idx.tolist())
        assert covered == set(range(len(sample_df)))


class TestNoOverlapCheck:
    """Tests for the no_overlap_check function."""

    def test_no_overlap(self):
        """Test when training label windows don't overlap with test."""
        # Train: [0, 1, 2, 3, 4, 5], Test: [10, 11, 12], label_span=2
        # Train windows: [0,1,2], [1,2,3], ..., [5,6,7] — no overlap with [10,11,12]
        all_idx = np.arange(20)
        train = [0, 1, 2, 3, 4, 5]
        test = [10, 11, 12]
        assert no_overlap_check(all_idx, train, test, label_span=2) is True

    def test_overlap_detected(self):
        """Test when training label window overlaps with test."""
        # Train: [5], Test: [6], label_span=2
        # Train window for sample 5: {5, 6, 7} — overlaps with test at 6
        all_idx = np.arange(20)
        train = [5]
        test = [6]
        assert no_overlap_check(all_idx, train, test, label_span=2) is False


class TestGeneratePurgedSplits:
    """Tests for generate_purged_splits helper."""

    def test_returns_correct_n_splits(self, sample_df):
        """Test correct number of split summaries returned."""
        results = generate_purged_splits(sample_df, n_splits=5, pct_embargo=0.05, label_span=3)
        assert len(results) == 5
        assert results[0]["fold"] == 1
        assert results[4]["fold"] == 5

    def test_purged_count_positive(self, sample_df):
        """Test purged sample count is positive when label_span > 0."""
        results = generate_purged_splits(sample_df, n_splits=3, label_span=5)
        for r in results:
            assert r["n_purged"] > 0

    def test_no_overlaps(self, sample_df):
        """Test that no overlaps detected in purged splits."""
        results = generate_purged_splits(sample_df, n_splits=5, label_span=5, pct_embargo=0.05)
        for r in results:
            assert r["has_overlap"] is False

    def test_result_keys(self, sample_df):
        """Test that result dict has expected keys."""
        results = generate_purged_splits(sample_df, n_splits=3)
        for r in results:
            assert set(r.keys()) == {"fold", "n_train", "n_test", "n_purged", "has_overlap"}
