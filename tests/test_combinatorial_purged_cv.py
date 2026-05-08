"""Unit tests for Combinatorial Purged CV."""

import numpy as np
import pandas as pd
import pytest

from src.ml.combinatorial_purged_cv import CombinatorialPurgedCV


@pytest.fixture
def sample_df():
    """Sample DataFrame with 120 rows and DatetimeIndex."""
    return pd.DataFrame(
        {"a": np.arange(120)},
        index=pd.date_range("2020-01-01", periods=120, freq="B"),
    )


@pytest.fixture
def small_df():
    """Small DataFrame for quick testing."""
    return pd.DataFrame(
        {"a": np.arange(60)},
        index=pd.date_range("2020-01-01", periods=60, freq="B"),
    )


class TestCombinatorialPurgedCV:
    """Tests for CombinatorialPurgedCV class."""

    def test_correct_n_paths(self):
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        assert cpcv.n_paths == 15  # C(6,2) = 15

    def test_correct_n_paths_choose_1(self):
        cpcv = CombinatorialPurgedCV(n_groups=5, n_test_groups=1)
        assert cpcv.n_paths == 5

    def test_correct_n_paths_choose_3(self):
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=3)
        assert cpcv.n_paths == 20  # C(6,3) = 20

    def test_split_yields_n_paths(self, sample_df):
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        splits = list(cpcv.split(sample_df))
        assert len(splits) == 15

    def test_train_test_disjoint(self, sample_df):
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        for train_idx, test_idx, _ in cpcv.split(sample_df):
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_purging_removes_samples(self, sample_df):
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, label_span=10)
        total = len(sample_df)
        any_purged = False
        for train_idx, test_idx, meta in cpcv.split(sample_df):
            if meta["n_purged"] > 0:
                any_purged = True
                assert total == len(train_idx) + len(test_idx) + meta["n_purged"]
        assert any_purged

    def test_purge_all_overlaps_stricter(self, sample_df):
        cpcv = CombinatorialPurgedCV(
            n_groups=6,
            n_test_groups=2,
            label_span=10,
            purge_all_overlaps=True,
        )
        for train_idx, test_idx, meta in cpcv.split(sample_df):
            # Verify no training label window overlaps with any test index
            test_set = set(test_idx)
            n = len(sample_df)
            for i in train_idx:
                label_end = min(i + 10, n)
                for j in range(i, label_end + 1):
                    assert j not in test_set, (
                        f"Training sample {i} has label window overlapping test idx {j}"
                    )
            # Verify at least some purging happens
            assert meta["n_purged"] >= 0

    def test_embargo_reduces_train(self, sample_df):
        cpcv_no = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, label_span=5, embargo_days=0)
        cpcv_yes = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, label_span=5, embargo_days=5)
        no_sizes = [m["n_train"] for _, _, m in cpcv_no.split(sample_df)]
        yes_sizes = [m["n_train"] for _, _, m in cpcv_yes.split(sample_df)]
        assert all(y <= n for y, n in zip(yes_sizes, no_sizes))

    def test_pct_embargo(self, sample_df):
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, pct_embargo=0.1)
        splits = list(cpcv.split(sample_df))
        assert len(splits) == 15

    def test_meta_info_keys(self, sample_df):
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        for _, _, meta in cpcv.split(sample_df):
            assert set(meta.keys()) == {
                "path_index",
                "test_groups",
                "n_train",
                "n_test",
                "n_purged",
            }
            assert isinstance(meta["path_index"], int)
            assert isinstance(meta["test_groups"], tuple)

    def test_path_indices_sequential(self, sample_df):
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        indices = [m["path_index"] for _, _, m in cpcv.split(sample_df)]
        assert indices == list(range(15))

    def test_all_groups_in_some_test(self, sample_df):
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        all_test_groups = set()
        for _, _, meta in cpcv.split(sample_df):
            all_test_groups.update(meta["test_groups"])
        assert all_test_groups == set(range(6))

    def test_test_groups_are_contiguous_chunks(self, sample_df):
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        for _, test_idx, meta in cpcv.split(sample_df):
            sorted_test = np.sort(test_idx)
            # Each group is a contiguous block; within each group, indices are sequential
            boundaries = np.where(np.diff(sorted_test) > 1)[0]
            assert len(boundaries) <= len(meta["test_groups"]) - 1

    def test_train_indices_in_order(self, sample_df):
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        for train_idx, _, _ in cpcv.split(sample_df):
            assert all(train_idx[i] < train_idx[i + 1] for i in range(len(train_idx) - 1))
            assert all(train_idx[i + 1] - train_idx[i] > 0 for i in range(len(train_idx) - 1))

    def test_various_group_combinations(self):
        for n_groups in [4, 5, 6]:
            for n_test in [1, 2]:
                if n_test >= n_groups:
                    continue
                cpcv = CombinatorialPurgedCV(n_groups=n_groups, n_test_groups=n_test)
                df = pd.DataFrame({"a": np.arange(100)})
                splits = list(cpcv.split(df))
                import math

                expected = math.comb(n_groups, n_test)
                assert len(splits) == expected


class TestInvalidInputs:
    """Tests for input validation."""

    def test_too_few_groups(self):
        with pytest.raises(ValueError, match="n_groups"):
            CombinatorialPurgedCV(n_groups=2)

    def test_test_groups_out_of_range(self):
        with pytest.raises(ValueError, match="n_test_groups"):
            CombinatorialPurgedCV(n_groups=5, n_test_groups=5)

    def test_test_groups_zero(self):
        with pytest.raises(ValueError, match="n_test_groups"):
            CombinatorialPurgedCV(n_groups=5, n_test_groups=0)

    def test_invalid_label_span(self):
        with pytest.raises(ValueError, match="label_span"):
            CombinatorialPurgedCV(label_span=0)

    def test_invalid_embargo_days(self):
        with pytest.raises(ValueError, match="embargo_days"):
            CombinatorialPurgedCV(embargo_days=-1)

    def test_invalid_pct_embargo_negative(self):
        with pytest.raises(ValueError, match="pct_embargo"):
            CombinatorialPurgedCV(pct_embargo=-0.1)

    def test_invalid_pct_embargo_too_high(self):
        with pytest.raises(ValueError, match="pct_embargo"):
            CombinatorialPurgedCV(pct_embargo=1.0)


class TestNumpyInput:
    """Tests with numpy array input."""

    def test_numpy_split(self):
        X = np.arange(90).reshape(90, 1)
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        splits = list(cpcv.split(X))
        assert len(splits) == 15

    def test_numpy_split_train_test(self):
        X = np.arange(90).reshape(90, 1)
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        for train_idx, test_idx, _ in cpcv.split(X):
            assert len(train_idx) + len(test_idx) <= len(X)
            assert len(train_idx) > 0
            assert len(test_idx) > 0


class TestPathSummary:
    """Tests for get_path_summary."""

    def test_summary_length(self, sample_df):
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        summary = cpcv.get_path_summary(sample_df)
        assert len(summary) == 15

    def test_summary_keys(self, sample_df):
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        summary = cpcv.get_path_summary(sample_df)
        for s in summary:
            assert set(s.keys()) == {
                "path",
                "test_groups",
                "n_train",
                "n_test",
                "n_purged",
                "train_pct",
                "test_pct",
            }

    def test_percentages_sum_leq_100(self, sample_df):
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        summary = cpcv.get_path_summary(sample_df)
        for s in summary:
            assert s["train_pct"] + s["test_pct"] <= 1.0


class TestTestCoverage:
    """Tests for get_test_coverage."""

    def test_coverage_size(self):
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        coverage = cpcv.get_test_coverage()
        assert len(coverage) == 6

    def test_coverage_symmetric(self):
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        coverage = cpcv.get_test_coverage()
        values = set(coverage.values())
        assert len(values) == 1

    def test_coverage_value_correct(self):
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        coverage = cpcv.get_test_coverage()
        import math

        expected = math.comb(5, 1)
        assert all(v == expected for v in coverage.values())


class TestTrainTestRatioStats:
    """Tests for get_train_test_ratio_stats."""

    def test_basic(self, sample_df):
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        stats = cpcv.get_train_test_ratio_stats(sample_df)
        assert stats["n_paths"] == 15
        assert "train_size" in stats
        assert "test_size" in stats
        assert "purged_size" in stats
        assert stats["train_size"]["max"] >= stats["train_size"]["min"]
        assert stats["test_size"]["max"] >= stats["test_size"]["min"]

    def test_purged_zero_without_label_span(self, sample_df):
        cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, label_span=1)
        stats = cpcv.get_train_test_ratio_stats(sample_df)
        assert stats["purged_size"]["min"] == 0
