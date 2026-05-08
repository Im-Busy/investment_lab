"""Unit tests for ML signal evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from src.ml.metrics import (
    compute_ic,
    compute_rank_ic,
    compute_ic_decay,
    compute_hit_rate,
    compute_information_ratio,
    ic_summary,
    filter_features_by_ic,
)


@pytest.fixture
def correlated_data():
    """Data with known correlation structure."""
    np.random.seed(42)
    n = 200
    returns = np.random.randn(n) * 0.02
    feature_good = returns + np.random.randn(n) * 0.01
    feature_noise = np.random.randn(n)
    return pd.DataFrame(
        {
            "good_feature": feature_good,
            "noise_feature": feature_noise,
        }
    ), pd.Series(returns, name="forward_return")


class TestComputeIC:
    def test_positive_correlation(self, correlated_data):
        features, returns = correlated_data
        result = compute_ic(features, returns)
        good_row = result[result["feature"] == "good_feature"].iloc[0]
        noise_row = result[result["feature"] == "noise_feature"].iloc[0]
        assert good_row["ic"] > noise_row["ic"]
        assert result["n_samples"].iloc[0] == 200

    def test_single_series_input(self, correlated_data):
        _, returns = correlated_data
        series = pd.Series(np.random.randn(200))
        result = compute_ic(series, returns)
        assert len(result) == 1

    def test_insufficient_data_returns_empty(self):
        values = pd.DataFrame({"a": [1.0, 2.0]})
        returns = pd.Series([0.01, 0.02])
        result = compute_ic(values, returns)
        assert len(result) == 0


class TestComputeRankIC:
    def test_rank_ic_positive(self, correlated_data):
        features, returns = correlated_data
        result = compute_rank_ic(features, returns)
        good_row = result[result["feature"] == "good_feature"].iloc[0]
        assert good_row["rank_ic"] > 0

    def test_rank_ic_bounds(self, correlated_data):
        features, returns = correlated_data
        result = compute_rank_ic(features, returns)
        for _, row in result.iterrows():
            assert -1.0 <= row["rank_ic"] <= 1.0

    def test_insufficient_data(self):
        values = pd.DataFrame({"a": [1.0, 2.0]})
        returns = pd.Series([0.01, 0.02])
        result = compute_rank_ic(values, returns)
        assert len(result) == 0


class TestComputeICDecay:
    def test_decay_over_horizons(self, correlated_data):
        features, returns = correlated_data
        decay = compute_ic_decay(features["good_feature"], returns, horizons=[1, 5, 10])
        assert len(decay) == 3
        assert decay["horizon"].tolist() == [1, 5, 10]

    def test_decay_ratio_computed(self, correlated_data):
        features, returns = correlated_data
        decay = compute_ic_decay(features["good_feature"], returns, horizons=[1, 5, 10, 20])
        assert "decay_ratio" in decay.columns
        assert decay["decay_ratio"].notna().any()

    def test_dataframe_returns(self, correlated_data):
        features, returns = correlated_data
        returns_df = returns.to_frame("ret")
        decay = compute_ic_decay(features["good_feature"], returns_df)
        assert len(decay) > 0


class TestComputeHitRate:
    def test_perfect_hits(self):
        pred = np.array([1.0, -1.0, 1.0, -1.0])
        actual = np.array([0.02, -0.01, 0.03, -0.02])
        rate = compute_hit_rate(pred, actual)
        assert rate == 1.0

    def test_no_hits(self):
        pred = np.array([1.0, 1.0, -1.0])
        actual = np.array([-0.02, -0.01, 0.03])
        rate = compute_hit_rate(pred, actual)
        assert rate == 0.0

    def test_mixed_hits(self):
        pred = np.array([1.0, 1.0, -1.0, -1.0])
        actual = np.array([0.02, -0.01, 0.01, -0.02])
        rate = compute_hit_rate(pred, actual)
        assert 0.4 <= rate <= 0.6

    def test_below_threshold_ignored(self):
        pred = np.array([1.0, 1.0])
        actual = np.array([0.0001, -0.0001])
        rate = compute_hit_rate(pred, actual, threshold=0.001)
        assert rate == 0.0

    def test_empty_mask_returns_zero(self):
        pred = np.array([0.0, 0.0])
        actual = np.array([0.0, 0.0])
        rate = compute_hit_rate(pred, actual, threshold=0.01)
        assert rate == 0.0


class TestComputeInformationRatio:
    def test_positive_ir(self):
        ic = [0.05, 0.06, 0.04, 0.07, 0.05]
        ir = compute_information_ratio(ic)
        assert ir > 0

    def test_zero_ir_for_single_value(self):
        ir = compute_information_ratio([0.05])
        assert ir == 0.0

    def test_annualization(self):
        ic = [0.05, 0.06, 0.04, 0.07, 0.03, 0.05, 0.06, 0.04, 0.05, 0.06]
        daily = compute_information_ratio(ic, periods_per_year=252)
        monthly = compute_information_ratio(ic, periods_per_year=21)
        assert daily > monthly > 0


class TestICSummary:
    def test_summary_columns(self, correlated_data):
        features, returns = correlated_data
        summary = ic_summary(features, returns)
        expected = {
            "feature",
            "ic",
            "rank_ic",
            "abs_ic",
            "abs_rank_ic",
            "t_stat",
            "p_value",
            "hit_rate",
            "n_samples",
            "significant",
        }
        assert expected.issubset(set(summary.columns))

    def test_ranking_by_abs_rank_ic(self, correlated_data):
        features, returns = correlated_data
        summary = ic_summary(features, returns)
        assert summary["abs_rank_ic"].is_monotonic_decreasing

    def test_empty_data(self):
        values = pd.DataFrame()
        returns = pd.Series(dtype=float)
        summary = ic_summary(values, returns)
        assert summary.empty


class TestFilterFeaturesByIC:
    def test_filters_noise(self, correlated_data):
        features, returns = correlated_data
        selected = filter_features_by_ic(features, returns, min_abs_ic=0.5, min_abs_rank_ic=0.5)
        assert "noise_feature" not in selected
        assert "good_feature" in selected

    def test_all_pass_low_threshold(self, correlated_data):
        features, returns = correlated_data
        selected = filter_features_by_ic(features, returns, min_abs_ic=0.0, min_abs_rank_ic=0.0)
        assert len(selected) == 2

    def test_empty_if_none_pass(self, correlated_data):
        features, returns = correlated_data
        selected = filter_features_by_ic(features, returns, min_abs_ic=1.0, min_abs_rank_ic=1.0)
        assert len(selected) == 0
