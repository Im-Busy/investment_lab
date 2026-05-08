"""
Tests for HistoricalAnalogMatcher (FS2 - Historical Analog Matching)
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.historical_analog import HistoricalAnalogMatcher, AnalogMatchSet


def make_features(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create synthetic feature data with some repeating patterns."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    trend = np.linspace(0, 10, n)
    noise = np.random.randn(n, 4) * 0.5

    X = np.zeros((n, 4))
    X[:, 0] = np.sin(np.linspace(0, 4 * np.pi, n)) + noise[:, 0]  # Cyclical
    X[:, 1] = trend * 0.01 + noise[:, 1]  # Trending
    X[:, 2] = np.cos(np.linspace(0, 6 * np.pi, n)) + noise[:, 2]  # Cyclical 2
    X[:, 3] = noise[:, 3]  # Pure noise

    return pd.DataFrame(
        X,
        index=dates,
        columns=["feature_a", "feature_b", "feature_c", "feature_d"],
    )


def make_prices(n: int = 300, seed: int = 42) -> pd.Series:
    """Create synthetic price series."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    returns = np.random.randn(n) * 0.01
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices, index=dates, name="close")


class TestHistoricalAnalogMatcher:
    """Core tests for HistoricalAnalogMatcher."""

    def test_fit_basic(self):
        """Test basic fit operation."""
        features = make_features(n=300)
        matcher = HistoricalAnalogMatcher(k=10)
        matcher.fit(features)

        assert matcher.is_fitted
        assert matcher.fitted_n_ == 300
        assert matcher.knn_ is not None
        assert matcher.feature_index_ is not None

    def test_find_analogs_returns_analog_match_set(self):
        """Test find_analogs returns correct type."""
        features = make_features(n=300)
        matcher = HistoricalAnalogMatcher(k=5)
        matcher.fit(features)

        query = features.iloc[-5:]
        result = matcher.find_analogs(query)

        assert isinstance(result, AnalogMatchSet)
        assert len(result.analogs) >= 1

    def test_find_analogs_k_respected(self):
        """Test find_analogs returns at most k results."""
        features = make_features(n=300)
        matcher = HistoricalAnalogMatcher(k=3)
        matcher.fit(features)

        query = features.iloc[-10:]
        result = matcher.find_analogs(query)

        assert len(result.analogs) <= 3

    def test_find_analogs_each_has_date(self):
        """Test each analog has a valid date."""
        features = make_features(n=300)
        matcher = HistoricalAnalogMatcher(k=5)
        matcher.fit(features)

        query = features.iloc[-5:]
        result = matcher.find_analogs(query)

        for analog in result.analogs:
            assert isinstance(analog.analog_date, pd.Timestamp)
            assert analog.rank >= 1

    def test_find_analogs_distances_positive(self):
        """Test all analog distances are non-negative."""
        features = make_features(n=300)
        matcher = HistoricalAnalogMatcher(k=5)
        matcher.fit(features)

        query = features.iloc[-5:]
        result = matcher.find_analogs(query)

        for analog in result.analogs:
            assert analog.distance >= 0

    def test_get_analog_forward_returns(self):
        """Test forward return computation for analogs."""
        features = make_features(n=300)
        prices = make_prices(n=300)
        matcher = HistoricalAnalogMatcher(k=5)
        matcher.fit(features)

        query = features.iloc[-5:]
        matcher.find_analogs(query)

        result = matcher.get_analog_forward_returns(prices, horizons=[5, 10])

        assert hasattr(result, "avg_forward_return")
        assert hasattr(result, "median_forward_return")
        assert hasattr(result, "win_rate")
        assert 5 in result.avg_forward_return
        assert 10 in result.avg_forward_return

        for analog in result.analogs:
            assert 5 in analog.forward_returns
            assert 10 in analog.forward_returns

    def test_win_rate_between_zero_and_one(self):
        """Test win rate is in [0, 1]."""
        features = make_features(n=300)
        prices = make_prices(n=300)
        matcher = HistoricalAnalogMatcher(k=10)
        matcher.fit(features)

        query = features.iloc[-5:]
        matcher.find_analogs(query)
        result = matcher.get_analog_forward_returns(prices, horizons=[5, 10, 20])

        for h in [5, 10, 20]:
            wr = result.win_rate[h]
            assert 0 <= wr <= 1

    def test_predict_before_fit_raises(self):
        """Test find_analogs before fit raises error."""
        features = make_features(n=100)
        matcher = HistoricalAnalogMatcher(k=5)

        with pytest.raises(ValueError, match="not fitted"):
            matcher.find_analogs(features.iloc[:5])

    def test_get_analog_forward_returns_before_find_raises(self):
        """Test forward returns before find_analogs raises error."""
        features = make_features(n=100)
        prices = make_prices(n=100)
        matcher = HistoricalAnalogMatcher(k=5)
        matcher.fit(features)

        with pytest.raises(ValueError, match="No analogs found"):
            matcher.get_analog_forward_returns(prices)

    def test_empty_data_raises(self):
        """Test empty data raises ValueError."""
        matcher = HistoricalAnalogMatcher(k=5)
        with pytest.raises(ValueError, match="empty"):
            matcher.fit(pd.DataFrame())

    def test_insufficient_samples_raises(self):
        """Test too few samples raises error."""
        features = make_features(n=3)
        matcher = HistoricalAnalogMatcher(k=10)
        with pytest.raises(ValueError, match="Insufficient samples"):
            matcher.fit(features)

    def test_nan_data_raises(self):
        """Test NaN in data raises error."""
        features = make_features(n=100)
        features.iloc[10, 1] = np.nan
        matcher = HistoricalAnalogMatcher(k=5)
        with pytest.raises(ValueError, match="NaN"):
            matcher.fit(features)

    def test_nan_query_raises(self):
        """Test NaN in query raises error."""
        features = make_features(n=200)
        matcher = HistoricalAnalogMatcher(k=5)
        matcher.fit(features)

        query = features.iloc[:5].copy()
        query.iloc[2, 0] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            matcher.find_analogs(query)

    def test_invalid_k_raises(self):
        """Test invalid k raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            HistoricalAnalogMatcher(k=0)

    def test_cosine_metric(self):
        """Test cosine distance metric works."""
        features = make_features(n=200)
        matcher = HistoricalAnalogMatcher(k=5, metric="cosine")
        matcher.fit(features)

        query = features.iloc[-5:]
        result = matcher.find_analogs(query)
        assert len(result.analogs) >= 1

    def test_get_analog_distribution(self):
        """Test analog distribution DataFrame."""
        features = make_features(n=200)
        matcher = HistoricalAnalogMatcher(k=5)
        matcher.fit(features)

        query = features.iloc[-5:]
        matcher.find_analogs(query)
        dist = matcher.get_analog_distribution()

        assert "distance" in dist.columns
        assert "analog_date" in dist.columns
        assert len(dist) >= 1

    def test_get_analog_distribution_before_find_raises(self):
        """Test distribution before find_analogs raises."""
        features = make_features(n=100)
        matcher = HistoricalAnalogMatcher(k=5)
        matcher.fit(features)

        with pytest.raises(ValueError, match="No analogs found"):
            matcher.get_analog_distribution()

    def test_get_analog_consensus(self):
        """Test consensus forecast from analogs."""
        features = make_features(n=300)
        prices = make_prices(n=300)
        matcher = HistoricalAnalogMatcher(k=10)
        matcher.fit(features)

        query = features.iloc[-5:]
        matcher.find_analogs(query)
        consensus = matcher.get_analog_consensus(prices, horizons=[5, 10, 20])

        assert "avg_return" in consensus.columns
        assert "median_return" in consensus.columns
        assert "win_rate" in consensus.columns
        assert "n_analogs" in consensus.columns
        assert 5 in consensus.index
        assert 10 in consensus.index
        assert 20 in consensus.index

    def test_closest_analog_min_distance(self):
        """Test that the first analog has the smallest distance."""
        features = make_features(n=300)
        matcher = HistoricalAnalogMatcher(k=5)
        matcher.fit(features)

        query = features.iloc[-5:]
        result = matcher.find_analogs(query)

        if len(result.analogs) >= 2:
            for i in range(len(result.analogs) - 1):
                assert result.analogs[i].distance <= result.analogs[i + 1].distance

    def test_find_sequential_analogs(self):
        """Test sequential analog search across time."""
        features = make_features(n=300)
        prices = make_prices(n=300)
        matcher = HistoricalAnalogMatcher(k=3)
        matcher.fit(features.iloc[:200])

        result = matcher.find_sequential_analogs(
            prices.iloc[:200],
            features.iloc[:200],
            window_size=5,
            step=10,
            k=2,
        )

        assert isinstance(result, pd.DataFrame)
        assert "query_date" in result.columns
        assert "analog_date" in result.columns
        assert "distance" in result.columns
        assert "forward_return_10" in result.columns
        assert len(result) > 0

    def test_various_query_sizes(self):
        """Test different query sizes work."""
        features = make_features(n=200)
        matcher = HistoricalAnalogMatcher(k=5)
        matcher.fit(features)

        for ws in [1, 5, 10]:
            query = features.iloc[-ws:]
            result = matcher.find_analogs(query)
            assert len(result.analogs) >= 1
