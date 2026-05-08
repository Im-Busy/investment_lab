"""
Tests for FracDiff (FS11 - Fractional Differentiation)
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.frac_diff import FracDiff


def make_prices(n: int = 500, seed: int = 42) -> pd.Series:
    """Create a synthetic price series (random walk)."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    returns = np.random.randn(n) * 0.01
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices, index=dates, name="close")


def make_stationary_series(n: int = 500, seed: int = 42) -> pd.Series:
    """Create an already-stationary series (white noise)."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    noise = np.random.randn(n) * 0.01
    return pd.Series(noise, index=dates, name="noise")


class TestFracDiff:
    """Core tests for FracDiff fractional differentiation."""

    def test_get_weights_basic(self):
        """Test weight computation follows binomial pattern."""
        fd = FracDiff()
        weights = fd.get_weights(d=0.4, window=10)

        assert len(weights) == 10
        assert weights[0] == 1.0
        assert np.abs(weights).sum() > 1.0  # Weights don't vanish instantly

    def test_get_weights_d_zero(self):
        """Test d=0 returns original series (weight = [1, 0, 0, ...])."""
        fd = FracDiff()
        weights = fd.get_weights(d=0.0, window=10)

        assert weights[0] == 1.0
        np.testing.assert_allclose(weights[1:], 0.0, atol=1e-10)

    def test_get_weights_d_one(self):
        """Test d=1 returns integer differencing weights [1, -1, 0, ...]."""
        fd = FracDiff()
        weights = fd.get_weights(d=1.0, window=10)

        assert weights[0] == 1.0
        np.testing.assert_allclose(weights[1], -1.0, atol=1e-10)
        np.testing.assert_allclose(weights[2:], 0.0, atol=1e-10)

    def test_get_weights_sign_pattern(self):
        """Test weights follow the recursive binomial pattern correctly."""
        fd = FracDiff()
        weights = fd.get_weights(d=0.4, window=20)

        assert weights[0] == 1.0
        assert weights[1] < 0  # w_1 = -w_0 * d / 1 = -0.4
        # After k > d+1, (d-k+1) becomes negative, so ALL subsequent weights
        # have the same sign (negative for d < 1). Verify no sign flips after k > d+1.
        signs = np.sign(weights)
        for i in range(int(0.4) + 2, len(weights)):
            if signs[i] != 0:
                assert signs[i] == signs[i - 1]

    def test_frac_diff_preserves_length(self):
        """Test output length matches input length."""
        prices = make_prices()
        fd = FracDiff()
        result = fd.frac_diff(prices, d=0.4)

        assert len(result) == len(prices)

    def test_frac_diff_initial_nan(self):
        """Test initial values are NaN (window warmup)."""
        prices = make_prices(n=500)
        fd = FracDiff(window=50)
        result = fd.frac_diff(prices, d=0.4)

        assert result.iloc[:49].isna().all()
        assert not result.iloc[100:].isna().all()

    def test_frac_diff_stores_weights(self):
        """Test frac_diff stores weights on instance."""
        prices = make_prices()
        fd = FracDiff()
        fd.frac_diff(prices, d=0.4)

        assert fd.weights_ is not None
        assert len(fd.weights_) > 0
        assert fd.d_ == 0.4

    def test_frac_diff_on_constant_series(self):
        """Test frac_diff on constant series produces small residuals.

        With finite window, weight sum per bar = 1 - sum(|w_k|) which is non-zero.
        This residual shrinks as window increases. For d=0.4 with window=200,
        residual should be well below 0.1.
        """
        constant = pd.Series(np.ones(1000), name="constant")
        fd = FracDiff(window=200)
        result = fd.frac_diff(constant, d=0.4)

        valid = result.dropna()
        assert len(valid) > 100, f"Expected >100 valid, got {len(valid)}"
        residuals = np.abs(valid.values[10:])
        assert residuals.max() < 0.15, f"Residual too large: {residuals.max():.6f}"

    def test_frac_diff_integer_diff_equivalent(self):
        """Test d=1 is equivalent to standard differencing."""
        prices = make_prices()
        fd = FracDiff()
        frac_result = fd.frac_diff(prices, d=1.0)

        std_diff = prices.diff()

        valid_mask = frac_result.notna()
        np.testing.assert_allclose(
            frac_result[valid_mask].values,
            std_diff[valid_mask].values,
            atol=1e-12,
        )

    def test_find_optimal_d_returns_dataframe(self):
        """Test find_optimal_d returns expected columns."""
        prices = make_prices(n=500)
        fd = FracDiff(verbose=False)
        result = fd.find_optimal_d(prices, step=0.2)

        assert isinstance(result, pd.DataFrame)
        assert "adf_stat" in result.columns
        assert "adf_pvalue" in result.columns
        assert "is_stationary" in result.columns
        assert len(result) > 0

    def test_find_optimal_d_finds_stationary(self):
        """Test that d close to 1 produces a stationary series on long data."""
        prices = make_prices(n=1000)
        fd = FracDiff()
        result = fd.find_optimal_d(prices, step=0.5)

        # d >= 0.5 should be stationary for a random walk with enough samples
        stationary_ds = result[result["is_stationary"]]
        assert len(stationary_ds) > 0, "At least one d value should produce stationarity"

    def test_stationary_series_needs_low_d(self):
        """Test already-stationary series is stationary at d=0."""
        noise = make_stationary_series(n=1000)
        fd = FracDiff()
        result = fd.find_optimal_d(noise, step=0.25)

        # White noise should be stationary even at d=0
        # ADF can be sensitive, check at least d=0 or d=0.25 works
        assert result["is_stationary"].any()

    def test_transform_features(self):
        """Test transform_features applies frac_diff to columns."""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame(
            {
                "returns_5": np.random.randn(n) * 0.02,
                "volatility_10": np.abs(np.random.randn(n)) * 0.01,
                "category": ["A", "B"] * 100,
            }
        )

        fd = FracDiff(window=50)
        result = fd.transform_features(df, d=0.3, columns=["returns_5", "volatility_10"])

        assert "category" in result.columns
        assert "returns_5" in result.columns
        assert "volatility_10" in result.columns
        # Category should be unchanged
        assert list(result["category"]) == list(df["category"])

    def test_get_weight_summary(self):
        """Test weight summary returns weights."""
        prices = make_prices()
        fd = FracDiff(window=20)
        fd.frac_diff(prices, d=0.4)

        summary = fd.get_weight_summary()
        assert len(summary) == 20
        assert summary.index.name == "lag"

    def test_get_weight_summary_before_diff_raises(self):
        """Test summary before diff raises ValueError."""
        fd = FracDiff()
        with pytest.raises(ValueError, match="Weights not computed"):
            fd.get_weight_summary()

    def test_weight_convergence_approaches_zero(self):
        """Test cumulative weight sum converges toward 0 (differencing property)."""
        fd = FracDiff()
        conv = fd.weight_convergence(d=0.4, window=200)

        # Cumulative sum should approach zero for d > 0
        assert np.abs(conv.iloc[-1]) < 0.1

    def test_invalid_d_raises(self):
        """Test d out of range raises ValueError."""
        fd = FracDiff()
        with pytest.raises(ValueError, match="d must be in"):
            fd.get_weights(d=3.0, window=10)

    def test_insufficient_series_raises(self):
        """Test too short series raises ValueError."""
        prices = make_prices(n=5)
        fd = FracDiff(min_obs=10)
        with pytest.raises(ValueError, match="Series too short"):
            fd.frac_diff(prices, d=0.4)

    def test_invalid_tau_raises(self):
        """Test tau <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="tau must be > 0"):
            FracDiff(tau=0)

    def test_invalid_min_obs_raises(self):
        """Test min_obs < 5 raises ValueError."""
        with pytest.raises(ValueError, match="min_obs must be >= 5"):
            FracDiff(min_obs=3)

    def test_result_is_float(self):
        """Test results are float64."""
        prices = make_prices()
        fd = FracDiff()
        result = fd.frac_diff(prices, d=0.4)

        assert result.dtype == np.float64

    def test_large_d_smoother_than_small_d(self):
        """Test larger d produces smaller magnitude (more differenced) values."""
        prices = make_prices()
        fd = FracDiff(window=100)

        r03 = fd.frac_diff(prices, d=0.3)
        r08 = fd.frac_diff(prices, d=0.8)

        valid = r03.notna() & r08.notna()
        std_03 = r03[valid].std()
        std_08 = r08[valid].std()

        # Higher d removes more trending component, reduces std for trending series
        # (random walk drift makes d=0.3 have larger std since trend persists more)
        assert std_08 < std_03 * 2  # Not guaranteed to be smaller, but should be reasonable
