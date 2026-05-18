"""Tests for D9: Kalman Hedge Ratios."""

import numpy as np
import pandas as pd
import pytest

from src.ml.kalman_hedge import (
    KalmanHedgeEstimator,
    KalmanHedgePair,
    KalmanHedgeResult,
    PortfolioHedgeEstimator,
)


@pytest.fixture
def random_data():
    """Generate correlated pair data."""
    np.random.seed(42)
    n = 200
    x = np.cumsum(np.random.randn(n) * 0.01 + 0.0005)
    y = 0.6 * x + np.random.randn(n) * 0.005
    return y, x


def test_estimator_fit(random_data):
    """Basic Kalman hedge ratio estimation."""
    y, x = random_data
    est = KalmanHedgeEstimator(delta=0.98)
    result = est.fit(y, x)
    assert isinstance(result, KalmanHedgeResult)
    assert len(result.beta) == len(y)
    assert len(result.alpha) == len(y)


def test_beta_convergence():
    """Hedge ratio converges to true beta with enough data."""
    np.random.seed(42)
    n = 500
    true_beta = 0.7
    x = np.cumsum(np.random.randn(n) * 0.01 + 0.0005)
    y = true_beta * x + np.random.randn(n) * 0.003

    est = KalmanHedgeEstimator(delta=0.98, init_beta=1.0)
    result = est.fit(y, x)

    late_beta = np.nanmean(result.beta[-100:])
    assert abs(late_beta - true_beta) < 0.2


def test_empty_input():
    """Handle empty/short input gracefully."""
    est = KalmanHedgeEstimator()
    result = est.fit(np.array([]), np.array([]))
    assert result.terminal_beta == 1.0


def test_short_input():
    """Handle very short input."""
    est = KalmanHedgeEstimator()
    result = est.fit(np.array([1.0, 2.0]), np.array([0.5, 0.6]))
    assert len(result.beta) > 0


def test_nan_handling(random_data):
    """Handle NaN in input series."""
    y, x = random_data
    y[50] = np.nan
    x[100] = np.nan
    est = KalmanHedgeEstimator()
    result = est.fit(y, x)
    assert len(result.beta) == len(y)
    assert np.any(np.isnan(result.beta))


def test_variance_reduction(random_data):
    """Variance reduction from hedging."""
    y, x = random_data
    pair = KalmanHedgePair()
    pair.fit(y, x)
    vr = pair.variance_reduction()
    assert 0.0 <= vr <= 1.0


def test_hedge_correlation(random_data):
    """Hedge correlation computation."""
    y, x = random_data
    pair = KalmanHedgePair()
    pair.fit(y, x)
    corr = pair.hedge_correlation()
    assert -1.0 <= corr <= 1.0


def test_compare_methods(random_data):
    """Compare Kalman vs rolling OLS vs static."""
    y, x = random_data
    pair = KalmanHedgePair()
    pair.fit(y, x)
    df = pair.compare_methods(lookback=60)
    assert "kalman" in df.columns
    assert "rolling_60" in df.columns
    assert "static" in df.columns


def test_hedge_pnl(random_data):
    """Hedged P&L computation."""
    y, x = random_data
    prices_y = np.cumprod(1 + y)
    prices_x = np.cumprod(1 + x)
    pair = KalmanHedgePair()
    df = pair.hedge_pnl(prices_y, prices_x)
    assert "unhedged_cum" in df.columns
    assert "hedged_cum" in df.columns


def test_fit_dataframe(random_data):
    """Fit from pandas Series."""
    y, x = random_data
    idx = pd.date_range("2020-01-01", periods=len(y))
    pair = KalmanHedgePair()
    df = pair.fit_dataframe(pd.Series(y, index=idx), pd.Series(x, index=idx))
    assert "beta" in df.columns
    assert len(df) == len(y)


def test_result_to_dict(random_data):
    """Result serialization."""
    y, x = random_data
    est = KalmanHedgeEstimator()
    result = est.fit(y, x)
    d = result.to_dict()
    assert "mean_beta" in d
    assert "variance_reduction" in d


def test_portfolio_hedge():
    """Multi-asset portfolio hedge ratios."""
    np.random.seed(42)
    T, N = 200, 3
    returns = np.random.randn(T, N) * 0.01
    hedge = np.random.randn(T) * 0.01
    est = PortfolioHedgeEstimator(delta=0.98)
    df = est.fit(returns, hedge)
    assert df.shape == (T, N)
    assert len(est.results) == N
