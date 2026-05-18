"""Tests for D6: VAR + Granger Causality."""

import numpy as np
import pandas as pd
import pytest

from src.ml.var_granger import (
    VARModel,
    VARResult,
    GrangerCausalityTest,
    GrangerResult,
    CrossAssetLeadLag,
    LeadLagResult,
)


@pytest.fixture
def random_multivariate():
    """Generate random multivariate time series."""
    np.random.seed(42)
    T = 200
    K = 3
    data = np.random.randn(T, K) * 0.01
    data = np.cumsum(data, axis=0)
    return data + np.arange(T).reshape(-1, 1) * 0.0001


@pytest.fixture
def causal_data():
    """Generate data with known Granger causal relationship."""
    np.random.seed(42)
    T = 300
    x = np.random.randn(T) * 0.02
    y = np.zeros(T)
    for t in range(2, T):
        y[t] = 0.3 * y[t - 1] + 0.2 * y[t - 2] + 0.5 * x[t - 1] + np.random.randn() * 0.01
    return y, x


def test_var_fit(random_multivariate):
    """Basic VAR model fitting."""
    var = VARModel(maxlags=3)
    result = var.fit(random_multivariate, var_names=["A", "B", "C"])
    assert isinstance(result, VARResult)
    assert result.convergence
    assert result.aic < float("inf")


def test_var_fit_small_data():
    """VAR with very little data returns empty result or limited model."""
    var = VARModel(maxlags=10)
    data = np.random.randn(10, 2)
    result = var.fit(data)
    if result.convergence:
        assert result.nobs <= 10


def test_var_fit_dataframe(random_multivariate):
    """VAR from DataFrame."""
    var = VARModel(maxlags=3)
    df = pd.DataFrame(random_multivariate, columns=["A", "B", "C"])
    result = var.fit_dataframe(df)
    assert "aic" in result
    assert result["convergence"]


def test_var_forecast(random_multivariate):
    """VAR forecast produces output."""
    var = VARModel(maxlags=3)
    var.fit(random_multivariate)
    fc = var.forecast(steps=5)
    assert fc.shape[0] == 5
    assert fc.shape[1] == random_multivariate.shape[1]


def test_var_result_to_dict(random_multivariate):
    """VARResult serialization."""
    var = VARModel(maxlags=3)
    result = var.fit(random_multivariate, var_names=["X", "Y", "Z"])
    d = result.to_dict()
    assert "aic" in d
    assert "bic" in d
    assert d["n_variables"] == 3


def test_granger_random(random_multivariate):
    """Granger test on random data should not find causality."""
    gc = GrangerCausalityTest(maxlag=5, significance_level=0.05)
    result = gc.test(random_multivariate[:, 0], random_multivariate[:, 1])
    assert isinstance(result, GrangerResult)
    assert not result.significant_01


def test_granger_causal(causal_data):
    """Granger test on data with known causality."""
    y, x = causal_data
    gc = GrangerCausalityTest(maxlag=5, significance_level=0.05)
    result = gc.test(y, x)
    assert result.p_value < 0.05
    assert result.causality_direction != "NONE"


def test_granger_named(causal_data):
    """Named Granger test."""
    y, x = causal_data
    gc = GrangerCausalityTest(maxlag=5)
    result = gc.test_named(y, "Target", x, "Predictor")
    assert "Target" in result.causality_direction or "Predictor" in result.causality_direction


def test_granger_short_input():
    """Very short input returns empty result."""
    gc = GrangerCausalityTest(maxlag=10)
    result = gc.test(np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.6, 0.7]))
    assert result.p_value == 1.0


def test_cross_asset_lead_lag():
    """Cross-asset lead/lag analysis."""
    np.random.seed(42)
    n = 300
    prices = {
        "SPY": np.cumsum(np.random.randn(n) * 0.01 + 0.0005),
        "QQQ": np.cumsum(np.random.randn(n) * 0.01 + 0.0008),
        "GLD": np.cumsum(np.random.randn(n) * 0.01 + 0.0003),
    }

    ll = CrossAssetLeadLag(maxlag=3, significance_level=0.05)
    result = ll.analyze_basket(prices)
    assert isinstance(result, LeadLagResult)
    assert result.granger_matrix.shape == (3, 3)
    assert 0.0 <= result.causal_density <= 1.0
    assert len(result.lead_ranking) == 3


def test_lead_lag_dataframe():
    """Cross-asset analysis from DataFrame."""
    np.random.seed(42)
    n = 200
    idx = pd.date_range("2020-01-01", periods=n)
    df = pd.DataFrame(
        {
            "SPY": np.cumsum(np.random.randn(n) * 0.01 + 0.0005),
            "QQQ": np.cumsum(np.random.randn(n) * 0.01 + 0.0008),
        },
        index=idx,
    )

    ll = CrossAssetLeadLag(maxlag=3)
    result = ll.analyze_dataframe(df)
    assert result.granger_matrix.shape == (2, 2)


def test_lead_lag_to_dict():
    """LeadLagResult serialization."""
    np.random.seed(42)
    n = 200
    prices = {
        "A": np.cumsum(np.random.randn(n) * 0.01),
        "B": np.cumsum(np.random.randn(n) * 0.01),
    }
    ll = CrossAssetLeadLag(maxlag=3)
    result = ll.analyze_basket(prices)
    d = result.to_dict()
    assert "most_predictive" in d
    assert "causal_density" in d
    assert "lead_ranking" in d


def test_granger_result_to_dict(causal_data):
    """GrangerResult serialization."""
    y, x = causal_data
    gc = GrangerCausalityTest(maxlag=5)
    result = gc.test(y, x)
    d = result.to_dict()
    assert "p_value" in d
    assert "f_stat" in d
    assert "direction" in d
