"""P28-18: skfolio portfolio optimization wrapper.

Scikit-learn compatible portfolio optimization via skfolio — mean-variance,
CVaR, hierarchical risk parity, and shrinkage estimators. Augments or
replaces existing position sizing logic.

Source: awesome-ai-in-finance — skfolio provides sklearn-compatible
portfolio optimization with robust covariance estimators.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import skfolio
from skfolio import (
    PerfMeasure,
    Population,
    Portfolio,
    RatioMeasure,
    RiskMeasure,
)
from skfolio.optimization import (
    EqualWeighted,
    InverseVolatility,
    MeanRisk,
    NestedClustersOptimization,
    ObjectiveFunction,
    RiskBudgeting,
)
from skfolio.preprocessing import prices_to_returns

logger = logging.getLogger(__name__)


@dataclass
class OptWeights:
    """Optimized portfolio weights result."""

    weights: dict[str, float]
    method: str
    expected_return: float
    expected_volatility: float
    expected_sharpe: float
    concentration: float


@dataclass
class OptComparison:
    """Side-by-side comparison of multiple optimization methods."""

    methods: list[OptWeights]
    best_by_sharpe: OptWeights
    best_by_diversification: OptWeights
    equal_weight_sharpe: float


def optimize_mean_variance(
    prices: pd.DataFrame,
    max_weight: float = 0.30,
    min_weight: float = 0.01,
) -> OptWeights:
    """Classic mean-variance optimization with weight constraints.

    Args:
        prices: DataFrame with ticker columns, DateTimeIndex, daily prices.
        max_weight: Maximum allocation per instrument (0-1).
        min_weight: Minimum allocation per instrument (0-1).

    Returns:
        OptWeights with optimized allocations.
    """
    returns = prices_to_returns(prices)
    model = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        max_weights=max_weight,
        min_weights=min_weight,
        portfolio_params={"name": "Mean-Variance"},
    )
    model.fit(returns)
    portfolio = model.predict(returns)

    w = _extract_weights(portfolio, prices.columns)
    return OptWeights(
        weights=w,
        method="mean_variance",
        expected_return=float(portfolio.annualized_mean),
        expected_volatility=float(portfolio.annualized_standard_deviation),
        expected_sharpe=float(portfolio.annualized_sharpe_ratio),
        concentration=_concentration(w),
    )


def optimize_cvar(
    prices: pd.DataFrame,
    max_weight: float = 0.30,
    min_weight: float = 0.01,
    confidence_level: float = 0.95,
) -> OptWeights:
    """CVaR (Expected Shortfall) minimization.

    More robust than mean-variance for fat-tailed return distributions.
    """
    returns = prices_to_returns(prices)
    model = MeanRisk(
        risk_measure=RiskMeasure.CVAR,
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        max_weights=max_weight,
        min_weights=min_weight,
        cvar_beta=confidence_level,
        portfolio_params={"name": "CVaR"},
    )
    model.fit(returns)
    portfolio = model.predict(returns)

    w = _extract_weights(portfolio, prices.columns)
    return OptWeights(
        weights=w,
        method=f"cvar_{int(confidence_level * 100)}",
        expected_return=float(portfolio.annualized_mean),
        expected_volatility=float(portfolio.annualized_standard_deviation),
        expected_sharpe=float(portfolio.annualized_sharpe_ratio),
        concentration=_concentration(w),
    )


def optimize_hrp(prices: pd.DataFrame) -> OptWeights:
    """Hierarchical Risk Parity (HRP) using nested clustering.

    No return estimates needed — purely covariance-based. Robust to
    estimation error in return forecasts.
    """
    returns = prices_to_returns(prices)
    model = NestedClustersOptimization(
        portfolio_params={"name": "HRP"},
    )
    model.fit(returns)
    portfolio = model.predict(returns)

    w = _extract_weights(portfolio, prices.columns)
    return OptWeights(
        weights=w,
        method="hrp",
        expected_return=float(portfolio.annualized_mean),
        expected_volatility=float(portfolio.annualized_standard_deviation),
        expected_sharpe=float(portfolio.annualized_sharpe_ratio),
        concentration=_concentration(w),
    )


def optimize_risk_budgeting(
    prices: pd.DataFrame,
    max_weight: float = 0.30,
    min_weight: float = 0.01,
) -> OptWeights:
    """Risk Budgeting (Equal Risk Contribution / ERC).

    Each instrument contributes equally to portfolio risk. Better
    diversification than equal-weight without requiring return forecasts.
    """
    returns = prices_to_returns(prices)
    model = RiskBudgeting(
        risk_measure=RiskMeasure.VARIANCE,
        max_weights=max_weight,
        min_weights=min_weight,
        portfolio_params={"name": "RiskBudget"},
    )
    model.fit(returns)
    portfolio = model.predict(returns)

    w = _extract_weights(portfolio, prices.columns)
    return OptWeights(
        weights=w,
        method="risk_budgeting",
        expected_return=float(portfolio.annualized_mean),
        expected_volatility=float(portfolio.annualized_standard_deviation),
        expected_sharpe=float(portfolio.annualized_sharpe_ratio),
        concentration=_concentration(w),
    )


def optimize_inverse_vol(prices: pd.DataFrame) -> OptWeights:
    """Inverse volatility weighting — simplest robust allocation."""
    returns = prices_to_returns(prices)
    model = InverseVolatility(portfolio_params={"name": "InvVol"})
    model.fit(returns)
    portfolio = model.predict(returns)

    w = _extract_weights(portfolio, prices.columns)
    return OptWeights(
        weights=w,
        method="inverse_vol",
        expected_return=float(portfolio.annualized_mean),
        expected_volatility=float(portfolio.annualized_standard_deviation),
        expected_sharpe=float(portfolio.annualized_sharpe_ratio),
        concentration=_concentration(w),
    )


def compare_methods(
    prices: pd.DataFrame,
    max_weight: float = 0.30,
    min_weight: float = 0.01,
) -> OptComparison:
    """Run all optimization methods and compare side-by-side.

    Args:
        prices: DataFrame with ticker columns, DateTimeIndex, daily prices.
        max_weight: Maximum allocation per instrument.
        min_weight: Minimum allocation per instrument.

    Returns:
        OptComparison with all results ranked by Sharpe and diversification.
    """
    methods = [
        optimize_mean_variance(prices, max_weight, min_weight),
        optimize_cvar(prices, max_weight, min_weight),
        optimize_hrp(prices),
        optimize_risk_budgeting(prices, max_weight, min_weight),
        optimize_inverse_vol(prices),
    ]

    equal_w = {c: 1.0 / len(prices.columns) for c in prices.columns}
    returns = prices_to_returns(prices)
    eq_ret = (returns @ pd.Series(equal_w)).mean() * 252
    eq_vol = (returns @ pd.Series(equal_w)).std() * np.sqrt(252)

    return OptComparison(
        methods=methods,
        best_by_sharpe=max(methods, key=lambda m: m.expected_sharpe),
        best_by_diversification=min(methods, key=lambda m: m.concentration),
        equal_weight_sharpe=float(eq_ret / eq_vol) if eq_vol > 0 else 0.0,
    )


def weights_to_dataframe(methods: list[OptWeights], columns: list[str]) -> pd.DataFrame:
    """Convert optimization results to a comparison DataFrame.

    Args:
        methods: List of OptWeights from different optimization methods.
        columns: Ticker names for the weight columns.

    Returns:
        DataFrame with one row per method, columns for each ticker weight.
    """
    rows = []
    for m in methods:
        row = {
            "method": m.method,
            "sharpe": round(m.expected_sharpe, 3),
            "return_pct": round(m.expected_return * 100, 2),
            "vol_pct": round(m.expected_volatility * 100, 2),
            "concentration": round(m.concentration, 3),
        }
        for col in columns:
            row[col] = round(m.weights.get(col, 0.0), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def _extract_weights(
    portfolio: Portfolio,
    columns: pd.Index,
) -> dict[str, float]:
    """Extract clean weight dict from skfolio portfolio."""
    w = dict(zip(columns, portfolio.weights))
    return {k: float(v) for k, v in w.items() if v > 0.0001}


def _concentration(weights: dict[str, float]) -> float:
    """Herfindahl-Hirschman Index for portfolio concentration.

    0 = perfectly diversified, 1 = single asset.
    """
    w = np.array(list(weights.values()))
    w = w / w.sum() if w.sum() > 0 else w
    return float((w**2).sum())
