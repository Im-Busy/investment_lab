"""PyPortfolioOpt integration — extends portfolio_optimizer with HRP, CVaR, and efficient frontier.

Wraps PyPortfolioOpt methods to complement the existing scipy-based optimizer and
Black-Litterman implementation in src/portfolio/black_litterman.py.

Usage:
    from src.optimizer.pypfopt_integration import (
        optimize_hrp, optimize_cvar, optimize_efficient_frontier, compare_methods
    )
    weights = optimize_hrp(returns_df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AllocationResult:
    """Result from PyPortfolioOpt allocation."""

    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    method: str = ""


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Drop tickers with NaN/zero variance and fill remaining NaN with 0."""
    df = returns.dropna(axis=1, how="all").fillna(0.0)
    valid = df.columns[df.std() > 1e-10]
    if len(valid) < len(df.columns):
        logger.debug(f"Dropped {len(df.columns) - len(valid)} zero-variance tickers")
    return df[list(valid)]


def optimize_hrp(
    returns: pd.DataFrame,
    frequency: int = 252,
) -> AllocationResult:
    """Hierarchical Risk Parity allocation using PyPortfolioOpt.

    HRP uses hierarchical clustering on the correlation matrix to build a
    tree, then allocates weights from the leaves up via inverse-variance.
    Robust to estimation errors — no expected returns needed.

    Use when: want risk-balanced allocation without return forecasting errors.
    """
    from pypfopt.hierarchical_portfolio import HRPOpt

    returns_clean = _clean_returns(returns)
    if returns_clean.empty or returns_clean.shape[1] < 2:
        return AllocationResult(
            weights=pd.Series(dtype=float),
            expected_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            method="hrp",
        )

    hrp = HRPOpt(returns_clean)
    weights_dict = hrp.optimize()
    weights = pd.Series(weights_dict)
    port_returns = returns_clean.dot(weights)
    ann_ret = port_returns.mean() * frequency
    ann_vol = port_returns.std() * np.sqrt(frequency)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    return AllocationResult(
        weights=weights,
        expected_return=ann_ret,
        volatility=ann_vol,
        sharpe_ratio=sharpe,
        method="hrp",
    )


def optimize_efficient_frontier(
    returns: pd.DataFrame,
    objective: str = "max_sharpe",
    long_only: bool = True,
    frequency: int = 252,
) -> AllocationResult:
    """Mean-variance efficient frontier optimization via PyPortfolioOpt.

    Args:
        objective: 'max_sharpe', 'min_volatility', or 'max_return'.
        long_only: If True, weights >= 0.
        frequency: Periods per year for annualization.

    Use when: want classic Markowitz optimization with proper constraints.
    """
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.expected_returns import mean_historical_return
    from pypfopt.risk_models import sample_cov

    returns_clean = _clean_returns(returns)
    if returns_clean.empty or returns_clean.shape[1] < 2:
        return AllocationResult(
            weights=pd.Series(dtype=float),
            expected_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            method="efficient_frontier",
        )

    mu = mean_historical_return(returns_clean, frequency=frequency)
    mu = mu.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    S = sample_cov(returns_clean, frequency=frequency)
    S = S.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    try:
        ef = EfficientFrontier(mu, S, weight_bounds=(0 if long_only else -1, 1))
        ef.tickers = list(returns_clean.columns)

        if objective == "max_sharpe":
            ef.max_sharpe()
        elif objective == "min_volatility":
            ef.min_volatility()
        elif objective == "max_return":
            ef.efficient_return(float(max(mu)))
        else:
            raise ValueError(f"Unknown objective: {objective}")

        cleaned = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=False)
    except (ValueError, Exception) as exc:
        logger.debug(f"EfficientFrontier optimization failed: {exc}")
        return AllocationResult(
            weights=pd.Series(0.0, index=returns_clean.columns),
            expected_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            method=f"efficient_frontier_{objective}",
        )

    perf_list = list(perf) if isinstance(perf, tuple) else [perf, 0.0, 0.0]
    return AllocationResult(
        weights=pd.Series(cleaned),
        expected_return=float(perf_list[0]) if len(perf_list) > 0 else 0.0,
        volatility=float(perf_list[1]) if len(perf_list) > 1 else 0.0,
        sharpe_ratio=float(perf_list[2]) if len(perf_list) > 2 else 0.0,
        method=f"efficient_frontier_{objective}",
    )

    mu = mean_historical_return(returns_clean, frequency=frequency)
    S = sample_cov(returns_clean, frequency=frequency)
    S = S.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    ef = EfficientFrontier(mu, S, weight_bounds=(0 if long_only else -1, 1))
    ef.tickers = list(returns_clean.columns)

    if objective == "max_sharpe":
        ef.max_sharpe()
    elif objective == "min_volatility":
        ef.min_volatility()
    elif objective == "max_return":
        ef.efficient_return(max(mu))
    else:
        raise ValueError(f"Unknown objective: {objective}")

    cleaned = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False)
    ef_sharpe = perf[2] if len(perf) > 2 else 0.0

    return AllocationResult(
        weights=pd.Series(cleaned),
        expected_return=perf[0],
        volatility=perf[1],
        sharpe_ratio=ef_sharpe,
        method=f"efficient_frontier_{objective}",
    )


def optimize_cvar(
    returns: pd.DataFrame,
    beta: float = 0.95,
    frequency: int = 252,
) -> AllocationResult:
    """Conditional Value-at-Risk (CVaR) optimization via PyPortfolioOpt.

    Minimizes expected shortfall rather than variance — better for tail-risk
    management.

    Args:
        beta: Confidence level (e.g. 0.95 for CVaR_95).

    Use when: want to minimize tail risk rather than variance.
    """
    from pypfopt.efficient_frontier import EfficientCVaR
    from pypfopt.expected_returns import mean_historical_return

    returns_clean = _clean_returns(returns)
    if returns_clean.empty or returns_clean.shape[1] < 2:
        return AllocationResult(
            weights=pd.Series(dtype=float),
            expected_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            method="cvar",
        )

    mu = mean_historical_return(returns_clean, frequency=frequency)
    mu = mu.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    try:
        ec = EfficientCVaR(mu, returns_clean, beta=beta, weight_bounds=(0, 1))
        ec.min_cvar()
        cleaned = ec.clean_weights()
        perf = ec.portfolio_performance(verbose=False)
    except (ValueError, Exception) as exc:
        logger.debug(f"CVaR optimization failed: {exc}")
        return AllocationResult(
            weights=pd.Series(0.0, index=returns_clean.columns),
            expected_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            method=f"cvar_{beta:.0%}",
        )

    perf_list = list(perf) if isinstance(perf, tuple) else [perf, 0.0, 0.0]
    return AllocationResult(
        weights=pd.Series(cleaned),
        expected_return=float(perf_list[0]) if len(perf_list) > 0 else 0.0,
        volatility=float(perf_list[1]) if len(perf_list) > 1 else 0.0,
        sharpe_ratio=float(perf_list[2]) if len(perf_list) > 2 else 0.0,
        method=f"cvar_{beta:.0%}",
    )


def optimize_black_litterman_pypfopt(
    returns: pd.DataFrame,
    market_caps: Optional[pd.Series] = None,
    views: Optional[list] = None,
    tau: float = 0.05,
    frequency: int = 252,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Black-Litterman optimization using PyPortfolioOpt.

    Combines market-implied equilibrium returns with investor views.

    Args:
        returns: DataFrame of asset returns.
        market_caps: Market capitalizations for each ticker.
        views: List of PyPortfolioOpt view dicts (Q, P).
        tau: Uncertainty in the prior (default 0.05).
        frequency: Periods per year.

    Returns:
        Tuple of (weights Series, metadata dict).

    Use when: have both market equilibrium and subjective views (e.g. ML predictions).
    """
    from pypfopt.black_litterman import BlackLittermanModel
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.risk_models import sample_cov

    returns_clean = _clean_returns(returns)
    if returns_clean.empty or returns_clean.shape[1] < 2:
        empty = pd.Series(dtype=float)
        return empty, {"method": "black_litterman", "tau": tau}

    S = sample_cov(returns_clean, frequency=frequency)
    S = S.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    n_assets = len(returns_clean.columns)

    if market_caps is None:
        market_caps = pd.Series(1.0 / n_assets, index=returns_clean.columns)

    try:
        bl = BlackLittermanModel(S, pi="market", market_caps=market_caps, tau=tau)
        bl.tickers = list(returns_clean.columns)

        if views:
            for view in views:
                bl.add_absolute_view(**view) if "Q" in view and "P" not in view else None

        bl_returns = bl.bl_returns()

        ef = EfficientFrontier(bl_returns, S)
        ef.tickers = list(returns_clean.columns)
        ef.max_sharpe()
        cleaned = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=False)
    except (ValueError, Exception) as exc:
        logger.debug(f"BlackLitterman optimization failed: {exc}")
        return pd.Series(1.0 / n_assets, index=returns_clean.columns), {
            "method": "black_litterman",
            "tau": tau,
        }

    perf_list = list(perf) if isinstance(perf, tuple) else [perf, 0.0, 0.0]
    meta = {
        "method": "black_litterman",
        "tau": tau,
        "n_assets": n_assets,
        "n_views": len(views) if views else 0,
        "expected_return": float(perf_list[0]) if len(perf_list) > 0 else 0.0,
        "volatility": float(perf_list[1]) if len(perf_list) > 1 else 0.0,
        "sharpe_ratio": float(perf_list[2]) if len(perf_list) > 2 else 0.0,
    }

    return pd.Series(cleaned), meta


def compare_methods(
    returns: pd.DataFrame,
    frequency: int = 252,
) -> pd.DataFrame:
    """Compare allocation methods on historical returns.

    Returns a DataFrame with one row per method, columns:
    expected_return, volatility, sharpe_ratio, concentration (HHI).
    """
    methods: Dict[str, Any] = {}

    methods["equal_weight"] = _equal_weight(returns, frequency)
    methods["hrp"] = optimize_hrp(returns, frequency)
    methods["ef_max_sharpe"] = optimize_efficient_frontier(
        returns, "max_sharpe", frequency=frequency
    )
    methods["ef_min_vol"] = optimize_efficient_frontier(
        returns, "min_volatility", frequency=frequency
    )
    methods["cvar_95"] = optimize_cvar(returns, beta=0.95, frequency=frequency)

    rows = []
    for name, result in methods.items():
        if result is None:
            continue
        if hasattr(result, "weights"):
            w = result.weights.dropna()
            row = {
                "method": name,
                "expected_return": result.expected_return,
                "volatility": result.volatility,
                "sharpe_ratio": result.sharpe_ratio,
                "hhi": float((w**2).sum()) if len(w) > 0 else 1.0,
                "n_assets": int((w > 0.001).sum()) if len(w) > 0 else 0,
            }
        else:
            w = result.get("weights", pd.Series(dtype=float))
            row = {
                "method": name,
                "expected_return": result.get("expected_return", 0),
                "volatility": result.get("volatility", 0),
                "sharpe_ratio": result.get("sharpe_ratio", 0),
                "hhi": float((w**2).sum()) if len(w) > 0 else 1.0,
                "n_assets": int((w > 0.001).sum()) if len(w) > 0 else 0,
            }
        rows.append(row)

    return pd.DataFrame(rows).set_index("method")


def _equal_weight(returns: pd.DataFrame, frequency: int) -> Dict[str, Any]:
    """Equal-weight baseline."""
    returns_clean = _clean_returns(returns)
    n = returns_clean.shape[1]
    if n == 0:
        return {
            "weights": pd.Series(dtype=float),
            "expected_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
        }
    w = pd.Series(1.0 / n, index=returns_clean.columns)
    port_ret = returns_clean.dot(w)
    ann_ret = port_ret.mean() * frequency
    ann_vol = port_ret.std() * np.sqrt(frequency)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    return {"weights": w, "expected_return": ann_ret, "volatility": ann_vol, "sharpe_ratio": sharpe}
