from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_performance(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    periods_per_year: int = 252,
) -> Tuple[float, float]:
    """Compute annualized portfolio return and standard deviation.

    Args:
        weights: Asset weights (must sum to 1 in absolute value).
        mean_returns: Series of per-period expected returns.
        cov_matrix: Covariance matrix of returns.
        periods_per_year: Number of periods for annualization.

    Returns:
        Tuple of (annualized_return, annualized_std_dev).
    """
    port_return = weights @ mean_returns.values
    port_std = np.sqrt(weights @ cov_matrix.values @ weights)
    annualized_return = (1 + port_return) ** periods_per_year - 1
    annualized_std = port_std * np.sqrt(periods_per_year)
    return annualized_return, annualized_std


def neg_sharpe_ratio(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    rf_rate: float = 0.0,
) -> float:
    """Negative Sharpe ratio (for minimization).

    Args:
        weights: Asset weights.
        mean_returns: Per-period expected returns.
        cov_matrix: Covariance matrix.
        rf_rate: Annualized risk-free rate.

    Returns:
        Negative Sharpe ratio value.
    """
    port_return, port_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(port_return - rf_rate) / port_std if port_std > 0 else 0.0


def max_sharpe_ratio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    rf_rate: float = 0.0,
    short_allowed: bool = False,
    weight_bounds: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """Find portfolio weights that maximize the Sharpe ratio.

    Uses scipy SLSQP optimizer with constraints that weights sum to +/-1.

    Args:
        mean_returns: Per-period expected returns.
        cov_matrix: Return covariance matrix.
        rf_rate: Annualized risk-free rate.
        short_allowed: If True, allows negative weights.
        weight_bounds: Custom (min, max) bounds per asset. Overrides short_allowed.

    Returns:
        Dictionary mapping ticker to optimized weight.
    """
    n_assets = len(mean_returns)
    x0 = np.array([1.0 / n_assets] * n_assets)

    if weight_bounds is not None:
        bounds = (weight_bounds,) * n_assets
    elif short_allowed:
        bounds = ((-1.0, 1.0),) * n_assets
    else:
        bounds = ((0.0, 1.0),) * n_assets

    constraints = [{"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0}]

    result = minimize(
        fun=neg_sharpe_ratio,
        x0=x0,
        args=(mean_returns, cov_matrix, rf_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return dict(zip(mean_returns.index, result.x.round(6)))


def min_volatility(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    short_allowed: bool = False,
) -> Dict[str, float]:
    """Find portfolio weights that minimize volatility.

    Args:
        mean_returns: Per-period expected returns (unused but kept for API consistency).
        cov_matrix: Return covariance matrix.
        short_allowed: If True, allows negative weights.

    Returns:
        Dictionary mapping ticker to optimized weight.
    """
    n_assets = len(mean_returns)
    x0 = np.array([1.0 / n_assets] * n_assets)
    bounds = ((-1.0, 1.0) if short_allowed else (0.0, 1.0),) * n_assets
    constraints = [{"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0}]

    def portfolio_variance(w: np.ndarray, cov: pd.DataFrame) -> float:
        return w @ cov.values @ w

    result = minimize(
        fun=portfolio_variance,
        x0=x0,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(f"Min vol optimization failed: {result.message}")

    return dict(zip(mean_returns.index, result.x.round(6)))


def efficient_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    target_returns: np.ndarray,
    short_allowed: bool = False,
) -> list:
    """Compute the efficient frontier by solving min-variance at each target return.

    Args:
        mean_returns: Per-period expected returns.
        cov_matrix: Return covariance matrix.
        target_returns: Array of target return levels.
        short_allowed: If True, allows negative weights.

    Returns:
        List of (return, std_dev, weights_dict) tuples along the frontier.
    """
    n_assets = len(mean_returns)
    x0 = np.array([1.0 / n_assets] * n_assets)
    bounds = ((-1.0, 1.0) if short_allowed else (0.0, 1.0),) * n_assets

    def port_std(w: np.ndarray, cov: pd.DataFrame) -> float:
        return np.sqrt(w @ cov.values @ w)

    frontier = []
    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0},
            {"type": "eq", "fun": lambda w, mr=mean_returns: w @ mr.values - target},
        ]
        result = minimize(
            fun=port_std,
            x0=x0,
            args=(cov_matrix,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if result.success:
            port_return, port_std = portfolio_performance(result.x, mean_returns, cov_matrix)
            frontier.append(
                (port_return, port_std, dict(zip(mean_returns.index, result.x.round(4))))
            )

    return frontier


def kelly_allocation_single(
    mean_return: float,
    variance: float,
) -> float:
    """Compute the Kelly fraction for a single asset using the analytical approximation.

    Kelly fraction f* ≈ excess_return / variance for small returns.

    Args:
        mean_return: Expected per-period return.
        variance: Per-period return variance.

    Returns:
        Optimal Kelly fraction (bet size as proportion of capital).
    """
    if variance <= 0:
        return 0.0
    return mean_return / variance


def kelly_allocation_multi(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
) -> pd.Series:
    """Compute multi-asset Kelly allocations via the precision matrix method.

    Kelly weights are proportional to: Sigma^{-1} * mu (the tangency portfolio).

    Args:
        mean_returns: Per-period expected returns.
        cov_matrix: Return covariance matrix.

    Returns:
        Series of Kelly fractions per asset.
    """
    precision = np.linalg.inv(cov_matrix.values)
    kelly_weights = precision @ mean_returns.values
    return pd.Series(kelly_weights, index=mean_returns.index)


def monte_carlo_portfolios(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    rf_rate: float = 0.0,
    n_portfolios: int = 100_000,
    allow_short: bool = False,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Generate random portfolio allocations via Dirichlet sampling.

    Args:
        mean_returns: Per-period expected returns.
        cov_matrix: Return covariance matrix.
        rf_rate: Annualized risk-free rate.
        n_portfolios: Number of random portfolios to generate.
        allow_short: If True, randomly flip signs on some weights.
        periods_per_year: Number of periods for annualization.

    Returns:
        DataFrame with columns ['Annualized Standard Deviation',
        'Annualized Returns', 'Sharpe Ratio'].
    """
    n_assets = len(mean_returns)
    alpha = np.full(n_assets, 0.05)
    weights = np.random.dirichlet(alpha=alpha, size=n_portfolios)

    if allow_short:
        signs = np.random.choice([-1, 1], size=weights.shape)
        weights *= signs

    ann_returns = (weights @ mean_returns.values + 1) ** periods_per_year - 1
    ann_stds = np.apply_along_axis(
        lambda w: np.sqrt(w @ cov_matrix.values @ w) * np.sqrt(periods_per_year),
        1,
        weights,
    )
    sharpe_ratios = (ann_returns - rf_rate) / ann_stds

    return pd.DataFrame(
        {
            "Annualized Standard Deviation": ann_stds,
            "Annualized Returns": ann_returns,
            "Sharpe Ratio": sharpe_ratios,
        }
    )
