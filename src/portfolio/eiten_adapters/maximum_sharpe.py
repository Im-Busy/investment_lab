"""Maximum Sharpe Ratio Portfolio (MSR / Tangency Portfolio).

Closed-form solution: w = Σ⁻¹ μ / (1ᵀ Σ⁻¹ μ).
Maximizes (μᵀ w) / sqrt(wᵀ Σ w) for a given expected return vector.

Reference: Eigen Portfolio Selection — A Robust Approach to Sharpe Ratio Maximization
(https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3070416)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class MaximumSharpePortfolio:
    """Maximum Sharpe ratio portfolio via closed-form solution.

    Parameters:
        risk_free_rate: Annualized risk-free rate (default 0.0).
    """

    def __init__(self, risk_free_rate: float = 0.0) -> None:
        self.risk_free_rate = risk_free_rate / 252.0  # daily

    def build(
        self,
        symbols: list[str],
        covariance_matrix: np.ndarray,
        expected_returns: np.ndarray,
    ) -> dict[str, float]:
        """Compute maximum Sharpe portfolio weights.

        Args:
            symbols: List of asset symbols.
            covariance_matrix: N x N covariance matrix.
            expected_returns: N-length vector of expected daily returns.

        Returns:
            Dict mapping symbol -> weight (sums to 1.0).
        """
        excess = expected_returns - self.risk_free_rate
        inv_cov = np.linalg.pinv(covariance_matrix)
        ones = np.ones(len(inv_cov))

        numerator = inv_cov @ excess
        denominator = ones @ numerator

        if abs(denominator) < 1e-12:
            logger.warning("MSR denominator near zero, falling back to equal weight")
            return dict(zip(symbols, [1.0 / len(symbols)] * len(symbols)))

        weights = numerator / denominator

        if np.sum(weights) < 0:
            weights = -weights

        return dict(zip(symbols, weights))

    def build_with_constraints(
        self,
        symbols: list[str],
        covariance_matrix: np.ndarray,
        expected_returns: np.ndarray,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
    ) -> dict[str, float]:
        """Compute MSR with box constraints using scipy SLSQP.

        Args:
            symbols: List of asset symbols.
            covariance_matrix: N x N covariance matrix.
            expected_returns: N-length vector of expected daily returns.
            lower_bound: Minimum weight per asset.
            upper_bound: Maximum weight per asset.

        Returns:
            Dict mapping symbol -> weight.
        """
        from scipy.optimize import minimize

        n = len(symbols)
        excess = expected_returns - self.risk_free_rate

        def neg_sharpe(w: np.ndarray) -> float:
            port_return = float(w @ excess)
            port_std = float(np.sqrt(w @ covariance_matrix @ w))
            if port_std < 1e-12:
                return 0.0
            return -port_return / port_std

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(lower_bound, upper_bound) for _ in range(n)]

        x0 = np.ones(n) / n
        result = minimize(
            neg_sharpe,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            logger.warning("MSR constrained optimization did not converge: %s", result.message)

        return dict(zip(symbols, result.x))
