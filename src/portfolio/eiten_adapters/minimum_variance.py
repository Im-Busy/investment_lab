"""Minimum Variance Portfolio (MVP).

Closed-form solution: w = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1).
Minimizes portfolio variance without considering expected returns.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class MinimumVariancePortfolio:
    """Minimum variance portfolio via closed-form QP solution.

    No expected return estimates needed — only covariance.
    Suitable when return forecasts are unreliable.
    """

    def build(
        self,
        symbols: list[str],
        covariance_matrix: np.ndarray,
    ) -> dict[str, float]:
        """Compute minimum variance portfolio weights.

        Args:
            symbols: List of asset symbols.
            covariance_matrix: N x N covariance matrix.

        Returns:
            Dict mapping symbol -> weight (sums to 1.0).
        """
        inv_cov = np.linalg.pinv(covariance_matrix)
        ones = np.ones(len(inv_cov))
        inv_dot_ones = inv_cov @ ones
        weights = inv_dot_ones / (ones @ inv_dot_ones)

        return dict(zip(symbols, weights))

    def build_with_constraints(
        self,
        symbols: list[str],
        covariance_matrix: np.ndarray,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
    ) -> dict[str, float]:
        """Compute MVP with box constraints using scipy SLSQP.

        Args:
            symbols: List of asset symbols.
            covariance_matrix: N x N covariance matrix.
            lower_bound: Minimum weight per asset.
            upper_bound: Maximum weight per asset.

        Returns:
            Dict mapping symbol -> weight.
        """
        from scipy.optimize import minimize

        n = len(symbols)

        def objective(w: np.ndarray) -> float:
            return float(w @ covariance_matrix @ w)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(lower_bound, upper_bound) for _ in range(n)]

        x0 = np.ones(n) / n
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            logger.warning("MVP constrained optimization did not converge: %s", result.message)

        return dict(zip(symbols, result.x))
