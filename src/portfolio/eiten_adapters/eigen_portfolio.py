"""Eigen (PCA) portfolio strategy.

The N-th eigen portfolio is constructed from the N-th largest eigenvector
of the covariance matrix, yielding a portfolio uncorrelated to the market
(1st eigen). Based on: https://srome.github.io/Eigenvesting-I/
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class EigenPortfolio:
    """PCA-based eigen portfolio construction.

    Each eigen portfolio is long/short based on the eigenvector components.
    The 1st eigen = market portfolio. The 2nd eigen = highest-return portfolio
    orthogonal to market. The 3rd eigen is orthogonal to both, etc.

    Parameters:
        eigen_number: Which eigen portfolio to use (1=market, 2=orthogonal, etc.).
    """

    def __init__(self, eigen_number: int = 2) -> None:
        if eigen_number < 1:
            raise ValueError("eigen_number must be >= 1")
        self.eigen_number = eigen_number

    def build(
        self,
        symbols: list[str],
        covariance_matrix: np.ndarray,
    ) -> dict[str, float]:
        """Compute eigen portfolio weights.

        Args:
            symbols: List of asset symbols.
            covariance_matrix: N x N covariance matrix.

        Returns:
            Dict mapping symbol -> weight (sums to 1.0).
        """
        eig_values, eig_vectors = np.linalg.eigh(covariance_matrix)

        if self.eigen_number > len(symbols):
            raise ValueError(f"eigen_number {self.eigen_number} > number of assets {len(symbols)}")

        eigen_vector = eig_vectors[:, -self.eigen_number]
        weights = eigen_vector / np.sum(np.abs(eigen_vector))

        return dict(zip(symbols, weights))

    def build_all(
        self,
        symbols: list[str],
        covariance_matrix: np.ndarray,
        max_eigens: int | None = None,
    ) -> dict[int, dict[str, float]]:
        """Build all eigen portfolios up to max_eigens.

        Returns:
            Dict mapping eigen_number -> {symbol: weight}.
        """
        n = min(len(symbols), max_eigens or len(symbols))
        eig_values, eig_vectors = np.linalg.eigh(covariance_matrix)

        result: dict[int, dict[str, float]] = {}
        for i in range(1, n + 1):
            vec = eig_vectors[:, -i]
            w = vec / np.sum(np.abs(vec))
            result[i] = dict(zip(symbols, w))

        return result

    @staticmethod
    def market_portfolio(
        symbols: list[str],
        covariance_matrix: np.ndarray,
    ) -> dict[str, float]:
        """Convenience: build the market (1st eigen) portfolio."""
        return EigenPortfolio(eigen_number=1).build(symbols, covariance_matrix)
