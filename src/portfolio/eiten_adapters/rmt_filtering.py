"""Random Matrix Theory covariance matrix denoising.

Filters out noise eigenvalues below the Marchenko-Pastur theoretical bound,
reconstructing a denoised covariance matrix. Based on:
https://srome.github.io/Eigenvesting-III-Random-Matrix-Filtering-In-Finance/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RMTResult:
    """Result of RMT filtering."""

    filtered_cov: np.ndarray
    """Denoised covariance matrix (N x N)."""
    original_eigenvalues: np.ndarray
    """Eigenvalues of the original correlation matrix (sorted ascending)."""
    filtered_eigenvalues: np.ndarray
    """Eigenvalues after zeroing noise components."""
    max_theoretical_eval: float
    """Marchenko-Pastur upper bound for noise eigenvalues."""
    n_kept: int
    """Number of eigenvalues surviving the filter."""
    n_total: int
    """Total number of eigenvalues."""


class RMTFiltering:
    """Random Matrix Theory covariance denoising.

    Parameters:
        sigma: Standard deviation of standardized returns (default 1.0).
    """

    def __init__(self, sigma: float = 1.0) -> None:
        self.sigma = sigma

    def denoise(self, returns_matrix: np.ndarray) -> RMTResult:
        """Apply RMT filtering to a returns matrix.

        Args:
            returns_matrix: Shape (T x N) — T time steps, N assets.

        Returns:
            RMTResult with filtered covariance and eigenvalue diagnostics.
        """
        n_assets = returns_matrix.shape[1]
        n_bars = returns_matrix.shape[0]
        q_ratio = n_bars / n_assets if n_assets > 0 else 0.0

        variances = np.var(returns_matrix, axis=0)
        std_devs = np.sqrt(variances)

        corr_matrix = np.corrcoef(returns_matrix, rowvar=False)
        eig_values, eig_vectors = np.linalg.eigh(corr_matrix)

        max_theoretical_eval = self.sigma**2 * (1 + np.sqrt(1.0 / q_ratio)) ** 2

        original_eigenvalues = eig_values.copy()
        eig_values[eig_values <= max_theoretical_eval] = 0.0

        temp = eig_vectors @ np.diag(eig_values) @ eig_vectors.T
        np.fill_diagonal(temp, 1.0)
        filtered_corr = temp

        filtered_cov = np.diag(std_devs) @ filtered_corr @ np.diag(std_devs)

        n_kept = int(np.sum(original_eigenvalues > max_theoretical_eval))

        return RMTResult(
            filtered_cov=filtered_cov,
            original_eigenvalues=original_eigenvalues,
            filtered_eigenvalues=eig_values,
            max_theoretical_eval=float(max_theoretical_eval),
            n_kept=n_kept,
            n_total=n_assets,
        )

    @staticmethod
    def ensure_psd(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """Ensure a matrix is positive semi-definite by adding regularization."""
        min_eig = np.linalg.eigvalsh(matrix)[0]
        if min_eig < 0:
            return matrix - min_eig * np.eye(len(matrix)) + epsilon * np.eye(len(matrix))
        return matrix + epsilon * np.eye(len(matrix))
