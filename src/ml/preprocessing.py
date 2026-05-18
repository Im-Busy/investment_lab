"""
R8: MAD + Rank Data Standardization Pipeline.

Two-step preprocessing pipeline with:
1. MADOutlierClipper — median-based outlier removal (|x - median| > n * MAD)
2. RankStandardizer — non-parametric rank standardization

Why rank standardization: Handles fat tails (crypto, financial returns) without
assuming normality. 华泰 recommends it for broader applicability.

Source: 华泰多因子 §1.2

Architecture: sklearn TransformerMixin pipeline, insertable into FeatureExtractor
or train_ml_pipeline_v3.py.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from scipy.special import erfinv
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

# Default MAD multiplier for outlier clipping
DEFAULT_MAD_MULTIPLIER = 5.0


def _mad(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Median Absolute Deviation (MAD).

    MAD = median(|x - median(x)|) * 1.4826 (consistency factor for normal distribution).
    """
    median = np.nanmedian(x, axis=axis, keepdims=True)
    abs_dev = np.abs(x - median)
    mad = np.nanmedian(abs_dev, axis=axis, keepdims=True) * 1.4826
    return np.squeeze(mad)


class MADOutlierClipper(BaseEstimator, TransformerMixin):
    """Clip outliers using Median Absolute Deviation.

    For each feature column, values beyond ``n_mad × MAD`` from the median
    are clipped to the boundary. This preserves feature rank order while
    capping extreme values.

    Parameters:
        n_mad: Number of MADs from median before clipping. Default 5.0.
            Use 3.0 for aggressive clipping, 5.0 for moderate.

    Attributes:
        median_: Per-column medians learned during fit.
        mad_: Per-column MAD values learned during fit.
        lower_bound_: Per-column lower clip bounds.
        upper_bound_: Per-column upper clip bounds.

    Example:
        >>> clipper = MADOutlierClipper(n_mad=5.0)
        >>> X_clipped = clipper.fit_transform(X)
    """

    def __init__(self, n_mad: float = DEFAULT_MAD_MULTIPLIER) -> None:
        self.n_mad = n_mad

    def fit(self, X: np.ndarray, y: Any = None) -> "MADOutlierClipper":
        """Learn per-column medians and MAD boundaries.

        Args:
            X: (n_samples, n_features) array.
            y: Ignored.

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.median_ = np.nanmedian(X, axis=0)
        self.mad_ = np.array([_mad(X[:, j]) for j in range(X.shape[1])])

        mad_scaled = self.mad_ * self.n_mad
        self.lower_bound_ = self.median_ - mad_scaled
        self.upper_bound_ = self.median_ + mad_scaled

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip outliers to MAD boundaries.

        Args:
            X: (n_samples, n_features) array.

        Returns:
            Clipped array with same shape.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        result = X.copy()
        for j in range(min(X.shape[1], len(self.lower_bound_))):
            col = result[:, j]
            mask_low = col < self.lower_bound_[j]
            mask_high = col > self.upper_bound_[j]
            col[mask_low] = self.lower_bound_[j]
            col[mask_high] = self.upper_bound_[j]

        return result

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        """Return feature names (passthrough)."""
        if input_features is not None:
            return list(input_features)
        return [f"x{i}" for i in range(getattr(self, "n_features_in_", 0))]


class RankStandardizer(BaseEstimator, TransformerMixin):
    """Non-parametric rank-based standardization.

    Replaces each value with its percentile rank, then optionally maps to
    a target distribution (uniform [0,1] or normal via probit).

    Robust to outliers and does not assume normality. Ideal for financial
    data with fat tails.

    Parameters:
        output_distribution: Target distribution for output.
            - 'uniform': Values in [0, 1] representing percentile rank.
            - 'normal': Values mapped to standard normal via probit
              (inverse CDF). Produces ~N(0, 1) output for any input
              distribution shape, except ties.

    Attributes:
        n_samples_: Number of samples seen during fit (for rank normalization).

    Example:
        >>> stand = RankStandardizer(output_distribution='normal')
        >>> X_ranked = stand.fit_transform(X)
    """

    def __init__(self, output_distribution: str = "uniform") -> None:
        if output_distribution not in ("uniform", "normal"):
            raise ValueError(
                f"output_distribution must be 'uniform' or 'normal', got {output_distribution!r}"
            )
        self.output_distribution = output_distribution

    def fit(self, X: np.ndarray, y: Any = None) -> "RankStandardizer":
        """Record sample count for rank normalization.

        Args:
            X: (n_samples, n_features) array.
            y: Ignored.

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_samples_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Convert values to percentile ranks (and optionally to normal).

        Args:
            X: (n_samples, n_features) array.

        Returns:
            Rank-transformed array in [0, 1] (uniform) or ~N(0, 1) (normal).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n = X.shape[0]
        result = np.empty_like(X)

        for j in range(X.shape[1]):
            col = X[:, j]
            # Percentile rank: argsort-based, handles NaN by placing at bottom
            valid = ~np.isnan(col)
            ranks = np.zeros(n, dtype=np.float64)
            if valid.sum() > 0:
                order = np.argsort(col[valid])
                n_valid = len(order)
                rank_vals = np.arange(1, n_valid + 1, dtype=np.float64)
                rank_vals /= n_valid  # [1/N, 1]
                ranks[valid] = rank_vals[np.argsort(order)]

            if self.output_distribution == "normal":
                # Probit: map [eps, 1-eps] → N(0,1) via inverse error function
                eps = 1.0 / (2.0 * n + 1.0)
                clipped = np.clip(ranks, eps, 1.0 - eps)
                result[:, j] = np.sqrt(2.0) * erfinv(2.0 * clipped - 1.0)
            else:
                result[:, j] = ranks

        return result

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        """Return feature names (passthrough)."""
        if input_features is not None:
            return list(input_features)
        return [f"x{i}" for i in range(getattr(self, "n_features_in_", 0))]


def mad_clip(
    x: np.ndarray,
    n_mad: float = DEFAULT_MAD_MULTIPLIER,
) -> np.ndarray:
    """Convenience function: clip a single array using MAD.

    Args:
        x: (N,) or (N, M) array.
        n_mad: Number of MADs from median for clipping.

    Returns:
        Clipped array with same shape.
    """
    clipper = MADOutlierClipper(n_mad=n_mad)
    return clipper.fit_transform(x)


def rank_standardize(
    x: np.ndarray,
    output_distribution: str = "normal",
) -> np.ndarray:
    """Convenience function: rank-standardize a single array.

    Args:
        x: (N,) or (N, M) array.
        output_distribution: 'uniform' or 'normal'.

    Returns:
        Rank-transformed array.
    """
    stand = RankStandardizer(output_distribution=output_distribution)
    return stand.fit_transform(x)


def mad_rank_pipeline(
    x: np.ndarray,
    n_mad: float = DEFAULT_MAD_MULTIPLIER,
    output_distribution: str = "normal",
) -> np.ndarray:
    """Full MAD clip → rank standardize pipeline.

    Args:
        x: (N,) or (N, M) array.
        n_mad: MAD multiplier for outlier clipping.
        output_distribution: 'uniform' or 'normal' rank output.

    Returns:
        Outlier-clipped, rank-standardized array.
    """
    clipped = mad_clip(x, n_mad=n_mad)
    return rank_standardize(clipped, output_distribution=output_distribution)
