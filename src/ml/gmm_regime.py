"""
Gaussian Mixture Model (GMM) Regime Detector

Uses sklearn.mixture.GaussianMixture for soft regime assignments with
ellipsoidal cluster shapes (vs KMeans spherical). The key advantage over
KMeans: provides true posterior probabilities, not inverse-distance approximations.

Features:
- Optimal n_components selection via BIC (Bayesian Information Criterion)
- Soft regime probabilities from GMM posterior (sum to 1 per row)
- Ellipsoidal cluster shapes capture covariance structure
- Follows RegimeDetectorBase interface (fit/predict/predict_proba/get_regime_summary)

Example:
    >>> from src.ml.gmm_regime import GMMRegimeDetector
    >>> detector = GMMRegimeDetector(n_range=(2, 6))
    >>> detector.fit(features)
    >>> regimes = detector.predict(features)
    >>> probs = detector.predict_proba(features)  # True GMM probabilities
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.ml.regime_base import RegimeDetectorBase, RegimeSummary


class GMMRegimeDetector(RegimeDetectorBase):
    """
    Gaussian Mixture Model regime detector.

    Uses GMM for soft regime assignments with full covariance modeling.
    Unlike KMeans (hard spherical clusters), GMM provides posterior
    probabilities and elliptical cluster shapes.

    Args:
        n_range: Range of n_components to evaluate via BIC
        covariance_type: GMM covariance type ("full", "tied", "diag", "spherical")
        max_iter: Maximum EM iterations
        random_state: Random seed for reproducibility
        n_init: Number of GMM initializations per n_components
        min_samples: Minimum samples required for fitting

    Attributes:
        gmm_: Fitted GaussianMixture model
        optimal_k_: Selected number of components
        bic_scores_: BIC scores for each n_components tested
        regime_mapping_: Mapping from component index to interpretable name
    """

    def __init__(
        self,
        n_range: Tuple[int, int] = (2, 6),
        covariance_type: str = "full",
        max_iter: int = 200,
        random_state: int = 42,
        n_init: int = 5,
        min_samples: int = 50,
    ):
        super().__init__(name="GMMRegimeDetector")

        if n_range[0] < 2:
            raise ValueError("Minimum n_components must be >= 2")

        self.n_range = n_range
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_init = n_init
        self.min_samples = min_samples

        self.gmm_: Optional[GaussianMixture] = None
        self.optimal_k_: int = 0
        self.bic_scores_: Dict[int, float] = {}
        self.aic_scores_: Dict[int, float] = {}
        self.scaler_ = StandardScaler()
        self.regime_mapping_: Dict[int, str] = {}
        self.fitted_n_samples_ = 0

    def fit(self, data: pd.DataFrame) -> GMMRegimeDetector:
        """
        Fit GMM on input features, selecting optimal n_components via BIC.

        Args:
            data: DataFrame with technical features

        Returns:
            Self for method chaining

        Raises:
            ValueError: If insufficient samples or features
        """
        self._validate_data(data)

        X_scaled = self.scaler_.fit_transform(data)

        self._select_optimal_components(X_scaled)

        self.is_fitted = True
        self.fitted_n_samples_ = len(data)

        self._create_regime_mapping(X_scaled)

        return self

    def _validate_data(self, data: pd.DataFrame) -> None:
        if data.empty:
            raise ValueError("Input data cannot be empty")

        if len(data) < self.min_samples:
            raise ValueError(
                f"Insufficient samples: {len(data)} < {self.min_samples} (minimum required)"
            )

        if data.isna().any().any():
            raise ValueError("Input data contains NaN values. Handle missing values first.")

        if len(data.columns) == 0:
            raise ValueError("Input data has no features")

    def _select_optimal_components(self, X: np.ndarray) -> None:
        """Select optimal n_components via BIC minimization."""
        k_min, k_max = self.n_range
        k_max = min(k_max, len(X) - 1)

        best_bic = float("inf")
        best_k = k_min

        for k in range(k_min, k_max + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=self.covariance_type,
                max_iter=self.max_iter,
                n_init=self.n_init,
                random_state=self.random_state,
                reg_covar=1e-6,
            )
            gmm.fit(X)

            bic = gmm.bic(X)
            aic = gmm.aic(X)

            self.bic_scores_[k] = bic
            self.aic_scores_[k] = aic

            if bic < best_bic:
                best_bic = bic
                best_k = k

        self.optimal_k_ = best_k

        self.gmm_ = GaussianMixture(
            n_components=self.optimal_k_,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state,
            reg_covar=1e-6,
        )
        self.gmm_.fit(X)

    def _create_regime_mapping(self, X: np.ndarray) -> None:
        """Create interpretable regime labels based on component characteristics."""
        if self.gmm_ is None:
            return

        means = self.gmm_.means_
        covariances = self.gmm_.covariances_

        if self.covariance_type == "spherical":
            variances = covariances
        elif self.covariance_type == "diag":
            variances = np.array([np.mean(cov) for cov in covariances])
        elif self.covariance_type == "tied":
            variances = np.full(len(means), np.mean(covariances))
        else:
            variances = np.array([np.trace(cov) / cov.shape[0] for cov in covariances])

        mean_returns = means.mean(axis=1)
        sorted_by_return = np.argsort(mean_returns)[::-1]

        n_clusters = self.optimal_k_
        n_high = max(1, n_clusters // 3)
        n_low = max(1, n_clusters // 3)

        var_ranks = np.argsort(variances)[::-1]
        var_rank_map = {int(idx): int(rank) for rank, idx in enumerate(var_ranks)}

        self.regime_mapping_ = {}
        for rank, comp_idx in enumerate(sorted_by_return):
            v_rank = var_rank_map.get(int(comp_idx), 0)
            if v_rank < n_high:
                label = f"High_Vol_{rank}"
            elif v_rank >= n_clusters - n_low:
                label = f"Low_Vol_{rank}"
            else:
                label = f"Neutral_{rank}"
            self.regime_mapping_[int(comp_idx)] = label

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict regime labels (hard assignment) for input data.

        Args:
            data: DataFrame with features (same columns as training)

        Returns:
            Series of regime labels (strings)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler_.transform(data)
        hard_labels = self.gmm_.predict(X_scaled)

        labels = [self.regime_mapping_.get(int(c), f"Component_{c}") for c in hard_labels]

        return pd.Series(labels, index=data.index, name="gmm_regime")

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime probabilities (GMM posterior) for input data.

        Unlike KMeans which approximates probabilities via inverse distance,
        GMM provides true posterior probabilities P(component | sample).

        Args:
            data: DataFrame with features (same columns as training)

        Returns:
            DataFrame with probability for each regime (columns sum to 1)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler_.transform(data)
        probs = self.gmm_.predict_proba(X_scaled)

        columns = [self.regime_mapping_.get(i, f"Component_{i}") for i in range(self.optimal_k_)]

        return pd.DataFrame(probs, columns=columns, index=data.index)

    def get_raw_components(self, data: pd.DataFrame) -> pd.Series:
        """Get raw component indices (0 to k-1)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler_.transform(data)
        labels = self.gmm_.predict(X_scaled)
        return pd.Series(labels, index=data.index, name="gmm_component")

    def get_regime_summary(self) -> RegimeSummary:
        """
        Get summary information about detected regimes.

        Returns:
            RegimeSummary with label distribution and metadata
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        labels = list(self.regime_mapping_.values())
        unique_labels = list(set(labels))

        label_counts: Dict[str, int] = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        total = sum(label_counts.values())
        proportions = {k: v / total for k, v in label_counts.items()}

        component_weights = self.gmm_.weights_.tolist()

        return RegimeSummary(
            name=self.name,
            n_regimes=self.optimal_k_,
            regime_labels=unique_labels,
            label_distribution=label_counts,
            label_proportions=proportions,
            metadata={
                "bic_scores": self.bic_scores_,
                "aic_scores": self.aic_scores_,
                "best_bic": self.bic_scores_.get(self.optimal_k_, float("inf")),
                "covariance_type": self.covariance_type,
                "component_weights": component_weights,
                "n_samples_fitted": self.fitted_n_samples_,
                "n_features": self.gmm_.means_.shape[1] if self.gmm_ else 0,
                "converged": bool(self.gmm_.converged_),
                "n_iter": int(self.gmm_.n_iter_),
            },
        )

    def get_bic_analysis(self) -> pd.DataFrame:
        """
        Get BIC and AIC scores for all tested n_components.

        Returns:
            DataFrame with BIC/AIC scores per k value
        """
        if not self.bic_scores_:
            raise ValueError("Model not fitted. Call fit() first.")

        return pd.DataFrame(
            {
                "bic": self.bic_scores_,
                "aic": self.aic_scores_,
            }
        ).rename_axis("k")

    def get_component_weights(self) -> pd.Series:
        """
        Get the mixing weights (priors) for each component.

        Returns:
            Series with weight per regime label
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        weights = self.gmm_.weights_
        index = [self.regime_mapping_.get(i, f"Component_{i}") for i in range(self.optimal_k_)]

        return pd.Series(weights, index=index, name="component_weight")

    def get_component_means(self) -> pd.DataFrame:
        """
        Get the mean vector for each component.

        Returns:
            DataFrame with components as rows and feature means as columns
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        n_components, n_features = self.gmm_.means_.shape
        columns = [f"feature_{i}" for i in range(n_features)]
        index = [self.regime_mapping_.get(i, f"Component_{i}") for i in range(n_components)]

        return pd.DataFrame(self.gmm_.means_, index=index, columns=columns)
