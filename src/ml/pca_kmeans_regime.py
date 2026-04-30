"""
PCA + k-means Regime Detector

Unsupervised clustering approach to regime detection:
1. PCA for dimensionality reduction (95% variance retained)
2. k-means clustering with optimal k (selected via silhouette score)
3. Cluster assignments become regime labels

Features:
- Automatic dimension selection via explained variance
- Optimal k selection via silhouette analysis
- Cluster center interpretation
- Works well with high-dimensional feature sets

Example:
    >>> from src.ml.pca_kmeans_regime import PCAKMeansRegimeDetector
    >>> detector = PCAKMeansRegimeDetector(k_range=(2, 6))
    >>> detector.fit(features)
    >>> regimes = detector.predict(features)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.ml.regime_base import RegimeDetectorBase, RegimeSummary


class PCAKMeansRegimeDetector(RegimeDetectorBase):
    """
    PCA + k-means regime detector.

    Two-stage approach:
    1. PCA reduces dimensionality while preserving 95% variance
    2. k-means clusters the PCA-transformed data into regimes

    Optimal k is selected via silhouette score analysis over a range.

    Args:
        k_range: Range of k values to evaluate for optimal clustering
        variance_threshold: Fraction of variance to retain in PCA
        max_components: Maximum number of PCA components (None = auto)
        random_state: Random seed for reproducibility
        n_init: Number of k-means initializations

    Attributes:
        pca_: Fitted PCA transformer
        kmeans_: Fitted k-means model
        optimal_k_: Selected number of clusters
        silhouette_scores_: Silhouette scores for each k tested
    """

    def __init__(
        self,
        k_range: Tuple[int, int] = (2, 6),
        variance_threshold: float = 0.95,
        max_components: Optional[int] = None,
        random_state: int = 42,
        n_init: int = 10,
        min_samples: int = 50,
    ):
        super().__init__(name="PCAKMeansRegimeDetector")

        self.k_range = k_range
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        self.random_state = random_state
        self.n_init = n_init
        self.min_samples = min_samples

        self.pca_: Optional[PCA] = None
        self.kmeans_: Optional[KMeans] = None
        self.optimal_k_: int = 0
        self.silhouette_scores_: Dict[int, float] = {}
        self.cluster_centers_: Optional[np.ndarray] = None
        self.scaler_ = StandardScaler()
        self.fitted_n_samples_ = 0
        self.regime_mapping_: Dict[int, str] = {}

    def fit(self, data: pd.DataFrame) -> PCAKMeansRegimeDetector:
        """
        Fit PCA + k-means on input features.

        Args:
            data: DataFrame with technical features

        Returns:
            Self for method chaining

        Raises:
            ValueError: If insufficient samples or features
        """
        self._validate_data(data)

        # Scale features
        X_scaled = self.scaler_.fit_transform(data)

        # Fit PCA
        self._fit_pca(X_scaled)

        # Transform with PCA
        X_pca = self.pca_.transform(X_scaled)

        # Find optimal k and fit k-means
        self._find_optimal_k(X_pca)

        self.is_fitted = True
        self.fitted_n_samples_ = len(data)

        # Create interpretable regime mapping
        self._create_regime_mapping(X_pca)

        return self

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data before fitting."""
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

    def _fit_pca(self, X: np.ndarray) -> None:
        """Fit PCA transformer."""
        n_components = min(
            min(X.shape) - 1,
            self.max_components if self.max_components else X.shape[1],
        )

        self.pca_ = PCA(
            n_components=min(n_components, int(self.variance_threshold * n_components) + 1),
            random_state=self.random_state,
        )
        self.pca_.fit(X)

        # Adjust components to meet variance threshold
        if self.pca_.explained_variance_ratio_.sum() < self.variance_threshold:
            n_comp_needed = (
                np.argmax(np.cumsum(self.pca_.explained_variance_ratio_) >= self.variance_threshold)
                + 1
            )
            n_comp_needed = min(n_comp_needed, n_components)

            self.pca_ = PCA(
                n_components=n_comp_needed,
                random_state=self.random_state,
            )
            self.pca_.fit(X)

    def _find_optimal_k(self, X_pca: np.ndarray) -> None:
        """Find optimal k via silhouette score analysis."""
        k_min, k_max = self.k_range
        k_max = min(k_max, len(X_pca) - 1)

        best_score = -1
        best_k = k_min

        for k in range(k_min, k_max + 1):
            kmeans = KMeans(
                n_clusters=k,
                n_init=self.n_init,
                random_state=self.random_state,
                max_iter=300,
            )
            labels = kmeans.fit_predict(X_pca)

            if len(np.unique(labels)) < 2:
                continue

            score = silhouette_score(X_pca, labels)
            self.silhouette_scores_[k] = score

            if score > best_score:
                best_score = score
                best_k = k

        self.optimal_k_ = best_k

        # Refit with optimal k
        self.kmeans_ = KMeans(
            n_clusters=self.optimal_k_,
            n_init=self.n_init,
            random_state=self.random_state,
            max_iter=300,
        )
        self.kmeans_.fit(X_pca)
        self.cluster_centers_ = self.kmeans_.cluster_centers_

    def _create_regime_mapping(self, X_pca: np.ndarray) -> None:
        """
        Create interpretable regime labels based on cluster characteristics.

        Maps clusters to regime names based on:
        - Distance from origin (volatility proxy)
        - Position in PCA space
        """
        labels = self.kmeans_.labels_
        centers = self.cluster_centers_

        # Calculate volatility proxy (distance from origin in PCA space)
        distances = np.sqrt((centers**2).sum(axis=1))

        # Rank clusters by distance
        sorted_indices = np.argsort(distances)

        # Map to regime names
        self.regime_mapping_ = {}
        n_clusters = self.optimal_k_

        for rank, cluster_idx in enumerate(sorted_indices):
            if rank < n_clusters // 3:
                label = f"Low_Vol_{rank}"
            elif rank >= n_clusters - n_clusters // 3:
                label = f"High_Vol_{rank}"
            else:
                label = f"Neutral_{rank}"
            self.regime_mapping_[cluster_idx] = label

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict regime labels for input data.

        Args:
            data: DataFrame with features (same columns as training)

        Returns:
            Series of regime labels (strings) with same index as input
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler_.transform(data)
        X_pca = self.pca_.transform(X_scaled)
        cluster_labels = self.kmeans_.predict(X_pca)

        # Map to interpretable labels
        labels = [self.regime_mapping_.get(c, f"Cluster_{c}") for c in cluster_labels]

        return pd.Series(labels, index=data.index, name="pca_kmeans_regime")

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime probabilities based on distance to cluster centers.

        Uses inverse distance weighting to estimate membership probabilities.

        Args:
            data: DataFrame with features

        Returns:
            DataFrame with probability for each regime
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler_.transform(data)
        X_pca = self.pca_.transform(X_scaled)

        # Calculate distances to all cluster centers
        distances = np.zeros((len(X_pca), self.optimal_k_))
        for i in range(self.optimal_k_):
            distances[:, i] = np.sqrt((X_pca - self.cluster_centers_[i]) ** 2).sum(axis=1)

        # Convert to probabilities via inverse distance
        epsilon = 1e-10
        inverse_dist = 1 / (distances + epsilon)
        probs = inverse_dist / inverse_dist.sum(axis=1, keepdims=True)

        # Create column names
        columns = [self.regime_mapping_.get(i, f"Cluster_{i}") for i in range(self.optimal_k_)]

        return pd.DataFrame(
            probs,
            columns=columns,
            index=data.index,
        )

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

        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        total = sum(label_counts.values())
        proportions = {k: v / total for k, v in label_counts.items()}

        return RegimeSummary(
            name=self.name,
            n_regimes=self.optimal_k_,
            regime_labels=unique_labels,
            label_distribution=label_counts,
            label_proportions=proportions,
            metadata={
                "n_components_pca": self.pca_.n_components_,
                "explained_variance_ratio": float(self.pca_.explained_variance_ratio_.sum()),
                "silhouette_score": self.silhouette_scores_.get(self.optimal_k_, 0),
                "all_silhouette_scores": self.silhouette_scores_,
                "n_samples_fitted": self.fitted_n_samples_,
                "n_features_original": len(self.scaler_.mean_),
            },
        )

    def get_cluster_centers(self) -> pd.DataFrame:
        """
        Get cluster centers in PCA space.

        Returns:
            DataFrame with cluster centers (rows=clusters, columns=PCs)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        columns = [f"PC{i + 1}" for i in range(self.pca_.n_components_)]
        index = [self.regime_mapping_.get(i, f"Cluster_{i}") for i in range(self.optimal_k_)]

        return pd.DataFrame(
            self.cluster_centers_,
            index=index,
            columns=columns,
        )

    def get_pca_info(self) -> Dict[str, float]:
        """
        Get PCA transformation information.

        Returns:
            Dict with variance explained and component count
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return {
            "n_components": self.pca_.n_components_,
            "explained_variance_ratio": float(self.pca_.explained_variance_ratio_.sum()),
            "explained_variance_per_component": self.pca_.explained_variance_ratio_.tolist(),
        }

    def get_silhouette_analysis(self) -> pd.DataFrame:
        """
        Get silhouette score analysis for all tested k values.

        Returns:
            DataFrame with silhouette scores for each k
        """
        if not self.silhouette_scores_:
            raise ValueError("Model not fitted. Call fit() first.")

        return pd.DataFrame(
            list(self.silhouette_scores_.items()),
            columns=["k", "silhouette_score"],
        ).set_index("k")

    def transform_to_pca(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to PCA space.

        Args:
            data: DataFrame with features

        Returns:
            DataFrame with PCA-transformed features
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler_.transform(data)
        X_pca = self.pca_.transform(X_scaled)

        columns = [f"PC{i + 1}" for i in range(self.pca_.n_components_)]

        return pd.DataFrame(
            X_pca,
            columns=columns,
            index=data.index,
        )

    def get_regime_volatility(self) -> pd.DataFrame:
        """
        Get volatility metrics for each regime based on PCA space position.

        Returns:
            DataFrame with distance from origin and regime label
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        results = []
        for i in range(self.optimal_k_):
            label = self.regime_mapping_.get(i, f"Cluster_{i}")
            distance = np.sqrt((self.cluster_centers_[i] ** 2).sum())
            results.append(
                {
                    "regime": label,
                    "distance_from_origin": distance,
                    "volatility_rank": i,
                }
            )

        df = pd.DataFrame(results).set_index("regime")
        df["volatility_rank"] = df["distance_from_origin"].rank(ascending=False).astype(int)

        return df
