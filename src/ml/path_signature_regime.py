"""
Path Signature Regime Detection

Regime detection using rough path signatures.

Signatures capture:
- Path geometry (trends, reversals)
- Higher-order interactions (volatility clustering)
- Lead-lag relationships

The signature of a path is a mathematical object that encodes:
1. First order: Direction and magnitude of movements
2. Second order: Area enclosed (volatility, oscillation)
3. Higher orders: Complex path interactions

Regimes are clustered in signature space.

Library: esig (or iisignature)
- esig: Streaming (online) signature computation
- iisignature: Batch signature computation

Example:
    >>> from src.ml.path_signature_regime import PathSignatureRegimeDetector
    >>> detector = PathSignatureRegimeDetector(depth=3)
    >>> detector.fit(features)
    >>> regimes = detector.predict(features)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.ml.regime_base import RegimeDetectorBase, RegimeSummary


class PathSignatureRegimeDetector(RegimeDetectorBase):
    """
    Path signature regime detector.

    Uses rough path signatures to capture the geometric structure of
    price/volume paths. Signatures provide a rigorous way to encode:
    - Trend information (first level)
    - Volatility/oscillation (second level area)
    - Higher-order path interactions

    Args:
        depth: Signature depth (order of iterated integrals, default 3)
        n_clusters: Number of signature-based regimes
        window_size: Window for signature computation
        random_state: Random seed for clustering

    Attributes:
        signature_features_: Matrix of signature features
        kmeans_: Fitted k-means model
        regime_labels_: Detected regime for each observation
        silhouette_score_: Clustering quality metric
    """

    def __init__(
        self,
        depth: int = 3,
        n_clusters: int = 4,
        window_size: int = 20,
        n_init: int = 10,
        random_state: int = 42,
        min_samples: int = 50,
    ):
        super().__init__(name="PathSignatureRegimeDetector")

        self.depth = depth
        self.n_clusters = n_clusters
        self.window_size = window_size
        self.n_init = n_init
        self.random_state = random_state
        self.min_samples = min_samples

        # Fitted attributes
        self.kmeans_ = None
        self.scaler_ = StandardScaler()
        self.signature_features_: Optional[np.ndarray] = None
        self.regime_labels_: Optional[pd.Series] = None
        self.regime_mapping_: Dict[int, str] = {}
        self.cluster_centers_: Optional[np.ndarray] = None
        self.silhouette_score_: float = 0.0

    def fit(self, data: pd.DataFrame) -> PathSignatureRegimeDetector:
        """
        Fit path signature detector on input data.

        Args:
            data: DataFrame with price/feature columns

        Returns:
            Self for method chaining
        """
        self._validate_data(data)

        # Compute path signatures
        self._compute_signatures(data)

        # Standardize signature features
        X_scaled = self.scaler_.fit_transform(self.signature_features_)

        # Find optimal number of clusters
        self._find_optimal_clusters(X_scaled)

        self.is_fitted = True

        return self

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        if data.empty:
            raise ValueError("Input data cannot be empty")

        if len(data) < self.min_samples:
            raise ValueError(
                f"Insufficient samples: {len(data)} < {self.min_samples} (minimum required)"
            )

        if data.isna().any().any():
            raise ValueError("Input data contains NaN values")

    def _compute_signatures(self, data: pd.DataFrame) -> None:
        """
        Compute signature features from input data.

        Signature computation:
        1. Extract leading information (returns, volatility)
        2. Compute streaming signatures over rolling windows
        3. Flatten signature tensor into feature vector
        """
        # Select continuous columns for path construction
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.shape[1] == 0:
            raise ValueError("No numeric columns for signature computation")

        # Normalize to comparable scales
        normalized = (numeric_data - numeric_data.mean()) / (numeric_data.std() + 1e-8)

        # Compute signatures in windows
        signatures = []
        n_windows = len(normalized) - self.window_size + 1

        if n_windows < self.n_clusters:
            raise ValueError(f"Insufficient data: {n_windows} windows < {self.n_clusters} clusters")

        for i in range(n_windows):
            window_data = normalized.iloc[i : i + self.window_size].values

            # Compute signature using esig
            sig = self._compute_window_signature(window_data)
            signatures.append(sig)

        self.signature_features_ = np.array(signatures)

    def _compute_window_signature(self, window: np.ndarray) -> np.ndarray:
        """
        Compute signature for a single window.

        Uses esig library for signature computation.

        Signature levels:
        - Level 1: Mean of path increments (trend)
        - Level 2: Levy area (volatility, oscillation)
        - Level 3+: Higher-order interactions

        Args:
            window: Window of data (time_steps × features)

        Returns:
            Flattened signature feature vector
        """
        try:
            from esig import tosig

            # Convert to continuous path (cumulative sum)
            path = np.cumsum(window, axis=0)

            # Compute signature up to specified depth
            signature = tosig.stream2sig(path, self.depth)

            return signature

        except (ImportError, AttributeError):
            # Fallback: manual signature approximation
            # Handles: ImportError (esig not installed) OR AttributeError (esig broken)
            return self._approximate_signature(window)

    def _approximate_signature(self, window: np.ndarray) -> np.ndarray:
        """
        Approximate signature without esig library.

        Computes:
        - Level 1: Mean increments
        - Level 2: Covariance/variance terms
        - Level 3: Skewness-like terms
        """
        n_steps, n_features = window.shape

        features = []

        # Level 1: Mean increments (trend)
        level1 = window.mean(axis=0)
        features.extend(level1.tolist())

        # Level 2: Variance and pairwise products (volatility)
        for i in range(n_features):
            features.append(float(window[:, i].var()))
            for j in range(i + 1, n_features):
                # Cross terms (area proxy)
                cross = (window[:, i] * window[:, j]).mean()
                features.append(float(cross))

        # Level 3: Higher moments (skewness proxy)
        for i in range(n_features):
            centered = window[:, i] - window[:, i].mean()
            skew = (centered**3).mean()
            features.append(float(skew))

        return np.array(features)

    def _find_optimal_clusters(self, X: np.ndarray) -> None:
        """Find optimal number of clusters via silhouette analysis."""
        k_range = range(2, min(8, len(X) // 10))

        best_score = -1
        best_k = 2
        best_model = None
        scores = {}

        for k in k_range:
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=300,
            )
            labels = kmeans.fit_predict(X)

            if len(np.unique(labels)) < 2:
                continue

            score = silhouette_score(X, labels)
            scores[k] = score

            if score > best_score:
                best_score = score
                best_k = k
                best_model = kmeans

        self.silhouette_score_ = best_score
        self.n_clusters = best_k
        self.kmeans_ = best_model

        # Assign labels
        labels = self.kmeans_.fit_predict(X)
        self._create_regime_mapping(labels)

    def _create_regime_mapping(self, labels: np.ndarray) -> None:
        """Create interpretable regime labels from clusters."""
        # Get cluster centers in signature space
        centers = self.kmeans_.cluster_centers_

        # Score clusters by activity (signature magnitude)
        cluster_activity = []
        for i in range(self.n_clusters):
            # Activity = sum of squared signature terms
            activity = np.sum(centers[i] ** 2)
            cluster_activity.append({"cluster": i, "activity": activity})

        # Sort by activity
        sorted_clusters = sorted(cluster_activity, key=lambda x: x["activity"])

        # Map to regime names
        self.regime_mapping_ = {}
        n_clusters = self.n_clusters

        for rank, cluster_info in enumerate(sorted_clusters):
            cluster_idx = cluster_info["cluster"]

            if rank < n_clusters // 4:
                label = "Low_Activity"
            elif rank >= n_clusters - n_clusters // 4:
                label = "High_Activity"
            else:
                label = "Medium_Activity"

            # Make unique
            self.regime_mapping_[cluster_idx] = f"{label}_{rank}"

        # Create regime labels series
        regime_names = [self.regime_mapping_.get(c, f"Cluster_{c}") for c in labels]
        self.regime_labels_ = pd.Series(
            regime_names,
            index=pd.RangeIndex(start=0, stop=len(regime_names)),
            name="signature_regime",
        )

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict regime labels for new data.

        Args:
            data: DataFrame with features

        Returns:
            Series of regime labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        self._validate_data(data)

        # Compute signatures for new data
        signatures = self._compute_signatures_for_predict(data)

        # Standardize
        X_scaled = self.scaler_.transform(signatures)

        # Predict
        cluster_labels = self.kmeans_.predict(X_scaled)

        # Map to regime names
        regime_names = [self.regime_mapping_.get(c, f"Cluster_{c}") for c in cluster_labels]

        # Align index with input data (accounting for window loss)
        n_lost = len(data) - len(regime_names)
        new_index = (
            data.index[n_lost:] if len(data.index) == len(data) else data.index[: len(regime_names)]
        )

        return pd.Series(regime_names, index=new_index, name="signature_regime")

    def _compute_signatures_for_predict(self, data: pd.DataFrame) -> np.ndarray:
        """Compute signatures for predict method."""
        numeric_data = data.select_dtypes(include=[np.number])
        normalized = (numeric_data - numeric_data.mean()) / (numeric_data.std() + 1e-8)

        signatures = []
        n_windows = len(normalized) - self.window_size + 1

        for i in range(n_windows):
            window_data = normalized.iloc[i : i + self.window_size].values
            sig = self._compute_window_signature(window_data)
            signatures.append(sig)

        return np.array(signatures)

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime probabilities based on distance to centers.

        Args:
            data: DataFrame with features

        Returns:
            DataFrame with probability for each regime
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        signatures = self._compute_signatures_for_predict(data)
        X_scaled = self.scaler_.transform(signatures)

        # Distance to each center
        distances = np.zeros((len(X_scaled), self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.sqrt(
                ((X_scaled - self.kmeans_.cluster_centers_[i]) ** 2).sum(axis=1)
            )

        # Inverse distance → probability
        epsilon = 1e-6
        inverse_dist = 1 / (distances + epsilon)
        probs = inverse_dist / inverse_dist.sum(axis=1, keepdims=True)

        columns = [self.regime_mapping_.get(i, f"Cluster_{i}") for i in range(self.n_clusters)]

        return pd.DataFrame(probs, columns=columns, index=data.index[: len(probs)])

    def get_regime_summary(self) -> RegimeSummary:
        """
        Get summary information about detected regimes.

        Returns:
            RegimeSummary with label distribution and metadata
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.regime_labels_ is None:
            return RegimeSummary(
                name=self.name,
                n_regimes=0,
                regime_labels=[],
                label_distribution={},
                label_proportions={},
            )

        label_counts = self.regime_labels_.value_counts().to_dict()
        total = len(self.regime_labels_)
        proportions = {k: v / total for k, v in label_counts.items()}

        return RegimeSummary(
            name=self.name,
            n_regimes=self.n_clusters,
            regime_labels=list(label_counts.keys()),
            label_distribution=label_counts,
            label_proportions=proportions,
            metadata={
                "depth": self.depth,
                "window_size": self.window_size,
                "silhouette_score": self.silhouette_score_,
                "n_signature_features": (
                    self.signature_features_.shape[1] if self.signature_features_ is not None else 0
                ),
            },
        )

    def get_signature_features(self) -> np.ndarray:
        """
        Get the computed signature features.

        Returns:
            Array of signature feature vectors
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.signature_features_ is None:
            return np.array([])

        return self.signature_features_

    def get_silhouette_score(self) -> float:
        """
        Get clustering quality metric.

        Returns:
            Silhouette score (-1 to 1, higher is better)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.silhouette_score_

    def interpret_signature(self, signature: np.ndarray) -> Dict[str, float]:
        """
        Interpret signature components.

        Args:
            signature: Signature feature vector

        Returns:
            Dict with interpretation of signature levels
        """
        n_features_original = len(signature)

        # Rough breakdown by signature level
        # Level 1: n features (mean increments)
        # Level 2: n + n*(n-1)/2 features (variance + area)
        # Level 3: n features (higher moments)

        interpretation = {
            "level1_norm": float(np.linalg.norm(signature[: min(5, n_features_original // 3)])),
            "level2_norm": float(
                np.linalg.norm(
                    signature[
                        min(5, n_features_original // 3) : min(15, 2 * n_features_original // 3)
                    ]
                )
            ),
            "total_variation": float(np.sum(np.abs(signature))),
        }

        return interpretation
