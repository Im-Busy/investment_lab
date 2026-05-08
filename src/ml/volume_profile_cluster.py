"""
Volume-Price Profile Clustering (FS9)

Extracts volume-at-price profiles per rolling window and clusters them into
accumulation, distribution, and churning regimes using KMeans or GMM.

The volume-at-price profile (VAP) shows where the most volume traded at each
price level within a rolling window. The shape of this profile reveals market
participant behaviour:
- Accumulation: Volume concentrated near the upper end (buyers absorbing)
- Distribution: Volume concentrated near the lower end (sellers unloading)
- Churning: Volume concentrated in the middle (no directional conviction)
- Thin: Low volume overall (consolidation)

Features:
- Rolling VAP extraction from OHLCV data
- Profile shape features: skew, kurtosis, POC position, VAH/VAL
- KMeans / GMM clustering of profile shapes
- Cluster labels as features for PatternClassifier / RegimeClassifier
- Cluster transition matrix for regime changes

Example:
    >>> from src.ml.volume_profile_cluster import VolumeProfileCluster
    >>> vpc = VolumeProfileCluster(window=21, n_clusters=4)
    >>> labels = vpc.fit_predict(ohlcv_df)
    >>> features = vpc.get_cluster_features(ohlcv_df)
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class VolumeProfileCluster:
    """
    Volume-at-price profile extractor and clusterer.

    Args:
        window: Rolling window size for profile extraction
        n_clusters: Number of clusters for KMeans/GMM
        method: Clustering method ("kmeans" or "gmm")
        n_bins: Number of price bins for VAP discretization
        profile_features: List of profile features to compute
        min_samples: Minimum samples per window
        random_state: Random seed

    Attributes:
        labels_: Cluster labels for each time point
        cluster_stats_: Per-cluster statistics (POC position, VA shape, etc.)
        centroids_: Cluster centroids (profile feature space)
        transition_matrix_: Cluster-to-cluster transition probabilities
        vap_profiles_: Raw VAP profiles per window
    """

    def __init__(
        self,
        window: int = 21,
        n_clusters: int = 4,
        method: Literal["kmeans", "gmm"] = "kmeans",
        n_bins: int = 20,
        profile_features: Optional[List[str]] = None,
        min_samples: int = 10,
        random_state: int = 42,
    ):
        if window < 5:
            raise ValueError("window must be >= 5")

        self.window = window
        self.n_clusters = n_clusters
        self.method = method
        self.n_bins = n_bins
        self.min_samples = min_samples
        self.random_state = random_state

        self.profile_features = profile_features or [
            "poc_position",
            "va_ratio_top",
            "va_ratio_bottom",
            "profile_skew",
            "profile_kurtosis",
            "total_volume_ratio",
            "high_volume_density",
        ]

        self.scaler_ = StandardScaler()
        self.model_: Any = None
        self.labels_: Optional[pd.Series] = None
        self.cluster_stats_: Dict[int, Dict[str, float]] = {}
        self.centroids_: Optional[np.ndarray] = None
        self.transition_matrix_: Optional[pd.DataFrame] = None
        self.vap_profiles_: List[np.ndarray] = []
        self.profile_dates_: List[Any] = []

    def _compute_vap_profile(
        self,
        window_data: pd.DataFrame,
    ) -> Optional[np.ndarray]:
        """
        Compute volume-at-price profile for a single window.

        Args:
            window_data: OHLCV data for the window (must have Open, High, Low, Close, Volume)

        Returns:
            Array of volume per price bin, or None if insufficient data
        """
        if len(window_data) < self.min_samples:
            return None

        high = window_data["High"].values
        low = window_data["Low"].values
        volume = window_data["Volume"].values

        price_min = float(np.min(low))
        price_max = float(np.max(high))
        price_range = price_max - price_min

        if price_range <= 1e-10:
            return None

        bin_edges = np.linspace(price_min, price_max, self.n_bins + 1)
        vol_profile = np.zeros(self.n_bins)

        for i in range(len(window_data)):
            if volume[i] <= 0:
                continue
            bar_high = high[i]
            bar_low = low[i]
            bar_vol = float(volume[i])
            if bar_high <= bar_low:
                continue

            bar_range = bar_high - bar_low
            vol_per_unit = bar_vol / bar_range

            for j in range(self.n_bins):
                bin_low = bin_edges[j]
                bin_high = bin_edges[j + 1]
                overlap_low = max(bar_low, bin_low)
                overlap_high = min(bar_high, bin_high)
                if overlap_high > overlap_low:
                    overlap = overlap_high - overlap_low
                    vol_profile[j] += overlap * vol_per_unit

        total_vol = vol_profile.sum()
        if total_vol > 0:
            vol_profile /= total_vol

        return vol_profile

    def _extract_profile_features(self, profile: np.ndarray, total_vol: float) -> Dict[str, float]:
        """
        Extract descriptive features from a VAP profile.

        Args:
            profile: Normalized volume-at-price profile array
            total_vol: Total volume in the window

        Returns:
            Dict of feature name -> value
        """
        n = len(profile)
        if n == 0:
            return {f: 0.0 for f in self.profile_features}

        price_axis = np.arange(n) / (n - 1)

        total = profile.sum() or 1.0
        mean_price = np.dot(profile, price_axis) / total
        var = max(np.dot(profile, (price_axis - mean_price) ** 2) / total, 1e-10)
        std = np.sqrt(var)

        skew = (
            np.dot(profile, (price_axis - mean_price) ** 3) / (total * std**3)
            if std > 1e-10
            else 0.0
        )
        kurt = (
            np.dot(profile, (price_axis - mean_price) ** 4) / (total * std**4)
            if std > 1e-10
            else 0.0
        )

        poc_idx = int(np.argmax(profile))
        poc_position = poc_idx / max(n - 1, 1)

        cum_vol = np.cumsum(profile) / total
        val_idx = int(np.searchsorted(cum_vol, 0.15))
        vah_idx = int(np.searchsorted(cum_vol, 0.85))
        va_ratio_top = vah_idx / max(n - 1, 1)
        va_ratio_bottom = val_idx / max(n - 1, 1)

        density = profile.max() / (profile.mean() + 1e-10)

        raw_volume_pct = float(np.clip(total_vol / max(self.window * 1e6, 1), 0, 10))

        features = {}
        if "poc_position" in self.profile_features:
            features["poc_position"] = float(poc_position)
        if "va_ratio_top" in self.profile_features:
            features["va_ratio_top"] = float(va_ratio_top)
        if "va_ratio_bottom" in self.profile_features:
            features["va_ratio_bottom"] = float(va_ratio_bottom)
        if "profile_skew" in self.profile_features:
            features["profile_skew"] = float(skew)
        if "profile_kurtosis" in self.profile_features:
            features["profile_kurtosis"] = float(kurt)
        if "total_volume_ratio" in self.profile_features:
            features["total_volume_ratio"] = float(raw_volume_pct)
        if "high_volume_density" in self.profile_features:
            features["high_volume_density"] = float(density)

        return features

    def _build_feature_matrix(
        self,
        ohlcv: pd.DataFrame,
    ) -> Tuple[np.ndarray, pd.Index, List[np.ndarray]]:
        """
        Build feature matrix from rolling VAP profiles.

        Args:
            ohlcv: OHLCV DataFrame

        Returns:
            Tuple of (feature array, date index, raw profiles)
        """
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        missing = required_cols - set(ohlcv.columns)
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        feature_rows = []
        date_index = []
        self.vap_profiles_ = []
        self.profile_dates_ = []

        for i in range(self.window - 1, len(ohlcv)):
            window_data = ohlcv.iloc[i - self.window + 1 : i + 1]
            profile = self._compute_vap_profile(window_data)
            if profile is None:
                continue

            self.vap_profiles_.append(profile)
            self.profile_dates_.append(ohlcv.index[i])

            total_vol = float(window_data["Volume"].sum())
            feats = self._extract_profile_features(profile, total_vol)
            feature_rows.append(feats)
            date_index.append(ohlcv.index[i])

        if not feature_rows:
            return np.empty((0, 0)), pd.Index([]), []

        X = pd.DataFrame(feature_rows, index=date_index)
        return X.values, X.index, self.vap_profiles_

    def fit(self, ohlcv: pd.DataFrame) -> VolumeProfileCluster:
        """
        Fit the clustering model on OHLCV data.

        Args:
            ohlcv: OHLCV DataFrame with Open, High, Low, Close, Volume

        Returns:
            Self for method chaining
        """
        X, idx, _ = self._build_feature_matrix(ohlcv)

        if len(X) < self.n_clusters:
            raise ValueError(
                f"Insufficient data: {len(X)} windows < {self.n_clusters} clusters. "
                f"Increase date range or reduce window."
            )

        X_scaled = self.scaler_.fit_transform(X)

        if self.method == "kmeans":
            self.model_ = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
        elif self.method == "gmm":
            self.model_ = GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state,
                n_init=5,
                reg_covar=1e-6,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.model_.fit(X_scaled)

        cluster_labels = (
            self.model_.predict(X_scaled)
            if self.method == "kmeans"
            else self.model_.predict(X_scaled)
        )
        self.labels_ = pd.Series(cluster_labels, index=idx, name="volume_cluster")

        if hasattr(self.model_, "cluster_centers_"):
            self.centroids_ = self.model_.cluster_centers_
        elif hasattr(self.model_, "means_"):
            self.centroids_ = self.model_.means_

        self._compute_cluster_stats()
        self._compute_transition_matrix()
        self._label_clusters()

        return self

    def fit_predict(self, ohlcv: pd.DataFrame) -> pd.Series:
        """
        Fit and return cluster labels in one call.

        Args:
            ohlcv: OHLCV DataFrame

        Returns:
            Series of cluster labels with datetime index
        """
        self.fit(ohlcv)
        return self.labels_ if self.labels_ is not None else pd.Series(dtype=int)

    def predict(self, ohlcv: pd.DataFrame) -> pd.Series:
        """
        Predict cluster labels for new OHLCV data (requires prior fit).

        Args:
            ohlcv: OHLCV DataFrame

        Returns:
            Series of cluster labels
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X, idx, _ = self._build_feature_matrix(ohlcv)
        if len(X) == 0:
            return pd.Series(dtype=int)

        X_scaled = self.scaler_.transform(X)
        labels = (
            self.model_.predict(X_scaled)
            if self.method == "kmeans"
            else self.model_.predict(X_scaled)
        )
        return pd.Series(labels, index=idx, name="volume_cluster")

    def predict_proba(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Predict cluster probabilities (available only for GMM method).

        Args:
            ohlcv: OHLCV DataFrame

        Returns:
            DataFrame with probability for each cluster
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.method != "gmm":
            raise ValueError("predict_proba only available for method='gmm'")

        X, idx, _ = self._build_feature_matrix(ohlcv)
        if len(X) == 0:
            return pd.DataFrame()

        X_scaled = self.scaler_.transform(X)
        probs = self.model_.predict_proba(X_scaled)
        columns = [f"cluster_{i}" for i in range(self.n_clusters)]
        return pd.DataFrame(probs, columns=columns, index=idx)

    def _compute_cluster_stats(self) -> None:
        """Compute per-cluster statistics."""
        if self.labels_ is None or self.centroids_ is None:
            return

        self.cluster_stats_ = {}
        for c in range(self.n_clusters):
            mask = self.labels_ == c
            poc_values = []
            skew_values = []
            if len(self.vap_profiles_) > 0:
                for i, prof in enumerate(self.vap_profiles_):
                    if i < len(self.labels_) and self.labels_.iloc[i] == c:
                        n = len(prof)
                        if n > 0:
                            price_axis = np.arange(n) / max(n - 1, 1)
                            total = prof.sum() or 1.0
                            poc_values.append(float(np.argmax(prof)) / max(n - 1, 1))
                            mean_p = np.dot(prof, price_axis) / total
                            var = max(np.dot(prof, (price_axis - mean_p) ** 2) / total, 1e-10)
                            std = np.sqrt(var)
                            skew_values.append(
                                float(np.dot(prof, (price_axis - mean_p) ** 3) / (total * std**3))
                                if std > 1e-10
                                else 0.0
                            )

            self.cluster_stats_[c] = {
                "count": int(mask.sum()),
                "pct": float(mask.sum() / max(len(self.labels_), 1)),
                "avg_poc": float(np.mean(poc_values)) if poc_values else 0.0,
                "avg_skew": float(np.mean(skew_values)) if skew_values else 0.0,
            }

    def _compute_transition_matrix(self) -> None:
        """Compute cluster-to-cluster transition probability matrix."""
        if self.labels_ is None:
            return

        n = self.n_clusters
        counts = np.zeros((n, n))
        vals = self.labels_.values

        for i in range(len(vals) - 1):
            counts[int(vals[i])][int(vals[i + 1])] += 1

        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans = counts / row_sums

        self.transition_matrix_ = pd.DataFrame(
            trans,
            index=[f"from_{i}" for i in range(n)],
            columns=[f"to_{i}" for i in range(n)],
        )

    def _label_clusters(self) -> None:
        """Assign interpretable labels to clusters based on profile features."""
        if self.cluster_stats_ is None:
            return

        ranked = sorted(
            self.cluster_stats_.items(),
            key=lambda x: x[1]["avg_poc"],
            reverse=True,
        )

        n = len(ranked)
        n_accum = max(1, n // 3)
        n_dist = max(1, n // 3)

        self._interpretable_labels_: Dict[int, str] = {}
        for rank, (c_id, stats) in enumerate(ranked):
            if rank < n_accum:
                self._interpretable_labels_[c_id] = "Accumulation"
            elif rank >= n - n_dist:
                self._interpretable_labels_[c_id] = "Distribution"
            elif stats["avg_skew"] > 0.2:
                self._interpretable_labels_[c_id] = "Churning_BiasUp"
            elif stats["avg_skew"] < -0.2:
                self._interpretable_labels_[c_id] = "Churning_BiasDown"
            else:
                self._interpretable_labels_[c_id] = "Churning_Neutral"

    def get_cluster_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Generate cluster labels as features for downstream classifiers.

        Returns a multi-column DataFrame with:
        - cluster_label: integer cluster assignment
        - cluster_name: interpretable label (Accumulation/Distribution/Churning)
        - cluster_onehot_*: one-hot encoded cluster membership

        Args:
            ohlcv: OHLCV DataFrame

        Returns:
            DataFrame with cluster feature columns
        """
        labels = self.predict(ohlcv)
        result = pd.DataFrame(index=labels.index)
        result["cluster_label"] = labels

        if hasattr(self, "_interpretable_labels_"):
            result["cluster_name"] = labels.map(self._interpretable_labels_)

        for c in range(self.n_clusters):
            result[f"cluster_onehot_{c}"] = (labels == c).astype(int)

        return result

    def get_transition_matrix(self) -> Optional[pd.DataFrame]:
        """Get the cluster transition probability matrix."""
        return self.transition_matrix_

    def get_regime_at(self, date: pd.Timestamp) -> Optional[str]:
        """
        Get the interpretable regime label at a specific date.

        Args:
            date: Timestamp to query

        Returns:
            Interpretable label or None
        """
        if self.labels_ is None:
            return None
        if date not in self.labels_.index:
            return None
        if not hasattr(self, "_interpretable_labels_"):
            return None
        return self._interpretable_labels_.get(int(self.labels_.loc[date]))

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the clustering results.

        Returns:
            Dict with cluster stats, transition matrix, and method info
        """
        if self.labels_ is None:
            return {"status": "not_fitted"}

        return {
            "method": self.method,
            "window": self.window,
            "n_clusters": self.n_clusters,
            "n_samples": len(self.labels_),
            "cluster_distribution": {
                k: v for k, v in enumerate(self.labels_.value_counts().sort_index().to_dict())
            },
            "interpretable_labels": getattr(self, "_interpretable_labels_", {}),
            "transition_probabilities": (
                self.transition_matrix_.to_dict() if self.transition_matrix_ is not None else None
            ),
        }
