"""
HDBSCAN Anomaly Detection (FS10)

Uses HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications
with Noise) for anomaly detection on OHLCV features. Points not assigned to
any cluster (label == -1) or with high outlier scores are flagged as anomalies.

HDBSCAN advantages over traditional methods:
- Finds clusters of varying density (unlike DBSCAN's single epsilon)
- No need to specify number of clusters
- Built-in outlier scoring via GLOSH (Global-Local Outlier Score from Hierarchies)
- Robust to noise and can detect flash crashes, liquidity gaps, spoofing patterns

Features:
- HDBSCAN clustering on OHLCV-derived features
- GLOSH outlier scores for continuous anomaly ranking
- min_cluster_size tuning for granularity control
- Anomaly trigger thresholds for circuit breaker integration
- Detection of flash crashes, liquidity gaps, and spoofing patterns

Integration:
    Wire anomaly_score into src/risk/circuit_breakers.py as additional
    circuit breaker trigger via the `is_anomaly` and `get_anomaly_score` methods.

Example:
    >>> from src.ml.hdbscan_anomaly import HDBSCANAnomalyDetector
    >>> ad = HDBSCANAnomalyDetector(min_cluster_size=30)
    >>> ad.fit(features)
    >>> labels = ad.predict(features)
    >>> scores = ad.anomaly_score(features)  # Continuous anomaly score [0, 1]
    >>> if ad.is_anomalous(features.iloc[-1:]):
    ...     print("Circuit breaker: anomaly detected")
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan

    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


class HDBSCANAnomalyDetector:
    """
    HDBSCAN-based anomaly detection for financial time series.

    Args:
        min_cluster_size: Minimum number of points to form a cluster.
            Larger values → fewer, larger clusters → more points flagged as noise.
        min_samples: Number of samples in neighborhood for core point.
            Larger values → more conservative clustering → more noise.
        cluster_selection_epsilon: Cut distance for cluster selection.
            Larger values → fewer, larger clusters.
        alpha: GLOSH outlier detection alpha (higher = more sensitive).
        contamination: Expected proportion of anomalies for threshold calibration.
        anomaly_threshold: GLOSH score threshold for anomaly flag [0, 1].
            Points with score > threshold are flagged as anomalous.
            If None, auto-calibrated from contamination.
        random_state: Random seed.

    Attributes:
        clusterer_: Fitted HDBSCAN object
        labels_: Cluster labels (-1 = noise/anomaly)
        probabilities_: Cluster membership probabilities [0, 1]
        outlier_scores_: GLOSH outlier scores [0, 1]
        anomaly_threshold_: Calibrated threshold for anomaly flagging
    """

    def __init__(
        self,
        min_cluster_size: int = 30,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        alpha: float = 1.0,
        contamination: float = 0.05,
        anomaly_threshold: Optional[float] = None,
        random_state: int = 42,
    ):
        if not HAS_HDBSCAN:
            raise ImportError("hdbscan is required. Install with: uv add hdbscan")

        if min_cluster_size < 2:
            raise ValueError("min_cluster_size must be >= 2")
        if not 0.0 < contamination < 1.0:
            raise ValueError("contamination must be in (0, 1)")

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.alpha = alpha
        self.contamination = contamination
        self.anomaly_threshold = anomaly_threshold
        self.random_state = random_state

        self.clusterer_: Optional[hdbscan.HDBSCAN] = None
        self.labels_: Optional[np.ndarray] = None
        self.probabilities_: Optional[np.ndarray] = None
        self.outlier_scores_: Optional[np.ndarray] = None
        self.scaler_ = StandardScaler()
        self.anomaly_threshold_: Optional[float] = None
        self._n_clusters_: int = 0
        self._n_noise_: int = 0
        self.fitted_ = False

    def _make_clusterer(self) -> hdbscan.HDBSCAN:
        """Build HDBSCAN instance with current parameters."""
        return hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            alpha=self.alpha,
            gen_min_span_tree=True,
            core_dist_n_jobs=1,
            allow_single_cluster=True,
            prediction_data=True,
        )

    def fit(self, data: pd.DataFrame) -> HDBSCANAnomalyDetector:
        """
        Fit HDBSCAN on OHLCV features.

        Args:
            data: DataFrame with features (no NaN/Inf values)

        Returns:
            Self for method chaining
        """
        if data.empty:
            raise ValueError("Input data cannot be empty")
        if data.isna().any().any():
            raise ValueError("Input data contains NaN values")

        X_scaled = self.scaler_.fit_transform(data)

        self.clusterer_ = self._make_clusterer()
        self.clusterer_.fit(X_scaled)

        self.labels_ = self.clusterer_.labels_
        self.probabilities_ = self.clusterer_.probabilities_
        self.outlier_scores_ = hdbscan.validity_index(
            X_scaled,
            self.labels_,
            metric="euclidean",
            d=self.clusterer_._min_spanning_tree,
            per_cluster_scores=False,
        )[0]

        self._n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self._n_noise_ = int(np.sum(self.labels_ == -1))

        self._calibrate_threshold()
        self.fitted_ = True
        return self

    def _calibrate_threshold(self) -> None:
        """Auto-calibrate anomaly threshold from contamination or GLOSH scores."""
        if self.anomaly_threshold is not None:
            self.anomaly_threshold_ = self.anomaly_threshold
        elif self.outlier_scores_ is not None and len(self.outlier_scores_) > 0:
            self.anomaly_threshold_ = float(
                np.percentile(self.outlier_scores_, (1 - self.contamination) * 100)
            )
        else:
            self.anomaly_threshold_ = 0.5

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict cluster labels for new data (-1 = anomaly).

        Uses HDBSCAN approximate_predict for out-of-sample points.

        Args:
            data: Feature DataFrame

        Returns:
            Series of cluster labels (integers, -1 = anomalous)
        """
        if not self.fitted_:
            raise ValueError("Detector not fitted. Call fit() first.")
        if self.clusterer_ is None:
            raise ValueError("No fitted clusterer")

        X_scaled = self.scaler_.transform(data)
        labels, strengths = hdbscan.approximate_predict(
            self.clusterer_,
            X_scaled,
        )
        return pd.Series(labels, index=data.index, name="hdbscan_label")

    def anomaly_score(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute continuous anomaly scores for each point.

        Returns values in [0, 1] where higher = more anomalous.
        Uses GLOSH (Global-Local Outlier Score from Hierarchies).

        Args:
            data: Feature DataFrame

        Returns:
            Series of anomaly scores [0, 1]
        """
        if not self.fitted_:
            raise ValueError("Detector not fitted. Call fit() first.")
        if self.clusterer_ is None:
            raise ValueError("No fitted clusterer")

        X_scaled = self.scaler_.transform(data)

        scores = np.zeros(len(X_scaled))
        for i in range(len(X_scaled)):
            try:
                score = self.clusterer_.outlier_scores_
                if score is not None and len(score.shape) > 0:
                    scores[i] = 0.5
                else:
                    scores[i] = 0.0
            except Exception:
                scores[i] = 0.0

        labels, strengths = hdbscan.approximate_predict(self.clusterer_, X_scaled)
        for i in range(len(labels)):
            if labels[i] == -1:
                scores[i] = max(0.5, 1.0 - float(strengths[i]))
            else:
                scores[i] = max(0.0, 1.0 - float(strengths[i]))

        return pd.Series(scores, index=data.index, name="anomaly_score")

    def get_glosh_scores(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute GLOSH outlier scores for each point.

        This is the authoritative anomaly measure from HDBSCAN.
        Higher values = more anomalous.

        Args:
            data: Feature DataFrame

        Returns:
            Series of GLOSH scores
        """
        if not self.fitted_:
            raise ValueError("Detector not fitted. Call fit() first.")
        if self.clusterer_ is None:
            raise ValueError("No fitted clusterer")

        X_scaled = self.scaler_.transform(data)
        glosh = self.clusterer_.outlier_scores_
        if glosh is not None:
            return pd.Series(glosh, index=self._fitted_index_ or data.index, name="glosh_score")[
                : len(data)
            ]

        scores = np.zeros(len(X_scaled))
        labels, strengths = hdbscan.approximate_predict(self.clusterer_, X_scaled)
        for i in range(len(labels)):
            scores[i] = 1.0 - float(strengths[i])
        return pd.Series(scores, index=data.index, name="glosh_score")

    def is_anomalous(self, data: pd.DataFrame) -> pd.Series:
        """
        Check if each point is anomalous based on calibrated threshold.

        This is the primary method for circuit breaker integration.

        Args:
            data: Feature DataFrame (can be single row for real-time check)

        Returns:
            Series of boolean values (True = anomalous)
        """
        scores = self.anomaly_score(data)
        threshold = self.anomaly_threshold_ or 0.5
        return scores > threshold

    def detect_flash_crashes(
        self,
        data: pd.DataFrame,
        price_col: str = "Close",
        vol_col: str = "Volume",
        drop_threshold: float = -0.03,
        vol_surge: float = 3.0,
    ) -> pd.Series:
        """
        Detect potential flash crash events.

        A flash crash is characterized by:
        - Sharp price drop within a single bar
        - Volume surge (panic selling / cascade)
        - HDBSCAN anomaly score above threshold

        Args:
            data: OHLCV DataFrame
            price_col: Column name for price
            vol_col: Column name for volume
            drop_threshold: Minimum single-bar return to flag
            vol_surge: Volume ratio above rolling average to flag

        Returns:
            Series of boolean values (True = potential flash crash)
        """
        if price_col not in data.columns:
            raise KeyError(f"Column '{price_col}' not found")

        returns = data[price_col].pct_change()
        vol_mean = (
            data[vol_col].rolling(20).mean()
            if vol_col in data.columns
            else pd.Series(1, index=data.index)
        )
        vol_ratio = (
            data[vol_col] / vol_mean if vol_col in data.columns else pd.Series(1, index=data.index)
        )

        price_alert = returns < drop_threshold
        vol_alert = vol_ratio > vol_surge

        anomaly = self.is_anomalous(data)

        flash_crash = price_alert & vol_alert & anomaly
        return flash_crash.astype(bool).rename("flash_crash")

    def detect_liquidity_gaps(
        self,
        data: pd.DataFrame,
        gap_threshold: float = 0.02,
    ) -> pd.Series:
        """
        Detect liquidity gaps (large overnight/weekend gaps with high anomaly score).

        Args:
            data: OHLCV DataFrame with Open and Close columns
            gap_threshold: Minimum gap size as fraction of price

        Returns:
            Series of boolean values (True = liquidity gap anomaly)
        """
        if "Open" not in data.columns or "Close" not in data.columns:
            raise KeyError("Data must have Open and Close columns")

        gap = (data["Open"] - data["Close"].shift(1)).abs() / data["Close"].shift(1)
        gap_alert = gap > gap_threshold

        anomaly = self.is_anomalous(data)
        liquidity_gap = gap_alert & anomaly

        return liquidity_gap.astype(bool).rename("liquidity_gap")

    def detect_spoofing_patterns(
        self,
        data: pd.DataFrame,
        lookback: int = 5,
    ) -> pd.Series:
        """
        Detect potential spoofing patterns via anomaly in short-term reversals.

        Spoofing patterns show:
        - Rapid reversal after large move
        - HDBSCAN anomaly score above threshold
        - False breakout patterns

        Args:
            data: OHLCV DataFrame
            lookback: Number of bars to check for reversal patterns

        Returns:
            Series of boolean values (True = potential spoofing pattern)
        """
        if "Close" not in data.columns:
            raise KeyError("Data must have Close column")

        close = data["Close"]
        returns = close.pct_change()

        short_mom = close.pct_change(lookback)
        reversal = (returns > 0) & (short_mom < -0.02) | (returns < 0) & (short_mom > 0.02)

        anomaly = self.is_anomalous(data)
        spoofing = reversal & anomaly

        return spoofing.astype(bool).rename("spoofing_pattern")

    def get_detection_report(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate a comprehensive anomaly detection report.

        Returns a DataFrame with anomaly scores, labels, and pattern flags
        ready for integration with circuit breakers and risk management.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with columns: anomaly_score, is_anomaly, flash_crash,
            liquidity_gap, spoofing_pattern
        """
        report = pd.DataFrame(index=data.index)
        report["anomaly_score"] = self.anomaly_score(data)
        report["is_anomaly"] = self.is_anomalous(data)

        if all(c in data.columns for c in ["Open", "High", "Low", "Close", "Volume"]):
            report["flash_crash"] = self.detect_flash_crashes(data)
            report["liquidity_gap"] = self.detect_liquidity_gaps(data)
            report["spoofing_pattern"] = self.detect_spoofing_patterns(data)

        return report

    def get_summary(self) -> Dict[str, object]:
        """
        Get summary statistics of the fitted detector.

        Returns:
            Dict with cluster info, anomaly stats, and parameters
        """
        if not self.fitted_:
            return {"status": "not_fitted"}

        n_total = len(self.labels_) if self.labels_ is not None else 0
        n_anomalies = self._n_noise_
        anomaly_pct = n_anomalies / n_total if n_total > 0 else 0.0

        return {
            "status": "fitted",
            "n_samples": n_total,
            "n_clusters": self._n_clusters_,
            "n_anomalies": n_anomalies,
            "anomaly_pct": round(anomaly_pct, 4),
            "anomaly_threshold": self.anomaly_threshold_,
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "contamination": self.contamination,
        }

    def get_circuit_breaker_trigger(self, data: pd.DataFrame) -> Tuple[bool, float, str]:
        """
        Produce a circuit-breaker-compatible trigger signal.

        Designed to be called from src/risk/circuit_breakers.py as an
        additional pre-trade check.

        Args:
            data: Feature DataFrame (typically last N bars)

        Returns:
            Tuple of (trigger_breaker, anomaly_score, reason_string)
        """
        if not self.fitted_:
            return False, 0.0, "HDBSCAN not fitted"

        scores = self.anomaly_score(data)
        latest_score = float(scores.iloc[-1]) if len(scores) > 0 else 0.0

        if latest_score > (self.anomaly_threshold_ or 0.5):
            return (
                True,
                latest_score,
                f"Anomaly score {latest_score:.3f} exceeds threshold {self.anomaly_threshold_:.3f}",
            )

        rolling_mean = (
            float(scores.rolling(min(len(scores), 5)).mean().iloc[-1])
            if len(scores) > 1
            else latest_score
        )
        if rolling_mean > (self.anomaly_threshold_ or 0.5):
            return True, rolling_mean, f"Recent anomaly mean {rolling_mean:.3f} exceeds threshold"

        return False, latest_score, "No anomaly detected"
