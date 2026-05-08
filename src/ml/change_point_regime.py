"""
Change-Point Regime Detection (FS7)

Uses the `ruptures` library for online change-point detection in time series,
wrapping PELT and BinSeg algorithms behind the RegimeDetectorBase interface.

PELT (Pruned Exact Linear Time): Finds optimal change points by minimising a
    cost function + penalty. Best when the number of change points is unknown.
BinSeg (Binary Segmentation): Greedy top-down search for a fixed number of
    change points. Faster for known k.

Features:
- Penality-based (PELT) and fixed-segment (BinSeg) detection
- Map change-point segments to interpretable regime labels
- Soft regime probabilities via distance-to-boundary weighting
- Merge real-time change-point signals with rolling-window features
- Follows RegimeDetectorBase interface (fit/predict/predict_proba/get_regime_summary)

Example:
    >>> from src.ml.change_point_regime import ChangePointRegimeDetector
    >>> detector = ChangePointRegimeDetector(method="pelt", pen=5.0)
    >>> detector.fit(features)
    >>> regimes = detector.predict(features)
    >>> probs = detector.predict_proba(features)  # Soft by boundary distance
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.preprocessing import StandardScaler

from src.ml.regime_base import RegimeDetectorBase, RegimeSummary

_CP_METHODS = ("pelt", "binseg", "window", "bottomup")


class ChangePointRegimeDetector(RegimeDetectorBase):
    """
    Change-point based regime detector using ruptures.

    Detects structural breaks in time series features and assigns each
    segment between change points a regime label. Soft probabilities
    are derived from distance-to-nearest-boundary.

    Args:
        method: Detection algorithm ("pelt", "binseg", "window", "bottomup")
        pen: Penalty value for PELT (higher = fewer change points)
        model: Cost model ("l2", "l1", "rbf", "normal", "ar", "rank")
        n_cps: Number of change points for BinSeg (ignored for PELT)
        min_size: Minimum segment length between change points
        jump: Subsampling factor for faster detection (BinSeg only)
        random_state: Random seed

    Attributes:
        change_points_: Detected change point indices (sorted)
        segment_labels_: Regime label for each segment
        boundary_scores_: Distance-to-boundary scores [0, 1] for each point
        segment_stats_: Per-segment mean/variance statistics
    """

    def __init__(
        self,
        method: Literal["pelt", "binseg", "window", "bottomup"] = "pelt",
        pen: float = 5.0,
        model: str = "l2",
        n_cps: int = 5,
        min_size: int = 30,
        jump: int = 5,
        random_state: int = 42,
    ):
        super().__init__(name="ChangePointRegimeDetector")

        if method not in _CP_METHODS:
            raise ValueError(f"method must be one of {_CP_METHODS}")

        self.method = method
        self.pen = pen
        self.model = model
        self.n_cps = n_cps
        self.min_size = min_size
        self.jump = jump
        self.random_state = random_state

        self.change_points_: List[int] = []
        self.segment_labels_: Dict[int, str] = {}
        self.boundary_scores_: Optional[np.ndarray] = None
        self.segment_stats_: Dict[int, Dict[str, float]] = {}
        self.scaler_ = StandardScaler()
        self._labels_: pd.Series = pd.Series(dtype=str)
        self.fitted_n_samples_ = 0

    @staticmethod
    def _build_algorithm(
        method: str,
        pen: float,
        model: str,
        n_cps: int,
        min_size: int,
        jump: int,
    ) -> object:
        """Build the appropriate ruptures algorithm instance."""
        if method == "pelt":
            algo = rpt.Pelt(
                model=model,
                min_size=min_size,
                jump=jump,
            )
        elif method == "binseg":
            algo = rpt.Binseg(
                model=model,
                min_size=min_size,
                jump=jump,
            )
        elif method == "window":
            algo = rpt.Window(
                model=model,
                width=min_size * 2,
                min_size=min_size,
                jump=jump,
            )
        elif method == "bottomup":
            algo = rpt.BottomUp(
                model=model,
                min_size=min_size,
                jump=jump,
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        return algo

    def fit(self, data: pd.DataFrame) -> ChangePointRegimeDetector:
        """
        Fit change-point detection on input features.

        Args:
            data: DataFrame with features (columns) and observations (rows)

        Returns:
            Self for method chaining
        """
        if data.empty:
            raise ValueError("Input data cannot be empty")

        if len(data) < self.min_size * 3:
            raise ValueError(f"Need at least {self.min_size * 3} samples, got {len(data)}")

        if data.isna().any().any():
            raise ValueError("Input data contains NaN values")

        X_scaled = self.scaler_.fit_transform(data)
        signal = X_scaled.mean(axis=1)

        algo = self._build_algorithm(
            self.method,
            self.pen,
            self.model,
            self.n_cps,
            self.min_size,
            self.jump,
        )

        algo.fit(signal.reshape(-1, 1))

        if self.method == "pelt":
            result = algo.predict(pen=self.pen)
        elif self.method in ("binseg", "bottomup"):
            result = algo.predict(n_bkps=self.n_cps)
        elif self.method == "window":
            result = algo.predict(n_bkps=self.n_cps)
        else:
            result = algo.predict(pen=self.pen)

        self.change_points_ = [0] + sorted(result)
        self.change_points_ = [cp for cp in self.change_points_ if cp < len(data)]

        self._build_segment_stats(X_scaled)
        self._build_boundary_scores(len(data))

        labels = np.zeros(len(data), dtype=int)
        for i, (start, end) in enumerate(self._segment_ranges()):
            labels[start:end] = i

        self._labels_ = pd.Series(
            [self._label_segment(self.segment_stats_[i]) for i in labels],
            index=data.index,
        )

        self.is_fitted = True
        self.fitted_n_samples_ = len(data)
        return self

    def _segment_ranges(self):
        """Generate (start, end) index pairs for each segment."""
        for i in range(len(self.change_points_) - 1):
            yield self.change_points_[i], self.change_points_[i + 1]

    def _build_segment_stats(self, X: np.ndarray) -> None:
        """Compute per-segment mean and volatility."""
        self.segment_stats_ = {}
        for i, (start, end) in enumerate(self._segment_ranges()):
            seg = X[start:end]
            self.segment_stats_[i] = {
                "mean": float(np.mean(seg)),
                "std": float(np.std(seg)),
                "size": end - start,
                "trend": float(
                    np.mean(seg[-min(10, len(seg)) :]) - np.mean(seg[: min(10, len(seg))])
                ),
            }

    def _label_segment(self, stats: Dict[str, float]) -> str:
        """Assign interpretable regime label to a segment."""
        trend = stats["trend"]
        vol = stats["std"]
        if trend > 0.5 and vol < 0.8:
            return "Bull_Calm"
        elif trend > 0.5 and vol >= 0.8:
            return "Bull_Volatile"
        elif trend < -0.5 and vol < 0.8:
            return "Bear_Calm"
        elif trend < -0.5 and vol >= 0.8:
            return "Bear_Volatile"
        elif vol < 0.5:
            return "Ranging_LowVol"
        elif vol >= 0.8:
            return "Ranging_HighVol"
        else:
            return "Transition"

    def _build_boundary_scores(self, n: int) -> None:
        """Compute distance-to-boundary scores for soft probability."""
        boundaries = set(self.change_points_)
        self.boundary_scores_ = np.ones(n, dtype=float)
        for i in range(n):
            dist = min(abs(i - b) for b in boundaries) if boundaries else n
            self.boundary_scores_[i] = min(dist / self.min_size, 1.0)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict regime labels for input data.

        Uses the full signal for prediction — refits change points
        on the provided data (online detection).

        Args:
            data: DataFrame with features

        Returns:
            Series of regime labels (strings)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler_.transform(data)
        signal = X_scaled.mean(axis=1)

        algo = self._build_algorithm(
            self.method,
            self.pen,
            self.model,
            self.n_cps,
            self.min_size,
            self.jump,
        )
        algo.fit(signal.reshape(-1, 1))

        if self.method == "pelt":
            result = algo.predict(pen=self.pen)
        elif self.method in ("binseg", "bottomup", "window"):
            result = algo.predict(n_bkps=self.n_cps)
        else:
            result = algo.predict(pen=self.pen)

        cp = [0] + sorted(result)
        cp = [c for c in cp if c < len(data)]

        labels = np.zeros(len(data), dtype=int)
        for i in range(len(cp) - 1):
            labels[cp[i] : cp[i + 1]] = i

        stats = {}
        for i in range(len(cp) - 1):
            seg = X_scaled[cp[i] : cp[i + 1]]
            stats[i] = {
                "mean": float(np.mean(seg)),
                "std": float(np.std(seg)),
                "size": cp[i + 1] - cp[i],
                "trend": float(
                    np.mean(seg[-min(10, len(seg)) :]) - np.mean(seg[: min(10, len(seg))])
                ),
            }

        return pd.Series(
            [self._label_segment(stats[i]) for i in labels],
            index=data.index,
            name="cp_regime",
        )

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime probabilities based on boundary distance.

        Points near change-point boundaries receive higher entropy
        probabilities (more uncertain), while points deep within a
        segment receive higher confidence.

        Args:
            data: DataFrame with features

        Returns:
            DataFrame with probability for each regime
        """
        labels = self.predict(data)
        unique_labels = sorted(labels.unique())
        n_labels = len(unique_labels)
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}

        X_scaled = self.scaler_.transform(data)
        signal = X_scaled.mean(axis=1)
        algo_pts = self._build_algorithm(
            self.method,
            self.pen,
            self.model,
            self.n_cps,
            self.min_size,
            self.jump,
        )
        algo_pts.fit(signal.reshape(-1, 1))

        if self.method == "pelt":
            result = algo_pts.predict(pen=self.pen)
        else:
            result = algo_pts.predict(n_bkps=self.n_cps)

        boundaries = set([0] + sorted(result))
        probs = np.zeros((len(data), n_labels))

        for i in range(len(data)):
            dist = min(abs(i - b) for b in boundaries)
            confidence = min(dist / self.min_size, 0.95)
            row = np.ones(n_labels) * ((1 - confidence) / (n_labels - 1))
            row[label_to_idx[labels.iloc[i]]] = confidence
            probs[i] = row / row.sum()

        return pd.DataFrame(probs, columns=unique_labels, index=data.index)

    def get_regime_summary(self) -> RegimeSummary:
        """
        Get summary information about detected regimes.

        Returns:
            RegimeSummary with label distribution and change-point metadata
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        label_counts = self._labels_.value_counts().to_dict()
        total = len(self._labels_)
        proportions = {k: v / total for k, v in label_counts.items()}

        return RegimeSummary(
            name=self.name,
            n_regimes=len(self.segment_stats_),
            regime_labels=list(label_counts.keys()),
            label_distribution=label_counts,
            label_proportions=proportions,
            metadata={
                "method": self.method,
                "pen": self.pen,
                "n_change_points": len(self.change_points_) - 2,
                "min_size": self.min_size,
                "change_point_indices": self.change_points_,
                "segment_sizes": [s["size"] for s in self.segment_stats_.values()],
                "n_samples_fitted": self.fitted_n_samples_,
            },
        )

    def get_change_points(self) -> List[int]:
        """Get detected change point indices (excluding start/end)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.change_points_[1:-1]

    def get_segment_boundaries(self) -> pd.DataFrame:
        """
        Get segment boundaries as a DataFrame with regime labels.

        Returns:
            DataFrame with columns: start_idx, end_idx, regime, size, trend, std
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        rows = []
        for i, (start, end) in enumerate(self._segment_ranges()):
            stats = self.segment_stats_[i]
            rows.append(
                {
                    "start_idx": start,
                    "end_idx": end,
                    "regime": self._label_segment(stats),
                    "size": stats["size"],
                    "trend": stats["trend"],
                    "std": stats["std"],
                }
            )

        return pd.DataFrame(rows)

    def merge_with_existing(
        self,
        data: pd.DataFrame,
        existing_regimes: pd.Series,
        weight: float = 0.5,
    ) -> pd.Series:
        """
        Merge change-point regime signals with existing rolling-window regimes.

        Uses weighted voting: when change-point regime differs from existing
        regime, the weight parameter determines which one dominates.

        Args:
            data: Feature DataFrame
            existing_regimes: Existing regime labels (from HMM/GMM/etc.)
            weight: Weight for change-point regime [0, 1]

        Returns:
            Series of merged regime labels
        """
        cp_regimes = self.predict(data)
        merged = cp_regimes.copy()

        for i, (cp_r, ex_r) in enumerate(zip(cp_regimes, existing_regimes)):
            prob = self.boundary_scores_[i] if self.boundary_scores_ is not None else weight
            if prob > weight:
                merged.iloc[i] = cp_r
            else:
                merged.iloc[i] = ex_r

        return merged
