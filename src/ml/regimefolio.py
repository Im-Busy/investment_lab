"""
RegimeFolio: VIX Term Structure Regime Detection

VIX-based regime detection using the VIX futures term structure.

Approach:
1. Compute VIX futures curve slope (contango vs backwardation)
2. Measure term structure shape (steepness, curvature)
3. Classify regimes based on VIX curve characteristics:
   - Contango (upward sloping) → Low vol / complacent regime
   - Backwardation (downward sloping) → High vol / stress regime
   - Flat → Transition / uncertain regime

Key Insight:
- VIX futures curve shape reflects market expectations of future volatility
- Contango → Calm markets, risk-on sentiment
- Backwardation → Fear, hedging demand, risk-off sentiment

Data Requirements:
- VIX spot (^VIX)
- VIX3M (3-month VIX) OR VIX futures term structure

Example:
    >>> from src.ml.regimefolio import RegimeFolioDetector
    >>> detector = RegimeFolioDetector()
    >>> detector.fit(vix_data)
    >>> regimes = detector.predict(vix_data)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.ml.regime_base import RegimeDetectorBase, RegimeSummary


class RegimeFolioDetector(RegimeDetectorBase):
    """
    VIX term structure regime detector.

    Classifies market regimes based on the shape of the VIX futures
    term structure. The VIX curve contains information about market
    expectations of future volatility.

    Args:
        vix-spot_col: Column name for VIX spot (default: 'vix')
        vix3m_col: Column name for 3-month VIX (default: 'vix3m')
        contango_threshold: Slope threshold for contango classification
        backwardation_threshold: Slope threshold for backwardation
        lookback_window: Window for rolling slope calculation

    Attributes:
        regime_labels_: Detected regime for each observation
        vix_slope_: Computed VIX term structure slope
        regime_summary_: Summary statistics per regime
    """

    def __init__(
        self,
        vix_spot_col: str = "vix",
        vix3m_col: str = "vix3m",
        contango_threshold: float = 0.0,
        backwardation_threshold: float = -0.1,
        lookback_window: int = 5,
        min_samples: int = 30,
    ):
        super().__init__(name="RegimeFolioDetector")

        self.vix_spot_col = vix_spot_col
        self.vix3m_col = vix3m_col
        self.contango_threshold = contango_threshold
        self.backwardation_threshold = backwardation_threshold
        self.lookback_window = lookback_window
        self.min_samples = min_samples

        # Fitted attributes
        self.regime_labels_: Optional[pd.Series] = None
        self.vix_slope_: Optional[pd.Series] = None
        self.regime_summary_: Optional[pd.DataFrame] = None
        self.fitted_index_: Optional[pd.DatetimeIndex] = None
        self.scaler_ = StandardScaler()

    def fit(self, data: pd.DataFrame) -> RegimeFolioDetector:
        """
        Fit RegimeFolio detector on VIX data.

        Args:
            data: DataFrame with VIX spot and 3M VIX columns

        Returns:
            Self for method chaining

        Raises:
            ValueError: If required columns missing or insufficient data
        """
        self._validate_data(data)

        # Compute VIX term structure slope
        self._compute_vix_slope(data)

        # Classify regimes based on slope
        self._classify_regimes(data)

        # Compute summary statistics
        self._compute_regime_summary()

        self.fitted_index_ = data.index
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

        # Check for required columns
        has_spot = self.vix_spot_col in data.columns
        has_3m = self.vix3m_col in data.columns

        if not has_spot and not has_3m:
            # Check for alternative column names
            alternatives = ["VIX", "VIX3M", "vix", "vix3m", "VIX_spot", "VIX_3M"]
            found_cols = [
                c for c in data.columns if any(a.lower() in c.lower() for a in alternatives)
            ]
            if len(found_cols) < 2:
                raise ValueError(
                    f"Missing required VIX columns. Expected '{self.vix_spot_col}' and '{self.vix3m_col}'. "
                    f"Found columns: {list(data.columns)}"
                )

    def _compute_vix_slope(self, data: pd.DataFrame) -> None:
        """
        Compute VIX term structure slope.

        Slope = (VIX3M - VIX) / VIX
        - Positive slope → Contango (VIX3M > VIX)
        - Negative slope → Backwardation (VIX3M < VIX)
        """
        # Get VIX columns
        spot_col = self._find_column(data, [self.vix_spot_col, "VIX", "vix"])
        vix3m_col = self._find_column(data, [self.vix3m_col, "VIX3M", "vix3m"])

        if spot_col is None or vix3m_col is None:
            raise ValueError("Cannot find VIX spot or 3M columns")

        vix_spot = data[spot_col].values
        vix_3m = data[vix3m_col].values

        # Compute slope (normalized spread)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = (vix_3m - vix_spot) / vix_spot
            slope = np.where(np.isfinite(slope), slope, 0.0)

        self.vix_slope_ = pd.Series(slope, index=data.index, name="vix_slope")

    def _find_column(self, data: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find column matching any of the candidate names."""
        for candidate in candidates:
            if candidate in data.columns:
                return candidate
            # Case-insensitive match
            for col in data.columns:
                if col.lower() == candidate.lower():
                    return col
        return None

    def _classify_regimes(self, data: pd.DataFrame) -> None:
        """
        Classify regimes based on VIX slope.

        Regime categories:
        - Contango: slope > contango_threshold (calm, risk-on)
        - Backwardation: slope < backwardation_threshold (stress, risk-off)
        - Flat/Neutral: between thresholds (transition)
        """
        if self.vix_slope_ is None:
            raise ValueError("VIX slope not computed")

        slope = self.vix_slope_.values
        labels = []

        for s in slope:
            if s > self.contango_threshold:
                labels.append("Contango")
            elif s < self.backwardation_threshold:
                labels.append("Backwardation")
            else:
                labels.append("Neutral")

        self.regime_labels_ = pd.Series(labels, index=data.index, name="vix_regime")

    def _compute_regime_summary(self) -> None:
        """Compute summary statistics per regime."""
        if self.regime_labels_ is None or self.vix_slope_ is None:
            return

        summary_data = []
        for regime in ["Contango", "Backwardation", "Neutral"]:
            mask = self.regime_labels_ == regime
            if mask.sum() == 0:
                continue

            slope_subset = self.vix_slope_[mask]
            summary_data.append(
                {
                    "regime": regime,
                    "count": mask.sum(),
                    "slope_mean": slope_subset.mean(),
                    "slope_std": slope_subset.std(),
                    "slope_min": slope_subset.min(),
                    "slope_max": slope_subset.max(),
                }
            )

        self.regime_summary_ = pd.DataFrame(summary_data).set_index("regime")

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict regime labels for input data.

        Args:
            data: DataFrame with VIX spot and 3M columns

        Returns:
            Series of regime labels with same index as input
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        self._validate_data(data)

        # Compute slope
        spot_col = self._find_column(data, [self.vix_spot_col, "VIX", "vix"])
        vix3m_col = self._find_column(data, [self.vix3m_col, "VIX3M", "vix3m"])

        if spot_col is None or vix3m_col is None:
            raise ValueError("Cannot find VIX columns in predict data")

        vix_spot = data[spot_col].values
        vix_3m = data[vix3m_col].values

        with np.errstate(divide="ignore", invalid="ignore"):
            slope = (vix_3m - vix_spot) / vix_spot
            slope = np.where(np.isfinite(slope), slope, 0.0)

        labels = []
        for s in slope:
            if s > self.contango_threshold:
                labels.append("Contango")
            elif s < self.backwardation_threshold:
                labels.append("Backwardation")
            else:
                labels.append("Neutral")

        return pd.Series(labels, index=data.index, name="vix_regime")

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime probabilities based on slope distance.

        Uses distance from regime boundaries to estimate probabilities.

        Args:
            data: DataFrame with VIX columns

        Returns:
            DataFrame with probability for each regime
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Compute slope
        spot_col = self._find_column(data, [self.vix_spot_col, "VIX", "vix"])
        vix3m_col = self._find_column(data, [self.vix3m_col, "VIX3M", "vix3m"])

        if spot_col is None or vix3m_col is None:
            raise ValueError("Cannot find VIX columns")

        vix_spot = data[spot_col].values
        vix_3m = data[vix3m_col].values

        with np.errstate(divide="ignore", invalid="ignore"):
            slope = (vix_3m - vix_spot) / vix_spot
            slope = np.where(np.isfinite(slope), slope, 0.0)

        # Compute distances to regime centers
        contango_center = max(self.contango_threshold, 0.1)
        neutral_center = (self.contango_threshold + self.backwardation_threshold) / 2
        backwardation_center = min(self.backwardation_threshold, -0.1)

        dist_contango = np.abs(slope - contango_center)
        dist_neutral = np.abs(slope - neutral_center)
        dist_backwardation = np.abs(slope - backwardation_center)

        # Inverse distance → probability
        epsilon = 1e-6
        inverse_dist = np.column_stack(
            [
                1 / (dist_contango + epsilon),
                1 / (dist_neutral + epsilon),
                1 / (dist_backwardation + epsilon),
            ]
        )

        probs = inverse_dist / inverse_dist.sum(axis=1, keepdims=True)

        return pd.DataFrame(
            probs,
            columns=["Contango", "Neutral", "Backwardation"],
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

        if self.regime_labels_ is None:
            return RegimeSummary(
                name=self.name,
                n_regimes=0,
                regime_labels=[],
                label_distribution={},
                label_proportions={},
                metadata={"error": "No regimes detected"},
            )

        label_counts = self.regime_labels_.value_counts().to_dict()
        total = len(self.regime_labels_)
        proportions = {k: v / total for k, v in label_counts.items()}

        return RegimeSummary(
            name=self.name,
            n_regimes=len(label_counts),
            regime_labels=list(label_counts.keys()),
            label_distribution=label_counts,
            label_proportions=proportions,
            metadata={
                "contango_threshold": self.contango_threshold,
                "backwardation_threshold": self.backwardation_threshold,
                "regime_summary": self.regime_summary_.to_dict()
                if self.regime_summary_ is not None
                else {},
                "vix_slope_mean": float(self.vix_slope_.mean())
                if self.vix_slope_ is not None
                else 0,
                "vix_slope_std": float(self.vix_slope_.std()) if self.vix_slope_ is not None else 0,
            },
        )

    def get_vix_slope(self) -> pd.Series:
        """
        Get the computed VIX term structure slope.

        Returns:
            Series of slope values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.vix_slope_ is None:
            return pd.Series(dtype=float)

        return self.vix_slope_

    def get_regime_summary_df(self) -> pd.DataFrame:
        """
        Get regime summary statistics.

        Returns:
            DataFrame with statistics per regime
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.regime_summary_ is None:
            return pd.DataFrame()

        return self.regime_summary_

    def get_regime_dates(self, regime: str = "Backwardation") -> pd.DatetimeIndex:
        """
        Get dates when market was in a specific regime.

        Args:
            regime: Regime name ("Contango", "Backwardation", "Neutral")

        Returns:
            DatetimeIndex of dates in that regime
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.regime_labels_ is None:
            return pd.DatetimeIndex([])

        mask = self.regime_labels_ == regime
        return self.regime_labels_[mask].index

    def get_stress_periods(self, threshold: float = -0.15) -> pd.DataFrame:
        """
        Identify market stress periods (extreme backwardation).

        Args:
            threshold: VIX slope threshold for stress classification

        Returns:
            DataFrame with stress period start/end dates
        """
        if not self.is_fitted or self.vix_slope_ is None:
            return pd.DataFrame()

        stress_mask = self.vix_slope_ < threshold

        # Find contiguous stress periods
        stress_periods = []
        in_stress = False
        start_date = None

        for idx in self.vix_slope_.index:
            if stress_mask.loc[idx] and not in_stress:
                in_stress = True
                start_date = idx
            elif not stress_mask.loc[idx] and in_stress:
                in_stress = False
                stress_periods.append(
                    {
                        "start": start_date,
                        "end": idx,
                        "min_slope": self.vix_slope_.loc[start_date:idx].min(),
                    }
                )

        if in_stress:
            stress_periods.append(
                {
                    "start": start_date,
                    "end": self.vix_slope_.index[-1],
                    "min_slope": self.vix_slope_.loc[start_date:].min(),
                }
            )

        return pd.DataFrame(stress_periods)
