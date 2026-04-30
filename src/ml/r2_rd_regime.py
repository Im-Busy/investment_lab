"""
Robust Rolling Regime Detection (R²-RD)

Rolling R² regime detection via structural break testing.

Approach:
1. Fit rolling window regression (returns ~ lagged returns + volatility)
2. Compute R² statistic for each window
3. Detect breaks where R² drops significantly (Chow test)
4. Segment data into regimes based on break points

This method detects structural changes in the return generation process
by monitoring the predictive power (R²) of autoregressive models.

Key Insight:
- High R² → Stable regime (predictable dynamics)
- Low R² → Transition/break regime (structural instability)

Reference:
- "Robust Rolling Regime Detection" - Statistical structural break literature
- Chow test for parameter stability

Example:
    >>> from src.ml.r2_rd_regime import R2RDRegimeDetector
    >>> detector = R2RDRegimeDetector(window_size=60)
    >>> detector.fit(features)
    >>> regimes = detector.predict(features)
    >>> breaks = detector.get_breakpoints()
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.ml.regime_base import RegimeDetectorBase, RegimeSummary


class R2RDRegimeDetector(RegimeDetectorBase):
    """
    Rolling R² regime detector via structural break testing.

    Detects regime changes by monitoring the stability of regression
    relationships in rolling windows. Structural breaks are identified
    where R² drops significantly.

    Args:
        window_size: Rolling window size for R² computation
        min_r2_threshold: R² threshold for stable regime detection
        break_threshold: Number of std devs for break detection
        min_regime_length: Minimum bars between breaks
        use_chow_test: Use Chow test formalism (default True)

    Attributes:
        breakpoints_: Detected structural break dates
        regime_segments_: DataFrame with regime start/end and labels
        r2_series_: Rolling R² values over time
        regression_stats_: Coefficients and fit stats per window
    """

    def __init__(
        self,
        window_size: int = 60,
        min_r2_threshold: float = 0.3,
        break_threshold: float = 2.0,
        min_regime_length: int = 10,
        use_chow_test: bool = True,
        min_samples: int = 50,
    ):
        super().__init__(name="R2RDRegimeDetector")

        self.window_size = window_size
        self.min_r2_threshold = min_r2_threshold
        self.break_threshold = break_threshold
        self.min_regime_length = min_regime_length
        self.use_chow_test = use_chow_test
        self.min_samples = min_samples

        # Fitted attributes
        self.breakpoints_: Optional[pd.DataFrame] = None
        self.regime_segments_: Optional[pd.DataFrame] = None
        self.r2_series_: Optional[pd.Series] = None
        self.regression_stats_: Optional[pd.DataFrame] = None
        self.fitted_index_: Optional[pd.DatetimeIndex] = None
        self.n_regimes_: int = 0

    def fit(self, data: pd.DataFrame) -> R2RDRegimeDetector:
        """
        Fit R²-RD detector on input data.

        Computes rolling R² statistics and detects structural breaks.

        Args:
            data: DataFrame with features (must include 'close' and/or 'returns')

        Returns:
            Self for method chaining

        Raises:
            ValueError: If insufficient data or features
        """
        self._validate_data(data)

        # Compute returns if not present
        if "returns" not in data.columns:
            if "close" in data.columns:
                data = data.copy()
                data["returns"] = data["close"].pct_change()
            elif data.shape[1] > 0:
                data = data.copy()
                data["returns"] = data.iloc[:, 0].pct_change()
            else:
                raise ValueError("Input must have 'close', 'returns', or feature columns")

        # Remove NaN from pct_change
        data_clean = data.dropna()

        # Compute rolling R²
        self._compute_rolling_r2(data_clean)

        # Detect breakpoints
        self._detect_breakpoints()

        # Segment into regimes
        self._segment_regimes(data_clean.index)

        # Compute regression statistics
        self._compute_regression_stats(data_clean)

        self.fitted_index_ = data_clean.index
        self.is_fitted = True

        return self

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data before fitting."""
        if data.empty:
            raise ValueError("Input data cannot be empty")

        if len(data) < self.min_samples:
            raise ValueError(
                f"Insufficient samples: {len(data)} < {self.min_samples} (minimum required)"
            )

        if len(data.columns) == 0:
            raise ValueError("Input data has no features")

    def _compute_rolling_r2(self, data: pd.DataFrame) -> None:
        """
        Compute rolling R² statistics.

        For each window, fits AR(1) + volatility regression:
            returns_t = α + β₁ * returns_{t-1} + β₂ * volatility_{t-1} + ε

        R² measures how well lagged features predict current returns.
        """
        returns = data["returns"].values
        n = len(returns)

        # Create lag features
        lagged_returns = np.roll(returns, 1)
        lagged_returns[0] = np.nan

        # Rolling volatility (std of returns)
        vol = pd.Series(returns).rolling(window=min(20, len(returns) // 3)).std().values
        lagged_vol = np.roll(vol, 1)
        lagged_vol[0] = np.nan

        # Prepare feature matrix
        valid_mask = ~np.isnan(lagged_returns) & ~np.isnan(lagged_vol) & ~np.isnan(returns)

        if valid_mask.sum() < self.window_size + 1:
            raise ValueError(f"Insufficient valid observations after lagging: {valid_mask.sum()}")

        X_full = np.column_stack([lagged_returns[valid_mask], lagged_vol[valid_mask]])
        y_full = returns[valid_mask]

        # Rolling window R² computation
        r2_values = []
        r2_index = []

        for i in range(self.window_size, len(X_full)):
            X_window = X_full[i - self.window_size : i]
            y_window = y_full[i - self.window_size : i]

            if len(np.unique(y_window)) < 2:
                r2 = 0.0
            else:
                model = LinearRegression()
                model.fit(X_window, y_window)
                y_pred = model.predict(X_window)
                ss_res = np.sum((y_window - y_pred) ** 2)
                ss_tot = np.sum((y_window - np.mean(y_window)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            r2_values.append(r2)
            r2_index.append(i)

        self.r2_series_ = pd.Series(
            r2_values, index=data.index[r2_index[0] : r2_index[-1] + 1], name="rolling_r2"
        )

    def _detect_breakpoints(self) -> None:
        """
        Detect structural breakpoints using R² drops.

        Breakpoints are where R² drops significantly below its moving average,
        indicating a change in the return generation process.
        """
        if self.r2_series_ is None:
            raise ValueError("R² series not computed")

        r2 = self.r2_series_.values

        # Compute rolling mean and std of R²
        window = min(20, len(r2) // 5)
        rolling_mean = pd.Series(r2).rolling(window=window, min_periods=1).mean().values
        rolling_std = pd.Series(r2).rolling(window=window, min_periods=1).std().values
        rolling_std = np.where(rolling_std == 0, 1e-6, rolling_std)

        # Z-score of R² drops (negative deviations)
        z_scores = (r2 - rolling_mean) / rolling_std

        # Detect breaks where R² drops significantly
        break_candidates = z_scores < -self.break_threshold

        # Additional: detect sharp level shifts in R²
        r2_diff = np.diff(r2, prepend=r2[0])
        large_drops = r2_diff < -np.std(r2_diff) * self.break_threshold

        # Combine signals
        if len(large_drops) > 1:
            # Align shapes: large_drops[1:] excludes first element, so exclude first from break_candidates too
            break_signal = break_candidates[1:] | large_drops[1:]
        else:
            break_signal = break_candidates

        # Enforce minimum regime length between breaks
        break_indices = np.where(break_signal)[0]

        final_breaks = []
        if len(break_indices) > 0:
            last_break = -self.min_regime_length
            for idx in break_indices:
                if idx - last_break >= self.min_regime_length:
                    final_breaks.append(idx)
                    last_break = idx

        # Create breakpoint DataFrame
        if len(final_breaks) > 0:
            break_positions = [
                self.r2_series_.index[i] for i in final_breaks if i < len(self.r2_series_)
            ]
            self.breakpoints_ = pd.DataFrame(
                {
                    "date": break_positions,
                    "r2_value": [
                        self.r2_series_.iloc[i] for i in final_breaks if i < len(self.r2_series_)
                    ],
                    "z_score": [z_scores[i] for i in final_breaks if i < len(z_scores)],
                }
            ).set_index("date")
        else:
            self.breakpoints_ = pd.DataFrame(columns=["date", "r2_value", "z_score"]).set_index(
                "date"
            )

    def _segment_regimes(self, index: pd.DatetimeIndex) -> None:
        """
        Segment data into regimes based on breakpoints.

        Each regime segment gets a unique label (Regime_0, Regime_1, etc.)
        """
        if self.breakpoints_ is None:
            # No breaks detected - single regime
            self.regime_segments_ = pd.DataFrame(
                {
                    "start_date": [index[0]],
                    "end_date": [index[-1]],
                    "regime_label": ["Regime_0"],
                    "r2_mean": [self.r2_series_.mean() if self.r2_series_ is not None else 0],
                }
            )
            self.n_regimes_ = 1
            return

        # Create segments between breakpoints
        break_dates = self.breakpoints_.index.tolist()

        segments = []
        regime_idx = 0

        # First segment (start to first break)
        if len(break_dates) > 0:
            first_break_idx = index.get_loc(break_dates[0])
            segments.append(
                {
                    "start_date": index[0],
                    "end_date": break_dates[0],
                    "regime_label": f"Stable_{regime_idx}",
                    "r2_mean": self.r2_series_.loc[index[0] : break_dates[0]].mean()
                    if self.r2_series_ is not None
                    else 0,
                }
            )
            regime_idx += 1

            # Middle segments
            for i, (current_break, next_break) in enumerate(zip(break_dates[:-1], break_dates[1:])):
                current_idx = index.get_loc(current_break)
                next_idx = index.get_loc(next_break)

                r2_mean = (
                    self.r2_series_.loc[current_break:next_break].mean()
                    if self.r2_series_ is not None
                    else 0
                )

                # Classify regime based on R² level
                if r2_mean > self.min_r2_threshold:
                    label = f"Stable_{regime_idx}"
                else:
                    label = f"Transition_{regime_idx}"

                segments.append(
                    {
                        "start_date": current_break,
                        "end_date": next_break,
                        "regime_label": label,
                        "r2_mean": r2_mean,
                    }
                )
                regime_idx += 1

            # Last segment
            last_break_idx = index.get_loc(break_dates[-1])
            r2_mean = (
                self.r2_series_.loc[break_dates[-1] : index[-1]].mean()
                if self.r2_series_ is not None
                else 0
            )

            if r2_mean > self.min_r2_threshold:
                label = f"Stable_{regime_idx}"
            else:
                label = f"Transition_{regime_idx}"

            segments.append(
                {
                    "start_date": break_dates[-1],
                    "end_date": index[-1],
                    "regime_label": label,
                    "r2_mean": r2_mean,
                }
            )
            regime_idx += 1

        self.regime_segments_ = pd.DataFrame(segments)
        self.n_regimes_ = regime_idx

    def _compute_regression_stats(self, data: pd.DataFrame) -> None:
        """
        Compute regression statistics for each regime segment.

        Stores coefficients from the rolling regression for interpretation.
        """
        if self.regime_segments_ is None:
            return

        stats = []
        for _, seg in self.regime_segments_.iterrows():
            mask = (data.index >= seg["start_date"]) & (data.index <= seg["end_date"])
            subset = data[mask]

            if len(subset) < 10:
                continue

            # Fit regression on segment
            returns = subset["returns"].values
            lagged_ret = np.roll(returns, 1)
            lagged_ret[0] = np.nan

            # Simple volatility proxy
            vol = pd.Series(returns).rolling(5).std().values
            lagged_vol = np.roll(vol, 1)
            lagged_vol[0] = np.nan

            valid = ~np.isnan(lagged_ret) & ~np.isnan(lagged_vol) & ~np.isnan(returns)
            if valid.sum() < 10:
                continue

            X = np.column_stack([lagged_ret[valid], lagged_vol[valid]])
            y = returns[valid]

            if len(np.unique(y)) < 2:
                continue

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            stats.append(
                {
                    "regime_label": seg["regime_label"],
                    "coef_lagged_return": model.coef_[0],
                    "coef_volatility": model.coef_[1] if len(model.coef_) > 1 else 0,
                    "intercept": model.intercept_,
                    "r2": r2,
                }
            )

        self.regression_stats_ = pd.DataFrame(stats)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict regime labels for input data.

        Args:
            data: DataFrame with features

        Returns:
            Series of regime labels (strings) with same index as input
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Use regime segments to assign labels
        if self.regime_segments_ is None:
            return pd.Series("Regime_0", index=data.index, name="r2_regime")

        labels = []
        for idx in data.index:
            assigned = False
            for _, seg in self.regime_segments_.iterrows():
                if idx >= seg["start_date"] and idx <= seg["end_date"]:
                    labels.append(seg["regime_label"])
                    assigned = True
                    break
            if not assigned:
                # Extrapolate - assign to nearest regime
                if idx < self.regime_segments_.iloc[0]["start_date"]:
                    labels.append(self.regime_segments_.iloc[0]["regime_label"])
                else:
                    labels.append(self.regime_segments_.iloc[-1]["regime_label"])

        return pd.Series(labels, index=data.index, name="r2_regime")

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime probabilities based on R² proximity.

        Uses distance of current R² from regime R² means.

        Args:
            data: DataFrame with features

        Returns:
            DataFrame with probability for each regime
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.regime_segments_ is None or self.r2_series_ is None:
            # Single regime - return certainty
            return pd.DataFrame(1.0, index=data.index, columns=["Regime_0"])

        # Get R² for each data point
        r2_values = self.r2_series_.reindex(data.index, method="ffill").fillna(0)

        # Compute distance to each regime's R² mean
        regime_r2_means = self.regime_segments_.set_index("regime_label")["r2_mean"]
        unique_regimes = regime_r2_means.index.unique().tolist()

        probs = pd.DataFrame(index=data.index, columns=unique_regimes, dtype=float)

        for regime in unique_regimes:
            mean_r2 = regime_r2_means[regime]
            # Convert distance to probability (inverse distance weighting)
            distance = np.abs(r2_values - mean_r2)
            probs[regime] = 1 / (distance + 0.01)

        # Normalize to probabilities
        probs = probs.div(probs.sum(axis=1), axis=0)

        return probs

    def get_regime_summary(self) -> RegimeSummary:
        """
        Get summary information about detected regimes.

        Returns:
            RegimeSummary with label distribution and metadata
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.regime_segments_ is None:
            return RegimeSummary(
                name=self.name,
                n_regimes=1,
                regime_labels=["Regime_0"],
                label_distribution={"Regime_0": 0},
                label_proportions={"Regime_0": 1.0},
                metadata={"breakpoints_detected": 0},
            )

        label_counts = self.regime_segments_["regime_label"].value_counts().to_dict()
        total = len(self.regime_segments_)
        proportions = {k: v / total for k, v in label_counts.items()}

        return RegimeSummary(
            name=self.name,
            n_regimes=self.n_regimes_,
            regime_labels=list(label_counts.keys()),
            label_distribution=label_counts,
            label_proportions=proportions,
            metadata={
                "n_breakpoints": len(self.breakpoints_) if self.breakpoints_ is not None else 0,
                "r2_mean": float(self.r2_series_.mean()) if self.r2_series_ is not None else 0,
                "r2_std": float(self.r2_series_.std()) if self.r2_series_ is not None else 0,
                "window_size": self.window_size,
                "break_threshold": self.break_threshold,
            },
        )

    def get_breakpoints(self) -> pd.DataFrame:
        """
        Get detected structural break dates.

        Returns:
            DataFrame with break dates, R² values, and z-scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.breakpoints_ is None:
            return pd.DataFrame(columns=["date", "r2_value", "z_score"]).set_index("date")

        return self.breakpoints_

    def get_regression_stats(self) -> pd.DataFrame:
        """
        Get regression statistics per regime.

        Returns:
            DataFrame with coefficients and fit stats per regime
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.regression_stats_ is None:
            return pd.DataFrame()

        return self.regression_stats_

    def get_regime_segments(self) -> pd.DataFrame:
        """
        Get regime segment information.

        Returns:
            DataFrame with start/end dates and labels for each regime segment
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.regime_segments_ is None:
            return pd.DataFrame()

        return self.regime_segments_

    def get_r2_series(self) -> pd.Series:
        """
        Get the rolling R² time series.

        Returns:
            Series of R² values over time
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.r2_series_ is None:
            return pd.Series(dtype=float)

        return self.r2_series_
