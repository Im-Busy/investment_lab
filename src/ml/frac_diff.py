"""
Fractional Differentiation (FS11)

Applies fractional differencing to price series to achieve stationarity while
preserving memory. Unlike integer differencing (d=1) which removes all memory,
fractional differencing (d≈0.3-0.5) retains long-term trends while stabilizing
variance. This eliminates the look-ahead bias introduced by StandardScaler.

Method: Fixed-width window fractional differencing via binomial weights
(Chapter 5, Lopez de Prado, "Advances in Financial Machine Learning")

Weight formula:
    w_0 = 1
    w_k = -w_{k-1} * (d - k + 1) / k   (recursive, for k >= 1)

Features:
- frac_diff(series, d): Apply fractional differencing with order d
- find_optimal_d(series): Find minimum d achieving ADF stationarity
- get_weights(d, window): Get the binomial weight vector
- Weight threshold (tau) truncation for computational efficiency

Example:
    >>> from src.ml.frac_diff import FracDiff
    >>> fd = FracDiff()
    >>> diff_series = fd.frac_diff(prices, d=0.4)
    >>> optimal_d = fd.find_optimal_d(prices)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


class FracDiff:
    """
    Fractional differencing transformer for price/return series.

    Preserves memory while achieving stationarity. The key insight:
    integer differencing removes ALL memory (including predictive signals),
    while fractional differencing with d≈0.3-0.5 retains the slow-moving
    trend components useful for ML features.

    Args:
        window: Fixed window size for weight truncation (None = auto-select)
        tau: Weight threshold — stop expanding when |w_k| < tau
        min_obs: Minimum observations required after differencing
        verbose: Print ADF test statistics during optimal d search

    Attributes:
        weights_: Computed fractional differencing weights
        d_: Current differencing order
        is_stationary_: Result of ADF test at current d

    Reference:
        Marcos Lopez de Prado, "Advances in Financial Machine Learning" (2018)
        Chapter 5: Fractional Differentiation
    """

    def __init__(
        self,
        window: Optional[int] = None,
        tau: float = 1e-4,
        min_obs: int = 10,
        verbose: bool = False,
    ):
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        if min_obs < 5:
            raise ValueError(f"min_obs must be >= 5, got {min_obs}")

        self.window = window
        self.tau = tau
        self.min_obs = min_obs
        self.verbose = verbose

        self.weights_: Optional[np.ndarray] = None
        self.d_: Optional[float] = None
        self.is_stationary_: bool = False
        self.adf_statistic_: Optional[float] = None
        self.adf_pvalue_: Optional[float] = None

    def get_weights(self, d: float, window: int) -> np.ndarray:
        """
        Compute fractional differencing weights for order d.

        Uses the recursive binomial formula to avoid numerical overflow:
            w_0 = 1
            w_k = -w_{k-1} * (d - k + 1) / k

        Args:
            d: Fractional differencing order (0 < d < 1)
            window: Maximum number of weights to compute

        Returns:
            Array of weights [w_0, w_1, ..., w_{window-1}]

        Raises:
            ValueError: If d is out of range
        """
        if d < 0 or d > 2:
            raise ValueError(f"d must be in [0, 2], got {d}")

        weights = np.zeros(window, dtype=np.float64)
        weights[0] = 1.0

        for k in range(1, window):
            weights[k] = -weights[k - 1] * (d - k + 1) / k

        return weights

    def _get_effective_window(self, d: float) -> int:
        """Determine window size based on weight threshold."""
        if self.window is not None:
            return self.window

        # Start with a reasonable window and expand until weights drop below tau
        window = 100
        max_window = 10000

        while window < max_window:
            weights = self.get_weights(d, window)
            if np.abs(weights[-1]) < self.tau:
                # Truncate at the last significant weight
                cutoff = np.argmax(np.abs(weights) < self.tau)
                if cutoff > self.min_obs:
                    return max(cutoff, self.min_obs)
            window *= 2

        return window

    def frac_diff(self, series: pd.Series, d: float = 0.4) -> pd.Series:
        """
        Apply fractional differencing to a series.

        Args:
            series: Input price or feature series (1D)
            d: Fractional differencing order (0 < d < 1)
                0 = original series (no differencing)
                1 = standard integer differencing (returns)
                ~0.3-0.5 = fractional differencing (stationary + memory)

        Returns:
            Fractionally differenced series (same index, NaN at start)

        Raises:
            ValueError: If series has insufficient length or d is invalid
        """
        if len(series) < self.min_obs:
            raise ValueError(f"Series too short: {len(series)} < {self.min_obs} (min_obs)")

        window = self._get_effective_window(d)

        if window > len(series):
            window = len(series)

        weights = self.get_weights(d, window)
        self.weights_ = weights
        self.d_ = d

        values = series.values.astype(np.float64)

        result = np.full(len(series), np.nan, dtype=np.float64)

        # Fixed-width window convolution
        for i in range(window - 1, len(series)):
            window_slice = values[i - window + 1 : i + 1][::-1]
            w_slice = weights[:window]
            result[i] = np.dot(w_slice, window_slice)

        return pd.Series(result, index=series.index, name=f"fracdiff_d{d}")

    def find_optimal_d(
        self,
        series: pd.Series,
        d_range: Tuple[float, float] = (0.0, 1.0),
        step: float = 0.05,
        significance: float = 0.05,
    ) -> pd.DataFrame:
        """
        Find the minimum d that achieves stationarity via ADF test.

        Iterates d from low to high, applies fractional differencing,
        and tests stationarity. Returns the first d where the series
        passes the ADF test at the given significance level.

        Args:
            series: Input price series
            d_range: (min_d, max_d) search range
            step: Increment step for d
            significance: p-value threshold for ADF stationarity test

        Returns:
            DataFrame with columns: d, adf_stat, adf_pvalue, is_stationary
        """
        from statsmodels.tsa.stattools import adfuller

        results = []
        best_d = d_range[1]

        d_values = np.arange(d_range[0], d_range[1] + step / 2, step)

        for d in d_values:
            d = round(d, 4)
            diff_series = self.frac_diff(series, d=d)
            valid = diff_series.dropna()

            if len(valid) < self.min_obs:
                results.append(
                    {
                        "d": d,
                        "adf_stat": np.nan,
                        "adf_pvalue": np.nan,
                        "is_stationary": False,
                        "n_obs": len(valid),
                    }
                )
                continue

            adf_result = adfuller(valid.values, maxlag=int(len(valid) ** 0.33))
            adf_stat = float(adf_result[0])
            adf_pvalue = float(adf_result[1])
            is_stat = adf_pvalue < significance

            if self.verbose:
                print(f"d={d:.2f} ADF={adf_stat:.4f} p={adf_pvalue:.4f} stationary={is_stat}")

            results.append(
                {
                    "d": d,
                    "adf_stat": adf_stat,
                    "adf_pvalue": adf_pvalue,
                    "is_stationary": is_stat,
                    "n_obs": len(valid),
                }
            )

            if is_stat and d < best_d:
                best_d = d
                self.adf_statistic_ = adf_stat
                self.adf_pvalue_ = adf_pvalue
                self.is_stationary_ = True

        if best_d == d_range[1] and results:
            self.is_stationary_ = results[-1]["is_stationary"]
            self.adf_statistic_ = results[-1]["adf_stat"]
            self.adf_pvalue_ = results[-1]["adf_pvalue"]

        return pd.DataFrame(results).set_index("d")

    def transform_features(
        self,
        df: pd.DataFrame,
        d: float = 0.4,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Apply fractional differencing to selected columns in a DataFrame.

        Useful for transforming price-derived features without differencing
        categorical or already-stationary columns.

        Args:
            df: Feature DataFrame
            d: Fractional differencing order
            columns: Columns to transform (default: all numeric)

        Returns:
            DataFrame with fractionally differenced features
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        result = df.copy()

        for col in columns:
            if col in result.columns:
                series = result[col]
                diff_series = self.frac_diff(series, d=d)
                result[col] = diff_series

        return result

    def get_weight_summary(self) -> pd.Series:
        """
        Get summary of the last computed weights.

        Returns:
            Series with computed weights (index = lag, values = weight)
        """
        if self.weights_ is None:
            raise ValueError("Weights not computed. Call frac_diff() first.")

        return pd.Series(
            self.weights_,
            name="weights",
            index=pd.Index(range(len(self.weights_)), name="lag"),
        )

    def weight_convergence(self, d: float = 0.4, window: int = 200) -> pd.Series:
        """
        Compute weight convergence for inspection.

        The sum of fractional differencing weights should converge to 0
        for d > 0, indicating the differencing property holds.

        Args:
            d: Fractional differencing order
            window: Number of weights to compute

        Returns:
            Series with cumulative weight sums
        """
        weights = self.get_weights(d, window)
        cumsum = np.cumsum(weights)
        return pd.Series(
            cumsum,
            name=f"cumulative_weight_sum_d{d}",
            index=range(window),
        )
