"""
D11: State Space Models for Time Series Decomposition.

Structural time series models decompose a time series into interpretable
components: trend, seasonal, cycle, and irregular (noise). Unlike ARIMA/GARCH
which model the data-generating process, state space models explicitly separate
signal from noise — making them ideal for regime detection and signal extraction.

Models:
  LocalLinearTrend: Random walk level + random walk slope.
    Extracts smooth trend from noisy price series. Adaptive — level and slope
    update with new data via Kalman filter.
  UnobservedComponents: Full structural decomposition.
    Trend + seasonal + cycle components via statsmodels. Produces clean
    signal/noise ratios for trading decisions.
  KalmanSignal: Online Kalman filter for real-time signal extraction.
    Minimal state: level + velocity. Produces filtered and predicted states
    without storing full history.

Usage:
    >>> model = LocalLinearTrend()
    >>> model.fit(prices)
    >>> trend = model.trend()
    >>> signal = model.slope_signal()  # positive slope = bullish
    >>> model.update(new_price)  # online update
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

logger = logging.getLogger(__name__)


@dataclass
class SSMDecomposition:
    trend: np.ndarray
    slope: np.ndarray
    seasonal: np.ndarray
    cycle: np.ndarray
    residual: np.ndarray
    trend_signal: np.ndarray
    slope_signal: np.ndarray

    @property
    def signal_to_noise(self) -> float:
        var_signal = np.nanvar(self.trend) + np.nanvar(self.slope)
        var_noise = np.nanvar(self.residual)
        if var_noise < 1e-12:
            return float("inf")
        return float(var_signal / var_noise)


class LocalLinearTrend:
    """Local linear trend model via Kalman filter.

    State vector: [level_t, slope_t]
    Level follows random walk with drift = slope.
    Both level and slope disturbances are stochastic.

    Produces:
      - Filtered level (smooth price trend)
      - Filtered slope (instantaneous trend direction)
      - One-step-ahead predictions
      - Signal: sign of slope * signal_strength

    Args:
        level_var: Variance of level innovations (default 0.01).
        slope_var: Variance of slope innovations (default 0.001).
        obs_var: Observation noise variance (default 1.0).
    """

    def __init__(
        self,
        level_var: float = 0.01,
        slope_var: float = 0.001,
        obs_var: float = 1.0,
    ):
        self.level_var = level_var
        self.slope_var = slope_var
        self.obs_var = obs_var
        self._level: Optional[np.ndarray] = None
        self._slope: Optional[np.ndarray] = None
        self._predicted: Optional[np.ndarray] = None
        self._n: int = 0

    def fit(self, series: np.ndarray) -> LocalLinearTrend:
        """Run Kalman filter on full series.

        Args:
            series: 1-D array of observations.

        Returns:
            Self for chaining.
        """
        series = np.asarray(series, dtype=float)
        n = len(series)
        self._n = n

        level = np.zeros(n)
        slope = np.zeros(n)
        predicted = np.zeros(n)

        # State: [level, slope]
        x_level = series[0] if not np.isnan(series[0]) else 0.0
        x_slope = 0.0

        # Covariance: P
        P = np.eye(2) * 1.0

        # Transition: F = [[1, 1], [0, 1]]
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        # Observation: H = [1, 0]
        H = np.array([1.0, 0.0])
        # State noise: Q
        Q = np.diag([self.level_var, self.slope_var])
        R = self.obs_var

        for t in range(n):
            # Predict
            x_pred = F @ np.array([x_level, x_slope])
            P_pred = F @ P @ F.T + Q

            # Predicted observation
            y_pred = H @ x_pred
            predicted[t] = y_pred

            # Update (if observation is valid)
            if not np.isnan(series[t]):
                y = series[t]
                v = y - y_pred  # innovation
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T / S  # Kalman gain

                x_update = x_pred + K * v
                P = (np.eye(2) - np.outer(K, H)) @ P_pred
            else:
                x_update = x_pred

            x_level, x_slope = x_update[0], x_update[1]
            level[t] = x_level
            slope[t] = x_slope

        self._level = level
        self._slope = slope
        self._predicted = predicted
        return self

    def trend(self) -> np.ndarray:
        """Get filtered level (trend estimate)."""
        if self._level is None:
            raise ValueError("Call fit() first")
        return self._level

    def filtered_slope(self) -> np.ndarray:
        """Get filtered slope (instantaneous trend direction)."""
        if self._slope is None:
            raise ValueError("Call fit() first")
        return self._slope

    def predicted(self) -> np.ndarray:
        """Get one-step-ahead predictions."""
        if self._predicted is None:
            raise ValueError("Call fit() first")
        return self._predicted

    def slope_signal(self, threshold: float = 0.0) -> np.ndarray:
        """Trading signal from slope direction.

        +1 = bullish (slope > threshold), -1 = bearish (slope < -threshold),
        0 = neutral.

        Args:
            threshold: Minimum absolute slope for non-zero signal.

        Returns:
            Array of -1, 0, +1 signals.
        """
        if self._slope is None:
            raise ValueError("Call fit() first")
        signal = np.zeros(self._n)
        signal[self._slope > threshold] = 1.0
        signal[self._slope < -threshold] = -1.0
        return signal

    def slope_strength(self) -> np.ndarray:
        """Normalized slope strength [-1, 1] via tanh."""
        if self._slope is None:
            raise ValueError("Call fit() first")
        std = np.nanstd(self._slope)
        if std < 1e-12:
            return np.zeros(self._n)
        return np.tanh(self._slope / std)

    def update(self, observation: float) -> Tuple[float, float]:
        """Online single-step update (Kalman filter one iteration).

        Args:
            observation: New observation value.

        Returns:
            Tuple of (updated_level, updated_slope).
        """
        if self._level is None or len(self._level) == 0:
            level = observation if not np.isnan(observation) else 0.0
            slope = 0.0
            self._level = np.array([level])
            self._slope = np.array([slope])
            return level, slope

        prev_level = self._level[-1]
        prev_slope = self._slope[-1]

        # Predict
        pred_level = prev_level + prev_slope
        pred_slope = prev_slope

        # Simple gain (0.5 for level, 0.1 for slope)
        if not np.isnan(observation):
            alpha = 0.5
            beta = 0.1
            new_level = pred_level + alpha * (observation - pred_level)
            new_slope = pred_slope + beta * (observation - pred_level)
        else:
            new_level = pred_level
            new_slope = pred_slope

        self._level = np.append(self._level, new_level)
        self._slope = np.append(self._slope, new_slope)
        self._n += 1

        return new_level, new_slope

    def fit_dataframe(
        self,
        prices: pd.Series,
    ) -> pd.DataFrame:
        """Fit on a price series and return results as DataFrame.

        Args:
            prices: Price series with datetime index.

        Returns:
            DataFrame with trend, slope, predicted, signal columns.
        """
        values = prices.values
        self.fit(values)
        df = pd.DataFrame(
            {
                "price": values,
                "trend": self._level,
                "slope": self._slope,
                "predicted": self._predicted,
                "signal": self.slope_signal(),
                "strength": self.slope_strength(),
            },
            index=prices.index,
        )
        return df


class StateSpaceDecomposer:
    """Full structural time series decomposition via statsmodels.

    Uses UnobservedComponents to decompose into:
      - Local linear trend (level + stochastic slope)
      - Seasonal component (configurable period)
      - Cycle component (stochastic cycle with configurable frequency)
      - Irregular component (residual noise)

    Produces clean signal/noise ratios and trading signals.

    Args:
        seasonal_period: Period for seasonal component (0 = no seasonal).
        cycle: Include stochastic cycle component.
        cycle_period: Approximate cycle length in bars.
        method: Estimation method ("bfgs" or "powell").
    """

    def __init__(
        self,
        seasonal_period: int = 0,
        cycle: bool = True,
        cycle_period: int = 21,
        method: str = "powell",
    ):
        self.seasonal_period = seasonal_period
        self.cycle = cycle
        self.cycle_period = cycle_period
        self.method = method
        self._result = None
        self._decomp: Optional[SSMDecomposition] = None

    def fit(self, series: np.ndarray) -> StateSpaceDecomposer:
        """Fit structural model.

        Args:
            series: 1-D array of values.

        Returns:
            Self for chaining.
        """
        series = np.asarray(series, dtype=float)
        n = len(series)
        valid = ~np.isnan(series)
        clean = series[valid]

        if len(clean) < 20:
            logger.warning("Insufficient data (%d obs) for state space model", len(clean))
            return self

        try:
            mod = UnobservedComponents(
                clean,
                level="local linear trend",
                freq_seasonal=[{"period": self.seasonal_period}]
                if self.seasonal_period > 0
                else None,
                cycle=self.cycle,
                cycle_period_bounds=[max(2, self.cycle_period // 2), self.cycle_period * 2],
            )
            res = mod.fit(disp=False, method=self.method, maxiter=200)
            self._result = res

            m = len(clean)
            trend = np.full(n, np.nan)
            slope = np.full(n, np.nan)
            seasonal_arr = np.full(n, np.nan)
            cycle_arr = np.full(n, np.nan)
            residual = np.full(n, np.nan)
            trend_signal = np.full(n, np.nan)
            slope_signal = np.full(n, np.nan)

            # Extract trend from results
            filtered_level = res.level.get("filtered", np.full(m, np.nan))
            trend[valid] = filtered_level

            # Approximate slope from differenced trend
            slope_arr = np.diff(filtered_level, prepend=filtered_level[0])
            slope[valid] = slope_arr

            # Extract cycle if present
            if self.cycle and hasattr(res, "cycle"):
                cycle_data = res.cycle
                if cycle_data is not None:
                    cycle_filtered = cycle_data.get("filtered", np.full(m, np.nan))
                    cycle_arr[valid] = cycle_filtered

            # Extract seasonal if present
            if self.seasonal_period > 0 and hasattr(res, "seasonal"):
                seasonal_data = res.seasonal
                if seasonal_data is not None and isinstance(seasonal_data, dict):
                    seasonal_filtered = seasonal_data.get("filtered", np.full(m, np.nan))
                    seasonal_arr[valid] = seasonal_filtered

            # Residual = observed - trend - cycle - seasonal
            residual[valid] = clean - filtered_level
            if not np.all(np.isnan(cycle_arr[valid])):
                residual[valid] -= cycle_arr[valid]
            if not np.all(np.isnan(seasonal_arr[valid])):
                residual[valid] -= seasonal_arr[valid]

            trend_signal[valid] = np.where(slope_arr > 0, 1.0, -1.0)
            slope_accel = np.diff(slope_arr, prepend=0)
            slope_signal[valid] = np.tanh(slope_accel / (np.nanstd(slope_accel) + 1e-12))

            self._decomp = SSMDecomposition(
                trend=trend,
                slope=slope,
                seasonal=seasonal_arr,
                cycle=cycle_arr,
                residual=residual,
                trend_signal=trend_signal,
                slope_signal=slope_signal,
            )
        except Exception as e:
            logger.warning("State space fit failed: %s", e)
            self._result = None

        return self

    @property
    def decomposition(self) -> Optional[SSMDecomposition]:
        """Get the decomposition result."""
        return self._decomp

    @property
    def signal_to_noise(self) -> float:
        """Ratio of trend+slope variance to residual variance."""
        if self._decomp is None:
            return 0.0
        return self._decomp.signal_to_noise

    def fit_dataframe(
        self,
        prices: pd.Series,
    ) -> pd.DataFrame:
        """Fit on a price series and return decomposition as DataFrame.

        Args:
            prices: Price series with datetime index.

        Returns:
            DataFrame with all components and signals.
        """
        values = prices.values
        self.fit(values)
        if self._decomp is None:
            return pd.DataFrame({"price": values}, index=prices.index)

        return pd.DataFrame(
            {
                "price": values,
                "trend": self._decomp.trend,
                "slope": self._decomp.slope,
                "seasonal": self._decomp.seasonal,
                "cycle": self._decomp.cycle,
                "residual": self._decomp.residual,
                "trend_signal": self._decomp.trend_signal,
                "slope_signal": self._decomp.slope_signal,
            },
            index=prices.index,
        )


class KalmanSignal:
    """Lightweight online Kalman filter for real-time trading signals.

    Minimal 2-state model (level + velocity). Designed for streaming data
    with constant noise parameters. Produces filtered states and signals
    without storing full history.

    Args:
        process_noise: State transition noise covariance (diagonal).
        measurement_noise: Observation noise variance.
    """

    def __init__(
        self,
        process_noise: float = 0.001,
        measurement_noise: float = 0.1,
    ):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self._x: np.ndarray = np.zeros(2)  # [level, velocity]
        self._P: np.ndarray = np.eye(2)
        self._F: np.ndarray = np.array([[1.0, 1.0], [0.0, 1.0]])
        self._H: np.ndarray = np.array([[1.0, 0.0]])
        self._Q: np.ndarray = np.diag([process_noise, process_noise * 0.1])
        self._R: float = measurement_noise
        self._initialized: bool = False
        self._history: List[float] = []

    def update(self, observation: float) -> Tuple[float, float]:
        """Single Kalman filter update step.

        Args:
            observation: New scalar observation.

        Returns:
            (filtered_level, filtered_velocity).
        """
        if not self._initialized:
            self._x = np.array([observation if not np.isnan(observation) else 0.0, 0.0])
            self._initialized = True

        # Predict
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q

        if not np.isnan(observation):
            # Innovation
            y = observation - (self._H @ x_pred)[0]
            S = (self._H @ P_pred @ self._H.T)[0, 0] + self._R
            K = P_pred @ self._H.T / S  # Kalman gain

            # Update
            self._x = x_pred + K.flatten() * y
            self._P = P_pred - K @ self._H @ P_pred
        else:
            self._x = x_pred

        self._history.append(float(self._x[0]))
        return float(self._x[0]), float(self._x[1])

    @property
    def level(self) -> float:
        return float(self._x[0])

    @property
    def velocity(self) -> float:
        return float(self._x[1])

    @property
    def signal(self) -> float:
        """Normalized trading signal from velocity. Range [-1, 1]."""
        v = self._x[1]
        return float(np.tanh(v / (self.measurement_noise + 1e-9)))

    def reset(self) -> None:
        """Reset filter state."""
        self._x = np.zeros(2)
        self._P = np.eye(2)
        self._initialized = False
        self._history = []
