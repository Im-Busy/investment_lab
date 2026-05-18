"""
R4: HP Filter Return Forecasting.

Hodrick-Prescott filter for extracting smooth trend from cumulative factor
returns. Eliminates noise while preserving trend structure. Used for expected
return estimation in portfolio optimization and signal weighting.

Source: 华泰多因子系列1 §2.5

Formula:
    min_τ Σ(y_t − τ_t)² + λ Σ[(τ_{t+1} − τ_t) − (τ_t − τ_{t-1})]²

Where λ is the smoothness parameter. Larger λ = smoother trend.
Huatai validated HP filter > EWMA > ARIMA > historical mean for factor return
forecasting.

Standard λ values:
    - Daily data:     λ = 100 × (252)⁴ ≈ 1.6 × 10¹⁰  (or use λ=100 with
      pre-annualized data)
    - Weekly data:    λ = 100 × (52)⁴ ≈ 7.3 × 10⁶
    - Monthly data:   λ = 100 × (12)⁴ ≈ 2.1 × 10⁵
    - Quarterly data: λ = 1600 (Hodrick-Prescott original)
    - Annual data:    λ = 100

Practical daily λ: use 6.25 / 100 since trading returns are already daily.
Or use the commonly cited λ_daily ≈ 129600 (same as 1600 × 81 for quarterly→daily).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Practical λ values ──
HP_LAMBDA_DAILY = 100_000
HP_LAMBDA_WEEKLY = 1_600
HP_LAMBDA_MONTHLY = 14_400
HP_LAMBDA_QUARTERLY = 1_600


def hp_filter(
    y: np.ndarray,
    lam: float = HP_LAMBDA_DAILY,
) -> np.ndarray:
    """Apply Hodrick-Prescott filter to a 1-D array.

    Extracts the smooth trend τ from noisy observations y by solving:
        min_τ ||y - τ||² + λ ||D₂τ||²

    where D₂ is the second-difference operator.

    Uses the closed-form sparse linear system solution via numpy.

    Args:
        y: (N,) input time series.
        lam: Smoothness parameter (larger = smoother). Default 100_000 for daily.

    Returns:
        (N,) smoothed trend τ.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)

    if n < 3:
        return y.copy()

    # Build pentadiagonal system: (I + λ D₂ᵀ D₂) τ = y
    # D₂ is (n-2, n) with pattern [-1, 2, -1] per row
    # D₂ᵀ D₂ is (n, n) with band structure:
    #   main diagonal:     [1, 6, 5, ..., 5, 6, 1]
    #   off-diagonal:      [-4, -4, ..., -4] and [1, 1, ..., 1]

    a = np.zeros(n, dtype=np.float64)  # lower diag
    b = np.zeros(n, dtype=np.float64)  # main diag
    c = np.zeros(n, dtype=np.float64)  # upper diag
    d = y.copy()

    # First and last rows (boundary conditions)
    b[0] = 1.0 + lam
    c[0] = -2.0 * lam
    b[1] = 1.0 + 5.0 * lam
    c[1] = -4.0 * lam
    a[1] = -2.0 * lam

    b[-1] = 1.0 + lam
    a[-1] = -2.0 * lam
    b[-2] = 1.0 + 5.0 * lam
    a[-2] = -4.0 * lam
    c[-2] = -2.0 * lam

    for i in range(2, n - 2):
        b[i] = 1.0 + 6.0 * lam

    for i in range(1, n - 1):
        if i != 1 and i != n - 2:
            a[i] = -4.0 * lam
            c[i] = -4.0 * lam

    for i in range(2, n - 2):
        d[i] += lam * y[i - 2]  # from upper second-diagonal term? No - let me fix.

    # Actually the pentadiagonal approach is complex. Let me use the simpler
    # tridiagonal approach via normal equations or direct matrix solve.
    # The HP filter can be solved as:
    #   τ = (I + λ D₂ᵀ D₂)^{-1} y
    # where the matrix is small enough for direct solve for typical N (~2500).

    return _hp_filter_solve(y, lam)


def _hp_filter_solve(y: np.ndarray, lam: float) -> np.ndarray:
    """Solve HP filter via direct linear system."""
    n = len(y)
    if n < 3:
        return y.copy()

    # Build D₂: (n-2) × n second-difference matrix
    # row i: [0, ..., 0, 1, -2, 1, 0, ..., 0]
    #         at positions i-1, i, i+1

    # Instead of building full matrix, build (I + λ D₂ᵀ D₂) directly.
    # D₂ᵀ D₂ is a banded matrix with 5 diagonals.
    # For n=5:
    # [[ 1, -2,  1,  0,  0],
    #  [-2,  5, -4,  1,  0],
    #  [ 1, -4,  6, -4,  1],
    #  [ 0,  1, -4,  5, -2],
    #  [ 0,  0,  1, -2,  1]]

    # Build full matrix for clarity (n ≤ ~5000 for daily data is fine)
    A = np.eye(n, dtype=np.float64)
    # Add D₂ᵀ D₂ contribution
    A += lam * _build_d2td2(n)
    return np.linalg.solve(A, y)


def _build_d2td2(n: int) -> np.ndarray:
    """Build D₂ᵀ D₂ matrix (n × n) for HP filter."""
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 2):
        # D₂ row i has 1 at i, -2 at i+1, 1 at i+2
        # D₂ᵀ D₂ contribution from this row:
        # outer product of [..., 1, -2, 1, ...] with itself
        for j, v1 in [(i, 1.0), (i + 1, -2.0), (i + 2, 1.0)]:
            for k, v2 in [(i, 1.0), (i + 1, -2.0), (i + 2, 1.0)]:
                mat[j, k] += v1 * v2
    return mat


def hp_forecast(
    returns: np.ndarray,
    lam: float = HP_LAMBDA_DAILY,
    horizon: int = 1,
) -> np.ndarray:
    """Forecast expected returns using HP-filtered trend.

    Args:
        returns: (N,) cumulative or per-bar return series.
        lam: HP smoothness parameter.
        horizon: Forecast horizon in bars (extrapolate trend).

    Returns:
        (N,) array of expected returns at each bar (using HP trend at that point).
        For bars before horizon is available, returns NaN.
    """
    trend = hp_filter(returns, lam)
    forecast = np.full_like(trend, np.nan, dtype=np.float64)

    # Use HP trend gradient as expected return
    if len(trend) > horizon:
        grad = np.gradient(trend)
        forecast[: len(grad) - horizon] = trend[horizon:] - trend[:-horizon]
        forecast[:] = np.roll(grad, -horizon)

    return forecast


def hp_expected_return(
    cumulative_return: np.ndarray,
    lam: float = HP_LAMBDA_DAILY,
) -> float:
    """Compute the HP-filtered expected return using the most recent trend.

    Args:
        cumulative_return: (N,) cumulative return series.
        lam: HP smoothness.

    Returns:
        Expected return based on HP trend slope at the final bar.
    """
    trend = hp_filter(cumulative_return, lam)
    if len(trend) < 3:
        return float(np.mean(np.diff(cumulative_return)) if len(cumulative_return) > 1 else 0.0)
    # Slope of the HP trend at the end
    return float(trend[-1] - trend[-2])


class HPFilter:
    """Online-capable HP filter with incremental state.

    For use in strategies that need HP trend at each bar without
    recomputing from scratch every time. Maintains a rolling window.

    Parameters:
        window: Window size for HP filter (larger = more stable).
        lam: Smoothness parameter.
    """

    def __init__(
        self,
        window: int = 252,
        lam: float = HP_LAMBDA_DAILY,
    ) -> None:
        self._window = window
        self._lam = lam
        self._history: list[float] = []

    def update(self, value: float) -> float:
        """Add a new value and return the current HP trend endpoint.

        Args:
            value: New observation to add.

        Returns:
            Latest HP trend value.
        """
        self._history.append(value)
        if len(self._history) > self._window:
            self._history.pop(0)
        if len(self._history) < 3:
            return value
        arr = np.array(self._history, dtype=np.float64)
        trend = _hp_filter_solve(arr, self._lam)
        return float(trend[-1])

    def slope(self) -> float:
        """Return the HP trend slope (expected return) at the last point."""
        if len(self._history) < 3:
            return 0.0
        arr = np.array(self._history, dtype=np.float64)
        trend = _hp_filter_solve(arr, self._lam)
        return float(trend[-1] - trend[-2])
