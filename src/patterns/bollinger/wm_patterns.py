"""W-Type Bottom and M-Type Top Bollinger Band patterns.

Paper: "Bollinger Band W/M Patterns" (B2 in MASTER_COMPARISON_REPORT)
4-step confirmation: (1) touch BB extreme → (2) retrace to middle →
(3) hold/sustain → (4) break support/resistance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _bollinger_bands(
    close: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Bollinger Bands (middle, upper, lower)."""
    sma = np.full_like(close, np.nan)
    upper = np.full_like(close, np.nan)
    lower = np.full_like(close, np.nan)
    if len(close) < period:
        return sma, upper, lower
    for i in range(period - 1, len(close)):
        window = close[i - period + 1 : i + 1]
        mu = float(np.mean(window))
        sigma = float(np.std(window, ddof=1))
        sma[i] = mu
        upper[i] = mu + num_std * sigma
        lower[i] = mu - num_std * sigma
    return sma, upper, lower


def detect_w_bottom(
    close: np.ndarray,
    bb_period: int = 20,
    bb_std: float = 2.0,
    lookback: int = 20,
) -> np.ndarray:
    """Detect W-Type Bottom Bollinger pattern.

    Steps:
      1. First trough touches or breaches lower BB
      2. Price retraces to middle BB
      3. Second trough is ABOVE lower BB (higher low)
      4. Price breaks above middle BB after second trough

    Args:
        close: Array of close prices.
        bb_period: Bollinger Band period.
        bb_std: Standard deviation multiplier.
        lookback: Maximum bars to search for the second trough.

    Returns:
        int8 signal array (1 at pattern completion bar).
    """
    n = len(close)
    middle, upper, lower = _bollinger_bands(close, bb_period, bb_std)
    signal = np.zeros(n, dtype=np.int8)

    if n < bb_period + lookback:
        return signal

    for i in range(bb_period + lookback, n):
        first_trough_idx = -1
        for j in range(i - lookback, i - 5):
            if np.isnan(lower[j]):
                continue
            if close[j] <= lower[j]:
                if j > 1 and close[j] < close[j - 1] and close[j] < close[j + 1]:
                    first_trough_idx = j
                    break

        if first_trough_idx < 0:
            continue

        retraced = False
        for j in range(first_trough_idx + 1, i - 3):
            if np.isnan(middle[j]):
                continue
            if close[j] >= middle[j]:
                retraced = True
                break
        if not retraced:
            continue

        second_trough_found = False
        second_trough_idx = -1
        for j in range(first_trough_idx + 5, i - 2):
            if np.isnan(lower[j]):
                continue
            if close[j] > lower[j]:
                if j > 1 and close[j] < close[j - 1] and close[j] <= close[j + 1]:
                    if close[j] > close[first_trough_idx]:
                        second_trough_found = True
                        second_trough_idx = j
                        break
        if not second_trough_found:
            continue

        if close[i] > middle[i] and close[i - 1] <= middle[i - 1]:
            signal[i] = 1

    return signal


def detect_m_top(
    close: np.ndarray,
    bb_period: int = 20,
    bb_std: float = 2.0,
    lookback: int = 20,
) -> np.ndarray:
    """Detect M-Type Top Bollinger pattern.

    Steps:
      1. First peak touches or breaches upper BB
      2. Price retraces to middle BB
      3. Second peak is BELOW upper BB (lower high)
      4. Price breaks below middle BB after second peak

    Returns:
        int8 signal array (-1 at pattern completion bar).
    """
    n = len(close)
    middle, upper, lower = _bollinger_bands(close, bb_period, bb_std)
    signal = np.zeros(n, dtype=np.int8)

    if n < bb_period + lookback:
        return signal

    for i in range(bb_period + lookback, n):
        first_peak_idx = -1
        for j in range(i - lookback, i - 5):
            if np.isnan(upper[j]):
                continue
            if close[j] >= upper[j]:
                if j > 1 and close[j] > close[j - 1] and close[j] > close[j + 1]:
                    first_peak_idx = j
                    break

        if first_peak_idx < 0:
            continue

        retraced = False
        for j in range(first_peak_idx + 1, i - 3):
            if np.isnan(middle[j]):
                continue
            if close[j] <= middle[j]:
                retraced = True
                break
        if not retraced:
            continue

        second_peak_found = False
        for j in range(first_peak_idx + 5, i - 2):
            if np.isnan(upper[j]):
                continue
            if close[j] < upper[j]:
                if j > 1 and close[j] > close[j - 1] and close[j] >= close[j + 1]:
                    if close[j] < close[first_peak_idx]:
                        second_peak_found = True
                        break
        if not second_peak_found:
            continue

        if close[i] < middle[i] and close[i - 1] >= middle[i - 1]:
            signal[i] = -1

    return signal


def detect_wm_bollinger(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    lookback: int = 20,
) -> pd.DataFrame:
    """Detect W-Bottom and M-Top Bollinger patterns on a DataFrame.

    Returns:
        DataFrame with added 'w_bottom' and 'm_top' signal columns.
    """
    result = df.copy()
    close = df["Close"].to_numpy(dtype=np.float64)
    result["w_bottom"] = detect_w_bottom(close, bb_period, bb_std, lookback)
    result["m_top"] = detect_m_top(close, bb_period, bb_std, lookback)
    return result
