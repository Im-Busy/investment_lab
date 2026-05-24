"""Swing point detector based on configurable value-change threshold.

Paper: "A Swing Point Detector for Financial Time Series" (A2 in MASTER_COMPARISON_REPORT)
Algorithm: Mark swing high when price drops X% from last running max;
           mark swing low when price rises X% from last running min.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_swing_points(
    high: np.ndarray,
    low: np.ndarray,
    threshold: float = 0.005,
    min_bars: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect swing highs and lows via configurable percentage threshold.

    Args:
        high: Array of high prices.
        low: Array of low prices.
        threshold: Fractional drop from max / rise from min to trigger swing (default 0.5%).
        min_bars: Minimum bars between consecutive swing points of same type (default 2).

    Returns:
        Tuple of (swing_high_signal, swing_low_signal) as np.int8 arrays.
        swing_high_signal: 1 at swing high bar, 0 elsewhere.
        swing_low_signal: -1 at swing low bar, 0 elsewhere.
    """
    n = len(high)
    swing_high = np.zeros(n, dtype=np.int8)
    swing_low = np.zeros(n, dtype=np.int8)

    if n < min_bars + 1:
        return swing_high, swing_low

    running_max = high[0]
    running_min = low[0]
    max_idx = 0
    min_idx = 0

    for i in range(1, n):
        if high[i] > running_max:
            running_max = high[i]
            max_idx = i
        if low[i] < running_min:
            running_min = low[i]
            min_idx = i

        drop_from_max = (running_max - low[i]) / running_max if running_max > 0 else 0.0
        rise_from_min = (high[i] - running_min) / running_min if running_min > 0 else 0.0

        if drop_from_max >= threshold and max_idx > 0 and i - max_idx >= min_bars:
            swing_high[max_idx] = 1
            running_min = low[i]
            min_idx = i

        if rise_from_min >= threshold and min_idx > 0 and i - min_idx >= min_bars:
            swing_low[min_idx] = -1
            running_max = high[i]
            max_idx = i

    return swing_high, swing_low


def detect_swing_points_df(
    df: pd.DataFrame,
    threshold: float = 0.005,
    min_bars: int = 2,
) -> pd.DataFrame:
    """Detect swing points on a DataFrame with OHLC columns.

    Args:
        df: DataFrame with 'High' and 'Low' columns.
        threshold: Fractional drop/rise threshold.
        min_bars: Minimum bars between same-type swing points.

    Returns:
        DataFrame with added 'swing_high' and 'swing_low' columns.
    """
    result = df.copy()
    high_arr = df["High"].to_numpy(dtype=np.float64)
    low_arr = df["Low"].to_numpy(dtype=np.float64)
    sh, sl = detect_swing_points(high_arr, low_arr, threshold, min_bars)
    result["swing_high"] = sh
    result["swing_low"] = sl
    return result
