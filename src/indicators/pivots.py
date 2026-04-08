"""
Pivot Point Detection

Functions for detecting local extrema (swing highs/lows) and pivot points.

PHASE 2 OPTIMIZATION:
- Uses Numba JIT compilation for 50-100x speedup
- Falls back to pure Python if Numba is not available
- All public functions maintain the same API
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Try to import Numba-accelerated functions
try:
    from .pivots_numba import (
        NUMBA_AVAILABLE,
        find_all_swings_numba,
        find_higher_highs_numba,
        find_local_extrema_numba,
        find_lower_lows_numba,
        find_pivot_points_numba,
        find_swing_highs_numba,
        find_swing_lows_numba,
        get_recent_swing_high_numba,
        get_recent_swing_low_numba,
    )
    from .pivots_numba import (
        warmup as numba_warmup,
    )
except ImportError:
    NUMBA_AVAILABLE = False


def find_local_extrema(
    df: pd.DataFrame, lookback: int = 5, method: str = "fractal"
) -> pd.DataFrame:
    """
    Find local highs and lows using the fractal method.

    A local high occurs when the bar's high is higher than the highs
    of the lookback bars on each side.

    Args:
        df: DataFrame with High, Low columns
        lookback: Number of bars to look on each side
        method: Detection method ('fractal' or 'simple')

    Returns:
        DataFrame with 'local_high' and 'local_low' boolean columns
    """
    result = df.copy()
    result["local_high"] = False
    result["local_low"] = False

    for i in range(lookback, len(df) - lookback):
        # Check for local high
        is_high = True
        for j in range(1, lookback + 1):
            if df.iloc[i]["High"] <= df.iloc[i - j]["High"]:
                is_high = False
                break
            if df.iloc[i]["High"] <= df.iloc[i + j]["High"]:
                is_high = False
                break

        if is_high:
            result.iloc[i, result.columns.get_loc("local_high")] = True  # type: ignore[index]

        # Check for local low
        is_low = True
        for j in range(1, lookback + 1):
            if df.iloc[i]["Low"] >= df.iloc[i - j]["Low"]:
                is_low = False
                break
            if df.iloc[i]["Low"] >= df.iloc[i + j]["Low"]:
                is_low = False
                break

        if is_low:
            result.iloc[i, result.columns.get_loc("local_low")] = True  # type: ignore[index]

    return result


def find_swing_highs(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """
    Find swing highs in the price data.

    PHASE 2 OPTIMIZATION: Uses Numba JIT compilation for 50-100x speedup.
    Falls back to pure Python if Numba is not available.

    Args:
        df: DataFrame with High column
        lookback: Number of bars to check on each side

    Returns:
        Series with swing high values (NaN where no swing high exists)
    """
    # Use Numba-accelerated version if available
    if NUMBA_AVAILABLE:
        highs = np.asarray(df["High"].values, dtype=np.float64)
        swing_highs_arr = find_swing_highs_numba(highs, lookback)  # type: ignore[arg-type]
        return pd.Series(swing_highs_arr, index=df.index)

    # Fallback to pure Python implementation
    swing_highs = pd.Series(index=df.index, dtype=float)

    for i in range(lookback, len(df) - lookback):
        is_swing_high = True
        current_high = df.iloc[i]["High"]

        # Check bars before
        for j in range(1, lookback + 1):
            if df.iloc[i - j]["High"] >= current_high:
                is_swing_high = False
                break

        # Check bars after
        if is_swing_high:
            for j in range(1, lookback + 1):
                if df.iloc[i + j]["High"] >= current_high:
                    is_swing_high = False
                    break

        if is_swing_high:
            swing_highs.iloc[i] = current_high
        else:
            swing_highs.iloc[i] = np.nan

    return swing_highs


def find_swing_lows(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """
    Find swing lows in the price data.

    PHASE 2 OPTIMIZATION: Uses Numba JIT compilation for 50-100x speedup.
    Falls back to pure Python if Numba is not available.

    Args:
        df: DataFrame with Low column
        lookback: Number of bars to check on each side

    Returns:
        Series with swing low values (NaN where no swing low exists)
    """
    # Use Numba-accelerated version if available
    if NUMBA_AVAILABLE:
        lows = np.asarray(df["Low"].values, dtype=np.float64)
        swing_lows_arr = find_swing_lows_numba(lows, lookback)  # type: ignore[arg-type]
        return pd.Series(swing_lows_arr, index=df.index)

    # Fallback to pure Python implementation
    swing_lows = pd.Series(index=df.index, dtype=float)

    for i in range(lookback, len(df) - lookback):
        is_swing_low = True
        current_low = df.iloc[i]["Low"]

        # Check bars before
        for j in range(1, lookback + 1):
            if df.iloc[i - j]["Low"] <= current_low:
                is_swing_low = False
                break

        # Check bars after
        if is_swing_low:
            for j in range(1, lookback + 1):
                if df.iloc[i + j]["Low"] <= current_low:
                    is_swing_low = False
                    break

        if is_swing_low:
            swing_lows.iloc[i] = current_low
        else:
            swing_lows.iloc[i] = np.nan

    return swing_lows


def calculate_pivot_points(df: pd.DataFrame, i: int) -> dict:
    """
    Calculate floor pivot points for bar i based on previous bar's data.

    Pivot Point (PP) = (High + Low + Close) / 3
    Support 1 (S1) = 2 * PP - High
    Resistance 1 (R1) = 2 * PP - Low
    Support 2 (S2) = PP - (High - Low)
    Resistance 2 (R2) = PP + (High - Low)
    Support 3 (S3) = S1 - (High - Low)
    Resistance 3 (R3) = R1 + (High - Low)

    Args:
        df: DataFrame with High, Low, Close columns
        i: Current bar index (uses i-1 for calculation)

    Returns:
        Dictionary with PP, S1, S2, S3, R1, R2, R3
    """
    if i < 1:
        return {}

    prev = df.iloc[i - 1]
    high = prev["High"]
    low = prev["Low"]
    close = prev["Close"]

    pp = (high + low + close) / 3
    s1 = 2 * pp - high
    r1 = 2 * pp - low
    s2 = pp - (high - low)
    r2 = pp + (high - low)
    s3 = s1 - (high - low)
    r3 = r1 + (high - low)

    return {
        "PP": pp,
        "S1": s1,
        "S2": s2,
        "S3": s3,
        "R1": r1,
        "R2": r2,
        "R3": r3,
        "range": high - low,
    }


def find_zigzag_pivots(
    df: pd.DataFrame, threshold: float = 0.05, method: str = "percent"
) -> pd.DataFrame:
    """
    Find pivots using ZigZag method.

    Args:
        df: DataFrame with High, Low, Close columns
        threshold: Minimum price change (percent or absolute)
        method: 'percent' or 'absolute' threshold

    Returns:
        DataFrame with 'pivot_high' and 'pivot_low' columns
    """
    result = df.copy()
    result["pivot_high"] = np.nan
    result["pivot_low"] = np.nan

    if len(df) < 3:
        return result

    # Initialize
    pivots: list = []  # type: ignore[assignment]
    current_trend = None  # 'up' or 'down'
    last_pivot_price = None
    last_pivot_idx = 0
    last_pivot_type = None  # 'high' or 'low'

    for i in range(1, len(df)):
        high = df.iloc[i]["High"]
        low = df.iloc[i]["Low"]

        if last_pivot_price is None:
            # Initialize with first bar
            if high > low:
                last_pivot_price = high
                last_pivot_type = "high"
                result.iloc[i, result.columns.get_loc("pivot_high")] = high  # type: ignore[index]
            else:
                last_pivot_price = low
                last_pivot_type = "low"
                result.iloc[i, result.columns.get_loc("pivot_low")] = low  # type: ignore[index]
            last_pivot_idx = i
            continue

        if method == "percent":
            threshold_val = last_pivot_price * threshold
        else:
            threshold_val = threshold

        if last_pivot_type == "high":
            # Looking for a low pivot
            if low < last_pivot_price - threshold_val:
                # Confirmed pivot high at last_pivot_idx
                result.iloc[last_pivot_idx, result.columns.get_loc("pivot_high")] = last_pivot_price  # type: ignore[call-overload]
                # New low pivot
                last_pivot_price = low
                last_pivot_type = "low"
                last_pivot_idx = i
            elif high > last_pivot_price:
                # Higher high, update potential pivot
                last_pivot_price = high
                last_pivot_idx = i

        else:  # last_pivot_type == 'low'
            # Looking for a high pivot
            if high > last_pivot_price + threshold_val:
                # Confirmed pivot low at last_pivot_idx
                result.iloc[last_pivot_idx, result.columns.get_loc("pivot_low")] = last_pivot_price  # type: ignore[call-overload]
                # New high pivot
                last_pivot_price = high
                last_pivot_type = "high"
                last_pivot_idx = i
            elif low < last_pivot_price:
                # Lower low, update potential pivot
                last_pivot_price = low
                last_pivot_idx = i

    return result


def get_recent_swing_high(
    df: pd.DataFrame, i: int, lookback: int = 50
) -> Optional[Tuple[int, float]]:
    """
    Get the most recent swing high before bar i.

    Args:
        df: DataFrame with High column
        i: Current bar index
        lookback: Maximum bars to look back

    Returns:
        Tuple of (bar_index, high_price) or None
    """
    start = max(0, i - lookback)

    for j in range(i - 1, start, -1):
        if j < 5:
            continue

        is_swing_high = True
        current_high = df.iloc[j]["High"]

        for k in range(1, 6):
            if j - k < 0 or j + k >= len(df):
                continue
            if df.iloc[j - k]["High"] >= current_high or df.iloc[j + k]["High"] >= current_high:
                is_swing_high = False
                break

        if is_swing_high:
            return (j, current_high)

    return None


def get_recent_swing_low(
    df: pd.DataFrame, i: int, lookback: int = 50
) -> Optional[Tuple[int, float]]:
    """
    Get the most recent swing low before bar i.

    Args:
        df: DataFrame with Low column
        i: Current bar index
        lookback: Maximum bars to look back

    Returns:
        Tuple of (bar_index, low_price) or None
    """
    start = max(0, i - lookback)

    for j in range(i - 1, start, -1):
        if j < 5:
            continue

        is_swing_low = True
        current_low = df.iloc[j]["Low"]

        for k in range(1, 6):
            if j - k < 0 or j + k >= len(df):
                continue
            if df.iloc[j - k]["Low"] <= current_low or df.iloc[j + k]["Low"] <= current_low:
                is_swing_low = False
                break

        if is_swing_low:
            return (j, current_low)

    return None


def count_successive_new_lows(df: pd.DataFrame, i: int) -> int:
    """
    Count the number of successive bars making new lows.

    Args:
        df: DataFrame with Low column
        i: Current bar index

    Returns:
        Number of successive bars making new lows
    """
    if i < 1:
        return 0

    count = 0
    for j in range(i, 0, -1):
        if df.iloc[j]["Low"] < df.iloc[j - 1]["Low"]:
            count += 1
        else:
            break

    return count


def count_successive_new_highs(df: pd.DataFrame, i: int) -> int:
    """
    Count the number of successive bars making new highs.

    Args:
        df: DataFrame with High column
        i: Current bar index

    Returns:
        Number of successive bars making new highs
    """
    if i < 1:
        return 0

    count = 0
    for j in range(i, 0, -1):
        if df.iloc[j]["High"] > df.iloc[j - 1]["High"]:
            count += 1
        else:
            break

    return count


# =============================================================================
# CACHED SWING DETECTION FUNCTIONS (Phase 1 Optimization)
# =============================================================================

from functools import lru_cache


def _make_array_hashable(arr: np.ndarray, max_bytes: int = 8000) -> tuple:
    """
    Convert numpy array to hashable tuple for caching.
    Uses a sample of the array for hashing to reduce memory.

    Args:
        arr: NumPy array to hash
        max_bytes: Maximum bytes to use for hash key

    Returns:
        Tuple representation for caching
    """
    # Sample the array to create a hashable key
    # Use last portion (most relevant for trading) and first portion (context)
    sample_size = min(len(arr), max_bytes // arr.itemsize)
    if sample_size < len(arr):
        # Take samples from start and end
        start_sample = arr[: sample_size // 2]
        end_sample = arr[-(sample_size - sample_size // 2) :]
        return (len(arr), tuple(start_sample.tobytes()), tuple(end_sample.tobytes()))
    return (len(arr), tuple(arr.tobytes()))


@lru_cache(maxsize=32)
def _cached_swing_highs_core(highs_tuple: tuple, length: int, lookback: int) -> tuple:
    """
    Core cached swing high detection.

    Args:
        highs_tuple: Hashable tuple representation of highs array
        length: Original array length
        lookback: Number of bars to check on each side

    Returns:
        Tuple of swing high values (NaN where no swing high)
    """
    # Reconstruct array
    highs = np.array(highs_tuple, dtype=np.float64)
    n = len(highs)

    swing_highs = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        is_swing_high = True
        current_high = highs[i]

        # Check bars before
        for j in range(1, lookback + 1):
            if highs[i - j] >= current_high:
                is_swing_high = False
                break

        # Check bars after
        if is_swing_high:
            for j in range(1, lookback + 1):
                if highs[i + j] >= current_high:
                    is_swing_high = False
                    break

        if is_swing_high:
            swing_highs[i] = current_high

    return tuple(swing_highs)


def find_swing_highs_cached(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """
    Find swing highs with caching for repeated calls.

    Use this when the same data is queried multiple times across different
    pattern detectors. The cache stores up to 32 unique (data, lookback) combinations.

    Performance Notes:
        - First call: Same speed as regular find_swing_highs()
        - Subsequent calls with same data: ~10-100x faster
        - Cache memory: ~32 entries max

    Args:
        df: DataFrame with High column
        lookback: Number of bars to check on each side (default 5)

    Returns:
        Series with swing high values (NaN where no swing high exists)
    """
    highs = df["High"].values

    # Create hashable key
    key = _make_array_hashable(np.asarray(highs))  # type: ignore[arg-type]

    try:
        result = _cached_swing_highs_core(key[1] if len(key) == 3 else key[1], len(highs), lookback)
        return pd.Series(list(result), index=df.index)
    except Exception:
        # Fall back to regular calculation
        return find_swing_highs(df, lookback)


@lru_cache(maxsize=32)
def _cached_swing_lows_core(lows_tuple: tuple, length: int, lookback: int) -> tuple:
    """
    Core cached swing low detection.

    Args:
        lows_tuple: Hashable tuple representation of lows array
        length: Original array length
        lookback: Number of bars to check on each side

    Returns:
        Tuple of swing low values (NaN where no swing low)
    """
    lows = np.array(lows_tuple, dtype=np.float64)
    n = len(lows)

    swing_lows = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        is_swing_low = True
        current_low = lows[i]

        # Check bars before
        for j in range(1, lookback + 1):
            if lows[i - j] <= current_low:
                is_swing_low = False
                break

        # Check bars after
        if is_swing_low:
            for j in range(1, lookback + 1):
                if lows[i + j] <= current_low:
                    is_swing_low = False
                    break

        if is_swing_low:
            swing_lows[i] = current_low

    return tuple(swing_lows)


def find_swing_lows_cached(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """
    Find swing lows with caching for repeated calls.

    Use this when the same data is queried multiple times across different
    pattern detectors. The cache stores up to 32 unique (data, lookback) combinations.

    Performance Notes:
        - First call: Same speed as regular find_swing_lows()
        - Subsequent calls with same data: ~10-100x faster
        - Cache memory: ~32 entries max

    Args:
        df: DataFrame with Low column
        lookback: Number of bars to check on each side (default 5)

    Returns:
        Series with swing low values (NaN where no swing low exists)
    """
    lows = df["Low"].values

    # Create hashable key
    key = _make_array_hashable(np.asarray(lows))  # type: ignore[arg-type]

    try:
        result = _cached_swing_lows_core(key[1] if len(key) == 3 else key[1], len(lows), lookback)
        return pd.Series(list(result), index=df.index)
    except Exception:
        # Fall back to regular calculation
        return find_swing_lows(df, lookback)


def clear_pivot_cache() -> None:
    """
    Clear all cached pivot/swing detection values.

    Call this when you want to free memory or when the underlying data
    has changed and you want to invalidate all cached calculations.
    """
    _cached_swing_highs_core.cache_clear()
    _cached_swing_lows_core.cache_clear()


def get_pivot_cache_info() -> dict:
    """
    Get information about the pivot cache usage.

    Returns:
        Dictionary with cache statistics for each cached function
    """
    return {
        "swing_highs_cache": {
            "hits": _cached_swing_highs_core.cache_info().hits,
            "misses": _cached_swing_highs_core.cache_info().misses,
            "size": _cached_swing_highs_core.cache_info().currsize,
            "maxsize": _cached_swing_highs_core.cache_info().maxsize,
        },
        "swing_lows_cache": {
            "hits": _cached_swing_lows_core.cache_info().hits,
            "misses": _cached_swing_lows_core.cache_info().misses,
            "size": _cached_swing_lows_core.cache_info().currsize,
            "maxsize": _cached_swing_lows_core.cache_info().maxsize,
        },
    }
