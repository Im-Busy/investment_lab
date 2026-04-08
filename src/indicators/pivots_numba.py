"""
Numba-Accelerated Pivot Point Detection

This module provides JIT-compiled versions of pivot detection functions
for significant performance improvements (50-100x speedup).

Performance Notes:
- First call has JIT compilation overhead (~100-500ms)
- Subsequent calls are extremely fast
- Use cache=True to save compiled functions to disk
- nopython=True ensures full compilation without Python fallback
"""

import numpy as np

try:
    import numba  # type: ignore[import-untyped]
    jit = numba.jit
    prange = numba.prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Fallback: create a no-op decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    prange = range


@jit(nopython=True, cache=True)
def find_swing_highs_numba(highs: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    Find swing highs in price data using JIT compilation.

    A swing high is a bar whose high is higher than the highs of
    the lookback bars on each side.

    Args:
        highs: NumPy array of high prices
        lookback: Number of bars to check on each side (default: 5)

    Returns:
        NumPy array with swing high values (NaN where no swing high exists)

    Example:
        >>> highs = np.array([10, 12, 15, 14, 13, 16, 14, 12, 11, 13.0])
        >>> swing_highs = find_swing_highs_numba(highs, lookback=2)
        # Index 2 (value 15) is a swing high
    """
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

    return swing_highs


@jit(nopython=True, cache=True)
def find_swing_lows_numba(lows: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    Find swing lows in price data using JIT compilation.

    A swing low is a bar whose low is lower than the lows of
    the lookback bars on each side.

    Args:
        lows: NumPy array of low prices
        lookback: Number of bars to check on each side (default: 5)

    Returns:
        NumPy array with swing low values (NaN where no swing low exists)

    Example:
        >>> lows = np.array([10, 8, 5, 7, 9, 4, 6, 8, 10, 7.0])
        >>> swing_lows = find_swing_lows_numba(lows, lookback=2)
        # Index 2 (value 5) and index 5 (value 4) are swing lows
    """
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

    return swing_lows


@jit(nopython=True, cache=True)
def find_local_extrema_numba(highs: np.ndarray, lows: np.ndarray, lookback: int = 5) -> tuple:
    """
    Find local highs and lows using the fractal method with JIT compilation.

    A local high occurs when the bar's high is higher than the highs
    of the lookback bars on each side. Similarly for local lows.

    Args:
        highs: NumPy array of high prices
        lows: NumPy array of low prices
        lookback: Number of bars to look on each side (default: 5)

    Returns:
        Tuple of (local_highs, local_lows) as boolean NumPy arrays
    """
    n = len(highs)
    local_highs = np.zeros(n, dtype=np.bool_)
    local_lows = np.zeros(n, dtype=np.bool_)

    for i in range(lookback, n - lookback):
        # Check for local high
        is_high = True
        is_low = True

        for j in range(1, lookback + 1):
            # Check for local high
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_high = False
            # Check for local low
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_low = False

        local_highs[i] = is_high
        local_lows[i] = is_low

    return local_highs, local_lows


@jit(nopython=True, cache=True)
def find_pivot_points_numba(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> tuple:
    """
    Calculate floor pivot points for all bars using JIT compilation.

    Pivot Point (PP) = (High + Low + Close) / 3
    Support 1 (S1) = 2 * PP - High
    Resistance 1 (R1) = 2 * PP - Low
    Support 2 (S2) = PP - (High - Low)
    Resistance 2 (R2) = PP + (High - Low)
    Support 3 (S3) = S1 - (High - Low)
    Resistance 3 (R3) = R1 + (High - Low)

    Args:
        highs: NumPy array of high prices
        lows: NumPy array of low prices
        closes: NumPy array of close prices

    Returns:
        Tuple of (pp, s1, s2, s3, r1, r2, r3) arrays
    """
    n = len(highs)

    pp = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    r3 = np.full(n, np.nan)

    for i in range(1, n):
        prev_high = highs[i - 1]
        prev_low = lows[i - 1]
        prev_close = closes[i - 1]

        pp[i] = (prev_high + prev_low + prev_close) / 3.0
        s1[i] = 2.0 * pp[i] - prev_high
        r1[i] = 2.0 * pp[i] - prev_low
        s2[i] = pp[i] - (prev_high - prev_low)
        r2[i] = pp[i] + (prev_high - prev_low)
        s3[i] = s1[i] - (prev_high - prev_low)
        r3[i] = r1[i] + (prev_high - prev_low)

    return pp, s1, s2, s3, r1, r2, r3


@jit(nopython=True, cache=True)
def get_recent_swing_high_numba(
    highs: np.ndarray, current_idx: int, lookback: int = 5, max_bars: int = 50
) -> float:
    """
    Find the most recent swing high before the current bar.

    Args:
        highs: NumPy array of high prices
        current_idx: Current bar index
        lookback: Lookback period for swing detection
        max_bars: Maximum bars to look back

    Returns:
        Most recent swing high price, or NaN if not found
    """
    n = len(highs)
    start_idx = max(lookback, current_idx - max_bars)

    for i in range(current_idx - 1, start_idx - 1, -1):
        if i < lookback or i >= n - lookback:
            continue

        is_swing_high = True
        current_high = highs[i]

        for j in range(1, lookback + 1):
            if highs[i - j] >= current_high or highs[i + j] >= current_high:
                is_swing_high = False
                break

        if is_swing_high:
            return float(current_high)

    return np.nan


@jit(nopython=True, cache=True)
def get_recent_swing_low_numba(
    lows: np.ndarray, current_idx: int, lookback: int = 5, max_bars: int = 50
) -> float:
    """
    Find the most recent swing low before the current bar.

    Args:
        lows: NumPy array of low prices
        current_idx: Current bar index
        lookback: Lookback period for swing detection
        max_bars: Maximum bars to look back

    Returns:
        Most recent swing low price, or NaN if not found
    """
    n = len(lows)
    start_idx = max(lookback, current_idx - max_bars)

    for i in range(current_idx - 1, start_idx - 1, -1):
        if i < lookback or i >= n - lookback:
            continue

        is_swing_low = True
        current_low = lows[i]

        for j in range(1, lookback + 1):
            if lows[i - j] <= current_low or lows[i + j] <= current_low:
                is_swing_low = False
                break

        if is_swing_low:
            return float(current_low)

    return np.nan


@jit(nopython=True, cache=True)
def find_higher_highs_numba(highs: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    Find higher highs in price series.

    A higher high is a swing high that is higher than the previous swing high.

    Args:
        highs: NumPy array of high prices
        lookback: Lookback period for swing detection

    Returns:
        Boolean array where True indicates a higher high
    """
    n = len(highs)
    swing_highs = find_swing_highs_numba(highs, lookback)
    higher_highs = np.zeros(n, dtype=np.bool_)

    last_swing_high = np.nan

    for i in range(n):
        if not np.isnan(swing_highs[i]):
            if np.isnan(last_swing_high):
                last_swing_high = swing_highs[i]
            elif swing_highs[i] > last_swing_high:
                higher_highs[i] = True
                last_swing_high = swing_highs[i]
            else:
                last_swing_high = swing_highs[i]

    return higher_highs


@jit(nopython=True, cache=True)
def find_lower_lows_numba(lows: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    Find lower lows in price series.

    A lower low is a swing low that is lower than the previous swing low.

    Args:
        lows: NumPy array of low prices
        lookback: Lookback period for swing detection

    Returns:
        Boolean array where True indicates a lower low
    """
    n = len(lows)
    swing_lows = find_swing_lows_numba(lows, lookback)
    lower_lows = np.zeros(n, dtype=np.bool_)

    last_swing_low = np.nan

    for i in range(n):
        if not np.isnan(swing_lows[i]):
            if np.isnan(last_swing_low):
                last_swing_low = swing_lows[i]
            elif swing_lows[i] < last_swing_low:
                lower_lows[i] = True
                last_swing_low = swing_lows[i]
            else:
                last_swing_low = swing_lows[i]

    return lower_lows


@jit(nopython=True, cache=True, parallel=True)
def find_all_swings_numba(highs: np.ndarray, lows: np.ndarray, lookback: int = 5) -> tuple:
    """
    Find all swing highs and swing lows in parallel.

    This is an optimized version that computes both swing highs and lows
    in a single pass with parallel processing.

    Args:
        highs: NumPy array of high prices
        lows: NumPy array of low prices
        lookback: Lookback period for swing detection

    Returns:
        Tuple of (swing_highs, swing_lows) arrays
    """
    n = len(highs)
    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)

    for i in prange(lookback, n - lookback):
        # Check swing high
        is_swing_high = True
        current_high = highs[i]
        for j in range(1, lookback + 1):
            if highs[i - j] >= current_high or highs[i + j] >= current_high:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs[i] = current_high

        # Check swing low
        is_swing_low = True
        current_low = lows[i]
        for j in range(1, lookback + 1):
            if lows[i - j] <= current_low or lows[i + j] <= current_low:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows[i] = current_low

    return swing_highs, swing_lows


# Warm-up function to pre-compile JIT functions
def warmup():
    """
    Pre-compile all JIT functions by calling them with small arrays.

    Call this at application startup to avoid JIT compilation overhead
    during actual backtesting.
    """
    if not NUMBA_AVAILABLE:
        return

    # Small test arrays for warm-up
    n = 20
    highs = np.random.random(n) * 100 + 100
    lows = np.random.random(n) * 100 + 95
    closes = np.random.random(n) * 100 + 100

    # Call all JIT functions to trigger compilation
    find_swing_highs_numba(highs, 5)
    find_swing_lows_numba(lows, 5)
    find_local_extrema_numba(highs, lows, 5)
    find_pivot_points_numba(highs, lows, closes)
    get_recent_swing_high_numba(highs, 15, 5, 10)
    get_recent_swing_low_numba(lows, 15, 5, 10)
    find_higher_highs_numba(highs, 5)
    find_lower_lows_numba(lows, 5)
    find_all_swings_numba(highs, lows, 5)


if __name__ == "__main__":
    # Test the functions
    print("Testing Numba-accelerated pivot detection...")

    # Generate test data
    np.random.seed(42)
    n = 1000
    highs = np.random.random(n) * 100 + 100
    lows = np.random.random(n) * 100 + 95
    closes = np.random.random(n) * 100 + 100

    # Warm up JIT
    print("Warming up JIT compilation...")
    warmup()

    # Test swing highs
    import time

    start = time.perf_counter()
    swing_highs = find_swing_highs_numba(highs, 5)
    elapsed = time.perf_counter() - start
    print(f"Swing highs computed in {elapsed * 1000:.3f}ms")
    print(f"Found {np.sum(~np.isnan(swing_highs))} swing highs")

    # Test swing lows
    start = time.perf_counter()
    swing_lows = find_swing_lows_numba(lows, 5)
    elapsed = time.perf_counter() - start
    print(f"Swing lows computed in {elapsed * 1000:.3f}ms")
    print(f"Found {np.sum(~np.isnan(swing_lows))} swing lows")

    print("\nNumba functions are ready!")
