"""
Numba-Accelerated Technical Indicators

This module provides JIT-compiled versions of common technical indicators
for significant performance improvements (30-50x speedup).

Performance Notes:
- First call has JIT compilation overhead (~100-500ms)
- Subsequent calls are extremely fast
- Use cache=True to save compiled functions to disk
- nopython=True ensures full compilation without Python fallback
"""

import numpy as np

try:
    from numba import jit, prange

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
def sma_numba(values: np.ndarray, period: int) -> np.ndarray:
    """
    Simple Moving Average - JIT compiled.

    Args:
        values: NumPy array of price values
        period: SMA period

    Returns:
        NumPy array with SMA values (NaN for insufficient data)

    Example:
        >>> values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> sma = sma_numba(values, 3)
        # sma[2] = 2.0, sma[3] = 3.0, sma[4] = 4.0
    """
    n = len(values)
    result = np.full(n, np.nan)

    if n < period:
        return result

    # Calculate first SMA value
    total = 0.0
    for i in range(period):
        total += values[i]
    result[period - 1] = total / period

    # Calculate remaining values using rolling window
    for i in range(period, n):
        total = total - values[i - period] + values[i]
        result[i] = total / period

    return result


@jit(nopython=True, cache=True)
def ema_numba(values: np.ndarray, period: int) -> np.ndarray:
    """
    Exponential Moving Average - JIT compiled.

    Uses the standard EMA formula:
    EMA = (Close - EMA_prev) * multiplier + EMA_prev
    where multiplier = 2 / (period + 1)

    Args:
        values: NumPy array of price values
        period: EMA period

    Returns:
        NumPy array with EMA values (NaN for insufficient data)
    """
    n = len(values)
    result = np.full(n, np.nan)

    if n < period:
        return result

    multiplier = 2.0 / (period + 1)

    # Start with SMA for first value
    total = 0.0
    for i in range(period):
        total += values[i]
    result[period - 1] = total / period

    # Calculate EMA
    for i in range(period, n):
        result[i] = (values[i] - result[i - 1]) * multiplier + result[i - 1]

    return result


@jit(nopython=True, cache=True)
def atr_numba(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """
    Average True Range - JIT compiled.

    True Range is the greatest of:
    - High - Low
    - |High - Previous Close|
    - |Low - Previous Close|

    ATR is the EMA of True Range.

    Args:
        highs: NumPy array of high prices
        lows: NumPy array of low prices
        closes: NumPy array of close prices
        period: ATR period

    Returns:
        NumPy array with ATR values (NaN for insufficient data)
    """
    n = len(highs)
    tr = np.empty(n)
    result = np.full(n, np.nan)

    # Calculate True Range
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, max(hc, lc))

    # Calculate ATR using Wilder's smoothing (similar to EMA)
    if n < period:
        return result

    # Initial ATR is simple average
    total = 0.0
    for i in range(period):
        total += tr[i]
    result[period - 1] = total / period

    # Wilder's smoothing
    for i in range(period, n):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return result


@jit(nopython=True, cache=True)
def rsi_numba(closes: np.ndarray, period: int) -> np.ndarray:
    """
    Relative Strength Index - JIT compiled.

    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss

    Uses Wilder's smoothing for average gain/loss.

    Args:
        closes: NumPy array of close prices
        period: RSI period (typically 14)

    Returns:
        NumPy array with RSI values (0-100, NaN for insufficient data)
    """
    n = len(closes)
    result = np.full(n, np.nan)

    if n < period + 1:
        return result

    # Calculate price changes
    gains = np.empty(n - 1)
    losses = np.empty(n - 1)

    for i in range(n - 1):
        change = closes[i + 1] - closes[i]
        if change > 0:
            gains[i] = change
            losses[i] = 0.0
        else:
            gains[i] = 0.0
            losses[i] = -change

    # Calculate initial average gain/loss
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(period):
        avg_gain += gains[i]
        avg_loss += losses[i]
    avg_gain /= period
    avg_loss /= period

    # Calculate first RSI value
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))

    # Calculate remaining RSI values using Wilder's smoothing
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return result


@jit(nopython=True, cache=True)
def adx_numba(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """
    Average Directional Index - JIT compiled.

    ADX measures trend strength without regard to direction.
    Values above 25 indicate a strong trend.

    Args:
        highs: NumPy array of high prices
        lows: NumPy array of low prices
        closes: NumPy array of close prices
        period: ADX period (typically 14)

    Returns:
        NumPy array with ADX values (0-100, NaN for insufficient data)
    """
    n = len(highs)
    result = np.full(n, np.nan)

    if n < period + 1:
        return result

    # Calculate True Range and Directional Movement
    tr = np.empty(n)
    plus_dm = np.empty(n)
    minus_dm = np.empty(n)

    tr[0] = highs[0] - lows[0]
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0

    for i in range(1, n):
        # True Range
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, max(hc, lc))

        # Directional Movement
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0.0

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0.0

    # Smooth TR and DM using Wilder's smoothing
    smooth_tr = np.full(n, np.nan)
    smooth_plus_dm = np.full(n, np.nan)
    smooth_minus_dm = np.full(n, np.nan)

    # Initial values
    tr_sum = 0.0
    plus_dm_sum = 0.0
    minus_dm_sum = 0.0
    for i in range(period):
        tr_sum += tr[i]
        plus_dm_sum += plus_dm[i]
        minus_dm_sum += minus_dm[i]

    smooth_tr[period - 1] = tr_sum
    smooth_plus_dm[period - 1] = plus_dm_sum
    smooth_minus_dm[period - 1] = minus_dm_sum

    # Wilder's smoothing
    for i in range(period, n):
        smooth_tr[i] = smooth_tr[i - 1] - (smooth_tr[i - 1] / period) + tr[i]
        smooth_plus_dm[i] = smooth_plus_dm[i - 1] - (smooth_plus_dm[i - 1] / period) + plus_dm[i]
        smooth_minus_dm[i] = (
            smooth_minus_dm[i - 1] - (smooth_minus_dm[i - 1] / period) + minus_dm[i]
        )

    # Calculate DI+ and DI-
    plus_di = np.empty(n)
    minus_di = np.empty(n)

    for i in range(period - 1, n):
        if smooth_tr[i] > 0:
            plus_di[i] = 100.0 * smooth_plus_dm[i] / smooth_tr[i]
            minus_di[i] = 100.0 * smooth_minus_dm[i] / smooth_tr[i]
        else:
            plus_di[i] = 0.0
            minus_di[i] = 0.0

    # Calculate DX
    dx = np.full(n, np.nan)
    for i in range(period - 1, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0

    # Calculate ADX (smoothed DX)
    adx = np.full(n, np.nan)

    # Initial ADX
    dx_sum = 0.0
    for i in range(period - 1, period * 2 - 1):
        if i < n:
            dx_sum += dx[i]
    adx[period * 2 - 2] = dx_sum / period

    # Smooth ADX
    for i in range(period * 2 - 1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    # Copy to result
    for i in range(period * 2 - 2, n):
        result[i] = adx[i]

    return result


@jit(nopython=True, cache=True)
def volume_sma_numba(volumes: np.ndarray, period: int) -> np.ndarray:
    """
    Volume Simple Moving Average - JIT compiled.

    Args:
        volumes: NumPy array of volume values
        period: SMA period

    Returns:
        NumPy array with volume SMA values (NaN for insufficient data)
    """
    return sma_numba(volumes, period)  # type: ignore[no-any-return]


@jit(nopython=True, cache=True)
def true_range_numba(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """
    True Range - JIT compiled.

    Args:
        highs: NumPy array of high prices
        lows: NumPy array of low prices
        closes: NumPy array of close prices

    Returns:
        NumPy array with True Range values
    """
    n = len(highs)
    tr = np.empty(n)

    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, max(hc, lc))

    return tr


@jit(nopython=True, cache=True)
def bollinger_bands_numba(closes: np.ndarray, period: int, std_dev: float) -> tuple:
    """
    Bollinger Bands - JIT compiled.

    Middle Band = SMA(period)
    Upper Band = Middle Band + (std_dev * Standard Deviation)
    Lower Band = Middle Band - (std_dev * Standard Deviation)

    Args:
        closes: NumPy array of close prices
        period: SMA period (typically 20)
        std_dev: Standard deviation multiplier (typically 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band) arrays
    """
    n = len(closes)
    middle = sma_numba(closes, period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    for i in range(period - 1, n):
        # Calculate standard deviation
        total = 0.0
        for j in range(period):
            total += closes[i - j]
        mean = total / period

        variance = 0.0
        for j in range(period):
            diff = closes[i - j] - mean
            variance += diff * diff
        std = np.sqrt(variance / period)

        upper[i] = middle[i] + std_dev * std
        lower[i] = middle[i] - std_dev * std

    return upper, middle, lower


@jit(nopython=True, cache=True)
def donchian_channel_numba(highs: np.ndarray, lows: np.ndarray, period: int) -> tuple:
    """
    Donchian Channel - JIT compiled.

    Upper Channel = Highest High over period
    Lower Channel = Lowest Low over period
    Middle Channel = (Upper + Lower) / 2

    Args:
        highs: NumPy array of high prices
        lows: NumPy array of low prices
        period: Channel period

    Returns:
        Tuple of (upper, middle, lower) arrays
    """
    n = len(highs)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)

    for i in range(period - 1, n):
        # Find highest high and lowest low
        highest = highs[i]
        lowest = lows[i]
        for j in range(period):
            if highs[i - j] > highest:
                highest = highs[i - j]
            if lows[i - j] < lowest:
                lowest = lows[i - j]

        upper[i] = highest
        lower[i] = lowest
        middle[i] = (highest + lowest) / 2.0

    return upper, middle, lower


@jit(nopython=True, cache=True, parallel=True)
def macd_numba(closes: np.ndarray, fast_period: int, slow_period: int, signal_period: int) -> tuple:
    """
    MACD (Moving Average Convergence Divergence) - JIT compiled.

    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD, signal_period)
    Histogram = MACD - Signal

    Args:
        closes: NumPy array of close prices
        fast_period: Fast EMA period (typically 12)
        slow_period: Slow EMA period (typically 26)
        signal_period: Signal EMA period (typically 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram) arrays
    """
    ema_fast = ema_numba(closes, fast_period)
    ema_slow = ema_numba(closes, slow_period)

    n = len(closes)
    macd_line = np.full(n, np.nan)

    for i in range(n):
        if not np.isnan(ema_fast[i]) and not np.isnan(ema_slow[i]):
            macd_line[i] = ema_fast[i] - ema_slow[i]

    signal_line = ema_numba(macd_line[~np.isnan(macd_line)], signal_period)

    # Align signal line with macd_line
    signal_aligned = np.full(n, np.nan)
    valid_start = 0
    for i in range(n):
        if not np.isnan(macd_line[i]):
            valid_start = i
            break

    signal_idx = 0
    for i in range(valid_start + slow_period - 1, n):
        if signal_idx < len(signal_line) and not np.isnan(signal_line[signal_idx]):
            signal_aligned[i] = signal_line[signal_idx]
        signal_idx += 1

    histogram = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(macd_line[i]) and not np.isnan(signal_aligned[i]):
            histogram[i] = macd_line[i] - signal_aligned[i]

    return macd_line, signal_aligned, histogram


@jit(nopython=True, cache=True)
def stochastic_numba(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, k_period: int, d_period: int
) -> tuple:
    """
    Stochastic Oscillator - JIT compiled.

    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA(%K, d_period)

    Args:
        highs: NumPy array of high prices
        lows: NumPy array of low prices
        closes: NumPy array of close prices
        k_period: %K period (typically 14)
        d_period: %D period (typically 3)

    Returns:
        Tuple of (k_line, d_line) arrays
    """
    n = len(highs)
    k_line = np.full(n, np.nan)

    for i in range(k_period - 1, n):
        # Find highest high and lowest low
        highest = highs[i]
        lowest = lows[i]
        for j in range(k_period):
            if highs[i - j] > highest:
                highest = highs[i - j]
            if lows[i - j] < lowest:
                lowest = lows[i - j]

        # Calculate %K
        if highest != lowest:
            k_line[i] = (closes[i] - lowest) / (highest - lowest) * 100.0
        else:
            k_line[i] = 50.0  # Neutral when range is zero

    # Calculate %D (SMA of %K)
    d_line = sma_numba(k_line, d_period)

    return k_line, d_line


@jit(nopython=True, cache=True)
def williams_r_numba(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int
) -> np.ndarray:
    """
    Williams %R - JIT compiled.

    %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

    Args:
        highs: NumPy array of high prices
        lows: NumPy array of low prices
        closes: NumPy array of close prices
        period: Lookback period (typically 14)

    Returns:
        NumPy array with Williams %R values (-100 to 0, NaN for insufficient data)
    """
    n = len(highs)
    result = np.full(n, np.nan)

    for i in range(period - 1, n):
        # Find highest high and lowest low
        highest = highs[i]
        lowest = lows[i]
        for j in range(period):
            if highs[i - j] > highest:
                highest = highs[i - j]
            if lows[i - j] < lowest:
                lowest = lows[i - j]

        # Calculate Williams %R
        if highest != lowest:
            result[i] = (highest - closes[i]) / (highest - lowest) * -100.0
        else:
            result[i] = -50.0  # Neutral when range is zero

    return result


# Warm-up function to pre-compile all JIT functions
def warmup():
    """
    Pre-compile all JIT functions by calling them with small arrays.

    Call this at application startup to avoid JIT compilation overhead
    during actual backtesting.
    """
    if not NUMBA_AVAILABLE:
        return

    # Small test arrays for warm-up
    n = 50
    values = np.random.random(n) * 100 + 100
    highs = np.random.random(n) * 100 + 105
    lows = np.random.random(n) * 100 + 95
    closes = np.random.random(n) * 100 + 100
    volumes = np.random.random(n) * 1000000

    # Call all JIT functions to trigger compilation
    sma_numba(values, 10)
    ema_numba(values, 10)
    atr_numba(highs, lows, closes, 14)
    rsi_numba(closes, 14)
    adx_numba(highs, lows, closes, 14)
    volume_sma_numba(volumes, 10)
    true_range_numba(highs, lows, closes)
    bollinger_bands_numba(closes, 20, 2.0)
    donchian_channel_numba(highs, lows, 20)
    macd_numba(closes, 12, 26, 9)
    stochastic_numba(highs, lows, closes, 14, 3)
    williams_r_numba(highs, lows, closes, 14)


if __name__ == "__main__":
    # Test the functions
    print("Testing Numba-accelerated technical indicators...")

    # Generate test data
    np.random.seed(42)
    n = 1000
    values = np.random.random(n) * 100 + 100
    highs = np.random.random(n) * 100 + 105
    lows = np.random.random(n) * 100 + 95
    closes = np.random.random(n) * 100 + 100

    # Warm up JIT
    print("Warming up JIT compilation...")
    warmup()

    # Test SMA
    import time

    start = time.perf_counter()
    sma = sma_numba(values, 20)
    elapsed = time.perf_counter() - start
    print(f"SMA computed in {elapsed * 1000:.3f}ms")

    # Test EMA
    start = time.perf_counter()
    ema = ema_numba(values, 20)
    elapsed = time.perf_counter() - start
    print(f"EMA computed in {elapsed * 1000:.3f}ms")

    # Test RSI
    start = time.perf_counter()
    rsi = rsi_numba(closes, 14)
    elapsed = time.perf_counter() - start
    print(f"RSI computed in {elapsed * 1000:.3f}ms")
    print(f"RSI range: {np.nanmin(rsi):.2f} - {np.nanmax(rsi):.2f}")

    print("\nNumba functions are ready!")
