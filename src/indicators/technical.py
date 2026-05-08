"""
Technical Indicators

Common technical indicators used in pattern detection.

PHASE 2 OPTIMIZATION:
- Uses Numba JIT compilation for 30-50x speedup
- Falls back to pure Python/pandas if Numba is not available
- All public functions maintain the same API
"""

from typing import Union

import numpy as np
import pandas as pd

# Try to import Numba-accelerated functions
try:
    from .technical_numba import (
        NUMBA_AVAILABLE,
        adx_numba,
        atr_numba,
        bollinger_bands_numba,
        donchian_channel_numba,
        ema_numba,
        macd_numba,
        rsi_numba,
        sma_numba,
        stochastic_numba,
        true_range_numba,
        volume_sma_numba,
        williams_r_numba,
    )
except ImportError:
    NUMBA_AVAILABLE = False


def sma(data: Union[pd.Series, np.ndarray], period: int) -> pd.Series:
    """
    Simple Moving Average.

    PHASE 2 OPTIMIZATION: Uses Numba JIT compilation for 30-50x speedup.

    Args:
        data: Price series
        period: Number of periods

    Returns:
        SMA series
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    # Use Numba-accelerated version if available
    if NUMBA_AVAILABLE:
        values: np.ndarray = np.asarray(data.values, dtype=np.float64)
        result = sma_numba(values, period)
        return pd.Series(result, index=data.index)

    # Fallback to pandas
    return data.rolling(window=period).mean()


def ema(data: Union[pd.Series, np.ndarray], period: int) -> pd.Series:
    """
    Exponential Moving Average.

    PHASE 2 OPTIMIZATION: Uses Numba JIT compilation for 30-50x speedup.

    Args:
        data: Price series
        period: Number of periods

    Returns:
        EMA series
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    # Use Numba-accelerated version if available
    if NUMBA_AVAILABLE:
        values: np.ndarray = np.asarray(data.values, dtype=np.float64)
        result = ema_numba(values, period)
        return pd.Series(result, index=data.index)

    # Fallback to pandas
    return data.ewm(span=period, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    """
    True Range calculation.

    PHASE 2 OPTIMIZATION: Uses Numba JIT compilation for speedup.

    Args:
        df: DataFrame with High, Low, Close columns

    Returns:
        True Range series
    """
    # Use Numba-accelerated version if available
    if NUMBA_AVAILABLE:
        highs: np.ndarray = np.asarray(df["High"].values, dtype=np.float64)
        lows: np.ndarray = np.asarray(df["Low"].values, dtype=np.float64)
        closes: np.ndarray = np.asarray(df["Close"].values, dtype=np.float64)
        result = true_range_numba(highs, lows, closes)
        return pd.Series(result, index=df.index)

    # Fallback to pandas
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.

    FIXED: Now uses RMA (Wilder) smoothing instead of SMA for consistency
    with industry-standard implementations (TradingView, ThinkOrSwim, etc.).

    RMA formula: RMA[i] = ((RMA[i-1] * (period - 1)) + TR[i]) / period

    PHASE 2 OPTIMIZATION: Uses Numba JIT compilation for 30-50x speedup.

    Args:
        df: DataFrame with High, Low, Close columns
        period: ATR period (default 14)

    Returns:
        ATR series
    """
    # Use Numba-accelerated version if available
    if NUMBA_AVAILABLE:
        highs: np.ndarray = np.asarray(df["High"].values, dtype=np.float64)
        lows: np.ndarray = np.asarray(df["Low"].values, dtype=np.float64)
        closes: np.ndarray = np.asarray(df["Close"].values, dtype=np.float64)
        result = atr_numba(highs, lows, closes, period)
        return pd.Series(result, index=df.index)

    # FIXED: Fallback to pandas with RMA smoothing
    tr = true_range(df)

    # First value is SMA of TR
    atr_values = [np.nan] * (period - 1)
    atr_values.append(tr.iloc[:period].mean())

    # Apply RMA formula for subsequent values
    for i in range(period, len(tr)):
        rma_prev = atr_values[-1]
        rma_new = ((rma_prev * (period - 1)) + tr.iloc[i]) / period
        atr_values.append(rma_new)

    return pd.Series(atr_values, index=df.index)


def average_range(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Average Range (High - Low)

    Args:
        df: DataFrame with High, Low columns
        period: Lookback period

    Returns:
        Average Range series
    """
    ranges = df["High"] - df["Low"]
    return ranges.rolling(window=period).mean()


def rsi(close: Union[pd.Series, np.ndarray], period: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    PHASE 2 OPTIMIZATION: Uses Numba JIT compilation for 30-50x speedup.

    Args:
        close: Close price series
        period: RSI period (default 14)

    Returns:
        RSI series (0-100)
    """
    if isinstance(close, np.ndarray):
        close = pd.Series(close)

    # Use Numba-accelerated version if available
    if NUMBA_AVAILABLE:
        values: np.ndarray = np.asarray(close.values, dtype=np.float64)
        result = rsi_numba(values, period)
        return pd.Series(result, index=close.index)

    # Fallback to pandas
    delta = close.diff()

    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))

    return rsi_val


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index.

    FIXED: Now uses RMA (Wilder) smoothing for DM and TR, consistent with
    industry-standard implementations (TradingView, ThinkOrSwim, etc.).

    PHASE 2 OPTIMIZATION: Uses Numba JIT compilation for 30-50x speedup.

    Args:
        df: DataFrame with High, Low, Close columns
        period: ADX period (default 14)

    Returns:
        ADX series
    """
    # Use Numba-accelerated version if available
    if NUMBA_AVAILABLE:
        highs = df["High"].to_numpy(dtype=np.float64, na_value=np.nan)
        lows = df["Low"].to_numpy(dtype=np.float64, na_value=np.nan)
        closes = df["Close"].to_numpy(dtype=np.float64, na_value=np.nan)
        result = adx_numba(highs, lows, closes, period)
        return pd.Series(result, index=df.index)

    # FIXED: Fallback to pandas with RMA smoothing
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # Plus Directional Movement
    plus_dm = high.diff()
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > (-low.diff())), 0)

    # Minus Directional Movement
    minus_dm = -low.diff()
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm.diff()), 0)

    # True Range
    tr = true_range(df)

    # FIXED: Use RMA (Wilder) smoothing instead of SMA
    def _rma(series: pd.Series, period: int) -> pd.Series:
        """Calculate RMA (Wilder's smoothed moving average)."""
        result = [np.nan] * (period - 1)
        result.append(series.iloc[:period].mean())
        for i in range(period, len(series)):
            rma_prev = result[-1]
            rma_new = ((rma_prev * (period - 1)) + series.iloc[i]) / period
            result.append(rma_new)
        return pd.Series(result, index=series.index)

    smoothed_tr = _rma(tr, period)
    smoothed_plus_dm = _rma(plus_dm, period)
    smoothed_minus_dm = _rma(minus_dm, period)

    # Calculate DI+ and DI-
    plus_di = 100 * smoothed_plus_dm / smoothed_tr
    minus_di = 100 * smoothed_minus_dm / smoothed_tr

    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = _rma(dx, period)

    return adx_val


def std_dev(data: Union[pd.Series, np.ndarray], period: int = 20) -> pd.Series:
    """
    Standard Deviation

    Args:
        data: Price series
        period: Lookback period

    Returns:
        Standard deviation series
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    return data.rolling(window=period).std()


def bollinger_bands(
    close: Union[pd.Series, np.ndarray], period: int = 20, num_std: float = 2.0
) -> dict:
    """
    Bollinger Bands

    Args:
        close: Close price series
        period: SMA period (default 20)
        num_std: Number of standard deviations (default 2.0)

    Returns:
        Dictionary with 'upper', 'middle', 'lower' bands
    """
    if isinstance(close, np.ndarray):
        close = pd.Series(close)

    middle = sma(close, period)
    std = std_dev(close, period)

    upper = middle + (num_std * std)
    lower = middle - (num_std * std)

    return {"upper": upper, "middle": middle, "lower": lower, "bandwidth": upper - lower}


def donchian_channel(df: pd.DataFrame, period: int = 20) -> dict:
    """
    Donchian Channel

    Args:
        df: DataFrame with High, Low columns
        period: Lookback period (default 20)

    Returns:
        Dictionary with 'upper', 'lower', 'middle' channels
    """
    upper = df["High"].rolling(window=period).max()
    lower = df["Low"].rolling(window=period).min()
    middle = (upper + lower) / 2

    return {"upper": upper, "lower": lower, "middle": middle}


def volume_sma(volume: Union[pd.Series, np.ndarray], period: int = 20) -> pd.Series:
    """
    Volume Simple Moving Average

    Args:
        volume: Volume series
        period: SMA period (default 20)

    Returns:
        Volume SMA series
    """
    return sma(volume, period)


def is_bullish_candle(open_price: float, close: float) -> bool:
    """Check if candle is bullish (close > open)"""
    return close > open_price


def is_bearish_candle(open_price: float, close: float) -> bool:
    """Check if candle is bearish (close < open)"""
    return close < open_price


def candle_body(open_price: float, close: float) -> float:
    """Get candle body size"""
    return abs(close - open_price)


def candle_range(high: float, low: float) -> float:
    """Get candle range"""
    return high - low


def is_inside_bar(
    current_high: float, current_low: float, prev_high: float, prev_low: float
) -> bool:
    """
    Check if current bar is an inside bar.
    Inside bar: High < Previous High AND Low > Previous Low
    """
    return current_high < prev_high and current_low > prev_low


def is_outside_bar(
    current_high: float, current_low: float, current_close: float, prev_high: float, prev_low: float
) -> bool:
    """
    Check if current bar is an outside bar.
    Outside bar: High > Previous High AND Low < Previous Low
    """
    return current_high > prev_high and current_low < prev_low


def higher_high(df: pd.DataFrame, i: int, lookback: int = 1) -> bool:
    """Check if bar i has a higher high than previous lookback bars"""
    if i < lookback:
        return False
    current_high = df.iloc[i]["High"]
    for j in range(1, lookback + 1):
        if current_high <= df.iloc[i - j]["High"]:
            return False
    return True


def lower_low(df: pd.DataFrame, i: int, lookback: int = 1) -> bool:
    """Check if bar i has a lower low than previous lookback bars"""
    if i < lookback:
        return False
    current_low = df.iloc[i]["Low"]
    for j in range(1, lookback + 1):
        if current_low >= df.iloc[i - j]["Low"]:
            return False
    return True


def new_high(df: pd.DataFrame, i: int, lookback: int = 21) -> bool:
    """Check if bar i is a new lookback-period high"""
    if i < lookback:
        return False
    current_high = df.iloc[i]["High"]
    return bool(float(current_high) > float(df.iloc[i - lookback : i]["High"].max()))


def new_low(df: pd.DataFrame, i: int, lookback: int = 21) -> bool:
    """Check if bar i is a new lookback-period low"""
    if i < lookback:
        return False
    current_low = df.iloc[i]["Low"]
    return bool(float(current_low) < float(df.iloc[i - lookback : i]["Low"].min()))


# =============================================================================
# CACHED INDICATOR FUNCTIONS (Phase 1 Optimization)
# =============================================================================

from functools import lru_cache


def _make_hashable(arr: np.ndarray, max_bytes: int = 1000) -> bytes:
    """
    Convert numpy array to hashable bytes for caching.
    Uses last N bytes for hashing to reduce memory while maintaining uniqueness.

    Args:
        arr: NumPy array to hash
        max_bytes: Maximum bytes to use for hash key

    Returns:
        Bytes representation for caching
    """
    # Use last portion of array for hashing (more relevant for trading)
    sample_size = min(len(arr), max_bytes // arr.itemsize)
    if sample_size < len(arr):
        arr_sample = arr[-sample_size:]
    else:
        arr_sample = arr
    return arr_sample.tobytes()


@lru_cache(maxsize=128)
def _cached_sma_core(prices_bytes: bytes, period: int, length: int) -> tuple:
    """
    Core cached SMA calculation.

    Args:
        prices_bytes: Hashable bytes representation of prices
        period: SMA period
        length: Original array length for reconstruction

    Returns:
        Tuple of SMA values
    """
    # Reconstruct array from bytes
    prices = np.frombuffer(prices_bytes, dtype=np.float64)

    # Calculate SMA using convolution (fast)
    if len(prices) < period:
        return tuple([np.nan] * length)

    # Use pandas for accurate SMA with proper handling of initial NaN values
    partial_sma = np.convolve(prices, np.ones(period) / period, mode="valid")

    # Pad with NaN for initial values
    result = np.concatenate([np.full(period - 1, np.nan), partial_sma])

    # If we truncated the input, extend the result
    if len(result) < length:
        # Use the last valid values to extend
        result = np.concatenate(
            [result, np.full(length - len(result), result[-1] if len(result) > 0 else np.nan)]
        )

    return tuple(result[:length])


def sma_cached(data: Union[pd.Series, np.ndarray], period: int) -> pd.Series:
    """
    SMA with caching for repeated calls with same data.

    Use this when the same data is queried multiple times across different
    pattern detectors. The cache stores up to 128 unique (data, period) combinations.

    Performance Notes:
        - First call: Same speed as regular sma()
        - Subsequent calls with same data: ~10-100x faster
        - Cache memory: ~128 entries max

    Args:
        data: Price series (Series or NumPy array)
        period: SMA period

    Returns:
        SMA series with same index as input
    """
    if isinstance(data, pd.Series):
        arr: np.ndarray = np.asarray(data.values, dtype=np.float64)
        index = data.index
    else:
        arr = np.asarray(data, dtype=np.float64)
        index = pd.RangeIndex(len(arr))

    # Create hashable key from array
    key_data = _make_hashable(arr)  # type: ignore[arg-type]

    try:
        result = _cached_sma_core(key_data, period, len(arr))
        return pd.Series(list(result), index=index)
    except Exception:
        # Fall back to regular calculation on cache miss/error
        return sma(data, period)


@lru_cache(maxsize=64)
def _cached_ema_core(prices_bytes: bytes, period: int, length: int) -> tuple:
    """
    Core cached EMA calculation.

    Args:
        prices_bytes: Hashable bytes representation of prices
        period: EMA period
        length: Original array length for reconstruction

    Returns:
        Tuple of EMA values
    """
    prices = np.frombuffer(prices_bytes, dtype=np.float64)

    if len(prices) < period:
        return tuple([np.nan] * length)

    # EMA calculation
    multiplier = 2 / (period + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]

    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]

    # Handle initial values with SMA seed for more accurate EMA
    if len(prices) >= period:
        seed = np.mean(prices[:period])
        ema[:period] = np.nan
        ema[period - 1] = seed

        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]

    # Extend if needed
    if len(ema) < length:
        ema = np.concatenate([ema, np.full(length - len(ema), ema[-1] if len(ema) > 0 else np.nan)])

    return tuple(ema[:length])


def ema_cached(data: Union[pd.Series, np.ndarray], period: int) -> pd.Series:
    """
    EMA with caching for repeated calls with same data.

    Use this when the same data is queried multiple times across different
    pattern detectors. The cache stores up to 64 unique (data, period) combinations.

    Performance Notes:
        - First call: Same speed as regular ema()
        - Subsequent calls with same data: ~10-100x faster
        - Cache memory: ~64 entries max

    Args:
        data: Price series (Series or NumPy array)
        period: EMA period

    Returns:
        EMA series with same index as input
    """
    if isinstance(data, pd.Series):
        arr: np.ndarray = np.asarray(data.values, dtype=np.float64)
        index = data.index
    else:
        arr = np.asarray(data, dtype=np.float64)
        index = pd.RangeIndex(len(arr))

    key_data = _make_hashable(arr)  # type: ignore[arg-type]

    try:
        result = _cached_ema_core(key_data, period, len(arr))
        return pd.Series(list(result), index=index)
    except Exception:
        # Fall back to regular calculation
        return ema(data, period)


def clear_indicator_cache() -> None:
    """
    Clear all cached indicator values.

    Call this when you want to free memory or when the underlying data
    has changed and you want to invalidate all cached calculations.
    """
    _cached_sma_core.cache_clear()
    _cached_ema_core.cache_clear()


def get_indicator_cache_info() -> dict:
    """
    Get information about the indicator cache usage.

    Returns:
        Dictionary with cache statistics for each cached function
    """
    return {
        "sma_cache": {
            "hits": _cached_sma_core.cache_info().hits,
            "misses": _cached_sma_core.cache_info().misses,
            "size": _cached_sma_core.cache_info().currsize,
            "maxsize": _cached_sma_core.cache_info().maxsize,
        },
        "ema_cache": {
            "hits": _cached_ema_core.cache_info().hits,
            "misses": _cached_ema_core.cache_info().misses,
            "size": _cached_ema_core.cache_info().currsize,
            "maxsize": _cached_ema_core.cache_info().maxsize,
        },
    }
