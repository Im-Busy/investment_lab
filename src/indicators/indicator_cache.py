"""
Indicator Cache for Performance Optimization

Pre-computes and caches common indicators to avoid redundant calculations
across multiple pattern detectors.

This module provides a centralized caching mechanism for frequently-used
indicators like swing highs/lows, moving averages, ATR, and volume metrics.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .pivots import find_swing_highs, find_swing_lows
from .technical import atr, ema, sma, true_range


def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Volume Simple Moving Average.

    Args:
        df: DataFrame with Volume column
        period: SMA period

    Returns:
        Volume SMA series
    """
    if "Volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["Volume"].rolling(window=period).mean()


class IndicatorCache:
    """
    Pre-compute indicators once per backtest and cache for reuse.

    This class addresses the performance bottleneck where each pattern
    detector independently calculates the same indicators (swing highs/lows,
    moving averages, ATR, etc.). By pre-computing and caching these values,
    we eliminate redundant calculations.

    Usage:
        cache = IndicatorCache(df)
        cache.pre_compute_common()  # Pre-compute most used indicators

        # Get cached indicators
        swing_highs = cache.get_swing_highs(lookback=5)
        sma_20 = cache.get_sma('Close', 20)
        atr_14 = cache.get_atr(14)

        # Access pre-extracted NumPy arrays for faster access
        close_array = cache.arrays['close']

    Performance Notes:
        - Pre-extracted NumPy arrays provide O(1) access vs O(n) DataFrame access
        - Cached indicators avoid recalculation across 20+ pattern detectors
        - Expected speedup: 5-10x for indicator-heavy patterns
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize cache with DataFrame.

        Args:
            df: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
        """
        self._df = df
        self._cache: Dict[str, Any] = {}

        # Pre-extract NumPy arrays for faster access
        # This avoids DataFrame column access overhead in hot loops
        self._arrays: Dict[str, Optional[np.ndarray]] = {
            "open": df["Open"].to_numpy().copy() if "Open" in df.columns else None,
            "high": df["High"].to_numpy().copy() if "High" in df.columns else None,
            "low": df["Low"].to_numpy().copy() if "Low" in df.columns else None,
            "close": df["Close"].to_numpy().copy() if "Close" in df.columns else None,
            "volume": df["Volume"].to_numpy().copy() if "Volume" in df.columns else None,
        }
        self._index = df.index
        self._length = len(df)

    @property
    def arrays(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Get pre-extracted NumPy arrays for fast access.

        Returns:
            Dictionary with 'open', 'high', 'low', 'close', 'volume' arrays
        """
        return self._arrays

    @property
    def df(self) -> pd.DataFrame:
        """Get the underlying DataFrame."""
        return self._df

    @property
    def length(self) -> int:
        """Get the number of bars in the dataset."""
        return self._length

    @property
    def index(self) -> pd.Index:
        """Get the DataFrame index."""
        return self._index

    def get_swing_highs(self, lookback: int = 5) -> pd.Series:
        """
        Get cached swing highs.

        Args:
            lookback: Number of bars to check on each side

        Returns:
            Series with swing high values (NaN where no swing high exists)
        """
        key = f"swing_highs_{lookback}"
        if key not in self._cache:
            self._cache[key] = find_swing_highs(self._df, lookback)
        return self._cache[key]

    def get_swing_lows(self, lookback: int = 5) -> pd.Series:
        """
        Get cached swing lows.

        Args:
            lookback: Number of bars to check on each side

        Returns:
            Series with swing low values (NaN where no swing low exists)
        """
        key = f"swing_lows_{lookback}"
        if key not in self._cache:
            self._cache[key] = find_swing_lows(self._df, lookback)
        return self._cache[key]

    def get_sma(self, column: str, period: int) -> pd.Series:
        """
        Get cached Simple Moving Average.

        Args:
            column: Column name (e.g., 'Close', 'High', 'Low')
            period: SMA period

        Returns:
            SMA series
        """
        key = f"sma_{column}_{period}"
        if key not in self._cache:
            if column in self._df.columns:
                self._cache[key] = sma(self._df[column], period)
            else:
                self._cache[key] = pd.Series(np.nan, index=self._index)
        return self._cache[key]

    def get_ema(self, column: str, period: int) -> pd.Series:
        """
        Get cached Exponential Moving Average.

        Args:
            column: Column name (e.g., 'Close', 'High', 'Low')
            period: EMA period

        Returns:
            EMA series
        """
        key = f"ema_{column}_{period}"
        if key not in self._cache:
            if column in self._df.columns:
                self._cache[key] = ema(self._df[column], period)
            else:
                self._cache[key] = pd.Series(np.nan, index=self._index)
        return self._cache[key]

    def get_atr(self, period: int = 14) -> pd.Series:
        """
        Get cached Average True Range.

        Args:
            period: ATR period (default 14)

        Returns:
            ATR series
        """
        key = f"atr_{period}"
        if key not in self._cache:
            self._cache[key] = atr(self._df, period)
        return self._cache[key]

    def get_volume_sma(self, period: int = 20) -> pd.Series:
        """
        Get cached Volume SMA.

        Args:
            period: SMA period for volume

        Returns:
            Volume SMA series
        """
        key = f"volume_sma_{period}"
        if key not in self._cache:
            self._cache[key] = volume_sma(self._df, period)
        return self._cache[key]

    def get_true_range(self) -> pd.Series:
        """
        Get cached True Range.

        Returns:
            True Range series
        """
        key = "true_range"
        if key not in self._cache:
            self._cache[key] = true_range(self._df)
        return self._cache[key]

    def get_average_range(self, period: int = 20) -> pd.Series:
        """
        Get cached Average Range (High - Low).

        Args:
            period: Lookback period

        Returns:
            Average Range series
        """
        key = f"avg_range_{period}"
        if key not in self._cache:
            if self._arrays["high"] is not None and self._arrays["low"] is not None:
                ranges = self._arrays["high"] - self._arrays["low"]
                self._cache[key] = (
                    pd.Series(ranges, index=self._index).rolling(window=period).mean()
                )
            else:
                self._cache[key] = pd.Series(np.nan, index=self._index)
        return self._cache[key]

    def pre_compute_common(self) -> None:
        """
        Pre-compute most commonly used indicators.

        Call this method during strategy initialization to populate
        the cache with indicators used by most pattern detectors.

        Pre-computed indicators:
            - Swing highs/lows (lookback=5)
            - SMA(20), SMA(50) on Close
            - EMA(20) on Close
            - ATR(14)
            - Volume SMA(20)
        """
        # Swing points with default lookback
        self.get_swing_highs(5)
        self.get_swing_lows(5)

        # Common moving averages
        self.get_sma("Close", 20)
        self.get_sma("Close", 50)
        self.get_ema("Close", 20)

        # ATR for stop-loss calculations
        self.get_atr(14)

        # Volume metrics
        self.get_volume_sma(20)

        # Average range
        self.get_average_range(20)

    def get_cached(self, key: str) -> Optional[Any]:
        """
        Get a cached value by key.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        return self._cache.get(key)

    def set_cached(self, key: str, value: Any) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = value

    def has_cached(self, key: str) -> bool:
        """
        Check if a key is cached.

        Args:
            key: Cache key

        Returns:
            True if key exists in cache
        """
        return key in self._cache

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "total_cached": len(self._cache),
            "cached_keys": list(self._cache.keys()),
            "data_length": self._length,
        }

    def get_bar(self, i: int) -> Optional[Dict[str, float]]:
        """
        Get bar data at index i using cached arrays (faster than DataFrame access).

        Args:
            i: Bar index

        Returns:
            Dictionary with OHLCV values or None if index out of bounds
        """
        if i < 0 or i >= self._length:
            return None

        return {
            "open": self._arrays["open"][i] if self._arrays["open"] is not None else np.nan,
            "high": self._arrays["high"][i] if self._arrays["high"] is not None else np.nan,
            "low": self._arrays["low"][i] if self._arrays["low"] is not None else np.nan,
            "close": self._arrays["close"][i] if self._arrays["close"] is not None else np.nan,
            "volume": self._arrays["volume"][i] if self._arrays["volume"] is not None else np.nan,
        }

    def get_bars_range(self, start: int, end: int) -> Optional[Dict[str, Optional[np.ndarray]]]:
        """
        Get a range of bars using cached arrays (faster than DataFrame slicing).

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)

        Returns:
            Dictionary with OHLCV arrays or None if range invalid
        """
        if start < 0 or end > self._length or start >= end:
            return None

        return {
            "open": self._arrays["open"][start:end] if self._arrays["open"] is not None else None,
            "high": self._arrays["high"][start:end] if self._arrays["high"] is not None else None,
            "low": self._arrays["low"][start:end] if self._arrays["low"] is not None else None,
            "close": self._arrays["close"][start:end]
            if self._arrays["close"] is not None
            else None,
            "volume": self._arrays["volume"][start:end]
            if self._arrays["volume"] is not None
            else None,
        }
