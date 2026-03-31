"""
Unit Tests for IndicatorCache

Tests the indicator caching functionality for Phase 1 optimizations.
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators.indicator_cache import IndicatorCache


class TestIndicatorCache:
    """Tests for IndicatorCache class."""

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
        return pd.DataFrame(
            {
                "Open": 100 + np.cumsum(np.random.randn(n)),
                "High": 101 + np.cumsum(np.random.randn(n)),
                "Low": 99 + np.cumsum(np.random.randn(n)),
                "Close": 100 + np.cumsum(np.random.randn(n)),
                "Volume": np.random.randint(1000, 10000, n),
            },
            index=dates,
        )

    def test_cache_initialization(self, sample_df):
        """Test IndicatorCache initializes correctly."""
        cache = IndicatorCache(sample_df)

        assert cache._df is sample_df
        assert len(cache._cache) == 0
        assert cache._length == 100
        assert cache.arrays is not None

    def test_arrays_extraction(self, sample_df):
        """Test NumPy arrays are extracted correctly."""
        cache = IndicatorCache(sample_df)

        arrays = cache.arrays
        assert "open" in arrays
        assert "high" in arrays
        assert "low" in arrays
        assert "close" in arrays
        assert "volume" in arrays

        assert len(arrays["close"]) == 100
        assert isinstance(arrays["close"], np.ndarray)

    def test_get_bar_fast(self, sample_df):
        """Test fast bar access returns correct values."""
        cache = IndicatorCache(sample_df)

        bar = cache.get_bar(0)
        assert bar is not None
        assert bar["open"] == sample_df["Open"].iloc[0]
        assert bar["close"] == sample_df["Close"].iloc[0]

        bar = cache.get_bar(50)
        assert bar is not None
        assert bar["high"] == sample_df["High"].iloc[50]

    def test_get_bar_out_of_bounds(self, sample_df):
        """Test get_bar returns None for out of bounds indices."""
        cache = IndicatorCache(sample_df)

        assert cache.get_bar(-1) is None
        assert cache.get_bar(100) is None
        assert cache.get_bar(1000) is None

    def test_get_bars_range(self, sample_df):
        """Test range extraction returns correct slice."""
        cache = IndicatorCache(sample_df)

        bars = cache.get_bars_range(10, 20)
        assert bars is not None
        assert len(bars["close"]) == 10
        np.testing.assert_array_equal(bars["close"], sample_df["Close"].values[10:20])

    def test_get_bars_range_invalid(self, sample_df):
        """Test get_bars_range returns None for invalid ranges."""
        cache = IndicatorCache(sample_df)

        assert cache.get_bars_range(-1, 10) is None
        assert cache.get_bars_range(50, 200) is None
        assert cache.get_bars_range(20, 10) is None

    def test_get_swing_highs_caching(self, sample_df):
        """Test swing highs are cached properly."""
        cache = IndicatorCache(sample_df)

        # First call should compute
        result1 = cache.get_swing_highs(lookback=5)
        assert "swing_highs_5" in cache._cache

        # Second call should return cached value
        result2 = cache.get_swing_highs(lookback=5)
        assert result1 is result2  # Same object reference

    def test_get_swing_lows_caching(self, sample_df):
        """Test swing lows are cached properly."""
        cache = IndicatorCache(sample_df)

        result1 = cache.get_swing_lows(lookback=5)
        assert "swing_lows_5" in cache._cache

        result2 = cache.get_swing_lows(lookback=5)
        assert result1 is result2

    def test_get_sma_caching(self, sample_df):
        """Test SMA is cached properly."""
        cache = IndicatorCache(sample_df)

        result1 = cache.get_sma("Close", 20)
        assert "sma_Close_20" in cache._cache

        result2 = cache.get_sma("Close", 20)
        assert result1 is result2

    def test_get_ema_caching(self, sample_df):
        """Test EMA is cached properly."""
        cache = IndicatorCache(sample_df)

        result1 = cache.get_ema("Close", 20)
        assert "ema_Close_20" in cache._cache

        result2 = cache.get_ema("Close", 20)
        assert result1 is result2

    def test_get_atr_caching(self, sample_df):
        """Test ATR is cached properly."""
        cache = IndicatorCache(sample_df)

        result1 = cache.get_atr(14)
        assert "atr_14" in cache._cache

        result2 = cache.get_atr(14)
        assert result1 is result2

    def test_get_volume_sma_caching(self, sample_df):
        """Test Volume SMA is cached properly."""
        cache = IndicatorCache(sample_df)

        result1 = cache.get_volume_sma(20)
        assert "volume_sma_20" in cache._cache

        result2 = cache.get_volume_sma(20)
        assert result1 is result2

    def test_pre_compute_common(self, sample_df):
        """Test pre_compute_common populates cache."""
        cache = IndicatorCache(sample_df)
        cache.pre_compute_common()

        # Check that common indicators are cached
        assert "swing_highs_5" in cache._cache
        assert "swing_lows_5" in cache._cache
        assert "sma_Close_20" in cache._cache
        assert "sma_Close_50" in cache._cache
        assert "ema_Close_20" in cache._cache
        assert "atr_14" in cache._cache
        assert "volume_sma_20" in cache._cache

    def test_clear_cache(self, sample_df):
        """Test clear() empties the cache."""
        cache = IndicatorCache(sample_df)
        cache.get_swing_highs(5)
        cache.get_sma("Close", 20)

        assert len(cache._cache) > 0

        cache.clear()
        assert len(cache._cache) == 0

    def test_get_cache_stats(self, sample_df):
        """Test get_cache_stats returns correct info."""
        cache = IndicatorCache(sample_df)
        cache.get_swing_highs(5)
        cache.get_sma("Close", 20)

        stats = cache.get_cache_stats()
        assert "total_cached" in stats
        assert "cached_keys" in stats
        assert "data_length" in stats
        assert stats["total_cached"] == 2
        assert stats["data_length"] == 100

    def test_has_cached(self, sample_df):
        """Test has_cached method."""
        cache = IndicatorCache(sample_df)

        assert not cache.has_cached("swing_highs_5")
        cache.get_swing_highs(5)
        assert cache.has_cached("swing_highs_5")

    def test_set_and_get_cached(self, sample_df):
        """Test set_cached and get_cached methods."""
        cache = IndicatorCache(sample_df)

        cache.set_cached("custom_key", "custom_value")
        assert cache.get_cached("custom_key") == "custom_value"
        assert cache.get_cached("nonexistent") is None

    def test_df_without_volume(self):
        """Test cache works with DataFrame without Volume column."""
        np.random.seed(42)
        n = 50
        dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
        df = pd.DataFrame(
            {
                "Open": 100 + np.cumsum(np.random.randn(n)),
                "High": 101 + np.cumsum(np.random.randn(n)),
                "Low": 99 + np.cumsum(np.random.randn(n)),
                "Close": 100 + np.cumsum(np.random.randn(n)),
            },
            index=dates,
        )

        cache = IndicatorCache(df)
        assert cache.arrays["volume"] is None

        # Volume SMA should return NaN series
        vol_sma = cache.get_volume_sma(20)
        assert vol_sma.isna().all()
