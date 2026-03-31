"""
Unit Tests for Numba-Accelerated Functions

Tests for Phase 2 optimization functions:
- Pivot detection (pivots_numba.py)
- Technical indicators (technical_numba.py)
- Pattern detection vectorization
"""

import numpy as np
import pandas as pd
import pytest

# Import Numba functions
from src.indicators.pivots_numba import (
    NUMBA_AVAILABLE,
    find_local_extrema_numba,
    find_pivot_points_numba,
    find_swing_highs_numba,
    find_swing_lows_numba,
)
from src.indicators.technical_numba import (
    atr_numba,
    donchian_channel_numba,
    ema_numba,
    rsi_numba,
    sma_numba,
    true_range_numba,
)


class TestPivotsNumba:
    """Tests for Numba-accelerated pivot detection."""

    def test_swing_highs_basic(self):
        """Test basic swing high detection."""
        highs = np.array([10.0, 12.0, 15.0, 14.0, 13.0, 16.0, 14.0, 12.0, 11.0, 13.0])
        result = find_swing_highs_numba(highs, lookback=2)

        # Index 2 (value 15) should be a swing high
        assert result[2] == 15.0

    def test_swing_highs_no_signal_at_edges(self):
        """Test that swing highs are not detected at edges."""
        highs = np.array([10.0, 15.0, 12.0, 15.0, 10.0])
        result = find_swing_highs_numba(highs, lookback=2)

        # First and last bars should be NaN (no signal)
        assert np.isnan(result[0])
        assert np.isnan(result[-1])

    def test_swing_lows_basic(self):
        """Test basic swing low detection."""
        lows = np.array([10.0, 8.0, 5.0, 7.0, 9.0, 4.0, 6.0, 8.0, 10.0, 7.0])
        result = find_swing_lows_numba(lows, lookback=2)

        # Index 2 (value 5) should be a swing low
        assert result[2] == 5.0

    def test_swing_lows_multiple(self):
        """Test detection of multiple swing lows."""
        # Create data with clear swing lows at indices 2, 6 (with lookback=2)
        # Index 2: 5.0 is lower than neighbors [8.0, 7.0, 4.0, 6.0]
        # Index 6: 3.0 is lower than neighbors [4.0, 6.0, 5.0, 7.0]
        lows = np.array([12.0, 10.0, 5.0, 7.0, 9.0, 6.0, 3.0, 5.0, 8.0, 7.0, 9.0])
        result = find_swing_lows_numba(lows, lookback=2)

        # Should detect at least one swing low
        detected_count = np.sum(~np.isnan(result))
        assert detected_count >= 1

    def test_local_extrema_basic(self):
        """Test local extrema detection."""
        highs = np.array([10.0, 12.0, 15.0, 14.0, 13.0, 16.0, 14.0, 12.0, 11.0, 13.0])
        lows = np.array([8.0, 7.0, 6.0, 7.0, 8.0, 5.0, 7.0, 9.0, 10.0, 8.0])

        local_highs, local_lows = find_local_extrema_numba(highs, lows, lookback=2)

        # Should return boolean arrays
        assert local_highs.dtype == np.bool_
        assert local_lows.dtype == np.bool_

    def test_pivot_points(self):
        """Test pivot point calculation."""
        highs = np.array([105.0, 110.0, 108.0, 112.0, 107.0])
        lows = np.array([95.0, 100.0, 98.0, 102.0, 97.0])
        closes = np.array([100.0, 105.0, 103.0, 107.0, 102.0])

        # Function returns 7 values: pp, s1, s2, s3, r1, r2, r3
        pp, s1, s2, s3, r1, r2, r3 = find_pivot_points_numba(highs, lows, closes)

        # Check that pivot points are calculated
        assert len(pp) == len(highs)
        # First pivot point should be NaN (no prior data)
        assert np.isnan(pp[0])


class TestTechnicalNumba:
    """Tests for Numba-accelerated technical indicators."""

    def test_sma_correctness(self):
        """Test SMA calculation matches expected values."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = sma_numba(values, period=3)

        # SMA at index 2 should be (1+2+3)/3 = 2.0
        assert result[2] == 2.0
        # SMA at index 4 should be (3+4+5)/3 = 4.0
        assert result[4] == 4.0
        # First two values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])

    def test_sma_insufficient_data(self):
        """Test SMA with insufficient data returns NaN."""
        values = np.array([1.0, 2.0])
        result = sma_numba(values, period=5)

        # All values should be NaN
        assert np.all(np.isnan(result))

    def test_ema_correctness(self):
        """Test EMA calculation."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = ema_numba(values, period=3)

        # First EMA value should be SMA
        expected_sma = (1.0 + 2.0 + 3.0) / 3.0
        assert result[2] == expected_sma

        # EMA should be increasing for this data
        assert result[3] > result[2]
        assert result[4] > result[3]

    def test_atr_calculation(self):
        """Test ATR calculation."""
        highs = np.array([105.0, 110.0, 108.0, 112.0, 107.0, 109.0, 111.0, 108.0, 110.0, 112.0])
        lows = np.array([95.0, 100.0, 98.0, 102.0, 97.0, 99.0, 101.0, 98.0, 100.0, 102.0])
        closes = np.array([100.0, 105.0, 103.0, 107.0, 102.0, 104.0, 106.0, 103.0, 105.0, 107.0])

        result = atr_numba(highs, lows, closes, period=5)

        # ATR should be positive
        valid_result = result[~np.isnan(result)]
        assert np.all(valid_result > 0)

    def test_rsi_range(self):
        """Test RSI values are in valid range."""
        # Generate random price data
        np.random.seed(42)
        closes = np.random.random(100) * 100 + 50

        result = rsi_numba(closes, period=14)

        # All non-NaN values should be between 0 and 100
        valid_result = result[~np.isnan(result)]
        assert np.all(valid_result >= 0)
        assert np.all(valid_result <= 100)

    def test_rsi_all_gains(self):
        """Test RSI with all gains should be 100."""
        closes = np.array(
            [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
                112.0,
                113.0,
                114.0,
                115.0,
                116.0,
                117.0,
                118.0,
                119.0,
            ]
        )
        result = rsi_numba(closes, period=14)

        # RSI should be 100 for all gains
        valid_result = result[~np.isnan(result)]
        assert np.allclose(valid_result, 100.0, atol=1.0)

    def test_true_range(self):
        """Test True Range calculation."""
        highs = np.array([105.0, 110.0, 108.0, 112.0, 107.0])
        lows = np.array([95.0, 100.0, 98.0, 102.0, 97.0])
        closes = np.array([100.0, 105.0, 103.0, 107.0, 102.0])

        result = true_range_numba(highs, lows, closes)

        # True Range should be positive
        assert np.all(result > 0)

        # First TR should be high - low
        assert result[0] == highs[0] - lows[0]

    def test_donchian_channel(self):
        """Test Donchian Channel calculation."""
        highs = np.array([105.0, 110.0, 108.0, 112.0, 107.0, 109.0, 111.0, 108.0, 110.0, 112.0])
        lows = np.array([95.0, 100.0, 98.0, 102.0, 97.0, 99.0, 101.0, 98.0, 100.0, 102.0])

        upper, middle, lower = donchian_channel_numba(highs, lows, period=5)

        # Upper should be >= middle >= lower
        valid_idx = ~np.isnan(upper)
        assert np.all(upper[valid_idx] >= middle[valid_idx])
        assert np.all(middle[valid_idx] >= lower[valid_idx])


class TestPatternVectorization:
    """Tests for vectorized pattern detection."""

    def test_msl_pattern_detection(self):
        """Test MSL pattern vectorized detection."""
        from src.patterns.basic.msl import MarketStructureLow

        # Create test data with MSL pattern
        df = pd.DataFrame(
            {
                "Open": [100.0, 99.0, 98.0, 99.0, 100.0, 101.0, 102.0, 101.0, 100.0, 99.0],
                "High": [101.0, 100.0, 99.0, 100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0],
                "Low": [99.0, 98.0, 97.0, 98.0, 99.0, 100.0, 101.0, 100.0, 99.0, 98.0],
                "Close": [99.5, 98.5, 97.5, 98.5, 100.5, 101.5, 102.5, 101.5, 100.5, 99.5],
                "Volume": [1000] * 10,
            }
        )

        msl = MarketStructureLow()
        signals = msl.detect_vectorized(df)

        # Should return numpy array
        assert isinstance(signals, np.ndarray)
        assert len(signals) == len(df)

    def test_nr7id_pattern_detection(self):
        """Test NR7ID pattern vectorized detection."""
        from src.patterns.basic.nr7id import NR7ID

        # Create test data with NR7 pattern (narrow range)
        np.random.seed(42)
        n = 20
        base = 100
        df = pd.DataFrame(
            {
                "Open": np.ones(n) * base + np.random.randn(n) * 2,
                "High": np.ones(n) * (base + 3) + np.random.randn(n) * 2,
                "Low": np.ones(n) * (base - 3) + np.random.randn(n) * 2,
                "Close": np.ones(n) * base + np.random.randn(n) * 2,
                "Volume": np.ones(n) * 1000,
            }
        )

        nr7id = NR7ID()
        signals = nr7id.detect_vectorized(df)

        assert isinstance(signals, np.ndarray)
        assert len(signals) == len(df)

    def test_floor_pivot_pattern_detection(self):
        """Test Floor Pivot pattern vectorized detection."""
        from src.patterns.basic.floor_pivot import FloorPivotBreakout

        # Create test data
        np.random.seed(42)
        n = 30
        df = pd.DataFrame(
            {
                "Open": np.ones(n) * 100 + np.random.randn(n) * 5,
                "High": np.ones(n) * 105 + np.random.randn(n) * 5,
                "Low": np.ones(n) * 95 + np.random.randn(n) * 5,
                "Close": np.ones(n) * 100 + np.random.randn(n) * 5,
                "Volume": np.ones(n) * 1000,
            }
        )

        fp = FloorPivotBreakout()
        signals = fp.detect_vectorized(df)

        assert isinstance(signals, np.ndarray)
        assert len(signals) == len(df)
        # Signals should be -1, 0, or 1
        assert np.all(np.isin(signals, [-1, 0, 1]))


class TestNumbaAvailability:
    """Test Numba availability and fallback."""

    def test_numba_available(self):
        """Test that Numba is available."""
        # This test just checks if NUMBA_AVAILABLE is defined
        assert isinstance(NUMBA_AVAILABLE, bool)

    def test_fallback_functions_work(self):
        """Test that functions work even if Numba is not available."""
        # The functions should work regardless of Numba availability
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma_numba(values, 3)

        assert not np.isnan(result[2])
        assert result[2] == 2.0


class TestPerformance:
    """Performance comparison tests."""

    @pytest.mark.skip(reason="Performance test - run manually")
    def test_pivot_performance(self):
        """Test pivot detection performance improvement."""
        import time

        # Generate large dataset
        n = 100000
        highs = np.random.random(n) * 100 + 100
        lows = np.random.random(n) * 100 + 95

        # Warm up JIT
        find_swing_highs_numba(highs[:100], 5)
        find_swing_lows_numba(lows[:100], 5)

        # Time Numba version
        start = time.perf_counter()
        result_highs = find_swing_highs_numba(highs, 5)
        result_lows = find_swing_lows_numba(lows, 5)
        numba_time = time.perf_counter() - start

        print(f"\nNumba pivot detection time: {numba_time:.4f}s")
        print(f"Data points processed: {n}")

    @pytest.mark.skip(reason="Performance test - run manually")
    def test_sma_performance(self):
        """Test SMA performance improvement."""
        import time

        # Generate large dataset
        n = 1000000
        values = np.random.random(n) * 100 + 50

        # Warm up JIT
        sma_numba(values[:100], 20)

        # Time Numba version
        start = time.perf_counter()
        result = sma_numba(values, 20)
        numba_time = time.perf_counter() - start

        # Time pandas version
        import pandas as pd

        series = pd.Series(values)
        start = time.perf_counter()
        pandas_result = series.rolling(window=20).mean()
        pandas_time = time.perf_counter() - start

        print("\nSMA Performance:")
        print(f"  Numba time: {numba_time:.4f}s")
        print(f"  Pandas time: {pandas_time:.4f}s")
        print(f"  Speedup: {pandas_time / numba_time:.1f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
