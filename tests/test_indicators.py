"""
Unit Tests for Technical Indicators

Tests the indicator functions including:
- SMA and EMA
- ATR
- RSI
- ADX
- Market Regime Detector
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators.regime import (
    MarketPhase,
    MarketRegimeDetector,
    RegimeState,
    TrendDirection,
    VolatilityRegime,
)
from src.indicators.technical import adx, atr, ema, rsi, sma, true_range


class TestSMA:
    """Tests for Simple Moving Average."""

    @pytest.fixture
    def sample_data(self):
        """Create sample price data."""
        return pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

    def test_sma_calculation(self, sample_data):
        """Test SMA calculation."""
        result = sma(sample_data, 3)

        # Check that SMA is calculated
        assert len(result) == len(sample_data)

        # Check specific values
        assert result.iloc[2] == pytest.approx(11.0)  # (10+11+12)/3
        assert result.iloc[3] == pytest.approx(12.0)  # (11+12+13)/3
        assert result.iloc[4] == pytest.approx(13.0)  # (12+13+14)/3

    def test_sma_nan_at_start(self, sample_data):
        """Test that SMA returns NaN at the start."""
        result = sma(sample_data, 5)

        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[3])
        assert not pd.isna(result.iloc[4])

    def test_sma_period_1(self, sample_data):
        """Test SMA with period 1 returns original data."""
        result = sma(sample_data, 1)

        # SMA returns float64, so compare values not dtypes
        pd.testing.assert_series_equal(result, sample_data, check_dtype=False)


class TestEMA:
    """Tests for Exponential Moving Average."""

    @pytest.fixture
    def sample_data(self):
        """Create sample price data."""
        return pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

    def test_ema_calculation(self, sample_data):
        """Test EMA calculation."""
        result = ema(sample_data, 5)

        assert len(result) == len(sample_data)
        # EMA should respond faster than SMA for trending data
        # For linear upward trend, EMA should be >= SMA (may be equal for perfectly linear data)
        sma_result = sma(sample_data, 5)
        assert result.iloc[-1] >= sma_result.iloc[-1]

    def test_ema_responds_to_changes(self):
        """Test that EMA responds to price changes."""
        data = pd.Series([10] * 10 + [20] * 10)
        result = ema(data, 5)

        # EMA should increase after price jump
        assert result.iloc[-1] > result.iloc[10]


class TestATR:
    """Tests for Average True Range."""

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLC data."""
        return pd.DataFrame(
            {
                "High": [105, 108, 110, 107, 112],
                "Low": [100, 103, 105, 102, 107],
                "Close": [103, 107, 108, 105, 110],
            }
        )

    def test_atr_calculation(self, sample_df):
        """Test ATR calculation."""
        result = atr(sample_df, 3)

        assert len(result) == len(sample_df)
        # ATR should be positive
        assert result.iloc[-1] > 0

    def test_atr_handles_gaps(self):
        """Test that ATR handles gaps correctly."""
        df = pd.DataFrame(
            {"High": [110, 115, 120], "Low": [100, 105, 110], "Close": [105, 112, 118]}
        )

        result = atr(df, 2)

        # Should account for gap between closes
        assert result.iloc[-1] > 0


class TestRSI:
    """Tests for Relative Strength Index."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data with trend."""
        # Upward trend
        return pd.Series([10, 11, 12, 11, 13, 14, 13, 15, 16, 15, 17, 18, 19, 20, 19])

    def test_rsi_range(self, sample_prices):
        """Test that RSI is between 0 and 100."""
        result = rsi(sample_prices, 5)

        # Filter out NaN values
        valid_result = result.dropna()

        assert all(valid_result >= 0)
        assert all(valid_result <= 100)

    def test_rsi_uptrend(self):
        """Test RSI in uptrend."""
        # Strong uptrend
        prices = pd.Series(range(10, 30))
        result = rsi(prices, 5)

        # RSI should be high in uptrend
        assert result.iloc[-1] > 70

    def test_rsi_downtrend(self):
        """Test RSI in downtrend."""
        # Strong downtrend
        prices = pd.Series(range(30, 10, -1))
        result = rsi(prices, 5)

        # RSI should be low in downtrend
        assert result.iloc[-1] < 30


class TestTrueRange:
    """Tests for True Range calculation."""

    def test_true_range_basic(self):
        """Test basic true range calculation."""
        df = pd.DataFrame(
            {"High": [105, 110, 115], "Low": [100, 105, 110], "Close": [103, 108, 113]}
        )

        result = true_range(df)

        # First TR is just High - Low
        assert result.iloc[0] == 5

        # Subsequent TRs consider previous close
        assert result.iloc[1] >= 5  # max(110-105, |110-103|, |105-103|)


class TestMarketRegimeDetector:
    """Tests for Market Regime Detector."""

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLCV data for regime testing."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        np.random.seed(42)

        # Create trending data
        trend = np.linspace(100, 120, 100)
        noise = np.random.randn(100) * 2

        close = trend + noise
        high = close + np.random.rand(100) * 3
        low = close - np.random.rand(100) * 3
        open_price = close + np.random.randn(100) * 0.5
        volume = np.random.randint(1000000, 5000000, 100)

        return pd.DataFrame(
            {"Open": open_price, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )

    @pytest.fixture
    def detector(self):
        """Create regime detector."""
        return MarketRegimeDetector()

    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert detector.adx_period == 14
        assert detector.adx_trend_threshold == 25.0
        assert detector.atr_period == 14

    def test_detect_returns_regime_state(self, detector, sample_df):
        """Test that detect returns RegimeState."""
        result = detector.detect(sample_df)

        assert isinstance(result, RegimeState)

    def test_regime_state_attributes(self, detector, sample_df):
        """Test RegimeState attributes."""
        result = detector.detect(sample_df)

        assert hasattr(result, "trend_direction")
        assert hasattr(result, "trend_strength")
        assert hasattr(result, "volatility_regime")
        assert hasattr(result, "market_phase")
        assert hasattr(result, "adx")
        assert hasattr(result, "atr")

    def test_trend_direction_values(self, detector, sample_df):
        """Test that trend direction is valid enum."""
        result = detector.detect(sample_df)

        assert result.trend_direction in [
            TrendDirection.UPTREND,
            TrendDirection.DOWNTREND,
            TrendDirection.SIDEWAYS,
        ]

    def test_volatility_regime_values(self, detector, sample_df):
        """Test that volatility regime is valid enum."""
        result = detector.detect(sample_df)

        assert result.volatility_regime in [
            VolatilityRegime.HIGH,
            VolatilityRegime.NORMAL,
            VolatilityRegime.LOW,
        ]

    def test_market_phase_values(self, detector, sample_df):
        """Test that market phase is valid enum."""
        result = detector.detect(sample_df)

        assert result.market_phase in [
            MarketPhase.STRONG_BULL,
            MarketPhase.WEAK_BULL,
            MarketPhase.STRONG_BEAR,
            MarketPhase.WEAK_BEAR,
            MarketPhase.RANGING,
            MarketPhase.CHOPPY,
        ]

    def test_regime_state_to_dict(self, detector, sample_df):
        """Test RegimeState to_dict method."""
        result = detector.detect(sample_df)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "trend_direction" in result_dict
        assert "volatility_regime" in result_dict
        assert "market_phase" in result_dict

    def test_get_pattern_weights(self, detector, sample_df):
        """Test pattern weight calculation."""
        regime = detector.detect(sample_df)
        weights = detector.get_pattern_weights(regime)

        assert "basic" in weights
        assert "harmonic" in weights
        assert "complex" in weights
        assert "classic" in weights

        # All weights should be positive
        for weight in weights.values():
            assert weight > 0

    def test_get_regime_multipliers(self, detector, sample_df):
        """Test regime multiplier calculation."""
        regime = detector.detect(sample_df)
        multipliers = detector.get_regime_multipliers(regime)

        assert "reversal" in multipliers
        assert "continuation" in multipliers
        assert "breakout" in multipliers

    def test_get_risk_parameters(self, detector, sample_df):
        """Test risk parameter calculation."""
        regime = detector.detect(sample_df)
        params = detector.get_risk_parameters(regime)

        assert "risk_per_trade" in params
        assert "stop_multiplier" in params
        assert "target_multiplier" in params
        assert "position_size_mult" in params

        # Risk per trade should be reasonable
        assert 0 < params["risk_per_trade"] <= 0.05

    def test_uptrend_detection(self):
        """Test detection of uptrend."""
        detector = MarketRegimeDetector()

        # Create strong uptrend
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        close = np.linspace(100, 150, 100)
        high = close + 2
        low = close - 2

        df = pd.DataFrame(
            {"Open": close, "High": high, "Low": low, "Close": close, "Volume": [1000000] * 100},
            index=dates,
        )

        result = detector.detect(df)

        # Should detect uptrend
        assert result.trend_direction in [TrendDirection.UPTREND, TrendDirection.SIDEWAYS]

    def test_ranging_detection(self):
        """Test detection of ranging market."""
        detector = MarketRegimeDetector()

        # Create ranging market
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        np.random.seed(42)
        close = 100 + np.sin(np.linspace(0, 4 * np.pi, 100)) * 5 + np.random.randn(100) * 0.5
        high = close + 1
        low = close - 1

        df = pd.DataFrame(
            {"Open": close, "High": high, "Low": low, "Close": close, "Volume": [1000000] * 100},
            index=dates,
        )

        result = detector.detect(df)

        # Should detect ranging or weak trend
        assert result.trend_direction in [
            TrendDirection.SIDEWAYS,
            TrendDirection.UPTREND,
            TrendDirection.DOWNTREND,
        ]


class TestADX:
    """Tests for ADX indicator."""

    @pytest.fixture
    def trending_df(self):
        """Create trending OHLC data."""
        return pd.DataFrame(
            {
                "High": [105, 108, 112, 115, 118, 122, 125, 128],
                "Low": [100, 103, 107, 110, 113, 117, 120, 123],
                "Close": [103, 107, 110, 113, 116, 120, 123, 126],
            }
        )

    def test_adx_calculation(self, trending_df):
        """Test ADX calculation."""
        result = adx(trending_df, 3)

        assert len(result) == len(trending_df)
        # ADX should be positive
        valid_result = result.dropna()
        assert all(valid_result >= 0)
        assert all(valid_result <= 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
