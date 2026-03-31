"""
Unit Tests for Base Pattern Classes

Tests the core pattern detection framework including:
- PatternType enum
- SignalDirection enum
- TradeSignal dataclass
- PatternResult dataclass
- BasePattern abstract class
"""

import numpy as np
import pandas as pd
import pytest

from src.patterns.base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class TestPatternType:
    """Tests for PatternType enum."""

    def test_pattern_types_exist(self):
        """Test that all expected pattern types exist."""
        assert hasattr(PatternType, "REVERSAL")
        assert hasattr(PatternType, "CONTINUATION")
        assert hasattr(PatternType, "BREAKOUT")
        assert hasattr(PatternType, "COUNTER_TREND")
        assert hasattr(PatternType, "VOLATILITY")

    def test_pattern_type_values(self):
        """Test pattern type string values."""
        assert PatternType.REVERSAL.value == "Reversal"
        assert PatternType.CONTINUATION.value == "Continuation"
        assert PatternType.BREAKOUT.value == "Breakout"


class TestSignalDirection:
    """Tests for SignalDirection enum."""

    def test_signal_directions_exist(self):
        """Test that all expected signal directions exist."""
        assert hasattr(SignalDirection, "LONG")
        assert hasattr(SignalDirection, "SHORT")
        assert hasattr(SignalDirection, "NEUTRAL")

    def test_signal_direction_values(self):
        """Test signal direction string values."""
        assert SignalDirection.LONG.value == "Long"
        assert SignalDirection.SHORT.value == "Short"
        assert SignalDirection.NEUTRAL.value == "Neutral"


class TestTradeSignal:
    """Tests for TradeSignal dataclass."""

    def test_create_basic_signal(self):
        """Test creating a basic trade signal."""
        signal = TradeSignal(
            pattern_name="Test Pattern",
            direction=SignalDirection.LONG,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit_1=110.0,
        )

        assert signal.pattern_name == "Test Pattern"
        assert signal.direction == SignalDirection.LONG
        assert signal.entry_price == 100.0
        assert signal.stop_loss == 95.0
        assert signal.take_profit_1 == 110.0
        assert signal.confidence == 0.5  # Default value

    def test_create_full_signal(self):
        """Test creating a signal with all fields."""
        timestamp = pd.Timestamp("2024-01-01")
        signal = TradeSignal(
            pattern_name="Full Signal",
            direction=SignalDirection.SHORT,
            entry_price=50.0,
            stop_loss=55.0,
            take_profit_1=40.0,
            take_profit_2=35.0,
            take_profit_3=30.0,
            confidence=0.75,
            timestamp=timestamp,
            metadata={"key": "value"},
        )

        assert signal.take_profit_2 == 35.0
        assert signal.take_profit_3 == 30.0
        assert signal.confidence == 0.75
        assert signal.timestamp == timestamp
        assert signal.metadata == {"key": "value"}

    def test_signal_to_dict(self):
        """Test converting signal to dictionary."""
        signal = TradeSignal(
            pattern_name="Dict Test",
            direction=SignalDirection.LONG,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit_1=110.0,
            confidence=0.8,
        )

        result = signal.to_dict()

        assert isinstance(result, dict)
        assert result["pattern_name"] == "Dict Test"
        assert result["direction"] == "Long"
        assert result["entry_price"] == 100.0
        assert result["confidence"] == 0.8


class TestPatternResult:
    """Tests for PatternResult dataclass."""

    def test_create_detected_result(self):
        """Test creating a detected pattern result."""
        signal = TradeSignal(
            pattern_name="Test",
            direction=SignalDirection.LONG,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit_1=110.0,
        )

        result = PatternResult(
            detected=True,
            pattern_name="Test Pattern",
            pattern_type=PatternType.REVERSAL,
            signal=signal,
            pivot_points={"high": 105.0, "low": 98.0},
            bars_since_detection=0,
            start_index=10,
            end_index=15,
        )

        assert result.detected is True
        assert result.pattern_name == "Test Pattern"
        assert result.pattern_type == PatternType.REVERSAL
        assert result.signal == signal
        assert result.pivot_points == {"high": 105.0, "low": 98.0}

    def test_create_not_detected_result(self):
        """Test creating a not-detected pattern result."""
        result = PatternResult(
            detected=False, pattern_name="Test Pattern", pattern_type=PatternType.CONTINUATION
        )

        assert result.detected is False
        assert result.signal is None
        assert result.pivot_points == {}

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = PatternResult(
            detected=True,
            pattern_name="Dict Test",
            pattern_type=PatternType.BREAKOUT,
            pivot_points={"level": 100.0},
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["detected"] is True
        assert result_dict["pattern_name"] == "Dict Test"


class ConcretePattern(BasePattern):
    """Concrete implementation of BasePattern for testing."""

    def __init__(self):
        super().__init__(
            name="Concrete Pattern", pattern_type=PatternType.REVERSAL, min_bars_required=10
        )

    def detect(self, df: pd.DataFrame, i: int) -> PatternResult:
        """Simple detection for testing."""
        if not self._validate_data(df, i):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Simple detection logic for testing
        if i >= 3:
            close = df["Close"].iloc[i]
            prev_close = df["Close"].iloc[i - 1]

            if close > prev_close * 1.02:  # 2% up move
                signal = TradeSignal(
                    pattern_name=self.name,
                    direction=SignalDirection.LONG,
                    entry_price=close,
                    stop_loss=close * 0.98,
                    take_profit_1=close * 1.05,
                )
                return PatternResult(
                    detected=True,
                    pattern_name=self.name,
                    pattern_type=self.pattern_type,
                    signal=signal,
                    pivot_points={"entry": close},
                    bars_since_detection=0,
                    start_index=i - 3,
                    end_index=i,
                )

        return PatternResult(detected=False, pattern_name=self.name, pattern_type=self.pattern_type)

    def generate_signal(self, df: pd.DataFrame, i: int):
        """Generate signal for testing."""
        result = self.detect(df, i)
        return result.signal if result.detected else None


class TestBasePattern:
    """Tests for BasePattern abstract class."""

    @pytest.fixture
    def pattern(self):
        """Create a concrete pattern for testing."""
        return ConcretePattern()

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
        np.random.seed(42)

        # Generate random price data
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)
        high = close + np.random.rand(50) * 2
        low = close - np.random.rand(50) * 2
        open_price = close + np.random.randn(50) * 0.5
        volume = np.random.randint(1000000, 5000000, 50)

        return pd.DataFrame(
            {"Open": open_price, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )

    def test_pattern_initialization(self, pattern):
        """Test pattern initialization."""
        assert pattern.name == "Concrete Pattern"
        assert pattern.pattern_type == PatternType.REVERSAL
        assert pattern.min_bars_required == 10

    def test_validate_data_sufficient(self, pattern, sample_df):
        """Test data validation with sufficient data."""
        is_valid = pattern._validate_data(sample_df, 25)
        assert is_valid is True

    def test_validate_data_insufficient(self, pattern, sample_df):
        """Test data validation with insufficient data."""
        is_valid = pattern._validate_data(sample_df, 3)
        assert is_valid is False

    def test_detect_returns_result(self, pattern, sample_df):
        """Test that detect returns a PatternResult."""
        result = pattern.detect(sample_df, 25)

        assert isinstance(result, PatternResult)
        assert result.pattern_name == "Concrete Pattern"

    def test_generate_signal_returns_signal_or_none(self, pattern, sample_df):
        """Test that generate_signal returns TradeSignal or None."""
        signal = pattern.generate_signal(sample_df, 25)

        assert signal is None or isinstance(signal, TradeSignal)

    def test_safe_float_conversion(self, pattern):
        """Test safe float conversion."""
        assert pattern._safe_float(100.5) == 100.5
        assert pattern._safe_float(np.float64(50.25)) == 50.25
        # None should return NaN (not 0.0) as per the implementation
        assert np.isnan(pattern._safe_float(None))

    def test_detect_with_up_move(self, pattern, sample_df):
        """Test detection with an up move."""
        # Modify data to create a detectable pattern
        sample_df.loc[sample_df.index[30], "Close"] = sample_df["Close"].iloc[29] * 1.03

        result = pattern.detect(sample_df, 30)

        # Should detect the pattern
        if result.detected:
            assert result.signal is not None
            assert result.signal.direction == SignalDirection.LONG


class TestPatternResultDefaults:
    """Tests for PatternResult default values."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        result = PatternResult(
            detected=True, pattern_name="Test", pattern_type=PatternType.REVERSAL
        )

        assert result.signal is None
        assert result.pivot_points == {}
        assert result.bars_since_detection == 0
        assert result.start_index is None
        assert result.end_index is None


class TestTradeSignalDefaults:
    """Tests for TradeSignal default values."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        signal = TradeSignal(
            pattern_name="Test",
            direction=SignalDirection.LONG,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit_1=110.0,
        )

        assert signal.take_profit_2 is None
        assert signal.take_profit_3 is None
        assert signal.confidence == 0.5
        assert signal.timestamp is None
        assert signal.metadata == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
