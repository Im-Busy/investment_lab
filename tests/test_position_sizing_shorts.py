# -*- coding: utf-8 -*-
"""
Tests for position sizing with short positions.
"""

import pytest

from src.risk.position_sizing import PositionSizer, PositionSizeResult


class TestPositionSizingShorts:
    """Test position sizing for short positions."""

    def test_short_position_stop_validation(self):
        """Test that stop price must be above entry for short positions."""
        sizer = PositionSizer(method="fixed_fractional", risk_per_trade=0.02)

        # Valid short: stop above entry
        result = sizer.calculate(
            equity=100000,
            entry_price=100.0,
            stop_price=105.0,
            direction="short",
        )
        assert result.size > 0
        assert result.stop_price > 100.0

    def test_short_position_invalid_stop_raises(self):
        """Test that stop below entry raises error for short positions."""
        sizer = PositionSizer(method="fixed_fractional", risk_per_trade=0.02)

        with pytest.raises(ValueError, match="Stop price must be above entry price"):
            sizer.calculate(
                equity=100000,
                entry_price=100.0,
                stop_price=95.0,
                direction="short",
            )

    def test_long_position_stop_validation(self):
        """Test that stop price must be below entry for long positions."""
        sizer = PositionSizer(method="fixed_fractional", risk_per_trade=0.02)

        # Valid long: stop below entry
        result = sizer.calculate(
            equity=100000,
            entry_price=100.0,
            stop_price=95.0,
            direction="long",
        )
        assert result.size > 0

    def test_long_position_invalid_stop_raises(self):
        """Test that stop above entry raises error for long positions."""
        sizer = PositionSizer(method="fixed_fractional", risk_per_trade=0.02)

        with pytest.raises(ValueError, match="Stop price must be below entry price"):
            sizer.calculate(
                equity=100000,
                entry_price=100.0,
                stop_price=105.0,
                direction="long",
            )

    def test_short_position_atr_stop(self):
        """Test ATR-based stop calculation for short positions."""
        sizer = PositionSizer(method="fixed_fractional", risk_per_trade=0.02, atr_multiplier=2.0)

        result = sizer.calculate(
            equity=100000,
            entry_price=100.0,
            atr=2.0,
            direction="short",
        )
        # Stop should be entry + (ATR * multiplier) = 100 + 4 = 104
        assert result.stop_price == pytest.approx(104.0)

    def test_long_position_atr_stop(self):
        """Test ATR-based stop calculation for long positions."""
        sizer = PositionSizer(method="fixed_fractional", risk_per_trade=0.02, atr_multiplier=2.0)

        result = sizer.calculate(
            equity=100000,
            entry_price=100.0,
            atr=2.0,
            direction="long",
        )
        # Stop should be entry - (ATR * multiplier) = 100 - 4 = 96
        assert result.stop_price == pytest.approx(96.0)

    def test_short_position_default_stop(self):
        """Test default stop calculation for short positions."""
        sizer = PositionSizer(method="fixed_fractional", risk_per_trade=0.02)

        result = sizer.calculate(
            equity=100000,
            entry_price=100.0,
            direction="short",
        )
        # Default stop should be entry * 1.05 = 105
        assert result.stop_price == pytest.approx(105.0)

    def test_long_position_default_stop(self):
        """Test default stop calculation for long positions."""
        sizer = PositionSizer(method="fixed_fractional", risk_per_trade=0.02)

        result = sizer.calculate(
            equity=100000,
            entry_price=100.0,
            direction="long",
        )
        # Default stop should be entry * 0.95 = 95
        assert result.stop_price == pytest.approx(95.0)

    def test_short_position_risk_calculation(self):
        """Test risk calculation for short positions."""
        sizer = PositionSizer(method="fixed_fractional", risk_per_trade=0.02)

        result = sizer.calculate(
            equity=100000,
            entry_price=100.0,
            stop_price=110.0,
            direction="short",
        )
        # Risk per share = 110 - 100 = 10
        # Risk amount = 100000 * 0.02 = 2000
        # Size = 2000 / 10 = 200
        assert result.risk_amount == pytest.approx(2000.0)
        assert result.size == pytest.approx(200.0)
