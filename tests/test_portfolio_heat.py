# -*- coding: utf-8 -*-
"""
Tests for portfolio heat tracking.
"""

from datetime import date

import pytest

from src.risk.daily_limits import DailyLossLimiter, DailyLossState, TradingState


class TestPortfolioHeat:
    """Test portfolio heat tracking functionality."""

    def test_portfolio_heat_calculation(self):
        """Test portfolio heat calculation."""
        state = DailyLossState(
            date=date.today(),
            starting_equity=100000.0,
            current_equity=100000.0,
            peak_equity=100000.0,
        )

        # Add position risks
        state.update_position_risk("pos1", 2000.0)
        state.update_position_risk("pos2", 1500.0)

        # Total risk = 3500, equity = 100000, heat = 3.5%
        assert state.portfolio_heat == pytest.approx(0.035)

    def test_portfolio_heat_warning_level(self):
        """Test portfolio heat warning at 4%."""
        limiter = DailyLossLimiter()
        state = DailyLossState(
            date=date.today(),
            starting_equity=100000.0,
            current_equity=100000.0,
            peak_equity=100000.0,
        )

        # Add position risks to reach 4.5%
        state.update_position_risk("pos1", 4500.0)

        breached, limit_type, message = limiter.check_portfolio_heat(state)

        assert not breached  # Not breached yet, just warning
        assert state.state == TradingState.WARNING

    def test_portfolio_heat_halt_level(self):
        """Test portfolio heat halt at 6%."""
        limiter = DailyLossLimiter()
        state = DailyLossState(
            date=date.today(),
            starting_equity=100000.0,
            current_equity=100000.0,
            peak_equity=100000.0,
        )

        # Add position risks to reach 6.5%
        state.update_position_risk("pos1", 6500.0)

        breached, limit_type, message = limiter.check_portfolio_heat(state)

        assert breached
        assert "Portfolio heat limit reached" in message

    def test_remove_position_risk(self):
        """Test removing position risk."""
        state = DailyLossState(
            date=date.today(),
            starting_equity=100000.0,
            current_equity=100000.0,
            peak_equity=100000.0,
        )

        state.update_position_risk("pos1", 2000.0)
        state.update_position_risk("pos2", 1500.0)
        assert state.portfolio_heat == pytest.approx(0.035)

        # Remove pos1
        state.remove_position_risk("pos1")
        assert state.portfolio_heat == pytest.approx(0.015)

    def test_portfolio_heat_empty(self):
        """Test portfolio heat with no positions."""
        state = DailyLossState(
            date=date.today(),
            starting_equity=100000.0,
            current_equity=100000.0,
            peak_equity=100000.0,
        )

        assert state.portfolio_heat == pytest.approx(0.0)

    def test_portfolio_heat_zero_equity(self):
        """Test portfolio heat with zero equity."""
        state = DailyLossState(
            date=date.today(),
            starting_equity=0.0,
            current_equity=0.0,
            peak_equity=0.0,
        )

        state.update_position_risk("pos1", 1000.0)
        assert state.portfolio_heat == pytest.approx(0.0)
