# -*- coding: utf-8 -*-
"""
Tests for SMC trade management.
"""

from datetime import datetime

import pandas as pd
import pytest

from src.strategies.smc_reversal import (
    DailyState,
    SMCConfig,
    SMCReversalStrategy,
    StrategyState,
    TradeSignal,
)


class TestSMCTradeManagement:
    """Test SMC trade management functionality."""

    def test_breakeven_at_1r(self):
        """Test that stop moves to breakeven at 1R profit."""
        strategy, state = self._setup_strategy_and_state()

        # Current profit = 1R
        state.active_signal.entry_price = 100.0
        state.active_signal.stop_loss = 98.0  # 2R risk
        state.active_signal.metadata = {}

        # Create mock DataFrame with price at 102 (1R profit)
        df = pd.DataFrame({"Close": [102.0]}, index=[datetime.now()])

        strategy._manage_active_trade(df, datetime.now(), 0)

        # Stop should be moved to breakeven
        assert state.active_signal.stop_loss == pytest.approx(100.0)
        assert state.active_signal.metadata.get("be_moved") is True

    def test_scale_out_at_2r(self):
        """Test that 50% scales out at 2R profit."""
        strategy, state = self._setup_strategy_and_state()

        # Original stop was 98, entry 100, so risk_distance = 2
        # 2R profit = 100 + 2*2 = 104
        state.active_signal.entry_price = 100.0
        state.active_signal.stop_loss = 98.0  # Original stop
        state.active_signal.metadata = {}

        # Create mock DataFrame with price at 104 (2R profit)
        df = pd.DataFrame({"Close": [104.0]}, index=[datetime.now()])

        strategy._manage_active_trade(df, datetime.now(), 0)

        # Should be scaled and stop moved to breakeven
        assert state.active_signal.metadata.get("be_moved") is True
        assert state.active_signal.metadata.get("scaled") is True

    def test_close_at_final_target(self):
        """Test that position closes at 2.5R target."""
        strategy, state = self._setup_strategy_and_state()

        # Original stop was 98, entry 100, so risk_distance = 2
        # 2.5R profit = 100 + 2*2.5 = 105
        state.active_signal.entry_price = 100.0
        state.active_signal.stop_loss = 98.0  # Original stop
        state.active_signal.metadata = {}

        # Create mock DataFrame with price at 105 (2.5R profit)
        df = pd.DataFrame({"Close": [105.0]}, index=[datetime.now()])

        strategy._manage_active_trade(df, datetime.now(), 0)

        # Position should be closed
        assert state.active_signal is None
        assert state.state == StrategyState.TRACKING_ASIA

    def test_stop_loss_enforcement_long(self):
        """Test stop loss enforcement for long positions."""
        strategy, state = self._setup_strategy_and_state()

        state.active_signal.entry_price = 100.0
        state.active_signal.stop_loss = 98.0
        state.active_signal.direction = "long"
        state.active_signal.metadata = {}

        # Create mock DataFrame with price at stop
        df = pd.DataFrame({"Close": [97.0]}, index=[datetime.now()])

        strategy._manage_active_trade(df, datetime.now(), 0)

        # Position should be closed
        assert state.active_signal is None

    def test_stop_loss_enforcement_short(self):
        """Test stop loss enforcement for short positions."""
        strategy, state = self._setup_strategy_and_state()

        state.active_signal.entry_price = 100.0
        state.active_signal.stop_loss = 102.0
        state.active_signal.direction = "short"
        state.active_signal.metadata = {}

        # Create mock DataFrame with price at stop
        df = pd.DataFrame({"Close": [103.0]}, index=[datetime.now()])

        strategy._manage_active_trade(df, datetime.now(), 0)

        # Position should be closed
        assert state.active_signal is None

    def _setup_strategy_and_state(self):
        """Setup strategy and state for testing."""
        config = SMCConfig(risk_per_trade=0.01)
        strategy = SMCReversalStrategy(config)

        state = DailyState(
            date=datetime.now().date(),
            state=StrategyState.IN_TRADE,
        )

        signal = TradeSignal(
            timestamp=datetime.now(),
            direction="long",
            entry_price=100.0,
            stop_loss=98.0,
            target_1=102.0,
            target_2=104.0,
            target_final=105.0,
            position_size=100.0,
            risk_amount=200.0,
            asia_high=101.0,
            asia_low=99.0,
            sweep_price=98.5,
            ifvg_zone=(99.0, 101.0),
            confidence=0.7,
            metadata={},
        )
        state.active_signal = signal
        strategy.state = state

        return strategy, state
