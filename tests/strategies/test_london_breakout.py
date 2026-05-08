"""
Tests for London Breakout Strategy.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def strategy_params():
    """Default strategy parameters."""
    return {
        "tokyo_start_hour": 2,
        "tokyo_end_hour": 3,
        "london_trading_minutes": 30,
        "london_close_hour": 12,
        "risky_stop": 0.01,
        "param": 0.5,
    }


@pytest.fixture
def sample_fx_data():
    """Generate sample FX minute data for testing."""
    rng = np.random.default_rng(42)
    n_points = 10000

    dates = pd.date_range(start="2023-01-01 00:00:00", periods=n_points, freq="1min", tz="UTC")

    base_price = 1.2500
    volatility = 0.0002

    prices = np.cumsum(rng.normal(0, volatility, n_points)) + base_price

    data = pd.DataFrame(
        {
            "Open": prices + rng.normal(0, 0.0001, n_points),
            "High": prices + np.abs(rng.normal(0, 0.0003, n_points)),
            "Low": prices - np.abs(rng.normal(0, 0.0003, n_points)),
            "Close": prices,
            "Volume": rng.integers(100, 1000, n_points),
        },
        index=dates,
    )

    return data


class TestLondonBreakoutIntegration:
    """Integration tests for London Breakout Strategy."""

    def test_strategy_runs(self, sample_fx_data, strategy_params) -> None:
        """Test strategy runs without errors."""
        from src.strategies.london_breakout import LondonBreakoutStrategy
        from backtesting import Backtest

        bt = Backtest(sample_fx_data, LondonBreakoutStrategy, cash=10000)

        try:
            bt.run()
        except Exception as e:
            pytest.fail(f"Strategy failed to run: {e}")

    def test_strategy_has_parameters(self, strategy_params) -> None:
        """Test strategy has correct parameters."""
        from src.strategies.london_breakout import LondonBreakoutStrategy

        assert hasattr(LondonBreakoutStrategy, "tokyo_start_hour")
        assert hasattr(LondonBreakoutStrategy, "tokyo_end_hour")
        assert hasattr(LondonBreakoutStrategy, "london_trading_minutes")
        assert hasattr(LondonBreakoutStrategy, "london_close_hour")
        assert hasattr(LondonBreakoutStrategy, "risky_stop")
        assert hasattr(LondonBreakoutStrategy, "param")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
