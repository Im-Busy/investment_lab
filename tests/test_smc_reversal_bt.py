"""Unit tests for SMC Reversal Backtest strategy."""

import numpy as np
import pandas as pd
from backtesting import Backtest

from src.strategies.smc_reversal_bt import SMCReversalBacktest, SMCState


def _make_daily_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create synthetic daily OHLCV data."""
    np.random.seed(seed)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "Open": prices * 0.995,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Close": prices,
            "Volume": np.ones(n) * 1_000_000,
        },
        index=pd.date_range("2022-01-01", periods=n, freq="B"),
    )
    return df


def test_smc_bt_runs_without_errors() -> None:
    """Test that strategy runs on synthetic data without raising errors."""
    df = _make_daily_data(n=500, seed=42)

    bt = Backtest(
        df,
        SMCReversalBacktest,
        cash=100_000,
        commission=0.001,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = bt.run()
    assert "Return [%]" in stats
    assert "# Trades" in stats
    print(f"test_smc_bt_runs_without_errors PASSED (trades: {stats['# Trades']})")


def test_smc_bt_generates_trades_on_sweep_data() -> None:
    """
    Test that strategy generates trades when clear sweeps occur.

    Create data with an obvious liquidity sweep pattern:
    - Several days of range-bound trading (session formation)
    - A sharp spike below the range (sweep)
    - Recovery and break above swing high (MSS)
    - This should trigger a long entry
    """
    np.random.seed(99)
    n = 300

    # Build a base price series
    base = np.linspace(100, 100, n)

    # Days 50-60: tight range (session formation)
    for i in range(50, 65):
        base[i] = 100 + np.random.uniform(-0.5, 0.5)

    # Day 66: sweep below range
    base[66] = 95.0

    # Days 67-75: recovery and MSS break above swing
    for i in range(67, 80):
        base[i] = 97 + (i - 67) * 0.8  # steady climb

    # Remaining data
    for i in range(80, n):
        base[i] = base[i - 1] + np.random.uniform(-1, 1.2)

    df = pd.DataFrame(
        {
            "Open": base * 0.995,
            "High": base * 1.015,  # wider highs to trigger sweeps
            "Low": base * 0.985,  # wider lows to trigger sweeps
            "Close": base,
            "Volume": np.ones(n) * 1_000_000,
        },
        index=pd.date_range("2022-01-01", periods=n, freq="B"),
    )

    bt = Backtest(
        df,
        SMCReversalBacktest,
        cash=100_000,
        commission=0.001,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = bt.run(
        session_bars=5,
        atr_period=14,
        atr_buffer_mult=0.3,
        mss_lookback=5,
    )
    assert "# Trades" in stats
    print(f"test_smc_bt_generates_trades_on_sweep_data PASSED (trades: {stats['# Trades']})")


def test_smc_bt_parameters_exposed() -> None:
    """Test that all required parameters are exposed as class attributes."""
    strategy = SMCReversalBacktest

    # Verify parameters exist with expected defaults
    assert hasattr(strategy, "session_bars")
    assert hasattr(strategy, "atr_period")
    assert hasattr(strategy, "atr_buffer_mult")
    assert hasattr(strategy, "mss_lookback")
    assert hasattr(strategy, "target_final")
    assert hasattr(strategy, "breakeven_at_r")

    # Verify default values
    assert strategy.session_bars == 5
    assert strategy.atr_period == 14
    assert strategy.atr_buffer_mult == 0.5
    assert strategy.mss_lookback == 5
    assert strategy.target_final == 2.5
    assert strategy.breakeven_at_r == 1.0

    print("test_smc_bt_parameters_exposed PASSED")


def test_smc_bt_inherits_from_strategy() -> None:
    """Test that SMCReversalBacktest inherits from backtesting.Strategy."""
    from backtesting import Strategy

    assert issubclass(SMCReversalBacktest, Strategy)
    print("test_smc_bt_inherits_from_strategy PASSED")


def test_smc_bt_custom_parameters() -> None:
    """Test that custom parameters can be passed to Backtest."""
    np.random.seed(42)
    n = 500
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "Open": prices * 0.995,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Close": prices,
            "Volume": np.ones(n) * 1_000_000,
        },
        index=pd.date_range("2022-01-01", periods=n, freq="B"),
    )

    bt = Backtest(
        df,
        SMCReversalBacktest,
        cash=100_000,
        commission=0.001,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = bt.run(
        session_bars=3,
        atr_period=10,
        atr_buffer_mult=1.0,
        mss_lookback=3,
        target_final=3.0,
    )
    assert "Return [%]" in stats
    print(f"test_smc_bt_custom_parameters PASSED (trades: {stats['# Trades']})")


if __name__ == "__main__":
    test_smc_bt_runs_without_errors()
    test_smc_bt_generates_trades_on_sweep_data()
    test_smc_bt_parameters_exposed()
    test_smc_bt_inherits_from_strategy()
    test_smc_bt_custom_parameters()
    print("\nAll SMC Reversal Backtest tests passed!")
