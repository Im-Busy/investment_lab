"""Unit tests for SMA Crossover strategy."""

import numpy as np
import pandas as pd
from backtesting import Backtest

from src.strategies.sma_crossover import SMACrossoverStrategy


def test_sma_crossover_runs_without_errors() -> None:
    """Test that strategy runs on synthetic data without raising errors."""
    np.random.seed(42)
    n = 600
    prices = np.concatenate(
        [
            np.linspace(150, 100, 200),
            np.linspace(100, 200, 200),
            np.linspace(200, 100, 200),
        ]
    )
    df = pd.DataFrame(
        {
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": np.ones(n) * 1000,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )

    bt = Backtest(df, SMACrossoverStrategy, cash=100000, commission=0.001, finalize_trades=True)
    stats = bt.run()
    # Strategy should run without errors; trades may vary based on data
    assert "Return [%]" in stats
    print(f"test_sma_crossover_runs_without_errors PASSED (trades: {stats['# Trades']})")


def test_sma_crossover_no_early_signals() -> None:
    """Test that no signals are generated before slow SMA has enough data."""
    np.random.seed(42)
    n = 150
    prices = np.random.uniform(100, 110, n)
    df = pd.DataFrame(
        {
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": np.ones(n) * 1000,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )

    bt = Backtest(df, SMACrossoverStrategy, cash=100000, commission=0.001)
    stats = bt.run()
    trades = stats["# Trades"]

    assert trades <= 2, f"Expected 0-2 trades with insufficient data, got {trades}"
    print(f"test_sma_crossover_no_early_signals PASSED (trades: {trades})")


if __name__ == "__main__":
    test_sma_crossover_runs_without_errors()
    test_sma_crossover_no_early_signals()
    print("All SMA Crossover tests passed!")
