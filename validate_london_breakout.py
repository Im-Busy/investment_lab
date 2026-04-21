"""
Validation script for London Breakout Strategy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.strategies.london_breakout import LondonBreakoutStrategy
from backtesting import Backtest
import json


def generate_sample_fx_data(n_points: int = 10000) -> pd.DataFrame:
    """Generate sample FX minute data for validation."""
    rng = np.random.default_rng(42)

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


def run_validation() -> None:
    """Run full validation suite."""
    print("=== London Breakout Strategy Validation ===\n")

    data = generate_sample_fx_data(10000)
    print(f"Loaded {len(data)} data points\n")

    bt = Backtest(data, LondonBreakoutStrategy, cash=10000, commission=0.0001)

    try:
        stats = bt.run()
    except Exception as e:
        print(f"Backtest failed: {e}")
        return

    print("=== Backtest Results ===")
    print(f"Total Return: {stats.get('Return [%]', 0):.2f}%")
    print(f"Buy & Hold Return: {stats.get('Buy & Hold Return [%]', 0):.2f}%")
    print(f"Sharpe Ratio: {stats.get('Sharpe Ratio', 0):.2f}")
    print(f"Max Drawdown: {stats.get('Max. Drawdown [%]', 0):.2f}%")
    print(f"Win Rate: {stats.get('Win Rate [%]', 0):.2f}%")
    print(f"Total Trades: {stats.get('# Trades', 0)}\n")

    validate_results(stats)
    save_validation_results(stats)


def validate_results(stats: pd.Series) -> None:
    """Validate results meet minimum criteria."""
    print("=== Validation ===")

    trades = stats.get("# Trades", 0)
    if trades < 30:
        print(f"Warning: Only {trades} trades (expected 30+)")
    else:
        print(f"OK: Trade count: {trades} (meets 30+ minimum)")

    sharpe = stats.get("Sharpe Ratio", 0)
    if pd.isna(sharpe) or sharpe < 0.5:
        print(f"Warning: Sharpe {sharpe:.2f} (expected > 0.5)")
    else:
        print(f"OK: Sharpe ratio: {sharpe:.2f} (exceeds 0.5 threshold)")

    mdd = stats.get("Max. Drawdown [%]", 0)
    if mdd > -30:
        print(f"OK: Max drawdown: {mdd:.2f}% (within -30% limit)")
    else:
        print(f"Warning: Max drawdown {mdd:.2f}% (exceeds -30%)")


def save_validation_results(stats: pd.Series) -> None:
    """Save validation results to file."""
    results_path = Path("docs/london_breakout_results.json")

    results_data = {
        "total_return": stats.get("Return [%]", 0),
        "buy_hold_return": stats.get("Buy & Hold Return [%]", 0),
        "sharpe_ratio": stats.get("Sharpe Ratio", 0),
        "max_drawdown": stats.get("Max. Drawdown [%]", 0),
        "win_rate": stats.get("Win Rate [%]", 0),
        "total_trades": stats.get("# Trades", 0),
    }

    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    run_validation()
