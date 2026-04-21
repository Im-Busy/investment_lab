"""Debug trades extraction with minimal data."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from src.strategies.backtest_py.multi_pattern_strategy_optimized import (
    MultiPatternStrategyOptimized,
)
from src.strategies.backtest_py.runner import BacktestPyRunner

data_path = project_root / "data" / "raw" / "SPY_daily.csv"
df = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Only first 1000 rows
df = df.head(1000)
print(f"Data shape: {df.shape}")

runner = BacktestPyRunner(data=df, cash=100000, commission=0.001, exclusive_orders=True)
MultiPatternStrategyOptimized.enable_signal_log = True
results = runner.run(
    strategy_class=MultiPatternStrategyOptimized,
    min_confidence=0.60,
    min_confluence_count=2,
    risk_per_trade=0.02,
    max_open_positions=5,
)

print(f"\nResults['stats']['# Trades']: {results['stats']['# Trades']}")
print(f"runner.results._trades shape: {runner.results._trades.shape}")
print(f"runner.get_trades() shape: {runner.get_trades().shape}")
