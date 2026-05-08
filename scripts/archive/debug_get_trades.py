"""Quick debug: check _trades columns and structure."""

import pandas as pd

# Load just a small subset of data for speed
data_path = r"C:\Dev\projects\investment_trying\data\raw\SPY_daily.csv"
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
df = df.head(500)  # Use only first 500 rows to keep it fast

sys_path = r"C:\Dev\projects\investment_trying"
import sys

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from src.strategies.backtest_py.multi_pattern_strategy_optimized import (
    MultiPatternStrategyOptimized,
)
from src.strategies.backtest_py.runner import BacktestPyRunner

runner = BacktestPyRunner(
    data=df,
    cash=100000,
    commission=0.001,
    exclusive_orders=True,
)
MultiPatternStrategyOptimized.enable_signal_log = True
results = runner.run(
    strategy_class=MultiPatternStrategyOptimized,
    min_confidence=0.60,
    min_confluence_count=2,
    risk_per_trade=0.02,
    max_open_positions=5,
)

bt_results = runner.results
trades = bt_results._trades
print("Type:", type(trades))
print("Shape:", trades.shape)
print("Columns:", list(trades.columns))
print("Index:", trades.index.name if trades.index.name else "no name")
if len(trades) > 0:
    print("\nFirst row values:")
    row = trades.iloc[0]
    for col in trades.columns:
        print(f"  {col}: {row[col]} (type: {type(row[col])})")
