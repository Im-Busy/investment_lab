import sys

sys.path.insert(0, ".")

import pandas as pd
from src.strategies.backtest_py.runner import BacktestPyRunner
from src.strategies.backtest_py.multi_pattern_strategy_optimized import (
    MultiPatternStrategyOptimized,
)

# Load data
df = pd.read_csv("data/raw/SPY_daily.csv", index_col=0, parse_dates=True)
print(f"Data shape: {df.shape}")

# Create runner
runner = BacktestPyRunner(
    data=df,
    cash=100000,
    commission=0.001,
    exclusive_orders=True,
)

# Run with signal logging
print("Running backtest with enable_signal_log=True...")
results = runner.run(
    strategy_class=MultiPatternStrategyOptimized,
    enable_signal_log=True,
    min_confidence=0.60,
    min_confluence_count=2,
    risk_per_trade=0.02,
    max_open_positions=5,
)

# Check if signal event log exists
print(f"runner.bt exists: {runner.bt is not None}")
if runner.bt:
    print(f"runner.bt._strategy exists: {runner.bt._strategy is not None}")
    if runner.bt._strategy:
        has_log = hasattr(runner.bt._strategy, "_signal_event_log")
        print(f"Has _signal_event_log: {has_log}")
        if has_log:
            signal_log = runner.bt._strategy._signal_event_log
            print(f"signal_log is None: {signal_log is None}")
            if signal_log:
                print(f"Signal log type: {type(signal_log)}")
                print(f"Signal log length: {len(signal_log)}")
