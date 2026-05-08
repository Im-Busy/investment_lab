import sys

sys.path.insert(0, ".")

import pandas as pd
from src.strategies.backtest_py.multi_pattern_strategy_optimized import (
    MultiPatternStrategyOptimized,
)

# Test if signal event log is created when enable_signal_log=True
print("Testing signal event log initialization...")


# Create a mock strategy instance
class MockData:
    def __init__(self):
        self.Close = [100, 101, 102]
        self.Open = [99, 100, 101]
        self.High = [102, 103, 104]
        self.Low = [98, 99, 100]
        self.Volume = [1000, 1100, 1200]
        self.index = pd.date_range("2020-01-01", periods=3)


# Test with enable_signal_log=False (default)
print("\n1. Testing with enable_signal_log=False (default):")
strategy_no_log = MultiPatternStrategyOptimized()
strategy_no_log.data = MockData()
strategy_no_log.enable_signal_log = False
strategy_no_log.init()
print(f"   Has _signal_event_log: {hasattr(strategy_no_log, '_signal_event_log')}")
if hasattr(strategy_no_log, "_signal_event_log"):
    print(f"   _signal_event_log is None: {strategy_no_log._signal_event_log is None}")

# Test with enable_signal_log=True
print("\n2. Testing with enable_signal_log=True:")
strategy_with_log = MultiPatternStrategyOptimized()
strategy_with_log.data = MockData()
strategy_with_log.enable_signal_log = True
strategy_with_log.init()
print(f"   Has _signal_event_log: {hasattr(strategy_with_log, '_signal_event_log')}")
if hasattr(strategy_with_log, "_signal_event_log"):
    print(f"   _signal_event_log is None: {strategy_with_log._signal_event_log is None}")
    if strategy_with_log._signal_event_log:
        print(f"   _signal_event_log type: {type(strategy_with_log._signal_event_log)}")

print("\nTest completed!")
