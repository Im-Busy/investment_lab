"""
backtesting.py Integration Module

Provides strategy wrappers and runners for using backtesting.py framework
with the multi-pattern confluence strategy.
"""

from .multi_pattern_strategy import MultiPatternStrategy, MultiPatternStrategySimple
from .multi_pattern_strategy_optimized import (
    MultiPatternStrategyFast,
    MultiPatternStrategyOptimized,
)
from .runner import BacktestPyRunner

__all__ = [
    "MultiPatternStrategy",
    "MultiPatternStrategySimple",
    "MultiPatternStrategyOptimized",
    "MultiPatternStrategyFast",
    "BacktestPyRunner",
]
