"""
Backtest Module

Contains backtesting engine, performance metrics, benchmark comparison,
advanced risk metrics, and unified adapter for multiple backtesting frameworks.
"""

from .benchmark import BenchmarkComparison
from .engine import BacktestEngine
from .metrics import PerformanceMetrics
from .risk_metrics import DrawdownAnalyzer, RiskMetrics
from .adapter import (
    BacktestAdapter,
    UnifiedBacktestConfig,
    UnifiedBacktestResult,
    BacktestEngineType,
    create_backtest_adapter,
)

__all__ = [
    "BacktestEngine",
    "PerformanceMetrics",
    "BenchmarkComparison",
    "RiskMetrics",
    "DrawdownAnalyzer",
    # New adapter components
    "BacktestAdapter",
    "UnifiedBacktestConfig",
    "UnifiedBacktestResult",
    "BacktestEngineType",
    "create_backtest_adapter",
]
