"""
Backtest Module

Contains backtesting engine, performance metrics, benchmark comparison,
advanced risk metrics, and unified adapter for multiple backtesting frameworks.
"""

from .adapter import (
    BacktestAdapter,
    BacktestEngineType,
    UnifiedBacktestConfig,
    UnifiedBacktestResult,
    create_backtest_adapter,
)
from .benchmark import BenchmarkComparison
from .engine import BacktestConfig, BacktestEngine
from .metrics import PerformanceMetrics
from .risk_metrics import DrawdownAnalyzer, RiskMetrics

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
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
