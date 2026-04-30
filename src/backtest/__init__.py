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

# Friction scoring (optional - used by scripts)
from .friction_scoring import (
    FrictionConfig,
    FrictionCosts,
    FrictionSummary,
    FrictionScorer,
    TurnoverBudgetEnforcer,
)

# VectorBT adapter (optional - requires vectorbt package)
try:
    from .vectorbt_adapter import VectorBTAdapter, VectorBTConfig, warmup_vectorbt

    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False

# Vectorized alternative engine (used by scripts)
from .vectorbt_alternative import VectorizedPortfolioEngine, VectorizedConfig

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
    # Friction scoring
    "FrictionConfig",
    "FrictionCosts",
    "FrictionSummary",
    "FrictionScorer",
    "TurnoverBudgetEnforcer",
    # VectorBT adapter
    "VectorBTAdapter",
    "VectorBTConfig",
    "warmup_vectorbt",
    "VECTORBT_AVAILABLE",
    # Vectorized alternative engine
    "VectorizedPortfolioEngine",
    "VectorizedConfig",
]
