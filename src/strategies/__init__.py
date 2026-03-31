"""
Strategies Module

Contains trading strategy components including confluence scoring,
strategy orchestration, SMC/ICT reversal strategy, and backtesting.py integration.
"""

from .confluence import ConfluenceScorer, ConfluenceScore, PatternCompatibility
from .smc_reversal import (
    SMCReversalStrategy,
    SMCConfig,
    TradeSignal,
    TradeDirection,
    StrategyState,
    DailyState
)

# backtesting.py integration (optional - requires backtesting package)
try:
    from .backtest_py import MultiPatternStrategy, BacktestPyRunner
    BACKTEST_PY_AVAILABLE = True
except ImportError:
    BACKTEST_PY_AVAILABLE = False

__all__ = [
    # Confluence
    'ConfluenceScorer',
    'ConfluenceScore',
    'PatternCompatibility',
    # SMC Strategy
    'SMCReversalStrategy',
    'SMCConfig',
    'TradeSignal',
    'TradeDirection',
    'StrategyState',
    'DailyState',
    # backtesting.py integration
    'MultiPatternStrategy',
    'BacktestPyRunner',
    'BACKTEST_PY_AVAILABLE',
]
