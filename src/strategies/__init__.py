"""
Strategies Module

Contains trading strategy components including confluence scoring,
strategy orchestration, SMC/ICT reversal strategy, influencer framework,
and backtesting.py integration.
"""

from .confluence import ConfluenceScorer, ConfluenceScore, PatternCompatibility
from .smc_reversal import (
    SMCReversalStrategy,
    SMCConfig,
    TradeSignal,
    TradeDirection,
    StrategyState,
    DailyState,
)

# Influencer Framework (new modular system)
from .multi_timeframe_bias import (
    MultiTimeframeBiasDetector,
    MTFConfig,
    TimeframeBias,
    BiasState,
    TrendDirection,
    MultiTimeframeSignal,
)
from .volume_confirmation import (
    VolumeConfirmation,
    VolumeConfig,
    VolumeSignal,
    VolumeSignalType,
)
from .vwap_sma_confluence import (
    VWAPConfluenceAnalyzer,
    VWAPConfig,
    VWAPConfluence,
    VWAPPosition,
    SMAPosition,
)
from .influencer_confluence import (
    InfluencerConfluenceScorer,
    InfluencerConfig,
    ConfluenceSignal,
    ConfluenceLevel,
)
from .strategy_registry import (
    StrategyRegistry,
    StrategyPlugin,
    StrategySignal,
    StrategyType,
    SignalQuality,
    PluginConfig,
    SMCReversalPlugin,
    InfluencerMTFPlugin,
    ConfluenceAggregatorPlugin,
)

# backtesting.py integration (optional - requires backtesting package)
try:
    from .backtest_py import MultiPatternStrategy, BacktestPyRunner

    BACKTEST_PY_AVAILABLE = True
except ImportError:
    BACKTEST_PY_AVAILABLE = False

# London Breakout Strategy (backtesting.py)
try:
    from .london_breakout import LondonBreakoutStrategy

    LONDON_BREAKOUT_AVAILABLE = True
except ImportError:
    LondonBreakoutStrategy = None
    LONDON_BREAKOUT_AVAILABLE = False

__all__ = [
    # Confluence
    "ConfluenceScorer",
    "ConfluenceScore",
    "PatternCompatibility",
    # SMC Strategy
    "SMCReversalStrategy",
    "SMCConfig",
    "TradeSignal",
    "TradeDirection",
    "StrategyState",
    "DailyState",
    # Multi-Timeframe Bias (Influencer Framework)
    "MultiTimeframeBiasDetector",
    "MTFConfig",
    "TimeframeBias",
    "BiasState",
    "TrendDirection",
    "MultiTimeframeSignal",
    # Volume Confirmation
    "VolumeConfirmation",
    "VolumeConfig",
    "VolumeSignal",
    "VolumeSignalType",
    # VWAP + SMA Confluence
    "VWAPConfluenceAnalyzer",
    "VWAPConfig",
    "VWAPConfluence",
    "VWAPPosition",
    "SMAPosition",
    # Influencer Confluence Scorer
    "InfluencerConfluenceScorer",
    "InfluencerConfig",
    "ConfluenceSignal",
    "ConfluenceLevel",
    # Strategy Registry (Plugin System)
    "StrategyRegistry",
    "StrategyPlugin",
    "StrategySignal",
    "StrategyType",
    "SignalQuality",
    "PluginConfig",
    "SMCReversalPlugin",
    "InfluencerMTFPlugin",
    "ConfluenceAggregatorPlugin",
    # backtesting.py integration
    "MultiPatternStrategy",
    "BacktestPyRunner",
    "BACKTEST_PY_AVAILABLE",
    # London Breakout Strategy
    "LondonBreakoutStrategy",
    "LONDON_BREAKOUT_AVAILABLE",
]
