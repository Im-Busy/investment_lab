"""
Strategies Module

Contains trading strategy components including confluence scoring,
strategy orchestration, SMC/ICT reversal strategy, influencer framework,
and backtesting.py integration.
"""

from .confluence import ConfluenceScorer, ConfluenceScore, PatternCompatibility
from .smc_strategy import SMCStrategy

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
    SMCPlugin,
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

# Instrument-Specific Alpha Strategies (backtesting.py)
try:
    from .gap_fill_strategy import GapFillStrategy
    from .bb_squeeze_strategy import BBSqueezeStrategy
    from .rsi_oversold_strategy import RSIOversoldStrategy
    from .dip_buy_strategy import DipBuyStrategy
    from .lowvol_momentum_strategy import LowVolMomentumStrategy
    from .sector_oversold_strategy import SectorOversoldStrategy
    from .gold_trend_strategy import GoldTrendStrategy
    from .silver_breakout_strategy import SilverBreakoutStrategy
    from .bond_fade_strategy import BondFadeStrategy
    from .crypto_momentum_strategy import CryptoMomentumStrategy

    ALPHA_STRATEGIES_AVAILABLE = True
except ImportError:
    GapFillStrategy = None
    BBSqueezeStrategy = None
    RSIOversoldStrategy = None
    DipBuyStrategy = None
    LowVolMomentumStrategy = None
    SectorOversoldStrategy = None
    GoldTrendStrategy = None
    SilverBreakoutStrategy = None
    BondFadeStrategy = None
    CryptoMomentumStrategy = None
    ALPHA_STRATEGIES_AVAILABLE = False

__all__ = [
    # Confluence
    "ConfluenceScorer",
    "ConfluenceScore",
    "PatternCompatibility",
    # SMC Strategy
    "SMCStrategy",
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
    "SMCPlugin",
    "InfluencerMTFPlugin",
    "ConfluenceAggregatorPlugin",
    # backtesting.py integration
    "MultiPatternStrategy",
    "BacktestPyRunner",
    "BACKTEST_PY_AVAILABLE",
    # London Breakout Strategy
    "LondonBreakoutStrategy",
    "LONDON_BREAKOUT_AVAILABLE",
    # Instrument-Specific Alpha Strategies
    "GapFillStrategy",
    "BBSqueezeStrategy",
    "RSIOversoldStrategy",
    "DipBuyStrategy",
    "LowVolMomentumStrategy",
    "SectorOversoldStrategy",
    "GoldTrendStrategy",
    "SilverBreakoutStrategy",
    "BondFadeStrategy",
    "CryptoMomentumStrategy",
    "ALPHA_STRATEGIES_AVAILABLE",
]
