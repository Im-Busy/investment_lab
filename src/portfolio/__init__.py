"""Portfolio optimization and multi-strategy backtesting."""

from .signal_aggregator import SignalAggregator, AggregationMethod, NormalizationMethod

from .multi_strategy_engine import (
    MultiStrategyEngine,
    EqualWeightScheme,
    SharpeWeightScheme,
    InverseVolatilityWeightScheme,
    KellyWeightScheme,
)

from .strategy_ranker import (
    StrategyRanker,
    RankingMethod,
    StrategyPerformance,
    RegimeDetector,
    RegimeState,
    AdaptiveStrategySelector,
)

from .portfolio_risk import (
    PortfolioRiskManager,
    PositionSizer,
    DynamicRiskAdjuster,
    RiskLimit,
    RiskLimitType,
)

__all__ = [
    "SignalAggregator",
    "AggregationMethod",
    "NormalizationMethod",
    "MultiStrategyEngine",
    "EqualWeightScheme",
    "SharpeWeightScheme",
    "InverseVolatilityWeightScheme",
    "KellyWeightScheme",
    "StrategyRanker",
    "RankingMethod",
    "StrategyPerformance",
    "RegimeDetector",
    "RegimeState",
    "AdaptiveStrategySelector",
    "PortfolioRiskManager",
    "PositionSizer",
    "DynamicRiskAdjuster",
    "RiskLimit",
    "RiskLimitType",
]
