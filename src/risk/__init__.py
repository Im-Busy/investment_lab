# -*- coding: utf-8 -*-
"""
Risk Management Module

Provides position sizing and risk management components for trading strategies.
"""

from .position_sizing import calculate_position_size, PositionSizeResult, PositionSizer
from .daily_limits import DailyLossLimiter, DailyLossState, CircuitBreaker
from .purged_cv import PurgedTimeSeriesCV
from .turnover_penalty import TurnoverPenalty, TurnoverPenaltyConfig
from .position_probability import PositionRiskModel, PositionRiskConfig
from .dynamic_rebalancing import DynamicRebalancer, DynamicRebalanceConfig
from .diversity_score import (
    DiversityScorer,
    DiversityAdjustedSizer,
    DiversityConfig,
    DiversityMethod,
    DiversityResult,
)
from .crash_factor import (
    CrashFactorModel,
    CrashFactorFilter,
    CrashFactorConfig,
    CrashFactorResult,
    BehavioralCrashDetector,
)
from .kelly_allocator import (
    KellyAllocator,
    KellyAllocation,
    KellyEdge,
    compute_kelly_from_history,
    compute_kelly_from_probability,
    estimate_minimum_capital,
    format_kelly_report,
)
from .failure_set_analyzer import (
    FailureSetAnalyzer,
    StrategyFailureMonitor,
    FailureTestConfig,
    FailureSetDiagnosis,
)
from .circuit_breakers import (
    CircuitBreaker as PortfolioCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
)
from .mc_var import MCVaR, VaRResult
from .copula_risk import GaussianCopulaRisk, TCopulaRisk, CopulaRiskResult  # Q6
from .market_impact import MarketImpact, ImpactResult  # Q7
from .delta_hedging import (
    DeltaHedgeSimulator,
    DeltaHedgeResult,
    GammaScalper,
    PortfolioHedgeOptimizer,
)  # A11
from .vol_trading import VolTradeAnalyzer, VolTradeResult, VolArbitrageResult  # A13

__all__ = [
    # Position Sizing
    "calculate_position_size",
    "PositionSizeResult",
    "PositionSizer",
    # Daily Limits
    "DailyLossLimiter",
    "DailyLossState",
    "CircuitBreaker",
    # Cross-validation
    "PurgedTimeSeriesCV",
    # Phase 6 Tier 1 Components
    "TurnoverPenalty",
    "TurnoverPenaltyConfig",
    "PositionRiskModel",
    "PositionRiskConfig",
    "DynamicRebalancer",
    "DynamicRebalanceConfig",
    # High Priority Research Items
    "DiversityScorer",
    "DiversityAdjustedSizer",
    "DiversityConfig",
    "DiversityMethod",
    "DiversityResult",
    "CrashFactorModel",
    "CrashFactorFilter",
    "CrashFactorConfig",
    "CrashFactorResult",
    "BehavioralCrashDetector",
    "KellyAllocator",
    "KellyAllocation",
    "KellyEdge",
    "compute_kelly_from_history",
    "compute_kelly_from_probability",
    "estimate_minimum_capital",
    "format_kelly_report",
    "FailureSetAnalyzer",
    "StrategyFailureMonitor",
    "FailureTestConfig",
    "FailureSetDiagnosis",
    # Portfolio Circuit Breaker
    "PortfolioCircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    # Monte Carlo VaR / CVaR (T6)
    "MCVaR",
    "VaRResult",
    # Phase 21 Q6: Copula Risk
    "GaussianCopulaRisk",
    "TCopulaRisk",
    "CopulaRiskResult",
    # Phase 21 Q7: Market Impact
    "MarketImpact",
    "ImpactResult",
    # A11: Delta Hedging
    "DeltaHedgeSimulator",
    "DeltaHedgeResult",
    "GammaScalper",
    "PortfolioHedgeOptimizer",
    # A13: Volatility Trading Strategies
    "VolTradeAnalyzer",
    "VolTradeResult",
    "VolArbitrageResult",
]
