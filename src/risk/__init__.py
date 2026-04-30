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
from .diversity_score import DiversityScorer, DiversityAdjustedSizer, DiversityConfig, DiversityMethod, DiversityResult
from .crash_factor import CrashFactorModel, CrashFactorFilter, CrashFactorConfig, CrashFactorResult
from .failure_set_analyzer import FailureSetAnalyzer, StrategyFailureMonitor, FailureTestConfig, FailureSetDiagnosis
from .circuit_breakers import (
    CircuitBreaker as PortfolioCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
)

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
    "FailureSetAnalyzer",
    "StrategyFailureMonitor",
    "FailureTestConfig",
    "FailureSetDiagnosis",
    # Portfolio Circuit Breaker
    "PortfolioCircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
]
