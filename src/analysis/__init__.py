"""
Contribution Analysis System and Pattern Selection Pipeline

This module provides tools for analyzing pattern contributions in multi-pattern trading strategies
and selecting profitable patterns through a multi-phase filtering pipeline.

Layers of Analysis:
1. Signal Event Log (Layer 1) - Record all pattern detection events
2. Trade Attributor (Layer 2) - Match trades to contributing patterns
3. Ablation Engine (Layer 3) - Leave-one-out contribution analysis
4. Synergy Analyzer (Layer 4) - Pairwise pattern interactions

Pattern Selection Pipeline:
- Statistical Filter - Filter patterns with insufficient data
- Performance Filter - Filter unprofitable patterns
- Correlation Analyzer - Remove redundant patterns
- Walk-Forward Validator - Validate patterns out-of-sample
- Signal Quality Filter - Quality gate before confluence
- Pattern Performance Tracker - Rolling metrics dashboard
"""

from .signal_event_log import SignalEvent, SignalEventLog
from .trade_attributor import AttributedTrade, TradeAttributor
from .ablation_engine import AblationResult, AblationEngine
from .synergy_analyzer import SynergyResult, SynergyAnalyzer
from .contribution_report import ContributionReport
from .contribution_charts import create_contribution_charts
from .statistical_filter import StatisticalFilter, StatisticalFilterResult
from .performance_filter import PerformanceFilter, PerformanceFilterResult
from .correlation_analyzer import CorrelationAnalyzer, CorrelationResult, CorrelationGroup
from .walk_forward_validator import (
    WalkForwardValidator,
    WalkForwardResult,
    OverfitStatus,
    detect_overfitting,
)
from .signal_quality_filter import SignalQualityFilter, SignalQualityResult, SignalQualityConfig
from .pattern_performance_tracker import PatternPerformanceTracker, RollingMetrics

__all__ = [
    # Layer 1
    "SignalEvent",
    "SignalEventLog",
    # Layer 2
    "AttributedTrade",
    "TradeAttributor",
    # Layer 3
    "AblationResult",
    "AblationEngine",
    # Layer 4
    "SynergyResult",
    "SynergyAnalyzer",
    # Aggregation & Reporting
    "ContributionReport",
    "create_contribution_charts",
    # Pattern Selection Pipeline
    # Phase 1
    "StatisticalFilter",
    "StatisticalFilterResult",
    # Phase 2
    "PerformanceFilter",
    "PerformanceFilterResult",
    # Phase 3
    "CorrelationAnalyzer",
    "CorrelationResult",
    "CorrelationGroup",
    # Phase 4 - Covered by AblationEngine & SynergyAnalyzer
    # Phase 5
    "WalkForwardValidator",
    "WalkForwardResult",
    "OverfitStatus",
    "detect_overfitting",
    # Signal Quality Gate
    "SignalQualityFilter",
    "SignalQualityResult",
    "SignalQualityConfig",
    # Performance Tracking
    "PatternPerformanceTracker",
    "RollingMetrics",
]
