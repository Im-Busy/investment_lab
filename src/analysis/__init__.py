"""Pattern Selection Improvement Pipeline.

Exports all key classes from the pattern selection analysis modules:
- StatisticalSignificanceFilter: t-tests and performance filtering
- CorrelationAnalyzer: correlation matrices and cluster deduplication
- ContributionAnalyzer: leave-one-out ablation analysis
- WalkForwardValidator: walk-forward analysis with overfitting detection
- SignalQualityFilter: signal quality scoring and filtering
- PatternSelector: orchestrator for the full selection pipeline
"""

from .statistical_filter import (
    DEFAULT_MIN_PROFIT_FACTOR,
    DEFAULT_MIN_SHARPE,
    DEFAULT_P_VALUE_THRESHOLD,
    SignificanceResult,
    StatisticalSignificanceFilter,
)

from .correlation_analyzer import (
    DEFAULT_CORRELATION_THRESHOLD,
    CorrelationAnalyzer,
)

from .contribution_analyzer import (
    DEFAULT_REDUNDANCY_THRESHOLD,
    AblationResult,
    ContributionAnalyzer,
)

from .walk_forward_validator import (
    DEFAULT_OVERFITTING_DEGRADATION,
    DEFAULT_TEST_DAYS,
    DEFAULT_TRAIN_DAYS,
    ValidationReport,
    WalkForwardValidator,
    WindowResult,
)

from .signal_quality_filter import (
    DEFAULT_MIN_SIGNALS,
    DEFAULT_MIN_WIN_RATE,
    QualityMetrics,
    QualityScore,
    SignalQualityFilter,
)

from .pattern_selector import (
    PatternSelector,
    SelectionConfig,
    SelectionResult,
)

__all__ = [
    # Statistical Significance Filter
    "StatisticalSignificanceFilter",
    "SignificanceResult",
    "DEFAULT_P_VALUE_THRESHOLD",
    "DEFAULT_MIN_SHARPE",
    "DEFAULT_MIN_PROFIT_FACTOR",
    # Correlation Analyzer
    "CorrelationAnalyzer",
    "DEFAULT_CORRELATION_THRESHOLD",
    # Contribution Analyzer
    "ContributionAnalyzer",
    "AblationResult",
    "DEFAULT_REDUNDANCY_THRESHOLD",
    # Walk-Forward Validator
    "WalkForwardValidator",
    "WindowResult",
    "ValidationReport",
    "DEFAULT_TRAIN_DAYS",
    "DEFAULT_TEST_DAYS",
    "DEFAULT_OVERFITTING_DEGRADATION",
    # Signal Quality Filter
    "SignalQualityFilter",
    "QualityMetrics",
    "QualityScore",
    "DEFAULT_MIN_WIN_RATE",
    "DEFAULT_MIN_SIGNALS",
    # Pattern Selector Orchestrator
    "PatternSelector",
    "SelectionConfig",
    "SelectionResult",
]
