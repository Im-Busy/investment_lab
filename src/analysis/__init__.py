"""Pattern Selection Improvement Pipeline.

Exports all key classes from the pattern selection analysis modules:
- StatisticalSignificanceFilter: t-tests and performance filtering
- CorrelationAnalyzer: correlation matrices and cluster deduplication
- ContributionAnalyzer: leave-one-out ablation analysis
- WalkForwardValidator: walk-forward analysis with overfitting detection
- SignalQualityFilter: signal quality scoring and filtering
- PatternSelector: orchestrator for the full selection pipeline

Phase 08 - Contribution & Attribution System (4-Layer Architecture):
- SignalEventLog: per-bar signal capture with pattern/regime metadata (Layer 1)
- TradeAttributor: P&L attribution to individual patterns (Layer 2)
- AblationEngine: leave-one-out pattern ablation studies (Layer 3)
- SynergyAnalyzer: pattern interaction effect analysis (Layer 4)
- ContributionReport: aggregated contribution report generation
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

# Phase 08 - Contribution & Attribution System
from .signal_event_log import (
    SignalEvent,
    SignalEventLog,
)

from .trade_attributor import (
    AttributedTrade,
    TradeAttributor,
)

from .ablation_engine import (
    AblationEngine,
    AblationResult,
)

from .synergy_analyzer import (
    SynergyAnalyzer,
    SynergyResult,
)

from .contribution_report import (
    ContributionReport,
)

from .contribution_charts import (
    create_contribution_charts,
)

from .deflated_sharpe import (
    PSRResult,
    FDRResult,
    compute_psr,
    compute_psr_from_returns,
    compute_dsr,
    compute_dsr_from_returns,
    benjamini_hochberg,
    deflated_sharpe_batch,
    dsr_significance,
)

from .dual_alpha_beta import (
    DualAlphaBetaResult,
    compute_dual_alpha_beta,
    rolling_dual_alpha_beta,
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
    # Phase 08 - Contribution & Attribution System
    "SignalEvent",
    "SignalEventLog",
    "AttributedTrade",
    "TradeAttributor",
    "AblationEngine",
    "AblationResult",
    "SynergyAnalyzer",
    "SynergyResult",
    "ContributionReport",
    "create_contribution_charts",
    # DSR/PSR/FDR
    "PSRResult",
    "FDRResult",
    "compute_psr",
    "compute_psr_from_returns",
    "compute_dsr",
    "compute_dsr_from_returns",
    "benjamini_hochberg",
    "deflated_sharpe_batch",
    "dsr_significance",
    # Dual Alpha/Beta
    "DualAlphaBetaResult",
    "compute_dual_alpha_beta",
    "rolling_dual_alpha_beta",
]
