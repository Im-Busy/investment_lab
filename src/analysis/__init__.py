"""Pattern Selection Improvement Pipeline.

Exports all key classes from the pattern selection analysis modules plus
the Phase 25 post-backtest validation suite:
  - Monte Carlo robustness (return reshuffling, replacement, param perturbation)
  - Purged Walk-Forward Analysis (WFA with WFE, majority-pass, catastrophic veto)
  - Regime Audit (per-regime Sharpe/DSR with VIX or SVM classification)
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
    BootstrapCIResult,
    PermutationResult,
    SignificanceSummary,
    _sharpe_ratio,
    compute_psr,
    compute_psr_from_returns,
    compute_dsr,
    compute_dsr_from_returns,
    benjamini_hochberg,
    deflated_sharpe_batch,
    dsr_significance,
    bootstrap_sharpe_ci,
    permutation_test,
    format_significance_summary,
    print_significance_report,
)

from .dual_alpha_beta import (
    DualAlphaBetaResult,
    compute_dual_alpha_beta,
    rolling_dual_alpha_beta,
)

from .monte_carlo_robustness import (
    ReshuffleResult,
    PerturbResult,
    MonteCarloRobustnessReport,
    return_reshuffling,
    return_replacement,
    parameter_perturbation,
    parameter_perturbation_from_returns,
    compute_combined_robustness_score,
    run_full_robustness_check,
    format_mc_report,
)

from .purged_walk_forward import (
    PurgedWindowResult,
    PurgedWFAReport,
    PurgedWalkForwardValidator,
)

from .regime_audit import (
    MarketRegime,
    RegimeStats,
    RegimeAuditReport,
    classify_regimes_from_returns,
    compute_per_regime_metrics,
    audit_regimes,
    format_regime_report,
)

from .divergence_bits import (
    DivergenceBitsResult,
    compute_divergence_bits,
    compare_vs_benchmark,
    compare_multiple,
)

__all__ = [
    "StatisticalSignificanceFilter",
    "SignificanceResult",
    "DEFAULT_P_VALUE_THRESHOLD",
    "DEFAULT_MIN_SHARPE",
    "DEFAULT_MIN_PROFIT_FACTOR",
    "CorrelationAnalyzer",
    "DEFAULT_CORRELATION_THRESHOLD",
    "ContributionAnalyzer",
    "DEFAULT_REDUNDANCY_THRESHOLD",
    "WalkForwardValidator",
    "WindowResult",
    "ValidationReport",
    "DEFAULT_TRAIN_DAYS",
    "DEFAULT_TEST_DAYS",
    "DEFAULT_OVERFITTING_DEGRADATION",
    "SignalQualityFilter",
    "QualityMetrics",
    "QualityScore",
    "DEFAULT_MIN_WIN_RATE",
    "DEFAULT_MIN_SIGNALS",
    "PatternSelector",
    "SelectionConfig",
    "SelectionResult",
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
    "PSRResult",
    "FDRResult",
    "BootstrapCIResult",
    "PermutationResult",
    "SignificanceSummary",
    "_sharpe_ratio",
    "compute_psr",
    "compute_psr_from_returns",
    "compute_dsr",
    "compute_dsr_from_returns",
    "benjamini_hochberg",
    "deflated_sharpe_batch",
    "dsr_significance",
    "bootstrap_sharpe_ci",
    "permutation_test",
    "format_significance_summary",
    "print_significance_report",
    "DualAlphaBetaResult",
    "compute_dual_alpha_beta",
    "rolling_dual_alpha_beta",
    "ReshuffleResult",
    "PerturbResult",
    "MonteCarloRobustnessReport",
    "return_reshuffling",
    "return_replacement",
    "parameter_perturbation",
    "parameter_perturbation_from_returns",
    "compute_combined_robustness_score",
    "run_full_robustness_check",
    "format_mc_report",
    "PurgedWindowResult",
    "PurgedWFAReport",
    "PurgedWalkForwardValidator",
    "MarketRegime",
    "RegimeStats",
    "RegimeAuditReport",
    "classify_regimes_from_returns",
    "compute_per_regime_metrics",
    "audit_regimes",
    "format_regime_report",
    "DivergenceBitsResult",
    "compute_divergence_bits",
    "compare_vs_benchmark",
    "compare_multiple",
]
