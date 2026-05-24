"""
Signals Module

Contains signal generation and position management components.
"""

from .signal_generator import SignalGenerator
from .position_manager import PositionManager
from .event_weighting import (
    EventWeightedAggregator,
    EventWeightedSignal,
    EventType,
    EventTypeInfo,
    PatternEventMapping,
    EVENT_TYPE_WEIGHTS,
    DEFAULT_PATTERN_EVENT_MAPPING,
)
from .pattern_boost import PatternBoostFilter, PatternBoost, PATTERN_RELIABILITY
from .ir_weighting import IRWeighting, ir_from_pattern_signals
from .factor_purification import FactorPurifier, PurificationReport, purify_pattern_signals
from .evaluation_gate import (
    PatternEvaluationGate,
    EvaluationReport,
    FactorType,
    GateResult,
    GateStepResult,
)
from .scoring import MultiAxisScorer, MultiAxisScore, quick_5axis_report
from .collinearity import (
    analyze_collinearity,
    analyze_pattern_overlap,
    CollinearityReport,
    RedundancyPair,
    get_discard_recommendations,
    get_synthesize_recommendations,
)
from .classification import (
    FactorClassifier,
    FactorClassification,
    FactorClass,
    BulkClassification,
    classify_patterns,
    get_entry_patterns,
    get_risk_patterns,
)
from .sentiment_scorer import (
    SentimentProvider,
    CSVSentimentProvider,
    SyntheticSentimentProvider,
    SentimentSignalModifier,
)
from .event_detector import (
    FinancialSpikeDetector,
    EventPatternDetector,
    EventSignalSuppressor,
)
from .sentiment.dictionary import (
    LMDictionary,
    LMSentimentScorer,
    LMTradingSignalModifier,
)
from .pattern_quality_registry import PatternQualityRegistry, PatternQualityEntry
from .fundamental_scorer import FundamentalScorer, SectorFundamentalScore, SECTOR_MODELS
from .vix_regime_gate import VixRegimeGate
from .yield_curve_gate import YieldCurveGate
from .options_sentiment import OptionsSentimentProvider
from .order_book_features import OrderBookFeatures, OrderBookSignal
from .treasury_auctions import TreasuryAuctionCalendar, AuctionDay  # B8
from .bond_etf_proxy import (
    BondETFProxy,
    TLT_PROXY,
    IEF_PROXY,
    SHY_PROXY,
    LQD_PROXY,
    rate_sensitivity,
)  # B9
from .real_yield_analysis import RealYieldAnalyzer, RealYieldSnapshot  # B10
from .combined_trend import (
    TrendConfirmation,
    four_indicator_trend,
    trend_multiplication_factor,
)  # P24-15
from .signal_alignment import (
    AlignmentResult,
    check_signal_alignment,
    compute_combined_fundamental_bias,
)  # P24-16
from .event_type_trading import (
    EventCategory,
    EventConfig,
    EVENT_CONFIGS,
    get_event_config,
    event_aware_entry,
    classify_event,
)  # P24-20
from .event_weighted_sentiment import (
    WeightedSentimentEvent,
    compute_time_weighted_sentiment,
    decay_fn,
    batch_weight_events,
)  # P24-21
from .indicator_voting import compute_voting_signals, compute_voting_signals_df  # B6
from .cross_currency_signals import compute_cross_currency_signals, implied_signal_scalar  # B18
from .divergence_detector import detect_divergences, detect_all_divergences  # B10
from .rules_catalog import generate_rules_catalog  # B1
from .fuzzy_system import (
    FuzzyInferenceSystem,
    FuzzyResult,
    FuzzyState,
    FuzzyRule,
    FuzzyVariable,
    TrapezoidMF,
)  # B32

__all__ = [
    "SignalGenerator",
    "PositionManager",
    "EventWeightedAggregator",
    "EventWeightedSignal",
    "EventType",
    "EventTypeInfo",
    "PatternEventMapping",
    "EVENT_TYPE_WEIGHTS",
    "DEFAULT_PATTERN_EVENT_MAPPING",
    "PatternBoostFilter",
    "PatternBoost",
    "PATTERN_RELIABILITY",
    "SentimentProvider",
    "CSVSentimentProvider",
    "SyntheticSentimentProvider",
    "SentimentSignalModifier",
    "FinancialSpikeDetector",
    "EventPatternDetector",
    "EventSignalSuppressor",
    "LMDictionary",
    "LMSentimentScorer",
    "LMTradingSignalModifier",
    "IRWeighting",
    "ir_from_pattern_signals",
    "FactorPurifier",
    "PurificationReport",
    "purify_pattern_signals",
    "PatternEvaluationGate",
    "EvaluationReport",
    "FactorType",
    "GateResult",
    "GateStepResult",
    "MultiAxisScorer",
    "MultiAxisScore",
    "quick_5axis_report",
    "analyze_collinearity",
    "analyze_pattern_overlap",
    "CollinearityReport",
    "RedundancyPair",
    "get_discard_recommendations",
    "get_synthesize_recommendations",
    "FactorClassifier",
    "FactorClassification",
    "FactorClass",
    "BulkClassification",
    "classify_patterns",
    "get_entry_patterns",
    "get_risk_patterns",
    "PatternQualityRegistry",
    "PatternQualityEntry",
    "FundamentalScorer",
    "SectorFundamentalScore",
    "SECTOR_MODELS",
    "VixRegimeGate",
    "YieldCurveGate",
    "OptionsSentimentProvider",
    "OrderBookFeatures",
    "OrderBookSignal",
    # B8: Treasury Auctions
    "TreasuryAuctionCalendar",
    "AuctionDay",
    # B9: Bond ETF Proxy
    "BondETFProxy",
    "TLT_PROXY",
    "IEF_PROXY",
    "SHY_PROXY",
    "LQD_PROXY",
    "rate_sensitivity",
    # B10: Real Yield Analysis
    "RealYieldAnalyzer",
    "RealYieldSnapshot",
    # P24-15: Combined 4-index Trend
    "TrendConfirmation",
    "four_indicator_trend",
    "trend_multiplication_factor",
    # P24-16: Signal Alignment
    "AlignmentResult",
    "check_signal_alignment",
    "compute_combined_fundamental_bias",
    # P24-20: Event-Type Trading
    "EventCategory",
    "EventConfig",
    "EVENT_CONFIGS",
    "get_event_config",
    "event_aware_entry",
    "classify_event",
    # P24-21: Event-Weighted Sentiment
    "WeightedSentimentEvent",
    "compute_time_weighted_sentiment",
    "decay_fn",
    "batch_weight_events",
    # B6: 6-Indicator Voting
    "compute_voting_signals",
    "compute_voting_signals_df",
    # B18: Cross-Currency Signals
    "compute_cross_currency_signals",
    "implied_signal_scalar",
    # B10: Divergence Detection
    "detect_divergences",
    "detect_all_divergences",
    # B1: 35-Rule Catalog
    "generate_rules_catalog",
    # B32: Fuzzy Rule System
    "FuzzyInferenceSystem",
    "FuzzyResult",
    "FuzzyState",
    "FuzzyRule",
    "FuzzyVariable",
    "TrapezoidMF",
]
