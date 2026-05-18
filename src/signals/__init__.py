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
]
