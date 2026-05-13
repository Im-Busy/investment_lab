"""
Machine Learning Module for Trading

Provides ML components for:
- Feature engineering from OHLCV data
- Regime classification using supervised learning
- Signal quality scoring using trained classifiers
- Unified pipeline orchestrating all ML components
- Experiment logging, purged CV, feature store, metrics, registry, backtest bridge

Modern ML Models (2025):
- CatBoost: Gradient boosting with ordered boosting for time series
- Chronos-2: Amazon's zero-shot time series foundation model
- FinCast: Financial foundation model (pending release)
- xLSTM: Extended LSTM with exponential gating (NeurIPS 2024)

Part A Infrastructure:
    from src.ml.experiment_logger import ExperimentLogger
    from src.ml.purged_cv import PurgedKFold
    from src.ml.feature_store import FeatureStore
    from src.ml.metrics import compute_ic, compute_rank_ic, ic_summary
    from src.ml.registry import ModelRegistry
    from src.ml.backtest_bridge import BacktestBridge
"""

from src.ml.experiment_logger import ExperimentLogger
from src.ml.purged_cv import PurgedKFold
from src.ml.feature_store import FeatureStore
from src.ml.metrics import (
    compute_ic,
    compute_rank_ic,
    compute_information_ratio,
    ic_summary,
    filter_features_by_ic,
)
from src.ml.registry import ModelRegistry
from src.ml.backtest_bridge import BacktestBridge, MLBacktestResult

from src.ml.features import FeatureEngineer
from src.ml.regime_model import RegimeClassifier, MLRegimeState
from src.ml.regime_base import RegimeDetectorBase, RegimeSummary, RegimeType
from src.ml.hmm_regime import HMMRegimeDetector
from src.ml.pca_kmeans_regime import PCAKMeansRegimeDetector
from src.ml.r2_rd_regime import R2RDRegimeDetector
from src.ml.regimefolio import RegimeFolioDetector
from src.ml.macro_regime import MacroRegimeDetector
from src.ml.path_signature_regime import PathSignatureRegimeDetector
from src.ml.signal_scorer import SignalScorer, ScoredSignal
from src.ml.pipeline import MLPipeline, PipelineResult
from src.ml.models import (
    CatBoostForecaster,
    ChronosForecaster,
    FinCastForecaster,
    xLSTMForecaster,
)

# Additional components (used by scripts and tests)
from src.ml.model_selector import ModelSelector, ModelConfig, ModelResult, Recommendation
from src.ml.pattern_classifier import PatternClassifier
from src.ml.feature_engineering import FeatureExtractor, FeatureConfig
from src.ml.feature_selector import FeatureSelector, SelectionResult
from src.ml.ensemble_regime import EnsembleRegimeDetector, EnsembleStrategy
from src.ml.cross_asset_features import (
    CrossAssetFeatures,
    prepare_cross_asset_data,
    CrossAssetFeatureExtractor,
    load_market_data,
)
from src.ml.triple_barrier import TripleBarrierLabeler, BarrierLabel
from src.ml.combinatorial_purged_cv import CombinatorialPurgedCV
from src.ml.gmm_regime import GMMRegimeDetector
from src.ml.drawdown_target import DrawdownTargetGenerator
from src.ml.frac_diff import FracDiff
from src.ml.historical_analog import HistoricalAnalogMatcher, AnalogMatchSet, AnalogResult

# Phase 3 — New Models (FS1, FS8, FS13, T5, T8)
from src.ml.survival_analyzer import (
    SurvivalAnalyzer,
    SurvivalResult,
    SurvivalTarget,
    compare_survival_models,
)
from src.ml.volatility_forecaster import VolatilityForecaster, VolForecastResult
from src.ml.breakout_classifier import BreakoutClassifier, BreakoutResult
from src.ml.ensemble_models import (
    EnsembleBuilder,
    EnsembleResult,
    EnsembleMethod,
    TaskType,
    BaseModelConfig,
)
from src.ml.shap_dashboard import (
    SHAPDashboard,
    PredictionExplanation,
    GlobalImportance,
    quick_shap_analysis,
)

# Phase 4 — Advanced Pipeline (FS7, FS9, FS10, FS12)
from src.ml.change_point_regime import ChangePointRegimeDetector
from src.ml.volume_profile_cluster import VolumeProfileCluster
from src.ml.hdbscan_anomaly import HDBSCANAnomalyDetector
from src.ml.cross_symbol_cluster import CrossSymbolCluster

# Phase 6b — Pioneer Research (T9, FS19)
from src.ml.meta_labeler import MetaLabeler, MetaLabelResult, MetaLabelPrediction
from src.ml.gap_fill_predictor import (
    GapFillPredictor,
    GapFillPrediction,
    GapFillTrainingResult,
    GapDetection,
)

# Phase 4+ — Risk, Alpha Research & Ensemble (T6, QW2, RS1, RS2)
from src.ml.stop_loss_optimizer import (
    StopLossOptimizer,
    StopLossRecommendation,
    LabelGenerator,
)
from src.ml.ebm_alpha import (
    EBMAlphaPipeline,
    AlphaResult,
    AlphaFeature,
    AlphaShape,
)
from src.ml.dream_team_ensemble import (
    DreamTeamEnsemble,
    DreamTeamResult,
    DreamTeamConfig,
)

__all__ = [
    # Part A infrastructure
    "ExperimentLogger",
    "PurgedKFold",
    "FeatureStore",
    "compute_ic",
    "compute_rank_ic",
    "compute_information_ratio",
    "ic_summary",
    "filter_features_by_ic",
    "ModelRegistry",
    "BacktestBridge",
    "MLBacktestResult",
    # Existing components
    "FeatureEngineer",
    "RegimeClassifier",
    "MLRegimeState",
    "RegimeDetectorBase",
    "RegimeSummary",
    "RegimeType",
    "HMMRegimeDetector",
    "PCAKMeansRegimeDetector",
    "R2RDRegimeDetector",
    "RegimeFolioDetector",
    "MacroRegimeDetector",
    "PathSignatureRegimeDetector",
    "SignalScorer",
    "ScoredSignal",
    "MLPipeline",
    "PipelineResult",
    # New ML models (2025 stack)
    "CatBoostForecaster",
    "ChronosForecaster",
    "FinCastForecaster",
    "xLSTMForecaster",
    # Additional components (scripts/tests)
    "ModelSelector",
    "ModelConfig",
    "ModelResult",
    "Recommendation",
    "PatternClassifier",
    "FeatureExtractor",
    "FeatureConfig",
    "FeatureSelector",
    "SelectionResult",
    "EnsembleRegimeDetector",
    "EnsembleStrategy",
    "CrossAssetFeatures",
    "prepare_cross_asset_data",
    "CrossAssetFeatureExtractor",
    "load_market_data",
    # Triple barrier + combinatorial purged CV
    "TripleBarrierLabeler",
    "BarrierLabel",
    "CombinatorialPurgedCV",
    # Phase 2 — Quick Wins (FS2, FS3, FS5, FS11)
    "GMMRegimeDetector",
    "DrawdownTargetGenerator",
    "FracDiff",
    "HistoricalAnalogMatcher",
    "AnalogMatchSet",
    "AnalogResult",
    # Phase 3 — New Models (FS1, FS8, FS13, T5, T8)
    "SurvivalAnalyzer",
    "SurvivalResult",
    "SurvivalTarget",
    "compare_survival_models",
    "VolatilityForecaster",
    "VolForecastResult",
    "BreakoutClassifier",
    "BreakoutResult",
    "EnsembleBuilder",
    "EnsembleResult",
    "EnsembleMethod",
    "TaskType",
    "BaseModelConfig",
    "SHAPDashboard",
    "PredictionExplanation",
    "GlobalImportance",
    "quick_shap_analysis",
    # Phase 4 — Advanced Pipeline (FS7, FS9, FS10, FS12)
    "ChangePointRegimeDetector",
    "VolumeProfileCluster",
    "HDBSCANAnomalyDetector",
    "CrossSymbolCluster",
    # Phase 4+ — Risk, Alpha Research & Ensemble (T6, QW2, RS1, RS2)
    "StopLossOptimizer",
    "StopLossRecommendation",
    "LabelGenerator",
    "EBMAlphaPipeline",
    "AlphaResult",
    "AlphaFeature",
    "AlphaShape",
    "DreamTeamEnsemble",
    "DreamTeamResult",
    "DreamTeamConfig",
    # Phase 6b — Pioneer Research
    "MetaLabeler",
    "MetaLabelResult",
    "MetaLabelPrediction",
    "GapFillPredictor",
    "GapFillPrediction",
    "GapFillTrainingResult",
    "GapDetection",
]
