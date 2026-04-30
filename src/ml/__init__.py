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
from src.ml.cross_asset_features import CrossAssetFeatures, prepare_cross_asset_data

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
]
