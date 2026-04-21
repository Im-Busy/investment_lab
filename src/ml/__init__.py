"""
Machine Learning Module for Trading

Provides ML components for:
- Feature engineering from OHLCV data
- Regime classification using supervised learning
- Signal quality scoring using trained classifiers
- Unified pipeline orchestrating all ML components
- Experiment logging, purged CV, feature store, metrics, registry, backtest bridge

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
    compute_ic_decay,
    compute_hit_rate,
    compute_information_ratio,
    ic_summary,
    filter_features_by_ic,
)
from src.ml.registry import ModelRegistry
from src.ml.backtest_bridge import BacktestBridge, MLBacktestResult

from src.ml.features import FeatureEngineer
from src.ml.regime_model import RegimeClassifier, MLRegimeState
from src.ml.signal_scorer import SignalScorer, ScoredSignal
from src.ml.pipeline import MLPipeline, PipelineResult

__all__ = [
    # Part A infrastructure
    "ExperimentLogger",
    "PurgedKFold",
    "FeatureStore",
    "compute_ic",
    "compute_rank_ic",
    "compute_ic_decay",
    "compute_hit_rate",
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
    "SignalScorer",
    "ScoredSignal",
    "MLPipeline",
    "PipelineResult",
]
