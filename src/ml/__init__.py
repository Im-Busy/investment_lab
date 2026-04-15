"""
Machine Learning Module for Trading

Provides ML components for:
- Feature engineering from OHLCV data
- Regime classification using supervised learning
- Signal quality scoring using trained classifiers
- Unified pipeline orchestrating all ML components

Usage:
    from src.ml.features import FeatureEngineer
    from src.ml.regime_model import RegimeClassifier
    from src.ml.signal_scorer import SignalScorer
    from src.ml.pipeline import MLPipeline, PipelineResult
"""

from src.ml.features import FeatureEngineer
from src.ml.regime_model import RegimeClassifier, MLRegimeState
from src.ml.signal_scorer import SignalScorer, ScoredSignal
from src.ml.pipeline import MLPipeline, PipelineResult

__all__ = [
    "FeatureEngineer",
    "RegimeClassifier",
    "MLRegimeState",
    "SignalScorer",
    "ScoredSignal",
    "MLPipeline",
    "PipelineResult",
]
