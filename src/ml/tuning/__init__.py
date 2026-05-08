"""Hyperparameter tuning and feature selection via nature-inspired metaheuristics.

ARO → GWO → GA → WOA pipeline for end-to-end OHLCV model optimization.
"""

from src.ml.tuning.base import (
    BaseOptimizer,
    OptimizerResult,
    ParamSpec,
    SearchSpace,
)
from src.ml.tuning.aro_selector import AROFeatureSelector
from src.ml.tuning.ga_tuner import GARegimeOptimizer, RegimeDiscovery, REGIME_PARAM_SPACE
from src.ml.tuning.gwo_tuner import CATBOOST_PARAM_SPACE, GWOTuner
from src.ml.tuning.woa_tuner import WOATuner, build_threshold_search_space

__all__ = [
    "BaseOptimizer",
    "OptimizerResult",
    "ParamSpec",
    "SearchSpace",
    "AROFeatureSelector",
    "GARegimeOptimizer",
    "GWOTuner",
    "WOATuner",
    "RegimeDiscovery",
    "CATBOOST_PARAM_SPACE",
    "REGIME_PARAM_SPACE",
    "build_threshold_search_space",
]
