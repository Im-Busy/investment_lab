"""
Route predictions to per-regime ML models.

Instead of one model for all regimes, maintain N models (one per regime).
At inference, detect current regime and route to the appropriate model.

Usage:
    from src.ml.regime_router import RegimeRouter
    from src.ml.simple_regime import CombinedSimpleRegimeDetector

    detector = CombinedSimpleRegimeDetector(ma_period=200, vol_pctile=80)
    router = RegimeRouter(
        regime_detector=detector,
        model_paths={
            "Bull_Low": Path("models/regime_Bull_Low_SPY.pkl"),
            "Bull_High": Path("models/regime_Bull_High_SPY.pkl"),
            "Bear_Low": Path("models/regime_Bear_Low_SPY.pkl"),
            "Bear_High": Path("models/regime_Bear_High_SPY.pkl"),
        },
        fallback_model_path=Path("models/pattern_classifier_v3_SPY.pkl"),
    )
    probs = router.predict(features)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.ml.pattern_classifier import PatternClassifier

logger = logging.getLogger(__name__)


class RegimeRouter:
    """Route predictions to regime-specific CatBoost models.

    Each regime has its own trained model. The router:
    1. Detects current regime
    2. Routes features to the correct model
    3. Returns probability

    Args:
        regime_detector: Any RegimeDetectorBase instance (already fitted).
        model_paths: Dict mapping regime_label -> .pkl model path.
        fallback_model_path: Used when no per-regime model matches.
    """

    def __init__(
        self,
        regime_detector,
        model_paths: dict[str, Path],
        fallback_model_path: Path,
    ):
        self.detector = regime_detector
        self.model_paths = model_paths
        self.fallback_model_path = fallback_model_path
        self.models: dict[str, object] = {}
        self.fallback_model: object | None = None

        for regime_id, path in model_paths.items():
            if path.exists():
                model = PatternClassifier()
                model.load(str(path))
                self.models[regime_id] = model
            else:
                logger.warning("Model not found for regime %s: %s", regime_id, path)

        if fallback_model_path.exists():
            model = PatternClassifier()
            model.load(str(fallback_model_path))
            self.fallback_model = model

    def predict(self, features: pd.DataFrame, price_data: pd.DataFrame | None = None) -> pd.Series:
        """Predict probabilities using regime-specific models.

        Args:
            features: Feature matrix for model prediction.
            price_data: Optional OHLCV DataFrame for regime detection
                (required if detector needs Close/High/Low columns).

        Returns:
            Series of probabilities [0, 1] with same index.
        """
        # Detect regimes from price data if provided, otherwise from features
        data_for_regime = price_data if price_data is not None else features
        regimes = self.detector.predict(data_for_regime)
        regimes = regimes.loc[features.index]

        probabilities = pd.Series(0.5, index=features.index, dtype=float)

        for regime_id in regimes.unique():
            mask = regimes == regime_id
            regime_features = features.loc[mask]

            if regime_features.empty:
                continue

            model = self.models.get(regime_id, self.fallback_model)
            if model is None:
                continue

            try:
                feature_cols = getattr(model, "feature_names_", None)
                if feature_cols is not None:
                    regime_features = regime_features.reindex(columns=feature_cols, fill_value=0.0)
                result = model.predict(regime_features)
                if isinstance(result, pd.DataFrame):
                    prob_col = (
                        "probability_profitable"
                        if "probability_profitable" in result.columns
                        else result.columns[-1]
                    )
                    probabilities.loc[mask] = result[prob_col].values
                elif isinstance(result, pd.Series):
                    probabilities.loc[mask] = result.values
                elif isinstance(result, np.ndarray):
                    if result.ndim == 2 and result.shape[1] == 2:
                        probabilities.loc[mask] = result[:, 1]
                    else:
                        probabilities.loc[mask] = result.ravel()
                else:
                    probabilities.loc[mask] = float(result)
            except Exception:
                logger.warning("Prediction failed for regime %s, assigning 0.5", regime_id)

        return probabilities

    @property
    def loaded_regimes(self) -> list[str]:
        return list(self.models.keys())

    @property
    def n_models(self) -> int:
        return len(self.models)
