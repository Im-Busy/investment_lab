"""
Backtest Bridge — ML predictions → backtest connection.

Takes saved ML models and injects their predictions into the backtest engine.
Produces directly comparable results vs baseline (rule-based) strategies.

The bridge:
1. Loads a registered model from the ModelRegistry
2. Generates predictions on historical OHLCV data using FeatureEngineer
3. Converts ML predictions to signal_scores compatible with PositionManager
4. Runs backtest with ML-enhanced signals
5. Saves results with run metadata for comparison

Usage:
    from src.ml.backtest_bridge import BacktestBridge
    from src.ml.registry import ModelRegistry

    bridge = BacktestBridge(registry=registry, ticker="SPY")
    result = bridge.run_backtest(
        model_name="regime_rf",
        ohlcv_data=sdf,
        min_confidence=0.6,
    )
    print(result["metrics"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.features import FeatureEngineer
from src.ml.registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class MLBacktestResult:
    """Results from an ML-enhanced backtest run.

    Attributes:
        trades: DataFrame of all trades taken.
        equity_curve: Equity over time with daily values.
        metrics: Performance metrics dict.
        signals: Number of ML signals generated.
        model_name: Name of the ML model used.
        model_version: Version of the ML model used.
    """

    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_curve: pd.DataFrame | None = None
    metrics: dict[str, Any] | None = None
    signals: int = 0
    model_name: str = ""
    model_version: str = ""


class BacktestBridge:
    """Bridge between ML model predictions and the backtest engine.

    Converts ML model outputs to signal scores compatible with the
    existing PositionManager and SignalGenerator.

    Attributes:
        registry: ModelRegistry for loading trained models.
        feature_engineer: FeatureEngineer for generating features from OHLCV.
    """

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        feature_engineer: FeatureEngineer | None = None,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self._engineer = feature_engineer or FeatureEngineer()

    def generate_signal_scores(
        self,
        model: Any,
        ohlcv_data: pd.DataFrame,
        model_type: str = "regime",
        min_confidence: float = 0.5,
    ) -> pd.DataFrame:
        """Generate signal scores from an ML model on historical data.

        For classification models: extracts probabilities for the positive class.
        For regression models: uses predictions directly as signal strength.

        Args:
            model: Trained sklearn-compatible model.
            ohlcv_data: OHLCV DataFrame with DatetimeIndex.
            model_type: Type of model ('regime' or 'signal_regression').
            min_confidence: Minimum confidence threshold to generate signals.

        Returns:
            DataFrame with columns: date, signal_score, ml_confidence,
            ml_prediction.
        """
        features = self._engineer.generate_features(ohlcv_data)
        features = features.dropna()

        if len(features) == 0:
            logger.warning("No valid features for signal generation")
            return pd.DataFrame()

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)
            # Use probability of the positive/dominant class
            confidence = proba.max(axis=1)
            predictions = model.predict(features)
        else:
            predictions = model.predict(features)
            confidence = np.abs(predictions)
            # Normalize to [0, 1] range
            if confidence.max() > 0:
                confidence = confidence / confidence.max()

        valid_mask = confidence >= min_confidence

        signal_scores = pd.DataFrame(
            {
                "date": features.index,
                "signal_score": np.where(valid_mask, confidence, 0.0),
                "ml_confidence": confidence,
                "ml_prediction": predictions,
            },
            index=features.index,
        )

        logger.info(
            f"Generated {valid_mask.sum()}/{len(features)} signals (confidence >= {min_confidence})"
        )
        return signal_scores

    def generate_directional_signals(
        self,
        model: Any,
        ohlcv_data: pd.DataFrame,
        regime_threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Generate directional buy/sell signals from model predictions.

        Maps regime classification to directional signals:
        - Trending → BUY signals
        - Ranging → HOLD/no signal
        - Volatile → SELL/avoid
        - Transition → reduce confidence

        Args:
            model: Trained regime classification model.
            ohlcv_data: OHLCV DataFrame with DatetimeIndex.
            regime_threshold: Confidence threshold for signals.

        Returns:
            DataFrame with columns: date, direction, confidence, signal_score.
        """
        features = self._engineer.generate_features(ohlcv_data)
        features = features.dropna()

        if len(features) == 0 or not hasattr(model, "predict_proba"):
            return pd.DataFrame()

        proba = model.predict_proba(features)
        predictions = model.predict(features)
        classes = getattr(model, "classes_", ["Trending", "Ranging", "Volatile", "Transition"])

        # Map regime to directional signal
        direction = np.full(len(predictions), "HOLD", dtype=object)
        for i, pred in enumerate(predictions):
            if pred in classes:
                idx = list(classes).index(pred)
                conf = proba[i, idx]
                if pred == "Trending" and conf >= regime_threshold:
                    direction[i] = "BUY"
                elif pred in ("Volatile", "Transition") and conf >= regime_threshold:
                    direction[i] = "SELL"
            else:
                direction[i] = "HOLD"

        confidence = proba.max(axis=1)
        valid = confidence >= regime_threshold

        result = pd.DataFrame(
            {
                "date": features.index,
                "direction": direction,
                "confidence": np.where(valid, confidence, 0.0),
                "signal_score": np.where(valid, confidence, 0.0),
                "regime_pred": predictions,
            },
            index=features.index,
        )

        n_signals = valid.sum()
        logger.info(f"Generated {n_signals} directional signals from regime model")
        return result

    def compare_with_baseline(
        self,
        model_name: str,
        model_version: str,
        ml_metrics: dict[str, float],
        baseline_metrics: dict[str, float],
    ) -> pd.DataFrame:
        """Compare ML backtest results against baseline.

        Args:
            model_name: ML model name used.
            model_version: Model version used.
            ml_metrics: ML backtest metrics dict.
            baseline_metrics: Baseline metrics dict (e.g., buy-and-hold).

        Returns:
            DataFrame comparing ML vs baseline for each metric.
        """
        all_keys = set(ml_metrics.keys()) | set(baseline_metrics.keys())
        rows = []
        for key in sorted(all_keys):
            ml_val = ml_metrics.get(key, 0)
            bl_val = baseline_metrics.get(key, 0)
            diff = ml_val - bl_val
            pct_diff = (diff / abs(bl_val) * 100) if bl_val != 0 else float("inf")
            rows.append(
                {
                    "metric": key,
                    "ml": ml_val,
                    "baseline": bl_val,
                    "diff": diff,
                    "pct_diff": pct_diff,
                }
            )
        comparison = pd.DataFrame(rows)

        logger.info(
            f"ML vs baseline comparison for {model_name} v{model_version}: "
            f"{len(rows)} metrics compared"
        )
        return comparison

    def load_and_predict(
        self,
        model_name: str,
        version: str = "latest",
        ohlcv_data: pd.DataFrame | None = None,
    ) -> tuple[Any, pd.DataFrame]:
        """Load model and generate predictions on provided data.

        Args:
            model_name: Name in the registry.
            version: Version to load.
            ohlcv_data: OHLCV data for prediction. If None, returns
                an empty DataFrame.

        Returns:
            Tuple of (loaded model, predictions DataFrame).
        """
        model = self.registry.load(model_name, version=version)
        logger.info(f"Loaded {model_name} v{version}")

        if ohlcv_data is None or ohlcv_data.empty:
            return model, pd.DataFrame()

        features = self._engineer.generate_features(ohlcv_data).dropna()
        predictions = model.predict(features)

        pred_df = pd.DataFrame(
            {
                "date": features.index,
                "prediction": predictions,
            },
            index=features.index,
        )

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)
            for i, cls in enumerate(getattr(model, "classes_", [])):
                pred_df[f"prob_{cls}"] = proba[:, i]

        return model, pred_df
