"""
ML-Enhanced Trading Pipeline

Orchestrates the ML components for regime detection and signal scoring,
integrating with the existing confluence scoring system.

Usage:
    from src.ml.pipeline import MLPipeline

    pipeline = MLPipeline()
    pipeline.fit(df, regime_labels, signal_labels)
    regime_prediction = pipeline.predict_regime(df)
    signal_scores = pipeline.score_signals(df)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.ml.features import FeatureEngineer
from src.ml.regime_model import RegimeClassifier
from src.ml.signal_scorer import SignalScorer


@dataclass
class PipelineResult:
    """Combined result from the ML pipeline."""

    regime_prediction: pd.Series
    regime_probabilities: pd.DataFrame
    signal_scores: Optional[pd.DataFrame]
    feature_importance_regime: pd.DataFrame
    feature_importance_signals: pd.DataFrame
    pipeline_metrics: Dict[str, Any]


class MLPipeline:
    """
    Orchestrates ML regime detection and signal scoring.

    Combines FeatureEngineer, RegimeClassifier, and SignalScorer
    into a single pipeline for training and inference.

    Example:
        >>> pipeline = MLPipeline()
        >>> results = pipeline.fit(df, regime_labels, signal_labels)
        >>> regimes = pipeline.predict_regime(new_df)
        >>> scores = pipeline.score_signals(signal_features)
    """

    def __init__(
        self,
        regime_model_type: str = "random_forest",
        signal_model_type: str = "gradient_boosting",
        n_estimators: int = 100,
        max_depth: int = 5,
        random_state: int = 42,
    ):
        """
        Initialize ML pipeline.

        Args:
            regime_model_type: Model type for regime classification
            signal_model_type: Model type for signal scoring
            n_estimators: Number of trees for tree-based models
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        self.feature_engineer = FeatureEngineer()
        self.regime_classifier = RegimeClassifier(
            model_type=regime_model_type,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.signal_scorer = SignalScorer(
            model_type=signal_model_type,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.feature_names_: List[str] = []
        self._is_fitted = False

    def _get_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features and return only numeric columns."""
        all_features = self.feature_engineer.generate_features(df)
        return all_features.select_dtypes(include=[np.number])

    def fit(
        self,
        df: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
        signal_features: Optional[pd.DataFrame] = None,
        signal_labels: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Train regime classifier and optionally signal scorer.

        Args:
            df: OHLCV DataFrame
            regime_labels: Series of regime labels (for regime classifier)
            signal_features: DataFrame of signal-specific features
            signal_labels: Series of signal outcomes (1=profitable, 0=not)

        Returns:
            Dict with training metrics for both models
        """
        results = {}

        # Generate numeric features only
        numeric_features = self._get_numeric_features(df)
        self.feature_names_ = list(numeric_features.columns)

        # Train regime classifier
        if regime_labels is not None:
            labels_aligned = regime_labels.reindex(df.index)
            valid = numeric_features.notna().all(axis=1) & labels_aligned.notna()

            X_regime = numeric_features[valid].reset_index(drop=True)
            y_regime = labels_aligned[valid].reset_index(drop=True)

            if len(X_regime) > 0:
                try:
                    regime_results = self.regime_classifier.train(X_regime, y_regime, test_size=0.3)
                    results["regime"] = regime_results
                except ValueError:
                    results["regime"] = {}
            else:
                results["regime"] = {}

        # Train signal scorer
        if signal_features is not None and signal_labels is not None:
            signal_results = self.signal_scorer.train(signal_features, signal_labels)
            results["signal"] = signal_results

        self._is_fitted = True
        return results

    def predict_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict market regime for OHLCV data.

        Args:
            df: OHLCV DataFrame

        Returns:
            Series of predicted regime labels
        """
        if not self._is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        numeric = self._get_numeric_features(df)
        numeric = numeric.ffill().bfill()
        numeric = numeric.reindex(columns=self.feature_names_, fill_value=0)

        return self.regime_classifier.predict(numeric)

    def get_regime_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get regime prediction probabilities.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with probability for each regime
        """
        if not self._is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        numeric = self._get_numeric_features(df)
        numeric = numeric.ffill().bfill()
        numeric = numeric.reindex(columns=self.feature_names_, fill_value=0)

        return self.regime_classifier.predict_proba(numeric)

    def score_signals(self, signal_features: pd.DataFrame) -> pd.DataFrame:
        """
        Score trading signals with ML quality assessment.

        Args:
            signal_features: DataFrame of signal features

        Returns:
            DataFrame with original features + ml_score + is_recommended
        """
        if not self._is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        if signal_features is None or self.signal_scorer.model is None:
            return pd.DataFrame()

        return self.signal_scorer.score(signal_features)

    def get_regime_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get most important features for regime classification.
        """
        return self.regime_classifier.get_feature_importance(top_n)

    def get_signal_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get most important features for signal quality scoring.
        """
        if self.signal_scorer.model is None:
            return pd.DataFrame()
        return self.signal_scorer.get_top_features_by_importance(top_n)

    def walk_forward_validation(
        self,
        df: pd.DataFrame,
        regime_labels: pd.Series,
        signal_features: Optional[pd.DataFrame] = None,
        signal_labels: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation for both models.
        """
        results = {}

        numeric = self._get_numeric_features(df)
        numeric = numeric.ffill().bfill()

        labels_aligned = regime_labels.reindex(numeric.index)
        valid = numeric.notna().all(axis=1) & labels_aligned.notna()

        X = numeric[valid].reset_index(drop=True)
        y = labels_aligned[valid].reset_index(drop=True)

        if len(X) > 100:
            wf_results = self.regime_classifier.walk_forward_validation(
                X, y, train_size=min(300, len(X) // 3), step_size=min(100, len(X) // 5)
            )
            if wf_results:
                results["regime"] = wf_results
            else:
                results["regime"] = {}
        else:
            results["regime"] = {}

        if signal_features is not None and signal_labels is not None:
            results["signal"] = self.signal_scorer.walk_forward_validation(
                signal_features, signal_labels, train_size=300, step_size=100
            )

        return results

    def run_pipeline(
        self,
        df: pd.DataFrame,
        regime_labels: pd.Series,
        signal_features: pd.DataFrame,
        signal_labels: pd.Series,
    ) -> PipelineResult:
        """
        Run complete pipeline: fit, predict, and evaluate.
        """
        self.fit(df, regime_labels, signal_features, signal_labels)

        regime_pred = self.predict_regime(df)
        regime_proba = self.get_regime_probabilities(df)

        if signal_features is not None and self.signal_scorer.model is not None:
            signal_scores = self.score_signals(signal_features)
            signal_importance = self.get_signal_feature_importance()
        else:
            signal_scores = pd.DataFrame()
            signal_importance = pd.DataFrame()

        regime_importance = self.get_regime_feature_importance()

        regime_valid = regime_pred.notna()
        regime_accuracy = (regime_pred[regime_valid] == regime_labels[regime_valid]).mean()

        return PipelineResult(
            regime_prediction=regime_pred,
            regime_probabilities=regime_proba,
            signal_scores=signal_scores,
            feature_importance_regime=regime_importance,
            feature_importance_signals=signal_importance,
            pipeline_metrics={
                "regime_accuracy": float(regime_accuracy),
                "n_features": len(self.feature_names_),
                "is_fitted": self._is_fitted,
            },
        )
