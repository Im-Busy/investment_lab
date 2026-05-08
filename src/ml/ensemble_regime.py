"""
Ensemble Regime Detection

Combines rule-based (ADX/ATR) and ML regime classifiers into an ensemble
for more stable and accurate regime detection.

Ensemble strategies:
1. Majority vote (if 3+ models)
2. Weighted average (ML confidence + rule-based signal)
3. Rule-based veto (ML must agree within tolerance)
4. Time-decay ensemble (ML dominates short-term, rules long-term)

Usage:
    from src.ml.ensemble_regime import EnsembleRegimeDetector
    from src.indicators.regime_detector import RegimeDetector
    from src.ml.regime_model import RegimeClassifier

    ens = EnsembleRegimeDetector(regime_detector, ml_classifier)
    labels = ens.predict(df)
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class EnsembleStrategy(str, Enum):
    """Ensemble combination strategy."""

    WEIGHTED = "weighted"
    RULE_VETO = "rule_veto"
    CONSENSUS = "consensus"
    TIME_DECAY = "time_decay"


class EnsembleRegimeDetector:
    """
    Ensemble regime detector combining rule-based and ML classifiers.

    Example:
        >>> rule_detector = RegimeDetector()
        >>> ml_classifier = RegimeClassifier("gradient_boosting")
        >>> ml_classifier.train(X_train, y_train)
        >>>
        >>> ensemble = EnsembleRegimeDetector(
        ...     rule_detector=rule_detector,
        ...     ml_classifier=ml_classifier,
        ...     strategy=EnsembleStrategy.WEIGHTED,
        ...     ml_weight=0.4,
        ... )
        >>> labels = ensemble.predict(df)
    """

    def __init__(
        self,
        rule_detector=None,
        ml_classifier=None,
        strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED,
        ml_weight: float = 0.4,
        consensus_threshold: float = 0.7,
        rule_veto_regime: str = "Volatile",
        time_decay_window: int = 50,
    ):
        """
        Initialize ensemble detector.

        Args:
            rule_detector: Rule-based RegimeDetector instance
            ml_classifier: Trained ML RegimeClassifier instance
            strategy: Ensemble combination strategy
            ml_weight: Weight for ML in weighted strategy (0-1)
            consensus_threshold: Fraction of models that must agree
            rule_veto_regime: Regime where rule-based always wins
            time_decay_window: Window for time-decay strategy
        """
        self.rule_detector = rule_detector
        self.ml_classifier = ml_classifier
        self.strategy = strategy
        self.ml_weight = ml_weight
        self.rule_weight = 1.0 - ml_weight
        self.consensus_threshold = consensus_threshold
        self.rule_veto_regime = rule_veto_regime
        self.time_decay_window = time_decay_window
        self._rule_labels_ = None
        self._ml_probas_ = None

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate ensemble regime labels.

        Args:
            df: OHLCV DataFrame

        Returns:
            Series of regime labels
        """
        if self.rule_detector is None and self.ml_classifier is None:
            raise ValueError("At least one detector must be provided")

        if self.rule_detector is None:
            return self.ml_classifier.predict(self._get_features(df))

        if self.ml_classifier is None:
            result = self.rule_detector.get_regime_series(df)
            return result["regime"].apply(lambda r: r.value)

        self._rule_labels_ = self._get_rule_labels(df)
        self._ml_probas_ = self._get_ml_predictions(df)

        if self.strategy == EnsembleStrategy.WEIGHTED:
            return self._weighted_ensemble(df)
        elif self.strategy == EnsembleStrategy.RULE_VETO:
            return self._rule_veto_ensemble(df)
        elif self.strategy == EnsembleStrategy.CONSENSUS:
            return self._consensus_ensemble(df)
        elif self.strategy == EnsembleStrategy.TIME_DECAY:
            return self._time_decay_ensemble(df)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _weighted_ensemble(self, df: pd.DataFrame) -> pd.Series:
        """Weighted combination: use ML probabilities weighted by rule confidence."""
        final_labels = []
        all_regimes = sorted(self._rule_labels_.unique())

        for idx in df.index:
            rule_label = self._rule_labels_.get(idx, None)
            if rule_label is None or idx not in self._ml_probas_.index:
                final_labels.append(rule_label)
                continue

            ml_proba = self._ml_probas_.loc[idx]

            # Find ML's top prediction
            ml_pred = ml_proba.idxmax() if hasattr(ml_proba, "idxmax") else None

            # If ML and rule agree, use that
            if ml_pred == rule_label:
                final_labels.append(rule_label)
            else:
                # Weighted: if ML confidence > rule threshold, go with ML
                ml_conf = ml_proba.get(rule_label, 0) if hasattr(ml_proba, "get") else 0
                if ml_conf > self.ml_weight:
                    final_labels.append(ml_pred)
                else:
                    final_labels.append(rule_label)

        return pd.Series(final_labels, index=df.index, name="ensemble_regime")

    def _rule_veto_ensemble(self, df: pd.DataFrame) -> pd.Series:
        """Rule-based veto: rule-based always wins in specific regimes (e.g., Volatile)."""
        ml_labels = self._ml_probas_.apply(lambda x: x.idxmax(), axis=1)

        # Rule veto: if rule detector says Volatile, always use rule
        result = ml_labels.copy()
        is_volatile = self._rule_labels_ == self.rule_veto_regime
        result[is_volatile] = self.rule_veto_regime

        return result

    def _consensus_ensemble(self, df: pd.DataFrame) -> pd.Series:
        """Consensus: only change from rule-based if ML confidence exceeds threshold."""
        result = self._rule_labels_.copy()

        for idx in df.index:
            if idx not in self._ml_probas_.index:
                continue

            ml_proba = self._ml_probas_.loc[idx]
            ml_pred = ml_proba.idxmax()
            ml_conf = ml_proba.max()

            # If ML confidence > threshold and disagrees with rule, use ML
            if ml_conf > self.consensus_threshold and ml_pred != result.get(idx):
                result[idx] = ml_pred

        return result

    def _time_decay_ensemble(self, df: pd.DataFrame) -> pd.Series:
        """Time-decay: ML dominates short-term patterns, rules dominate long-term."""
        result = self._rule_labels_.copy()

        for idx in df.index:
            if idx not in self._ml_probas_.index:
                continue

            ml_proba = self._ml_probas_.loc[idx]
            ml_pred = ml_proba.idxmax()
            ml_conf = ml_proba.max()
            rule_label = result.get(idx)

            if ml_pred == rule_label:
                continue

            if ml_conf > self.ml_weight * 2:
                result[idx] = ml_pred

        return result

    def _get_rule_labels(self, df: pd.DataFrame) -> pd.Series:
        """Get rule-based regime labels."""
        if self._rule_labels_ is not None:
            return self._rule_labels_

        result = self.rule_detector.get_regime_series(df)
        return result["regime"].apply(lambda r: r.value)

    def _get_ml_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get ML regime probabilities."""
        if self._ml_probas_ is not None:
            return self._ml_probas_

        # Need features - try cross-asset first, then basic
        features = None
        try:
            if hasattr(self.ml_classifier, "predict_proba"):
                features_df = self._get_features(df)
                return self.ml_classifier.predict_proba(features_df)
        except Exception:
            pass

        # Fallback: return uniform probabilities
        all_regimes = ["Trending", "Ranging", "Volatile", "Transition"]
        return pd.DataFrame(
            {r: 0.25 for r in all_regimes},
            index=df.index,
        )

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from OHLCV data."""
        from src.ml.features import FeatureEngineer

        engineer = FeatureEngineer()
        features = engineer.generate_features(df)
        numeric = features.select_dtypes(include=[np.number])
        return numeric.ffill().bfill().dropna()

    def get_agreement(self) -> float:
        """Get agreement rate between rule-based and ML."""
        if self._rule_labels_ is None or self._ml_probas_ is None:
            raise ValueError("Must call predict() first")

        ml_labels = self._ml_probas_.apply(lambda x: x.idxmax(), axis=1)
        common_idx = self._rule_labels_.index.intersection(ml_labels.index)
        return (self._rule_labels_[common_idx] == ml_labels[common_idx]).mean()

    def get_ml_confidence(self, df: pd.DataFrame) -> pd.Series:
        """Get ML prediction confidence (max probability)."""
        return self._ml_probas_.max(axis=1)
