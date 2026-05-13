"""Pattern Confidence Scorer — ML scores per pattern detection probability.

Wraps PatternClassifier to provide confidence scoring for pattern detections.
Each detected pattern is evaluated by a trained CatBoost model, producing:
- Probability of profitability (0.0–1.0)
- Confidence score (distance from 0.5, normalized to 0.0–1.0)
- Recommendation flag (passes threshold)
- Top contributory features (positive and negative)

Usage:
    from src.ml.pattern_scorer import PatternScorer

    scorer = PatternScorer()
    scorer.train(features, labels)
    results = scorer.score_detections(new_features)
    # results["probability_profitable"] → per-pattern probability
    # results["is_recommended"] → boolean filter
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PatternScore:
    """Scored result for a single pattern detection."""

    timestamp: pd.Timestamp | str
    pattern_name: str
    direction: str
    probability_profitable: float
    confidence: float
    is_recommended: bool
    top_reasons: List[str]


@dataclass
class ScoringResult:
    """Aggregate scoring result for a batch of pattern detections."""

    pattern_scores: List[PatternScore]
    summary: pd.DataFrame
    acceptance_rate: float
    mean_probability: float
    mean_confidence: float


class PatternScorer:
    """ML-based pattern confidence scorer.

    Trains a CatBoost classifier on historical pattern data and uses it to
    score new pattern detections for trading signal quality.

    Example:
        >>> scorer = PatternScorer()
        >>> result = scorer.train(X_train, y_train)
        >>> scored = scorer.score_detections(X_new)
        >>> recommended = scored[scored["is_recommended"]]
    """

    def __init__(
        self,
        model_type: str = "catboost",
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.03,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        bagging_temperature: float = 1.0,
        border_count: int = 128,
        min_data_in_leaf: int = 20,
        random_state: int = 42,
        threshold: float = 0.50,
    ):
        """Initialize pattern scorer.

        Args:
            model_type: Type of classifier (catboost recommended).
            n_estimators: Number of boosting iterations.
            max_depth: Tree depth.
            learning_rate: Learning rate.
            l2_leaf_reg: L2 regularization.
            random_strength: CatBoost random score strength.
            bagging_temperature: Bayesian bootstrap temperature.
            border_count: Number of splits for numeric features.
            min_data_in_leaf: Minimum samples per leaf.
            random_state: Random seed.
            threshold: Probability threshold for recommendation.
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.border_count = border_count
        self.min_data_in_leaf = min_data_in_leaf
        self.random_state = random_state
        self.threshold = threshold

        self._model: object | None = None
        self._feature_names: List[str] = []
        self._feature_importance: Dict[str, float] = {}
        self._is_trained: bool = False
        self._train_auc: float = 0.0
        self._test_auc: float = 0.0

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.3,
        purge_window: int = 5,
    ) -> Dict[str, float]:
        """Train the pattern scorer on historical data.

        Args:
            X: Feature matrix (patterns × features).
            y: Binary labels (1 = profitable, 0 = not).
            test_size: Fraction of data for test split.
            purge_window: Samples to exclude at split boundary.

        Returns:
            Dict with train_auc, test_auc, accuracy, n_samples.
        """
        from sklearn.metrics import accuracy_score, roc_auc_score

        clean_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[clean_mask].copy()
        y_clean = y[clean_mask].copy()

        if len(X_clean) < 50:
            raise ValueError(f"Need >=50 samples, got {len(X_clean)}")

        split_idx = int(len(X_clean) * (1 - test_size))
        train_end = max(0, split_idx - purge_window)

        X_train = X_clean.iloc[:train_end]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:train_end]
        y_test = y_clean.iloc[split_idx:]

        self._feature_names = list(X_train.columns)

        self._model = self._create_model()
        self._model.fit(X_train, y_train)

        train_proba = self._model.predict_proba(X_train)[:, 1]
        test_proba = self._model.predict_proba(X_test)[:, 1]

        self._train_auc = float(roc_auc_score(y_train, train_proba))
        self._test_auc = float(roc_auc_score(y_test, test_proba))

        train_pred = (train_proba >= self.threshold).astype(int)
        test_pred = (test_proba >= self.threshold).astype(int)

        if hasattr(self._model, "feature_importances_"):
            self._feature_importance = dict(
                sorted(
                    zip(self._feature_names, self._model.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )

        self._is_trained = True

        return {
            "train_auc": self._train_auc,
            "test_auc": self._test_auc,
            "train_accuracy": float(accuracy_score(y_train, train_pred)),
            "test_accuracy": float(accuracy_score(y_test, test_pred)),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    def _create_model(self) -> object:
        """Create the underlying ML model."""
        try:
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                iterations=self.n_estimators,
                depth=self.max_depth,
                learning_rate=self.learning_rate,
                l2_leaf_reg=self.l2_leaf_reg,
                random_strength=self.random_strength,
                bagging_temperature=self.bagging_temperature,
                border_count=self.border_count,
                min_data_in_leaf=self.min_data_in_leaf,
                random_state=self.random_state,
                verbose=False,
                loss_function="Logloss",
            )
        except ImportError:
            raise ImportError("catboost required. Run: uv add catboost")

    def score_detections(
        self,
        X: pd.DataFrame,
        pattern_metadata: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Score a batch of pattern detections.

        Args:
            X: Feature matrix for patterns to score (must match training features).
            pattern_metadata: Optional DataFrame with columns:
                timestamp, pattern_name, direction.

        Returns:
            DataFrame with columns:
                probability_profitable, confidence, is_recommended,
                plus original metadata columns.
        """
        if not self._is_trained:
            raise ValueError("Scorer not trained. Call train() first.")

        X_clean = X[self._feature_names].fillna(0)

        proba = self._model.predict_proba(X_clean.values)[:, 1]

        result = X.copy()
        result["probability_profitable"] = proba
        result["confidence"] = np.abs(proba - 0.5) * 2.0
        result["is_recommended"] = proba >= self.threshold

        if pattern_metadata is not None:
            for col in ["timestamp", "pattern_name", "direction"]:
                if col in pattern_metadata.columns:
                    result[col] = pattern_metadata.loc[
                        X.index.intersection(pattern_metadata.index), col
                    ].values

        return result

    def score_single(
        self,
        features: pd.Series | np.ndarray,
        pattern_name: str = "Unknown",
        direction: str = "Unknown",
        timestamp: pd.Timestamp | None = None,
        top_reasons_n: int = 3,
    ) -> PatternScore:
        """Score a single pattern detection.

        Args:
            features: Feature vector for the pattern.
            pattern_name: Name of the detected pattern.
            direction: Signal direction ('bullish' or 'bearish').
            timestamp: Detection timestamp.
            top_reasons_n: Number of top feature reasons to report.

        Returns:
            PatternScore with probability and recommendation.
        """
        if not self._is_trained:
            raise ValueError("Scorer not trained. Call train() first.")

        if isinstance(features, pd.Series):
            vals = features[self._feature_names].fillna(0).values.reshape(1, -1)
        else:
            vals = np.array(features).reshape(1, -1)

        proba = float(self._model.predict_proba(vals)[:, 1][0])
        confidence = abs(proba - 0.5) * 2.0
        is_recommended = proba >= self.threshold

        ts = timestamp or pd.Timestamp.now()
        reasons = self._explain_prediction(vals[0])

        return PatternScore(
            timestamp=ts,
            pattern_name=pattern_name,
            direction=direction,
            probability_profitable=proba,
            confidence=confidence,
            is_recommended=is_recommended,
            top_reasons=reasons[:top_reasons_n],
        )

    def _explain_prediction(self, feature_values: np.ndarray) -> List[str]:
        """Generate human-readable explanation for a prediction."""
        if not self._feature_importance:
            return ["No feature importance available"]

        contributions = feature_values * np.array(
            [self._feature_importance.get(f, 0) for f in self._feature_names]
        )

        sorted_idx = np.argsort(np.abs(contributions))[::-1][:5]
        reasons = []
        for idx in sorted_idx:
            name = self._feature_names[idx]
            value = feature_values[idx]
            contrib = contributions[idx]
            direction_str = "positive" if contrib > 0 else "negative"
            reasons.append(f"{name}={value:.2f} ({direction_str}, impact={abs(contrib):.4f})")
        return reasons

    def score_batch(
        self,
        X: pd.DataFrame,
        pattern_metadata: pd.DataFrame | None = None,
    ) -> ScoringResult:
        """Score a batch and return structured results.

        Args:
            X: Feature matrix.
            pattern_metadata: Optional metadata DataFrame.

        Returns:
            ScoringResult with list of PatternScore objects and summary.
        """
        scored = self.score_detections(X, pattern_metadata)

        pattern_scores = []
        for idx in scored.index:
            meta = {}
            if pattern_metadata is not None and idx in pattern_metadata.index:
                meta = pattern_metadata.loc[idx].to_dict()

            ps = PatternScore(
                timestamp=meta.get("timestamp", idx),
                pattern_name=meta.get("pattern_name", "Unknown"),
                direction=meta.get("direction", "Unknown"),
                probability_profitable=float(scored.loc[idx, "probability_profitable"]),
                confidence=float(scored.loc[idx, "confidence"]),
                is_recommended=bool(scored.loc[idx, "is_recommended"]),
                top_reasons=[],
            )
            pattern_scores.append(ps)

        summary = scored[["probability_profitable", "confidence", "is_recommended"]].describe()

        return ScoringResult(
            pattern_scores=pattern_scores,
            summary=summary,
            acceptance_rate=float(scored["is_recommended"].mean()),
            mean_probability=float(scored["probability_profitable"].mean()),
            mean_confidence=float(scored["confidence"].mean()),
        )

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return feature importance as a sorted DataFrame."""
        items = sorted(self._feature_importance.items(), key=lambda x: x[1], reverse=True)
        return pd.DataFrame(items[:top_n], columns=["feature", "importance"])

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def train_auc(self) -> float:
        return self._train_auc

    @property
    def test_auc(self) -> float:
        return self._test_auc

    def save(self, path: str | Path) -> None:
        """Save trained scorer to disk."""
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self._model,
            "feature_names": self._feature_names,
            "feature_importance": self._feature_importance,
            "train_auc": self._train_auc,
            "test_auc": self._test_auc,
            "threshold": self.threshold,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str | Path) -> None:
        """Load trained scorer from disk."""
        import pickle

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scorer file not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)  # nosec B301

        self._model = data["model"]
        self._feature_names = data["feature_names"]
        self._feature_importance = data.get("feature_importance", {})
        self._train_auc = data.get("train_auc", 0.0)
        self._test_auc = data.get("test_auc", 0.0)
        self.threshold = data.get("threshold", self.threshold)
        self._is_trained = True
