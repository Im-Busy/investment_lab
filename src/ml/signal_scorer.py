"""
ML Signal Scorer

Uses machine learning to score trading signals based on their probability
of being profitable. Augments rule-based confluence scoring with learned
patterns from historical signal outcomes.

Can:
1. Train on historical signal features and trade outcomes
2. Score new signals with probability of profitability
3. Provide feature importance for signal quality factors
4. Integrate with existing ConfluenceScorer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ScoredSignal:
    """A signal with ML-enhanced quality score."""

    timestamp: pd.Timestamp
    pattern_name: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    original_confidence: float
    ml_score: float
    ml_probability_profitable: float
    is_recommended: bool
    top_features: Dict[str, float]


class SignalScorer:
    """
    ML-based signal quality scorer.

    Trains on historical signals labeled with their outcomes (profitable or not)
    and scores new signals based on learned patterns.

    Example:
        >>> scorer = SignalScorer(model_type="gradient_boosting")
        >>> scorer.train(X_train, y_train)  # y = 1 if profitable, 0 otherwise
        >>> scores = scorer.score(X_new)
    """

    SUPPORTED_MODELS = [
        "gradient_boosting",
        "random_forest",
        "logistic_regression",
        "xgboost",
    ]

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        n_estimators: int = 100,
        max_depth: int = 4,
        random_state: int = 42,
    ):
        """
        Initialize signal scorer.

        Args:
            model_type: Type of classifier
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model type: {model_type}. Supported: {self.SUPPORTED_MODELS}"
            )

        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.feature_names_ = None
        self.threshold_ = 0.5  # Default threshold for recommendation

    def _create_model(self):
        """Create the underlying ML model."""
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            RandomForestClassifier,
        )
        from sklearn.linear_model import LogisticRegression

        if self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                class_weight="balanced",
            )
        elif self.model_type == "logistic_regression":
            return LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state,
            )
        else:
            try:
                import xgboost as xgb

                return xgb.XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    scale_pos_weight=1,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )
            except ImportError:
                raise ImportError("xgboost not installed. Run: uv add xgboost")

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Train the scorer on signal features and labels.

        Args:
            X: Signal features DataFrame
            y: Labels (1 = profitable, 0 = not profitable)
            threshold: Score threshold for recommending signals

        Returns:
            Dict with training metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            precision_recall_curve,
            roc_auc_score,
        )

        # Clean NaN
        valid_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        # Train/test split (time-based)
        split_idx = int(len(X_clean) * 0.7)
        X_train = X_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:split_idx]
        y_test = y_clean.iloc[split_idx:]

        # Create and train model
        self.model = self._create_model()
        self.feature_names_ = list(X_train.columns)
        self.threshold_ = threshold
        self.model.fit(X_train, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        test_proba = self.model.predict_proba(X_test)[:, 1]

        # Find optimal threshold if not specified
        if threshold is None:
            precision, recall, thresholds = precision_recall_curve(y_test, test_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            self.threshold_ = thresholds[min(optimal_idx, len(thresholds) - 1)]

        results = {
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "test_auc_roc": roc_auc_score(y_test, test_proba),
            "test_precision": precision
            if (precision := (y_test[test_pred == 1] == 1).mean()) > 0
            else 0,
            "test_report": classification_report(y_test, test_pred, output_dict=True),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "optimal_threshold": self.threshold_,
            "n_profitable_train": int(y_train.sum()),
            "n_profitable_test": int(y_test.sum()),
        }

        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            results["feature_importance"] = dict(
                sorted(
                    zip(self.feature_names_, self.model.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True,
                )[:15]
            )

        return results

    def score(
        self,
        X: pd.DataFrame,
        signals: Optional[List[Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """
        Score signals with ML probability.

        Args:
            X: Signal features DataFrame
            signals: Optional list of signal metadata dicts

        Returns:
            DataFrame with original features + ml_score + is_recommended
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_clean = X.fillna(0)
        proba = self.model.predict_proba(X_clean)[:, 1]

        scored = X.copy()
        scored["ml_score"] = proba
        scored["is_recommended"] = proba >= self.threshold_

        return scored

    def walk_forward_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_size: int = 500,
        step_size: int = 100,
    ) -> Dict[str, List[float]]:
        """
        Walk-forward validation for signal scoring.

        Args:
            X: Features DataFrame
            y: Labels series
            train_size: Initial training window
            step_size: Step for expanding window

        Returns:
            Dict with train/test accuracy lists
        """
        from sklearn.metrics import accuracy_score, roc_auc_score

        results = {"train_accuracy": [], "test_accuracy": [], "test_auc": []}
        start = 0

        while start + train_size < len(X):
            end = min(start + train_size + step_size, len(X))

            X_train = X.iloc[start : start + train_size]
            y_train = y.iloc[start : start + train_size]
            X_test = X.iloc[start + train_size : end]
            y_test = y.iloc[start + train_size : end]

            if len(X_test) == 0:
                break

            valid_train = X_train.notna().all(axis=1) & y_train.notna()
            valid_test = X_test.notna().all(axis=1) & y_test.notna()

            if valid_train.sum() < 50 or valid_test.sum() < 10:
                start += step_size
                continue

            model = self._create_model()
            model.fit(X_train[valid_train], y_train[valid_train])

            train_pred = model.predict(X_train[valid_train])
            test_pred = model.predict(X_test[valid_test])

            try:
                test_proba = model.predict_proba(X_test[valid_test])[:, 1]
                test_auc = roc_auc_score(y_test[valid_test], test_proba)
            except Exception:
                test_auc = 0.5

            results["train_accuracy"].append(accuracy_score(y_train[valid_train], train_pred))
            results["test_accuracy"].append(accuracy_score(y_test[valid_test], test_pred))
            results["test_auc"].append(test_auc)

            start += step_size

        return results

    def get_top_features_by_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get top N most important features for signal scoring.

        Args:
            top_n: Number of features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if not hasattr(self.model, "feature_importances_"):
            if hasattr(self.model, "coef_"):
                importance = np.abs(self.model.coef_).mean(axis=0)
                return (
                    pd.DataFrame({"feature": self.feature_names_, "importance": importance})
                    .sort_values("importance", ascending=False)
                    .head(top_n)
                )
            return pd.DataFrame()

        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "importance": self.model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(top_n)
        )
