"""
Pattern Classifier using LightGBM

ML-based classifier for trading pattern detection enhancement.
Provides probability calibration, feature importance, and walk-forward validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import brier_score_loss


@dataclass
class ClassifierTrainingResult:
    """Results from training the pattern classifier."""

    train_auc: float
    test_auc: float
    train_accuracy: float
    test_accuracy: float
    feature_importance: Dict[str, float]
    calibration_error: float
    n_train_samples: int
    n_test_samples: int
    n_profitable_train: int
    n_profitable_test: int


@dataclass
class PatternPrediction:
    """Prediction for a pattern instance."""

    timestamp: pd.Timestamp
    pattern_name: str
    direction: str
    probability_profitable: float
    confidence: float
    is_recommended: bool
    top_positive_features: List[Tuple[str, float]]
    top_negative_features: List[Tuple[str, float]]


class PatternClassifier:
    """
    LightGBM-based pattern classifier.

    Predicts probability that a detected pattern will be profitable.

    Example:
        >>> clf = PatternClassifier(n_estimators=200, max_depth=6)
        >>> result = clf.train(X_train, y_train)
        >>> predictions = clf.predict(X_test)
    """

    SUPPORTED_MODELS = ["catboost", "chronos", "fincast", "xlstm"]

    def __init__(
        self,
        model_type: str = "catboost",
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.03,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        bagging_temperature: float = 1.0,
        border_count: int = 128,
        min_data_in_leaf: int = 20,
    ):
        """
        Initialize pattern classifier.

        Args:
            model_type: Type of gradient boosting model
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            min_child_samples: Minimum samples per leaf
            subsample: Subsample ratio
            colsample_bytree: Feature subsample ratio
            random_state: Random seed
            n_jobs: Number of parallel jobs
            l2_leaf_reg: L2 regularization coefficient (CatBoost)
            random_strength: Random score strength for overfitting (CatBoost)
            bagging_temperature: Bayesian bootstrap temperature (CatBoost)
            border_count: Number of splits for numeric features (CatBoost)
            min_data_in_leaf: Minimum training samples in leaf (CatBoost)
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model type: {model_type}. Supported: {self.SUPPORTED_MODELS}"
            )

        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.border_count = border_count
        self.min_data_in_leaf = min_data_in_leaf

        self.model = None
        self.feature_names_: Optional[List[str]] = None
        self.calibration_slope_: Optional[float] = None
        self.calibration_intercept_: Optional[float] = None

    def _has_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _create_model(self):
        """Create the underlying ML model."""
        if self.model_type == "catboost":
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
                    task_type="GPU" if self._has_gpu() else "CPU",
                )
            except ImportError:
                raise ImportError("catboost not installed. Run: uv add catboost")

        elif self.model_type == "chronos":
            from src.ml.models.chronos import ChronosForecaster

            return ChronosForecaster(model_size="base")

        elif self.model_type == "fincast":
            from src.ml.models.fincast import FinCastForecaster

            return FinCastForecaster.from_zero_shot()

        elif self.model_type == "xlstm":
            from src.ml.models.xlstm import xLSTMForecaster

            return xLSTMForecaster(hidden_size=128, num_layers=4)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        calibration_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        purge_window: int = 5,
    ) -> ClassifierTrainingResult:
        """
        Train the pattern classifier.

        Args:
            X: Feature matrix (patterns x features)
            y: Labels (1 = profitable, 0 = not profitable)
            calibration_data: Optional held-out data for probability calibration
            purge_window: Number of training samples at the split boundary
                to exclude from training (prevents label overlap leakage).
                Set to match the forward-return horizon used for labels.

        Returns:
            Training results with metrics and feature importance
        """
        from sklearn.metrics import (
            accuracy_score,
            roc_auc_score,
        )

        clean_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[clean_mask].copy()
        y_clean = y[clean_mask].copy()

        if len(X_clean) < 50:
            raise ValueError(f"Insufficient samples after cleaning: {len(X_clean)}")

        split_idx = int(len(X_clean) * 0.7)
        train_end = max(0, split_idx - purge_window)
        X_train = X_clean.iloc[:train_end]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:train_end]
        y_test = y_clean.iloc[split_idx:]

        self.feature_names_ = list(X_train.columns)

        self.model = self._create_model()
        self.model.fit(X_train, y_train)

        train_proba = self.model.predict_proba(X_train)[:, 1]
        test_proba = self.model.predict_proba(X_test)[:, 1]

        train_pred = (train_proba >= 0.5).astype(int)
        test_pred = (test_proba >= 0.5).astype(int)

        train_auc = roc_auc_score(y_train, train_proba)
        test_auc = roc_auc_score(y_test, test_proba)

        calibration_error = self._calibrate_probabilities(
            test_proba, y_test.values, calibration_data
        )

        feature_importance = self._extract_feature_importance()

        result = ClassifierTrainingResult(
            train_auc=float(train_auc),
            test_auc=float(test_auc),
            train_accuracy=float(accuracy_score(y_train, train_pred)),
            test_accuracy=float(accuracy_score(y_test, test_pred)),
            feature_importance=feature_importance,
            calibration_error=calibration_error,
            n_train_samples=len(X_train),
            n_test_samples=len(X_test),
            n_profitable_train=int(y_train.sum()),
            n_profitable_test=int(y_test.sum()),
        )

        return result

    def _calibrate_probabilities(
        self,
        proba: np.ndarray,
        y_true: np.ndarray,
        calibration_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> float:
        """
        Calibrate predicted probabilities using Platt scaling or isotonic regression.

        Returns calibration error (Brier score loss).
        """
        from sklearn.calibration import calibration_curve

        if calibration_data is not None:
            X_cal, y_cal = calibration_data
            X_cal = X_cal.dropna()
            y_cal = y_cal.reindex(X_cal.index).dropna()

            if len(X_cal) > 50:
                cal_proba = self.model.predict_proba(X_cal)[:, 1]
                proba = cal_proba
                y_true = y_cal.values

        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, proba, n_bins=10, strategy="uniform"
            )

            from sklearn.linear_model import LogisticRegression

            calibrator = LogisticRegression()
            calibrator.fit(proba.reshape(-1, 1), y_true)

            self.calibration_slope_ = float(calibrator.coef_[0, 0])
            self.calibration_intercept_ = float(calibrator.intercept_[0])

            calibrated_proba = calibrator.predict_proba(proba.reshape(-1, 1))[:, 1]
            calibration_error = float(brier_score_loss(y_true, calibrated_proba))

        except Exception:
            calibration_error = float(brier_score_loss(y_true, proba))

        return calibration_error

    def _extract_feature_importance(self) -> Dict[str, float]:
        """Extract feature importance from trained model."""
        if not hasattr(self.model, "feature_importances_"):
            return {}

        importance = self.model.feature_importances_

        return dict(
            sorted(
                zip(self.feature_names_, importance),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Predict probability of profitability for patterns.

        Args:
            X: Feature matrix
            threshold: Probability threshold for recommendation

        Returns:
            DataFrame with predictions and recommendations
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_clean = X.fillna(0)

        proba = self.model.predict_proba(X_clean)[:, 1]

        if self.calibration_slope_ is not None:
            log_odds = (
                self.calibration_slope_ * proba / (1 - proba + 1e-10) + self.calibration_intercept_
            )
            proba_calibrated = 1 / (1 + np.exp(-log_odds))
            proba = np.clip(proba_calibrated, 0, 1)

        result = X.copy()
        result["probability_profitable"] = proba
        result["is_recommended"] = proba >= threshold
        result["confidence"] = np.abs(proba - 0.5) * 2

        return result

    def predict_patterns(
        self,
        X: pd.DataFrame,
        pattern_metadata: Optional[pd.DataFrame] = None,
        threshold: float = 0.5,
    ) -> List[PatternPrediction]:
        """
        Predict for pattern instances with metadata.

        Args:
            X: Feature matrix
            pattern_metadata: Optional DataFrame with timestamp, pattern_name, direction
            threshold: Recommendation threshold

        Returns:
            List of PatternPrediction objects
        """
        scored = self.predict(X, threshold)

        predictions = []
        for idx in scored.index:
            if pattern_metadata is not None and idx in pattern_metadata.index:
                meta = pattern_metadata.loc[idx]
                timestamp = meta.get("timestamp", idx)
                pattern_name = meta.get("pattern_name", "Unknown")
                direction = meta.get("direction", "Unknown")
            else:
                timestamp = idx if isinstance(idx, pd.Timestamp) else pd.Timestamp.now()
                pattern_name = "Unknown"
                direction = "Unknown"

            proba = float(scored.loc[idx, "probability_profitable"])

            row = X.loc[idx] if idx in X.index else X.iloc[0]
            row = row.fillna(0)

            if hasattr(self.model, "feature_importances_"):
                feature_values = row.values
                importance = self.model.feature_importances_

                contributions = feature_values * importance
                top_positive_idx = np.argsort(contributions)[-3:][::-1]
                top_negative_idx = np.argsort(contributions)[:3]

                top_positive = [
                    (self.feature_names_[i], float(feature_values[i] * importance[i]))
                    for i in top_positive_idx
                ]
                top_negative = [
                    (self.feature_names_[i], float(feature_values[i] * importance[i]))
                    for i in top_negative_idx
                ]
            else:
                top_positive = []
                top_negative = []

            pred = PatternPrediction(
                timestamp=timestamp,
                pattern_name=pattern_name,
                direction=direction,
                probability_profitable=proba,
                confidence=float(scored.loc[idx, "confidence"]),
                is_recommended=bool(scored.loc[idx, "is_recommended"]),
                top_positive_features=top_positive,
                top_negative_features=top_negative,
            )
            predictions.append(pred)

        return predictions

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N most important features.

        Args:
            top_n: Number of features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if not hasattr(self.model, "feature_importances_"):
            return pd.DataFrame()

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names_,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        return importance_df.head(top_n)

    def walk_forward_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_size: int = 500,
        step_size: int = 100,
    ) -> Dict[str, List[float]]:
        """
        Walk-forward validation for pattern classifier.

        Args:
            X: Feature matrix
            y: Labels series
            train_size: Initial training window
            step_size: Step for expanding window

        Returns:
            Dict with train/test AUC and accuracy lists
        """
        from sklearn.metrics import accuracy_score, roc_auc_score

        results = {
            "train_auc": [],
            "test_auc": [],
            "train_accuracy": [],
            "test_accuracy": [],
        }

        start = 0
        while start + train_size < len(X):
            end = min(start + train_size + step_size, len(X))

            X_train = X.iloc[start : start + train_size]
            y_train = y.iloc[start : start + train_size]
            X_test = X.iloc[start + train_size : end]
            y_test = y.iloc[start + train_size : end]

            if len(X_test) == 0 or len(X_train) < 100:
                start += step_size
                continue

            valid_train = X_train.notna().all(axis=1) & y_train.notna()
            valid_test = X_test.notna().all(axis=1) & y_test.notna()

            if valid_train.sum() < 50 or valid_test.sum() < 10:
                start += step_size
                continue

            model = self._create_model()
            model.fit(X_train[valid_train], y_train[valid_train])

            train_proba = model.predict_proba(X_train[valid_train])[:, 1]
            test_proba = model.predict_proba(X_test[valid_test])[:, 1]

            train_pred = (train_proba >= 0.5).astype(int)
            test_pred = (test_proba >= 0.5).astype(int)

            try:
                train_auc = roc_auc_score(y_train[valid_train], train_proba)
                test_auc = roc_auc_score(y_test[valid_test], test_proba)
            except Exception:
                train_auc = 0.5
                test_auc = 0.5

            results["train_auc"].append(float(train_auc))
            results["test_auc"].append(float(test_auc))
            results["train_accuracy"].append(
                float(accuracy_score(y_train[valid_train], train_pred))
            )
            results["test_accuracy"].append(float(accuracy_score(y_test[valid_test], test_pred)))

            start += step_size

        return results

    def save(self, path: str | Path) -> None:
        """Save trained model to disk."""
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "feature_names": self.feature_names_,
            "calibration_slope": self.calibration_slope_,
            "calibration_intercept": self.calibration_intercept_,
            "config": {
                "model_type": self.model_type,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "min_child_samples": self.min_child_samples,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
            },
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    def load(self, path: str | Path) -> None:
        """Load trained model from disk."""
        import pickle

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.feature_names_ = model_data["feature_names"]
        self.calibration_slope_ = model_data.get("calibration_slope")
        self.calibration_intercept_ = model_data.get("calibration_intercept")

        config = model_data.get("config", {})
        self.model_type = config.get("model_type", self.model_type)
        self.n_estimators = config.get("n_estimators", self.n_estimators)
        self.max_depth = config.get("max_depth", self.max_depth)
        self.learning_rate = config.get("learning_rate", self.learning_rate)
        self.min_child_samples = config.get("min_child_samples", self.min_child_samples)
        self.subsample = config.get("subsample", self.subsample)
        self.colsample_bytree = config.get("colsample_bytree", self.colsample_bytree)
        self.random_state = config.get("random_state", self.random_state)
        self.n_jobs = config.get("n_jobs", self.n_jobs)
