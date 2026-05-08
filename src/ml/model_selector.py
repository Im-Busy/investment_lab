"""
Model Selector — intelligent model and validation method selection.

Provides ModelSelector for automatically selecting the best ML model and
validation method based on dataset characteristics, task requirements,
and performance priorities.

Example:
    >>> selector = ModelSelector()
    >>> recommendation = selector.recommend(X, y, task="classification")
    >>> result = selector.train_and_evaluate(X, y, recommendation)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    roc_auc_score,
)


@dataclass
class ModelConfig:
    """Configuration for a specific model type."""

    name: str
    display_name: str
    model_type: str
    supports_multiclass: bool = True
    supports_regression: bool = True
    supports_classification: bool = True
    default_params: Dict[str, Any] = field(default_factory=dict)
    fast: bool = False
    interpretable: bool = True
    typically_accurate: bool = True


@dataclass
class ValidationConfig:
    """Configuration for a validation method."""

    name: str
    display_name: str
    validation_type: str
    requires_time_series: bool = False
    computationally_expensive: bool = False
    min_samples: int = 100


@dataclass
class Recommendation:
    """Model and validation recommendation."""

    model: ModelConfig
    validation: ValidationConfig
    reasoning: str
    estimated_time_seconds: float = 0.0
    confidence: float = 1.0


@dataclass
class ModelResult:
    """Result from training and evaluating a model."""

    model_name: str
    model_type: str
    validation_method: str
    train_score: float
    test_score: float
    overfit_gap: float
    training_time: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    model: Any = None
    feature_importance: Optional[Dict[str, float]] = None


class ModelSelector:
    """
       Intelligent model and validation method selector.

       Analyzes dataset characteristics and recommends the best model

    and validation method based on:
       - Dataset size and dimensionality
       - Task type (classification/regression)
       - Class distribution (for classification)
       - Performance priority (balanced/fast/accurate/interpretable)

       Example:
           >>> selector = ModelSelector()
           >>> rec = selector.recommend(X, y, priority="balanced")
           >>> result = selector.train_and_evaluate(X, y, rec.model, rec.validation)
    """

    MODEL_CONFIGS: Dict[str, ModelConfig] = {
        "catboost": ModelConfig(
            name="catboost",
            display_name="CatBoost",
            model_type="catboost",
            supports_multiclass=True,
            supports_regression=True,
            supports_classification=True,
            default_params={
                "n_estimators": 500,
                "depth": 6,
                "learning_rate": 0.03,
                "l2_leaf_reg": 3.0,
                "random_seed": 42,
            },
            fast=False,
            interpretable=True,
            typically_accurate=True,
        ),
        "chronos": ModelConfig(
            name="chronos",
            display_name="Chronos-2 (Foundation Model)",
            model_type="chronos",
            supports_multiclass=False,
            supports_regression=True,
            supports_classification=False,
            default_params={},
            fast=True,
            interpretable=False,
            typically_accurate=True,
        ),
        "fincast": ModelConfig(
            name="fincast",
            display_name="FinCast (Financial Foundation Model)",
            model_type="fincast",
            supports_multiclass=False,
            supports_regression=True,
            supports_classification=False,
            default_params={},
            fast=True,
            interpretable=False,
            typically_accurate=True,
        ),
        "xlstm": ModelConfig(
            name="xlstm",
            display_name="xLSTM",
            model_type="xlstm",
            supports_multiclass=False,
            supports_regression=True,
            supports_classification=False,
            default_params={
                "hidden_size": 128,
                "num_layers": 4,
            },
            fast=False,
            interpretable=False,
            typically_accurate=True,
        ),
        "logistic_regression": ModelConfig(
            name="logistic_regression",
            display_name="Logistic Regression",
            model_type="logistic_regression",
            supports_multiclass=True,
            supports_regression=False,
            supports_classification=True,
            default_params={"max_iter": 1000, "class_weight": "balanced"},
            fast=True,
            interpretable=True,
            typically_accurate=False,
        ),
    }

    VALIDATION_CONFIGS: Dict[str, ValidationConfig] = {
        "train_test": ValidationConfig(
            name="train_test",
            display_name="Train/Test Split",
            validation_type="train_test",
            requires_time_series=False,
            computationally_expensive=False,
            min_samples=100,
        ),
        "walk_forward": ValidationConfig(
            name="walk_forward",
            display_name="Walk-Forward Validation",
            validation_type="walk_forward",
            requires_time_series=True,
            computationally_expensive=True,
            min_samples=500,
        ),
        "purged_kfold": ValidationConfig(
            name="purged_kfold",
            display_name="Purged K-Fold CV",
            validation_type="purged_kfold",
            requires_time_series=True,
            computationally_expensive=True,
            min_samples=100,
        ),
    }

    def __init__(self, random_state: int = 42):
        """Initialize ModelSelector.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state

    def _has_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def recommend(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str = "classification",
        priority: str = "balanced",
        model_type: Optional[str] = None,
        validation_type: Optional[str] = None,
        is_time_series: bool = False,
    ) -> Recommendation:
        """Recommend model and validation method.

        Args:
            X: Feature matrix
            y: Target variable
            task: 'classification' or 'regression'
            priority: 'balanced', 'fast', 'accurate', 'interpretable'
            model_type: Force specific model type
            validation_type: Force specific validation type
            is_time_series: Whether data is time series

        Returns:
            Recommendation with model, validation, and reasoning
        """
        n_samples, n_features = X.shape
        n_classes = len(y.unique()) if task == "classification" else None

        reasons = []

        if model_type:
            model = self.MODEL_CONFIGS[model_type]
            reasons.append(f"User specified model: {model.display_name}")
        else:
            model = self._select_model(
                n_samples,
                n_features,
                task,
                priority,
                n_classes,
            )

        if validation_type:
            validation = self.VALIDATION_CONFIGS[validation_type]
            reasons.append(f"User specified validation: {validation.display_name}")
        else:
            validation = self._select_validation(
                n_samples,
                is_time_series,
                priority,
            )

        reasoning = "; ".join(reasons)
        estimated_time = self._estimate_time(n_samples, n_features, model, validation)

        return Recommendation(
            model=model,
            validation=validation,
            reasoning=reasoning,
            estimated_time_seconds=estimated_time,
        )

    def _select_model(
        self,
        n_samples: int,
        n_features: int,
        task: str,
        priority: str,
        n_classes: int | None,
    ) -> ModelConfig:
        """Select best model based on characteristics heuristics."""
        suitable_models = []

        for config in self.MODEL_CONFIGS.values():
            if task == "classification" and not config.supports_classification:
                continue
            if task == "regression" and not config.supports_regression:
                continue

            suitable_models.append(config)

        if not suitable_models:
            raise ValueError(f"No suitable models for task: {task}")

        if priority == "fast":
            suitable_models = [m for m in suitable_models if m.fast]
            if suitable_models:
                return suitable_models[0]

        if priority == "accurate":
            accurate_models = [m for m in suitable_models if m.typically_accurate]
            if accurate_models:
                return accurate_models[0]

        if priority == "interpretable":
            return suitable_models[0]

        if n_samples < 1000:
            fast_models = [m for m in suitable_models if m.fast]
            if fast_models:
                return fast_models[0]

        return suitable_models[0]

    def _select_validation(
        self,
        n_samples: int,
        is_time_series: bool,
        priority: str,
    ) -> ValidationConfig:
        """Select best validation method."""
        if priority == "fast":
            return self.VALIDATION_CONFIGS["train_test"]

        if is_time_series and n_samples >= 500:
            return self.VALIDATION_CONFIGS["walk_forward"]

        if is_time_series and n_samples >= 100:
            return self.VALIDATION_CONFIGS["purged_kfold"]

        return self.VALIDATION_CONFIGS["train_test"]

    def _estimate_time(
        self,
        n_samples: int,
        n_features: int,
        model: ModelConfig,
        validation: ValidationConfig,
    ) -> float:
        """Estimate training time in seconds."""
        base_time = n_samples * n_features * 0.0001

        if model.name in ["catboost", "random_forest"]:
            base_time *= 2

        if validation.computationally_expensive:
            base_time *= 3

        return min(base_time, 300)

    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_config: ModelConfig,
        validation_config: ValidationConfig,
        test_size: float = 0.3,
    ) -> ModelResult:
        """Train and evaluate model with specified validation method.

        Args:
            X: Feature matrix
            y: Target variable
            model_config: Model configuration
            validation_config: Validation configuration
            test_size: Test set proportion for train_test split

        Returns:
            ModelResult with metrics and trained model
        """
        start_time = time.time()

        X_clean = X.dropna()
        y_clean = y.reindex(X_clean.index).dropna()

        if len(X_clean) < validation_config.min_samples:
            raise ValueError(
                f"Insufficient samples: {len(X_clean)} < {validation_config.min_samples}"
            )

        if validation_config.validation_type == "train_test":
            result = self._train_test_split(X_clean, y_clean, model_config, test_size)
        elif validation_config.validation_type == "walk_forward":
            result = self._walk_forward_validation(X_clean, y_clean, model_config)
        elif validation_config.validation_type == "purged_kfold":
            result = self._purged_kfold_validation(X_clean, y_clean, model_config)
        else:
            raise ValueError(f"Unknown validation type: {validation_config.validation_type}")

        training_time = time.time() - start_time
        result.training_time = training_time

        return result

    def _train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_config: ModelConfig,
        test_size: float,
    ) -> ModelResult:
        """Train/test split validation."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, shuffle=False
        )

        model = self._create_model(model_config)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_score = self._compute_score(y_train, train_pred, model_config)
        test_score = self._compute_score(y_test, test_pred, model_config)
        overfit_gap = abs(train_score - test_score)

        feature_importance = self._extract_feature_importance(model, X_train.columns)

        return ModelResult(
            model_name=model_config.display_name,
            model_type=model_config.model_type,
            validation_method="Train/Test Split",
            train_score=float(train_score),
            test_score=float(test_score),
            overfit_gap=float(overfit_gap),
            training_time=0.0,
            metrics={"train_samples": len(X_train), "test_samples": len(X_test)},
            model=model,
            feature_importance=feature_importance,
        )

    def _walk_forward_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_config: ModelConfig,
        train_size: int = 500,
        step_size: int = 100,
    ) -> ModelResult:
        """Walk-forward validation."""
        train_scores = []
        test_scores = []

        start = 0
        while start + train_size < len(X):
            end = min(start + train_size + step_size, len(X))

            X_train = X.iloc[start : start + train_size]
            y_train = y.iloc[start : start + train_size]
            X_test = X.iloc[start + train_size : end]
            y_test = y.iloc[start + train_size : end]

            if len(X_test) == 0:
                start += step_size
                continue

            model = self._create_model(model_config)
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_scores.append(self._compute_score(y_train, train_pred, model_config))
            test_scores.append(self._compute_score(y_test, test_pred, model_config))

            start += step_size

        train_score = np.mean(train_scores) if train_scores else 0
        test_score = np.mean(test_scores) if test_scores else 0
        overfit_gap = abs(train_score - test_score)

        final_model = self._create_model(model_config)
        final_model.fit(X, y)
        feature_importance = self._extract_feature_importance(final_model, X.columns)

        return ModelResult(
            model_name=model_config.display_name,
            model_type=model_config.model_type,
            validation_method="Walk-Forward",
            train_score=float(train_score),
            test_score=float(test_score),
            overfit_gap=float(overfit_gap),
            training_time=0.0,
            metrics={"n_folds": len(train_scores)},
            model=final_model,
            feature_importance=feature_importance,
        )

    def _purged_kfold_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_config: ModelConfig,
        n_splits: int = 5,
    ) -> ModelResult:
        """Purged K-Fold cross-validation."""
        try:
            from src.ml.purged_cv import PurgedKFold

            purged_kf = PurgedKFold(n_splits=n_splits)
        except ImportError:
            from sklearn.model_selection import KFold

            purged_kf = KFold(n_splits=n_splits, shuffle=False)

        train_scores = []
        test_scores = []

        for train_idx, test_idx in purged_kf.split(X, y):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            model = self._create_model(model_config)
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_scores.append(self._compute_score(y_train, train_pred, model_config))
            test_scores.append(self._compute_score(y_test, test_pred, model_config))

        train_score = np.mean(train_scores) if train_scores else 0
        test_score = np.mean(test_scores) if test_scores else 0
        overfit_gap = abs(train_score - test_score)

        final_model = self._create_model(model_config)
        final_model.fit(X, y)
        feature_importance = self._extract_feature_importance(final_model, X.columns)

        return ModelResult(
            model_name=model_config.display_name,
            model_type=model_config.model_type,
            validation_method="Purged K-Fold",
            train_score=float(train_score),
            test_score=float(test_score),
            overfit_gap=float(overfit_gap),
            training_time=0.0,
            metrics={"n_folds": len(train_scores)},
            model=final_model,
            feature_importance=feature_importance,
        )

    def _create_model(self, config: ModelConfig) -> Any:
        """Create model instance from configuration."""
        if config.model_type == "catboost":
            try:
                from catboost import CatBoostClassifier

                return CatBoostClassifier(
                    iterations=config.default_params.get("n_estimators", 500),
                    depth=config.default_params.get("depth", 6),
                    learning_rate=config.default_params.get("learning_rate", 0.03),
                    l2_leaf_reg=config.default_params.get("l2_leaf_reg", 3.0),
                    random_seed=config.default_params.get("random_seed", 42),
                    verbose=False,
                    loss_function="Logloss",
                    task_type="GPU" if self._has_gpu() else "CPU",
                )
            except ImportError:
                raise ImportError("catboost not installed. Run: uv add catboost")

        elif config.model_type == "chronos":
            from src.ml.models.chronos import ChronosForecaster

            return ChronosForecaster(model_size="base")

        elif config.model_type == "fincast":
            from src.ml.models.fincast import FinCastForecaster

            return FinCastForecaster.from_zero_shot()

        elif config.model_type == "xlstm":
            from src.ml.models.xlstm import xLSTMForecaster

            return xLSTMForecaster(
                hidden_size=config.default_params.get("hidden_size", 128),
                num_layers=config.default_params.get("num_layers", 4),
            )

        elif config.model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(**config.default_params)

        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

    def _compute_score(self, y_true: pd.Series, y_pred: np.ndarray, config: ModelConfig) -> float:
        """Compute appropriate score metric based on task."""
        try:
            if hasattr(config, "supports_regression") and config.supports_regression:
                if len(y_true.unique()) > 2:
                    return r2_score(y_true, y_pred)
                else:
                    return r2_score(y_true, y_pred)

            if len(y_true.unique()) == 2:
                try:
                    return roc_auc_score(y_true, y_pred)
                except Exception:
                    return accuracy_score(y_true, y_pred)
            else:
                return accuracy_score(y_true, y_pred)

        except Exception:
            return accuracy_score(y_true, y_pred)

    def _extract_feature_importance(
        self, model: Any, feature_names: List[str]
    ) -> Optional[Dict[str, float]]:
        """Extract feature importance if available."""
        if not hasattr(model, "feature_importances_"):
            return None

        importance = model.feature_importances_

        return dict(sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True))

    def compare_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str = "classification",
        validation_config: Optional[ValidationConfig] = None,
    ) -> List[ModelResult]:
        """Compare all suitable models.

        Args:
            X: Feature matrix
            y: Target variable
            task: 'classification' or 'regression'
            validation_config: Validation method to use (uses default if None)

        Returns:
            List of ModelResult for each model
        """
        if validation_config is None:
            validation_config = self.VALIDATION_CONFIGS["train_test"]

        results = []

        for model_config in self.MODEL_CONFIGS.values():
            if task == "classification" and not model_config.supports_classification:
                continue
            if task == "regression" and not model_config.supports_regression:
                continue

            try:
                result = self.train_and_evaluate(X, y, model_config, validation_config)
                results.append(result)
            except Exception:
                continue

        return sorted(results, key=lambda x: x.test_score, reverse=True)
