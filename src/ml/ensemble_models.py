"""
Ensemble Methods for Financial ML.

Stacking, Voting, and Blending ensembles combining CatBoost, LightGBM, and
Random Forest for classification and regression tasks. All methods integrate
with PurgedKFold for time-series-safe cross-validation.

Ensemble strategies:
  - Voting: Soft (probability averaging) or hard (majority) voting
  - Stacking: Meta-learner trained on base model out-of-fold predictions
  - Blending: Weighted average of base model predictions with PurgedKFold OOF

Usage:
    from src.ml.ensemble_models import EnsembleBuilder

    builder = EnsembleBuilder(method="stacking", task="classification")
    builder.add_catboost()
    builder.add_lightgbm()
    builder.add_rf()
    result = builder.fit(X, y)
    preds = builder.predict(X_test)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
)


class EnsembleMethod(Enum):
    """Ensemble combination strategies."""

    VOTING_SOFT = "voting_soft"
    VOTING_HARD = "voting_hard"
    STACKING = "stacking"
    BLENDING = "blending"


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class BaseModelConfig:
    """Configuration for a base ensemble model.

    Attributes:
        name: Model identifier (e.g., 'catboost', 'lightgbm', 'rf').
        model: Pre-built sklearn-compatible model instance.
        weight: Ensemble weight (for blending/voting).
        params: Model hyperparameters dict.
    """

    name: str
    model: Any
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleResult:
    """Results from ensemble training and evaluation.

    Attributes:
        method: Ensemble method used.
        task: Classification or regression.
        n_models: Number of base models.
        model_names: List of base model names.
        metrics: Dict of metric name -> value for the ensemble.
        individual_metrics: Dict of model name -> Dict of metrics.
        feature_importance: Aggregated feature importance (where available).
        n_train: Number of training samples.
        oof_predictions: Out-of-fold predictions (blending only).
    """

    method: str
    task: str
    n_models: int
    model_names: List[str]
    metrics: Dict[str, float]
    individual_metrics: Dict[str, Dict[str, float]]
    feature_importance: Dict[str, float]
    n_train: int
    oof_predictions: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "task": self.task,
            "n_models": self.n_models,
            "model_names": self.model_names,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "individual_metrics": {
                name: {k: round(v, 4) for k, v in m.items()}
                for name, m in self.individual_metrics.items()
            },
            "n_train": self.n_train,
        }

    def __repr__(self) -> str:
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return (
            f"EnsembleResult(method={self.method}, task={self.task}, "
            f"models={self.n_models}, {metric_str})"
        )


class EnsembleBuilder:
    """Build and train ensembles of CatBoost, LightGBM, and Random Forest.

    Supports four ensemble methods:
        - voting_soft: Average predicted probabilities (classification) or values (regression)
        - voting_hard: Majority vote (classification only)
        - stacking: Train a meta-learner on base model out-of-fold predictions
        - blending: Weighted average of out-of-fold predictions

    All methods use PurgedKFold for out-of-fold prediction generation to
    prevent data leakage in time-series contexts.

    Example:
        >>> builder = EnsembleBuilder(method="stacking", task="classification")
        >>> builder.add_catboost(iterations=300)
        >>> builder.add_lightgbm(n_estimators=300)
        >>> builder.add_rf(n_estimators=200)
        >>> result = builder.fit(X_train, y_train)
        >>> probs = builder.predict_proba(X_test)
        >>> preds = builder.predict(X_test)
    """

    def __init__(
        self,
        method: str = "stacking",
        task: str = "classification",
        n_splits: int = 5,
        pct_embargo: float = 0.05,
        label_span: int = 5,
        meta_learner: str = "catboost",
        random_state: int = 42,
        verbose: bool = False,
    ):
        """Initialize ensemble builder.

        Args:
            method: Ensemble strategy (voting_soft, voting_hard, stacking, blending).
            task: "classification" or "regression".
            n_splits: Number of PurgedKFold splits for OOF predictions.
            pct_embargo: Embargo fraction for PurgedKFold.
            label_span: Label computation window for PurgedKFold.
            meta_learner: Meta-learner type for stacking ("catboost", "lightgbm", "rf", "lr").
            random_state: Random seed.
            verbose: Print training progress.
        """
        try:
            self.method = EnsembleMethod(method)
        except ValueError:
            raise ValueError(
                f"Unknown method: {method}. Available: {[e.value for e in EnsembleMethod]}"
            )
        try:
            self.task = TaskType(task)
        except ValueError:
            raise ValueError(f"Unknown task: {task}. Available: {[e.value for e in TaskType]}")

        self.n_splits = n_splits
        self.pct_embargo = pct_embargo
        self.label_span = label_span
        self.meta_learner_type = meta_learner
        self.random_state = random_state
        self.verbose = verbose

        self._models: List[BaseModelConfig] = []
        self._meta_model: Any = None
        self._base_models_trained: List[Any] = []
        self._blend_weights: Optional[np.ndarray] = None
        self._oof_predictions: Optional[np.ndarray] = None
        self._is_fitted: bool = False
        self._feature_names: Optional[List[str]] = None

        self._is_classification = task == "classification"
        if self.method == EnsembleMethod.VOTING_HARD and not self._is_classification:
            raise ValueError("Hard voting only supported for classification tasks.")

    def add_catboost(
        self,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        weight: float = 1.0,
        **kwargs,
    ) -> "EnsembleBuilder":
        """Add CatBoost base model.

        Args:
            iterations: Number of trees.
            depth: Tree depth.
            learning_rate: Learning rate.
            l2_leaf_reg: L2 regularization.
            weight: Ensemble weight.
            **kwargs: Additional CatBoost params.
        """
        if self._is_classification:
            from catboost import CatBoostClassifier

            model = CatBoostClassifier(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                random_seed=self.random_state,
                verbose=False,
                loss_function="Logloss",
                task_type="CPU",
                **kwargs,
            )
        else:
            from catboost import CatBoostRegressor

            model = CatBoostRegressor(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                random_seed=self.random_state,
                verbose=False,
                loss_function="RMSE",
                task_type="CPU",
                **kwargs,
            )

        self._models.append(
            BaseModelConfig(
                name="catboost",
                model=model,
                weight=weight,
                params={
                    "iterations": iterations,
                    "depth": depth,
                    "learning_rate": learning_rate,
                    "l2_leaf_reg": l2_leaf_reg,
                },
            )
        )
        return self

    def add_lightgbm(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        weight: float = 1.0,
        **kwargs,
    ) -> "EnsembleBuilder":
        """Add LightGBM base model.

        Args:
            n_estimators: Number of trees.
            max_depth: Tree depth.
            learning_rate: Learning rate.
            num_leaves: Number of leaves.
            weight: Ensemble weight.
            **kwargs: Additional LightGBM params.
        """
        if self._is_classification:
            from lightgbm import LGBMClassifier

            model = LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                random_state=self.random_state,
                verbose=-1,
                **kwargs,
            )
        else:
            from lightgbm import LGBMRegressor

            model = LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                random_state=self.random_state,
                verbose=-1,
                **kwargs,
            )

        self._models.append(
            BaseModelConfig(
                name="lightgbm",
                model=model,
                weight=weight,
                params={
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "num_leaves": num_leaves,
                },
            )
        )
        return self

    def add_rf(
        self,
        n_estimators: int = 200,
        max_depth: int = 8,
        min_samples_leaf: int = 10,
        weight: float = 1.0,
        **kwargs,
    ) -> "EnsembleBuilder":
        """Add Random Forest base model.

        Args:
            n_estimators: Number of trees.
            max_depth: Tree depth.
            min_samples_leaf: Minimum samples per leaf.
            weight: Ensemble weight.
            **kwargs: Additional RF params.
        """
        if self._is_classification:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
                **kwargs,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
                **kwargs,
            )

        self._models.append(
            BaseModelConfig(
                name="rf",
                model=model,
                weight=weight,
                params={
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                },
            )
        )
        return self

    def add_model(
        self,
        name: str,
        model: Any,
        weight: float = 1.0,
        params: Optional[Dict[str, Any]] = None,
    ) -> "EnsembleBuilder":
        """Add a custom base model.

        Args:
            name: Model identifier.
            model: sklearn-compatible model instance.
            weight: Ensemble weight.
            params: Model hyperparameters dict.

        Returns:
            Self for chaining.
        """
        self._models.append(
            BaseModelConfig(name=name, model=model, weight=weight, params=params or {})
        )
        return self

    def _generate_oof_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
    ) -> np.ndarray:
        """Generate out-of-fold predictions using PurgedKFold.

        Args:
            X: Feature array.
            y: Target array.
            model: Untrained model (will be cloned per fold).

        Returns:
            Array of OOF predictions of shape (n_samples, 1) for regression
            or (n_samples, n_classes) for classification.
        """
        from .purged_cv import PurgedKFold

        cv = PurgedKFold(
            n_splits=self.n_splits,
            pct_embargo=self.pct_embargo,
            label_span=self.label_span,
        )

        if self._is_classification:
            n_classes = len(np.unique(y))
            oof = np.zeros((len(X), n_classes), dtype=np.float64)
        else:
            oof = np.zeros(len(X), dtype=np.float64)
            n_classes = 1

        for train_idx, test_idx in cv.split(X):
            fold_model = clone(model)
            fold_model.fit(X[train_idx], y[train_idx])

            if self._is_classification:
                oof[test_idx] = fold_model.predict_proba(X[test_idx])
            else:
                oof[test_idx] = fold_model.predict(X[test_idx])

        return oof

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> EnsembleResult:
        """Fit the ensemble.

        For stacking and blending, use PurgedKFold to generate OOF predictions.
        For voting, train each base model on full data.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            EnsembleResult with metrics.
        """
        if len(self._models) < 2:
            raise ValueError(f"Need at least 2 base models. Currently: {len(self._models)}")

        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        y_arr = np.asarray(y)

        self._is_fitted = True  # must be set before _evaluate calls predict()

        if self.method in (EnsembleMethod.STACKING, EnsembleMethod.BLENDING):
            result = self._fit_with_oof(X_arr, y_arr)
        else:
            result = self._fit_voting(X_arr, y_arr)

        return result

    def _fit_with_oof(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> EnsembleResult:
        """Fit ensemble using out-of-fold predictions (stacking/blending)."""
        n_models = len(self._models)
        n_classes = len(np.unique(y)) if self._is_classification else 1

        if self._is_classification:
            oof_list = np.zeros((len(X), n_models, n_classes), dtype=np.float64)
        else:
            oof_list = np.zeros((len(X), n_models), dtype=np.float64)

        if self.verbose:
            print(f"Generating OOF predictions for {n_models} models...")

        for m_idx, config in enumerate(self._models):
            if self.verbose:
                print(f"  {config.name}...")
            oof_list[..., m_idx] = self._generate_oof_predictions(X, y, config.model)

        # Train base models on full data
        self._base_models_trained = []
        for config in self._models:
            model = clone(config.model)
            model.fit(X, y)
            self._base_models_trained.append((config.name, model))

        if self.method == EnsembleMethod.STACKING:
            return self._fit_stacking(X, y, oof_list)
        else:
            return self._fit_blending(X, y, oof_list)

    def _fit_stacking(
        self,
        X: np.ndarray,
        y: np.ndarray,
        oof_list: np.ndarray,
    ) -> EnsembleResult:
        """Train meta-learner on stacked OOF predictions."""
        n_models = len(self._models)

        if self._is_classification:
            n_classes = oof_list.shape[2]
            meta_X = oof_list.reshape(len(X), n_models * n_classes)
        else:
            meta_X = oof_list

        self._meta_model = self._create_meta_learner()
        self._meta_model.fit(meta_X, y)
        self._oof_predictions = oof_list

        return self._evaluate(X, y)

    def _fit_blending(
        self,
        X: np.ndarray,
        y: np.ndarray,
        oof_list: np.ndarray,
    ) -> EnsembleResult:
        """Compute optimal blend weights from OOF predictions."""
        n_models = len(self._models)

        if self._is_classification:
            # For classification, optimize weights for log loss or accuracy
            best_score = float("inf")
            best_weights = np.ones(n_models) / n_models

            for _ in range(50):
                weights = np.random.dirichlet(np.ones(n_models))
                blend = np.sum(oof_list * weights.reshape(1, n_models, 1), axis=1)
                if len(np.unique(y)) > 1:
                    score = log_loss(y, blend)
                else:
                    score = 0.0
                if score < best_score:
                    best_score = score
                    best_weights = weights
            self._blend_weights = best_weights
        else:
            # For regression, optimize for MSE
            from scipy.optimize import minimize

            def _mse(weights: np.ndarray) -> float:
                scaled = weights / np.sum(weights)
                blend = np.sum(oof_list * scaled.reshape(1, -1), axis=1)
                return float(mean_squared_error(y, blend))

            x0 = np.ones(n_models) / n_models
            bounds = [(0.01, 0.99)] * n_models
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
            result = minimize(_mse, x0, bounds=bounds, constraints=constraints, method="SLSQP")
            self._blend_weights = result.x / np.sum(result.x)

        self._oof_predictions = oof_list

        if self.verbose:
            for i, config in enumerate(self._models):
                print(f"  {config.name}: weight={self._blend_weights[i]:.3f}")

        return self._evaluate(X, y)

    def _fit_voting(self, X: np.ndarray, y: np.ndarray) -> EnsembleResult:
        """Fit base models on full data for voting."""
        self._base_models_trained = []
        for config in self._models:
            model = clone(config.model)
            model.fit(X, y)
            self._base_models_trained.append((config.name, model))
        return self._evaluate(X, y)

    def _create_meta_learner(self) -> Any:
        """Create meta-learner for stacking."""
        if self.meta_learner_type == "catboost":
            if self._is_classification:
                from catboost import CatBoostClassifier

                return CatBoostClassifier(
                    iterations=200,
                    depth=4,
                    learning_rate=0.03,
                    random_seed=self.random_state,
                    verbose=False,
                    task_type="CPU",
                )
            else:
                from catboost import CatBoostRegressor

                return CatBoostRegressor(
                    iterations=200,
                    depth=4,
                    learning_rate=0.03,
                    random_seed=self.random_state,
                    verbose=False,
                    task_type="CPU",
                )
        elif self.meta_learner_type == "lightgbm":
            if self._is_classification:
                from lightgbm import LGBMClassifier

                return LGBMClassifier(
                    n_estimators=100,
                    max_depth=3,
                    random_state=self.random_state,
                    verbose=-1,
                )
            else:
                from lightgbm import LGBMRegressor

                return LGBMRegressor(
                    n_estimators=100,
                    max_depth=3,
                    random_state=self.random_state,
                    verbose=-1,
                )
        elif self.meta_learner_type == "rf":
            if self._is_classification:
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=4,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
            else:
                return RandomForestRegressor(
                    n_estimators=100,
                    max_depth=4,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
        elif self.meta_learner_type == "lr":
            from sklearn.linear_model import LogisticRegression, Ridge

            if self._is_classification:
                return LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_state)
            else:
                return Ridge(alpha=1.0, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown meta_learner: {self.meta_learner_type}")

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate ensemble predictions.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels (classification) or values (regression).
        """
        if not self._is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        X_arr = self._preprocess_X(X)

        if self._is_classification:
            probs = self._predict_proba(X_arr)
            n_classes = probs.shape[1]
            if n_classes == 2:
                return (probs[:, 1] >= 0.5).astype(int)
            else:
                return np.argmax(probs, axis=1)
        else:
            return self._predict_regression(X_arr)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate class probability predictions (classification only)."""
        if not self._is_classification:
            raise ValueError("predict_proba() only available for classification tasks.")
        if not self._is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        X_arr = self._preprocess_X(X)
        return self._predict_proba(X_arr)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Internal probability prediction."""
        if self.method == EnsembleMethod.VOTING_SOFT:
            return self._voting_soft_proba(X)
        elif self.method == EnsembleMethod.VOTING_HARD:
            proba_list = [model.predict_proba(X) for _, model in self._base_models_trained]
            n_classes = proba_list[0].shape[1]
            votes = np.zeros((len(X), n_classes))
            for proba in proba_list:
                hard = np.eye(n_classes)[np.argmax(proba, axis=1)]
                votes += hard / len(proba_list)
            return votes
        elif self.method == EnsembleMethod.STACKING:
            base_probas = self._get_base_probas(X)
            n_models = len(self._models)
            n_classes = base_probas[0].shape[1]
            meta_X = np.hstack([p for p in base_probas]).reshape(len(X), n_models * n_classes)
            return self._meta_model.predict_proba(meta_X)
        elif self.method == EnsembleMethod.BLENDING:
            return self._blend_proba(X)

    def _predict_regression(self, X: np.ndarray) -> np.ndarray:
        """Internal regression prediction."""
        if self.method in (EnsembleMethod.VOTING_SOFT, EnsembleMethod.VOTING_HARD):
            preds = np.column_stack([model.predict(X) for _, model in self._base_models_trained])
            weights = np.array([c.weight for c in self._models])
            return np.average(preds, axis=1, weights=weights)

        elif self.method == EnsembleMethod.STACKING:
            base_preds = np.column_stack(
                [model.predict(X) for _, model in self._base_models_trained]
            )
            return self._meta_model.predict(base_preds)

        elif self.method == EnsembleMethod.BLENDING:
            base_preds = np.column_stack(
                [model.predict(X) for _, model in self._base_models_trained]
            )
            return np.sum(base_preds * self._blend_weights.reshape(1, -1), axis=1)

    def _get_base_probas(self, X: np.ndarray) -> List[np.ndarray]:
        """Get probability predictions from all base models."""
        return [model.predict_proba(X) for _, model in self._base_models_trained]

    def _voting_soft_proba(self, X: np.ndarray) -> np.ndarray:
        """Soft voting: average probabilities."""
        base_probas = self._get_base_probas(X)
        weights = np.array([c.weight for c in self._models])
        weights = weights / np.sum(weights)
        return np.average(np.stack(base_probas, axis=0), axis=0, weights=weights)

    def _blend_proba(self, X: np.ndarray) -> np.ndarray:
        """Blending: weighted average of probabilities."""
        base_probas = self._get_base_probas(X)
        weights = self._blend_weights / np.sum(self._blend_weights)
        return np.sum(np.stack(base_probas, axis=0) * weights.reshape(-1, 1, 1), axis=0)

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> EnsembleResult:
        """Compute ensemble and individual model metrics."""
        ensemble_metrics = self._compute_metrics(X, y, ensemble=True)
        individual_metrics = {}

        for name, model in self._base_models_trained:
            individual_metrics[name] = self._compute_model_metrics(X, y, model, name)

        feat_imp = self._compute_feature_importance()

        return EnsembleResult(
            method=self.method.value,
            task=self.task.value,
            n_models=len(self._models),
            model_names=[c.name for c in self._models],
            metrics=ensemble_metrics,
            individual_metrics=individual_metrics,
            feature_importance=feat_imp,
            n_train=len(X),
            oof_predictions=self._oof_predictions,
        )

    def _compute_metrics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ensemble: bool = True,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        if ensemble:
            if self._is_classification:
                preds = self.predict(X)
                probs = self.predict_proba(X)
            else:
                preds = self.predict(X)
        else:
            raise ValueError("Use _compute_model_metrics for individual models.")

        if self._is_classification:
            metrics = {
                "accuracy": accuracy_score(y, preds),
                "precision": precision_score(y, preds, average="weighted", zero_division=0),
                "recall": recall_score(y, preds, average="weighted", zero_division=0),
                "f1": f1_score(y, preds, average="weighted", zero_division=0),
            }
            if len(np.unique(y)) > 1:
                if probs.shape[1] == 2:
                    metrics["roc_auc"] = roc_auc_score(y, probs[:, 1])
                else:
                    metrics["roc_auc"] = roc_auc_score(
                        y, probs, multi_class="ovr", average="weighted"
                    )
            return metrics
        else:
            return {
                "rmse": float(np.sqrt(mean_squared_error(y, preds))),
                "mae": float(mean_absolute_error(y, preds)),
                "r2": float(r2_score(y, preds)),
            }

    def _compute_model_metrics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
        name: str,
    ) -> Dict[str, float]:
        """Compute metrics for an individual model."""
        if self._is_classification:
            preds = model.predict(X)
            probs = model.predict_proba(X)
            metrics = {
                "accuracy": accuracy_score(y, preds),
                "precision": precision_score(y, preds, average="weighted", zero_division=0),
                "recall": recall_score(y, preds, average="weighted", zero_division=0),
                "f1": f1_score(y, preds, average="weighted", zero_division=0),
            }
            if len(np.unique(y)) > 1:
                if probs.shape[1] == 2:
                    metrics["roc_auc"] = roc_auc_score(y, probs[:, 1])
                else:
                    metrics["roc_auc"] = roc_auc_score(
                        y, probs, multi_class="ovr", average="weighted"
                    )
            return metrics
        else:
            preds = model.predict(X)
            return {
                "rmse": float(np.sqrt(mean_squared_error(y, preds))),
                "mae": float(mean_absolute_error(y, preds)),
                "r2": float(r2_score(y, preds)),
            }

    def _compute_feature_importance(self) -> Dict[str, float]:
        """Aggregate feature importance across models."""
        if not self._feature_names:
            return {}

        importance: Dict[str, float] = {}
        count: Dict[str, int] = {}

        for name, model in self._base_models_trained:
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                for feat, val in zip(self._feature_names, imp):
                    importance[feat] = importance.get(feat, 0.0) + float(val)
                    count[feat] = count.get(feat, 0) + 1

        if not importance:
            return {}

        avg_imp = {k: v / count[k] for k, v in importance.items()}
        return dict(sorted(avg_imp.items(), key=lambda x: x[1], reverse=True))

    def _preprocess_X(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Convert X to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float64)
        return np.asarray(X, dtype=np.float64)

    def save(self, path: str | Path) -> None:
        """Save ensemble to disk."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "base_models_trained": self._base_models_trained,
                "meta_model": self._meta_model,
                "blend_weights": self._blend_weights,
                "models_config": [
                    {
                        "name": c.name,
                        "weight": c.weight,
                        "params": c.params,
                    }
                    for c in self._models
                ],
                "feature_names": self._feature_names,
                "config": {
                    "method": self.method.value,
                    "task": self.task.value,
                    "meta_learner_type": self.meta_learner_type,
                    "random_state": self.random_state,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleBuilder":
        """Load ensemble from disk."""
        import joblib

        data = joblib.load(path)
        config = data["config"]
        instance = cls(
            method=config["method"],
            task=config["task"],
            meta_learner=config["meta_learner_type"],
            random_state=config["random_state"],
        )
        instance._base_models_trained = data["base_models_trained"]
        instance._meta_model = data.get("meta_model")
        instance._blend_weights = data.get("blend_weights")
        instance._feature_names = data.get("feature_names")
        instance._is_fitted = True

        # Reconstruct model configs
        for mc in data.get("models_config", []):
            instance._models.append(
                BaseModelConfig(
                    name=mc["name"],
                    model=None,
                    weight=mc["weight"],
                    params=mc["params"],
                )
            )

        return instance

    def compare_to_individual(self, result: EnsembleResult) -> pd.DataFrame:
        """Generate comparison table: ensemble vs individual models.

        Args:
            result: EnsembleResult from fit().

        Returns:
            DataFrame with ensemble and individual model metrics.
        """
        rows = [{"model": "ensemble", **result.metrics}]
        for name, metrics in result.individual_metrics.items():
            rows.append({"model": name, **metrics})
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "untrained"
        names = [c.name for c in self._models]
        return (
            f"EnsembleBuilder(method={self.method.value}, task={self.task.value}, "
            f"models={names}, status={status})"
        )
