"""Dream Team Stacking Ensemble: CatBoost + LightGBM with GWO weight tuning.

Literature-validated combination of CatBoost (ordered boosting for time series)
and LightGBM (GOSS + EFB for speed/accuracy). Reported R2 of 0.815 vs 0.788
single-model in financial time series benchmarks.

Architecture:
  - Base models: CatBoost + LightGBM (both with PurgedKFold OOF predictions)
  - Meta-learner: Logistic regression (classification) or Ridge (regression)
  - GWO optimizer for ensemble weight optimization (from src/ml/tuning/gwo_tuner.py)
  - PurgedKFold CV for unbiased OOF generation

Builds on EnsembleBuilder pattern from src/ml/ensemble_models.py.

Usage:
    >>> ensemble = DreamTeamEnsemble(task="classification")
    >>> result = ensemble.train(X, y)
    >>> preds = ensemble.predict(X_test)
    >>> proba = ensemble.predict_proba(X_test)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DreamTeamConfig:
    """Configuration for Dream Team ensemble.

    Attributes:
        catboost_iterations: CatBoost tree count.
        catboost_depth: CatBoost tree depth.
        lightgbm_n_estimators: LightGBM tree count.
        lightgbm_num_leaves: LightGBM leaf count.
        learning_rate: Shared learning rate.
        n_splits: PurgedKFold splits for OOF predictions.
        pct_embargo: PurgedKFold embargo fraction.
        meta_learner: Meta-learner type ("lr", "catboost", "lightgbm").
        random_state: Random seed.
        n_jobs: Parallel workers.
    """

    catboost_iterations: int = 500
    catboost_depth: int = 6
    lightgbm_n_estimators: int = 500
    lightgbm_num_leaves: int = 31
    learning_rate: float = 0.05
    l2_leaf_reg: float = 3.0
    n_splits: int = 5
    pct_embargo: float = 0.05
    meta_learner: str = "lr"
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class DreamTeamResult:
    """Results from Dream Team ensemble training.

    Attributes:
        config: DreamTeamConfig used.
        metrics: Ensemble evaluation metrics.
        individual_metrics: Per-base-model metrics.
        catboost_metrics: CatBoost-only metrics.
        lightgbm_metrics: LightGBM-only metrics.
        ensemble_weights: Learned weights for blending.
        feature_importance: Aggregated importance across base models.
        n_train: Training samples.
        improvement_pct: % improvement over best single model.
    """

    config: Dict[str, Any]
    metrics: Dict[str, float]
    individual_metrics: Dict[str, Dict[str, float]]
    catboost_metrics: Dict[str, float]
    lightgbm_metrics: Dict[str, float]
    ensemble_weights: Dict[str, float]
    feature_importance: Dict[str, float]
    n_train: int
    improvement_pct: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "catboost": {k: round(v, 4) for k, v in self.catboost_metrics.items()},
            "lightgbm": {k: round(v, 4) for k, v in self.lightgbm_metrics.items()},
            "weights": {k: round(v, 4) for k, v in self.ensemble_weights.items()},
            "improvement": {k: round(v, 2) for k, v in self.improvement_pct.items()},
            "n_train": self.n_train,
            "config": self.config,
        }

    def compare_table(self) -> pd.DataFrame:
        rows = [
            {"model": "dream_team", **self.metrics},
            {"model": "catboost", **self.catboost_metrics},
            {"model": "lightgbm", **self.lightgbm_metrics},
        ]
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        m = self.metrics
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in m.items())
        return f"DreamTeamResult({metric_str})"


class DreamTeamEnsemble:
    """CatBoost + LightGBM stacking ensemble with GWO weight tuning.

    Combines the two strongest GBDT libraries:
      - CatBoost: Ordered boosting prevents target leakage in time series
      - LightGBM: GOSS sampling + EFB for memory efficiency and speed

    Meta-learner logistic regression provides calibrated probabilities
    with minimal overfitting risk.

    Example:
        >>> dt = DreamTeamEnsemble(task="classification")
        >>> dt.fit(X_train, y_train)
        >>> probs = dt.predict_proba(X_test)
        >>> result = dt.evaluate(X_val, y_val)
        >>> print(result.compare_table())
    """

    def __init__(
        self,
        task: str = "classification",
        config: Optional[DreamTeamConfig] = None,
        tune_weights: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize Dream Team ensemble.

        Args:
            task: "classification" or "regression".
            config: DreamTeamConfig (uses defaults if None).
            tune_weights: Use GWO to optimize ensemble blend weights.
            verbose: Print training progress.
        """
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be 'classification' or 'regression', got '{task}'")
        self._task = task
        self._is_classification = task == "classification"
        self.config = config or DreamTeamConfig()
        self._tune_weights = tune_weights
        self.verbose = verbose
        self._rng = np.random.default_rng(self.config.random_state)

        self._catboost: Any = None
        self._lightgbm: Any = None
        self._meta_model: Any = None
        self._blend_weight_cb: float = 0.5
        self._blend_weight_lgb: float = 0.5
        self._feature_names: List[str] = []
        self._is_fitted: bool = False

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> DreamTeamResult:
        """Fit the Dream Team ensemble.

        Steps:
          1. Generate PurgedKFold OOF predictions from CatBoost + LightGBM
          2. Optionally run GWO to find optimal blend weights
          3. Train meta-learner on stacked OOF predictions
          4. Train base models on full data for final predictions

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            DreamTeamResult with all metrics.
        """
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        n_samples = len(X_arr)
        if n_samples < 100:
            raise ValueError(f"Need >= 100 samples, got {n_samples}")

        oof_cb, oof_lgb = self._generate_oof_predictions(X_arr, y_arr)

        if self._tune_weights:
            self._optimize_weights(oof_cb, oof_lgb, y_arr)

        meta_X = self._build_meta_features(oof_cb, oof_lgb)
        self._meta_model = self._create_meta_learner()
        self._meta_model.fit(meta_X, y_arr)

        self._catboost = self._create_catboost()
        self._catboost.fit(X_arr, y_arr)
        self._lightgbm = self._create_lightgbm()
        self._lightgbm.fit(X_arr, y_arr)

        self._is_fitted = True

        return self._evaluate(X_arr, y_arr)

    def _generate_oof_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate OOF predictions using PurgedKFold for both base models."""
        from sklearn.base import clone

        n = len(X)
        if self._is_classification:
            n_classes = len(np.unique(y))
            oof_cb = np.zeros((n, n_classes), dtype=np.float64)
            oof_lgb = np.zeros((n, n_classes), dtype=np.float64)
        else:
            oof_cb = np.zeros(n, dtype=np.float64)
            oof_lgb = np.zeros(n, dtype=np.float64)

        splits = list(self._make_splits(n))

        if self.verbose:
            print(f"DreamTeam: {len(splits)}-fold PurgedKFold OOF generation...")

        cb_model = self._create_catboost()
        lgb_model = self._create_lightgbm()

        for fold, (train_idx, test_idx) in enumerate(splits):
            if self.verbose:
                print(f"  Fold {fold + 1}/{len(splits)}...")

            cb_fold = clone(cb_model)
            lgb_fold = clone(lgb_model)

            cb_fold.fit(X[train_idx], y[train_idx])
            lgb_fold.fit(X[train_idx], y[train_idx])

            if self._is_classification:
                oof_cb[test_idx] = cb_fold.predict_proba(X[test_idx])
                oof_lgb[test_idx] = lgb_fold.predict_proba(X[test_idx])
            else:
                oof_cb[test_idx] = cb_fold.predict(X[test_idx])
                oof_lgb[test_idx] = lgb_fold.predict(X[test_idx])

        return oof_cb, oof_lgb

    def _make_splits(self, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate PurgedKFold splits respecting time-series ordering."""
        from .purged_cv import PurgedKFold

        cv = PurgedKFold(
            n_splits=self.config.n_splits,
            pct_embargo=self.config.pct_embargo,
        )
        return list(cv.split(np.arange(n)))

    def _build_meta_features(
        self,
        oof_cb: np.ndarray,
        oof_lgb: np.ndarray,
    ) -> np.ndarray:
        """Build meta-feature matrix for stacking."""
        if self._is_classification:
            cb_flat = oof_cb.reshape(len(oof_cb), -1)
            lgb_flat = oof_lgb.reshape(len(oof_lgb), -1)
            diff = oof_cb - oof_lgb
            prod = oof_cb * oof_lgb

            return np.hstack([cb_flat, lgb_flat, diff, prod])
        else:
            diff = (oof_cb - oof_lgb).reshape(-1, 1)
            prod = (oof_cb * oof_lgb).reshape(-1, 1)
            return np.column_stack([oof_cb, oof_lgb, diff.ravel(), prod.ravel()])

    def _optimize_weights(
        self,
        oof_cb: np.ndarray,
        oof_lgb: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Use GWO to find optimal blend weights for CatBoost vs LightGBM."""
        from .tuning.base import OptimizerResult, ParamSpec, SearchSpace
        from .tuning.gwo_tuner import GWOTuner

        space = SearchSpace(
            [
                ParamSpec("cb_weight", "float", 0.1, 0.9),
                ParamSpec("lgb_weight", "float", 0.1, 0.9),
            ],
            seed=self.config.random_state,
        )

        def fitness(params: Dict[str, Any]) -> float:
            w_cb = params["cb_weight"]
            w_lgb = params["lgb_weight"]
            w_sum = w_cb + w_lgb
            w_cb /= w_sum
            w_lgb /= w_sum

            if self._is_classification:
                blend = w_cb * oof_cb + w_lgb * oof_lgb
                preds = np.argmax(blend, axis=1)
                acc = float(np.mean(preds == y))
                return acc
            else:
                blend = w_cb * oof_cb + w_lgb * oof_lgb
                mse = float(np.mean((y - blend) ** 2))
                return -mse  # maximize negative MSE

        tuner = GWOTuner(space, fitness, n_wolves=10, maximize=True, seed=self.config.random_state)
        result: OptimizerResult = tuner.optimize(max_iter=30, early_stop=5)

        w_cb = result.best_params["cb_weight"]
        w_lgb = result.best_params["lgb_weight"]
        w_sum = w_cb + w_lgb
        self._blend_weight_cb = w_cb / w_sum
        self._blend_weight_lgb = w_lgb / w_sum

        if self.verbose:
            print(
                f"DreamTeam GWO weights: CatBoost={self._blend_weight_cb:.3f}, "
                f"LightGBM={self._blend_weight_lgb:.3f}"
            )

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict target labels/values.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels or regression values.
        """
        if not self._is_fitted:
            raise ValueError("Not fitted. Call fit() first.")
        X_arr = self._to_array(X)

        if self._is_classification:
            probs = self.predict_proba(X_arr)
            n_classes = probs.shape[1]
            if n_classes == 2:
                return (probs[:, 1] >= 0.5).astype(np.int64)
            return np.argmax(probs, axis=1)
        else:
            pred_cb = self._catboost.predict(X_arr)
            pred_lgb = self._lightgbm.predict(X_arr)
            return self._blend_weight_cb * pred_cb + self._blend_weight_lgb * pred_lgb

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Uses meta-learner on base model predictions for calibrated output.
        """
        if not self._is_classification:
            raise ValueError("predict_proba() only for classification tasks.")
        if not self._is_fitted:
            raise ValueError("Not fitted. Call fit() first.")
        X_arr = self._to_array(X)

        proba_cb = self._catboost.predict_proba(X_arr)
        proba_lgb = self._lightgbm.predict_proba(X_arr)

        if isinstance(self._meta_model, type(None)):
            return self._blend_weight_cb * proba_cb + self._blend_weight_lgb * proba_lgb

        base_blend = self._blend_weight_cb * proba_cb + self._blend_weight_lgb * proba_lgb

        try:
            cb_flat = proba_cb.reshape(len(proba_cb), -1)
            lgb_flat = proba_lgb.reshape(len(proba_lgb), -1)
            diff = proba_cb - proba_lgb
            prod = proba_cb * proba_lgb
            meta_X_local = np.hstack(
                [cb_flat, lgb_flat, diff.reshape(len(diff), -1), prod.reshape(len(prod), -1)]
            )
            return self._meta_model.predict_proba(meta_X_local)
        except Exception:
            return base_blend

    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> DreamTeamResult:
        """Evaluate the fitted ensemble on holdout data.

        Args:
            X: Test features.
            y: Test labels.

        Returns:
            DreamTeamResult with test metrics.
        """
        if not self._is_fitted:
            raise ValueError("Not fitted. Call fit() first.")
        return self._evaluate(self._to_array(X), np.asarray(y))

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> DreamTeamResult:
        """Compute all evaluation metrics."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            log_loss,
            mean_absolute_error,
            mean_squared_error,
            precision_score,
            r2_score,
            recall_score,
            roc_auc_score,
        )

        cb_pred = self._catboost.predict(X)
        lgb_pred = self._lightgbm.predict(X)
        ensemble_pred = self.predict(X)

        if self._is_classification:
            cb_proba = self._catboost.predict_proba(X)
            lgb_proba = self._lightgbm.predict_proba(X)
            ensemble_proba = self.predict_proba(X)

            cb_metrics = {
                "accuracy": accuracy_score(y, cb_pred),
                "precision": precision_score(y, cb_pred, average="weighted", zero_division=0),
                "recall": recall_score(y, cb_pred, average="weighted", zero_division=0),
                "f1": f1_score(y, cb_pred, average="weighted", zero_division=0),
            }
            lgb_metrics = {
                "accuracy": accuracy_score(y, lgb_pred),
                "precision": precision_score(y, lgb_pred, average="weighted", zero_division=0),
                "recall": recall_score(y, lgb_pred, average="weighted", zero_division=0),
                "f1": f1_score(y, lgb_pred, average="weighted", zero_division=0),
            }
            ensemble_metrics = {
                "accuracy": accuracy_score(y, ensemble_pred),
                "precision": precision_score(y, ensemble_pred, average="weighted", zero_division=0),
                "recall": recall_score(y, ensemble_pred, average="weighted", zero_division=0),
                "f1": f1_score(y, ensemble_pred, average="weighted", zero_division=0),
            }

            if len(np.unique(y)) > 1:
                try:
                    ensemble_metrics["roc_auc"] = roc_auc_score(
                        y, ensemble_proba, multi_class="ovr", average="weighted"
                    )
                    ensemble_metrics["log_loss"] = log_loss(y, ensemble_proba)
                    cb_metrics["roc_auc"] = roc_auc_score(
                        y, cb_proba, multi_class="ovr", average="weighted"
                    )
                    lgb_metrics["roc_auc"] = roc_auc_score(
                        y, lgb_proba, multi_class="ovr", average="weighted"
                    )
                except Exception:
                    pass
        else:
            cb_metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y, cb_pred))),
                "mae": float(mean_absolute_error(y, cb_pred)),
                "r2": float(r2_score(y, cb_pred)),
            }
            lgb_metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y, lgb_pred))),
                "mae": float(mean_absolute_error(y, lgb_pred)),
                "r2": float(r2_score(y, lgb_pred)),
            }
            ensemble_metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y, ensemble_pred))),
                "mae": float(mean_absolute_error(y, ensemble_pred)),
                "r2": float(r2_score(y, ensemble_pred)),
            }

        best_individual = (
            "catboost"
            if cb_metrics.get("f1", cb_metrics.get("r2", 0))
            >= lgb_metrics.get("f1", lgb_metrics.get("r2", 0))
            else "lightgbm"
        )
        best_metrics = cb_metrics if best_individual == "catboost" else lgb_metrics

        improvement: Dict[str, float] = {}
        if self._is_classification:
            metric_keys = ["accuracy", "f1", "roc_auc"]
        else:
            metric_keys = ["rmse", "r2"]
        for k in metric_keys:
            if k in ensemble_metrics and k in best_metrics:
                e_val = ensemble_metrics[k]
                b_val = best_metrics[k]
                if k == "rmse":
                    imp = (b_val - e_val) / max(abs(b_val), 1e-8) * 100
                else:
                    imp = (e_val - b_val) / max(abs(b_val), 1e-8) * 100
                improvement[k] = imp

        feat_imp = self._compute_feature_importance()

        return DreamTeamResult(
            config={
                "task": self._task,
                "catboost_iterations": self.config.catboost_iterations,
                "lightgbm_n_estimators": self.config.lightgbm_n_estimators,
                "n_splits": self.config.n_splits,
            },
            metrics=ensemble_metrics,
            individual_metrics={
                "catboost": cb_metrics,
                "lightgbm": lgb_metrics,
            },
            catboost_metrics=cb_metrics,
            lightgbm_metrics=lgb_metrics,
            ensemble_weights={
                "catboost": self._blend_weight_cb,
                "lightgbm": self._blend_weight_lgb,
            },
            feature_importance=feat_imp,
            n_train=len(X),
            improvement_pct=improvement,
        )

    def _create_catboost(self) -> Any:
        if self._is_classification:
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                iterations=self.config.catboost_iterations,
                depth=self.config.catboost_depth,
                learning_rate=self.config.learning_rate,
                l2_leaf_reg=self.config.l2_leaf_reg,
                random_seed=self.config.random_state,
                verbose=False,
                loss_function="Logloss",
                task_type="CPU",
                thread_count=self.config.n_jobs,
            )
        else:
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                iterations=self.config.catboost_iterations,
                depth=self.config.catboost_depth,
                learning_rate=self.config.learning_rate,
                l2_leaf_reg=self.config.l2_leaf_reg,
                random_seed=self.config.random_state,
                verbose=False,
                loss_function="RMSE",
                task_type="CPU",
                thread_count=self.config.n_jobs,
            )

    def _create_lightgbm(self) -> Any:
        if self._is_classification:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=self.config.lightgbm_n_estimators,
                num_leaves=self.config.lightgbm_num_leaves,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
                verbose=-1,
                n_jobs=self.config.n_jobs,
            )
        else:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=self.config.lightgbm_n_estimators,
                num_leaves=self.config.lightgbm_num_leaves,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
                verbose=-1,
                n_jobs=self.config.n_jobs,
            )

    def _create_meta_learner(self) -> Any:
        if self.config.meta_learner == "lr":
            if self._is_classification:
                from sklearn.linear_model import LogisticRegression

                return LogisticRegression(
                    C=1.0, max_iter=2000, random_state=self.config.random_state
                )
            else:
                from sklearn.linear_model import Ridge

                return Ridge(alpha=1.0, random_state=self.config.random_state)
        elif self.config.meta_learner == "catboost":
            return self._create_catboost()
        elif self.config.meta_learner == "lightgbm":
            return self._create_lightgbm()
        else:
            raise ValueError(f"Unknown meta_learner: {self.config.meta_learner}")

    def _compute_feature_importance(self) -> Dict[str, float]:
        if not self._feature_names or not self._is_fitted:
            return {}

        importance: Dict[str, float] = {}
        count: Dict[str, int] = {}

        for model in (self._catboost, self._lightgbm):
            if hasattr(model, "feature_importances_"):
                for feat, val in zip(self._feature_names, model.feature_importances_):
                    importance[feat] = importance.get(feat, 0.0) + float(val)
                    count[feat] = count.get(feat, 0) + 1

        if not importance:
            return {}

        avg_imp = {k: v / count[k] for k, v in importance.items()}
        return dict(sorted(avg_imp.items(), key=lambda x: x[1], reverse=True))

    def _to_array(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float64)
        return np.asarray(X, dtype=np.float64)

    def save(self, path: str) -> None:
        """Save ensemble to disk."""
        import pickle

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "catboost": self._catboost,
            "lightgbm": self._lightgbm,
            "meta_model": self._meta_model,
            "blend_weight_cb": self._blend_weight_cb,
            "blend_weight_lgb": self._blend_weight_lgb,
            "feature_names": self._feature_names,
            "config": self.config,
            "task": self._task,
        }
        with open(out, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load ensemble from disk."""
        import pickle

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")

        with open(p, "rb") as f:
            data = pickle.load(f)

        self._catboost = data["catboost"]
        self._lightgbm = data["lightgbm"]
        self._meta_model = data.get("meta_model")
        self._blend_weight_cb = data.get("blend_weight_cb", 0.5)
        self._blend_weight_lgb = data.get("blend_weight_lgb", 0.5)
        self._feature_names = data.get("feature_names", [])
        self.config = data.get("config", DreamTeamConfig())
        self._task = data.get("task", "classification")
        self._is_classification = self._task == "classification"
        self._is_fitted = True

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def weights(self) -> Dict[str, float]:
        return {
            "catboost": self._blend_weight_cb,
            "lightgbm": self._blend_weight_lgb,
        }

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "untrained"
        return (
            f"DreamTeamEnsemble(task={self._task}, status={status}, "
            f"cb_weight={self._blend_weight_cb:.2f}, "
            f"lgb_weight={self._blend_weight_lgb:.2f})"
        )
