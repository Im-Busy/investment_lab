"""Optuna-based hyperparameter tuning for CatBoost and LightGBM models.

Replaces GWO/GA/WOA metaheuristics with Bayesian optimization (TPE sampler)
and automated pruning of unpromising trials.

Integrates with:
  - PurgedKFold CV (src/ml/purged_cv.py) for time-series-aware validation
  - MLflow experiment logger for trial-level tracking
  - scripts/tune_model.py CLI via --algo optuna

Usage:
    from src.ml.tuning.optuna_tuner import OptunaTuner

    tuner = OptunaTuner(model_type="catboost", n_trials=50)
    result = tuner.optimize(X, y, target="pattern_classifier")
    print(result.best_params, result.best_score)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import optuna
import pandas as pd

logger = logging.getLogger(__name__)

CATBOOST_OPTUNA_SPACE: Dict[str, dict] = {
    "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
    "max_depth": {"type": "int", "low": 3, "high": 12},
    "l2_leaf_reg": {"type": "float", "low": 0.5, "high": 30.0, "log": True},
    "random_strength": {"type": "float", "low": 0.1, "high": 5.0},
    "bagging_temperature": {"type": "float", "low": 0.0, "high": 2.0},
    "border_count": {"type": "int", "low": 32, "high": 255},
    "min_data_in_leaf": {"type": "int", "low": 5, "high": 100},
    "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    "colsample_bylevel": {"type": "float", "low": 0.3, "high": 1.0},
}

LIGHTGBM_OPTUNA_SPACE: Dict[str, dict] = {
    "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
    "max_depth": {"type": "int", "low": 3, "high": 15},
    "num_leaves": {"type": "int", "low": 15, "high": 255},
    "lambda_l1": {"type": "float", "low": 0.0, "high": 10.0},
    "lambda_l2": {"type": "float", "low": 0.0, "high": 30.0},
    "min_child_samples": {"type": "int", "low": 5, "high": 100},
    "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.3, "high": 1.0},
    "subsample_freq": {"type": "int", "low": 0, "high": 10},
}


def _suggest_param(trial: optuna.Trial, name: str, spec: dict) -> Any:
    """Suggest a parameter value to Optuna based on the space spec."""
    ptype = spec["type"]
    low = spec["low"]
    high = spec["high"]
    log_scale = spec.get("log", False)

    if ptype == "float":
        return trial.suggest_float(name, low, high, log=log_scale)
    if ptype == "int":
        return trial.suggest_int(name, low, high, log=log_scale)
    return low


@dataclass
class OptunaResult:
    """Result from an Optuna hyperparameter optimization run."""

    best_params: Dict[str, Any]
    best_score: float
    n_trials: int = 0
    study_name: str = ""
    best_trial_number: int = 0
    trial_scores: List[float] = field(default_factory=list)
    study: optuna.Study | None = None


class OptunaTuner:
    """Bayesian hyperparameter optimization via Optuna's TPE sampler.

    Uses PurgedKFold cross-validation for time-series-aware evaluation.
    Supports CatBoost and LightGBM model types with separate search spaces.

    Attributes:
        model_type: "catboost" or "lightgbm".
        n_trials: Number of Optuna trials.
        n_splits: Number of PurgedKFold splits.
        label_span: Forward return horizon for purging.
        pruner: Optuna pruner (MedianPruner by default).
        study_name: Name for the Optuna study.
    """

    def __init__(
        self,
        model_type: str = "catboost",
        n_trials: int = 50,
        n_splits: int = 5,
        label_span: int = 5,
        pruner_patience: int = 10,
        study_name: str = "",
        seed: int = 42,
    ) -> None:
        if model_type not in ("catboost", "lightgbm"):
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'catboost' or 'lightgbm'.")
        self.model_type = model_type
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.label_span = label_span
        self.pruner_patience = pruner_patience
        self.study_name = study_name
        self.seed = seed
        self._study: optuna.Study | None = None
        self._model_class: type | None = None
        self._target: str = ""

    def _objective(
        self,
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        """Optuna objective function: train model with PurgedKFold CV."""
        from src.ml.purged_cv import PurgedKFold

        space = CATBOOST_OPTUNA_SPACE if self.model_type == "catboost" else LIGHTGBM_OPTUNA_SPACE
        params = {name: _suggest_param(trial, name, spec) for name, spec in space.items()}

        cv = PurgedKFold(
            n_splits=self.n_splits,
            pct_embargo=0.01,
            label_span=self.label_span,
        )

        scores: List[float] = []
        for train_idx, test_idx in cv.split(X):
            X_tr = X.iloc[train_idx]
            _ = X.iloc[test_idx]
            y_tr = y.iloc[train_idx]
            y_te = y.iloc[test_idx]

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue

            try:
                model = self._model_class(
                    model_type=self.model_type,
                    n_estimators=200,
                    random_state=self.seed,
                    **params,
                )
                result = model.train(X_tr, y_tr)
                scores.append(result.test_auc)
            except Exception as exc:
                logger.debug(f"Trial {trial.number} fold failed: {exc}")
                return float("-inf")

        if not scores:
            return float("-inf")

        score = float(np.mean(scores))

        for i, s in enumerate(scores):
            trial.set_user_attr(f"fold_{i}_auc", float(s))

        return score

    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target: str = "pattern_classifier",
        model_class: type | None = None,
    ) -> OptunaResult:
        """Run Optuna hyperparameter optimization.

        Args:
            X: Feature DataFrame.
            y: Target series (binary labels).
            target: One of 'pattern_classifier', 'signal_regressor', 'regime'.
            model_class: Model class with .train() returning object with .test_auc.

        Returns:
            OptunaResult with best_params, best_score, trials info.
        """
        from src.ml.pattern_classifier import PatternClassifier
        from src.ml.signal_scorer import SignalRegressor
        from src.ml.regime_model import RegimeClassifier

        target_map: Dict[str, type] = {
            "pattern_classifier": PatternClassifier,
            "signal_regressor": SignalRegressor,
            "regime": RegimeClassifier,
        }
        self._model_class = model_class or target_map.get(target)
        if self._model_class is None:
            raise ValueError(f"Unknown target '{target}'. Use: {list(target_map)}")
        self._target = target

        study_name = self.study_name or f"{target}_{self.model_type}"
        storage = f"sqlite:///optuna_{study_name}.db"

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1,
        )

        self._study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )

        logger.info(
            f"Optuna: tuning {target} ({self.model_type}), "
            f"{self.n_trials} trials, {self.n_splits}-fold PurgedKFold, TPE sampler"
        )

        self._study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        best_score = self._study.best_value
        best_params = self._study.best_params
        trial_scores = [t.value for t in self._study.trials if t.value is not None]

        logger.info(f"Optuna complete: best AUC={best_score:.4f}, {len(trial_scores)} trials")
        logger.info(f"Best params: {best_params}")

        return OptunaResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=len(self._study.trials),
            study_name=study_name,
            best_trial_number=self._study.best_trial.number if self._study.best_trial else 0,
            trial_scores=trial_scores,
            study=self._study,
        )

    def log_to_mlflow(self, result: OptunaResult) -> None:
        """Log Optuna optimization results to MLflow."""
        try:
            import mlflow

            mlflow.log_param("tuner", "optuna")
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("n_trials", result.n_trials)
            mlflow.log_param("best_trial", result.best_trial_number)
            mlflow.log_metric("best_auc", result.best_score)
            for name, value in result.best_params.items():
                mlflow.log_param(f"best_{name}", value)
            logger.info("Optuna results logged to MLflow")
        except Exception as exc:
            logger.debug(f"MLflow logging skipped: {exc}")


def compare_optimizers(
    X: pd.DataFrame,
    y: pd.Series,
    target: str = "pattern_classifier",
    model_type: str = "catboost",
    n_trials: int = 30,
    label_span: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare Optuna TPE vs GWO on the same data.

    Runs both optimizers and returns their best scores + params.

    Returns:
        Dict with 'optuna' and 'gwo' keys, each containing 'score' and 'params'.
    """
    from src.ml.tuning.gwo_tuner import CATBOOST_PARAM_SPACE, GWOTuner
    from src.ml.tuning.base import SearchSpace

    results: Dict[str, Any] = {}

    logger.info(f"=== Optuna TPE ({n_trials} trials) ===")
    optuna_tuner = OptunaTuner(
        model_type=model_type,
        n_trials=n_trials,
        label_span=label_span,
        seed=seed,
        study_name=f"compare_{target}",
    )
    optuna_result = optuna_tuner.optimize(X, y, target=target)
    results["optuna"] = {"score": optuna_result.best_score, "params": optuna_result.best_params}

    logger.info(f"=== GWO (30 wolves, {n_trials} iter) ===")
    from src.ml.pattern_classifier import PatternClassifier
    from src.ml.signal_scorer import SignalRegressor
    from src.ml.regime_model import RegimeClassifier

    target_map: Dict[str, type] = {
        "pattern_classifier": PatternClassifier,
        "signal_regressor": SignalRegressor,
        "regime": RegimeClassifier,
    }
    model_class = target_map[target]

    from src.ml.purged_cv import PurgedKFold

    def fitness_fn(params: Dict[str, Any]) -> float:
        from src.ml.tuning.base import map_params

        mapped = map_params(params, target)
        cv = PurgedKFold(n_splits=3, pct_embargo=0.01, label_span=label_span)
        scores = []
        for train_idx, test_idx in cv.split(X):
            X_tr = X.iloc[train_idx]
            _ = X.iloc[test_idx]
            y_tr = y.iloc[train_idx]
            _ = y.iloc[test_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            try:
                model = model_class(
                    model_type=model_type, n_estimators=150, random_state=seed, **mapped
                )
                result = model.train(X_tr, y_tr)
                scores.append(result.test_auc)
            except Exception:
                return 0.0
        return float(np.mean(scores)) if scores else 0.0

    space = SearchSpace(CATBOOST_PARAM_SPACE)
    gwo = GWOTuner(space, fitness_fn, n_wolves=30, seed=seed)
    gwo_result = gwo.optimize(max_iter=n_trials)
    results["gwo"] = {"score": gwo_result.best_score, "params": gwo_result.best_params}

    logger.info(f"Optuna best AUC: {results['optuna']['score']:.4f}")
    logger.info(f"GWO    best AUC: {results['gwo']['score']:.4f}")
    diff = results["optuna"]["score"] - results["gwo"]["score"]
    logger.info(f"Delta (Optuna - GWO): {diff:+.4f}")
    results["winner"] = "optuna" if diff > 0 else "gwo"

    return results
