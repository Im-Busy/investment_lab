"""MLflow experiment tracking wrapper for the existing ExperimentLogger.

Additive layer — pipes all log calls to both the JSONL-based ExperimentLogger
and MLflow's tracking server. Provides a comparison UI and remote tracking
without replacing the existing logging infrastructure.

Usage:
    from src.ml.mlflow_logger import MlflowExperimentLogger

    logger = MlflowExperimentLogger("rf_regime_classifier", tracking_uri="http://localhost:5000")
    logger.log_metadata(model_type="random_forest", task="regime_classification")
    logger.log_config(hyperparams={"n_estimators": 100})
    logger.log_fold_metrics(fold=1, train_metrics={...}, test_metrics={...})
    logger.log_feature_importance(mdi={"atr": 0.15})
    logger.log_model(model)
    logger.finish()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.ml.experiment_logger import ExperimentLogger, EXPERIMENTS_DIR

mlflow_logger = logging.getLogger(__name__)


class MlflowExperimentLogger:
    """MLflow-tracked experiment logger wrapping the existing ExperimentLogger.

    All calls are forwarded to both the JSONL ExperimentLogger (file-based)
    and MLflow (tracking server). MLflow failures are logged but do not
    interrupt the experiment.

    Attributes:
        logger: The underlying ExperimentLogger instance.
        run: MLflow active run context.
    """

    def __init__(
        self,
        run_id: str = "",
        base_dir: Path | str = EXPERIMENTS_DIR,
        tracking_uri: str | None = None,
        experiment_name: str = "investment_trying",
        **kwargs: Any,
    ) -> None:
        import mlflow

        self._mlflow = mlflow
        self.logger = ExperimentLogger(run_id=run_id, base_dir=base_dir, **kwargs)

        self._tracking_uri = tracking_uri
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self._run: Any = None

    def _start_run(self) -> Any:
        """Lazily start an MLflow run on first log call."""
        if self._run is None:
            try:
                self._run = self._mlflow.start_run(run_name=self.logger.run_id)
                self._mlflow.set_tag("run_id", self.logger.run_id)
                mlflow_logger.info(f"MLflow run started: {self.logger.run_id}")
            except Exception as e:
                mlflow_logger.warning(f"MLflow unavailable, using file-only logging: {e}")
        return self._run

    # ── Forwarded calls ──────────────────────────────────────────────────────

    def log_metadata(
        self,
        description: str = "",
        model_type: str = "",
        task: str = "",
        tickers: Sequence[str] | None = None,
        data_start: str = "",
        data_end: str = "",
        n_samples: int = 0,
        n_features: int = 0,
        random_state: int = 42,
        author: str = "",
    ) -> None:
        self.logger.log_metadata(
            description=description,
            model_type=model_type,
            task=task,
            tickers=tickers,
            data_start=data_start,
            data_end=data_end,
            n_samples=n_samples,
            n_features=n_features,
            random_state=random_state,
            author=author,
        )
        self._start_run()
        if self._run:
            try:
                self._mlflow.set_tag("model_type", model_type)
                self._mlflow.set_tag("task", task)
                self._mlflow.set_tag("data_start", data_start)
                self._mlflow.set_tag("data_end", data_end)
                self._mlflow.log_param("n_samples", n_samples)
                self._mlflow.log_param("n_features", n_features)
                self._mlflow.log_param("random_state", random_state)
            except Exception:
                mlflow_logger.debug("MLflow log_metadata failed", exc_info=True)

    def log_config(
        self,
        hyperparams: dict[str, Any] | None = None,
        features: Sequence[str] | None = None,
        cv_params: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.logger.log_config(hyperparams, features, cv_params, extra)
        self._start_run()
        if self._run:
            try:
                if hyperparams:
                    self._mlflow.log_params({f"hp.{k}": v for k, v in hyperparams.items()})
                if cv_params:
                    self._mlflow.log_params({f"cv.{k}": v for k, v in cv_params.items()})
            except Exception:
                mlflow_logger.debug("MLflow log_config failed", exc_info=True)

    def log_data_hash(self, data: np.ndarray | pd.DataFrame) -> str:
        return self.logger.log_data_hash(data)

    def log_fold_metrics(
        self,
        fold: int,
        train_metrics: dict[str, float],
        test_metrics: dict[str, float],
        train_start: str = "",
        train_end: str = "",
        test_start: str = "",
        test_end: str = "",
        purged_samples: int = 0,
        embargo_days: int = 0,
        n_train: int = 0,
        n_test: int = 0,
    ) -> dict[str, Any]:
        result = self.logger.log_fold_metrics(
            fold=fold,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            purged_samples=purged_samples,
            embargo_days=embargo_days,
            n_train=n_train,
            n_test=n_test,
        )
        self._start_run()
        if self._run:
            try:
                for k, v in test_metrics.items():
                    self._mlflow.log_metric(f"fold_{fold}.test.{k}", v, step=fold)
                for k, v in train_metrics.items():
                    self._mlflow.log_metric(f"fold_{fold}.train.{k}", v, step=fold)
                overfit_gap = result.get("overfit_gap", {})
                for k, v in overfit_gap.items():
                    self._mlflow.log_metric(f"fold_{fold}.overfit_gap.{k}", v, step=fold)
            except Exception:
                mlflow_logger.debug("MLflow log_fold_metrics failed", exc_info=True)
        return result

    def log_feature_importance(
        self,
        mdi: dict[str, float] | None = None,
        permutation: dict[str, float] | None = None,
        shap_top: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        result = self.logger.log_feature_importance(mdi, permutation, shap_top)
        self._start_run()
        if self._run:
            try:
                if mdi:
                    self._mlflow.log_dict({"mdi": mdi}, "feature_importance_mdi.json")
                if permutation:
                    self._mlflow.log_dict(
                        {"permutation": permutation}, "feature_importance_permutation.json"
                    )
            except Exception:
                mlflow_logger.debug("MLflow log_feature_importance failed", exc_info=True)
        return result

    def log_predictions(self, predictions_df: pd.DataFrame) -> None:
        self.logger.log_predictions(predictions_df)
        self._start_run()
        if self._run:
            try:
                csv_path = self.logger.run_dir / "predictions.csv"
                if csv_path.exists():
                    self._mlflow.log_artifact(str(csv_path))
            except Exception:
                mlflow_logger.debug("MLflow log_predictions artifact failed", exc_info=True)

    def log_model(self, model: Any, filename: str = "model.joblib") -> Path:
        result = self.logger.log_model(model, filename)
        self._start_run()
        if self._run:
            try:
                self._mlflow.log_artifact(str(result))
            except Exception:
                mlflow_logger.debug("MLflow log_model artifact failed", exc_info=True)
        return result

    def log_summary_verdict(
        self,
        mean_oos_metrics: dict[str, float],
        n_folds: int,
        best_fold_metrics: dict[str, float] | None = None,
        worst_fold_metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        result = self.logger.log_summary_verdict(
            mean_oos_metrics, n_folds, best_fold_metrics, worst_fold_metrics
        )
        self._start_run()
        if self._run:
            try:
                self._mlflow.log_dict(result, "summary.json")
                for k, v in mean_oos_metrics.items():
                    if isinstance(v, (int, float)):
                        self._mlflow.log_metric(f"mean_oos.{k}", v)
            except Exception:
                mlflow_logger.debug("MLflow log_summary_verdict failed", exc_info=True)
        return result

    # ── Passthrough properties ───────────────────────────────────────────────

    @property
    def run_id(self) -> str:
        return self.logger.run_id

    @property
    def run_dir(self) -> Path:
        return self.logger.run_dir

    def finish(self) -> None:
        """Close the MLflow run. Call this after logging is complete."""
        if self._run:
            try:
                self._mlflow.end_run()
                mlflow_logger.info(f"MLflow run ended: {self.logger.run_id}")
            except Exception:
                mlflow_logger.debug("MLflow end_run failed", exc_info=True)
