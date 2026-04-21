"""
ML Experiment Logger — JSONL structured logging for all ML runs.

Provides ExperimentLogger class that creates a run directory hierarchy with:
- metadata.json: Run description, model type, data info, timestamps
- config.json: Hyperparameters, features, random state
- data_hash.json: SHA256 of training data for reproducibility
- fold_metrics.jsonl: Per-fold metrics (train/test dates, IC, accuracy, overfit gap)
- feature_importance.json: MDI, permutation, SHAP importance scores
- predictions.csv: Per-sample predictions with actuals
- model artifact: Serialized model via joblib

Directory structure:
    experiments/
    └── runs/
        └── <run_id>/
            ├── metadata.json
            ├── config.json
            ├── data_hash.json
            ├── fold_metrics.jsonl
            ├── feature_importance.json
            ├── predictions.csv
            └── model.joblib

Usage:
    logger = ExperimentLogger("my_run", base_dir="experiments")
    logger.log_metadata(description="...", model_type="random_forest", ...)
    logger.log_config(hyperparams={...}, features=[...])
    logger.log_fold_metrics(fold=1, train_metrics={...}, test_metrics={...})
    logger.log_feature_importance(mdi={...}, permutation={...})
    logger.log_predictions(df)  # DataFrame with pred, actual, features
    logger.log_model(model)  # sklearn-compatible model
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("experiments") / "runs"


@dataclass
class ExperimentMetadata:
    """Metadata for a single experiment run."""

    run_id: str
    description: str
    model_type: str
    task: str
    date: str
    data_start: str
    data_end: str
    n_samples: int
    n_features: int
    tickers: list[str] | None = None
    author: str = ""
    random_state: int = 42
    data_hash_sha256: str = ""


@dataclass
class FoldMetrics:
    """Metrics for a single CV fold."""

    fold: int
    train_start: str = ""
    train_end: str = ""
    test_start: str = ""
    test_end: str = ""
    purged_samples: int = 0
    embargo_days: int = 0
    n_train: int = 0
    n_test: int = 0
    train_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def overfit_gap(self) -> dict[str, float]:
        """Compute train - test gap for each metric."""
        return {
            k: self.train_metrics.get(k, 0) - self.test_metrics.get(k, 0)
            for k in self.test_metrics
            if k in self.train_metrics
        }


class ExperimentLogger:
    """Structured JSONL experiment logger for ML runs.

    Creates a run directory under experiments/runs/<run_id>/ and writes
    metadata, config, per-fold metrics, feature importance, predictions,
    and model artifacts in machine-readable formats.

    Attributes:
        run_id: Unique identifier for this experiment run.
        run_dir: Path to the run directory.
    """

    def __init__(
        self,
        run_id: str = "",
        base_dir: Path | str = EXPERIMENTS_DIR,
        description: str = "",
        model_type: str = "",
        task: str = "",
    ) -> None:
        self.run_id = run_id or f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / self.run_id
        self._metadata: dict[str, Any] = {
            "run_id": self.run_id,
            "description": description,
            "model_type": model_type,
            "task": task,
            "date": datetime.now().isoformat(),
        }
        self._config: dict[str, Any] = {}
        self._fold_metrics: list[dict[str, Any]] = []
        self._feature_importance: dict[str, Any] = {}
        self._predictions: list[dict[str, Any]] = []
        self._model_saved = False

    def _init_run_dir(self) -> Path:
        """Create the run directory if it doesn't exist."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

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
        """Log experiment metadata.

        Args:
            description: Human-readable description of the experiment.
            model_type: Model type string (e.g., 'random_forest', 'lightgbm').
            task: Task type (e.g., 'regime_classification', 'signal_generation').
            tickers: List of ticker symbols used in training.
            data_start: Start date of the training data (YYYY-MM-DD).
            data_end: End date of the training data (YYYY-MM-DD).
            n_samples: Number of samples in the dataset.
            n_features: Number of features used.
            random_state: Random seed for reproducibility.
            author: Author of the experiment.
        """
        self._metadata.update(
            {
                "description": description or self._metadata.get("description", ""),
                "model_type": model_type or self._metadata.get("model_type", ""),
                "task": task or self._metadata.get("task", ""),
                "tickers": list(tickers) if tickers else [],
                "data_start": data_start,
                "data_end": data_end,
                "n_samples": n_samples,
                "n_features": n_features,
                "random_state": random_state,
                "author": author,
            }
        )
        self._save_json("metadata.json", self._metadata)
        logger.info(f"Logged metadata for run {self.run_id}")

    def log_config(
        self,
        hyperparams: dict[str, Any] | None = None,
        features: Sequence[str] | None = None,
        cv_params: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log experiment configuration.

        Args:
            hyperparams: Model hyperparameters dict.
            features: List of feature names used.
            cv_params: Cross-validation parameters (n_splits, embargo_days, etc.).
            extra: Any additional configuration.
        """
        self._config = {
            "hyperparams": hyperparams or {},
            "features": list(features) if features else [],
            "cv_params": cv_params or {},
            "extra": extra or {},
        }
        self._save_json("config.json", self._config)
        logger.info(f"Logged config for run {self.run_id}")

    def log_data_hash(self, data: np.ndarray | pd.DataFrame) -> str:
        """Compute and save SHA256 hash of training data.

        Args:
            data: Training data array or DataFrame.

        Returns:
            SHA256 hex digest string.
        """
        if isinstance(data, pd.DataFrame):
            raw = data.to_numpy().tobytes()
        else:
            raw = data.tobytes()
        sha = hashlib.sha256(raw).hexdigest()
        self._save_json(
            "data_hash.json", {"sha256": sha, "computed_at": datetime.now().isoformat()}
        )
        logger.info(f"Logged data hash: {sha[:16]}...")
        return sha

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
        """Log metrics for a single CV fold. Appends to JSONL file.

        Args:
            fold: Fold number (1-indexed).
            train_metrics: Dict of metric_name → value for training set.
            test_metrics: Dict of metric_name → value for test set.
            train_start: Training data start date.
            train_end: Training data end date.
            test_start: Test data start date.
            test_end: Test data end date.
            purged_samples: Number of samples purged from training.
            embargo_days: Number of embargo days after test period.
            n_train: Number of training samples.
            n_test: Number of test samples.

        Returns:
            Dict with all logged metrics including overfit_gap.
        """
        overfit_gap = {
            k: train_metrics.get(k, 0) - test_metrics.get(k, 0)
            for k in test_metrics
            if k in train_metrics
        }

        record = {
            "fold": fold,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "purged_samples": purged_samples,
            "embargo_days": embargo_days,
            "n_train": n_train,
            "n_test": n_test,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "overfit_gap": overfit_gap,
        }
        self._fold_metrics.append(record)

        filepath = self._init_run_dir() / "fold_metrics.jsonl"
        with open(filepath, "a") as f:
            f.write(json.dumps(record, default=_json_serializer) + "\n")

        logger.info(
            f"Fold {fold}: test_metrics={test_metrics}, "
            f"overfit_gap={overfit_gap.get('accuracy', overfit_gap.get('spearman_ic', 'N/A'))}"
        )
        return record

    def log_feature_importance(
        self,
        mdi: dict[str, float] | None = None,
        permutation: dict[str, float] | None = None,
        shap_top: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Log feature importance from multiple methods.

        Args:
            mdi: Mean Decrease Impurity (Gini) importance scores.
            permutation: Permutation importance scores.
            shap_top: Top-N SHAP importance entries with feature and mean_abs_shap.
        """
        self._feature_importance = {
            "mdi": mdi or {},
            "permutation": permutation or {},
            "shap_top10": (shap_top or [])[:10],
        }
        self._save_json("feature_importance.json", self._feature_importance)
        logger.info(
            f"Logged feature importance: {len(mdi or {})} MDI, {len(permutation or {})} permutation"
        )
        return self._feature_importance

    def log_predictions(self, predictions_df: pd.DataFrame) -> None:
        """Log per-sample predictions to CSV.

        Expected columns: timestamp, ticker, pred, actual (and any extras).

        Args:
            predictions_df: DataFrame with prediction results.
        """
        filepath = self._init_run_dir() / "predictions.csv"
        predictions_df.to_csv(filepath, index=False)
        self._predictions = []
        logger.info(f"Logged {len(predictions_df)} predictions to {filepath}")

    def log_model(self, model: Any, filename: str = "model.joblib") -> Path:
        """Serialize and save the trained model.

        Args:
            model: Trained sklearn-compatible model object.
            filename: Output filename (default: model.joblib).

        Returns:
            Path to the saved model file.
        """
        filepath = self._init_run_dir() / filename
        joblib.dump(model, filepath)
        self._model_saved = True
        logger.info(f"Saved model to {filepath}")
        return filepath

    def log_summary_verdict(
        self,
        mean_oos_metrics: dict[str, float],
        n_folds: int,
        best_fold_metrics: dict[str, float] | None = None,
        worst_fold_metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Generate and save a summary verdict for the experiment run.

        Computes verdict string based on overfitting thresholds:
        - overfit_gap > 0.20 in accuracy → OVERFIT
        - OOS Spearman IC < 0.05 → WEAK SIGNAL
        - Otherwise → VIABLE

        Args:
            mean_oos_metrics: Mean out-of-sample metrics across folds.
            n_folds: Number of CV folds.
            best_fold_metrics: Best fold's test metrics.
            worst_fold_metrics: Worst fold's test metrics.

        Returns:
            Verdict dict with recommendation.
        """
        overfit_gap = mean_oos_metrics.get("overfit_gap", {}) or {}
        max_overfit = max((abs(v) for v in overfit_gap.values()), default=0)
        oos_ic = mean_oos_metrics.get("spearman_ic", mean_oos_metrics.get("rank_ic", 0))
        oos_acc = mean_oos_metrics.get("accuracy", 0)

        reasons = []
        if max_overfit > 0.20:
            reasons.append(f"overfit_gap={max_overfit:.2f} > 0.20")
        if oos_ic < 0.05:
            reasons.append(f"OOS IC={oos_ic:.4f} < 0.05")
        if oos_acc < 0.50 and oos_acc > 0:
            reasons.append(f"OOS accuracy={oos_acc:.2f} < 0.50 (worse than random)")

        if not reasons:
            verdict = "VIABLE"
            recommendation = "Model shows acceptable OOS performance. Consider further validation."
        else:
            verdict = "OVERFIT" if max_overfit > 0.20 else "WEAK SIGNAL"
            recommendation = (
                "Do not use in production. Try feature reduction, different model, or more data."
            )

        summary = {
            "run_id": self.run_id,
            "mean_oos_metrics": {
                k: round(v, 6) if isinstance(v, float) else v for k, v in mean_oos_metrics.items()
            },
            "max_overfit_gap": round(max_overfit, 4),
            "n_folds": n_folds,
            "verdict": verdict,
            "reasons": reasons,
            "recommendation": recommendation,
            "best_fold": best_fold_metrics,
            "worst_fold": worst_fold_metrics,
        }
        self._save_json("summary.json", summary)
        logger.info(
            f"Summary verdict: {verdict} — {', '.join(reasons) if reasons else 'No overfitting detected'}"
        )
        return summary

    def get_run_dir(self) -> Path:
        """Return the run directory path."""
        return self.run_dir

    def _save_json(self, filename: str, data: Any) -> None:
        """Save data as JSON to the run directory."""
        self._init_run_dir()
        filepath = self.run_dir / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=_json_serializer)

    def load_fold_metrics(self) -> list[dict[str, Any]]:
        """Load fold metrics from the JSONL file.

        Returns:
            List of fold metric dicts.
        """
        filepath = self.run_dir / "fold_metrics.jsonl"
        if not filepath.exists():
            return []
        records = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def load_model(self, filename: str = "model.joblib") -> Any:
        """Load a saved model from the run directory.

        Args:
            filename: Model filename (default: model.joblib).

        Returns:
            Loaded model object.
        """
        filepath = self.run_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Model not found at {filepath}")
        return joblib.load(filepath)

    def load_predictions(self) -> pd.DataFrame:
        """Load predictions from the CSV file.

        Returns:
            DataFrame with columns from the logged predictions.
        """
        filepath = self.run_dir / "predictions.csv"
        if not filepath.exists():
            return pd.DataFrame()
        return pd.read_csv(filepath)


def _json_serializer(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
