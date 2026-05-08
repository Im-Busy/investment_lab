"""AutoML baseline using AutoGluon TabularPredictor.

Trains 10+ models (CatBoost, LightGBM, XGBoost, RF, NN, EBM, etc.) in one shot
to establish the accuracy ceiling for hand-tuned models.

Usage:
    from src.ml.automl import AutoMLBaseline

    automl = AutoMLBaseline(label="profitability", time_limit=300)
    result = automl.fit(X_train, y_train)
    print(result.leaderboard)
    predictions = automl.predict(X_test)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AutoMLResult:
    """Results from an AutoML training run."""

    leaderboard: pd.DataFrame
    best_model: str
    best_score: float
    train_time_seconds: float
    models_trained: int
    feature_importance: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            "=" * 60,
            "AutoML Baseline Results",
            "=" * 60,
            f"Best model:      {self.best_model}",
            f"Best score:      {self.best_score:.4f}",
            f"Models trained:   {self.models_trained}",
            f"Training time:    {self.train_time_seconds:.1f}s",
            "",
            "Top 5 models:",
        ]
        if not self.leaderboard.empty:
            top5 = self.leaderboard.head(5)
            for _, row in top5.iterrows():
                lines.append(f"  {row['model']:30s} {row.get('score_val', 'N/A'):.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)


class AutoMLBaseline:
    """AutoGluon-powered AutoML baseline for accuracy benchmarking.

    Trains a diverse set of models (CatBoost, LightGBM, XGBoost, RF,
    Neural Networks, EBM, KNN, and ensembles) to establish the performance
    ceiling for manually-tuned classifiers.

    Example:
        >>> automl = AutoMLBaseline(label="profitable", time_limit=300)
        >>> result = automl.fit(X_train, y_train)
        >>> preds = automl.predict(X_test)
        >>> proba = automl.predict_proba(X_test)
    """

    def __init__(
        self,
        label: str = "profitable",
        time_limit: int = 300,
        eval_metric: str = "roc_auc",
        problem_type: str = "binary",
        presets: str = "medium_quality",
        path: Optional[str] = None,
        verbosity: int = 1,
        seed: int = 42,
    ):
        """Initialize AutoML baseline.

        Args:
            label: Name of the target column.
            time_limit: Maximum training time in seconds (None = no limit).
            eval_metric: Evaluation metric (roc_auc, accuracy, f1, log_loss).
            problem_type: 'binary', 'multiclass', or 'regression'.
            presets: AutoGluon quality preset:
                'best_quality' — most accurate, slowest
                'high_quality' — good accuracy
                'medium_quality' — balanced (default)
                'optimize_for_deployment' — small/fast models
            path: Directory to save AutoGluon artifacts.
            verbosity: Logging level (0=silent, 1=info, 2=debug).
            seed: Random seed.
        """
        self.label = label
        self.time_limit = time_limit
        self.eval_metric = eval_metric
        self.problem_type = problem_type
        self.presets = presets
        self.path = path
        self.verbosity = verbosity
        self.seed = seed

        self._predictor: Any = None
        self._result: Optional[AutoMLResult] = None
        self._is_fitted: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_limit: Optional[int] = None,
    ) -> AutoMLResult:
        """Train AutoGluon TabularPredictor on features and labels.

        Args:
            X: Feature DataFrame.
            y: Label series (binary 0/1 for classification, float for regression).
            time_limit: Override training time limit in seconds.

        Returns:
            AutoMLResult with leaderboard and best model info.
        """
        from autogluon.tabular import TabularPredictor

        tl = time_limit or self.time_limit

        train_data = X.copy()
        train_data[self.label] = y.values

        logger.info(
            f"Starting AutoML: {self.problem_type} classification, "
            f"{train_data.shape[0]} samples, {X.shape[1]} features, "
            f"time_limit={tl}s, presets={self.presets}"
        )

        self._predictor = TabularPredictor(
            label=self.label,
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
            path=self.path,
            verbosity=self.verbosity,
        )

        import time

        start = time.perf_counter()
        self._predictor.fit(
            train_data=train_data,
            time_limit=tl,
            presets=self.presets,
            verbosity=self.verbosity,
        )
        elapsed = time.perf_counter() - start

        leaderboard = self._predictor.leaderboard(silent=True)
        best_model = str(leaderboard.iloc[0]["model"]) if not leaderboard.empty else "unknown"
        best_score = float(leaderboard.iloc[0]["score_val"]) if not leaderboard.empty else 0.0

        feature_importance = {}
        try:
            fi = self._predictor.feature_importance(train_data)
            if not fi.empty:
                feature_importance = dict(
                    sorted(
                        zip(fi.index, fi["importance"]),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:20]
                )
        except Exception:
            pass

        self._result = AutoMLResult(
            leaderboard=leaderboard,
            best_model=best_model,
            best_score=best_score,
            train_time_seconds=elapsed,
            models_trained=len(leaderboard),
            feature_importance=feature_importance,
            hyperparameters={
                "time_limit": tl,
                "presets": self.presets,
                "eval_metric": self.eval_metric,
                "problem_type": self.problem_type,
            },
        )
        self._is_fitted = True

        logger.info(
            f"AutoML complete: {self._result.models_trained} models in "
            f"{elapsed:.1f}s, best={best_model} ({best_score:.4f})"
        )
        return self._result

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of predicted class labels.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._predictor.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict class probabilities.

        Args:
            X: Feature DataFrame.

        Returns:
            DataFrame with probability columns per class.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._predictor.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate the best model on held-out data.

        Args:
            X: Feature DataFrame.
            y: Label series.

        Returns:
            Dict with evaluation metrics.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._predictor.evaluate(X.assign(**{self.label: y.values}))

    def leaderboard(self) -> pd.DataFrame:
        """Return the model leaderboard."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._predictor.leaderboard(silent=True)

    def compare_to_baseline(
        self,
        baseline_score: float,
        baseline_name: str = "Hand-tuned CatBoost",
    ) -> Dict[str, Any]:
        """Compare AutoML best score against a baseline.

        Args:
            baseline_score: AUC/accuracy from your hand-tuned model.
            baseline_name: Label for the baseline model.

        Returns:
            Dict with uplift and winner.
        """
        if not self._result:
            raise ValueError("Run fit() first.")

        uplift = self._result.best_score - baseline_score
        uplift_pct = (uplift / abs(baseline_score)) * 100 if baseline_score != 0 else 0.0

        return {
            "baseline_name": baseline_name,
            "baseline_score": baseline_score,
            "automl_best_model": self._result.best_model,
            "automl_score": self._result.best_score,
            "uplift": uplift,
            "uplift_percent": uplift_pct,
            "winner": "AutoML" if uplift > 0 else baseline_name,
            "models_trained": self._result.models_trained,
            "train_time_seconds": self._result.train_time_seconds,
        }

    def save(self, path: str) -> None:
        """Save the trained AutoML predictor to disk."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        self._predictor.save(str(path_obj))

    @classmethod
    def load(cls, path: str) -> "AutoMLBaseline":
        """Load a saved AutoML predictor from disk."""
        from autogluon.tabular import TabularPredictor

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model path not found: {path}")

        predictor = TabularPredictor.load(str(path_obj))
        instance = cls()
        instance._predictor = predictor
        instance._is_fitted = True

        try:
            lb = predictor.leaderboard(silent=True)
            instance._result = AutoMLResult(
                leaderboard=lb,
                best_model=str(lb.iloc[0]["model"]) if not lb.empty else "unknown",
                best_score=float(lb.iloc[0]["score_val"]) if not lb.empty else 0.0,
                train_time_seconds=0.0,
                models_trained=len(lb),
            )
        except Exception:
            pass

        return instance

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def result(self) -> Optional[AutoMLResult]:
        return self._result


def run_automl_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    hand_tuned_auc: float,
    hand_tuned_name: str = "Hand-Tuned CatBoost",
    time_limit: int = 300,
    presets: str = "medium_quality",
) -> Dict[str, Any]:
    """Convenience function: run AutoML and compare against baseline.

    Args:
        X: Feature DataFrame.
        y: Binary label series.
        hand_tuned_auc: AUC from your existing hand-tuned model.
        hand_tuned_name: Label for your hand-tuned model.
        time_limit: AutoML training time limit.
        presets: AutoGluon quality preset.

    Returns:
        Dict with benchmark results, leaderboard, and comparison.
    """
    automl = AutoMLBaseline(
        label="target",
        time_limit=time_limit,
        presets=presets,
        eval_metric="roc_auc",
    )
    result = automl.fit(X, y)

    comparison = automl.compare_to_baseline(hand_tuned_auc, hand_tuned_name)

    return {
        "automl_result": result,
        "comparison": comparison,
    }
