"""
Survival Analysis for Time-to-Target Prediction.

Converts triple-barrier labels into survival data and predicts *when* a
take-profit or stop-loss barrier will be hit, using right-censored
survival models from scikit-survival.

The core idea: instead of just binary classification (TP vs SL vs timeout),
we model the time-to-event as a survival function. For each bar, we have:
  - time_to_event: bars until TP or SL is hit (or timeout limit)
  - event_occurred: True if TP/SL hit, False if censored (timeout)

Models supported:
  - Cox Proportional Hazards (interpretable coefficients)
  - Random Survival Forest (robust, non-parametric)
  - Gradient Boosting Survival Analysis (best performance)

Usage:
    from src.ml.survival_analyzer import SurvivalAnalyzer

    labeler = TripleBarrierLabeler()
    labels = labeler.fit(close, high, low, time_limit=20)

    analyzer = SurvivalAnalyzer(model_type="gbs")
    analyzer.fit(X_features, labels)
    survival_probs = analyzer.predict_survival(X_features, bars_ahead=20)
    time_to_exit = analyzer.predict_time_to_exit(X_features)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class SurvivalTarget:
    """Survival data built from triple-barrier labels.

    Attributes:
        time: Bars until barrier hit or timeout (must be > 0).
        event: True if TP or SL was hit, False if censored (timeout).
        barrier_type: 'tp', 'sl', or 'time' for each observation.
        return_pct: Percentage return at exit.
        raw_labels: Original triple-barrier labels (+1/-1/0).
    """

    time: np.ndarray
    event: np.ndarray
    barrier_type: np.ndarray
    return_pct: np.ndarray
    raw_labels: np.ndarray

    @classmethod
    def from_triple_barrier_labels(
        cls,
        labels: pd.Series,
        label_span: int | None = None,
    ) -> "SurvivalTarget":
        """Convert triple-barrier labels to survival format.

        The triple-barrier labeler stores auxiliary arrays in `.attrs`:
          - bars_to_exit: int array, number of bars until exit
          - barrier: str array, 'tp', 'sl', or 'time'
          - return_pct: float array

        Args:
            labels: Output from TripleBarrierLabeler.fit().
            label_span: Maximum bars to consider (defaults to max bars_to_exit).

        Returns:
            SurvivalTarget with time/event/barrier_type/return_pct.
        """
        bars_to_exit = labels.attrs.get("bars_to_exit")
        barrier = labels.attrs.get("barrier")
        return_pct = labels.attrs.get("return_pct")

        if bars_to_exit is None or barrier is None:
            raise ValueError(
                "labels must have .attrs with 'bars_to_exit' and 'barrier'. "
                "Use TripleBarrierLabeler.fit() to generate them."
            )

        time_arr = np.asarray(bars_to_exit, dtype=np.float64)
        barrier_arr = np.asarray(barrier, dtype=str)
        return_arr = (
            np.asarray(return_pct, dtype=np.float64)
            if return_pct is not None
            else np.zeros_like(time_arr)
        )
        raw = labels.values.astype(np.float64)

        # Event: True if TP or SL was hit, False if timeout
        event_arr = (barrier_arr != "time").astype(bool)

        # Ensure time > 0 for uncensored observations (scikit-survival requires)
        time_arr = np.maximum(time_arr, 1.0)

        if label_span is not None:
            time_arr = np.minimum(time_arr, float(label_span))

        return cls(
            time=time_arr,
            event=event_arr,
            barrier_type=barrier_arr,
            return_pct=return_arr,
            raw_labels=raw,
        )

    def to_structured_array(self) -> np.ndarray:
        """Convert to scikit-survival structured array format.

        Returns:
            Structured array with fields ('event', bool) and ('time', float64).
        """
        from sksurv.util import Surv

        return Surv.from_arrays(event=self.event, time=self.time)

    @property
    def censoring_rate(self) -> float:
        """Fraction of censored observations (timeouts)."""
        return float(1.0 - np.mean(self.event))

    @property
    def n_samples(self) -> int:
        return len(self.time)

    @property
    def n_events(self) -> int:
        return int(np.sum(self.event))


@dataclass
class SurvivalResult:
    """Results from survival model training and evaluation.

    Attributes:
        model_type: Type of survival model used.
        c_index: Concordance index (Uno's, robust to censoring).
        integrated_brier: Integrated Brier score (lower is better).
        time_auc: Mean time-dependent AUC.
        feature_importance: Dict of feature name -> importance score.
        n_train: Number of training samples.
        n_events: Number of events (uncensored) in training.
        train_time_seconds: Training wall-clock time.
    """

    model_type: str
    c_index: float
    integrated_brier: float
    time_auc: float
    feature_importance: Dict[str, float]
    n_train: int
    n_events: int
    train_time_seconds: float
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "c_index": round(self.c_index, 4),
            "integrated_brier": round(self.integrated_brier, 4),
            "time_auc": round(self.time_auc, 4),
            "n_train": self.n_train,
            "n_events": self.n_events,
            "train_time_seconds": round(self.train_time_seconds, 2),
        }

    def __repr__(self) -> str:
        return (
            f"SurvivalResult(model={self.model_type}, c_idx={self.c_index:.4f}, "
            f"ibs={self.integrated_brier:.4f}, auc={self.time_auc:.4f}, "
            f"n={self.n_train}, events={self.n_events})"
        )


class SurvivalAnalyzer:
    """Train and evaluate survival models for time-to-target prediction.

    Supports three model types:
        - "cox": Cox Proportional Hazards (interpretable)
        - "rsf": Random Survival Forest (robust, non-parametric)
        - "gbs": Gradient Boosting Survival Analysis (best performance)

    All models follow sklearn-compatible fit/predict API and integrate
    with PurgedKFold for time-series-safe cross-validation.

    Example:
        >>> analyzer = SurvivalAnalyzer(model_type="gbs", n_estimators=200)
        >>> result = analyzer.fit(X_train, labels)
        >>> surv_probs = analyzer.predict_survival(X_test, bars_ahead=[5, 10, 20])
        >>> time_pred = analyzer.predict_time_to_exit(X_test)
    """

    SUPPORTED_MODELS = ("cox", "rsf", "gbs")

    def __init__(
        self,
        model_type: str = "gbs",
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 10,
        random_state: int = 42,
        alpha: float = 0.1,
        verbose: bool = False,
    ):
        """Initialize survival analyzer.

        Args:
            model_type: "cox", "rsf", or "gbs".
            n_estimators: Number of trees (RSF/GBS).
            max_depth: Maximum tree depth (RSF/GBS).
            learning_rate: Learning rate (GBS only).
            min_samples_leaf: Minimum samples per leaf (RSF/GBS).
            random_state: Random seed.
            alpha: Regularization strength (Cox only).
            verbose: Whether to print training progress.
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model_type: {model_type}. Supported: {self.SUPPORTED_MODELS}"
            )
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.alpha = alpha
        self.verbose = verbose

        self._surv = None
        self._model = None
        self._feature_names: Optional[List[str]] = None
        self._train_times: Optional[np.ndarray] = None

    def _create_model(self) -> Any:
        """Create the survival model instance."""
        if self.model_type == "cox":
            from sksurv.linear_model import CoxPHSurvivalAnalysis

            return CoxPHSurvivalAnalysis(alpha=self.alpha)

        elif self.model_type == "rsf":
            from sksurv.ensemble import RandomSurvivalForest

            return RandomSurvivalForest(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
            )

        elif self.model_type == "gbs":
            from sksurv.ensemble import GradientBoostingSurvivalAnalysis

            return GradientBoostingSurvivalAnalysis(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                verbose=1 if self.verbose else 0,
            )

    def fit(
        self,
        X: pd.DataFrame,
        labels: pd.Series | SurvivalTarget,
        categorical_features: Optional[List[str]] = None,
    ) -> SurvivalAnalyzer:
        """Train the survival model.

        Args:
            X: Feature matrix.
            labels: Triple-barrier labels (from TripleBarrierLabeler) or SurvivalTarget.
            categorical_features: List of categorical column names.

        Returns:
            Self for chaining.
        """
        import time

        self._feature_names = list(X.columns)
        X_arr = X.values.astype(np.float64)

        if isinstance(labels, SurvivalTarget):
            target = labels
        else:
            target = SurvivalTarget.from_triple_barrier_labels(labels)

        self._surv = target
        self._train_times = target.time

        self._model = self._create_model()

        t0 = time.perf_counter()
        self._model.fit(X_arr, target.to_structured_array())
        self._train_time_seconds = time.perf_counter() - t0

        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Predict risk scores (higher = event sooner).

        Args:
            X: Feature matrix.

        Returns:
            Array of risk scores.
        """
        self._check_fitted()
        return self._model.predict(X.values.astype(np.float64))

    def predict_survival(
        self,
        X: pd.DataFrame,
        bars_ahead: List[int] | np.ndarray,
    ) -> np.ndarray:
        """Predict survival probability at given time horizons.

        Args:
            X: Feature matrix.
            bars_ahead: List of bar horizons to predict at (e.g., [5, 10, 20]).

        Returns:
            Array of shape (n_samples, n_horizons) with survival probabilities.
            Survival prob = probability the event has NOT occurred by that time.
            So 1 - survival = probability TP/SL was hit by that time.
        """
        self._check_fitted()
        times = np.asarray(bars_ahead, dtype=np.float64)
        surv_funcs = self._model.predict_survival_function(
            X.values.astype(np.float64), return_array=False
        )
        probs = np.zeros((len(X), len(times)), dtype=np.float64)
        for i, sf in enumerate(surv_funcs):
            probs[i, :] = sf(times)
        return probs

    def predict_time_to_exit(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict expected time to exit (TP or SL hit).

        Uses the survival function to estimate the time at which the
        probability of hitting a barrier exceeds the threshold.

        Args:
            X: Feature matrix.
            threshold: Survival probability threshold (default 0.5).
                        Lower = more conservative (shorter exit time).

        Returns:
            Array of predicted bars to exit.
        """
        self._check_fitted()
        surv_funcs = self._model.predict_survival_function(
            X.values.astype(np.float64), return_array=False
        )
        if self._train_times is None:
            max_time = 50
        else:
            max_time = int(np.percentile(self._train_times, 95))

        times = np.arange(1, max_time + 1, dtype=np.float64)
        predictions = np.full(len(X), max_time, dtype=np.float64)

        for i, sf in enumerate(surv_funcs):
            surv_probs = sf(times)
            crossing = np.where(surv_probs <= (1.0 - threshold))[0]
            if len(crossing) > 0:
                predictions[i] = times[crossing[0]]

        return predictions

    def evaluate(
        self,
        X: pd.DataFrame,
        labels: pd.Series | SurvivalTarget,
        eval_times: np.ndarray | None = None,
    ) -> SurvivalResult:
        """Evaluate the survival model.

        Args:
            X: Feature matrix.
            labels: Triple-barrier labels or SurvivalTarget.
            eval_times: Time points for Brier score / AUC evaluation.

        Returns:
            SurvivalResult with metrics.
        """
        from sksurv.metrics import (
            concordance_index_ipcw,
            cumulative_dynamic_auc,
            integrated_brier_score,
        )

        self._check_fitted()

        if isinstance(labels, SurvivalTarget):
            target = labels
        else:
            target = SurvivalTarget.from_triple_barrier_labels(labels)

        y_train = self._surv.to_structured_array()
        y_test = target.to_structured_array()
        risk_scores = self.predict_risk(X)

        c_index = float(concordance_index_ipcw(y_train, y_test, risk_scores)[0])

        if eval_times is None:
            min_time = int(np.percentile(target.time, 10))
            max_time = int(np.percentile(target.time, 90))
            eval_times = np.linspace(min_time, max_time, 10, dtype=np.float64)

        ibs = float(
            integrated_brier_score(
                y_train,
                y_test,
                self._model.predict_survival_function(X.values.astype(np.float64)),
                eval_times,
            )
        )

        try:
            time_auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, eval_times)
            mean_auc = float(np.mean(time_auc[~np.isnan(time_auc)]))
        except Exception:
            mean_auc = float("nan")

        feat_imp = self._compute_feature_importance(X)

        return SurvivalResult(
            model_type=self.model_type,
            c_index=c_index,
            integrated_brier=ibs,
            time_auc=mean_auc,
            feature_importance=feat_imp,
            n_train=len(X),
            n_events=target.n_events,
            train_time_seconds=self._train_time_seconds,
        )

    def cross_validate(
        self,
        X: pd.DataFrame,
        labels: pd.Series | SurvivalTarget,
        n_splits: int = 5,
        pct_embargo: float = 0.05,
        label_span: int | None = None,
        eval_times: np.ndarray | None = None,
    ) -> SurvivalResult:
        """Purged cross-validation for survival model.

        Args:
            X: Feature matrix.
            labels: Triple-barrier labels or SurvivalTarget.
            n_splits: Number of CV folds.
            pct_embargo: Embargo fraction.
            label_span: Label computation window (auto-detected if None).
            eval_times: Time points for Brier/AUC evaluation.

        Returns:
            SurvivalResult with mean fold metrics.
        """
        from sksurv.metrics import concordance_index_ipcw, integrated_brier_score

        self._feature_names = list(X.columns)

        if isinstance(labels, SurvivalTarget):
            target = labels
        else:
            target = SurvivalTarget.from_triple_barrier_labels(labels)

        y_struct = target.to_structured_array()
        X_arr = X.values.astype(np.float64)

        if label_span is None:
            label_span = int(np.percentile(target.time, 90))

        from .purged_cv import PurgedKFold

        cv = PurgedKFold(
            n_splits=n_splits,
            pct_embargo=pct_embargo,
            label_span=label_span,
        )

        if eval_times is None:
            min_time = int(np.percentile(target.time, 10))
            max_time = int(np.percentile(target.time, 90))
            eval_times = np.linspace(min_time, max_time, 10, dtype=np.float64)

        fold_metrics = []
        all_risk_scores = np.zeros(len(X_arr), dtype=np.float64)
        all_event = target.event.astype(bool)
        all_time = target.time

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X), 1):
            model = self._create_model()
            model.fit(X_arr[train_idx], y_struct[train_idx])

            risk = model.predict(X_arr[test_idx])
            all_risk_scores[test_idx] = risk

            c_idx = float(concordance_index_ipcw(y_struct[train_idx], y_struct[test_idx], risk)[0])

            try:
                surv_funcs = model.predict_survival_function(X_arr[test_idx])
                ibs = float(
                    integrated_brier_score(
                        y_struct[train_idx],
                        y_struct[test_idx],
                        surv_funcs,
                        eval_times,
                    )
                )
            except Exception:
                ibs = float("nan")

            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "c_index": round(c_idx, 4),
                    "integrated_brier": round(ibs, 4),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_events_test": int(np.sum(target.event[test_idx])),
                }
            )

        mean_c_idx = float(np.mean([m["c_index"] for m in fold_metrics]))
        mean_ibs = float(np.nanmean([m["integrated_brier"] for m in fold_metrics]))

        try:
            from sksurv.metrics import cumulative_dynamic_auc
            from sksurv.util import Surv

            y_all = Surv.from_arrays(event=all_event, time=all_time)
            time_auc, _ = cumulative_dynamic_auc(y_all, y_all, all_risk_scores, eval_times)
            mean_auc = float(np.mean(time_auc[~np.isnan(time_auc)]))
        except Exception:
            mean_auc = float("nan")

        feat_imp = {}
        full_model = self._create_model()
        full_model.fit(X_arr, y_struct)
        self._model = full_model
        self._surv = target
        self._train_times = target.time
        feat_imp = self._compute_feature_importance(X)

        return SurvivalResult(
            model_type=self.model_type,
            c_index=mean_c_idx,
            integrated_brier=mean_ibs,
            time_auc=mean_auc,
            feature_importance=feat_imp,
            n_train=len(X),
            n_events=target.n_events,
            train_time_seconds=0.0,
            fold_metrics=fold_metrics,
        )

    def _compute_feature_importance(self, X: pd.DataFrame) -> Dict[str, float]:
        """Compute feature importance for tree-based models.

        Uses permutation importance for robust estimates.
        """
        if self._model is None or self._surv is None:
            return {}

        if self.model_type == "cox":
            coef = self._model.coef_
            return dict(zip(self._feature_names or [], np.abs(coef)))

        try:
            from sklearn.inspection import permutation_importance
            from sksurv.metrics import concordance_index_ipcw

            y_train = self._surv.to_structured_array()
            X_arr = X.values.astype(np.float64)

            def _scorer(estimator, X_test, y_test):
                risk = estimator.predict(X_test)
                return concordance_index_ipcw(y_train, y_test, risk)[0]

            result = permutation_importance(
                self._model,
                X_arr,
                y_train,
                scoring=_scorer,
                n_repeats=5,
                random_state=self.random_state,
                n_jobs=-1,
            )
            return dict(
                zip(
                    self._feature_names or [],
                    result.importances_mean,
                )
            )
        except Exception:
            if hasattr(self._model, "feature_importances_"):
                return dict(
                    zip(
                        self._feature_names or [],
                        self._model.feature_importances_,
                    )
                )
            return {}

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "surv": self._surv,
                "feature_names": self._feature_names,
                "train_times": self._train_times,
                "config": {
                    "model_type": self.model_type,
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                    "min_samples_leaf": self.min_samples_leaf,
                    "random_state": self.random_state,
                    "alpha": self.alpha,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SurvivalAnalyzer":
        """Load model from disk."""
        import joblib

        data = joblib.load(path)
        config = data["config"]
        instance = cls(**config)
        instance._model = data["model"]
        instance._surv = data.get("surv")
        instance._feature_names = data.get("feature_names")
        instance._train_times = data.get("train_times")
        return instance

    def _check_fitted(self) -> None:
        if self._model is None:
            raise ValueError("Model not trained. Call fit() or cross_validate() first.")

    def __repr__(self) -> str:
        status = "fitted" if self._model is not None else "untrained"
        return f"SurvivalAnalyzer(model={self.model_type}, status={status})"


def compare_survival_models(
    X: pd.DataFrame,
    labels: pd.Series | SurvivalTarget,
    n_splits: int = 5,
    pct_embargo: float = 0.05,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compare Cox, RSF, and GBS survival models using purged CV.

    Args:
        X: Feature matrix.
        labels: Triple-barrier labels or SurvivalTarget.
        n_splits: Number of CV folds.
        pct_embargo: Embargo fraction.
        verbose: Print progress.

    Returns:
        DataFrame with model comparison metrics.
    """
    results = []
    model_configs = [
        ("cox", {}),
        ("rsf", {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 10}),
        (
            "gbs",
            {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.05,
                "min_samples_leaf": 10,
            },
        ),
    ]

    for model_type, extra_kwargs in model_configs:
        if verbose:
            print(f"\nEvaluating {model_type.upper()}...")
        analyzer = SurvivalAnalyzer(model_type=model_type, **extra_kwargs)
        result = analyzer.cross_validate(X, labels, n_splits=n_splits, pct_embargo=pct_embargo)
        results.append(result.to_dict())

    df = pd.DataFrame(results)
    if verbose:
        print("\n--- Survival Model Comparison ---")
        print(df.to_string(index=False))
    return df
