"""P25: Nested cross-validation for financial time series.

Inner loop tunes hyperparameters; outer loop evaluates performance.
NEVER mix inner and outer data — this prevents over-hyping from
iterative optimization on the same data.

Reference: "I Tried a Bunch of Things" (over-hyping), López de Prado (2018).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

from src.ml.purged_cv import PurgedKFold

logger = logging.getLogger(__name__)

SCORE_NAMES = ("accuracy", "auc", "sharpe", "profit_factor", "mre_gap")


@dataclass
class NestedCVResult:
    """Result of nested cross-validation."""

    outer_scores: list[float]
    outer_mean: float
    outer_std: float
    best_params_per_outer: list[dict[str, Any]]
    inner_scores_per_outer: list[list[float]]
    n_outer_folds: int
    n_inner_folds: int
    score_name: str
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"NestedCV: mean={self.outer_mean:.4f} ± {self.outer_std:.4f} "
            f"({self.score_name}) over {self.n_outer_folds} outer × {self.n_inner_folds} inner folds",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


class NestedPurgedCV:
    """Nested cross-validation using PurgedKFold for both loops.

    Outer loop: evaluates generalization (model trained on outer-train,
    evaluated on outer-test). Inner loop: tunes hyperparams within outer-train.

    Usage:
        from src.ml.nested_cv import NestedPurgedCV

        ncv = NestedPurgedCV(n_outer=5, n_inner=3, label_span=5)
        result = ncv.run(X, y, param_grid, build_model_fn, eval_fn)

    Attributes:
        n_outer: Number of outer folds.
        n_inner: Number of inner folds per outer split.
        label_span: Forward return horizon for purging.
        embargo_days: Days to embargo after test period.
        score_name: Metric name for reporting.
        maximize: True if higher score is better.
    """

    def __init__(
        self,
        n_outer: int = 5,
        n_inner: int = 3,
        label_span: int = 1,
        embargo_days: int = 0,
        score_name: str = "accuracy",
        maximize: bool = True,
        random_state: int | None = 42,
    ) -> None:
        if n_outer < 3:
            raise ValueError("n_outer must be >= 3")
        if n_inner < 2:
            raise ValueError("n_inner must be >= 2")
        if score_name not in SCORE_NAMES:
            raise ValueError(f"score_name must be one of {SCORE_NAMES}")

        self.n_outer = n_outer
        self.n_inner = n_inner
        self.label_span = label_span
        self.embargo_days = embargo_days
        self.score_name = score_name
        self.maximize = maximize
        self.random_state = random_state

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: dict[str, list[Any]],
        build_model: Callable[[dict[str, Any]], Any],
        eval_fn: Callable[[Any, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], float],
        fit_kwargs: dict[str, Any] | None = None,
    ) -> NestedCVResult:
        """Run nested purged cross-validation.

        Args:
            X: Feature matrix.
            y: Target labels.
            param_grid: Dict of param_name → [candidate values].
            build_model: Callable(params) → untrained model.
            eval_fn: Callable(model, X_train, y_train, X_test, y_test) → float score.
            fit_kwargs: Extra kwargs passed to model.fit().

        Returns:
            NestedCVResult with outer fold scores and best params.
        """
        outer_cv = PurgedKFold(
            n_splits=self.n_outer,
            label_span=self.label_span,
            embargo_days=self.embargo_days,
        )

        outer_scores: list[float] = []
        best_params_per_outer: list[dict[str, Any]] = []
        inner_scores_per_outer: list[list[float]] = []
        warnings: list[str] = []

        for outer_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X)):
            X_outer_train = X.iloc[outer_train_idx]
            y_outer_train = y.iloc[outer_train_idx]
            X_outer_test = X.iloc[outer_test_idx]
            y_outer_test = y.iloc[outer_test_idx]

            if len(X_outer_train) < 50:
                warnings.append(f"Outer fold {outer_idx}: only {len(X_outer_train)} train samples")
                continue

            inner_cv = PurgedKFold(
                n_splits=self.n_inner,
                label_span=self.label_span,
                embargo_days=self.embargo_days,
            )

            best_inner_score = float("-inf") if self.maximize else float("inf")
            best_params = {}
            inner_fold_scores: list[float] = []

            param_combinations = _grid_combinations(param_grid)
            for params in param_combinations:
                fold_scores: list[float] = []
                for inner_train_idx, inner_test_idx in inner_cv.split(X_outer_train):
                    X_inner_train = X_outer_train.iloc[inner_train_idx]
                    y_inner_train = y_outer_train.iloc[inner_train_idx]
                    X_inner_test = X_outer_train.iloc[inner_test_idx]
                    y_inner_test = y_outer_train.iloc[inner_test_idx]

                    if len(X_inner_train) < 30:
                        continue

                    model = build_model(params)
                    try:
                        if fit_kwargs:
                            model.fit(X_inner_train, y_inner_train, **fit_kwargs)
                        else:
                            model.fit(X_inner_train, y_inner_train)
                    except Exception:
                        continue

                    try:
                        score = eval_fn(
                            model, X_inner_train, y_inner_train, X_inner_test, y_inner_test
                        )
                        fold_scores.append(score)
                    except Exception:
                        continue

                if not fold_scores:
                    continue

                mean_score = float(np.mean(fold_scores))
                improved = (
                    mean_score > best_inner_score
                    if self.maximize
                    else mean_score < best_inner_score
                )
                if improved:
                    best_inner_score = mean_score
                    best_params = params.copy()
                    inner_fold_scores = fold_scores

            if not best_params:
                warnings.append(f"Outer fold {outer_idx}: no valid params found")
                continue

            inner_scores_per_outer.append(inner_fold_scores)

            model = build_model(best_params)
            try:
                if fit_kwargs:
                    model.fit(X_outer_train, y_outer_train, **fit_kwargs)
                else:
                    model.fit(X_outer_train, y_outer_train)
            except Exception as e:
                warnings.append(f"Outer fold {outer_idx}: final fit failed: {e}")
                continue

            try:
                outer_score = eval_fn(
                    model, X_outer_train, y_outer_train, X_outer_test, y_outer_test
                )
                outer_scores.append(outer_score)
                best_params_per_outer.append(best_params)
            except Exception as e:
                warnings.append(f"Outer fold {outer_idx}: eval failed: {e}")

        if not outer_scores:
            return NestedCVResult(
                outer_scores=[],
                outer_mean=0.0,
                outer_std=0.0,
                best_params_per_outer=[],
                inner_scores_per_outer=[],
                n_outer_folds=self.n_outer,
                n_inner_folds=self.n_inner,
                score_name=self.score_name,
                warnings=["All outer folds failed"] + warnings,
            )

        gap = (
            float(np.mean(inner_scores_per_outer[0])) - float(np.mean(outer_scores))
            if inner_scores_per_outer
            else 0
        )
        if gap > 0.10:
            warnings.append(f"Inner-outer gap {gap:.3f} > 0.10 — possible over-hyping")

        return NestedCVResult(
            outer_scores=outer_scores,
            outer_mean=float(np.mean(outer_scores)),
            outer_std=float(np.std(outer_scores, ddof=1)),
            best_params_per_outer=best_params_per_outer,
            inner_scores_per_outer=inner_scores_per_outer,
            n_outer_folds=self.n_outer,
            n_inner_folds=self.n_inner,
            score_name=self.score_name,
            warnings=warnings,
        )


def _grid_combinations(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Generate all combinations from a parameter grid."""
    import itertools

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    return combinations


def leave_one_group_out_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | np.ndarray,
    build_model: Callable[[], Any],
    eval_fn: Callable[[Any, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], float],
) -> list[float]:
    """Leave-One-Group-Out CV for time series by year or regime.

    Train on all groups except one; test on the held-out group.
    Training sets are supersets of earlier groups — correct for financial time series.
    """
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        raise ValueError("Need at least 2 groups for LOGO-CV")

    scores: list[float] = []
    for test_group in unique_groups:
        test_mask = np.array(groups) == test_group
        X_train = X[~test_mask]
        y_train = y[~test_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        if len(X_train) < 30 or len(X_test) < 10:
            logger.warning("Skipping group %s: insufficient data", test_group)
            continue

        model = build_model()
        model.fit(X_train, y_train)
        scores.append(eval_fn(model, X_train, y_train, X_test, y_test))

    return scores
