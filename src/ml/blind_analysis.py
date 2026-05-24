"""P25: Blind analysis protocol — optimize on scrambled labels, evaluate on true.

Shuffles target labels during hyperparameter tuning to prevent over-hyping.
After all hyperparameter decisions are final, evaluate once on true labels.

Reference: "I Tried a Bunch of Things" — even 1-shot hyperparameter selection
introduces ~1.6% measurement bias.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _build_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Generate all combinations from a parameter grid."""
    from itertools import product

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


@dataclass
class BlindAnalysisResult:
    """Result of blind analysis protocol."""

    scrambled_best_params: dict[str, Any]
    scrambled_best_score: float
    scrambled_all_scores: list[float]
    true_score: float
    true_random_baseline: float
    score_delta: float  # true_score - true_random_baseline
    is_honest: bool  # True if model beats random on true labels
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Blind Analysis: scrambled_best={self.scrambled_best_score:.4f} → true={self.true_score:.4f}",
            f"  random_baseline={self.true_random_baseline:.4f}, delta={self.score_delta:.4f}",
            f"  honest={'YES' if self.is_honest else 'NO — model performs at random'}",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def run_blind_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict[str, list[Any]],
    build_model: Callable[[dict[str, Any]], Any],
    eval_fn: Callable[[Any, pd.DataFrame, pd.Series], float],
    n_trials: int = 10,
    val_frac: float = 0.20,
    maximize: bool = True,
    random_state: int | None = 42,
) -> BlindAnalysisResult:
    """Run blind analysis: tune on scrambled labels, evaluate on true labels.

    Protocol:
    1. Split X into train_val and holdout.
    2. For each hyperparam combo, evaluate on VALIDATION with SCRAMBLED labels.
    3. Select best params based on scrambled-label performance only.
    4. Evaluate ONCE on true holdout labels with best params.
    5. Report delta vs random baseline.

    Args:
        X: Feature matrix.
        y: Target labels.
        param_grid: Dict of param_name → [candidate values].
        build_model: Callable(params) → untrained model.
        eval_fn: Callable(model, X, y) → float score.
        n_trials: Number of random trials to compute random baseline.
        val_frac: Fraction for holdout split.
        maximize: True if higher eval_fn score is better.
        random_state: Random seed.

    Returns:
        BlindAnalysisResult.
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    n_val = max(int(n * val_frac), 1)

    idx = rng.permutation(n)
    train_val_idx = idx[:-n_val]
    holdout_idx = idx[-n_val:]

    X_train_val = X.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    X_holdout = X.iloc[holdout_idx]
    y_holdout = y.iloc[holdout_idx]

    y_scrambled = y_train_val.copy()
    y_scrambled = pd.Series(rng.permutation(y_scrambled.values), index=y_scrambled.index)

    param_combinations = _build_grid(param_grid)
    if not param_combinations:
        raise ValueError("param_grid produced no combinations")

    best_score = float("-inf") if maximize else float("inf")
    best_params: dict[str, Any] = {}
    all_scores: list[float] = []

    for params in param_combinations:
        model = build_model(params)
        try:
            model.fit(X_train_val, y_scrambled)
            score = eval_fn(model, X_train_val, y_scrambled)
            all_scores.append(score)
            improved = score > best_score if maximize else score < best_score
            if improved:
                best_score = score
                best_params = params.copy()
        except Exception as e:
            logger.debug("Param combo %s failed: %s", params, e)
            continue

    if not best_params:
        return BlindAnalysisResult(
            scrambled_best_params={},
            scrambled_best_score=0.0,
            scrambled_all_scores=[],
            true_score=0.0,
            true_random_baseline=0.0,
            score_delta=0.0,
            is_honest=False,
            warnings=["No valid params found during blind tuning"],
        )

    model = build_model(best_params)
    model.fit(
        X_holdout.iloc[:-1] if len(X_holdout) > 1 else X_train_val,
        y_holdout.iloc[:-1] if len(y_holdout) > 1 else y_train_val,
    )
    true_score = eval_fn(model, X_holdout, y_holdout)

    random_scores: list[float] = []
    for _ in range(n_trials):
        y_rand = y_holdout.copy()
        y_rand = pd.Series(rng.permutation(y_rand.values), index=y_rand.index)
        random_model = build_model(best_params)
        try:
            random_model.fit(X_train_val, y_train_val)
            random_scores.append(eval_fn(random_model, X_holdout, y_rand))
        except Exception:
            continue

    true_random_baseline = float(np.mean(random_scores)) if random_scores else 0.0
    score_delta = true_score - true_random_baseline
    is_honest = score_delta > 0

    warnings = []
    scrambled_mean = float(np.mean(all_scores)) if all_scores else 0
    if abs(scrambled_mean - true_random_baseline) < 0.01 and true_score > scrambled_mean + 0.02:
        warnings.append(
            "Model may be exploiting noise — scrambled score near random but true score elevated"
        )

    return BlindAnalysisResult(
        scrambled_best_params=best_params,
        scrambled_best_score=best_score,
        scrambled_all_scores=all_scores,
        true_score=true_score,
        true_random_baseline=true_random_baseline,
        score_delta=score_delta,
        is_honest=is_honest,
        warnings=warnings,
    )
