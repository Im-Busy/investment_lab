"""P25: Label-shuffling baseline test for feature selection validation.

Shuffles target labels, trains model, verifies model does NOT exceed random
baseline. If it does, features are exploiting noise structure (overfitting signal).

Reference: López de Prado (2018), "I Tried a Bunch of Things".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SHUFFLE_WARNING_THRESHOLD = 0.10


@dataclass
class LabelShufflingResult:
    """Result of label-shuffling baseline test."""

    shuffled_scores: list[float]
    shuffled_mean: float
    shuffled_std: float
    true_score: float
    random_baseline: float
    exceed_probability: float  # P(shuffled ≥ true_score)
    is_noise_exploiting: bool
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"LabelShuffling: true={self.true_score:.4f} vs "
            f"shuffled={self.shuffled_mean:.4f}±{self.shuffled_std:.4f}, "
            f"n_shuffles={len(self.shuffled_scores)}, "
            f"p(exceed)={self.exceed_probability:.3f}, "
            f"noise_exploit={'YES' if self.is_noise_exploiting else 'no'}"
        )


def run_label_shuffling_test(
    X: pd.DataFrame,
    y: pd.Series,
    build_model: Callable[[], Any],
    eval_fn: Callable[[Any, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], float],
    n_shuffles: int = 50,
    test_frac: float = 0.20,
    random_state: int | None = 42,
) -> LabelShufflingResult:
    """Shuffle target labels, retrain, verify model doesn't beat random.

    If shuffled-label models perform similar to true labels → features have
    no genuine signal. If shuffled models AND true perform above random →
    features are exploiting noise structure.

    Args:
        X: Feature matrix.
        y: Target labels.
        build_model: Callable() → untrained model.
        eval_fn: (model, X_train, y_train, X_test, y_test) → float.
        n_shuffles: Number of shuffle iterations.
        test_frac: Test split fraction.
        random_state: Random seed.

    Returns:
        LabelShufflingResult.
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    n_test = max(int(n * test_frac), 1)
    indices = rng.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    model = build_model()
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        logger.error("True-label fit failed: %s", e)
        return LabelShufflingResult(
            shuffled_scores=[],
            shuffled_mean=0.0,
            shuffled_std=0.0,
            true_score=0.0,
            random_baseline=0.0,
            exceed_probability=1.0,
            is_noise_exploiting=True,
            warnings=[f"True-label fit failed: {e}"],
        )

    true_score = eval_fn(model, X_train, y_train, X_test, y_test)

    shuffled_scores: list[float] = []
    for _ in range(n_shuffles):
        y_shuffled = pd.Series(rng.permutation(y_train.values), index=y_train.index)
        model_shuffle = build_model()
        try:
            model_shuffle.fit(X_train, y_shuffled)
            score = eval_fn(model_shuffle, X_train, y_shuffled, X_test, y_test)
            shuffled_scores.append(score)
        except Exception:
            continue

    if not shuffled_scores:
        return LabelShufflingResult(
            shuffled_scores=[],
            shuffled_mean=0.0,
            shuffled_std=0.0,
            true_score=true_score,
            random_baseline=0.0,
            exceed_probability=1.0,
            is_noise_exploiting=True,
            warnings=["All shuffle trials failed"],
        )

    shuffled_mean = float(np.mean(shuffled_scores))
    shuffled_std = float(np.std(shuffled_scores, ddof=1))

    y_rand = pd.Series(rng.choice([-1, 1], size=len(y_test)), index=y_test.index)
    random_model = build_model()
    try:
        random_model.fit(
            X_train, y_rand.iloc[: len(X_train)] if len(y_rand) >= len(X_train) else y_train
        )
    except Exception:
        pass
    random_baseline = eval_fn(random_model, X_train, y_train, X_test, y_rand)

    exceed_count = sum(1 for s in shuffled_scores if s >= true_score)
    exceed_probability = exceed_count / len(shuffled_scores)

    is_noise_exploiting = shuffled_mean > (random_baseline + SHUFFLE_WARNING_THRESHOLD)

    warnings: list[str] = []
    if is_noise_exploiting:
        warnings.append(
            f"Shuffled mean ({shuffled_mean:.4f}) > random baseline "
            f"({random_baseline:.4f}) + {SHUFFLE_WARNING_THRESHOLD} — "
            "features may exploit noise structure"
        )
    if exceed_probability > 0.05:
        warnings.append(
            f"p(exceed)={exceed_probability:.3f} > 0.05 — true score not "
            "significantly different from shuffled distribution"
        )
    if abs(true_score - shuffled_mean) < 0.02:
        warnings.append(
            f"True-shuffled gap={abs(true_score - shuffled_mean):.4f} < 0.02 — "
            "features may have no genuine signal"
        )

    return LabelShufflingResult(
        shuffled_scores=shuffled_scores,
        shuffled_mean=shuffled_mean,
        shuffled_std=shuffled_std,
        true_score=true_score,
        random_baseline=random_baseline,
        exceed_probability=exceed_probability,
        is_noise_exploiting=is_noise_exploiting,
        warnings=warnings,
    )
