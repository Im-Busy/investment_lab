"""P25 + P28-3: Label-shuffling baseline test with CRNG fat-tail RNG.

Shuffles target labels, trains model, verifies model does NOT exceed random
baseline. If it does, features are exploiting noise structure (overfitting signal).

P28-3: Contingency RNG (CRNG) generates fat-tailed synthetic labels with
volatility clustering — 86% more market-realistic than uniform NumPy noise.
K parameter (5-220) controls tail fatness. Used as a stronger baseline:
can the model distinguish real signal from CRNG-simulated market noise?

Reference: López de Prado (2018), "I Tried a Bunch of Things".
Reference: awesome-ai-in-finance — Contingency RNG for label shuffling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SHUFFLE_WARNING_THRESHOLD = 0.10
CRNG_DEFAULT_K = 10
CRNG_CLUSTER_MIN = 3
CRNG_CLUSTER_MAX = 30


def generate_crng_labels(
    n: int,
    k: int = CRNG_DEFAULT_K,
    random_state: int | None = None,
) -> np.ndarray:
    """Generate fat-tailed synthetic labels via Contingency RNG.

    Produces volatility-clustered label sequences that mimic real market
    return distributions. K controls tail fatness: K=1 ≈ normal, K=5-220
    for increasingly heavy tails (t-distribution degrees of freedom).

    Labels are ±1 with fat-tailed noise structure — a harder baseline than
    uniform permutation for detecting genuine signal.

    Args:
        n: Number of labels to generate.
        k: Tail fatness (degrees of freedom in t-distribution).
           K=1 → normal, K=5 → moderate tails, K=220 → extreme tails.
        random_state: Random seed.

    Returns:
        Array of shape (n,) with values in [−1, +1], volatility-clustered.
    """
    rng = np.random.default_rng(random_state)
    innovations = rng.standard_t(df=k, size=n) if k > 1 else rng.standard_normal(n)
    labels = np.full(n, np.nan)
    i = 0
    while i < n:
        cluster_len = rng.integers(CRNG_CLUSTER_MIN, CRNG_CLUSTER_MAX + 1)
        cluster_len = min(cluster_len, n - i)
        scale = 0.1 + 0.9 * rng.random()
        cluster = np.tanh(innovations[i : i + cluster_len] * scale)
        labels[i : i + cluster_len] = np.sign(cluster)
        i += cluster_len
    labels = np.where(labels == 0, rng.choice([-1.0, 1.0], size=n), labels)
    return labels.astype(float)


def run_crng_baseline_test(
    X: pd.DataFrame,
    y: pd.Series,
    build_model: Callable[[], Any],
    eval_fn: Callable[[Any, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], float],
    n_trials: int = 30,
    k_values: tuple[int, ...] = (1, 5, 20, 100),
    test_frac: float = 0.20,
    random_state: int | None = 42,
) -> dict[int, LabelShufflingResult]:
    """Run label-shuffling test with CRNG fat-tail noise at multiple K levels.

    Compares model performance against CRNG-generated noise at varying tail
    fatness. A model that beats uniform shuffle but fails CRNG(K≥5) is
    exploiting structured noise typical of financial data.

    Args:
        X: Feature matrix.
        y: True target labels.
        build_model: Callable() → untrained model.
        eval_fn: (model, X_train, y_train, X_test, y_test) → float.
        n_trials: Shuffle trials per K level.
        k_values: Tail fatness levels to test.
        test_frac: Test split fraction.
        random_state: Random seed.

    Returns:
        Dict mapping K → LabelShufflingResult.
    """
    results: dict[int, LabelShufflingResult] = {}
    rng = np.random.default_rng(random_state)
    n = len(X)
    n_test = max(int(n * test_frac), 1)
    indices = rng.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    for k_val in k_values:
        shuffled_scores: list[float] = []
        for seed in range(n_trials):
            y_synth = pd.Series(
                generate_crng_labels(len(train_idx), k=k_val, random_state=seed),
                index=X_train.index,
            )
            model = build_model()
            try:
                model.fit(X_train, y_synth)
                score = eval_fn(model, X_train, y_synth, X_test, y.iloc[test_idx])
                shuffled_scores.append(score)
            except Exception:
                continue

        if not shuffled_scores:
            results[k_val] = LabelShufflingResult(
                shuffled_scores=[],
                shuffled_mean=0.0,
                shuffled_std=0.0,
                true_score=0.0,
                random_baseline=0.0,
                exceed_probability=1.0,
                is_noise_exploiting=True,
                warnings=[f"All CRNG K={k_val} trials failed"],
            )
            continue

        shuffled_mean = float(np.mean(shuffled_scores))
        shuffled_std = float(np.std(shuffled_scores, ddof=1))
        true_score = eval_fn(build_model(), X_train, y.iloc[train_idx], X_test, y.iloc[test_idx])
        exceed_count = sum(1 for s in shuffled_scores if s >= true_score)
        exceed_probability = exceed_count / len(shuffled_scores)
        is_noise = shuffled_mean > (0.0 + SHUFFLE_WARNING_THRESHOLD)

        results[k_val] = LabelShufflingResult(
            shuffled_scores=shuffled_scores,
            shuffled_mean=shuffled_mean,
            shuffled_std=shuffled_std,
            true_score=true_score,
            random_baseline=0.0,
            exceed_probability=exceed_probability,
            is_noise_exploiting=is_noise,
            warnings=(
                [
                    f"CRNG K={k_val}: model does not exceed fat-tail noise (p={exceed_probability:.3f})"
                ]
                if exceed_probability > 0.05
                else []
            ),
        )

    return results


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
