"""P24-10: Profit Mirage — counterfactual evaluation of model predictions.

Perturb input features and measure prediction consistency. Low consistency
indicates memorization (false alpha). Based on Fisher et al. 2020 "All
That Glitters Is Not Gold" counterfactual methodology.

Detects:
  - Input memorization: small perturbations flip predictions
  - Feature fragility: single-feature drops change prediction dramatically
  - Temporal instability: predictions shift under timeline perturbation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MirageFeatureImpact:
    """Per-feature counterfactual impact."""

    feature: str
    original_prediction: float
    perturbed_prediction: float
    delta: float
    delta_pct: float
    fragile: bool  # |delta| > 2 stdev of deltas


@dataclass
class MirageReport:
    """Profit Mirage evaluation result."""

    consistency_score: float  # 0-1, higher = more consistent
    input_stability: float  # perturbation consistency
    feature_fragility: float  # 1.0 - fraction of fragile features
    temporal_stability: float  # timeline perturbation consistency
    fragile_features: List[MirageFeatureImpact] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    is_mirage: bool = False  # True if model likely memorizing

    def summary(self) -> str:
        lines = [
            f"Consistency: {self.consistency_score:.2%}",
            f"Input Stability: {self.input_stability:.2%}",
            f"Feature Fragility: {self.feature_fragility:.2%}",
            f"Temporal Stability: {self.temporal_stability:.2%}",
            f"Fragile features: {len(self.fragile_features)}",
        ]
        for w in self.warnings:
            lines.append(f"WARN: {w}")
        if self.is_mirage:
            lines.append("*** PROFIT MIRAGE DETECTED — model may be memorizing ***")
        return "\n".join(lines)


def _feature_perturbation(
    features: np.ndarray,
    feature_names: List[str],
    predict_fn: Callable[[np.ndarray], np.ndarray],
    noise_scale: float = 0.05,
    n_perturbations: int = 10,
) -> Tuple[float, List[MirageFeatureImpact]]:
    """Perturb each feature with Gaussian noise, track prediction change.

    Returns:
        (input_stability, list of per-feature impacts)
    """
    if features.shape[1] != len(feature_names) or features.shape[0] < 1:
        return 1.0, []

    base_preds = predict_fn(features)
    impacts: List[MirageFeatureImpact] = []
    all_deltas: List[float] = []

    for j, name in enumerate(feature_names):
        feature_std = np.std(features[:, j])
        noise_std = noise_scale * feature_std if feature_std > 1e-10 else noise_scale

        perturbed_deltas: List[float] = []
        for _ in range(n_perturbations):
            feats_perturbed = features.copy()
            feats_perturbed[:, j] += np.random.default_rng().normal(0, noise_std, features.shape[0])
            pert_preds = predict_fn(feats_perturbed)
            delta = float(np.mean(np.abs(pert_preds - base_preds)))
            perturbed_deltas.append(delta)

        mean_delta = float(np.mean(perturbed_deltas))
        base = float(np.mean(base_preds))
        all_deltas.append(mean_delta)
        impacts.append(
            MirageFeatureImpact(
                feature=name,
                original_prediction=base,
                perturbed_prediction=base + mean_delta,
                delta=mean_delta,
                delta_pct=float(mean_delta / (abs(base) + 1e-10)),
                fragile=False,
            )
        )

    if all_deltas:
        mean_d = float(np.mean(all_deltas))
        std_d = float(np.std(all_deltas)) + 1e-10
        for imp in impacts:
            imp.fragile = abs(imp.delta - mean_d) > 2.0 * std_d

    fragile_count = sum(1 for i in impacts if i.fragile)
    total = max(len(impacts), 1)
    fragility = 1.0 - fragile_count / total

    if impacts:
        mean_abs_delta = float(np.mean([abs(i.delta) for i in impacts]))
        base_mag = abs(float(np.mean(base_preds))) + 1e-10
        stability = float(np.clip(1.0 - mean_abs_delta / base_mag, 0.0, 1.0))
    else:
        stability = 1.0

    return stability, impacts


def _temporal_perturbation(
    features: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    n_shifts: int = 5,
    shift_size: int = 5,
) -> float:
    """Shift prediction window, measure stability.

    Returns:
        temporal_stability: 0-1 score.
    """
    n = features.shape[0]
    if n < shift_size * 2:
        return 1.0

    base_preds = predict_fn(features[shift_size:])
    deltas: List[float] = []

    for i in range(1, n_shifts + 1):
        shift = min(i * shift_size, n - shift_size)
        if shift <= 0:
            continue
        shifted_preds = predict_fn(features[: n - shift])
        common_len = min(len(base_preds), len(shifted_preds))
        if common_len < 5:
            continue
        delta = float(np.mean(np.abs(base_preds[:common_len] - shifted_preds[:common_len])))
        deltas.append(delta)

    if not deltas:
        return 1.0

    mean_delta = float(np.mean(deltas))
    base_mag = float(np.mean(np.abs(base_preds))) + 1e-10
    return float(np.clip(1.0 - mean_delta / base_mag, 0.0, 1.0))


def run_profit_mirage(
    features: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    feature_names: Optional[List[str]] = None,
    noise_scale: float = 0.05,
    stability_threshold: float = 0.70,
) -> MirageReport:
    """Run full Profit Mirage counterfactual evaluation.

    Args:
        features: (n_samples, n_features) input array.
        predict_fn: Model predict_proba or equivalent.
        feature_names: Feature names for reporting.
        noise_scale: Gaussian noise std as fraction of feature std.
        stability_threshold: Below this → warn of potential memorization.

    Returns:
        MirageReport with all stability metrics.
    """
    if feature_names is None:
        feature_names = [f"f_{i}" for i in range(features.shape[1])]

    warnings: List[str] = []

    input_stability, fragile_features = _feature_perturbation(
        features, feature_names, predict_fn, noise_scale
    )

    temporal_stability = _temporal_perturbation(features, predict_fn)

    feature_fragility = 1.0 - sum(1 for f in fragile_features if f.fragile) / max(
        len(fragile_features), 1
    )
    consistency = float((input_stability + temporal_stability + feature_fragility) / 3.0)

    if consistency < stability_threshold:
        warnings.append(
            f"Low consistency ({consistency:.2%} < {stability_threshold:.0%}) "
            "— predictions may reflect memorization rather than signal"
        )

    if input_stability < 0.6:
        warnings.append(
            f"Low input stability ({input_stability:.2%}) — perturbation flips predictions"
        )
    if feature_fragility < 0.7:
        warnings.append(
            f"Feature fragility ({feature_fragility:.2%}) — single-feature drops shift output"
        )
    if temporal_stability < 0.6:
        warnings.append(f"Low temporal stability ({temporal_stability:.2%}) — timeline-sensitive")

    is_mirage = consistency < stability_threshold or (
        input_stability < 0.5 and temporal_stability < 0.5
    )

    return MirageReport(
        consistency_score=consistency,
        input_stability=input_stability,
        feature_fragility=feature_fragility,
        temporal_stability=temporal_stability,
        fragile_features=fragile_features,
        warnings=warnings,
        is_mirage=is_mirage,
    )
