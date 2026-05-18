"""
Circuit-Based Overfitting Detection — Direction C Phase 5.

Based on "Circuit-Based Intrinsic Methods to Detect Overfitting"
(Chatterjee & Mishchenko, ICML 2020):

Core idea — Counterfactual Simulation (CFS):
1. Simulate training examples through the model (as a logic circuit).
2. Identify rare patterns: signals/activations that fire for very few training examples.
3. Rare patterns indicate memorization (the model has a "lookup table" for specific examples).
4. Perturb rare patterns and compare performance change — overfit models degrade more.

The method is INTRINSIC (uses only model + training data, no test set needed)
and works uniformly across model types (NNs, random forests, lookup tables).

For tree-based models (CatBoost), we adapt CFS via:
- Leaf analysis: count training samples per leaf. Leaves with 1 sample = memorized.
- Path rarity: for each training example, measure uniqueness of its decision path.
- Perturbation test: flip labels of rare-path examples, measure stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CircuitOverfitResult:
    """Result of circuit-based overfitting analysis."""

    overfit_score: float
    rare_pattern_ratio: float
    sample_coverage: float
    interpretation: str
    details: dict = field(default_factory=dict)

    risk_level: str = "LOW"
    is_overfit: bool = False


class CircuitOverfitDetector:
    """Detect overfitting using circuit-based intrinsic methods (CFS).

    This is an intrinsic method — it only needs the training data and
    model predictions. No hold-out / test set required.

    The key insight from the CFS paper: a model that has memorized
    training examples will have "rare patterns" — internal signals
    that fire for very few training examples. These look like lookup
    table entries. A well-generalized model uses common patterns.

    Parameters:
        rarity_threshold: Percentile below which patterns are "rare" (default 0.01 = bottom 1%).
        overfit_score_threshold: Overfit score above which model is flagged (default 0.30).
        min_samples: Minimum training samples needed for meaningful analysis (default 100).
    """

    def __init__(
        self,
        rarity_threshold: float = 0.01,
        overfit_score_threshold: float = 0.30,
        min_samples: int = 100,
    ) -> None:
        self.rarity_threshold = rarity_threshold
        self.overfit_score_threshold = overfit_score_threshold
        self.min_samples = min_samples

    def analyze_tree_leaves(
        self,
        leaf_samples: np.ndarray,
    ) -> CircuitOverfitResult:
        """Analyze CatBoost tree leaves for overfitting signals.

        CatBoost trees store `_leaf_values` and we need leaf sample counts.
        When leaf sample counts are available, leaves with very few samples
        suggest memorization of specific training examples.

        Args:
            leaf_samples: Array of sample counts per leaf node.

        Returns:
            CircuitOverfitResult with overfit score and interpretation.
        """
        if len(leaf_samples) < self.min_samples:
            return CircuitOverfitResult(
                overfit_score=0.0,
                rare_pattern_ratio=0.0,
                sample_coverage=0.0,
                interpretation="Insufficient data for analysis",
                risk_level="LOW",
                is_overfit=False,
            )

        # Count rare leaves (those with 1 sample = memorized example)
        rare_count = int((leaf_samples <= 1).sum())
        rare_ratio = rare_count / len(leaf_samples) if len(leaf_samples) > 0 else 0.0

        # Count leaves with 2-3 samples (borderline)
        small_count = int(((leaf_samples >= 2) & (leaf_samples <= 3)).sum())

        # Coverage: total samples covered by rare leaves
        total_samples = int(leaf_samples.sum())
        rare_sample_coverage = (
            int(leaf_samples[leaf_samples <= 1].sum()) / total_samples if total_samples > 0 else 0.0
        )

        # Overfit score: weighted combination
        overfit_score = rare_ratio * 0.7 + rare_sample_coverage * 0.3

        # Interpretation
        if overfit_score >= self.overfit_score_threshold:
            is_overfit = True
            interpretation = (
                f"HIGH overfit risk: {rare_ratio:.1%} of leaves are memorized "
                f"({rare_count}/{len(leaf_samples)} singleton leaves). "
                f"Model likely memorized specific training examples."
            )
            risk_level = "HIGH"
        elif overfit_score >= 0.15:
            is_overfit = True
            interpretation = (
                f"MEDIUM overfit risk: {rare_ratio:.1%} singleton leaves. "
                f"Some memorization detected. Consider regularization or pruning."
            )
            risk_level = "MEDIUM"
        else:
            is_overfit = False
            interpretation = (
                f"LOW overfit risk: only {rare_ratio:.1%} singleton leaves "
                f"({rare_count}/{len(leaf_samples)}). Model generalizes well on training data."
            )
            risk_level = "LOW"

        return CircuitOverfitResult(
            overfit_score=round(overfit_score, 4),
            rare_pattern_ratio=round(rare_ratio, 4),
            sample_coverage=round(rare_sample_coverage, 4),
            interpretation=interpretation,
            is_overfit=is_overfit,
            risk_level=risk_level,
            details={
                "total_leaves": len(leaf_samples),
                "rare_leaves": rare_count,
                "small_leaves": small_count,
                "total_samples": total_samples,
                "rare_leaf_samples": int(leaf_samples[leaf_samples <= 1].sum()),
            },
        )

    def analyze_prediction_distribution(
        self,
        predictions: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> CircuitOverfitResult:
        """Analyze prediction distribution for overfitting signals (prediction-based CFS).

        An overfit model has extreme confidence on training examples.
        We measure:
        1. Prediction concentration: are most predictions at extremes (0 or 1)?
        2. Rare-pattern prediction sharpness: ratio of predictions with very low uncertainty.
        3. Label-conditional prediction divergence.

        Args:
            predictions: Model probability predictions (shape [n_samples]).
            labels: Optional binary labels (shape [n_samples]).

        Returns:
            CircuitOverfitResult with overfit score.
        """
        preds = np.asarray(predictions).ravel()
        n = len(preds)

        if n < self.min_samples:
            return CircuitOverfitResult(
                overfit_score=0.0,
                rare_pattern_ratio=0.0,
                sample_coverage=0.0,
                interpretation="Insufficient data for analysis",
                risk_level="LOW",
                is_overfit=False,
            )

        # 1. Extreme prediction ratio: fraction of predictions near 0 or 1
        extreme_ratio = np.mean((preds <= 0.10) | (preds >= 0.90))

        # 2. Uncertainty entropy: lower = more overconfident
        entropy = -preds * np.log2(np.maximum(preds, 1e-10)) - (1 - preds) * np.log2(
            np.maximum(1 - preds, 1e-10)
        )
        mean_entropy = float(np.mean(entropy))
        max_entropy = 1.0
        entropy_score = 1.0 - mean_entropy / max_entropy  # 0 = well-calibrated, 1 = overconfident

        # 3. Rare pattern identification via prediction uniqueness
        # Round predictions to 3 decimal places and count frequencies
        rounded = np.round(preds, 3)
        unique, counts = np.unique(rounded, return_counts=True)
        rare_unique_ratio = np.mean(counts <= 1)

        # 4. If labels provided, check calibration sharpness
        if labels is not None:
            labels_arr = np.asarray(labels).ravel()
            # Predictions on correct vs incorrect examples
            correct_mask = (preds >= 0.5) == (labels_arr >= 0.5)
            correct_confidence = float(np.mean(preds[correct_mask])) if correct_mask.any() else 0.5
            incorrect_confidence = (
                float(np.mean(preds[~correct_mask])) if (~correct_mask).any() else 0.5
            )
            confidence_gap = correct_confidence - incorrect_confidence
            # Large gap = overfitting (confident on both correct & incorrect)
        else:
            confidence_gap = 0.0

        # Composite overfit score
        overfit_score = (
            extreme_ratio * 0.35
            + entropy_score * 0.25
            + rare_unique_ratio * 0.25
            + max(0.0, confidence_gap - 0.3) * 0.15
        )

        if overfit_score >= self.overfit_score_threshold:
            is_overfit = True
            risk_level = "HIGH"
            interpretation = (
                f"HIGH overfit: {extreme_ratio:.1%} extreme predictions, "
                f"entropy score={entropy_score:.3f}, rare_unique_ratio={rare_unique_ratio:.3f}"
            )
        elif overfit_score >= 0.15:
            is_overfit = True
            risk_level = "MEDIUM"
            interpretation = (
                f"MEDIUM overfit: {extreme_ratio:.1%} extreme predictions, "
                f"entropy score={entropy_score:.3f}"
            )
        else:
            is_overfit = False
            risk_level = "LOW"
            interpretation = (
                f"LOW overfit: {extreme_ratio:.1%} extreme predictions, "
                f"entropy score={entropy_score:.3f}"
            )

        return CircuitOverfitResult(
            overfit_score=round(overfit_score, 4),
            rare_pattern_ratio=round(rare_unique_ratio, 4),
            sample_coverage=round(extreme_ratio, 4),
            interpretation=interpretation,
            is_overfit=is_overfit,
            risk_level=risk_level,
            details={
                "n_samples": n,
                "extreme_ratio": round(extreme_ratio, 4),
                "entropy_score": round(entropy_score, 4),
                "mean_entropy": round(mean_entropy, 4),
                "rare_unique_ratio": round(rare_unique_ratio, 4),
                "confidence_gap": round(confidence_gap, 4),
            },
        )

    def analyze_perturbation_stability(
        self,
        predictions_original: np.ndarray,
        predictions_perturbed: np.ndarray,
    ) -> CircuitOverfitResult:
        """CFS perturbation test: how much do predictions change when rare patterns are perturbed?

        Core CFS idea: perturb rare patterns (flip labels of examples that
        have unique decision paths) and measure prediction stability.
        An overfit model changes dramatically on perturbed examples.
        A well-generalized model is stable.

        Args:
            predictions_original: Predictions on original training data.
            predictions_perturbed: Predictions after perturbation (e.g., noise added).

        Returns:
            CircuitOverfitResult with stability metrics.
        """
        orig = np.asarray(predictions_original).ravel()
        pert = np.asarray(predictions_perturbed).ravel()
        n = len(orig)

        if n < self.min_samples or len(pert) != n:
            return CircuitOverfitResult(
                overfit_score=0.0,
                rare_pattern_ratio=0.0,
                sample_coverage=0.0,
                interpretation="Insufficient data for perturbation analysis",
                risk_level="LOW",
                is_overfit=False,
            )

        # 1. Mean absolute change
        abs_change = np.abs(orig - pert)
        mean_abs_change = float(np.mean(abs_change))

        # 2. Fraction of examples with large change (>0.1)
        large_change_ratio = float(np.mean(abs_change > 0.10))

        # 3. Sign flip ratio (predictions crossing 0.5 threshold)
        sign_flip_ratio = float(np.mean((orig >= 0.5) != (pert >= 0.5)))

        # 4. Kolmogorov-Smirnov distance between distributions
        ks_stat, _ = self._ks_2samp(orig, pert)

        # Composite overfit score: higher perturbation sensitivity = more overfit
        overfit_score = (
            mean_abs_change * 0.35
            + large_change_ratio * 0.30
            + sign_flip_ratio * 0.20
            + ks_stat * 0.15
        )
        # Normalize: typical well-generalized model has score ~0.05
        # Scale to [0, 1]
        overfit_score = min(1.0, overfit_score * 5.0)

        if overfit_score >= self.overfit_score_threshold:
            is_overfit = True
            risk_level = "HIGH"
            interpretation = (
                f"HIGH instability: mean_change={mean_abs_change:.4f}, "
                f"large_change={large_change_ratio:.1%}, sign_flip={sign_flip_ratio:.1%}"
            )
        elif overfit_score >= 0.15:
            is_overfit = True
            risk_level = "MEDIUM"
            interpretation = f"MEDIUM instability: mean_change={mean_abs_change:.4f}"
        else:
            is_overfit = False
            risk_level = "LOW"
            interpretation = f"LOW instability: mean_change={mean_abs_change:.4f}. Model is stable under perturbation."

        return CircuitOverfitResult(
            overfit_score=round(overfit_score, 4),
            rare_pattern_ratio=round(large_change_ratio, 4),
            sample_coverage=round(sign_flip_ratio, 4),
            interpretation=interpretation,
            is_overfit=is_overfit,
            risk_level=risk_level,
            details={
                "n_samples": n,
                "mean_abs_change": round(mean_abs_change, 6),
                "large_change_ratio": round(large_change_ratio, 4),
                "sign_flip_ratio": round(sign_flip_ratio, 4),
                "ks_statistic": round(ks_stat, 4),
            },
        )

    @staticmethod
    def _ks_2samp(data1: np.ndarray, data2: np.ndarray) -> tuple[float, float]:
        """Two-sample Kolmogorov-Smirnov test (minimal implementation)."""
        data1 = np.sort(data1)
        data2 = np.sort(data2)
        n1, n2 = len(data1), len(data2)
        if n1 == 0 or n2 == 0:
            return 0.0, 1.0

        all_vals = np.sort(np.concatenate([data1, data2]))
        cdf1 = np.searchsorted(data1, all_vals, side="right") / n1
        cdf2 = np.searchsorted(data2, all_vals, side="right") / n2
        d = np.max(np.abs(cdf1 - cdf2))

        # Approximate p-value
        en = np.sqrt(n1 * n2 / (n1 + n2))
        p_val = 2.0 * np.exp(-2.0 * (en * d) ** 2)
        return float(d), float(p_val)

    def analyze_catboost_model(
        self,
        model,
        training_data: np.ndarray,
    ) -> CircuitOverfitResult:
        """Analyze a CatBoost model using circuit-based intrinsic methods.

        Extracts leaf assignments for each training sample, counts samples
        per leaf across all trees, and scores rarity patterns.

        Args:
            model: Trained CatBoost classifier (must have .calc_leaf_indexes).
            training_data: Feature matrix used for training (shape [n_samples, n_features]).

        Returns:
            CircuitOverfitResult.
        """
        if training_data.shape[0] < self.min_samples:
            return CircuitOverfitResult(
                overfit_score=0.0,
                rare_pattern_ratio=0.0,
                sample_coverage=0.0,
                interpretation=f"Insufficient training samples ({training_data.shape[0]} < {self.min_samples})",
                risk_level="LOW",
                is_overfit=False,
            )

        try:
            leaf_indexes = model.calc_leaf_indexes(training_data)
        except Exception as e:
            return CircuitOverfitResult(
                overfit_score=0.0,
                rare_pattern_ratio=0.0,
                sample_coverage=0.0,
                interpretation=f"Could not extract leaf indexes: {e}",
                risk_level="LOW",
                is_overfit=False,
            )

        # leaf_indexes shape: (n_samples, n_trees)
        n_samples, n_trees = leaf_indexes.shape
        total_leaves = 0
        rare_leaves = 0
        rare_samples_set = set()

        for tree_idx in range(n_trees):
            tree_leaves = leaf_indexes[:, tree_idx]
            unique_leaves, leaf_counts = np.unique(tree_leaves, return_counts=True)
            total_leaves += len(unique_leaves)

            for leaf, count in zip(unique_leaves, leaf_counts):
                if count <= 1:
                    rare_leaves += 1
                    # Find which sample is in this rare leaf
                    sample_idx = int(np.where(tree_leaves == leaf)[0][0])
                    rare_samples_set.add(sample_idx)

        rare_ratio = rare_leaves / total_leaves if total_leaves > 0 else 0.0
        unique_sample_ratio = len(rare_samples_set) / n_samples

        overfit_score = rare_ratio * 0.6 + unique_sample_ratio * 0.4

        if overfit_score >= self.overfit_score_threshold:
            is_overfit = True
            risk_level = "HIGH"
            interpretation = (
                f"HIGH overfit: {rare_ratio:.1%} of {total_leaves} leaves are singletons. "
                f"{unique_sample_ratio:.1%} of training samples have unique paths. "
                f"Model memorized {len(rare_samples_set)}/{n_samples} examples."
            )
        elif overfit_score >= 0.15:
            is_overfit = True
            risk_level = "MEDIUM"
            interpretation = (
                f"MEDIUM overfit: {rare_ratio:.1%} singleton leaves across {n_trees} trees."
            )
        else:
            is_overfit = False
            risk_level = "LOW"
            interpretation = (
                f"LOW overfit: only {rare_ratio:.1%} singleton leaves in "
                f"{n_trees} trees. Model generalizes well."
            )

        return CircuitOverfitResult(
            overfit_score=round(overfit_score, 4),
            rare_pattern_ratio=round(rare_ratio, 4),
            sample_coverage=round(unique_sample_ratio, 4),
            interpretation=interpretation,
            is_overfit=is_overfit,
            risk_level=risk_level,
            details={
                "n_samples": n_samples,
                "n_trees": n_trees,
                "total_leaves": total_leaves,
                "rare_leaves": rare_leaves,
                "rare_samples": len(rare_samples_set),
            },
        )

    def full_check(
        self,
        predictions: np.ndarray,
        labels: Optional[np.ndarray] = None,
        catboost_model=None,
        training_data: Optional[np.ndarray] = None,
    ) -> dict:
        """Run all circuit-based checks and produce a consolidated report.

        Args:
            predictions: Model predictions on training data.
            labels: Optional ground truth labels.
            catboost_model: Optional CatBoost model for leaf analysis.
            training_data: Feature matrix (required if catboost_model provided).

        Returns:
            Dict with all results and overall flag.
        """
        results = {}

        # 1. Prediction distribution check (always possible)
        pred_result = self.analyze_prediction_distribution(predictions, labels)
        results["prediction_distribution"] = pred_result

        # 2. Perturbation stability (add small noise)
        noise = np.random.RandomState(42).normal(0, 0.02, size=len(predictions))
        perturbed_preds = np.clip(np.asarray(predictions).ravel() + noise, 0, 1)
        pert_result = self.analyze_perturbation_stability(
            np.asarray(predictions).ravel(), perturbed_preds
        )
        results["perturbation_stability"] = pert_result

        # 3. CatBoost leaf analysis (if model available)
        if catboost_model is not None and training_data is not None:
            cb_result = self.analyze_catboost_model(catboost_model, training_data)
            results["catboost_leaves"] = cb_result

        # Aggregate
        scores = [r.overfit_score for r in results.values()]
        overall_score = np.mean(scores) if scores else 0.0
        any_overfit = any(r.is_overfit for r in results.values())

        return {
            "overall_overfit_score": round(overall_score, 4),
            "is_overfit": any_overfit or overall_score >= self.overfit_score_threshold,
            "individual_results": results,
            "recommendation": (
                "RETRAIN: Model shows signs of overfitting across circuit checks."
                if (any_overfit or overall_score >= self.overfit_score_threshold)
                else "PASS: No circuit-based overfitting detected."
            ),
        }
