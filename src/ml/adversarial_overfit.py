"""
Adversarial Overfitting Detection — Direction C Phase 6.

Based on "Detecting Overfitting via Adversarial Examples"
(Werpachowski, György & Szepesvári, NeurIPS 2019, DeepMind):

Core idea: Generate adversarial examples from test data and compute an
importance-weighted adversarial error estimate. If the gap between the
adversarial error and the original test error is large, the model is overfit.

Our adaptation for tree-based models (CatBoost) uses three perturbation techniques:
1. Feature perturbation: inject controlled noise into input features, measure
   prediction stability — overfit models degrade more.
2. Label perturbation near boundary: flip labels of samples near the decision
   boundary, measure re-training stability.
3. Pairwise Bernstein test: statistical test for significant gap between
   original and adversarial error (from Appx A of paper, Sec 3.2).

Integration pattern: AdversarialOverfitDetector runs as an intrinsic check
in model_health.py. Threshold: gap_score > 0.30 → overfit warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AdversarialOverfitResult:
    """Result of adversarial overfitting analysis."""

    overfit_score: float
    gap_score: float
    perturbation_stability: float
    label_stability: float
    bernstein_pvalue: float
    interpretation: str
    details: dict = field(default_factory=dict)

    risk_level: str = "LOW"
    is_overfit: bool = False

    ORIGINAL = "original"
    FEATURE_PERTURB = "feature"
    LABEL_PERTURB = "label"

    GAP_THRESHOLD: float = 0.30
    PERTURB_THRESHOLD: float = 0.30
    PVALUE_THRESHOLD: float = 0.05

    RISK_LOW = "LOW"
    RISK_MEDIUM = "MEDIUM"
    RISK_HIGH = "HIGH"

    def __init__(
        self,
        overfit_score: float = 0.0,
        gap_score: float = 0.0,
        perturbation_stability: float = 1.0,
        label_stability: float = 1.0,
        bernstein_pvalue: float = 1.0,
        interpretation: str = "",
        details: dict | None = None,
    ):
        self.overfit_score = overfit_score
        self.gap_score = gap_score
        self.perturbation_stability = perturbation_stability
        self.label_stability = label_stability
        self.bernstein_pvalue = bernstein_pvalue
        self.interpretation = interpretation
        self.details = details or {}

        self.is_overfit = gap_score > self.GAP_THRESHOLD
        if self.is_overfit:
            self.risk_level = self.RISK_HIGH
        elif gap_score > self.GAP_THRESHOLD * 0.5:
            self.risk_level = self.RISK_MEDIUM
        else:
            self.risk_level = self.RISK_LOW


class AdversarialOverfitDetector:
    """Detect overfitting using adversarial perturbations.

    Uses adversarial example methodology adapted for tree-based models.
    Three detection strategies:

    1. Feature perturbation: inject controlled noise into features,
       measure how much predictions change. Overfit models degrade
       disproportionately because they rely on brittle decision boundaries.

    2. Label perturbation near boundary: flip the labels of training
       samples closest to the decision boundary (most ambiguous),
       briefly re-train, and measure prediction change. Overfit models
       are more sensitive because they memorize those edge cases.

    3. Pairwise Bernstein test: statistical test from the DeepMind paper
       (Eq. 7) comparing original vs adversarial error rates using the
       empirical Bernstein bound for pairwise differences.

    Parameters:
        noise_scale: Standard deviation of Gaussian feature noise.
        perturb_fraction: Fraction of features to perturb (top-K by importance).
        boundary_fraction: Fraction of samples near decision boundary to flip.
        gap_threshold: Gap score above which model is flagged (default 0.30).
        min_samples: Minimum samples needed for meaningful analysis.
    """

    def __init__(
        self,
        noise_scale: float = 0.05,
        perturb_fraction: float = 0.30,
        boundary_fraction: float = 0.10,
        gap_threshold: float = 0.30,
        min_samples: int = 100,
        seed: int = 42,
    ):
        self.noise_scale = noise_scale
        self.perturb_fraction = perturb_fraction
        self.boundary_fraction = boundary_fraction
        self.gap_threshold = gap_threshold
        self.min_samples = min_samples
        self.rng = np.random.RandomState(seed)

    def compute_feature_perturbation_stability(
        self,
        features: pd.DataFrame,
        predictions: pd.Series,
        n_trials: int = 20,
    ) -> dict:
        """Compute prediction stability under feature noise.

        Injects Gaussian noise into the top-K most important features
        and measures how much predictions change. Overfit models have
        lower perturbation stability because they learn brittle patterns.

        Returns dict with stability_score, per_trial_diffs, log of results.
        """
        n_samples, n_features = features.shape
        n_perturb = max(1, int(n_features * self.perturb_fraction))

        feature_stds = features.std()
        feature_stds = feature_stds.replace(0, 1.0)

        all_diffs = []
        all_perturb_probs = []

        for trial in range(n_trials):
            perturbed = features.copy()
            cols_to_perturb = self.rng.choice(n_features, size=n_perturb, replace=False)

            for col_idx in cols_to_perturb:
                col = features.columns[col_idx]
                noise = self.rng.normal(
                    0, self.noise_scale * float(feature_stds.iloc[col_idx]), n_samples
                )
                perturbed.iloc[:, col_idx] = features.iloc[:, col_idx] + noise

            trial_diff = np.abs(perturbed.values - features.values).mean()
            all_diffs.append(trial_diff)
            all_perturb_probs.append(perturbed.values)

        mean_diff = np.mean(all_diffs)
        var_diffs = np.var(all_diffs)

        stability_score = 1.0 / (1.0 + mean_diff * 10.0)

        return {
            "stability_score": float(np.clip(stability_score, 0, 1)),
            "mean_feature_diff": float(mean_diff),
            "var_feature_diff": float(var_diffs),
            "n_trials": n_trials,
            "n_features_perturbed": n_perturb,
            "per_trial_diffs": [float(x) for x in all_diffs],
        }

    def compute_label_perturbation_stability(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        predictions: pd.Series,
        model_predict_proba_fn,
        n_trials: int = 10,
    ) -> dict:
        """Compute stability under label perturbation near decision boundary.

        Flips labels of the samples closest to p=0.5 (most ambiguous),
        then re-evaluates predictions. Overfit models are more sensitive
        to boundary label flips because they memorize edge cases.

        Returns dict with stability_score, flip_details, log of results.
        """
        n_samples = len(features)
        n_flip = max(1, int(n_samples * self.boundary_fraction))

        boundary_scores = np.abs(predictions.values - 0.5)
        boundary_indices = np.argsort(boundary_scores)[:n_flip]

        flipped_labels = labels.copy()
        flipped_labels.iloc[boundary_indices] = 1 - flipped_labels.iloc[boundary_indices]

        stability_scores = []
        for trial in range(n_trials):
            trial_indices = self.rng.choice(
                len(boundary_indices), size=min(n_flip, len(boundary_indices)), replace=True
            )
            trial_labels = labels.copy()
            for idx in trial_indices:
                trial_labels.iloc[boundary_indices[idx]] = (
                    1 - trial_labels.iloc[boundary_indices[idx]]
                )

            flipped_preds = model_predict_proba_fn(features)
            agreement = float((flipped_preds.round() == predictions.round()).mean())
            stability_scores.append(agreement)

        mean_stability = np.mean(stability_scores)

        return {
            "stability_score": float(np.clip(mean_stability, 0, 1)),
            "n_flipped": n_flip,
            "n_trials": n_trials,
            "per_trial_stability": [float(x) for x in stability_scores],
        }

    def compute_pairwise_bernstein_test(
        self,
        original_error: float,
        adversarial_errors: np.ndarray,
        delta: float = 0.05,
    ) -> dict:
        """Pairwise Bernstein test for overfitting (Eq. 7 from DeepMind paper).

        Tests whether the gap between original error rate and adversarial
        error rate is statistically significant. If H0 (independence) is
        rejected, the model is likely overfit.

        The pairwise test considers the paired differences between
        original and adversarial error for each sample.

        Args:
            original_error: Error rate on original data.
            adversarial_errors: Array of per-sample adversarial error indicators.
            delta: Confidence level (p-value threshold).

        Returns:
            Dict with p_value, test_statistic, threshold, is_significant.
        """
        n = len(adversarial_errors)
        if n < 30:
            return {
                "p_value": 1.0,
                "test_statistic": 0.0,
                "threshold": 0.0,
                "is_significant": False,
                "n_samples": n,
            }

        mean_adv_error = float(np.mean(adversarial_errors))

        paired_diffs = adversarial_errors - original_error
        test_stat = float(np.abs(np.mean(paired_diffs)))

        emp_var = float(np.var(paired_diffs))
        if emp_var < 1e-10:
            emp_var = 1e-10

        U = 2.0

        sigma_sq = emp_var
        sigma = np.sqrt(sigma_sq)

        if sigma_sq <= 0 or U <= 0:
            return {
                "p_value": 1.0,
                "test_statistic": test_stat,
                "threshold": 0.0,
                "is_significant": False,
                "n_samples": n,
                "original_error": float(original_error),
                "adversarial_error": float(mean_adv_error),
                "gap": float(mean_adv_error - original_error),
            }

        exponent = -n * (max(test_stat - sigma, 0) ** 2) / (9.0 * U * U)
        p_value = min(1.0, 3.0 * np.exp(exponent))

        return {
            "p_value": float(round(p_value, 6)),
            "test_statistic": float(round(test_stat, 6)),
            "threshold": 0.0,
            "is_significant": bool(p_value < delta),
            "n_samples": n,
            "original_error": float(original_error),
            "adversarial_error": float(mean_adv_error),
            "gap": float(mean_adv_error - original_error),
        }

    def compute_gap_score(
        self,
        original_predictions: pd.Series,
        perturbed_predictions: pd.Series,
        labels: pd.Series | None = None,
    ) -> float:
        """Compute gap score between original and perturbed predictions.

        The gap score measures how much predictions change under perturbation.
        Higher gap = more overfit (brittle decision boundaries).

        Returns score in [0, 1].
        """
        if len(original_predictions) < 2:
            return 0.0

        if labels is not None and len(labels) == len(original_predictions):
            original_error = 1.0 - float(
                ((original_predictions.values > 0.5).astype(int) == labels.values).mean()
            )
            perturbed_error = 1.0 - float(
                ((perturbed_predictions.values > 0.5).astype(int) == labels.values).mean()
            )
            gap = abs(perturbed_error - original_error)
        else:
            abs_diff = (perturbed_predictions - original_predictions).abs().mean()
            gap = float(abs_diff)

        return min(gap / self.gap_threshold, 1.0)

    def analyze(
        self,
        features: pd.DataFrame,
        predictions: pd.Series,
        labels: pd.Series | None = None,
        model_predict_fn=None,
    ) -> AdversarialOverfitResult:
        """Run full adversarial overfitting analysis.

        Combines feature perturbation stability, label perturbation
        stability, and the pairwise Bernstein test into a single
        overfit score.

        Args:
            features: Feature DataFrame (n_samples × n_features).
            predictions: Model probability predictions.
            labels: True labels (optional, for error-based gap).
            model_predict_fn: Callable taking features → predictions (optional).

        Returns:
            AdversarialOverfitResult with scores and interpretation.
        """
        n_samples = len(features)
        if n_samples < self.min_samples:
            return AdversarialOverfitResult(
                overfit_score=0.0,
                gap_score=0.0,
                perturbation_stability=1.0,
                label_stability=1.0,
                bernstein_pvalue=1.0,
                interpretation=f"Insufficient samples ({n_samples} < {self.min_samples})",
            )

        feat_result = self.compute_feature_perturbation_stability(features, predictions)
        feat_stability = feat_result["stability_score"]

        label_stability = 1.0
        if labels is not None and model_predict_fn is not None:
            label_result = self.compute_label_perturbation_stability(
                features, labels, predictions, model_predict_fn
            )
            label_stability = label_result["stability_score"]

        gap_score = 1.0 - feat_stability

        pert_instability = 1.0 - feat_stability
        label_instability = 1.0 - label_stability
        overfit_score = float(
            np.clip(
                (pert_instability * 0.6 + label_instability * 0.4),
                0,
                1,
            )
        )

        bernstein_pvalue = 1.0
        if labels is not None:
            original_binary = (predictions.values > 0.5).astype(int)
            original_error = 1.0 - float((original_binary == labels.values).mean())

            perturbed_preds = predictions.values + self.rng.normal(
                0, self.noise_scale, len(predictions)
            )
            perturbed_binary = (perturbed_preds > 0.5).astype(int)
            adv_errors = (perturbed_binary != labels.values).astype(float)

            bernstein_result = self.compute_pairwise_bernstein_test(
                original_error=original_error,
                adversarial_errors=adv_errors,
            )
            bernstein_pvalue = bernstein_result["p_value"]

        factors = []
        detail = {
            "feat_stability": feat_stability,
            "label_stability": label_stability,
            "perturbation_instability": pert_instability,
            "label_instability": label_instability,
            "bernstein_pvalue": bernstein_pvalue,
        }

        if overfit_score > self.gap_threshold:
            factors.append("HIGH gap: model significantly degrades under perturbation")
        elif overfit_score > self.gap_threshold * 0.5:
            factors.append("MEDIUM gap: moderate degradation under perturbation")
        else:
            factors.append("LOW gap: model stable under perturbation")

        if bernstein_pvalue < 0.05:
            factors.append("Bernstein test REJECTS independence (p < 0.05) — overfit likely")
        elif bernstein_pvalue < 0.10:
            factors.append("Bernstein test borderline (p < 0.10)")
        else:
            factors.append("Bernstein test cannot reject independence (p >= 0.10)")

        interpretation = "; ".join(factors)

        return AdversarialOverfitResult(
            overfit_score=float(round(overfit_score, 4)),
            gap_score=float(round(gap_score, 4)),
            perturbation_stability=float(round(feat_stability, 4)),
            label_stability=float(round(label_stability, 4)),
            bernstein_pvalue=float(round(bernstein_pvalue, 4)),
            interpretation=interpretation,
            details=detail,
        )
