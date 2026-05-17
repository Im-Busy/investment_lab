"""
Q5: Model Validation Framework — PSI / KS / Gini.

Industry-standard model monitoring to prevent silent model degradation
in production. Based on credit risk model governance frameworks (Basel II)
adapted for trading model validation.

Metrics:
  PSI (Population Stability Index):
    Measures feature distribution drift vs training baseline.
    PSI < 0.1 → no drift, 0.1-0.25 → moderate, > 0.25 → significant.
    Formula: Σ (actual% - expected%) * ln(actual% / expected%)

  KS (Kolmogorov-Smirnov) Statistic:
    Measures discrimination power decay.
    Compares score distributions of positive vs negative classes.
    KS > 0.3 → good, 0.2-0.3 → fair, < 0.2 → poor.

  Gini Coefficient:
    Overall model quality (AUC equivalent: Gini = 2*AUC - 1).
    Gini > 0.6 → excellent, 0.4-0.6 → good, < 0.4 → fair.

  Feature Drift Monitor:
    Per-feature PSI tracking. Flags features with significant drift.

Usage:
    >>> validator = ModelValidator()
    >>> report = validator.validate(train_scores, test_scores, train_labels, test_labels)
    >>> print(report.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

N_BINS_PSI = 10
PSI_LOW = 0.10
PSI_MODERATE = 0.25
KS_GOOD = 0.30
KS_FAIR = 0.20
GINI_EXCELLENT = 0.60
GINI_GOOD = 0.40


@dataclass
class DriftResult:
    """Per-feature drift analysis result."""

    feature: str
    psi: float
    ks: float
    train_mean: float
    test_mean: float
    train_std: float
    test_std: float
    drift_level: str  # none, moderate, significant


@dataclass
class ValidationReport:
    """Model validation report with PSI/KS/Gini metrics."""

    psi_overall: float
    psi_level: str
    ks_statistic: float
    ks_level: str
    gini: float
    gini_level: str
    feature_drift: List[DriftResult] = field(default_factory=list)
    n_features: int = 0
    n_train: int = 0
    n_test: int = 0
    warnings: List[str] = field(default_factory=list)
    timestamp: str = ""

    def summary(self) -> str:
        lines = [
            "=== Model Validation Report ===",
            f"Samples: {self.n_train} train, {self.n_test} test",
            f"Features: {self.n_features}",
            f"PSI: {self.psi_overall:.4f} ({self.psi_level})",
            f"KS:  {self.ks_statistic:.4f} ({self.ks_level})",
            f"Gini: {self.gini:.4f} ({self.gini_level})",
        ]
        if self.feature_drift:
            sig = [d for d in self.feature_drift if d.drift_level == "significant"]
            mod = [d for d in self.feature_drift if d.drift_level == "moderate"]
            lines.append(f"Drifted features: {len(sig)} sig, {len(mod)} moderate")
            for d in sig[:5]:
                lines.append(f"  SIG: {d.feature} (PSI={d.psi:.4f})")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"WARN: {w}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "psi_overall": round(self.psi_overall, 6),
            "psi_level": self.psi_level,
            "ks_statistic": round(self.ks_statistic, 6),
            "ks_level": self.ks_level,
            "gini": round(self.gini, 6),
            "gini_level": self.gini_level,
            "n_features": self.n_features,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "warnings": self.warnings,
            "drifted_features": [
                {
                    "feature": d.feature,
                    "psi": round(d.psi, 6),
                    "ks": round(d.ks, 6),
                    "drift_level": d.drift_level,
                }
                for d in self.feature_drift
                if d.drift_level != "none"
            ],
        }

    @property
    def has_significant_drift(self) -> bool:
        return any(d.drift_level == "significant" for d in self.feature_drift)

    @property
    def discrimination_degraded(self) -> bool:
        return self.ks_level == "poor"

    @property
    def needs_retraining(self) -> bool:
        return (
            self.psi_overall > PSI_MODERATE
            or self.ks_statistic < KS_FAIR
            or self.has_significant_drift
        )


class ModelValidator:
    """Model validation and monitoring framework.

    Computes PSI (feature drift), KS (discrimination decay), and Gini
    (overall quality) comparing a test set against a training baseline.

    Args:
        n_bins: Number of bins for PSI calculation (default 10).
        psi_moderate: Threshold for moderate drift warning.
        psi_significant: Threshold for significant drift alert.
    """

    def __init__(
        self,
        n_bins: int = N_BINS_PSI,
        psi_moderate: float = PSI_LOW,
        psi_significant: float = PSI_MODERATE,
    ):
        self.n_bins = n_bins
        self.psi_moderate = psi_moderate
        self.psi_significant = psi_significant

    def validate(
        self,
        train_scores: np.ndarray,
        test_scores: np.ndarray,
        train_labels: Optional[np.ndarray] = None,
        test_labels: Optional[np.ndarray] = None,
        train_features: Optional[pd.DataFrame] = None,
        test_features: Optional[pd.DataFrame] = None,
    ) -> ValidationReport:
        """Run full validation suite.

        Args:
            train_scores: Model probability scores for training set.
            test_scores: Model probability scores for test set.
            train_labels: Binary labels for training set (for KS/Gini).
            test_labels: Binary labels for test set (for KS/Gini).
            train_features: Feature DataFrame for training set.
            test_features: Feature DataFrame for test set.

        Returns:
            ValidationReport with all metrics.
        """
        warnings = []
        psi = self._compute_psi(train_scores, test_scores)
        psi_level = self._psi_level(psi)

        if train_labels is not None and test_labels is not None:
            ks = self._compute_ks(test_scores, test_labels)
            gini = self._compute_gini(test_scores, test_labels)
        else:
            ks = np.nan
            gini = np.nan
            warnings.append("No labels provided; KS/Gini not computed")

        ks_level = self._ks_level(ks) if not np.isnan(ks) else "unknown"
        gini_level = self._gini_level(gini) if not np.isnan(gini) else "unknown"

        feature_drift: List[DriftResult] = []
        if train_features is not None and test_features is not None:
            feature_drift = self._feature_drift_analysis(train_features, test_features)

        return ValidationReport(
            psi_overall=psi,
            psi_level=psi_level,
            ks_statistic=ks,
            ks_level=ks_level,
            gini=gini,
            gini_level=gini_level,
            feature_drift=feature_drift,
            n_features=len(train_features.columns) if train_features is not None else 0,
            n_train=len(train_scores),
            n_test=len(test_scores),
            warnings=warnings,
            timestamp=pd.Timestamp.now().isoformat(),
        )

    def _compute_psi(self, expected: np.ndarray, actual: np.ndarray) -> float:
        """Compute Population Stability Index.

        PSI = Σ (A_i - E_i) * ln(A_i / E_i)

        Args:
            expected: Training distribution (baseline).
            actual: Test distribution (monitoring).
        """
        combined = np.concatenate([expected, actual])
        bin_edges = np.percentile(combined, np.linspace(0, 100, self.n_bins + 1))
        bin_edges[-1] += 1e-10

        e_hist, _ = np.histogram(expected, bins=bin_edges)
        a_hist, _ = np.histogram(actual, bins=bin_edges)

        e_pct = e_hist / max(len(expected), 1)
        a_pct = a_hist / max(len(actual), 1)

        psi_sum = 0.0
        for e_i, a_i in zip(e_pct, a_pct):
            e_i = max(e_i, 1e-10)
            a_i = max(a_i, 1e-10)
            psi_sum += (a_i - e_i) * np.log(a_i / e_i)

        return float(psi_sum)

    def _compute_ks(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute Kolmogorov-Smirnov statistic.

        KS = max |CDF_pos(s) - CDF_neg(s)|
        """
        pos = scores[labels == 1]
        neg = scores[labels == 0]

        if len(pos) == 0 or len(neg) == 0:
            return 0.0

        combined = np.sort(np.unique(np.concatenate([pos, neg])))
        cdf_pos = np.searchsorted(np.sort(pos), combined, side="right") / len(pos)
        cdf_neg = np.searchsorted(np.sort(neg), combined, side="right") / len(neg)

        return float(np.max(np.abs(cdf_pos - cdf_neg)))

    def _compute_gini(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute Gini coefficient from AUC.

        Gini = 2 * AUC - 1
        """
        from sklearn.metrics import roc_auc_score

        if len(np.unique(labels)) < 2:
            return 0.0
        auc = roc_auc_score(labels, scores)
        return float(2 * auc - 1)

    def _feature_drift_analysis(
        self, train_feat: pd.DataFrame, test_feat: pd.DataFrame
    ) -> List[DriftResult]:
        """Per-feature PSI and KS analysis."""
        common_cols = [c for c in train_feat.columns if c in test_feat.columns]
        results = []

        for col in common_cols:
            train_vals = train_feat[col].dropna().values
            test_vals = test_feat[col].dropna().values

            if len(train_vals) < 10 or len(test_vals) < 10:
                continue

            psi = self._compute_psi(train_vals, test_vals)

            sorted_all = np.sort(np.concatenate([train_vals, test_vals]))
            cdf_train = np.searchsorted(np.sort(train_vals), sorted_all, side="right") / len(
                train_vals
            )
            cdf_test = np.searchsorted(np.sort(test_vals), sorted_all, side="right") / len(
                test_vals
            )
            ks = float(np.max(np.abs(cdf_train - cdf_test)))

            drift_level = (
                "significant"
                if psi >= self.psi_significant
                else "moderate"
                if psi >= self.psi_moderate
                else "none"
            )

            results.append(
                DriftResult(
                    feature=col,
                    psi=psi,
                    ks=ks,
                    train_mean=float(np.mean(train_vals)),
                    test_mean=float(np.mean(test_vals)),
                    train_std=float(np.std(train_vals)),
                    test_std=float(np.std(test_vals)),
                    drift_level=drift_level,
                )
            )

        return sorted(results, key=lambda x: x.psi, reverse=True)

    @staticmethod
    def _psi_level(psi: float) -> str:
        if psi > PSI_MODERATE:
            return "significant_drift"
        elif psi > PSI_LOW:
            return "moderate_drift"
        return "stable"

    @staticmethod
    def _ks_level(ks: float) -> str:
        if ks >= KS_GOOD:
            return "good"
        elif ks >= KS_FAIR:
            return "fair"
        return "poor"

    @staticmethod
    def _gini_level(gini: float) -> str:
        if gini >= GINI_EXCELLENT:
            return "excellent"
        elif gini >= GINI_GOOD:
            return "good"
        return "fair"


def validate_from_arrays(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> ValidationReport:
    """Quick validation from score arrays only (no features)."""
    validator = ModelValidator()
    return validator.validate(train_scores, test_scores, train_labels, test_labels)
