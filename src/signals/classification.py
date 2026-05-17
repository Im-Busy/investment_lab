"""
R9: Return Factor vs Risk Factor Classification.

Classify each pattern signal as "return factor" (directional, IC significant)
or "risk factor" (variance explanatory, doesn't predict direction).

Source: 华泰多因子系列1 §1.3

Method:
    1. Regress forward return on pattern signal: r_{t+1} = α + β·s_t + ε
    2. Test H0: β = 0 via t-test.
    3. |t| > 2 AND β direction stable → RETURN factor (entry signal)
    4. |t| <= 2 OR β flips sign across subsamples → RISK factor (position sizing modifier)

Architecture:
    FactorClassifier consumed by PatternEvaluationGate step 2 and as standalone
    bulk classifier for all 45+ pattern detectors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)

# Significance threshold for return factor classification
DEFAULT_TSTAT_THRESHOLD = 2.0
DEFAULT_SIGNIFICANCE = 0.05


class FactorClass(Enum):
    """Classification of a pattern's factor type."""

    RETURN = "return"
    RISK = "risk"
    AMBIGUOUS = "ambiguous"


@dataclass
class FactorClassification:
    """Classification result for a single pattern signal.

    Attributes:
        pattern_name: Name of the pattern.
        factor_class: RETURN (entry signal), RISK (position sizing), or AMBIGUOUS.
        beta: Regression coefficient of signal on forward return.
        beta_se: Standard error of beta.
        tstat: t-statistic for H0: beta = 0.
        pvalue: Two-sided p-value.
        r_squared: Regression R².
        directional_ratio: Fraction of periods where signal direction aligns with
            subsequent return direction. > 0.5 suggests directional predictability.
    """

    pattern_name: str
    factor_class: FactorClass
    beta: float
    beta_se: float
    tstat: float
    pvalue: float
    r_squared: float
    directional_ratio: float = 0.5
    warning: str = ""


@dataclass
class BulkClassification:
    """Aggregated classification results for multiple patterns.

    Attributes:
        results: Per-pattern FactorClassification results.
        return_count: Number classified as RETURN factors.
        risk_count: Number classified as RISK factors.
        ambiguous_count: Number classified as AMBIGUOUS.
        summary_df: DataFrame with pattern_name, factor_class, tstat, pvalue, r_squared.
    """

    results: list[FactorClassification] = field(default_factory=list)
    return_count: int = 0
    risk_count: int = 0
    ambiguous_count: int = 0


class FactorClassifier:
    """Classify pattern signals as return factors or risk factors.

    Return factors produce statistically significant, directionally consistent
    predictions of forward returns — they should drive entry signals.

    Risk factors explain variance but don't predict direction — they should
    inform position sizing and risk allocation, not trigger entries.
    """

    def __init__(
        self,
        tstat_threshold: float = DEFAULT_TSTAT_THRESHOLD,
        significance: float = DEFAULT_SIGNIFICANCE,
        directional_min: float = 0.51,
    ) -> None:
        self._tstat_threshold = tstat_threshold
        self._significance = significance
        self._directional_min = directional_min

    def classify(
        self,
        pattern_name: str,
        signal: np.ndarray,
        forward_returns: np.ndarray,
    ) -> FactorClassification:
        """Classify a single pattern signal as return or risk factor.

        Args:
            pattern_name: Name of the pattern for reporting.
            signal: (N,) array of pattern signal values (non-NaN).
            forward_returns: (N,) array of forward returns aligned with signal.

        Returns:
            FactorClassification with factor_class and statistics.
        """
        n = min(len(signal), len(forward_returns))
        s = np.asarray(signal[:n], dtype=np.float64)
        r = np.asarray(forward_returns[:n], dtype=np.float64)

        mask = np.isfinite(s) & np.isfinite(r)
        s = s[mask]
        r = r[mask]

        if len(s) < 30:
            return FactorClassification(
                pattern_name=pattern_name,
                factor_class=FactorClass.AMBIGUOUS,
                beta=0.0,
                beta_se=0.0,
                tstat=0.0,
                pvalue=1.0,
                r_squared=0.0,
                warning=f"insufficient data: {len(s)} < 30 bars",
            )

        with np.errstate(invalid="ignore"):
            s_std = np.nan_to_num((s - np.nanmean(s)) / max(np.nanstd(s), 1e-12), nan=0.0)

        x = sm.add_constant(s_std, prepend=True)
        try:
            model = sm.OLS(r, x, missing="drop").fit()
        except Exception:
            return FactorClassification(
                pattern_name=pattern_name,
                factor_class=FactorClass.AMBIGUOUS,
                beta=0.0,
                beta_se=0.0,
                tstat=0.0,
                pvalue=1.0,
                r_squared=0.0,
                warning="OLS fit failed",
            )

        beta = float(model.params[1]) if len(model.params) > 1 else 0.0
        beta_se = float(model.bse[1]) if len(model.bse) > 1 else 0.0
        with np.errstate(invalid="ignore"):
            tstat = float(beta / max(beta_se, 1e-12)) if beta_se > 0 else 0.0
        pvalue = float(model.pvalues[1]) if len(model.pvalues) > 1 else 1.0
        r_squared = float(model.rsquared)

        dir_ratio = self._compute_directional_ratio(s, r)
        warnings: list[str] = []

        factor_class = FactorClass.AMBIGUOUS
        if abs(tstat) >= self._tstat_threshold and pvalue <= self._significance:
            if dir_ratio >= self._directional_min:
                factor_class = FactorClass.RETURN
            else:
                factor_class = FactorClass.RISK
                warnings.append(
                    f"|t|={abs(tstat):.2f} significant but directional_ratio="
                    f"{dir_ratio:.3f} < {self._directional_min}"
                )
        elif abs(tstat) >= 1.0:
            factor_class = FactorClass.RISK
            warnings.append(f"|t|={abs(tstat):.2f} below threshold {self._tstat_threshold}")

        return FactorClassification(
            pattern_name=pattern_name,
            factor_class=factor_class,
            beta=beta,
            beta_se=beta_se,
            tstat=tstat,
            pvalue=pvalue,
            r_squared=r_squared,
            directional_ratio=dir_ratio,
            warning="; ".join(warnings) if warnings else "",
        )

    def classify_multiple(
        self,
        signals: dict[str, np.ndarray],
        forward_returns: np.ndarray,
    ) -> BulkClassification:
        """Classify multiple pattern signals in bulk.

        Args:
            signals: Dict mapping pattern_name → signal array.
            forward_returns: (N,) array of forward returns.

        Returns:
            BulkClassification with per-pattern results and counts.
        """
        results: list[FactorClassification] = []
        return_count = 0
        risk_count = 0
        ambiguous_count = 0

        for name, sig in signals.items():
            fc = self.classify(name, sig, forward_returns)
            results.append(fc)
            if fc.factor_class == FactorClass.RETURN:
                return_count += 1
            elif fc.factor_class == FactorClass.RISK:
                risk_count += 1
            else:
                ambiguous_count += 1

        return BulkClassification(
            results=results,
            return_count=return_count,
            risk_count=risk_count,
            ambiguous_count=ambiguous_count,
        )

    @staticmethod
    def _compute_directional_ratio(signal: np.ndarray, forward_returns: np.ndarray) -> float:
        """Fraction of periods where signal sign matches forward return sign."""
        n = len(signal)
        if n < 2:
            return 0.5
        same_direction = 0
        for i in range(n):
            if np.sign(signal[i]) == np.sign(forward_returns[i]) and forward_returns[i] != 0:
                same_direction += 1
        return same_direction / n


def classify_patterns(
    signals_df: "pd.DataFrame",
    forward_returns: np.ndarray,
    tstat_threshold: float = DEFAULT_TSTAT_THRESHOLD,
) -> BulkClassification:
    """Convenience: classify all pattern signal columns in a DataFrame.

    Args:
        signals_df: DataFrame with pattern names as columns, signal values.
        forward_returns: (N,) array of forward returns.
        tstat_threshold: Minimum |t| for return factor classification.

    Returns:
        BulkClassification with per-pattern results.
    """
    if not isinstance(signals_df, pd.DataFrame):
        raise TypeError("signals_df must be a pandas DataFrame")

    fc = FactorClassifier(tstat_threshold=tstat_threshold)
    signals_dict = {col: signals_df[col].to_numpy(dtype=np.float64) for col in signals_df.columns}
    return fc.classify_multiple(signals_dict, forward_returns)


def get_entry_patterns(classification: BulkClassification) -> list[str]:
    """Return pattern names classified as RETURN factors (entry signals)."""
    return [r.pattern_name for r in classification.results if r.factor_class == FactorClass.RETURN]


def get_risk_patterns(classification: BulkClassification) -> list[str]:
    """Return pattern names classified as RISK factors (position sizing)."""
    return [r.pattern_name for r in classification.results if r.factor_class == FactorClass.RISK]
