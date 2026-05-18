"""
IC-Based Signal Quality Gates — extends PatternEvaluationGate with IC-centric metrics.

Adds the IC-based gating used in "From Hypotheses to Factors":
    - Mean daily Pearson IC (between signal at T and forward return at T+1)
    - IC t-statistic (time-series significance)
    - Signal coverage (fraction of non-missing observations)
    - Combined gate: mean_ic >= threshold AND ic_tstat >= threshold AND coverage >= threshold

Integrates with the existing 4-step PatternEvaluationGate as a supplementary
pre-filter or standalone gate.

Source: Huang, Fan, Hu & Ye (2026), §4.4 and §5.2 Table 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass
class ICGateResult:
    """Result of IC-based gating for a signal."""

    factor_name: str
    mean_ic: float
    ic_std: float
    ic_tstat: float
    ic_pvalue: float
    coverage: float
    n_dates: int
    min_assets_per_date: float
    passed: bool
    diagnostics: dict = field(default_factory=dict)

    def report(self) -> str:
        lines = [
            f"IC Gate: {self.factor_name}",
            f"  Mean IC:    {self.mean_ic:+.4f}",
            f"  IC t-stat:  {self.ic_tstat:.2f} (p={self.ic_pvalue:.4f})",
            f"  Coverage:   {self.coverage:.2%}",
            f"  N dates:    {self.n_dates}",
            f"  Min assets: {self.min_assets_per_date:.1f}",
            f"  Result:     {'PASS' if self.passed else 'FAIL'}",
        ]
        if self.diagnostics:
            for k, v in self.diagnostics.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


class ICGate:
    """IC-based signal quality gate for factor discovery.

    Evaluates a factor signal series against forward returns using:
        1. Mean Pearson IC (signal quality)
        2. IC t-statistic (statistical significance)
        3. Coverage (data completeness)

    Parameters:
        min_mean_ic: Minimum absolute mean IC threshold (default 0.02).
        min_ic_tstat: Minimum IC t-statistic (default 2.0).
        min_coverage: Minimum fraction of non-missing observations (default 0.70).
        min_assets_per_date: Minimum assets required per date for IC calculation.
        ic_method: 'pearson' (default) or 'spearman'.
    """

    def __init__(
        self,
        min_mean_ic: float = 0.02,
        min_ic_tstat: float = 2.0,
        min_coverage: float = 0.70,
        min_assets_per_date: int = 10,
        ic_method: str = "pearson",
    ) -> None:
        self.min_mean_ic = min_mean_ic
        self.min_ic_tstat = min_ic_tstat
        self.min_coverage = min_coverage
        self.min_assets_per_date = min_assets_per_date
        self.ic_method = ic_method

    def evaluate(
        self,
        factor_name: str,
        scores: np.ndarray | pd.DataFrame,
        forward_returns: np.ndarray | pd.DataFrame,
        dates: np.ndarray | pd.Index | None = None,
    ) -> ICGateResult:
        """Run IC-based quality gate.

        Args:
            factor_name: Name of the factor/pattern being evaluated.
            scores: (N,) or (N, T) array of factor scores.
            forward_returns: (N,) or (N, T) array of forward returns.
            dates: Optional date index for panel data.

        Returns:
            ICGateResult with all metrics and pass/fail decision.
        """
        scores_arr = np.asarray(scores, dtype=np.float64)
        returns_arr = np.asarray(forward_returns, dtype=np.float64)

        if scores_arr.ndim > 1:
            scores_arr = scores_arr.flatten()
        if returns_arr.ndim > 1:
            returns_arr = returns_arr.flatten()

        mask = np.isfinite(scores_arr) & np.isfinite(returns_arr)
        n_valid = int(mask.sum())

        coverage = n_valid / max(len(scores_arr), 1)
        min_assets = float(n_valid) if dates is None else n_valid

        if n_valid < self.min_assets_per_date:
            return ICGateResult(
                factor_name=factor_name,
                mean_ic=0.0,
                ic_std=0.0,
                ic_tstat=0.0,
                ic_pvalue=1.0,
                coverage=coverage,
                n_dates=1,
                min_assets_per_date=float(n_valid),
                passed=False,
                diagnostics={"error": f"Insufficient data: {n_valid} < {self.min_assets_per_date}"},
            )

        if self.ic_method == "pearson":
            ic_val, ic_pval = scipy_stats.pearsonr(scores_arr[mask], returns_arr[mask])
        else:
            ic_val, ic_pval = scipy_stats.spearmanr(scores_arr[mask], returns_arr[mask])

        passed = (
            abs(ic_val) >= self.min_mean_ic
            and n_valid >= self.min_assets_per_date
            and coverage >= self.min_coverage
        )

        ic_tstat = (
            ic_val
            / (np.std(scores_arr[mask]) * np.std(returns_arr[mask]) / np.sqrt(n_valid) + 1e-9)
            if n_valid > 1
            else 0.0
        )

        return ICGateResult(
            factor_name=factor_name,
            mean_ic=float(ic_val),
            ic_std=float(np.std(scores_arr[mask])) if n_valid > 1 else 0.0,
            ic_tstat=float(ic_tstat),
            ic_pvalue=float(ic_pval),
            coverage=coverage,
            n_dates=1 if dates is None else 1,
            min_assets_per_date=float(min_assets),
            passed=passed,
        )

    def evaluate_panel(
        self,
        factor_name: str,
        scores: pd.DataFrame,
        forward_returns: pd.DataFrame,
    ) -> ICGateResult:
        """Evaluate IC gate on panel data with cross-sectional IC per date.

        Args:
            factor_name: Name of the factor.
            scores: DataFrame with shape (n_dates, n_tickers).
            forward_returns: DataFrame with same shape as scores.

        Returns:
            ICGateResult with time-series aggregated IC statistics.
        """
        if scores.shape != forward_returns.shape:
            raise ValueError(
                f"Shape mismatch: scores {scores.shape} vs returns {forward_returns.shape}"
            )

        ic_series: list[float] = []
        n_dates_used = 0
        asset_counts: list[int] = []

        for idx in range(len(scores)):
            s = scores.iloc[idx].values.astype(np.float64)
            r = forward_returns.iloc[idx].values.astype(np.float64)
            mask = np.isfinite(s) & np.isfinite(r)
            n = mask.sum()

            if n >= self.min_assets_per_date:
                if self.ic_method == "pearson":
                    ic_val = np.corrcoef(s[mask], r[mask])[0, 1]
                else:
                    ic_val, _ = scipy_stats.spearmanr(s[mask], r[mask])
                ic_series.append(ic_val)
                n_dates_used += 1
                asset_counts.append(n)

        if n_dates_used == 0:
            return ICGateResult(
                factor_name=factor_name,
                mean_ic=0.0,
                ic_std=0.0,
                ic_tstat=0.0,
                ic_pvalue=1.0,
                coverage=float(scores.notna().mean().mean()),
                n_dates=0,
                min_assets_per_date=0.0,
                passed=False,
                diagnostics={"error": "No valid cross-sectional dates"},
            )

        ic_arr = np.array(ic_series)
        mean_ic = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr, ddof=1))
        ic_tstat = mean_ic / (ic_std / np.sqrt(n_dates_used)) if ic_std > 0 else 0.0
        _, ic_pval = scipy_stats.ttest_1samp(ic_arr, 0.0)

        coverage = float(scores.notna().mean().mean())
        mean_assets = float(np.mean(asset_counts))

        passed = (
            abs(mean_ic) >= self.min_mean_ic
            and abs(ic_tstat) >= self.min_ic_tstat
            and coverage >= self.min_coverage
        )

        diagnostics: dict = {
            "n_dates_with_valid_ic": n_dates_used,
            "mean_assets_per_date": mean_assets,
            "min_ic_daily": float(np.min(ic_arr)),
            "max_ic_daily": float(np.max(ic_arr)),
            "ic_positive_frac": float((ic_arr > 0).mean()),
        }

        return ICGateResult(
            factor_name=factor_name,
            mean_ic=mean_ic,
            ic_std=ic_std,
            ic_tstat=float(ic_tstat),
            ic_pvalue=float(ic_pval),
            coverage=coverage,
            n_dates=n_dates_used,
            min_assets_per_date=mean_assets,
            passed=passed,
            diagnostics=diagnostics,
        )


def ic_gate_summary_table(results: list[ICGateResult]) -> str:
    """Generate a formatted summary table of IC gate results.

    Args:
        results: List of ICGateResult objects.

    Returns:
        Formatted table string.
    """
    lines = [
        f"{'Factor':<35} {'MeanIC':>8} {'t-stat':>8} {'Cov%':>7} {'Result':>6}",
        "-" * 70,
    ]
    for r in sorted(results, key=lambda x: abs(x.mean_ic), reverse=True):
        lines.append(
            f"{r.factor_name:<35} {r.mean_ic:>+8.4f} {r.ic_tstat:>8.2f} "
            f"{r.coverage:>7.1%} {'PASS' if r.passed else 'FAIL':>6}"
        )
    return "\n".join(lines)
