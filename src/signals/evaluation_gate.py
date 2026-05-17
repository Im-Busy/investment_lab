"""
R3: 4-Step Pattern Evaluation Gate.

Formal entry requirement for any new pattern entering the detector catalog.
Four sequential tests applied to pattern signal series against forward returns.

Source: 华泰多因子系列1 §1.3

Gate sequence:
  1. Single-factor regression with industry dummies → t-stat on factor return
  2. |t|>2 ratio + directional t-test → return factor vs risk factor classification
  3. IC analysis — rank correlation of factor exposure at T vs return at T+1 (purified)
  4. Quantile backtest — sort by signal, hold top/bottom N, sector-neutral returns

Gate PASS requires: |t|>2 in step 1, IC significant in step 3, and
  top quantile > bottom quantile in step 4.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import stats as scipy_stats

from src.signals.factor_purification import FactorPurifier, PurificationReport

logger = logging.getLogger(__name__)


class FactorType(Enum):
    """Classification of a pattern as return factor (predicts direction)
    or risk factor (explains variance only)."""

    RETURN = "return"
    RISK = "risk"
    FAILED = "failed"


class GateResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"


@dataclass
class GateStepResult:
    """Result for a single evaluation step."""

    step: int
    name: str
    value: float
    threshold: float | None
    passed: bool
    details: dict = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Complete 4-step evaluation report for a pattern."""

    pattern_name: str

    step1_tstat: float
    """T-statistic from single-factor regression with industry dummies."""

    step1_pvalue: float
    """P-value of the t-statistic."""

    step2_factor_type: FactorType
    """Classification: return factor, risk factor, or failed."""

    step2_tstat_ratio: float
    """Fraction of rolling windows with |t| > 2."""

    step3_ic: float
    """Rank IC (Spearman correlation) of factor exposure vs forward return."""

    step3_ic_pvalue: float | None
    """P-value of IC significance test."""

    step4_top_ret: float
    """Mean return of top quantile portfolio (annualized)."""

    step4_bottom_ret: float
    """Mean return of bottom quantile portfolio (annualized)."""

    step4_spread: float
    """Top minus bottom quantile spread."""

    passed: bool
    """Overall gate result."""

    steps: list[GateStepResult]
    """Detailed per-step results."""

    purification: PurificationReport | None = None
    """Purification report from R2 (sector/size neutrality)."""

    warnings: list[str] = field(default_factory=list)
    """Warnings or caveats about the result."""


class PatternEvaluationGate:
    """4-step formal evaluation gate for pattern signal validation.

    Parameters:
        significance: P-value threshold for t-stat and IC significance.
        tstat_threshold: Minimum absolute t-statistic (typically 2.0).
        n_quantiles: Number of quantile bins for quantile backtest.
        min_samples: Minimum samples required for evaluation.
    """

    def __init__(
        self,
        significance: float = 0.05,
        tstat_threshold: float = 2.0,
        n_quantiles: int = 5,
        min_samples: int = 100,
    ) -> None:
        self._significance = significance
        self._tstat_threshold = tstat_threshold
        self._n_quantiles = n_quantiles
        self._min_samples = min_samples

    def evaluate(
        self,
        pattern_name: str,
        signal: np.ndarray,
        forward_return: np.ndarray,
        sectors: np.ndarray | None = None,
        log_mcap: np.ndarray | None = None,
        dates: np.ndarray | None = None,
    ) -> EvaluationReport:
        """Run the full 4-step evaluation gate.

        Args:
            pattern_name: Name of the pattern being evaluated.
            signal: (N,) array of pattern signals (-1, 0, 1 or continuous).
            forward_return: (N,) array of forward returns (return from T to T+1).
            sectors: (N,) array of sector labels for purification.
            log_mcap: (N,) array of log(market cap) for purification.
            dates: (N,) array of dates (for reporting only).

        Returns:
            EvaluationReport with all step results and overall gate decision.
        """
        signal = np.asarray(signal, dtype=np.float64)
        forward_return = np.asarray(forward_return, dtype=np.float64)
        warnings: list[str] = []

        # Mask: require valid signal and forward return
        mask = np.isfinite(signal) & np.isfinite(forward_return)
        n_valid = int(mask.sum())
        if n_valid < self._min_samples:
            return EvaluationReport(
                pattern_name=pattern_name,
                step1_tstat=0.0,
                step1_pvalue=1.0,
                step2_factor_type=FactorType.FAILED,
                step2_tstat_ratio=0.0,
                step3_ic=0.0,
                step3_ic_pvalue=None,
                step4_top_ret=0.0,
                step4_bottom_ret=0.0,
                step4_spread=0.0,
                passed=False,
                steps=[],
                warnings=[f"Insufficient samples: {n_valid} < {self._min_samples}"],
            )

        sig = signal[mask]
        ret = forward_return[mask]
        sec = sectors[mask] if sectors is not None else None
        lmc = log_mcap[mask] if log_mcap is not None else None

        # ── Purification ──
        purification: PurificationReport | None = None
        purified_sig = sig.copy()
        if sec is not None:
            purifier = FactorPurifier()
            purified_sig, purification = purifier.purify(sig, sec, lmc)
            purified_sig = np.nan_to_num(purified_sig, nan=0.0)

        steps: list[GateStepResult] = []

        # ── Step 1: Single-factor regression with industry dummies ──
        tstat, pval = self._step1_factor_regression(purified_sig, ret, sec, lmc)
        step1_pass = abs(tstat) >= self._tstat_threshold and pval < self._significance
        steps.append(
            GateStepResult(
                step=1,
                name="Factor Return t-stat",
                value=tstat,
                threshold=self._tstat_threshold,
                passed=step1_pass,
                details={"p_value": pval, "n_samples": n_valid},
            )
        )

        # ── Step 2: Return factor vs risk factor classification ──
        factor_type, tstat_ratio = self._step2_classify(purified_sig, ret, sec, lmc)
        step2_pass = factor_type == FactorType.RETURN
        steps.append(
            GateStepResult(
                step=2,
                name="Return vs Risk Factor",
                value=tstat_ratio,
                threshold=0.5,
                passed=step2_pass,
                details={"factor_type": factor_type.value},
            )
        )

        # ── Step 3: IC analysis (rank correlation) ──
        ic, ic_pval = self._step3_ic(purified_sig, ret)
        step3_pass = ic > 0.01 and (ic_pval is None or ic_pval < self._significance)
        steps.append(
            GateStepResult(
                step=3,
                name="Rank IC",
                value=ic,
                threshold=0.01,
                passed=step3_pass,
                details={"ic_pvalue": ic_pval},
            )
        )

        # ── Step 4: Quantile backtest ──
        top_ret, bottom_ret, spread = self._step4_quantile_test(purified_sig, ret, sec)
        step4_pass = spread > 0
        steps.append(
            GateStepResult(
                step=4,
                name="Quantile Spread",
                value=spread,
                threshold=0.0,
                passed=step4_pass,
                details={"top_ret": top_ret, "bottom_ret": bottom_ret},
            )
        )

        # ── Overall gate ──
        all_passed = all(s.passed for s in steps)
        if not all_passed:
            warnings.append(f"Gate FAILED at steps: {[s.step for s in steps if not s.passed]}")

        return EvaluationReport(
            pattern_name=pattern_name,
            step1_tstat=tstat,
            step1_pvalue=pval,
            step2_factor_type=factor_type,
            step2_tstat_ratio=tstat_ratio,
            step3_ic=ic,
            step3_ic_pvalue=ic_pval,
            step4_top_ret=top_ret,
            step4_bottom_ret=bottom_ret,
            step4_spread=spread,
            passed=all_passed,
            steps=steps,
            purification=purification,
            warnings=warnings,
        )

    def _step1_factor_regression(
        self,
        signal: np.ndarray,
        ret: np.ndarray,
        sectors: np.ndarray | None,
        log_mcap: np.ndarray | None,
    ) -> tuple[float, float]:
        """Step 1: Regress returns on signal + sector dummies + market cap."""
        n = len(ret)
        if sectors is None:
            # Simple regression: ret = alpha + beta * signal
            X = np.column_stack([np.ones(n), signal])
            coeffs, _, rank, _ = np.linalg.lstsq(X, ret, rcond=None)
            if rank < 2:
                return 0.0, 1.0
            residuals = ret - X @ coeffs
            beta = coeffs[1]
            se = np.sqrt(np.sum(residuals**2) / (n - 2)) / np.sqrt(
                np.sum((signal - signal.mean()) ** 2) + 1e-10
            )
            tstat = beta / max(se, 1e-10)
            pval = 2 * (1 - scipy_stats.t.cdf(abs(tstat), df=n - 2))
            return float(tstat), float(pval)

        # With sector dummies
        unique_sec = sorted(set(sectors))
        n_cols = 2 + len(unique_sec) + (1 if log_mcap is not None else 0)
        X = np.ones((n, n_cols), dtype=np.float64)
        X[:, 1] = signal
        for j, s in enumerate(unique_sec):
            X[:, 2 + j] = (sectors == s).astype(np.float64)
        if log_mcap is not None:
            X[:, -1] = np.asarray(log_mcap, dtype=np.float64)

        coeffs, residuals, rank, _ = np.linalg.lstsq(X, ret, rcond=None)
        if rank < 2:
            return 0.0, 1.0
        beta = coeffs[1]
        se_sq = np.sum(residuals**2) / (n - rank) if n > rank else 1e10
        xx_inv = np.linalg.inv(X.T @ X + np.eye(n_cols) * 1e-10)
        se = np.sqrt(max(se_sq * xx_inv[1, 1], 1e-20))
        tstat = float(beta / se)
        pval = float(2 * (1 - scipy_stats.t.cdf(abs(tstat), df=max(n - rank, 1))))
        return tstat, pval

    def _step2_classify(
        self,
        signal: np.ndarray,
        ret: np.ndarray,
        sectors: np.ndarray | None,
        log_mcap: np.ndarray | None,
    ) -> tuple[FactorType, float]:
        """Step 2: Classify as return factor or risk factor.

        A return factor has a directional t-stat that is consistently significant.
        We compute rolling window t-stats and check |t|>2 ratio.
        """
        n = len(ret)
        window = max(60, n // 4)
        if n < window * 2:
            return FactorType.FAILED, 0.0

        n_windows = n - window + 1
        tstats = np.zeros(n_windows, dtype=np.float64)
        for i in range(n_windows):
            ts, _ = self._step1_factor_regression(
                signal[i : i + window],
                ret[i : i + window],
                sectors[i : i + window] if sectors is not None else None,
                log_mcap[i : i + window] if log_mcap is not None else None,
            )
            tstats[i] = ts

        ratio = float(np.mean(np.abs(tstats) >= self._tstat_threshold))
        directional = float(np.mean(tstats > 0))

        if ratio >= 0.5 and directional >= 0.6:
            factor_type = FactorType.RETURN
        elif ratio >= 0.3:
            factor_type = FactorType.RISK
        else:
            factor_type = FactorType.FAILED

        return factor_type, ratio

    def _step3_ic(self, signal: np.ndarray, ret: np.ndarray) -> tuple[float, float | None]:
        """Step 3: Rank IC (Spearman correlation) of signal vs forward return."""
        result = scipy_stats.spearmanr(signal, ret)
        if isinstance(result, float):
            return result, None
        ic_tuple = result
        if hasattr(ic_tuple, "statistic"):
            return float(ic_tuple.statistic), float(
                ic_tuple.pvalue
            ) if ic_tuple.pvalue is not None else None
        corr, pval = ic_tuple
        if isinstance(pval, np.ndarray):
            pval = float(pval[()])
        else:
            pval = float(pval) if pval is not None else None
        return float(corr), pval

    def _step4_quantile_test(
        self,
        signal: np.ndarray,
        ret: np.ndarray,
        sectors: np.ndarray | None,
    ) -> tuple[float, float, float]:
        """Step 4: Quantile backtest — top vs bottom quantile returns.

        If sectors are provided, computes sector-neutral returns.
        """
        n = len(ret)
        if n < self._n_quantiles * 3:
            return 0.0, 0.0, 0.0

        # Assign quantiles
        quantile_bins = np.digitize(
            signal, np.percentile(signal, np.linspace(0, 100, self._n_quantiles + 1)[1:-1])
        )

        if sectors is not None:
            # Sector-neutral: subtract sector mean from each bar's return
            unique_sec = sorted(set(sectors))
            ret_neutral = ret.copy().astype(np.float64)
            for s in unique_sec:
                mask = sectors == s
                if mask.sum() > 0:
                    ret_neutral[mask] -= ret[mask].mean()
            ret = ret_neutral

        top_mask = quantile_bins >= self._n_quantiles - 1
        bottom_mask = quantile_bins <= 0

        top_ret = float(np.mean(ret[top_mask])) if top_mask.any() else 0.0
        bottom_ret = float(np.mean(ret[bottom_mask])) if bottom_mask.any() else 0.0
        spread = top_ret - bottom_ret

        return top_ret, bottom_ret, spread
