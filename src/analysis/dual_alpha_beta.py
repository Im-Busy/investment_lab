"""P25: Dual alpha/beta model — separate alpha/beta for bull vs bear markets.

Single-index models understate alpha for convex strategies (higher beta in bulls)
and overstate alpha for concave strategies (higher beta in bears). The dual model
splits alpha and beta by market regime to detect phantom alpha.

Reference: Mitchell & Pulvino (2001), E9 in Master Comparison Report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DualAlphaBetaResult:
    """Dual alpha/beta decomposition by regime."""

    # Single-index model
    single_alpha: float
    single_beta: float
    single_r2: float

    # Bull regime (market return > 0)
    bull_alpha: float
    bull_beta: float
    bull_n: int

    # Bear regime (market return < 0)
    bear_alpha: float
    bear_beta: float
    bear_n: int

    # Diagnostics
    beta_asymmetry: float  # bull_beta - bear_beta
    alpha_gap: float  # bull_alpha - bear_alpha
    phantom_alpha: float  # single_alpha - (bull_alpha * bull_weight + bear_alpha * bear_weight)
    chow_statistic: float
    chow_pvalue: float
    is_structural_break: bool

    # Classification
    is_convex: bool  # higher beta in bulls → true alpha is LOWER than single-index
    is_concave: bool  # higher beta in bears → true alpha is HIGHER than single-index
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Dual α/β: single α={self.single_alpha:.4f} β={self.single_beta:.3f} R²={self.single_r2:.3f}",
            f"  Bull (n={self.bull_n}): α={self.bull_alpha:.4f} β={self.bull_beta:.3f}",
            f"  Bear (n={self.bear_n}): α={self.bear_alpha:.4f} β={self.bear_beta:.3f}",
            f"  Asymmetry: Δβ={self.beta_asymmetry:.3f} Δα={self.alpha_gap:.4f} phantom={self.phantom_alpha:.4f}",
        ]
        if self.is_convex:
            lines.append("  STRATEGY: Convex — single-index OVERestimates alpha")
        if self.is_concave:
            lines.append("  STRATEGY: Concave — single-index UNDERestimates alpha")
        if self.is_structural_break:
            lines.append(
                f"  CHOW TEST: structural break (F={self.chow_statistic:.2f}, p={self.chow_pvalue:.4f})"
            )
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(__name__)


BETA_ASYMMETRY_THRESHOLD = 0.15
CHOW_SIGNIFICANCE = 0.05


def compute_dual_alpha_beta(
    strategy_returns: pd.Series | np.ndarray,
    market_returns: pd.Series | np.ndarray,
    regime_threshold: float = 0.0,
) -> DualAlphaBetaResult:
    """Decompose alpha/beta separately for bull and bear regimes.

    Args:
        strategy_returns: Strategy excess returns (over risk-free).
        market_returns: Market excess returns (over risk-free).
        regime_threshold: Market return threshold for bull/bear split (default 0).

    Returns:
        DualAlphaBetaResult with single and regime-specific alpha/beta.
    """
    strategy = np.asarray(strategy_returns).flatten()
    market = np.asarray(market_returns).flatten()

    if len(strategy) != len(market):
        raise ValueError(f"Length mismatch: strategy={len(strategy)}, market={len(market)}")
    if len(strategy) < 20:
        return DualAlphaBetaResult(
            single_alpha=0,
            single_beta=0,
            single_r2=0,
            bull_alpha=0,
            bull_beta=0,
            bull_n=0,
            bear_alpha=0,
            bear_beta=0,
            bear_n=0,
            beta_asymmetry=0,
            alpha_gap=0,
            phantom_alpha=0,
            chow_statistic=0,
            chow_pvalue=1.0,
            is_structural_break=False,
            is_convex=False,
            is_concave=False,
            warnings=["Insufficient data (<20 obs)"],
        )

    valid = ~(np.isnan(strategy) | np.isnan(market))
    strategy = strategy[valid]
    market = market[valid]

    if len(strategy) < 20:
        return DualAlphaBetaResult(
            single_alpha=0,
            single_beta=0,
            single_r2=0,
            bull_alpha=0,
            bull_beta=0,
            bull_n=0,
            bear_alpha=0,
            bear_beta=0,
            bear_n=0,
            beta_asymmetry=0,
            alpha_gap=0,
            phantom_alpha=0,
            chow_statistic=0,
            chow_pvalue=1.0,
            is_structural_break=False,
            is_convex=False,
            is_concave=False,
            warnings=["Insufficient valid data after NaN removal"],
        )

    X = np.column_stack([np.ones(len(strategy)), market])
    single_coef, single_resid, _, _ = np.linalg.lstsq(X, strategy, rcond=None)
    single_alpha, single_beta = float(single_coef[0]), float(single_coef[1])
    ss_total = np.sum((strategy - np.mean(strategy)) ** 2)
    ss_resid = np.sum(single_resid**2) if single_resid.size > 0 else ss_total
    single_r2 = 1 - (ss_resid / ss_total) if ss_total > 1e-15 else 0.0

    bull_mask = market > regime_threshold
    bear_mask = market <= regime_threshold

    bull_n = int(np.sum(bull_mask))
    bear_n = int(np.sum(bear_mask))

    bull_alpha, bull_beta = 0.0, 0.0
    bear_alpha, bear_beta = 0.0, 0.0

    warnings: list[str] = []

    if bull_n >= 5:
        X_bull = np.column_stack([np.ones(bull_n), market[bull_mask]])
        y_bull = strategy[bull_mask]
        coef_bull, _, _, _ = np.linalg.lstsq(X_bull, y_bull, rcond=None)
        bull_alpha, bull_beta = float(coef_bull[0]), float(coef_bull[1])
    else:
        warnings.append(f"Bull regime has only {bull_n} obs (< 5)")

    if bear_n >= 5:
        X_bear = np.column_stack([np.ones(bear_n), market[bear_mask]])
        y_bear = strategy[bear_mask]
        coef_bear, _, _, _ = np.linalg.lstsq(X_bear, y_bear, rcond=None)
        bear_alpha, bear_beta = float(coef_bear[0]), float(coef_bear[1])
    else:
        warnings.append(f"Bear regime has only {bear_n} obs (< 5)")

    beta_asymmetry = bull_beta - bear_beta
    alpha_gap = bull_alpha - bear_alpha
    bull_weight = bull_n / (bull_n + bear_n) if (bull_n + bear_n) > 0 else 0.5
    weighted_true_alpha = bull_alpha * bull_weight + bear_alpha * (1 - bull_weight)
    phantom_alpha = single_alpha - weighted_true_alpha

    is_convex = beta_asymmetry > BETA_ASYMMETRY_THRESHOLD
    is_concave = beta_asymmetry < -BETA_ASYMMETRY_THRESHOLD

    chow_stat, chow_pval, is_break = _chow_test(market, strategy, bull_mask, bear_mask)

    if is_convex:
        warnings.append(
            f"Convex strategy detected: β_bull({bull_beta:.3f}) >> β_bear({bear_beta:.3f}). "
            f"Single-index α ({single_alpha:.4f}) overestimates true α ({weighted_true_alpha:.4f}) "
            f"by {phantom_alpha:.4f}"
        )
    if is_concave:
        warnings.append(
            f"Concave strategy detected: β_bear({bear_beta:.3f}) >> β_bull({bull_beta:.3f}). "
            f"Single-index α ({single_alpha:.4f}) underestimates true α ({weighted_true_alpha:.4f}) "
            f"by {abs(phantom_alpha):.4f}"
        )

    return DualAlphaBetaResult(
        single_alpha=single_alpha,
        single_beta=single_beta,
        single_r2=float(single_r2),
        bull_alpha=bull_alpha,
        bull_beta=bull_beta,
        bull_n=bull_n,
        bear_alpha=bear_alpha,
        bear_beta=bear_beta,
        bear_n=bear_n,
        beta_asymmetry=beta_asymmetry,
        alpha_gap=alpha_gap,
        phantom_alpha=phantom_alpha,
        chow_statistic=chow_stat,
        chow_pvalue=chow_pval,
        is_structural_break=is_break,
        is_convex=is_convex,
        is_concave=is_concave,
        warnings=warnings,
    )


def _chow_test(
    market: np.ndarray,
    strategy: np.ndarray,
    bull_mask: np.ndarray,
    bear_mask: np.ndarray,
) -> tuple[float, float, bool]:
    """Chow test for structural break between bull and bear regimes."""
    try:
        n_bull = int(np.sum(bull_mask))
        n_bear = int(np.sum(bear_mask))
        n_total = len(market)
        k = 2  # intercept + beta

        if n_bull < k or n_bear < k:
            return 0.0, 1.0, False

        X_total = np.column_stack([np.ones(n_total), market])
        _, resid_total, _, _ = np.linalg.lstsq(X_total, strategy, rcond=None)
        rss_total = float(np.sum(resid_total**2))

        X_bull = np.column_stack([np.ones(n_bull), market[bull_mask]])
        _, resid_bull, _, _ = np.linalg.lstsq(X_bull, strategy[bull_mask], rcond=None)
        rss_bull = float(np.sum(resid_bull**2))

        X_bear = np.column_stack([np.ones(n_bear), market[bear_mask]])
        _, resid_bear, _, _ = np.linalg.lstsq(X_bear, strategy[bear_mask], rcond=None)
        rss_bear = float(np.sum(resid_bear**2))

        rss_unrestricted = rss_bull + rss_bear
        if rss_unrestricted < 1e-15:
            return 0.0, 1.0, False

        f_stat = ((rss_total - rss_unrestricted) / k) / (rss_unrestricted / (n_total - 2 * k))
        p_value = 1.0 - float(stats.f.cdf(f_stat, k, n_total - 2 * k))
        is_break = p_value < CHOW_SIGNIFICANCE

        return float(f_stat), p_value, is_break
    except Exception:
        return 0.0, 1.0, False


def rolling_dual_alpha_beta(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 252,
    step: int = 21,
) -> pd.DataFrame:
    """Rolling dual alpha/beta decomposition over time.

    Args:
        strategy_returns: Strategy excess returns.
        market_returns: Market excess returns.
        window: Rolling window size (default 252 = 1 year).
        step: Step size between windows (default 21 = 1 month).

    Returns:
        DataFrame with rolling alpha, beta, asymmetry metrics.
    """
    results = []
    dates = []

    for start in range(0, len(strategy_returns) - window, step):
        end = start + window
        slice_r = strategy_returns.iloc[start:end]
        slice_m = market_returns.iloc[start:end]

        result = compute_dual_alpha_beta(slice_r, slice_m)

        results.append(
            {
                "single_alpha": result.single_alpha,
                "single_beta": result.single_beta,
                "bull_alpha": result.bull_alpha,
                "bull_beta": result.bull_beta,
                "bear_alpha": result.bear_alpha,
                "bear_beta": result.bear_beta,
                "beta_asymmetry": result.beta_asymmetry,
                "phantom_alpha": result.phantom_alpha,
                "is_convex": result.is_convex,
                "chow_pvalue": result.chow_pvalue,
            }
        )
        dates.append(slice_r.index[-1])

    return pd.DataFrame(results, index=pd.DatetimeIndex(dates))
