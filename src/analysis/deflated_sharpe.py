"""
Deflated Sharpe Ratio, Probabilistic Sharpe Ratio, and False Discovery Rate.

Implements advanced backtest validation statistics from Bailey & López de Prado:

- PSR (Probabilistic Sharpe Ratio): Probability that observed Sharpe exceeds a benchmark,
  accounting for non-normal returns (skewness, kurtosis).

- DSR (Deflated Sharpe Ratio): PSR evaluated against the expected maximum Sharpe
  under multiple testing — controls for selection bias when testing many strategies.

- FDR (False Discovery Rate): Benjamini-Hochberg procedure for multiple hypothesis
  testing correction across strategy ensembles.

References:
    Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
    Journal of Portfolio Management.

    López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley.

Usage:
    from src.analysis.deflated_sharpe import compute_psr, compute_dsr, benjamini_hochberg

    psr = compute_psr(sharpe=1.5, benchmark=0.0, n_obs=252, skew=-0.3, kurt=3.5)
    dsr = compute_dsr(sharpe=1.5, n_obs=252, n_trials=100, skew=-0.3, kurt=3.5)
    rejected = benjamini_hochberg(p_values, alpha=0.05)
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import NamedTuple


class PSRResult(NamedTuple):
    """Result of Probabilistic Sharpe Ratio computation.

    Attributes:
        psr: Probability that true Sharpe > benchmark (0 to 1).
        sharpe: Observed Sharpe ratio.
        benchmark: Benchmark Sharpe ratio.
        standard_error: Standard error of the Sharpe ratio estimate.
        deflated: Whether this is a deflated (DSR) result.
    """

    psr: float
    sharpe: float
    benchmark: float
    standard_error: float
    deflated: bool


class FDRResult(NamedTuple):
    """Result of Benjamini-Hochberg FDR procedure.

    Attributes:
        rejected: Indices of rejected hypotheses (0-based).
        threshold: p-value threshold for rejection.
        n_rejected: Number of rejected hypotheses.
        alpha: Desired FDR level.
    """

    rejected: np.ndarray
    threshold: float
    n_rejected: int
    alpha: float


def _sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Compute annualized Sharpe ratio from excess returns.

    Args:
        returns: Array of period excess returns.
        periods_per_year: Frequency multiplier (252 daily, 52 weekly, 12 monthly).

    Returns:
        Annualized Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0.0
    return float((mean / std) * np.sqrt(periods_per_year))


def compute_psr(
    sharpe: float,
    benchmark: float = 0.0,
    n_obs: int = 252,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> PSRResult:
    """Compute Probabilistic Sharpe Ratio.

    PSR = Φ( (SR̂ - SR*) * √(T-1) / √(1 - γ̂₃·SR̂ + (γ̂₄-1)/4 · SR̂²) )

    where SR̂ is the observed Sharpe, SR* is the benchmark, T is the number
    of observations, γ̂₃ is skewness, γ̂₄ is kurtosis, and Φ is the standard
    normal CDF.

    Args:
        sharpe: Observed annualized Sharpe ratio.
        benchmark: Benchmark Sharpe ratio (default 0).
        n_obs: Number of observations used to estimate Sharpe.
        skew: Skewness of returns.
        kurt: Kurtosis of returns (normal = 3).

    Returns:
        PSRResult with psr, sharpe, benchmark, standard_error, deflated=False.
    """
    if n_obs < 2:
        return PSRResult(
            psr=0.0, sharpe=sharpe, benchmark=benchmark, standard_error=float("inf"), deflated=False
        )

    numer = (sharpe - benchmark) * np.sqrt(n_obs - 1)
    denom_sq = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe**2
    if denom_sq <= 0:
        return PSRResult(
            psr=0.0, sharpe=sharpe, benchmark=benchmark, standard_error=float("inf"), deflated=False
        )
    denom = np.sqrt(denom_sq)

    z_score = numer / denom
    psr_value = float(stats.norm.cdf(z_score))
    se = denom / np.sqrt(n_obs - 1)

    return PSRResult(
        psr=psr_value, sharpe=sharpe, benchmark=benchmark, standard_error=se, deflated=False
    )


def compute_psr_from_returns(
    returns: np.ndarray,
    benchmark: float = 0.0,
    periods_per_year: int = 252,
) -> PSRResult:
    """Compute PSR directly from a return series.

    Automatically estimates Sharpe, skewness, and kurtosis from the data.

    Args:
        returns: Array of period excess returns.
        benchmark: Benchmark Sharpe ratio (default 0).
        periods_per_year: Frequency multiplier.

    Returns:
        PSRResult with estimated psr.
    """
    n = len(returns)
    if n < 3:
        return PSRResult(
            psr=0.0, sharpe=0.0, benchmark=benchmark, standard_error=float("inf"), deflated=False
        )

    sharpe = _sharpe_ratio(returns, periods_per_year)
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=False))

    return compute_psr(sharpe=sharpe, benchmark=benchmark, n_obs=n, skew=skew, kurt=kurt)


def _expected_max_sharpe(
    n_trials: int,
    n_obs: int,
    variance: float = 1.0,
    n_simulations: int = 5000,
    seed: int | None = 42,
) -> float:
    """Estimate expected maximum Sharpe ratio under the null via Monte Carlo.

    Under the null hypothesis (zero mean returns), the t-statistic for each
    trial follows a Student's t distribution with n_obs-1 degrees of freedom.
    The maximum t-stat across N trials follows an extreme value distribution.

    Args:
        n_trials: Number of independent strategies tested.
        n_obs: Observations per strategy.
        variance: Variance of the returns (default 1 for standardized).
        n_simulations: Monte Carlo simulation count.
        seed: Random seed for reproducibility.

    Returns:
        Expected maximum annualized Sharpe ratio under the null.
    """
    rng = np.random.RandomState(seed)
    df = max(1, n_obs - 1)

    max_sharpes = np.empty(n_simulations)
    for i in range(n_simulations):
        draws = rng.standard_t(df, size=n_trials)
        max_t = np.max(np.abs(draws))
        max_sharpes[i] = max_t / np.sqrt(n_obs)

    return float(np.mean(max_sharpes) * np.sqrt(variance))


def compute_dsr(
    sharpe: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    variance: float = 1.0,
    n_simulations: int = 5000,
    seed: int | None = 42,
) -> PSRResult:
    """Compute Deflated Sharpe Ratio.

    DSR = PSR(SR̂, E[max{SR₀}], T, γ̂₃, γ̂₄)

    Instead of testing against a fixed benchmark (e.g., SR* = 0), the DSR
    tests against the expected maximum Sharpe ratio that would arise from
    testing n_trials random strategies. This corrects for selection bias.

    Args:
        sharpe: Observed annualized Sharpe ratio.
        n_obs: Number of observations used to estimate Sharpe.
        n_trials: Number of independent strategies tested.
        skew: Skewness of returns.
        kurt: Kurtosis of returns (normal = 3).
        variance: Variance of returns (default 1).
        n_simulations: Monte Carlo draws for E[max{SR₀}].
        seed: Random seed.

    Returns:
        PSRResult with deflated=True, benchmark = E[max{SR₀}].
    """
    emax = _expected_max_sharpe(
        n_trials=n_trials,
        n_obs=n_obs,
        variance=variance,
        n_simulations=n_simulations,
        seed=seed,
    )

    result = compute_psr(
        sharpe=sharpe,
        benchmark=float(emax),
        n_obs=n_obs,
        skew=skew,
        kurt=kurt,
    )

    return PSRResult(
        psr=result.psr,
        sharpe=result.sharpe,
        benchmark=float(emax),
        standard_error=result.standard_error,
        deflated=True,
    )


def compute_dsr_from_returns(
    returns: np.ndarray,
    n_trials: int,
    periods_per_year: int = 252,
    n_simulations: int = 5000,
    seed: int | None = 42,
) -> PSRResult:
    """Compute DSR directly from a return series.

    Args:
        returns: Array of period excess returns.
        n_trials: Number of independent strategies tested.
        periods_per_year: Frequency multiplier.
        n_simulations: Monte Carlo draws for E[max{SR₀}].
        seed: Random seed.

    Returns:
        PSRResult with deflated=True.
    """
    n = len(returns)
    if n < 3:
        return PSRResult(
            psr=0.0, sharpe=0.0, benchmark=0.0, standard_error=float("inf"), deflated=True
        )

    sharpe = _sharpe_ratio(returns, periods_per_year)
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=False))
    variance = float(np.var(returns, ddof=1))
    ann_variance = variance * periods_per_year

    return compute_dsr(
        sharpe=sharpe,
        n_obs=n,
        n_trials=n_trials,
        skew=skew,
        kurt=kurt,
        variance=ann_variance,
        n_simulations=n_simulations,
        seed=seed,
    )


def benjamini_hochberg(
    p_values: np.ndarray | list[float],
    alpha: float = 0.05,
) -> FDRResult:
    """Benjamini-Hochberg procedure for FDR control.

    Controls the expected proportion of false discoveries among rejected
    hypotheses. Sorts p-values, finds largest k such that:
        p(k) ≤ (k / m) * alpha
    and rejects hypotheses 1..k.

    Args:
        p_values: Array of p-values from multiple tests.
        alpha: Desired false discovery rate level (default 0.05).

    Returns:
        FDRResult with rejected indices, threshold, count, and alpha.
    """
    p = np.asarray(p_values, dtype=np.float64)
    p = p[~np.isnan(p)]

    if len(p) == 0:
        return FDRResult(
            rejected=np.array([], dtype=int),
            threshold=0.0,
            n_rejected=0,
            alpha=alpha,
        )

    m = len(p)
    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]

    ranks = np.arange(1, m + 1)
    bh_thresholds = (ranks / m) * alpha

    significant = sorted_p <= bh_thresholds

    if not np.any(significant):
        return FDRResult(
            rejected=np.array([], dtype=int),
            threshold=0.0,
            n_rejected=0,
            alpha=alpha,
        )

    max_k = int(np.max(np.where(significant)[0]))
    rejected = sorted_idx[: max_k + 1]
    threshold = float(sorted_p[max_k])

    return FDRResult(
        rejected=rejected,
        threshold=threshold,
        n_rejected=len(rejected),
        alpha=alpha,
    )


def deflated_sharpe_batch(
    sharpes: np.ndarray | list[float],
    n_obs: int,
    n_trials: int,
    skews: np.ndarray | list[float] | None = None,
    kurts: np.ndarray | list[float] | None = None,
    n_simulations: int = 5000,
    seed: int | None = 42,
) -> list[PSRResult]:
    """Compute DSR for a batch of strategies sharing the same data frequency.

    Uses a single E[max{SR₀}] estimate for all strategies since they were
    tested over the same period with the same number of observations.

    Args:
        sharpes: Array of observed annualized Sharpe ratios.
        n_obs: Observations shared by all strategies.
        n_trials: Number of independent strategies tested.
        skews: Optional per-strategy skewness (default 0).
        kurts: Optional per-strategy kurtosis (default 3).
        n_simulations: Monte Carlo draws.
        seed: Random seed.

    Returns:
        List of PSRResult, one per strategy.
    """
    s = np.asarray(sharpes, dtype=np.float64)
    if skews is None:
        skews = np.zeros_like(s)
    else:
        skews = np.asarray(skews, dtype=np.float64)
    if kurts is None:
        kurts = np.full_like(s, 3.0)
    else:
        kurts = np.asarray(kurts, dtype=np.float64)

    emax = _expected_max_sharpe(n_trials, n_obs, n_simulations=n_simulations, seed=seed)

    results = []
    for i in range(len(s)):
        result = compute_psr(
            sharpe=float(s[i]),
            benchmark=float(emax),
            n_obs=n_obs,
            skew=float(skews[i]),
            kurt=float(kurts[i]),
        )
        results.append(
            PSRResult(
                psr=result.psr,
                sharpe=result.sharpe,
                benchmark=float(emax),
                standard_error=result.standard_error,
                deflated=True,
            )
        )
    return results


def dsr_significance(
    dsr_value: float,
    thresholds: dict[str, float] | None = None,
) -> dict[str, bool]:
    """Interpret DSR significance level.

    Args:
        dsr_value: Deflated Sharpe Ratio probability (0 to 1).
        thresholds: Dict of {label: threshold}. Default levels:
            - significant: PSR >= 0.95
            - borderline: PSR >= 0.80
            - not_significant: otherwise

    Returns:
        Dict with boolean flags for each threshold.
    """
    if thresholds is None:
        thresholds = {
            "significant": 0.95,
            "borderline": 0.80,
        }

    flags = {}
    for label, thresh in thresholds.items():
        flags[label] = dsr_value >= thresh
    return flags
