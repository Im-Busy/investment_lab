"""
Deflated Sharpe Ratio, Probabilistic Sharpe Ratio, False Discovery Rate,
Bootstrap Confidence Intervals, and Permutation Tests.

Implements advanced backtest validation statistics from Bailey & López de Prado
and empirical resampling methods:

- PSR (Probabilistic Sharpe Ratio): Probability that observed Sharpe exceeds a benchmark,
  accounting for non-normal returns (skewness, kurtosis).

- DSR (Deflated Sharpe Ratio): PSR evaluated against the expected maximum Sharpe
  under multiple testing — controls for selection bias when testing many strategies.

- FDR (False Discovery Rate): Benjamini-Hochberg procedure for multiple hypothesis
  testing correction across strategy ensembles.

- Bootstrap Sharpe CI: Empirical 95% confidence interval via block bootstrap,
  with MBB (moving block bootstrap) support for autocorrelated returns.

- Permutation Test: Shuffles returns to destroy temporal structure, builds null
  distribution, computes empirical p-value for any performance metric.

References:
    Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
    Journal of Portfolio Management.

    López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley.

    Aronson, D. (2006). "Evidence-Based Technical Analysis." Wiley.
    — Bootstrap CI for backtest metrics.

    Efron, B., & Tibshirani, R. J. (1994). "An Introduction to the Bootstrap."
    — Block bootstrap methodology.

Usage:
    from src.analysis.deflated_sharpe import (
        compute_psr, compute_dsr, benjamini_hochberg,
        bootstrap_sharpe_ci, permutation_test, format_significance_summary,
    )

    psr = compute_psr(sharpe=1.5, benchmark=0.0, n_obs=252, skew=-0.3, kurt=3.5)
    dsr = compute_dsr(sharpe=1.5, n_obs=252, n_trials=100, skew=-0.3, kurt=3.5)
    rejected = benjamini_hochberg(p_values, alpha=0.05)

    # Bootstrap 95% CI on Sharpe
    ci = bootstrap_sharpe_ci(returns_array, n_bootstrap=5000)

    # Permutation test: is the observed Sharpe significant?
    p_result = permutation_test(returns_array, metric_fn=_sharpe_ratio, n_permutations=5000)
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


class BootstrapCIResult(NamedTuple):
    """Bootstrap confidence interval for a performance metric.

    Attributes:
        metric_name: Name of the metric (e.g., "Sharpe Ratio").
        observed: Observed value of the metric.
        ci_lower: Lower bound of the confidence interval.
        ci_upper: Upper bound of the confidence interval.
        confidence: Confidence level (e.g., 0.95).
        n_bootstrap: Number of bootstrap resamples.
    """

    metric_name: str
    observed: float
    ci_lower: float
    ci_upper: float
    confidence: float
    n_bootstrap: int


class PermutationResult(NamedTuple):
    """Permutation test result for a backtest metric.

    Attributes:
        metric_name: Name of the metric tested.
        observed: Observed value of the metric.
        null_mean: Mean of the null distribution.
        null_std: Standard deviation of the null distribution.
        p_value: Empirical one-sided p-value (P(null >= observed) for Sharpe).
        n_permutations: Number of permutations performed.
        significant: Whether p_value < alpha (default 0.05).
    """

    metric_name: str
    observed: float
    null_mean: float
    null_std: float
    p_value: float
    n_permutations: int
    significant: bool


class SignificanceSummary(NamedTuple):
    """Combined statistical significance summary for a strategy.

    Attributes:
        sharpe: Observed annualized Sharpe ratio.
        sharpe_ci_lower: Bootstrap 95% CI lower bound.
        sharpe_ci_upper: Bootstrap 95% CI upper bound.
        psr: Probabilistic Sharpe Ratio (vs benchmark=0).
        dsr: Deflated Sharpe Ratio (vs E[max{SR}]).
        permutation_pvalue: Permutation test empirical p-value.
        n_trades: Number of trades (for context).
        significant: Consensus — True if 2+ tests indicate significance.
    """

    sharpe: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float
    psr: float
    dsr: float | None
    permutation_pvalue: float
    n_trades: int
    significant: bool


def _moving_block_indices(
    n_samples: int,
    block_size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate indices for moving block bootstrap (MBB).

    Draws random block starting positions, wraps indices circularly (stationary
    bootstrap variant), and concatenates blocks until n_samples indices are
    obtained.

    Args:
        n_samples: Total number of samples to generate.
        block_size: Size of each contiguous block.
        rng: NumPy random state.

    Returns:
        Array of integer indices of length n_samples.
    """
    n_blocks = int(np.ceil(n_samples / block_size))
    indices = np.empty(n_blocks * block_size, dtype=np.intp)
    for i in range(n_blocks):
        start = rng.randint(0, n_samples)
        indices[i * block_size : (i + 1) * block_size] = (
            np.arange(start, start + block_size) % n_samples
        )
    return indices[:n_samples]


def bootstrap_sharpe_ci(
    returns: np.ndarray,
    periods_per_year: int = 252,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    use_block_bootstrap: bool = False,
    block_size: int = 10,
    seed: int | None = 42,
) -> BootstrapCIResult:
    """Compute bootstrap confidence interval for annualized Sharpe ratio.

    Uses percentile bootstrap: resamples returns with replacement (or block
    bootstrap for autocorrelated series), recomputes Sharpe each iteration,
    takes alpha/2 and 1-alpha/2 percentiles as CI bounds.

    Args:
        returns: Array of period returns.
        periods_per_year: Frequency multiplier (252 daily, 52 weekly).
        n_bootstrap: Number of bootstrap resamples (default 5000).
        confidence: Confidence level (default 0.95).
        use_block_bootstrap: If True, use moving block bootstrap to preserve
            autocorrelation structure. Recommended for daily returns.
        block_size: Block size for MBB (default 10 ≈ 2 weeks).
        seed: Random seed for reproducibility.

    Returns:
        BootstrapCIResult with observed Sharpe and confidence interval.
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    if len(returns) < 10:
        return BootstrapCIResult(
            metric_name="Sharpe Ratio",
            observed=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
        )

    observed = _sharpe_ratio(returns, periods_per_year)
    rng = np.random.RandomState(seed)
    n = len(returns)

    bootstrapped = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        if use_block_bootstrap:
            idx = _moving_block_indices(n, block_size, rng)
        else:
            idx = rng.randint(0, n, size=n)
        sample = returns[idx]
        bootstrapped[i] = _sharpe_ratio(sample, periods_per_year)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(bootstrapped, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrapped, 100 * (1 - alpha / 2)))

    return BootstrapCIResult(
        metric_name="Sharpe Ratio",
        observed=observed,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
    )


def permutation_test(
    returns: np.ndarray,
    metric_fn: callable | None = None,
    periods_per_year: int = 252,
    n_permutations: int = 5000,
    alternative: str = "greater",
    seed: int | None = 42,
) -> PermutationResult:
    """Monte Carlo permutation test for backtest overfitting.

    Randomly shuffles (permutes) the return series to destroy any temporal
    structure or predictive pattern. The metric computed on each permuted
    series forms the null distribution. The empirical p-value is the fraction
    of permuted metrics exceeding (or below) the observed metric.

    If the strategy has genuine predictive power, the observed Sharpe should
    be extreme relative to the null distribution (low p-value).

    Args:
        returns: Array of period returns.
        metric_fn: Callable(returns, periods_per_year) -> float. Defaults to
            _sharpe_ratio if None.
        periods_per_year: Frequency multiplier.
        n_permutations: Number of permutations (default 5000).
        alternative: 'greater' for Sharpe-like metrics (observed > null),
            'less' for drawdown-like metrics, 'two-sided' for symmetrical.
        seed: Random seed.

    Returns:
        PermutationResult with p-value and significance flag.
    """
    if metric_fn is None:
        metric_fn = lambda r, ppy: _sharpe_ratio(r, ppy)  # noqa: E731

    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    if len(returns) < 5:
        return PermutationResult(
            metric_name="Sharpe Ratio",
            observed=0.0,
            null_mean=0.0,
            null_std=0.0,
            p_value=1.0,
            n_permutations=0,
            significant=False,
        )

    observed = metric_fn(returns, periods_per_year)
    rng = np.random.RandomState(seed)
    n = len(returns)

    null_dist = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(returns)
        null_dist[i] = metric_fn(shuffled, periods_per_year)

    if alternative == "greater":
        p_value = float(np.mean(null_dist >= observed))
    elif alternative == "less":
        p_value = float(np.mean(null_dist <= observed))
    else:
        p_value = float(
            np.mean(np.abs(null_dist - np.mean(null_dist)) >= np.abs(observed - np.mean(null_dist)))
        )

    return PermutationResult(
        metric_name="Sharpe Ratio",
        observed=observed,
        null_mean=float(np.mean(null_dist)),
        null_std=float(np.std(null_dist)),
        p_value=p_value,
        n_permutations=n_permutations,
        significant=p_value < 0.05,
    )


def bootstrap_metric_ci(
    returns: np.ndarray,
    metric_fn: callable,
    metric_name: str = "Metric",
    periods_per_year: int = 252,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int | None = 42,
) -> BootstrapCIResult:
    """Bootstrap confidence interval for any user-supplied metric.

    Generic bootstrap wrapper that works with any metric function — Sortino,
    Calmar, max drawdown, win rate, etc.

    Args:
        returns: Array of period returns.
        metric_fn: Callable(returns, periods_per_year) -> float.
        metric_name: Display name for the metric.
        periods_per_year: Frequency multiplier.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level.
        seed: Random seed.

    Returns:
        BootstrapCIResult with observed value and CI.
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    if len(returns) < 10:
        return BootstrapCIResult(
            metric_name=metric_name,
            observed=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
        )

    observed = metric_fn(returns, periods_per_year)
    rng = np.random.RandomState(seed)
    n = len(returns)

    bootstrapped = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        bootstrapped[i] = metric_fn(returns[idx], periods_per_year)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(bootstrapped, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrapped, 100 * (1 - alpha / 2)))

    return BootstrapCIResult(
        metric_name=metric_name,
        observed=observed,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
    )


def format_significance_summary(
    returns: np.ndarray,
    n_trades: int = 0,
    n_trials: int = 100,
    periods_per_year: int = 252,
    n_bootstrap: int = 5000,
    n_permutations: int = 5000,
    seed: int | None = 42,
) -> SignificanceSummary:
    """Run all statistical significance tests and produce a summary.

    This is the one-call convenience function for comprehensive backtest
    significance. Runs bootstrap CI, PSR, DSR, and permutation test in
    one pass.

    Args:
        returns: Array of daily strategy returns.
        n_trades: Number of trades (for context, not used in computation).
        n_trials: Number of strategies tested (for DSR).
        periods_per_year: Trading days per year.
        n_bootstrap: Bootstrap resamples for CI.
        n_permutations: Permutations for null distribution.
        seed: Random seed.

    Returns:
        SignificanceSummary with consolidated results.
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 10:
        return SignificanceSummary(
            sharpe=0.0,
            sharpe_ci_lower=0.0,
            sharpe_ci_upper=0.0,
            psr=0.0,
            dsr=None,
            permutation_pvalue=1.0,
            n_trades=n_trades,
            significant=False,
        )

    sharpe = _sharpe_ratio(returns, periods_per_year)

    ci = bootstrap_sharpe_ci(
        returns,
        periods_per_year=periods_per_year,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    psr_result = compute_psr_from_returns(
        returns,
        benchmark=0.0,
        periods_per_year=periods_per_year,
    )

    try:
        dsr_result = compute_dsr_from_returns(
            returns,
            n_trials=n_trials,
            periods_per_year=periods_per_year,
            seed=seed,
        )
        dsr_val = dsr_result.psr
    except (ValueError, RuntimeError):
        dsr_val = None

    perm_result = permutation_test(
        returns,
        periods_per_year=periods_per_year,
        n_permutations=n_permutations,
        seed=seed,
    )

    significant_count = sum(
        [
            sharpe > 0 and ci.ci_lower > 0,
            psr_result.psr >= 0.90,
            perm_result.significant,
        ]
    )

    return SignificanceSummary(
        sharpe=sharpe,
        sharpe_ci_lower=ci.ci_lower,
        sharpe_ci_upper=ci.ci_upper,
        psr=psr_result.psr,
        dsr=dsr_val,
        permutation_pvalue=perm_result.p_value,
        n_trades=n_trades,
        significant=significant_count >= 2,
    )


def print_significance_report(
    returns: np.ndarray,
    n_trades: int = 0,
    n_trials: int = 100,
) -> str:
    """Generate a human-readable statistical significance report.

    Args:
        returns: Array of daily strategy returns.
        n_trades: Number of trades executed.
        n_trials: Number of strategies considered for DSR correction.

    Returns:
        Multi-line formatted string.
    """
    s = format_significance_summary(returns, n_trades=n_trades, n_trials=n_trials)

    lines = [
        "=" * 62,
        "STATISTICAL SIGNIFICANCE REPORT",
        "=" * 62,
        "",
        f"  Observed Sharpe Ratio:           {s.sharpe:>10.3f}",
        f"  Bootstrap 95% CI:                [{s.sharpe_ci_lower:.3f}, {s.sharpe_ci_upper:.3f}]",
        f"  Probabilistic Sharpe Ratio:      {s.psr:>10.3f}  (P(SR > 0))",
    ]

    if s.dsr is not None:
        lines.append(
            f"  Deflated Sharpe Ratio:           {s.dsr:>10.3f}  (vs E[max SR] across {n_trials} trials)"
        )

    lines.extend(
        [
            f"  Permutation Test p-value:        {s.permutation_pvalue:>10.4f}  (empirical, {5000} shuffles)",
            f"  Trades:                          {s.n_trades:>10}",
            "",
        ]
    )

    if s.significant:
        lines.append("  CONCLUSION: STATISTICALLY SIGNIFICANT (2+ tests pass)")
    else:
        lines.append("  CONCLUSION: NOT STATISTICALLY SIGNIFICANT -- treat with caution")

    lines.append("")
    lines.append("  Interpretation:")
    lines.append("    * Bootstrap CI > 0 ==> Sharpe unlikely due to chance")
    lines.append("    * PSR > 0.90 ==> High probability true Sharpe > 0")
    lines.append("    * DSR > 0.80 ==> Survives multiple-testing correction")
    lines.append("    * Permutation p < 0.05 ==> Strategy beats random shuffling")
    lines.append("=" * 62)

    return "\n".join(lines)
