"""P24-32: Divergence-in-bits strategy comparison metric.

Δg = D_KL(W* || W_A) − D_KL(W* || W_B)

Measures how much more information (in bits) Strategy A captures relative
to Strategy B by comparing each strategy's return distribution to an optimal
target distribution. Unit-independent — more robust than Sharpe for comparing
strategies with different risk profiles.

Reference: E5 in Master Comparison Report, "Investing Is Compression" (arXiv:2604.10758).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

EPSILON = 1e-10


@dataclass
class DivergenceBitsResult:
    """Result of divergence-in-bits comparison between two strategies."""

    strategy_a_name: str
    strategy_b_name: str
    delta_g_bits: float
    """Δg = D_KL(target || A) - D_KL(target || B).  Δg < 0 → A is better."""

    kl_a: float
    """KL divergence from target to strategy A."""

    kl_b: float
    """KL divergence from target to strategy B."""

    target_entropy: float
    """Entropy of the target distribution."""

    a_entropy: float
    b_entropy: float

    winner: str
    """Name of the strategy with lower KL divergence."""

    reduction_pct: float
    """Percent reduction in KL divergence of winner vs loser."""


def _to_pmf(returns: np.ndarray, bins: int = 50) -> np.ndarray:
    """Convert returns array to a probability mass function via histogram."""
    hist, _ = np.histogram(returns, bins=bins, density=True)
    bin_width = (returns.max() - returns.min()) / bins if len(returns) > 1 else 1.0
    pmf = hist * bin_width
    pmf = np.maximum(pmf, EPSILON)
    return pmf / pmf.sum()


def _kl_divergence(p_pmf: np.ndarray, q_pmf: np.ndarray) -> float:
    """KL divergence D_KL(P || Q) in bits (log base 2)."""
    return float(np.sum(p_pmf * np.log2(p_pmf / q_pmf)))


def compute_divergence_bits(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    target_returns: np.ndarray | None = None,
    bins: int = 50,
    name_a: str = "Strategy A",
    name_b: str = "Strategy B",
) -> DivergenceBitsResult:
    """Compare two strategies using the divergence-in-bits metric.

    Δg measures how much closer one strategy's return distribution is to a
    target distribution (e.g., a perfect-profit distribution or an optimal
    benchmark). A negative Δg means Strategy A is better (lower KL from target).

    Args:
        returns_a: Daily return series for strategy A.
        returns_b: Daily return series for strategy B.
        target_returns: Target return series. If None, uses a Dirac-at-best-profit
            distribution (all mass at the maximum individual-period return).
        bins: Number of histogram bins for PMF estimation.
        name_a: Name for strategy A.
        name_b: Name for strategy B.

    Returns:
        DivergenceBitsResult with Δg and per-strategy KL divergences.
    """
    returns_a = np.asarray(returns_a, dtype=np.float64)
    returns_b = np.asarray(returns_b, dtype=np.float64)

    if len(returns_a) < 10 or len(returns_b) < 10:
        raise ValueError("At least 10 observations required per strategy.")

    if target_returns is None:
        best_return = max(returns_a.max(), returns_b.max())
        target_returns = np.full(len(returns_a), best_return)

    target_returns = np.asarray(target_returns, dtype=np.float64)

    target_pmf = _to_pmf(target_returns, bins)
    a_pmf = _to_pmf(returns_a, bins)
    b_pmf = _to_pmf(returns_b, bins)

    kl_a = _kl_divergence(target_pmf, a_pmf)
    kl_b = _kl_divergence(target_pmf, b_pmf)
    delta_g = kl_a - kl_b

    winner = name_a if delta_g < 0 else name_b
    max_kl = max(abs(kl_a), abs(kl_b))
    reduction_pct = (1.0 - min(abs(kl_a), abs(kl_b)) / max_kl) * 100.0 if max_kl > 0 else 0.0

    return DivergenceBitsResult(
        strategy_a_name=name_a,
        strategy_b_name=name_b,
        delta_g_bits=round(delta_g, 4),
        kl_a=round(kl_a, 4),
        kl_b=round(kl_b, 4),
        target_entropy=round(float(stats.entropy(target_pmf, base=2)), 4),
        a_entropy=round(float(stats.entropy(a_pmf, base=2)), 4),
        b_entropy=round(float(stats.entropy(b_pmf, base=2)), 4),
        winner=winner,
        reduction_pct=round(reduction_pct, 2),
    )


def compare_vs_benchmark(
    returns_a: np.ndarray,
    benchmark_returns: np.ndarray,
    name_a: str = "Strategy",
    name_benchmark: str = "Benchmark",
    bins: int = 50,
) -> DivergenceBitsResult:
    """Compare a strategy against a benchmark using divergence-in-bits.

    Uses the benchmark as the target distribution. Δg < 0 means the strategy
    is better (more information captured than the benchmark itself).
    """
    return compute_divergence_bits(
        returns_a=returns_a,
        returns_b=benchmark_returns,
        target_returns=benchmark_returns,
        bins=bins,
        name_a=name_a,
        name_b=name_benchmark,
    )


def compare_multiple(
    strategies: dict[str, np.ndarray],
    target_returns: np.ndarray | None = None,
    bins: int = 50,
) -> list[DivergenceBitsResult]:
    """Pairwise comparison of multiple strategies.

    Args:
        strategies: Dict mapping strategy name → return series.
        target_returns: Optional target distribution.

    Returns:
        List of pairwise DivergenceBitsResult, sorted by Δg (best first).
    """
    names = list(strategies.keys())
    results = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            result = compute_divergence_bits(
                returns_a=strategies[names[i]],
                returns_b=strategies[names[j]],
                target_returns=target_returns,
                bins=bins,
                name_a=names[i],
                name_b=names[j],
            )
            results.append(result)
    results.sort(key=lambda r: r.delta_g_bits)
    return results
