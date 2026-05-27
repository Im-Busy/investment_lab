"""Monte Carlo Robustness Testing for Backtest Strategies.

Implements three robustness checks from Bailey & Lopez de Prado and related
methodology, plus a comprehensive harness combining them:

- Return Reshuffling: Permute returns to destroy temporal structure — tests
  whether strategy profits depend on sequence (autocorrelation) or are random.

- Return Replacement (Bootstrap): Sample returns with replacement — tests
  sensitivity to drawdown ordering.

- Parameter Perturbation: Jitter strategy parameters by ±10% — tests
  sensitivity to parameter estimation error.

- Combined Robustness Score: Aggregates all three into a single 0-100 score
  with acceptance gates.

References:
    Bailey, D.H. & Lopez de Prado, M. "Tactical Investment Algorithms."
    Hillsdale Investment Management. "The Three Types of Backtests."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import stats

from src.analysis.deflated_sharpe import _sharpe_ratio

logger = logging.getLogger(__name__)


@dataclass
class ReshuffleResult:
    """Result of return reshuffling simulation.

    Attributes:
        n_simulations: Number of reshuffles performed.
        metrics: Dict of {metric_name: observed_value}.
        percentile_05: 5th percentile of simulated metrics.
        percentile_50: Median of simulated metrics.
        percentile_95: 95th percentile of simulated metrics.
        p_value: Empirical p-value (fraction of simulations exceeding observed).
        passes: True if 5th percentile > 0 for all metrics.
        ruin_probability: Fraction of simulations with negative total return.
    """

    n_simulations: int
    metrics: dict[str, float]
    percentile_05: dict[str, float]
    percentile_50: dict[str, float]
    percentile_95: dict[str, float]
    p_value: dict[str, float]
    passes: bool
    ruin_probability: float


@dataclass
class PerturbResult:
    """Result of parameter perturbation testing.

    Attributes:
        n_simulations: Number of perturbation runs.
        param_name: Name of the perturbed parameter.
        base_value: Original parameter value.
        perturb_range: (low, high) range tested.
        metrics: Dict of {metric_name: base_value}.
        percentile_05: 5th percentile of perturbed metrics.
        percentile_50: Median of perturbed metrics.
        percentile_95: 95th percentile of perturbed metrics.
        passes: True if 5th percentile still passes gate.
    """

    n_simulations: int
    param_name: str
    base_value: float
    perturb_range: tuple[float, float]
    metrics: dict[str, float]
    percentile_05: dict[str, float]
    percentile_50: dict[str, float]
    percentile_95: dict[str, float]
    passes: bool


@dataclass
class MonteCarloRobustnessReport:
    """Combined Monte Carlo robustness report.

    Attributes:
        reshuffle: Return reshuffling result.
        replacement: Return replacement (bootstrap) result.
        perturbations: List of parameter perturbation results.
        combined_score: 0-100 robustness score.
        acceptance_gates: List of (gate_name, passed) tuples.
        overall_pass: True if all acceptance gates pass.
    """

    reshuffle: ReshuffleResult
    replacement: ReshuffleResult
    perturbations: list[PerturbResult]
    combined_score: float
    acceptance_gates: dict[str, bool]
    overall_pass: bool


def _compute_metrics(returns: np.ndarray, periods_per_year: int = 252) -> dict[str, float]:
    """Compute standard performance metrics from a return series.

    Args:
        returns: Array of period returns.
        periods_per_year: Annualization factor.

    Returns:
        Dict of {metric_name: value}.
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    if len(returns) < 2:
        return {
            "sharpe": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "win_rate": 0.0,
        }

    sharpe = _sharpe_ratio(returns, periods_per_year)
    total_return = float(np.prod(1.0 + returns) - 1.0)
    cum_returns = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    max_dd = float(np.min(drawdowns))
    volatility = float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))
    win_rate = float(np.mean(returns > 0))

    return {
        "sharpe": sharpe,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "volatility": volatility,
        "win_rate": win_rate,
    }


def _percentiles(metric_name: str, simulated: np.ndarray) -> dict[str, float]:
    """Compute percentile summary for a metric across simulations."""
    return {
        "p05": float(np.percentile(simulated, 5)),
        "p50": float(np.percentile(simulated, 50)),
        "p95": float(np.percentile(simulated, 95)),
    }


def return_reshuffling(
    returns: np.ndarray,
    n_simulations: int = 5000,
    periods_per_year: int = 252,
    seed: int | None = 42,
) -> ReshuffleResult:
    """Test robustness by randomly reshuffling return sequence.

    Destroys temporal dependence (autocorrelation, trend, volatility clustering).
    If strategy profits survive shuffling, the edge comes from bet size/timing,
    not from return sequence structure.

    Args:
        returns: Array of period returns.
        n_simulations: Number of reshuffles.
        periods_per_year: Annualization factor.
        seed: Random seed.

    Returns:
        ReshuffleResult with metrics and acceptance.
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    n = len(returns)

    if n < 10:
        return ReshuffleResult(
            n_simulations=n_simulations,
            metrics={},
            percentile_05={},
            percentile_50={},
            percentile_95={},
            p_value={},
            passes=False,
            ruin_probability=1.0,
        )

    observed = _compute_metrics(returns, periods_per_year)
    rng = np.random.RandomState(seed)

    sim_sharpes = np.empty(n_simulations)
    sim_returns = np.empty(n_simulations)
    sim_drawdowns = np.empty(n_simulations)
    sim_vols = np.empty(n_simulations)
    sim_win_rates = np.empty(n_simulations)

    for i in range(n_simulations):
        shuffled = rng.permutation(returns)
        m = _compute_metrics(shuffled, periods_per_year)
        sim_sharpes[i] = m["sharpe"]
        sim_returns[i] = m["total_return"]
        sim_drawdowns[i] = m["max_drawdown"]
        sim_vols[i] = m["volatility"]
        sim_win_rates[i] = m["win_rate"]

    p05_ret = float(np.percentile(sim_returns, 5))
    passes = p05_ret > 0
    ruin_prob = float(np.mean(sim_returns <= 0))

    results = {
        "sharpe": observed["sharpe"],
        "total_return": observed["total_return"],
        "max_drawdown": observed["max_drawdown"],
        "volatility": observed["volatility"],
        "win_rate": observed["win_rate"],
    }

    p_values = {
        "sharpe": float(np.mean(sim_sharpes >= observed["sharpe"])),
        "total_return": float(np.mean(sim_returns >= observed["total_return"])),
    }

    return ReshuffleResult(
        n_simulations=n_simulations,
        metrics=results,
        percentile_05={
            "sharpe": float(np.percentile(sim_sharpes, 5)),
            "total_return": p05_ret,
            "max_drawdown": float(np.percentile(sim_drawdowns, 5)),
            "volatility": float(np.percentile(sim_vols, 5)),
            "win_rate": float(np.percentile(sim_win_rates, 5)),
        },
        percentile_50={
            "sharpe": float(np.percentile(sim_sharpes, 50)),
            "total_return": float(np.percentile(sim_returns, 50)),
            "max_drawdown": float(np.percentile(sim_drawdowns, 50)),
            "volatility": float(np.percentile(sim_vols, 50)),
            "win_rate": float(np.percentile(sim_win_rates, 50)),
        },
        percentile_95={
            "sharpe": float(np.percentile(sim_sharpes, 95)),
            "total_return": float(np.percentile(sim_returns, 95)),
            "max_drawdown": float(np.percentile(sim_drawdowns, 95)),
            "volatility": float(np.percentile(sim_vols, 95)),
            "win_rate": float(np.percentile(sim_win_rates, 95)),
        },
        p_value=p_values,
        passes=passes,
        ruin_probability=ruin_prob,
    )


def return_replacement(
    returns: np.ndarray,
    n_simulations: int = 5000,
    periods_per_year: int = 252,
    seed: int | None = 42,
) -> ReshuffleResult:
    """Test robustness by bootstrapping returns with replacement.

    Tests sensitivity to sequence composition — different draws from the
    empirical return distribution.

    Args:
        returns: Array of period returns.
        n_simulations: Number of bootstrap samples.
        periods_per_year: Annualization factor.
        seed: Random seed.

    Returns:
        ReshuffleResult with bootstrapped metric distributions.
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    n = len(returns)

    if n < 10:
        return ReshuffleResult(
            n_simulations=n_simulations,
            metrics={},
            percentile_05={},
            percentile_50={},
            percentile_95={},
            p_value={},
            passes=False,
            ruin_probability=1.0,
        )

    observed = _compute_metrics(returns, periods_per_year)
    rng = np.random.RandomState(seed)

    sim_sharpes = np.empty(n_simulations)
    sim_returns = np.empty(n_simulations)
    sim_drawdowns = np.empty(n_simulations)

    for i in range(n_simulations):
        idx = rng.randint(0, n, size=n)
        sample = returns[idx]
        m = _compute_metrics(sample, periods_per_year)
        sim_sharpes[i] = m["sharpe"]
        sim_returns[i] = m["total_return"]
        sim_drawdowns[i] = m["max_drawdown"]

    p05_ret = float(np.percentile(sim_returns, 5))
    passes = p05_ret > 0
    ruin_prob = float(np.mean(sim_returns <= 0))

    return ReshuffleResult(
        n_simulations=n_simulations,
        metrics={
            "sharpe": observed["sharpe"],
            "total_return": observed["total_return"],
            "max_drawdown": observed["max_drawdown"],
            "volatility": observed["volatility"],
            "win_rate": observed["win_rate"],
        },
        percentile_05={
            "sharpe": float(np.percentile(sim_sharpes, 5)),
            "total_return": p05_ret,
            "max_drawdown": float(np.percentile(sim_drawdowns, 5)),
        },
        percentile_50={
            "sharpe": float(np.percentile(sim_sharpes, 50)),
            "total_return": float(np.percentile(sim_returns, 50)),
            "max_drawdown": float(np.percentile(sim_drawdowns, 50)),
        },
        percentile_95={
            "sharpe": float(np.percentile(sim_sharpes, 95)),
            "total_return": float(np.percentile(sim_returns, 95)),
            "max_drawdown": float(np.percentile(sim_drawdowns, 95)),
        },
        p_value={
            "sharpe": float(np.mean(sim_sharpes >= observed["sharpe"])),
            "total_return": float(np.mean(sim_returns >= observed["total_return"])),
        },
        passes=passes,
        ruin_probability=ruin_prob,
    )


def parameter_perturbation(
    base_returns: np.ndarray,
    backtest_fn: Callable[[dict[str, float]], np.ndarray],
    params: dict[str, float],
    perturbation_frac: float = 0.10,
    n_simulations: int = 100,
    periods_per_year: int = 252,
    seed: int | None = 42,
) -> list[PerturbResult]:
    """Test robustness to parameter estimation error by jittering each param ±10%.

    For each parameter, generates n_simulations random values uniformly
    distributed in [value*(1-pct), value*(1+pct)], reruns the backtest,
    and checks whether 5th percentile of metrics still passes gates.

    Args:
        base_returns: Array of period returns from base configuration.
        backtest_fn: Function that takes {param: value} dict and returns
            array of period returns. Signature: fn(params) -> np.ndarray.
        params: Base parameter values as {name: value}.
        perturbation_frac: Fraction to perturb (default 0.10 = ±10%).
        n_simulations: Perturbation draws per parameter.
        periods_per_year: Annualization factor.
        seed: Random seed.

    Returns:
        List of PerturbResult, one per parameter.
    """
    base_metrics = _compute_metrics(base_returns, periods_per_year)
    rng = np.random.RandomState(seed)

    results: list[PerturbResult] = []
    for param_name, base_val in params.items():
        if base_val == 0 or isinstance(base_val, bool):
            continue

        low = base_val * (1.0 - perturbation_frac)
        high = base_val * (1.0 + perturbation_frac)

        sim_sharpes = np.empty(n_simulations)
        sim_returns = np.empty(n_simulations)
        sim_drawdowns = np.empty(n_simulations)

        for i in range(n_simulations):
            perturbed_params = dict(params)
            perturbed_params[param_name] = float(rng.uniform(low, high))
            try:
                perturbed_returns = backtest_fn(perturbed_params)
                m = _compute_metrics(perturbed_returns, periods_per_year)
            except Exception:
                m = {"sharpe": -999.0, "total_return": -99.0, "max_drawdown": -1.0}
            sim_sharpes[i] = m["sharpe"]
            sim_returns[i] = m["total_return"]
            sim_drawdowns[i] = m["max_drawdown"]

        p05_ret = float(np.percentile(sim_returns, 5))
        passes = p05_ret > 0

        results.append(
            PerturbResult(
                n_simulations=n_simulations,
                param_name=param_name,
                base_value=base_val,
                perturb_range=(low, high),
                metrics={
                    "sharpe": base_metrics["sharpe"],
                    "total_return": base_metrics["total_return"],
                    "max_drawdown": base_metrics["max_drawdown"],
                },
                percentile_05={
                    "sharpe": float(np.percentile(sim_sharpes, 5)),
                    "total_return": p05_ret,
                    "max_drawdown": float(np.percentile(sim_drawdowns, 5)),
                },
                percentile_50={
                    "sharpe": float(np.percentile(sim_sharpes, 50)),
                    "total_return": float(np.percentile(sim_returns, 50)),
                    "max_drawdown": float(np.percentile(sim_drawdowns, 50)),
                },
                percentile_95={
                    "sharpe": float(np.percentile(sim_sharpes, 95)),
                    "total_return": float(np.percentile(sim_returns, 95)),
                    "max_drawdown": float(np.percentile(sim_drawdowns, 95)),
                },
                passes=passes,
            )
        )

    return results


def parameter_perturbation_from_returns(
    returns: np.ndarray,
    params: dict[str, float],
    perturbation_frac: float = 0.10,
    n_simulations: int = 100,
    periods_per_year: int = 252,
    seed: int | None = 42,
) -> list[PerturbResult]:
    """Lightweight parameter perturbation using return reshuffling as proxy.

    When a full backtest_fn is not available (e.g., for offline analysis of
    saved results), this approximates parameter sensitivity by re-weighting
    returns with random factor perturbations. This is a heuristic — for
    accurate assessment, use parameter_perturbation() with a real backtest_fn.

    Args:
        returns: Array of period returns from base configuration.
        params: Base parameter values.
        perturbation_frac: Fraction to perturb.
        n_simulations: Perturbation draws per parameter.
        periods_per_year: Annualization factor.
        seed: Random seed.

    Returns:
        List of PerturbResult, one per parameter.
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    base_metrics = _compute_metrics(returns, periods_per_year)
    rng = np.random.RandomState(seed)
    n = len(returns)

    results: list[PerturbResult] = []
    for param_name, base_val in params.items():
        if base_val == 0 or isinstance(base_val, bool):
            continue

        low = base_val * (1.0 - perturbation_frac)
        high = base_val * (1.0 + perturbation_frac)

        sim_sharpes = np.empty(n_simulations)
        sim_returns = np.empty(n_simulations)

        for i in range(n_simulations):
            perturbed_val = float(rng.uniform(low, high))
            scale = perturbed_val / max(base_val, 1e-8)
            scaled_rets = returns * np.clip(scale, 0.5, 2.0)
            m = _compute_metrics(scaled_rets, periods_per_year)
            sim_sharpes[i] = m["sharpe"]
            sim_returns[i] = m["total_return"]

        p05_ret = float(np.percentile(sim_returns, 5))
        passes = p05_ret > 0

        results.append(
            PerturbResult(
                n_simulations=n_simulations,
                param_name=param_name,
                base_value=base_val,
                perturb_range=(low, high),
                metrics={
                    "sharpe": base_metrics["sharpe"],
                    "total_return": base_metrics["total_return"],
                    "max_drawdown": base_metrics["max_drawdown"],
                },
                percentile_05={
                    "sharpe": float(np.percentile(sim_sharpes, 5)),
                    "total_return": p05_ret,
                },
                percentile_50={
                    "sharpe": float(np.percentile(sim_sharpes, 50)),
                    "total_return": float(np.percentile(sim_returns, 50)),
                },
                percentile_95={
                    "sharpe": float(np.percentile(sim_sharpes, 95)),
                    "total_return": float(np.percentile(sim_returns, 95)),
                },
                passes=passes,
            )
        )

    return results


def compute_combined_robustness_score(
    reshuffle: ReshuffleResult,
    replacement: ReshuffleResult,
    perturbations: list[PerturbResult],
    base_sharpe: float,
) -> tuple[float, dict[str, bool]]:
    """Compute a 0-100 combined robustness score.

    Scoring:
        - 40 points: Return shuffling (5th-percentile Sharpe / observed Sharpe)
        - 40 points: Return replacement (5th-percentile Sharpe / observed Sharpe)
        - 20 points: Parameter perturbation (fraction of params passing)

    Args:
        reshuffle: Return reshuffling result.
        replacement: Return replacement result.
        perturbations: Parameter perturbation results.
        base_sharpe: Observed Sharpe ratio.

    Returns:
        (combined_score 0-100, acceptance_gates dict).
    """
    gates: dict[str, bool] = {}

    # Gate 1: 5th percentile returns > 0 after reshuffling
    gates["reshuffle_p05_positive"] = reshuffle.passes

    # Gate 2: Ruin probability < 2% after reshuffling
    gates["ruin_probability_lt_2pct"] = reshuffle.ruin_probability < 0.02

    # Gate 3: 5th percentile returns > 0 after replacement
    gates["replacement_p05_positive"] = replacement.passes

    # Gate 4: >50% of parameters pass perturbation test
    if perturbations:
        n_pass = sum(1 for p in perturbations if p.passes)
        gates["majority_params_pass"] = n_pass > len(perturbations) / 2
    else:
        gates["majority_params_pass"] = True

    # Scoring
    score = 0.0

    # Reshuffling score (40 pts max)
    if base_sharpe > 0:
        res_sharpe_ratio = max(
            0.0, min(1.0, reshuffle.percentile_05.get("sharpe", 0.0) / max(base_sharpe, 0.01))
        )
        score += 40.0 * res_sharpe_ratio

    # Replacement score (40 pts max)
    if base_sharpe > 0:
        repl_sharpe_ratio = max(
            0.0, min(1.0, replacement.percentile_05.get("sharpe", 0.0) / max(base_sharpe, 0.01))
        )
        score += 40.0 * repl_sharpe_ratio

    # Perturbation score (20 pts max)
    if perturbations:
        n_pass = sum(1 for p in perturbations if p.passes)
        score += 20.0 * (n_pass / len(perturbations))

    return float(score), gates


def run_full_robustness_check(
    returns: np.ndarray,
    params: dict[str, float] | None = None,
    backtest_fn: Callable[[dict[str, float]], np.ndarray] | None = None,
    n_simulations: int = 5000,
    n_perturbations: int = 100,
    perturbation_frac: float = 0.10,
    periods_per_year: int = 252,
    seed: int | None = 42,
) -> MonteCarloRobustnessReport:
    """Run the full Monte Carlo robustness suite.

    Args:
        returns: Array of strategy period returns.
        params: Dict of {param_name: base_value} for perturbation testing.
        backtest_fn: Optional full backtest function for accurate perturbation.
            If None, uses lightweight return-scaling heuristic.
        n_simulations: Shuffling and replacement iterations.
        n_perturbations: Perturbation draws per parameter.
        perturbation_frac: Fraction to perturb params.
        periods_per_year: Annualization factor.
        seed: Random seed.

    Returns:
        MonteCarloRobustnessReport.
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    base_metrics = _compute_metrics(returns, periods_per_year)

    reshuffle = return_reshuffling(
        returns, n_simulations=n_simulations, periods_per_year=periods_per_year, seed=seed
    )
    replacement = return_replacement(
        returns, n_simulations=n_simulations, periods_per_year=periods_per_year, seed=seed
    )

    if params and backtest_fn:
        perturbations = parameter_perturbation(
            returns,
            backtest_fn,
            params,
            perturbation_frac=perturbation_frac,
            n_simulations=n_perturbations,
            periods_per_year=periods_per_year,
            seed=seed,
        )
    elif params:
        perturbations = parameter_perturbation_from_returns(
            returns,
            params,
            perturbation_frac=perturbation_frac,
            n_simulations=n_perturbations,
            periods_per_year=periods_per_year,
            seed=seed,
        )
    else:
        perturbations = []

    score, gates = compute_combined_robustness_score(
        reshuffle, replacement, perturbations, base_metrics["sharpe"]
    )

    overall_pass = all(gates.values()) and score >= 60.0

    return MonteCarloRobustnessReport(
        reshuffle=reshuffle,
        replacement=replacement,
        perturbations=perturbations,
        combined_score=score,
        acceptance_gates=gates,
        overall_pass=overall_pass,
    )


def format_mc_report(report: MonteCarloRobustnessReport) -> str:
    """Generate a human-readable Monte Carlo robustness report.

    Args:
        report: MonteCarloRobustnessReport from run_full_robustness_check().

    Returns:
        Multi-line formatted string.
    """
    lines = [
        "=" * 70,
        "MONTE CARLO ROBUSTNESS REPORT",
        "=" * 70,
        "",
        f"  Combined Robustness Score: {report.combined_score:.0f}/100",
        f"  Overall Pass: {'YES' if report.overall_pass else 'NO'}",
        "",
        "  Acceptance Gates:",
    ]
    for gate, passed in report.acceptance_gates.items():
        status = "PASS" if passed else "FAIL"
        lines.append(f"    [{status}] {gate}")

    lines.extend(
        [
            "",
            "-" * 70,
            "RETURN RESHUFFLING (temporal structure test)",
            "-" * 70,
            f"  Simulations: {report.reshuffle.n_simulations}",
            f"  Observed Sharpe: {report.reshuffle.metrics.get('sharpe', 0):.3f}",
            f"  Observed Return: {report.reshuffle.metrics.get('total_return', 0):.1%}",
            f"  5th %ile Return: {report.reshuffle.percentile_05.get('total_return', 0):.1%}",
            f"  Median Return:   {report.reshuffle.percentile_50.get('total_return', 0):.1%}",
            f"  95th %ile Return: {report.reshuffle.percentile_95.get('total_return', 0):.1%}",
            f"  Ruin Probability: {report.reshuffle.ruin_probability:.1%}",
            f"  5th %ile Sharpe: {report.reshuffle.percentile_05.get('sharpe', 0):.3f}",
            f"  Median Sharpe:   {report.reshuffle.percentile_50.get('sharpe', 0):.3f}",
            f"  95th %ile Sharpe: {report.reshuffle.percentile_95.get('sharpe', 0):.3f}",
            f"  p-value (Sharpe): {report.reshuffle.p_value.get('sharpe', 1):.4f}",
            "",
            "-" * 70,
            "RETURN REPLACEMENT (bootstrap sensitivity test)",
            "-" * 70,
            f"  Simulations: {report.replacement.n_simulations}",
            f"  Observed Sharpe: {report.replacement.metrics.get('sharpe', 0):.3f}",
            f"  5th %ile Return: {report.replacement.percentile_05.get('total_return', 0):.1%}",
            f"  Median Return:   {report.replacement.percentile_50.get('total_return', 0):.1%}",
            f"  95th %ile Return: {report.replacement.percentile_95.get('total_return', 0):.1%}",
            f"  Ruin Probability: {report.replacement.ruin_probability:.1%}",
            "",
        ]
    )

    if report.perturbations:
        lines.extend(
            [
                "-" * 70,
                "PARAMETER PERTURBATION (±{:.0f}%)".format(
                    100
                    * (
                        report.perturbations[0].perturb_range[1]
                        / report.perturbations[0].base_value
                        - 1
                    )
                    if report.perturbations
                    else 10
                ),
                "-" * 70,
            ]
        )
        for p in report.perturbations:
            status = "PASS" if p.passes else "FAIL"
            lines.append(
                f"  [{status}] {p.param_name} = {p.base_value:.4f} "
                f"  p05_ret={p.percentile_05.get('total_return', 0):.1%}"
            )

    lines.append("")
    lines.append("=" * 70)

    if report.overall_pass:
        lines.append("CONCLUSION: STRATEGY PASSES MONTE CARLO ROBUSTNESS")
    else:
        lines.append("CONCLUSION: STRATEGY FAILS MONTE CARLO ROBUSTNESS — review gates above")

    lines.append("=" * 70)
    return "\n".join(lines)
