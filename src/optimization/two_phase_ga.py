"""P24-29: Two-phase GA rule combination.

Phase 1: GA optimizes individual rule parameters (threshold, lookback, etc.).
Phase 2: GA optimizes weighted voting weights combining surviving rules.

Reference: B5 in Master Comparison Report — "GA-based Rule Combination for
Ensemble Pattern Trading Systems".
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from src.optimization.nsga2_optimizer import NSGA2Optimizer, NSGA2Config, NSGA2Result, ParamDef

logger = logging.getLogger(__name__)

MIN_SURVIVAL_SHARPE = 0.0


@dataclass
class RuleDef:
    """Definition of a single trading rule to be optimized."""

    name: str
    params: list[ParamDef]
    """Per-rule parameter definitions (thresholds, lookbacks, etc.)."""

    eval_fn: Callable[[dict[str, Any]], float]
    """Evaluates a rule with given params, returns a scalar fitness (e.g., Sharpe).

    Signature: eval_fn(params: dict) -> float

    Higher is better.
    """

    is_enabled: bool = True


@dataclass
class Phase1Result:
    """Results from Phase 1: per-rule GA optimization."""

    rule_name: str
    best_params: dict[str, Any]
    best_fitness: float
    passed: bool
    generation_history: list[dict[str, Any]]


@dataclass
class TwoPhaseGAResult:
    """Complete Two-Phase GA optimization result."""

    phase1_results: list[Phase1Result]
    """Per-rule optimization results."""

    surviving_rules: list[str]
    """Rule names that passed Phase 1 threshold."""

    rule_weights: dict[str, float]
    """Phase 2 optimized weight per surviving rule."""

    phase1_best_fitness: float
    """Best fitness from Phase 1."""

    phase2_best_fitness: float
    """Best fitness from Phase 2 (combined rules)."""

    fitness_improvement_pct: float
    """Percent improvement from Phase 2 over best Phase 1 rule."""

    def summary(self) -> str:
        lines = [
            f"Two-Phase GA: {len(self.surviving_rules)}/{len(self.phase1_results)} rules survived",
            f"  Phase 1 best fitness: {self.phase1_best_fitness:.4f}",
            f"  Phase 2 best fitness: {self.phase2_best_fitness:.4f}",
            f"  Improvement: {self.fitness_improvement_pct:.1f}%",
            f"  Rule weights: {self.rule_weights}",
        ]
        return "\n".join(lines)


class TwoPhaseGA:
    """Two-phase genetic algorithm for rule parameter + combination optimization.

    Phase 1 — Per-Rule Parameter Optimization:
        For each rule, run NSGA-II (or simple GA) to find the best parameter
        values for that rule's detector (threshold, lookback, etc.).

    Phase 2 — Weighted Voting Combination:
        Take surviving rules from Phase 1 and optimize the combination weights
        using NSGA-II on the ensemble fitness.

    Usage:
        rules = [
            RuleDef(
                name="double_bottom",
                params=[ParamDef("lookback", 5, 50, is_integer=True)],
                eval_fn=lambda p: backtest_rule("double_bottom", **p),
            ),
            ...
        ]
        two_phase = TwoPhaseGA(rules, phase1_config=NSGA2Config(population_size=20, generations=10))
        result = two_phase.optimize()
    """

    def __init__(
        self,
        rules: list[RuleDef],
        phase1_config: NSGA2Config | None = None,
        phase2_config: NSGA2Config | None = None,
        survival_threshold: float = MIN_SURVIVAL_SHARPE,
        max_surviving_rules: int = 10,
    ) -> None:
        self.rules = [r for r in rules if r.is_enabled]
        self.survival_threshold = survival_threshold
        self.max_surviving_rules = max_surviving_rules

        self.p1_config = phase1_config or self._default_phase_config()
        self.p2_config = phase2_config or NSGA2Config(
            population_size=40,
            generations=20,
            crossover_prob=0.9,
            mutation_prob=0.15,
            tournament_size=3,
        )

    @staticmethod
    def _default_phase_config() -> NSGA2Config:
        return NSGA2Config(
            population_size=20,
            generations=15,
            crossover_prob=0.9,
            mutation_prob=0.1,
            tournament_size=2,
        )

    def _run_phase1_single_rule(self, rule: RuleDef) -> Phase1Result:
        """Run GA for a single rule's parameters."""
        config = deepcopy(self.p1_config)
        config.objectives_minimize = [True]  # single objective: minimize negative fitness

        def _wrap_eval(params: dict[str, Any]) -> list[float]:
            fitness = float(rule.eval_fn(params))
            return [-fitness]

        if not rule.params:
            fitness = float(rule.eval_fn({}))
            return Phase1Result(
                rule_name=rule.name,
                best_params={},
                best_fitness=fitness,
                passed=fitness > self.survival_threshold,
                generation_history=[],
            )

        nsga = NSGA2Optimizer(rule.params, _wrap_eval, config)
        result = nsga.optimize()

        if not result.best_params:
            return Phase1Result(
                rule_name=rule.name,
                best_params={},
                best_fitness=float("-inf"),
                passed=False,
                generation_history=[],
            )

        best_p = result.best_params[0]
        best_fitness_val = abs(float(np.min(result.best_objectives[:, 0])))

        return Phase1Result(
            rule_name=rule.name,
            best_params=best_p,
            best_fitness=best_fitness_val,
            passed=best_fitness_val > self.survival_threshold,
            generation_history=result.generation_history,
        )

    def _run_phase1(self) -> list[Phase1Result]:
        """Run Phase 1 for all rules."""
        results = []
        for i, rule in enumerate(self.rules):
            logger.info(
                "Phase 1 [%d/%d]: optimizing rule '%s' (%d params)",
                i + 1,
                len(self.rules),
                rule.name,
                len(rule.params),
            )
            result = self._run_phase1_single_rule(rule)
            results.append(result)
            logger.debug(
                "  %s: fitness=%.4f, passed=%s", rule.name, result.best_fitness, result.passed
            )
        return results

    def _run_phase2(
        self,
        phase1_results: list[Phase1Result],
        combined_eval_fn: Callable[[dict[str, float]], float],
    ) -> tuple[dict[str, float], float]:
        """Run Phase 2: optimize combination weights for surviving rules.

        Args:
            phase1_results: Phase 1 results with per-rule fitness.
            combined_eval_fn: Evaluates a set of rule weights, returns scalar fitness.

        Returns:
            Tuple of (optimized weights dict, best fitness).
        """
        survivors = [
            (r.rule_name, r.best_fitness, r.best_params) for r in phase1_results if r.passed
        ]

        survivors.sort(key=lambda x: x[1], reverse=True)
        survivors = survivors[: self.max_surviving_rules]

        if not survivors:
            logger.warning("No rules survived Phase 1. Skipping Phase 2.")
            return {}, 0.0

        if len(survivors) == 1:
            name = survivors[0][0]
            return {name: 1.0}, survivors[0][1]

        weight_params = []
        for name, _fitness, _params in survivors:
            weight_params.append(ParamDef(f"w_{name}", 0.0, 1.0))

        config = deepcopy(self.p2_config)
        config.objectives_minimize = [True]

        def _wrap_combined_eval(params: dict[str, Any]) -> list[float]:
            weights = {name: params.get(f"w_{name}", 0.0) for name, _, _ in survivors}
            total = sum(weights.values()) or 1.0
            normalized = {k: v / total for k, v in weights.items()}
            fitness = float(combined_eval_fn(normalized))
            return [-fitness]

        nsga = NSGA2Optimizer(weight_params, _wrap_combined_eval, config)
        result = nsga.optimize()

        if not result.best_params:
            equal_w = 1.0 / len(survivors)
            return {name: equal_w for name, _, _ in survivors}, survivors[0][1]

        best_p = result.best_params[0]
        raw_weights = {name: best_p.get(f"w_{name}", 0.0) for name, _, _ in survivors}
        total = sum(raw_weights.values()) or 1.0
        normalized = {k: round(v / total, 4) for k, v in raw_weights.items()}
        best_fitness_val = abs(float(np.min(result.best_objectives[:, 0])))

        return normalized, best_fitness_val

    def optimize(
        self,
        combined_eval_fn: Callable[[dict[str, float]], float] | None = None,
    ) -> TwoPhaseGAResult:
        """Run full two-phase GA optimization.

        Args:
            combined_eval_fn: Function to evaluate combined rules via weighted voting.
                Signature: eval_fn(weights: dict[str, float]) -> float.
                If None, uses a simple weighted-average baseline (sum of per-rule
                fitnesses weighted by their Phase 2 weights).

        Returns:
            TwoPhaseGAResult with optimized per-rule params and combination weights.
        """
        p1_results = self._run_phase1()

        if combined_eval_fn is None:
            combined_eval_fn = self._make_default_combined_eval(p1_results)

        rule_weights, p2_fitness = self._run_phase2(p1_results, combined_eval_fn)

        p1_best = max((r.best_fitness for r in p1_results), default=0.0)
        improvement = (p2_fitness - p1_best) / abs(p1_best) * 100.0 if abs(p1_best) > 0 else 0.0

        return TwoPhaseGAResult(
            phase1_results=p1_results,
            surviving_rules=list(rule_weights.keys()),
            rule_weights=rule_weights,
            phase1_best_fitness=p1_best,
            phase2_best_fitness=p2_fitness,
            fitness_improvement_pct=round(improvement, 2),
        )

    @staticmethod
    def _make_default_combined_eval(
        p1_results: list[Phase1Result],
    ) -> Callable[[dict[str, float]], float]:
        """Create a default combined evaluation using per-rule stored fitness."""
        rule_fitness = {r.rule_name: max(r.best_fitness, 0.0) for r in p1_results}

        def _eval(weights: dict[str, float]) -> float:
            total = 0.0
            for name, w in weights.items():
                total += w * rule_fitness.get(name, 0.0)
            return total

        return _eval
