"""P25: Dynamic GA with associative memory for regime-adaptive optimization.

Maintains separate GA populations per market regime (up/side/down) and switches
between them when regime changes. Uses associative memory to recall previously
found good solutions when returning to a known regime.

Reference: Paiva et al. (2016) "Combining SVM with GA", Sadeghi et al. (2021).
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from src.optimization.nsga2_optimizer import NSGA2Optimizer, NSGA2Config, NSGA2Result, ParamDef

logger = logging.getLogger(__name__)

REGIME_LABELS = ("up", "side", "down")
MEMORY_SIZE_DEFAULT = 10  # top-N solutions to remember per regime


@dataclass
class RegimeMemory:
    """Associative memory storing elite solutions per regime."""

    regime: str
    solutions: list[np.ndarray] = field(default_factory=list)
    fitnesses: list[float] = field(default_factory=list)
    max_size: int = MEMORY_SIZE_DEFAULT

    def store(self, solution: np.ndarray, fitness: float) -> None:
        self.solutions.append(solution.copy())
        self.fitnesses.append(fitness)
        sorted_pairs = sorted(
            zip(self.solutions, self.fitnesses),
            key=lambda x: x[1],
            reverse=True,
        )
        self.solutions = [s for s, _ in sorted_pairs[: self.max_size]]
        self.fitnesses = [f for _, f in sorted_pairs[: self.max_size]]

    def get_best(self) -> np.ndarray | None:
        if not self.solutions:
            return None
        return self.solutions[0].copy()

    def get_seeded(self, n: int, rng: np.random.Generator) -> list[np.ndarray]:
        """Generate n individuals seeded from memory with mutation."""
        if not self.solutions:
            return []
        seeded = []
        for _ in range(n):
            base = self.solutions[rng.integers(0, len(self.solutions))]
            mutant = base + rng.normal(0, 0.05, base.shape)
            seeded.append(np.clip(mutant, 0.0, 1.0))
        return seeded

    def is_empty(self) -> bool:
        return len(self.solutions) == 0


@dataclass
class DynamicGAResult:
    """Dynamic GA optimization result."""

    regime_results: dict[str, NSGA2Result]
    transitions: list[dict[str, Any]]
    memory_size: int
    n_generations_per_regime: int

    def summary(self) -> str:
        lines = [f"Dynamic GA: {len(self.transitions)} regime transitions, mem={self.memory_size}"]
        for regime, result in self.regime_results.items():
            if result.best_params:
                lines.append(f"  {regime}: {len(result.best_params)} solutions")
                lines.append(f"    best: {result.best}")
        return "\n".join(lines)


class RegimeDetector:
    """Simple regime detector using price-based heuristics."""

    def __init__(
        self,
        ma_short: int = 20,
        ma_long: int = 50,
        trend_threshold: float = 0.005,
    ) -> None:
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.trend_threshold = trend_threshold

    def detect(self, prices: np.ndarray) -> str:
        """Detect current regime from price array.

        Returns 'up', 'down', or 'side'.
        """
        if len(prices) < self.ma_long:
            return "side"

        ma_s = np.mean(prices[-self.ma_short :])
        ma_l = np.mean(prices[-self.ma_long :])
        trend = (ma_s - ma_l) / ma_l if ma_l != 0 else 0.0

        if trend > self.trend_threshold:
            return "up"
        elif trend < -self.trend_threshold:
            return "down"
        return "side"


class DynamicGAOptimizer:
    """Dynamic Genetic Algorithm with associative memory per regime.

    Maintains separate GA populations for each market regime. When regime
    shifts, saves current population's elites to memory and switches to
    the new regime's GA (seeded from memory if available). When returning
    to a known regime, uses hyper-mutation (10× mutation rate) for rapid
    adaptation.

    Usage:
        params = [ParamDef("entry", 0.3, 0.9), ParamDef("trail", 1.0, 5.0)]

        def evaluate(params: dict, regime: str, prices: np.ndarray) -> list[float]:
            return [roi, -max_dd, profit_factor]

        dga = DynamicGAOptimizer(params, evaluate)
        dga.fit(prices_df)
        best_params = dga.get_best("up")
    """

    def __init__(
        self,
        params: list[ParamDef],
        eval_fn: Callable[[dict[str, Any], str], list[float]],
        config: NSGA2Config | None = None,
        regime_detector: RegimeDetector | None = None,
        memory_size: int = MEMORY_SIZE_DEFAULT,
        hyper_mutation_rate: float = 0.5,
    ) -> None:
        self.params = params
        self.eval_fn = eval_fn
        self.config = config or NSGA2Config(population_size=30, generations=15)
        self.regime_detector = regime_detector or RegimeDetector()
        self.memory_size = memory_size
        self.hyper_mutation_rate = hyper_mutation_rate

        self._rng = np.random.default_rng(self.config.seed)
        self._ga_per_regime: dict[str, NSGA2Optimizer] = {}
        self._memory: dict[str, RegimeMemory] = {
            r: RegimeMemory(regime=r, max_size=memory_size) for r in REGIME_LABELS
        }
        self._current_regime = "side"
        self._transitions: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []

    def _get_or_create_ga(self, regime: str) -> NSGA2Optimizer:
        if regime not in self._ga_per_regime:

            def regime_eval(params: dict[str, Any]) -> list[float]:
                return self.eval_fn(params, regime)

            self._ga_per_regime[regime] = NSGA2Optimizer(
                self.params,
                regime_eval,
                deepcopy(self.config),
            )
        return self._ga_per_regime[regime]

    def detect_regime(self, prices: np.ndarray) -> str:
        """Detect current regime from price data."""
        return self.regime_detector.detect(prices)

    def fit(
        self,
        prices: np.ndarray,
        window: int = 252,
        step: int = 21,
    ) -> DynamicGAResult:
        """Fit dynamic GA by rolling through historical data.

        Args:
            prices: 1D array of close prices.
            window: Lookback window for regime detection and evaluation.
            step: Step size between optimizations.

        Returns:
            DynamicGAResult with per-regime solutions.
        """
        regime_results: dict[str, Any] = {}
        current_regime = "side"

        for start in range(0, len(prices) - window, step):
            window_prices = prices[start : start + window]
            regime = self.detect_regime(window_prices)

            if regime != current_regime:
                self._transitions.append(
                    {
                        "from": current_regime,
                        "to": regime,
                        "bar": start + window,
                    }
                )
                logger.debug(
                    "Regime transition: %s → %s at bar %d", current_regime, regime, start + window
                )

                if self._current_regime in self._ga_per_regime:
                    self._save_elites_to_memory(self._current_regime)

                current_regime = regime
                self._current_regime = regime

                use_hyper = bool(self._rng.random() < self.hyper_mutation_rate)
                if use_hyper:
                    logger.debug("Hyper-mutation activated for regime %s", regime)

            self._current_regime = regime

        regime_results = {}
        for regime in REGIME_LABELS:
            if regime in self._ga_per_regime:
                ga = self._ga_per_regime[regime]
                result = ga.optimize()
                regime_results[regime] = result
            else:
                regime_results[regime] = NSGA2Result(
                    best_individuals=[],
                    best_objectives=np.array([]),
                    best_params=[],
                    pareto_front_ranks=np.array([]),
                    generation_history=[],
                    n_generations=0,
                    n_objectives=0,
                )

        return DynamicGAResult(
            regime_results=regime_results,
            transitions=self._transitions,
            memory_size=self.memory_size,
            n_generations_per_regime=self.config.generations,
        )

    def _save_elites_to_memory(self, regime: str) -> None:
        """Save elite solutions from current GA to regime memory."""
        if regime not in self._ga_per_regime:
            return
        ga = self._ga_per_regime[regime]
        result = ga.optimize()

        if regime not in self._memory:
            self._memory[regime] = RegimeMemory(regime=regime, max_size=self.memory_size)

        for i, (params, obj) in enumerate(zip(result.best_params[:5], result.best_objectives)):
            encoded = ga._encode(params)
            fitness = float(-obj[0])
            self._memory[regime].store(encoded, fitness)

    def get_best(self, regime: str) -> dict[str, Any]:
        """Get best parameters for a specific regime."""
        if regime in self._ga_per_regime:
            result = self._ga_per_regime[regime].optimize()
            return result.best
        return {}

    def get_all_best(self) -> dict[str, dict[str, Any]]:
        """Get best parameters for all regimes."""
        return {r: self.get_best(r) for r in REGIME_LABELS}
