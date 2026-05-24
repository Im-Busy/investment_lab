"""P25: NSGA-II multi-objective optimization for trading strategy parameters.

Implements Non-dominated Sorting Genetic Algorithm II for optimizing
multiple conflicting objectives simultaneously:
- Maximize return / Sharpe ratio
- Minimize max drawdown
- Maximize profit factor
- Minimize trade count instability

Reference: Deb et al. (2002), Sadeghi et al. (2021) "Combined Ensemble SVM + Fuzzy NSGA-II".
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NSGA2Config:
    """NSGA-II algorithm configuration."""

    population_size: int = 50
    generations: int = 30
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    crossover_eta: float = 20.0  # distribution index for SBX
    mutation_eta: float = 20.0  # distribution index for polynomial mutation
    tournament_size: int = 3
    seed: int | None = 42

    # Objective directions: True=minimize, False=maximize
    objectives_minimize: list[bool] = field(default_factory=lambda: [False, True, False, True])
    # Default: maximize return + profit factor, minimize DD + trade instability


@dataclass
class ParamDef:
    """Parameter definition for NSGA-II chromosome."""

    name: str
    lower: float
    upper: float
    is_integer: bool = False
    is_binary: bool = False


@dataclass
class NSGA2Result:
    """NSGA-II optimization result."""

    best_individuals: list[np.ndarray]
    best_objectives: np.ndarray  # shape (n_solutions, n_objectives)
    best_params: list[dict[str, Any]]
    pareto_front_ranks: np.ndarray
    generation_history: list[dict[str, Any]]
    n_generations: int
    n_objectives: int

    def summary(self) -> str:
        n_pareto = int(np.sum(self.pareto_front_ranks == 0))
        best_idx = -1 if len(self.best_params) > 1 else 0
        lines = [
            f"NSGA-II: {n_pareto} Pareto-optimal solutions from pop={len(self.best_params)}, gen={self.n_generations}",
            f"  Objectives: {self.best_objectives[best_idx].tolist()}",
            f"  Best params: {self.best_params[best_idx]}",
        ]
        return "\n".join(lines)

    @property
    def best(self) -> dict[str, Any]:
        """Return best solution (first Pareto-optimal individual by default)."""
        return self.best_params[0] if self.best_params else {}


class NSGA2Optimizer:
    """Non-dominated Sorting Genetic Algorithm II for multi-objective optimization.

    Chromosome encoding: real-valued vector + optional binary indicators.
    Selection: tournament on Pareto rank + crowding distance.
    Crossover: Simulated Binary Crossover (SBX).
    Mutation: Polynomial mutation.

    Usage:
        params = [
            ParamDef("entry_threshold", 0.3, 0.9),
            ParamDef("trail_stop", 1.0, 5.0),
            ParamDef("use_rsi", 0, 1, is_binary=True),
        ]

        def evaluate(x: np.array) -> list[float]:
            # Return [roi, max_dd, profit_factor, trade_instability]
            return [0.15, -0.08, 1.8, 0.05]

        nsga = NSGA2Optimizer(params, evaluate)
        result = nsga.optimize()
    """

    def __init__(
        self,
        params: list[ParamDef],
        eval_fn: Callable[[np.ndarray], list[float]],
        config: NSGA2Config | None = None,
    ) -> None:
        self.params = params
        self.eval_fn = eval_fn
        self.config = config or NSGA2Config()
        self.n_params = len(params)
        self.n_objectives = len(self.config.objectives_minimize)
        self._rng = np.random.default_rng(self.config.seed)

    def _encode(self, param_dict: dict[str, Any]) -> np.ndarray:
        """Convert param dict to normalized chromosome [0, 1]^n."""
        x = np.zeros(self.n_params)
        for i, p in enumerate(self.params):
            val = param_dict.get(p.name, p.lower)
            if p.is_binary:
                x[i] = 1.0 if val else 0.0
            else:
                x[i] = (val - p.lower) / (p.upper - p.lower) if p.upper > p.lower else 0.0
        return np.clip(x, 0.0, 1.0)

    def _decode(self, x: np.ndarray) -> dict[str, Any]:
        """Convert chromosome to parameter dict."""
        result = {}
        for i, p in enumerate(self.params):
            if p.is_binary:
                result[p.name] = x[i] >= 0.5
            elif p.is_integer:
                result[p.name] = int(round(p.lower + x[i] * (p.upper - p.lower)))
            else:
                result[p.name] = float(p.lower + x[i] * (p.upper - p.lower))
        return result

    def _decode_real(self, x: np.ndarray) -> np.ndarray:
        """Convert chromosome to real parameter values (for objective evaluation)."""
        return x  # NSGA-II operates in real space; eval_fn gets decoded values internally

    def _initialize_population(self) -> np.ndarray:
        """Create random initial population."""
        pop = self._rng.random((self.config.population_size, self.n_params))
        for i, p in enumerate(self.params):
            if p.is_binary:
                pop[:, i] = self._rng.binomial(1, 0.5, self.config.population_size)
        return pop

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate all individuals. Returns (population_size, n_objectives)."""
        obj = np.zeros((len(population), self.n_objectives))
        for i in range(len(population)):
            params = self._decode(population[i])
            raw = self.eval_fn(params)
            adjusted = []
            for j, (val, minimize) in enumerate(zip(raw, self.config.objectives_minimize)):
                adjusted.append(float(val) if minimize else -float(val))
            obj[i] = adjusted
        return obj

    def _non_dominated_sort(self, obj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fast non-dominated sorting (Deb 2002). Returns (fronts, rank array)."""
        n = len(obj)
        dominated_count = np.zeros(n, dtype=int)
        dominates_list: list[list[int]] = [[] for _ in range(n)]
        ranks = np.full(n, -1, dtype=int)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                i_dominates_j = np.all(obj[i] <= obj[j]) and np.any(obj[i] < obj[j])
                if i_dominates_j:
                    dominates_list[i].append(j)
                else:
                    j_dominates_i = np.all(obj[j] <= obj[i]) and np.any(obj[j] < obj[i])
                    if j_dominates_i:
                        dominated_count[i] += 1

        fronts: list[list[int]] = [[]]
        current_front = [i for i in range(n) if dominated_count[i] == 0]
        fronts[0] = current_front
        for idx in current_front:
            ranks[idx] = 0

        front_idx = 0
        while fronts[front_idx]:
            next_front: list[int] = []
            for i in fronts[front_idx]:
                for j in dominates_list[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        ranks[j] = front_idx + 1
                        next_front.append(j)
            front_idx += 1
            fronts.append(next_front)

        return ranks, np.array([len(f) for f in fronts if f])

    def _crowding_distance(self, obj: np.ndarray, front_indices: list[int]) -> np.ndarray:
        """Compute crowding distance for a single front."""
        if len(front_indices) <= 2:
            return np.full(len(front_indices), np.inf)

        front_obj = obj[front_indices]
        n = len(front_indices)
        distances = np.zeros(n)

        for m in range(self.n_objectives):
            sorted_idx = np.argsort(front_obj[:, m])
            f_min = front_obj[sorted_idx[0], m]
            f_max = front_obj[sorted_idx[-1], m]
            if f_max - f_min < 1e-12:
                continue

            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf

            for i in range(1, n - 1):
                distances[sorted_idx[i]] += (
                    front_obj[sorted_idx[i + 1], m] - front_obj[sorted_idx[i - 1], m]
                ) / (f_max - f_min)

        return distances

    def _tournament_select(
        self,
        population: np.ndarray,
        obj: np.ndarray,
        ranks: np.ndarray,
        crowding: np.ndarray,
    ) -> np.ndarray:
        """Binary tournament selection based on rank and crowding distance."""
        n = len(population)
        parents = np.zeros((n, self.n_params))

        for i in range(n):
            a, b = self._rng.choice(n, size=2, replace=False)
            if ranks[a] < ranks[b]:
                winner = a
            elif ranks[b] < ranks[a]:
                winner = b
            elif crowding[a] > crowding[b]:
                winner = a
            else:
                winner = b
            parents[i] = population[winner]

        return parents

    def _crossover(self, parents: np.ndarray) -> np.ndarray:
        """Simulated Binary Crossover (SBX)."""
        n = len(parents)
        children = parents.copy()
        eta = self.config.crossover_eta

        for i in range(0, n - 1, 2):
            if self._rng.random() > self.config.crossover_prob:
                continue
            for j in range(self.n_params):
                if self._rng.random() < 0.5:
                    u = self._rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    p1, p2 = parents[i, j], parents[i + 1, j]
                    children[i, j] = np.clip(0.5 * ((1 + beta) * p1 + (1 - beta) * p2), 0.0, 1.0)
                    children[i + 1, j] = np.clip(
                        0.5 * ((1 - beta) * p1 + (1 + beta) * p2), 0.0, 1.0
                    )

        return children

    def _mutate(self, population: np.ndarray) -> np.ndarray:
        """Polynomial mutation."""
        eta = self.config.mutation_eta
        pop = population.copy()

        for i in range(len(pop)):
            if self._rng.random() > self.config.mutation_prob:
                continue
            for j in range(self.n_params):
                if self._rng.random() < 1.0 / self.n_params:
                    u = self._rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    pop[i, j] = np.clip(pop[i, j] + delta, 0.0, 1.0)

        return pop

    def optimize(self) -> NSGA2Result:
        """Run NSGA-II optimization.

        Returns:
            NSGA2Result with Pareto-optimal solutions.
        """
        population = self._initialize_population()
        obj = self._evaluate_population(population)
        ranks, _ = self._non_dominated_sort(obj)
        crowding = np.zeros(len(population))
        for r in range(int(np.max(ranks)) + 1):
            front_idx = [i for i in range(len(ranks)) if ranks[i] == r]
            if front_idx:
                cd = self._crowding_distance(obj, front_idx)
                for idx, ci in zip(front_idx, cd):
                    crowding[idx] = ci

        history: list[dict[str, Any]] = []

        for gen in range(self.config.generations):
            parents = self._tournament_select(population, obj, ranks, crowding)
            children = self._crossover(parents)
            children = self._mutate(children)

            combined = np.vstack([population, children])
            combined_obj = self._evaluate_population(combined)
            combined_ranks, front_counts = self._non_dominated_sort(combined_obj)
            combined_crowding = np.zeros(len(combined))
            for r in range(int(np.max(combined_ranks)) + 1):
                front_idx = [i for i in range(len(combined_ranks)) if combined_ranks[i] == r]
                if front_idx:
                    cd = self._crowding_distance(combined_obj, front_idx)
                    for idx, ci in zip(front_idx, cd):
                        combined_crowding[idx] = ci

            sorted_idx = (
                np.lexsort((combined_crowding, -combined_ranks))[::-1][
                    : self.config.population_size
                ]
                if self.config.population_size < len(combined)
                else np.arange(len(combined))
            )

            sorted_idx = sorted_idx[: self.config.population_size]
            population = combined[sorted_idx]
            obj = combined_obj[sorted_idx]
            ranks = combined_ranks[sorted_idx]
            crowding = combined_crowding[sorted_idx]

            pareto_count = int(np.sum(ranks == 0))
            best_obj = obj[ranks == 0]
            history.append(
                {
                    "generation": gen,
                    "pareto_count": pareto_count,
                    "best_objective_0": float(np.min(obj[:, 0])),
                    "best_objective_1": float(np.min(obj[:, 1])) if self.n_objectives > 1 else 0,
                    "mean_rank": float(np.mean(ranks)),
                }
            )

            if gen % 10 == 0:
                logger.debug(
                    "NSGA-II gen %d: pareto=%d, obj0=%.4f, mean_rank=%.2f",
                    gen,
                    pareto_count,
                    history[-1]["best_objective_0"],
                    history[-1]["mean_rank"],
                )

        pareto_mask = ranks == 0
        if not np.any(pareto_mask):
            pareto_mask = ranks == np.min(ranks)

        best_pop = population[pareto_mask]
        best_obj = obj[pareto_mask]
        best_params = [self._decode(x) for x in best_pop]

        return NSGA2Result(
            best_individuals=list(best_pop),
            best_objectives=-best_obj,
            best_params=best_params,
            pareto_front_ranks=ranks[pareto_mask],
            generation_history=history,
            n_generations=self.config.generations,
            n_objectives=self.n_objectives,
        )
