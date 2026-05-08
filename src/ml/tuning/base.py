"""Shared abstractions for metaheuristic optimizers.

Provides ParamSpec, SearchSpace, and OptimizerResult used by ARO, GWO, GA, WOA.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np


@dataclass
class ParamSpec:
    """Specification for a single tunable hyperparameter."""

    name: str
    type: str  # 'int', 'float', 'choice'
    low: float | None = None
    high: float | None = None
    choices: list | None = None
    log_scale: bool = False

    def __post_init__(self) -> None:
        if self.type == "choice" and self.choices is None:
            raise ValueError(f"ParamSpec '{self.name}': 'choice' type requires choices")
        if self.type in ("int", "float") and (self.low is None or self.high is None):
            raise ValueError(
                f"ParamSpec '{self.name}': {self.type} type requires low and high bounds"
            )

    def sample(self, rng: np.random.Generator) -> Any:
        """Draw a random value within bounds."""
        if self.type == "choice":
            return rng.choice(self.choices)
        if self.type == "int":
            if self.log_scale:
                log_low = np.log(self.low)
                log_high = np.log(self.high)
                val = np.exp(rng.uniform(log_low, log_high))
            else:
                val = rng.uniform(self.low, self.high + 1)
            return int(np.clip(val, self.low, self.high))
        if self.type == "float":
            if self.log_scale:
                log_low = np.log(self.low)
                log_high = np.log(self.high)
                val = np.exp(rng.uniform(log_low, log_high))
            else:
                val = rng.uniform(self.low, self.high)
            return float(np.clip(val, self.low, self.high))
        raise ValueError(f"Unknown type: {self.type}")

    def clip(self, value: Any) -> Any:
        """Clip a value to valid bounds."""
        if self.type == "choice":
            return value if value in self.choices else self.choices[0]
        if self.type == "int":
            return int(np.clip(value, self.low, self.high))
        return float(np.clip(value, self.low, self.high))


@dataclass
class SearchSpace:
    """Collection of hyperparameters with shared random state."""

    params: List[ParamSpec]
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    @property
    def n_dims(self) -> int:
        return len(self.params)

    def random_position(self) -> np.ndarray:
        """Generate a random position vector in search space.

        Returns:
            Array of length n_dims with values in valid ranges.
        """
        pos = np.zeros(self.n_dims)
        for i, p in enumerate(self.params):
            pos[i] = p.sample(self._rng)
        return pos

    def clip_position(self, position: np.ndarray) -> np.ndarray:
        """Clip each dimension to its valid range."""
        clipped = position.copy()
        for i, p in enumerate(self.params):
            clipped[i] = p.clip(clipped[i])
        return clipped

    def to_dict(self, position: np.ndarray) -> Dict[str, Any]:
        """Convert a position vector to a keyword-argument dict."""
        return {p.name: p.clip(position[i]) for i, p in enumerate(self.params)}


@dataclass
class OptimizerResult:
    """Result from a metaheuristic optimization run."""

    best_params: Dict[str, Any]
    best_score: float
    all_scores: List[float] = field(default_factory=list)
    best_position: np.ndarray | None = None
    n_iterations: int = 0
    convergence_history: List[float] = field(default_factory=list)
    optimizer_name: str = ""

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.all_scores)) if self.all_scores else self.best_score

    @property
    def score_std(self) -> float:
        return float(np.std(self.all_scores)) if self.all_scores else 0.0


# Map CatBoost-native param names → PatternClassifier __init__ kwargs
CATBOOST_TO_PATTERNCLASSIFIER = {
    "learning_rate": "learning_rate",
    "depth": "max_depth",
    "l2_leaf_reg": "l2_leaf_reg",
    "random_strength": "random_strength",
    "bagging_temperature": "bagging_temperature",
    "border_count": "border_count",
    "min_data_in_leaf": "min_data_in_leaf",
}


def map_params(params: Dict[str, Any], target: str = "pattern_classifier") -> Dict[str, Any]:
    """Map CatBoost-native param names to target model kwargs.

    Args:
        params: Dict with CatBoost-native names (from CATBOOST_PARAM_SPACE).
        target: 'pattern_classifier', 'signal_regressor', or 'regime'.

    Returns:
        Dict with target-model kwarg names.
    """
    if target == "pattern_classifier":
        return {CATBOOST_TO_PATTERNCLASSIFIER.get(k, k): v for k, v in params.items()}
    return params


class BaseOptimizer(ABC):
    """Abstract base for population-based metaheuristic optimizers.

    Subclasses implement the position-update rule specific to each algorithm
    (GWO: wolf hierarchy, ARO: detour + hiding, GA: crossover + mutation, WOA: spiral).
    """

    def __init__(
        self,
        search_space: SearchSpace,
        fitness_fn: Callable[[Dict[str, Any]], float],
        population_size: int = 30,
        maximize: bool = True,
        seed: int = 42,
    ) -> None:
        self.search_space = search_space
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.maximize = maximize
        self._rng = np.random.default_rng(seed)
        self._best_score = -np.inf if maximize else np.inf

    def _evaluate(self, position: np.ndarray) -> float:
        """Evaluate fitness of a position vector."""
        params = self.search_space.to_dict(position)
        score = self.fitness_fn(params)
        return score

    def _is_better(self, score: float, reference: float) -> bool:
        """Check if score is better than reference given maximize/minimize."""
        return score > reference if self.maximize else score < reference

    def _initialize_population(self) -> np.ndarray:
        """Create random initial population."""
        population = np.zeros((self.population_size, self.search_space.n_dims))
        for i in range(self.population_size):
            population[i] = self.search_space.random_position()
        return population

    @abstractmethod
    def optimize(self, max_iter: int = 50, early_stop: int = 10) -> OptimizerResult:
        """Run the optimization loop."""
        ...
