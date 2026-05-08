"""GWO (Grey Wolf Optimizer) for CatBoost hyperparameter tuning.

Four-level wolf hierarchy: alpha (best), beta (second), delta (third), omega (rest).
Wolves converge toward prey (optimal HPs) via position updates:
  A = 2*a*r1 - a  (decaying coefficient)
  C = 2*r2         (stochastic coefficient)
  D = |C * X_prey - X_wolf|
  X_next = X_prey - A * D

Reference: Mirjalili et al., "Grey Wolf Optimizer", Advances in Engineering Software, 2014.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

import numpy as np

from src.ml.tuning.base import BaseOptimizer, OptimizerResult, ParamSpec, SearchSpace

logger = logging.getLogger(__name__)

# Standard CatBoost hyperparameter space
CATBOOST_PARAM_SPACE = [
    ParamSpec("learning_rate", "float", 0.01, 0.3, log_scale=True),
    ParamSpec("depth", "int", 3, 10),
    ParamSpec("l2_leaf_reg", "float", 1.0, 30.0, log_scale=True),
    ParamSpec("random_strength", "float", 0.5, 5.0),
    ParamSpec("bagging_temperature", "float", 0.0, 2.0),
    ParamSpec("border_count", "int", 32, 255),
    ParamSpec("min_data_in_leaf", "int", 5, 50),
]


class GWOTuner(BaseOptimizer):
    """GWO-based CatBoost hyperparameter tuner.

    Optimizes PatternClassifier, SignalRegressor, or RegimeClassifier HPs
    using PurgedKFold cross-validation as the fitness function.

    Usage:
        from src.ml.tuning import GWOTuner, SearchSpace, CATBOOST_PARAM_SPACE

        def fitness(params: dict) -> float:
            model = PatternClassifier(**params, n_estimators=200)
            result = model.train(X, y)
            return result.test_auc

        space = SearchSpace(CATBOOST_PARAM_SPACE)
        tuner = GWOTuner(space, fitness, n_wolves=20)
        result = tuner.optimize(max_iter=50)
    """

    def __init__(
        self,
        search_space: SearchSpace,
        fitness_fn: Callable[[Dict[str, Any]], float],
        n_wolves: int = 20,
        maximize: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__(
            search_space=search_space,
            fitness_fn=fitness_fn,
            population_size=n_wolves,
            maximize=maximize,
            seed=seed,
        )
        self._alpha_pos: np.ndarray | None = None
        self._alpha_score = -np.inf if maximize else np.inf
        self._beta_pos: np.ndarray | None = None
        self._beta_score = -np.inf if maximize else np.inf
        self._delta_pos: np.ndarray | None = None
        self._delta_score = -np.inf if maximize else np.inf

    def optimize(self, max_iter: int = 50, early_stop: int = 10) -> OptimizerResult:
        """Run GWO optimization.

        Args:
            max_iter: Maximum iterations.
            early_stop: Stop if alpha doesn't improve for this many iterations.

        Returns:
            OptimizerResult with best_params and best_score (AUC or accuracy).
        """
        n_dims = self.search_space.n_dims
        wolves = self._initialize_population()
        fitness = np.array([self._evaluate(w) for w in wolves])

        best_idx = int(np.argmax(fitness) if self.maximize else np.argmin(fitness))
        self._alpha_pos = wolves[best_idx].copy()
        self._alpha_score = float(fitness[best_idx])

        beta_idx = self._second_best(fitness, exclude=best_idx)
        self._beta_pos = wolves[beta_idx].copy()
        self._beta_score = float(fitness[beta_idx])

        delta_idx = self._second_best(fitness, exclude=(best_idx, beta_idx))
        self._delta_pos = wolves[delta_idx].copy()
        self._delta_score = float(fitness[delta_idx])

        no_improve = 0
        history: List[float] = [self._alpha_score]

        for t in range(max_iter):
            # Linearly decaying coefficient
            a = 2.0 - 2.0 * t / max_iter

            for i in range(len(wolves)):
                for leader in (self._alpha_pos, self._beta_pos, self._delta_pos):
                    r1 = self._rng.random(n_dims)
                    r2 = self._rng.random(n_dims)
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    D = np.abs(C * leader - wolves[i])
                    wolves[i] = leader - A * D

                wolves[i] = self.search_space.clip_position(wolves[i])

            fitness = np.array([self._evaluate(w) for w in wolves])

            sorted_idx = np.argsort(fitness)
            if not self.maximize:
                sorted_idx = sorted_idx[::-1]

            top_three = sorted_idx[-3:]

            best_score = float(fitness[top_three[-1]])
            if self._is_better(best_score, self._alpha_score):
                self._alpha_score = best_score
                self._alpha_pos = wolves[top_three[-1]].copy()
                no_improve = 0
            else:
                no_improve += 1

            if self._is_better(float(fitness[top_three[-2]]), self._beta_score):
                self._beta_score = float(fitness[top_three[-2]])
                self._beta_pos = wolves[top_three[-2]].copy()

            if self._is_better(float(fitness[top_three[-3]]), self._delta_score):
                self._delta_score = float(fitness[top_three[-3]])
                self._delta_pos = wolves[top_three[-3]].copy()

            history.append(self._alpha_score)

            if t % 5 == 0:
                best_vals = ", ".join(
                    f"{self.search_space.params[i].name}={self._alpha_pos[i]:.{'0' if self.search_space.params[i].type == 'int' else '4f'}}"
                    for i in range(len(self._alpha_pos))
                )
                logger.info(f"GWO iter {t:3d}: alpha {self._alpha_score:.4f}, {best_vals}")

            if no_improve >= early_stop:
                logger.info(f"GWO converged at iteration {t}")
                break

        return OptimizerResult(
            best_params=self.search_space.to_dict(self._alpha_pos),
            best_score=self._alpha_score,
            all_scores=fitness.tolist(),
            best_position=self._alpha_pos,
            n_iterations=t + 1,
            convergence_history=history,
            optimizer_name="GWO",
        )

    def _second_best(
        self,
        fitness: np.ndarray,
        exclude: int | tuple[int, ...],
    ) -> int:
        """Find the index of the second-best wolf, excluding given indices."""
        if isinstance(exclude, int):
            exclude = (exclude,)
        mask = np.ones(len(fitness), dtype=bool)
        for idx in exclude:
            mask[idx] = False
        if self.maximize:
            return int(np.argmax(np.where(mask, fitness, -np.inf)))
        return int(np.argmin(np.where(mask, fitness, np.inf)))
