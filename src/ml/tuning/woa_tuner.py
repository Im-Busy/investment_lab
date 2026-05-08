"""WOA (Whale Optimization Algorithm) for pattern threshold tuning.

Optimizes per-pattern min_confidence thresholds that control which
detected chart patterns pass through to strategy execution.

The WOA mimics humpback whale bubble-net hunting:
- Spiral update (exploitation): spiral toward best prey
- Shrinking circle (exploration): move toward random whale

Reference: Mirjalili & Lewis, "The Whale Optimization Algorithm",
Advances in Engineering Software, 2016.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

import numpy as np

from src.ml.tuning.base import BaseOptimizer, OptimizerResult, ParamSpec, SearchSpace

logger = logging.getLogger(__name__)

# Default pattern categories for threshold tuning
DEFAULT_PATTERN_CATEGORIES = [
    "reversal_bullish",
    "reversal_bearish",
    "continuation_bullish",
    "continuation_bearish",
    "candlestick_bullish",
    "candlestick_bearish",
    "harmonic_bullish",
    "harmonic_bearish",
]

# Lower-bound threshold that passes almost all signals
DEFAULT_THRESHOLD_MIN = 0.10
# Upper-bound threshold that passes only high-confidence signals
DEFAULT_THRESHOLD_MAX = 0.95


def build_threshold_search_space(
    pattern_categories: List[str] | None = None,
    threshold_min: float = DEFAULT_THRESHOLD_MIN,
    threshold_max: float = DEFAULT_THRESHOLD_MAX,
) -> SearchSpace:
    """Create a SearchSpace with one float param per pattern category.

    Args:
        pattern_categories: List of pattern category names.
        threshold_min: Minimum threshold value.
        threshold_max: Maximum threshold value.

    Returns:
        SearchSpace with one 0.0-1.0 float param per category.
    """
    categories = pattern_categories or DEFAULT_PATTERN_CATEGORIES
    params = [
        ParamSpec(name=cat, type="float", low=threshold_min, high=threshold_max)
        for cat in categories
    ]
    return SearchSpace(params)


class WOATuner(BaseOptimizer):
    """WOA-based pattern threshold tuner.

    Each whale position is a vector of per-pattern-category min_confidence
    thresholds. Fitness is backtest performance (Sharpe, return, profit factor).

    Usage:
        from src.ml.tuning.woa_tuner import WOATuner, build_threshold_search_space

        space = build_threshold_search_space()
        tuner = WOATuner(space, fitness_fn, n_whales=30)
        result = tuner.optimize(max_iter=50)
        # result.best_params → {"reversal_bullish": 0.55, "candlestick_bearish": 0.70, ...}
    """

    def __init__(
        self,
        search_space: SearchSpace,
        fitness_fn: Callable[[Dict[str, Any]], float],
        n_whales: int = 30,
        maximize: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__(
            search_space=search_space,
            fitness_fn=fitness_fn,
            population_size=n_whales,
            maximize=maximize,
            seed=seed,
        )
        self._best_whale: np.ndarray | None = None

    def optimize(self, max_iter: int = 50, early_stop: int = 10) -> OptimizerResult:
        """Run WOA optimization.

        Args:
            max_iter: Maximum iterations.
            early_stop: Stop if best doesn't improve for this many iterations.

        Returns:
            OptimizerResult with best_params (per-pattern thresholds) and score.
        """
        n_dims = self.search_space.n_dims
        whales = self._initialize_population()
        fitness = np.array([self._evaluate(w) for w in whales])

        best_idx = int(np.argmax(fitness) if self.maximize else np.argmin(fitness))
        self._best_whale = whales[best_idx].copy()
        best_score = float(fitness[best_idx])

        no_improve = 0
        history: List[float] = [best_score]

        for t in range(max_iter):
            # Linearly decaying exploration coefficient
            a = 2.0 - 2.0 * t / max_iter
            a2 = -1.0 - t / max_iter

            for i in range(len(whales)):
                r = self._rng.random()
                A = 2 * a * self._rng.random(n_dims) - a
                C = 2 * self._rng.random(n_dims)
                l = a2 * self._rng.random() + 1.0

                p = self._rng.random()

                if p < 0.5:
                    if np.all(np.abs(A) < 1):
                        # Shrinking encircling → exploit
                        D = np.abs(C * self._best_whale - whales[i])
                        whales[i] = self._best_whale - A * D
                    else:
                        # Random search → explore
                        rand_idx = self._rng.integers(0, len(whales))
                        D = np.abs(C * whales[rand_idx] - whales[i])
                        whales[i] = whales[rand_idx] - A * D
                else:
                    # Spiral update → exploit with local refinement
                    D_prime = np.abs(self._best_whale - whales[i])
                    whales[i] = D_prime * np.exp(l) * np.cos(2 * np.pi * l) + self._best_whale

                whales[i] = self.search_space.clip_position(whales[i])

            fitness = np.array([self._evaluate(w) for w in whales])

            best_idx = int(np.argmax(fitness) if self.maximize else np.argmin(fitness))
            current_best = float(fitness[best_idx])

            if self._is_better(current_best, best_score):
                self._best_whale = whales[best_idx].copy()
                best_score = current_best
                no_improve = 0
            else:
                no_improve += 1

            history.append(best_score)

            if t % 5 == 0:
                params_str = ", ".join(
                    f"{self.search_space.params[i].name}={self._best_whale[i]:.3f}"
                    for i in range(min(3, n_dims))
                )
                logger.info(
                    f"WOA iter {t:3d}: best {best_score:.4f}, [{params_str}], "
                    f"no_improve {no_improve}"
                )

            if no_improve >= early_stop:
                logger.info(f"WOA converged at iteration {t}")
                break

        return OptimizerResult(
            best_params=self.search_space.to_dict(self._best_whale),
            best_score=best_score,
            all_scores=fitness.tolist(),
            best_position=self._best_whale,
            n_iterations=t + 1,
            convergence_history=history,
            optimizer_name="WOA",
        )


class WOAPatternThresholdOptimizer:
    """High-level WOA wrapper for pattern confidence threshold optimization.

    Given a list of pattern names, this optimizer tunes per-pattern
    min_confidence thresholds to maximize backtest metrics.

    Usage:
        from src.ml.tuning.woa_tuner import WOAPatternThresholdOptimizer

        optimizer = WOAPatternThresholdOptimizer(pattern_names=patterns)
        result = optimizer.optimize(
            fitness_fn=backtest_evaluator,
            n_whales=20,
            max_iter=30,
        )
        # result.best_params → {"doji": 0.45, "hammer": 0.60, "engulfing": 0.70, ...}
    """

    def __init__(
        self,
        pattern_names: List[str],
        threshold_min: float = DEFAULT_THRESHOLD_MIN,
        threshold_max: float = DEFAULT_THRESHOLD_MAX,
        seed: int = 42,
    ) -> None:
        """Initialize pattern threshold optimizer.

        Args:
            pattern_names: List of unique pattern names from pattern detectors.
            threshold_min: Minimum allowed threshold.
            threshold_max: Maximum allowed threshold.
            seed: Random seed.
        """
        self.pattern_names = list(pattern_names)
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.seed = seed

    def optimize(
        self,
        fitness_fn: Callable[[Dict[str, float]], float],
        n_whales: int = 30,
        max_iter: int = 50,
        maximize: bool = True,
    ) -> OptimizerResult:
        """Run WOA to find optimal per-pattern thresholds.

        Args:
            fitness_fn: Function that takes {pattern: threshold} dict and returns
                a performance metric (e.g., Sharpe ratio).
            n_whales: Number of whales in the population.
            max_iter: Maximum WOA iterations.
            maximize: Whether to maximize or minimize fitness.

        Returns:
            OptimizerResult with best_params mapping pattern → threshold.
        """
        threshold_params = [
            ParamSpec(
                name=name,
                type="float",
                low=self.threshold_min,
                high=self.threshold_max,
            )
            for name in self.pattern_names
        ]
        space = SearchSpace(threshold_params, seed=self.seed)

        tuner = WOATuner(
            search_space=space,
            fitness_fn=fitness_fn,
            n_whales=n_whales,
            maximize=maximize,
            seed=self.seed,
        )
        return tuner.optimize(max_iter=max_iter)
