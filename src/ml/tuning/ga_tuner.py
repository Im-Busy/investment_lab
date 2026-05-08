"""GA (Genetic Algorithm) for unsupervised regime optimization.

Optimizes the number of market regimes (n_regimes) via KMeans clustering
with silhouette score as fitness. The GA searches for the natural number
of market states present in OHLCV-derived features.

Reference: Holland, "Adaptation in Natural and Artificial Systems", 1975.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.ml.tuning.base import BaseOptimizer, OptimizerResult, ParamSpec, SearchSpace

logger = logging.getLogger(__name__)

# Regime clustering search space: n_regimes + KMeans hyperparams
REGIME_PARAM_SPACE = [
    ParamSpec("n_regimes", "int", 2, 8),
    ParamSpec("n_init", "int", 5, 30),
    ParamSpec("max_iter", "int", 100, 500),
    ParamSpec("tol", "float", 1e-5, 1e-2, log_scale=True),
]


def silhouette_fitness(
    X: np.ndarray,
    n_regimes: int,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: int = 42,
) -> float:
    """Compute silhouette score for KMeans clustering.

    Args:
        X: Feature matrix (n_samples × n_features).
        n_regimes: Number of clusters.
        n_init: Number of KMeans initializations.
        max_iter: Max KMeans iterations.
        tol: KMeans convergence tolerance.
        random_state: Random seed.

    Returns:
        Silhouette score in [-1, 1]. Higher = better defined clusters.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if len(X) < n_regimes * 2:
        return -1.0

    kmeans = KMeans(
        n_clusters=n_regimes,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    labels = kmeans.fit_predict(X)

    if len(np.unique(labels)) < 2:
        return -1.0

    return float(silhouette_score(X, labels, random_state=random_state))


class GARegimeOptimizer(BaseOptimizer):
    """Genetic Algorithm for unsupervised regime optimization.

    Evolves a population of (n_regimes, n_init, max_iter, tol) tuples.
    Uses tournament selection, arithmetic crossover, and Gaussian mutation.
    Elite individual preserved each generation.

    Usage:
        from src.ml.tuning.ga_tuner import GARegimeOptimizer, REGIME_PARAM_SPACE

        def fitness(params: dict) -> float:
            return silhouette_fitness(X, **params)

        space = SearchSpace(REGIME_PARAM_SPACE)
        ga = GARegimeOptimizer(space, fitness, population_size=20)
        result = ga.optimize(max_iter=30)
    """

    def __init__(
        self,
        search_space: SearchSpace,
        fitness_fn: Any,
        population_size: int = 20,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15,
        tournament_size: int = 3,
        maximize: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__(
            search_space=search_space,
            fitness_fn=fitness_fn,
            population_size=population_size,
            maximize=maximize,
            seed=seed,
        )
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = min(tournament_size, population_size)

    def optimize(self, max_iter: int = 30, early_stop: int = 10) -> OptimizerResult:
        """Run GA optimization.

        Args:
            max_iter: Maximum generations.
            early_stop: Stop if no improvement for this many generations.

        Returns:
            OptimizerResult with best_params (n_regimes etc.) and silhouette score.
        """
        n_dims = self.search_space.n_dims
        pop = self._initialize_population()
        fitness = np.array([self._evaluate(p) for p in pop])

        best_idx = int(np.argmax(fitness) if self.maximize else np.argmin(fitness))
        best_pos = pop[best_idx].copy()
        best_score = float(fitness[best_idx])

        no_improve = 0
        history: List[float] = [best_score]

        for generation in range(max_iter):
            new_pop = np.zeros_like(pop)
            elite_idx = int(np.argmax(fitness) if self.maximize else np.argmin(fitness))
            new_pop[0] = pop[elite_idx].copy()

            for i in range(1, len(pop)):
                parent1 = self._tournament_select(pop, fitness)
                parent2 = self._tournament_select(pop, fitness)

                if self._rng.random() < self.crossover_rate:
                    child = self._arithmetic_crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                child = self._mutate(child, generation, max_iter)

                new_pop[i] = self.search_space.clip_position(child)

            pop = new_pop
            fitness = np.array([self._evaluate(p) for p in pop])

            best_idx = int(np.argmax(fitness) if self.maximize else np.argmin(fitness))
            current_best = float(fitness[best_idx])

            if self._is_better(current_best, best_score):
                best_score = current_best
                best_pos = pop[best_idx].copy()
                no_improve = 0
            else:
                no_improve += 1

            history.append(best_score)

            if generation % 5 == 0:
                regime_count = int(best_pos[0])
                logger.info(
                    f"GA gen {generation:3d}: silhouette {best_score:.4f}, "
                    f"n_regimes {regime_count}, no_improve {no_improve}"
                )

            if no_improve >= early_stop:
                logger.info(f"GA converged at generation {generation}")
                break

        return OptimizerResult(
            best_params=self.search_space.to_dict(best_pos),
            best_score=best_score,
            all_scores=fitness.tolist(),
            best_position=best_pos,
            n_iterations=generation + 1,
            convergence_history=history,
            optimizer_name="GA",
        )

    def _tournament_select(self, pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Select individual via tournament selection."""
        indices = self._rng.choice(len(pop), size=self.tournament_size, replace=False)
        if self.maximize:
            winner = indices[np.argmax(fitness[indices])]
        else:
            winner = indices[np.argmin(fitness[indices])]
        return pop[winner].copy()

    def _arithmetic_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Blend two parents with random weight."""
        alpha = self._rng.random(len(parent1))
        return alpha * parent1 + (1 - alpha) * parent2

    def _mutate(self, individual: np.ndarray, generation: int, max_generations: int) -> np.ndarray:
        """Gaussian mutation with adaptive step size.

        Mutation strength decays linearly from 20% to 5% of param range.
        """
        mutant = individual.copy()
        progress = generation / max(max_generations, 1)
        base_sigma = 0.2 * (1.0 - 0.75 * progress)

        for dim, param in enumerate(self.search_space.params):
            if self._rng.random() < self.mutation_rate:
                if param.type == "int":
                    sigma = max(base_sigma * (param.high - param.low), 1.0)
                    mutant[dim] += self._rng.normal(0, sigma)
                elif param.type == "float":
                    sigma = base_sigma * (param.high - param.low)
                    mutant[dim] += self._rng.normal(0, sigma)

        return mutant


class RegimeDiscovery:
    """Unsupervised regime discovery via GA-optimized KMeans clustering.

    Finds the optimal number of market regimes and their cluster centroids
    by maximizing silhouette score on OHLCV-derived features.

    Usage:
        discovery = RegimeDiscovery(features_df)
        result = discovery.optimize(population_size=20, max_generations=30)
        # result.best_params["n_regimes"] → optimal regime count
        # result.best_score → silhouette score
        # discovery.centroids_ → (n_regimes × n_features) centroid matrix
        # discovery.labels_ → regime label per sample
    """

    def __init__(
        self,
        X: pd.DataFrame,
        random_state: int = 42,
    ) -> None:
        """Initialize regime discovery.

        Args:
            X: Feature DataFrame (n_samples × n_features). Must be numeric.
            random_state: Random seed.
        """
        self.X = X.copy()
        self._X_numeric = self.X.select_dtypes(include=[np.number])
        self._X_scaled: np.ndarray | None = None
        self.random_state = random_state

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.n_regimes_: int = 0
        self.silhouette_: float = 0.0

    def _preprocess(self) -> np.ndarray:
        """Scale features for clustering."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        self._X_scaled = scaler.fit_transform(self._X_numeric)
        return self._X_scaled

    def optimize(
        self,
        population_size: int = 20,
        max_generations: int = 30,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15,
    ) -> OptimizerResult:
        """Run GA to find optimal regime count.

        Args:
            population_size: Number of individuals per generation.
            max_generations: Maximum generations.
            crossover_rate: Probability of crossover.
            mutation_rate: Per-gene mutation probability.

        Returns:
            OptimizerResult with best_params (n_regimes, etc.) and silhouette score.
        """
        X = self._preprocess()

        space = SearchSpace(REGIME_PARAM_SPACE, seed=self.random_state)

        def fitness(params: Dict[str, Any]) -> float:
            return silhouette_fitness(X, **params, random_state=self.random_state)

        ga = GARegimeOptimizer(
            search_space=space,
            fitness_fn=fitness,
            population_size=population_size,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            maximize=True,
            seed=self.random_state,
        )
        result = ga.optimize(max_iter=max_generations)

        self._fit_final(result.best_params)
        return result

    def _fit_final(self, params: Dict[str, Any]) -> None:
        """Fit final KMeans with best params to extract centroids and labels."""
        from sklearn.cluster import KMeans

        n_regimes = int(params["n_regimes"])
        X = self._X_scaled

        kmeans = KMeans(
            n_clusters=n_regimes,
            n_init=int(params.get("n_init", 10)),
            max_iter=int(params.get("max_iter", 300)),
            tol=float(params.get("tol", 1e-4)),
            random_state=self.random_state,
        )
        self.labels_ = kmeans.fit_predict(X)
        self.centroids_ = kmeans.cluster_centers_
        self.n_regimes_ = n_regimes

        from sklearn.metrics import silhouette_score

        if len(np.unique(self.labels_)) >= 2 and len(X) > n_regimes * 2:
            self.silhouette_ = float(
                silhouette_score(X, self.labels_, random_state=self.random_state)
            )

    def get_regime_labels(self) -> pd.Series:
        """Return regime labels as a Series with original DataFrame index."""
        if self.labels_ is None:
            raise ValueError("Model not fitted. Call optimize() first.")
        return pd.Series(
            self.labels_,
            index=self._X_numeric.index,
            name="regime",
        )

    def get_centroid_features(self, feature_names: List[str] | None = None) -> pd.DataFrame:
        """Return centroid matrix with feature names.

        Args:
            feature_names: List of feature names (default: X.columns).

        Returns:
            DataFrame with centroids (n_regimes × n_features).
        """
        if self.centroids_ is None:
            raise ValueError("Model not fitted. Call optimize() first.")
        cols = feature_names or list(self._X_numeric.columns)
        return pd.DataFrame(
            self.centroids_,
            index=[f"regime_{i}" for i in range(len(self.centroids_))],
            columns=cols[: self.centroids_.shape[1]],
        )
