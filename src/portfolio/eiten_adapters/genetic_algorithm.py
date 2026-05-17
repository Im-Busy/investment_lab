"""Genetic Algorithm portfolio optimization.

Evolves a population of portfolio weight vectors via selection (Sharpe fitness),
mutation (noise perturbation), and crossover. Maintains weight simplex constraint
via L1 normalization at each generation.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GAResult:
    """Result of GA portfolio optimization."""

    weights: dict[str, float]
    """Optimal portfolio weights."""
    sharpe: float
    """Sharpe ratio of the best gene."""
    generations: int
    """Number of generations run."""
    history: list[float]
    """Best Sharpe per generation."""


class GeneticAlgorithmPortfolio:
    """Genetic algorithm for portfolio weight optimization.

    Parameters:
        population_size: Initial gene pool size (default 200).
        generations: Number of evolution generations (default 50).
        selection_top: Number of top genes retained per generation (default 30).
        mutation_rate: Fraction of genes mutated per generation (default 0.5).
        weight_update_factor: Mutation step size (default 0.1).
        crossover_probability: Probability of crossover per gene pair (default 0.05).
        random_seed: Seed for reproducibility.
    """

    def __init__(
        self,
        population_size: int = 200,
        generations: int = 50,
        selection_top: int = 30,
        mutation_rate: float = 0.5,
        weight_update_factor: float = 0.1,
        crossover_probability: float = 0.05,
        random_seed: int | None = None,
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.selection_top = selection_top
        self.mutation_rate = mutation_rate
        self.weight_update_factor = weight_update_factor
        self.crossover_probability = crossover_probability

        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

    def build(
        self,
        symbols: list[str],
        returns_matrix: np.ndarray,
    ) -> dict[str, float]:
        """Run GA to find optimal portfolio weights.

        Args:
            symbols: List of asset symbols.
            returns_matrix: T x N matrix of percentage returns.

        Returns:
            Dict mapping symbol -> weight (sums to 1.0).
        """
        result = self.optimize(symbols, returns_matrix)
        return result.weights

    def optimize(
        self,
        symbols: list[str],
        returns_matrix: np.ndarray,
    ) -> GAResult:
        """Run full GA optimization and return detailed result.

        Args:
            symbols: List of asset symbols.
            returns_matrix: T x N matrix of percentage returns.

        Returns:
            GAResult with weights, Sharpe, and evolution history.
        """
        n_assets = len(symbols)

        genes = self._generate_initial_genes(n_assets)
        history: list[float] = []

        for generation in range(self.generations):
            scored = self._select(returns_matrix, genes)
            best_score = scored[0][0]
            history.append(float(best_score))
            top_genes = [item[1] for item in scored]

            genes = self._mutate(top_genes, n_assets)
            genes = genes + self._crossover(genes)

            if len(genes) > self.population_size:
                genes = random.sample(genes, self.population_size)

            if generation % 10 == 0:
                logger.debug(
                    "GA generation %d/%d best Sharpe: %.4f",
                    generation,
                    self.generations,
                    best_score,
                )

        final_scored = self._select(returns_matrix, genes)
        best_gene = final_scored[0][1]
        weights = self._normalize(np.array(best_gene))

        return GAResult(
            weights=dict(zip(symbols, weights)),
            sharpe=final_scored[0][0],
            generations=self.generations,
            history=history,
        )

    def _generate_initial_genes(self, n_assets: int) -> list[np.ndarray]:
        genes = []
        for _ in range(self.population_size):
            gene = np.random.uniform(-1.0, 1.0, n_assets)
            genes.append(gene)
        return genes

    def _select(
        self,
        returns_matrix: np.ndarray,
        genes: list[np.ndarray],
    ) -> list[tuple[float, np.ndarray]]:
        scored: list[tuple[float, np.ndarray]] = []

        for gene in genes:
            w = self._normalize(gene)
            port_returns = returns_matrix @ w
            score = self._fitness(port_returns)
            scored.append((score, gene))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self.selection_top]
        # Inject random genes for diversity
        n_assets = len(genes[0])
        for _ in range(5):
            random_gene = np.random.uniform(-1.0, 1.0, n_assets)
            top.append((0.0, random_gene))

        return top

    def _mutate(self, genes: list[np.ndarray], n_assets: int) -> list[np.ndarray]:
        mutated = list(genes)
        n_mutate = max(1, int(len(genes) * self.mutation_rate))

        for _ in range(n_mutate):
            idx = random.randrange(len(genes))
            noise = self.weight_update_factor * np.random.uniform(-1.0, 1.0, n_assets)
            mutated.append(genes[idx] + noise)

        return mutated

    def _crossover(self, genes: list[np.ndarray]) -> list[np.ndarray]:
        crossovers = []
        for _ in range(len(genes)):
            if random.random() < self.crossover_probability:
                a = random.choice(genes)
                b = random.choice(genes)
                split = random.randrange(1, len(a) - 1)
                child = np.concatenate([a[:split], b[split:]])
                crossovers.append(child)
        return crossovers

    @staticmethod
    def _normalize(w: np.ndarray) -> np.ndarray:
        total = float(np.sum(np.abs(w)))
        if total < 1e-12:
            return np.ones_like(w) / len(w)
        return w / total

    @staticmethod
    def _fitness(returns: np.ndarray) -> float:
        std = float(np.std(returns))
        if std < 1e-12:
            return 0.0
        return float(np.mean(returns) / std)
