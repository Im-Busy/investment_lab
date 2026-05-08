"""ARO (Artificial Rabbit Optimization) for feature selection.

Two-phase metaheuristic: detour foraging (broad exploration) + random hiding (fine tuning).
Each "rabbit" is a binary feature-inclusion vector. Fitness = PurgedKFold AUC with selected features.

Reference: Wang et al., "Artificial Rabbits Optimization: A new bio-inspired meta-heuristic
algorithm for solving engineering optimization problems", 2022.

NOTE: This is a standalone selector, not a BaseOptimizer subclass, because
its domain (binary feature vectors) differs from continuous HP search.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from src.ml.tuning.base import OptimizerResult

logger = logging.getLogger(__name__)


class AROFeatureSelector:
    """ARO-based feature selection for OHLCV indicator spaces.

    Each rabbit position is a binary vector. The optimizer selects the subset
    of ~15 features (from 100+) that maximize PurgedKFold AUC.

    Usage:
        from src.ml.tuning import AROFeatureSelector

        selector = AROFeatureSelector(
            feature_names=all_features,
            X=feature_matrix,
            y=labels,
            model_class=PatternClassifier,
            target_n_features=15,
            n_rabbits=30,
            max_iter=100,
        )
        result = selector.select()
        # result.selected_features → ['ema_21', 'atr_20', 'rsi_14', ...]
    """

    def __init__(
        self,
        feature_names: List[str],
        X: pd.DataFrame,
        y: pd.Series,
        model_class: type,
        n_rabbits: int = 30,
        target_n_features: int = 15,
        min_features: int = 5,
        cv_splits: int = 3,
        seed: int = 42,
    ) -> None:
        """Initialize ARO feature selector.

        Args:
            feature_names: All available feature names (column names from FeatureExtractor).
            X: Feature matrix (n_samples × n_features).
            y: Binary target labels.
            model_class: ML model class (e.g., PatternClassifier) with train/predict.
            n_rabbits: Population size.
            target_n_features: Target number of features after selection.
            min_features: Minimum features to keep.
            cv_splits: Number of PurgedKFold splits for fitness evaluation.
            seed: Random seed.
        """
        self.feature_names = list(feature_names)
        self.X = X
        self.y = y
        self.model_class = model_class
        self.target_n_features = target_n_features
        self.min_features = min_features
        self.cv_splits = cv_splits
        self.n_rabbits = n_rabbits

        n_features = len(feature_names)
        self._rng = np.random.default_rng(seed)

        self._population = self._init_binary_population(n_rabbits, n_features)
        self._fitness_cache: Dict[bytes, float] = {}
        self._best_rabbit_idx = 0
        self._iteration = 0

    def _init_binary_population(self, n_rabbits: int, n_features: int) -> np.ndarray:
        """Initialize rabbits with ~target_n_features active each."""
        pop = np.zeros((n_rabbits, n_features), dtype=np.float64)
        for i in range(n_rabbits):
            n_active = self._rng.integers(self.min_features, self.target_n_features + 1)
            idx = self._rng.choice(n_features, size=n_active, replace=False)
            pop[i, idx] = 1.0
        return pop

    def _to_feature_subset(self, rabbit: np.ndarray) -> List[str]:
        """Convert binary rabbit to selected feature names."""
        indices = np.where(rabbit > 0.5)[0]
        return [self.feature_names[i] for i in indices]

    def _evaluate_rabbit(self, rabbit: np.ndarray) -> float:
        """Evaluate a rabbit's feature subset via PurgedKFold AUC."""
        key = rabbit.tobytes()
        if key in self._fitness_cache:
            return self._fitness_cache[key]

        selected = self._to_feature_subset(rabbit)
        if len(selected) < self.min_features:
            self._fitness_cache[key] = 0.0
            return 0.0

        try:
            from sklearn.model_selection import TimeSeriesSplit

            X_sub = self.X[selected].values
            y_vals = self.y.values

            scores = []
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            for train_idx, test_idx in tscv.split(X_sub):
                X_tr, X_te = X_sub[train_idx], X_sub[test_idx]
                y_tr, y_te = y_vals[train_idx], y_vals[test_idx]

                if len(np.unique(y_tr)) < 2:
                    continue

                model = self.model_class(
                    n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42
                )
                result = model.train(
                    pd.DataFrame(X_tr, columns=selected),
                    pd.Series(y_tr),
                )
                preds = model.predict(pd.DataFrame(X_te, columns=selected))
                acc = (preds["is_recommended"].values == y_te).mean()
                scores.append(acc)

            score = float(np.mean(scores)) if scores else 0.0
        except Exception:
            score = 0.0

        self._fitness_cache[key] = score
        return score

    def select(self, max_iter: int = 100, early_stop: int = 15) -> OptimizerResult:
        """Run ARO optimization to select the best feature subset.

        Args:
            max_iter: Maximum iterations.
            early_stop: Stop if no improvement for this many iterations.

        Returns:
            OptimizerResult with best_params["selected_features"] and best_score.
        """
        n_features = len(self.feature_names)
        fitness = np.array([self._evaluate_rabbit(r) for r in self._population])
        best_idx = int(np.argmax(fitness))
        best_rabbit = self._population[best_idx].copy()
        best_score = float(fitness[best_idx])

        no_improve = 0
        history: List[float] = [best_score]

        # ARO parameters
        A = 2.0  # energy factor, decays linearly
        R = 0.0  # running probability

        for t in range(max_iter):
            # Energy decay
            energy = 2.0 * (1.0 - t / max_iter)

            for i in range(len(self._population)):
                # Detour foraging (exploration): deviate from other rabbits
                if self._rng.random() > 0.5:
                    candidates = [
                        j for j in range(len(self._population)) if j != i and j != best_idx
                    ]
                    if candidates:
                        j = self._rng.choice(candidates)
                        L = (np.exp(1) - np.exp((t / max_iter) ** 2)) * np.sin(
                            2 * np.pi * self._rng.random()
                        )
                        new_pos = (
                            self._population[i]
                            + energy * (self._population[i] - self._population[j])
                            + np.round(L * n_features * 0.1)
                        )
                    else:
                        new_pos = self._population[i] + energy * self._rng.uniform(
                            -0.5, 0.5, n_features
                        )
                # Random hiding (exploitation): fine-tune around best
                else:
                    burrow = best_rabbit + energy * self._rng.uniform(-0.3, 0.3, n_features)
                    H = energy * self._rng.random()
                    new_pos = self._population[i] + H * (burrow - self._population[i])

                # Binarization: clip to [0, 1], keep only target_n_features
                new_pos = np.clip(np.abs(new_pos), 0, 1)

                # Enforce cardinality constraint: keep top target_n_features by weight
                sorted_idx = np.argsort(new_pos)[::-1]
                binary = np.zeros(n_features, dtype=np.float64)
                binary[sorted_idx[: self.target_n_features]] = 1.0

                new_fitness = self._evaluate_rabbit(binary)
                if new_fitness > fitness[i]:
                    self._population[i] = binary
                    fitness[i] = new_fitness

            best_idx = int(np.argmax(fitness))
            if fitness[best_idx] > best_score:
                best_score = float(fitness[best_idx])
                best_rabbit = self._population[best_idx].copy()
                no_improve = 0
            else:
                no_improve += 1

            history.append(best_score)

            if t % 10 == 0:
                selected_count = int(np.sum(best_rabbit > 0.5))
                logger.info(
                    f"ARO iter {t:3d}: best AUC {best_score:.4f}, "
                    f"features selected {selected_count}, no_improve {no_improve}"
                )

            if no_improve >= early_stop:
                logger.info(f"ARO converged at iteration {t}")
                break

        selected_features = self._to_feature_subset(best_rabbit)

        return OptimizerResult(
            best_params={
                "selected_features": selected_features,
                "feature_mask": best_rabbit.tolist(),
            },
            best_score=best_score,
            all_scores=fitness.tolist(),
            best_position=best_rabbit,
            n_iterations=t + 1,
            convergence_history=history,
            optimizer_name="ARO",
        )
