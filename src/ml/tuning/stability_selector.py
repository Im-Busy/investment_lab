"""Stability Selection for robust feature selection.

Implements Meinshausen & Buehlmann (2010, JRSS-B): bootstrapped feature importance
aggregation eliminates selection instability in small-N, large-P settings.

Core algorithm:
1. Bootstrap N times, each time:
   - Sample a random subset of rows (with replacement)
   - Train a CatBoost model on the subset
   - Record feature importance rankings
2. For each feature, compute stability score = proportion of bootstraps
   where the feature appears in the top-k
3. Select features with stability_score >= threshold (default 0.6)

Reference:
    Meinshausen, N. & Buehlmann, P. (2010). Stability selection.
    Journal of the Royal Statistical Society: Series B, 72(4), 417-473.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StabilityResult:
    selected_features: list[str]
    stability_scores: dict[str, float]
    n_bootstraps: int
    threshold: float
    sample_fraction: float
    n_original_features: int
    n_selected_features: int
    mean_features_per_bootstrap: float
    top_features: list[tuple[str, float]] = field(default_factory=list)


class StabilitySelector:
    """Bootstrapped stability-based feature selector.

    Replaces ARO metaheuristic selector which collapses to ~5 features
    and overfits to cross-asset leakage features.

    Usage:
        selector = StabilitySelector(
            n_bootstraps=100,
            threshold=0.6,
            sample_fraction=0.8,
            top_k_fraction=0.5,
            random_state=42,
        )
        result = selector.select(X, y, feature_names)
        # result.selected_features -> list of stable feature names
        # result.stability_scores -> {feature: stability_score}
    """

    def __init__(
        self,
        n_bootstraps: int = 100,
        threshold: float = 0.6,
        sample_fraction: float = 0.8,
        top_k_fraction: float = 0.5,
        catboost_params: dict[str, Any] | None = None,
        random_state: int = 42,
    ) -> None:
        self.n_bootstraps = n_bootstraps
        self.threshold = threshold
        self.sample_fraction = sample_fraction
        self.top_k_fraction = top_k_fraction
        self.catboost_params = catboost_params or {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "l2_leaf_reg": 5.0,
            "random_strength": 2.0,
            "subsample": 0.8,
            "colsample_bylevel": 0.8,
            "min_data_in_leaf": 30,
            "verbose": 0,
        }
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str] | None = None,
    ) -> StabilityResult:
        """Run stability selection and return stable features.

        Args:
            X: Feature matrix (n_samples x n_features).
            y: Target labels (binary classification).
            feature_names: Column names. Defaults to X.columns.

        Returns:
            StabilityResult with selected features and diagnostics.
        """
        from catboost import CatBoostClassifier

        if feature_names is None:
            feature_names = list(X.columns)

        n_features = len(feature_names)
        n_samples = len(X)
        top_k = max(int(n_features * self.top_k_fraction), 5)

        X_arr = X[feature_names].fillna(0).values.astype(np.float64)
        y_arr = y.values.astype(np.float64)

        selection_counts = np.zeros(n_features, dtype=np.int32)
        bootstrap_feature_counts: list[int] = []

        for b in range(self.n_bootstraps):
            sample_size = max(int(n_samples * self.sample_fraction), 200)
            indices = self._rng.choice(n_samples, size=sample_size, replace=True)
            X_boot, y_boot = X_arr[indices], y_arr[indices]

            unique_classes = np.unique(y_boot)
            if len(unique_classes) < 2:
                continue

            try:
                model = CatBoostClassifier(
                    random_seed=self._rng.integers(0, 2**31),
                    **self.catboost_params,
                )
                model.fit(X_boot, y_boot, verbose=False)

                importances = model.get_feature_importance()
                top_indices = np.argsort(importances)[::-1][:top_k]
                selection_counts[top_indices] += 1
                bootstrap_feature_counts.append(len(top_indices))
            except Exception:
                continue

            if (b + 1) % 25 == 0:
                logger.info(
                    f"Stability selection: {b + 1}/{self.n_bootstraps} bootstraps, "
                    f"top features so far: "
                    f"{self._top_stable_features(selection_counts, feature_names, 5)}"
                )

        stability_scores = {
            name: float(selection_counts[i]) / max(self.n_bootstraps, 1)
            for i, name in enumerate(feature_names)
        }

        selected = [name for name, score in stability_scores.items() if score >= self.threshold]
        selected.sort(key=lambda n: stability_scores[n], reverse=True)

        sorted_scores = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)

        logger.info(
            f"Stability selection: {len(selected)}/{n_features} features selected "
            f"(threshold={self.threshold:.2f}, bootstraps={self.n_bootstraps})"
        )
        logger.info(f"Top 10 by stability: {[(n, f'{s:.3f}') for n, s in sorted_scores[:10]]}")

        cross_asset_count = sum(
            1
            for f in selected
            if any(prefix in f for prefix in ("spy_", "qqq_", "tlt_", "gld_", "xlk_"))
        )
        if cross_asset_count > 0:
            logger.info(
                f"Cross-asset features in selection: {cross_asset_count} "
                f"({cross_asset_count / max(len(selected), 1) * 100:.1f}%)"
            )

        mean_bootstrap = (
            float(np.mean(bootstrap_feature_counts)) if bootstrap_feature_counts else 0.0
        )

        return StabilityResult(
            selected_features=selected,
            stability_scores=stability_scores,
            n_bootstraps=self.n_bootstraps,
            threshold=self.threshold,
            sample_fraction=self.sample_fraction,
            n_original_features=n_features,
            n_selected_features=len(selected),
            mean_features_per_bootstrap=mean_bootstrap,
            top_features=sorted_scores[:15],
        )

    @staticmethod
    def _top_stable_features(
        counts: np.ndarray,
        names: list[str],
        n: int,
    ) -> list[str]:
        indices = np.argsort(counts)[::-1][:n]
        return [names[i] for i in indices if counts[i] > 0]
