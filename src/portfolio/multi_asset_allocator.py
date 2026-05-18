"""Correlation-aware multi-asset portfolio allocator.

Clusters instruments by rolling correlation, allocates equal weight to clusters
and signal-strength within clusters. Prevents concentration in correlated
instruments (e.g., all-tech SPY+QQQ+XLK).

Usage:
    from src.portfolio.multi_asset_allocator import MultiAssetAllocator

    allocator = MultiAssetAllocator(max_cluster_weight=0.30, corr_threshold=0.7)
    weights = allocator.allocate(
        signals={"SPY": 0.6, "QQQ": 0.8, "XLK": 0.7, "TLT": -0.3},
        returns_df=df,  # pd.DataFrame with instrument columns
    )
    # weights = {"SPY": 0.0, "QQQ": 0.0, "XLK": 0.15, "TLT": 0.20, ...}
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MultiAssetAllocator:
    """Correlation-aware multi-asset allocator.

    Parameters:
        max_cluster_weight: Maximum weight per correlation cluster (default 0.30).
        corr_threshold: Correlation threshold for clustering (default 0.7).
        corr_window: Rolling window for correlation computation (default 60).
        capital: Total capital to allocate (default 100_000).
        min_signal: Minimum absolute signal score to allocate (default 0.05).
    """

    def __init__(
        self,
        max_cluster_weight: float = 0.30,
        corr_threshold: float = 0.7,
        corr_window: int = 60,
        capital: float = 100_000.0,
        min_signal: float = 0.05,
    ) -> None:
        self.max_cluster_weight = max_cluster_weight
        self.corr_threshold = corr_threshold
        self.corr_window = corr_window
        self.capital = capital
        self.min_signal = min_signal

    def _build_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Build rolling correlation matrix from returns."""
        if len(returns_df) < self.corr_window:
            return returns_df.corr()

        tail = returns_df.tail(self.corr_window)
        return tail.corr()

    @staticmethod
    def _cluster_by_correlation(corr_matrix: pd.DataFrame, threshold: float) -> list[list[str]]:
        """Greedy clustering: group instruments with pairwise corr > threshold."""
        remaining = set(corr_matrix.columns)
        clusters: list[list[str]] = []

        while remaining:
            seed = remaining.pop()
            cluster = [seed]

            for other in list(remaining):
                if abs(corr_matrix.loc[seed, other]) > threshold:
                    cluster.append(other)
                    remaining.discard(other)

            clusters.append(cluster)

        return clusters

    def allocate(
        self,
        signals: dict[str, float],
        returns_df: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute allocation weights.

        Args:
            signals: Dict of instrument -> signal score (-1.0 to 1.0).
            returns_df: DataFrame with instrument return columns.

        Returns:
            Dict of instrument -> allocation weight (0.0 to 1.0).
        """
        instruments = [k for k in signals if k in returns_df.columns]
        if not instruments:
            return {}

        corr_matrix = self._build_correlation_matrix(returns_df[instruments])
        clusters = self._cluster_by_correlation(corr_matrix, self.corr_threshold)

        n_clusters = len(clusters)
        if n_clusters == 0:
            return {}

        base_weight_per_cluster = min(self.max_cluster_weight, 1.0 / n_clusters)

        weights: dict[str, float] = {}
        for cluster in clusters:
            cluster_instruments = [i for i in cluster if abs(signals.get(i, 0)) >= self.min_signal]
            if not cluster_instruments:
                continue

            cluster_weight = base_weight_per_cluster
            cluster_signals = {i: signals.get(i, 0.0) for i in cluster_instruments}
            total_abs_signal = sum(abs(s) for s in cluster_signals.values())

            if total_abs_signal > 0:
                for inst in cluster_instruments:
                    s = cluster_signals[inst]
                    weights[inst] = cluster_weight * (abs(s) / total_abs_signal)
            else:
                equal = cluster_weight / len(cluster_instruments)
                for inst in cluster_instruments:
                    weights[inst] = equal

        total = sum(weights.values())
        if total > 0 and total > 1.0:
            scale = 1.0 / total
            weights = {k: v * scale for k, v in weights.items()}

        return weights

    def get_clusters(self, returns_df: pd.DataFrame) -> dict[str, Any]:
        """Return clustering info for diagnostic display."""
        instruments = list(returns_df.columns)
        if len(instruments) < 2:
            return {"n_clusters": 1, "clusters": [instruments]}

        corr_matrix = self._build_correlation_matrix(returns_df[instruments])
        clusters = self._cluster_by_correlation(corr_matrix, self.corr_threshold)

        return {
            "n_clusters": len(clusters),
            "clusters": clusters,
            "max_cluster_size": max(len(c) for c in clusters) if clusters else 0,
            "corr_matrix": corr_matrix.to_dict(),
        }
