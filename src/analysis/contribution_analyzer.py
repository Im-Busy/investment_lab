"""Contribution Analyzer (Leave-One-Out Ablation) for Pattern Selection.

Measures marginal contribution of each pattern by removing it and
observing the impact on portfolio metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_REDUNDANCY_THRESHOLD = 0.01


@dataclass
class AblationResult:
    """Result of ablating a single pattern."""

    pattern_name: str
    base_metrics: Dict[str, float]
    without_metrics: Dict[str, float]
    marginal_contribution: Dict[str, float]
    is_redundant: bool = False


class ContributionAnalyzer:
    """Performs leave-one-out ablation to identify pattern contributions."""

    def __init__(
        self,
        redundancy_threshold: float = DEFAULT_REDUNDANCY_THRESHOLD,
        metric_backtest_fn: Optional[Callable[[List[str]], Dict[str, float]]] = None,
    ):
        self.redundancy_threshold = redundancy_threshold
        self.metric_backtest_fn = metric_backtest_fn

    def leave_one_out_backtest(
        self,
        all_patterns: List[str],
        data: Optional[pd.DataFrame] = None,
        base_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, AblationResult]:
        """Run leave-one-out backtest for each pattern.

        Args:
            all_patterns: List of all pattern names.
            data: Optional DataFrame (used if metric_backtest_fn is None).
            base_metrics: Pre-computed baseline metrics (if available).

        Returns:
            {pattern_name: AblationResult}
        """
        if self.metric_backtest_fn is None:
            raise ValueError("metric_backtest_fn must be provided or base_metrics must be pre-computed")

        if base_metrics is None:
            base_metrics = self.metric_backtest_fn(all_patterns)

        results: Dict[str, AblationResult] = {}

        for pattern in all_patterns:
            remaining = [p for p in all_patterns if p != pattern]
            if not remaining:
                without_metrics = {k: 0.0 for k in base_metrics}
            else:
                without_metrics = self.metric_backtest_fn(remaining)

            marginal = self.calculate_marginal_contribution(pattern, base_metrics, without_metrics)
            is_redundant = all(
                abs(v) < self.redundancy_threshold for v in marginal.values()
            )

            results[pattern] = AblationResult(
                pattern_name=pattern,
                base_metrics=base_metrics,
                without_metrics=without_metrics,
                marginal_contribution=marginal,
                is_redundant=is_redundant,
            )
            logger.info(
                "Ablation '%s': marginal_contribution=%s, redundant=%s",
                pattern,
                {k: f"{v:.4f}" for k, v in marginal.items()},
                is_redundant,
            )

        return results

    def calculate_marginal_contribution(
        self,
        pattern_name: str,
        base_metrics: Dict[str, float],
        without_metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate marginal contribution of a pattern.

        Marginal = base_metric - without_pattern_metric.

        Args:
            pattern_name: Pattern being ablated.
            base_metrics: Metrics with all patterns.
            without_metrics: Metrics without this pattern.

        Returns:
            {metric_name: marginal_contribution}
        """
        marginal = {}
        for key in base_metrics:
            base_val = base_metrics.get(key, 0.0)
            without_val = without_metrics.get(key, 0.0)
            marginal[key] = base_val - without_val
        return marginal

    def rank_patterns_by_contribution(
        self,
        ablation_results: Dict[str, AblationResult],
        rank_metric: str = "sharpe_ratio",
    ) -> List[Dict[str, Any]]:
        """Rank patterns by their marginal contribution.

        Args:
            ablation_results: Results from leave_one_out_backtest.
            rank_metric: Metric to rank by.

        Returns:
            List of {pattern_name, rank, contribution} dicts sorted by contribution.
        """
        contributions = []
        for name, result in ablation_results.items():
            contrib = result.marginal_contribution.get(rank_metric, 0.0)
            contributions.append(
                {
                    "pattern_name": name,
                    "contribution": contrib,
                    "is_redundant": result.is_redundant,
                }
            )

        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        for i, item in enumerate(contributions):
            item["rank"] = i + 1

        return contributions

    def identify_redundant_patterns(
        self,
        ablation_results: Dict[str, AblationResult],
        threshold: Optional[float] = None,
    ) -> List[str]:
        """Identify patterns whose removal barely affects metrics.

        Args:
            ablation_results: Results from leave_one_out_backtest.
            threshold: Max absolute marginal change to consider redundant.

        Returns:
            List of redundant pattern names.
        """
        thresh = threshold if threshold is not None else self.redundancy_threshold
        redundant = []
        for name, result in ablation_results.items():
            max_change = max(abs(v) for v in result.marginal_contribution.values()) if result.marginal_contribution else 0.0
            if max_change < thresh:
                redundant.append(name)
        return redundant
