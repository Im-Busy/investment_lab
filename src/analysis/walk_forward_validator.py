"""Walk-Forward Validator for Pattern Selection.

Implements rolling walk-forward analysis to detect overfitting by
comparing in-sample vs out-of-sample performance across windows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_DAYS = 252
DEFAULT_TEST_DAYS = 63
DEFAULT_OVERFITTING_DEGRADATION = 0.5


@dataclass
class WindowResult:
    """Result for a single walk-forward window."""

    window_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_metrics: Dict[str, float]
    oos_metrics: Dict[str, float]
    patterns: List[str]


@dataclass
class ValidationReport:
    """Aggregated walk-forward validation report."""

    windows: List[WindowResult]
    oos_aggregate: Dict[str, float]
    overfitting_detected: bool
    degradation_ratios: Dict[str, float]


class WalkForwardValidator:
    """Performs walk-forward analysis with configurable train/test windows."""

    def __init__(
        self,
        train_days: int = DEFAULT_TRAIN_DAYS,
        test_days: int = DEFAULT_TEST_DAYS,
        degradation_threshold: float = DEFAULT_OVERFITTING_DEGRADATION,
        metric_fn: Optional[Callable[[List[str], pd.DataFrame], Dict[str, float]]] = None,
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.degradation_threshold = degradation_threshold
        self.metric_fn = metric_fn

    def validate(
        self,
        patterns: List[str],
        data: pd.DataFrame,
        train_days: Optional[int] = None,
        test_days: Optional[int] = None,
    ) -> List[WindowResult]:
        """Run walk-forward validation across rolling windows.

        Args:
            patterns: List of pattern names to validate.
            data: OHLCV DataFrame with DatetimeIndex.
            train_days: Override training window size.
            test_days: Override test window size.

        Returns:
            List of WindowResult for each window.
        """
        td = train_days if train_days is not None else self.train_days
        ts = test_days if test_days is not None else self.test_days

        if self.metric_fn is None:
            raise ValueError("metric_fn must be set via constructor")

        total_days = len(data)
        window_span = td + ts
        if total_days < window_span:
            logger.warning(
                "Insufficient data for walk-forward: need %d days, have %d",
                window_span,
                total_days,
            )
            return []

        results = []
        idx = 0
        while idx + window_span <= total_days:
            train_end = idx + td
            test_end = train_end + ts

            train_data = data.iloc[idx:train_end]
            test_data = data.iloc[train_end:test_end]

            train_metrics = self.metric_fn(patterns, train_data)
            oos_metrics = self.metric_fn(patterns, test_data)

            wr = WindowResult(
                window_idx=idx,
                train_start=data.index[idx],
                train_end=data.index[train_end - 1],
                test_start=data.index[train_end],
                test_end=data.index[test_end - 1],
                train_metrics=train_metrics,
                oos_metrics=oos_metrics,
                patterns=patterns.copy(),
            )
            results.append(wr)

            idx += ts

        logger.info("Walk-forward: %d windows processed", len(results))
        return results

    def calculate_oos_metrics(self, wf_results: List[WindowResult]) -> Dict[str, float]:
        """Aggregate out-of-sample metrics across all windows.

        Args:
            wf_results: List of WindowResult from validate().

        Returns:
            {metric_name: mean_value} across windows.
        """
        if not wf_results:
            return {}

        all_metrics = {}
        for wr in wf_results:
            for key, val in wr.oos_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(val)

        return {key: float(np.mean(vals)) for key, vals in all_metrics.items()}

    def check_overfitting(
        self,
        train_metrics: List[Dict[str, float]],
        oos_metrics: Dict[str, float],
    ) -> bool:
        """Detect overfitting by comparing train vs OOS performance.

        Overfitting is flagged when OOS Sharpe degrades by more than
        the degradation_threshold relative to average train Sharpe.

        Args:
            train_metrics: List of per-window train metrics.
            oos_metrics: Aggregated OOS metrics.

        Returns:
            True if overfitting detected.
        """
        if not train_metrics or not oos_metrics:
            return False

        avg_train_sharpe = np.mean([m.get("sharpe_ratio", 0.0) for m in train_metrics])
        oos_sharpe = oos_metrics.get("sharpe_ratio", 0.0)

        if avg_train_sharpe <= 0:
            return oos_sharpe < 0

        degradation = (avg_train_sharpe - oos_sharpe) / abs(avg_train_sharpe)
        return degradation > self.degradation_threshold

    def generate_validation_report(self, wf_results: List[WindowResult]) -> str:
        """Generate a human-readable walk-forward validation report.

        Args:
            wf_results: List of WindowResult.

        Returns:
            Formatted report string.
        """
        if not wf_results:
            return "No walk-forward results available."

        oos_agg = self.calculate_oos_metrics(wf_results)
        train_metrics_list = [w.train_metrics for w in wf_results]
        is_overfit = self.check_overfitting(train_metrics_list, oos_agg)

        lines = [
            "=" * 70,
            "WALK-FORWARD VALIDATION REPORT",
            "=" * 70,
            f"Windows evaluated: {len(wf_results)}",
            f"Overfitting detected: {is_overfit}",
            "",
        ]

        lines.append("Out-of-Sample Aggregate Metrics:")
        for key, val in oos_agg.items():
            lines.append(f"  {key}: {val:.4f}")
        lines.append("")

        lines.append("Per-Window Details:")
        for w in wf_results:
            train_sharpe = w.train_metrics.get("sharpe_ratio", 0.0)
            oos_sharpe = w.oos_metrics.get("sharpe_ratio", 0.0)
            degradation = (
                (train_sharpe - oos_sharpe) / abs(train_sharpe)
                if train_sharpe != 0
                else float("inf")
            )
            lines.append(
                f"  Window {w.window_idx}: "
                f"Train Sharpe={train_sharpe:.4f}, "
                f"OOS Sharpe={oos_sharpe:.4f}, "
                f"Degradation={degradation:.2%}"
            )

        lines.append("")
        degradation_ratios = self._calc_degradation_ratios(wf_results)
        lines.append("Degradation Ratios by Metric:")
        for key, ratio in degradation_ratios.items():
            lines.append(f"  {key}: {ratio:.4f}")

        return "\n".join(lines)

    def _calc_degradation_ratios(self, wf_results: List[WindowResult]) -> Dict[str, float]:
        """Calculate per-metric degradation ratios."""
        ratios = {}
        if not wf_results:
            return ratios

        for key in wf_results[0].train_metrics:
            train_vals = [w.train_metrics.get(key, 0.0) for w in wf_results]
            oos_vals = [w.oos_metrics.get(key, 0.0) for w in wf_results]

            avg_train = float(np.mean(train_vals))
            avg_oos = float(np.mean(oos_vals))

            if abs(avg_train) > 1e-12:
                ratios[key] = (avg_train - avg_oos) / abs(avg_train)
            else:
                ratios[key] = 0.0

        return ratios
