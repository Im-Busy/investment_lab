"""Signal Quality Filter for Pattern Selection.

Evaluates pattern signal quality using win rate, signal count,
and composite scoring to filter out unreliable patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MIN_WIN_RATE = 0.45
DEFAULT_MIN_SIGNALS = 30


@dataclass
class QualityMetrics:
    """Quality metrics for a single pattern."""

    pattern_name: str
    total_signals: int
    win_rate: float
    avg_return: float
    std_return: float
    sharpe_estimate: float
    max_consecutive_losses: int
    signal_consistency: float


@dataclass
class QualityScore:
    """Composite quality score for a pattern."""

    pattern_name: str
    win_rate_score: float
    signal_count_score: float
    consistency_score: float
    composite_score: float
    passes: bool


class SignalQualityFilter:
    """Filters patterns by signal quality metrics."""

    def __init__(
        self,
        min_win_rate: float = DEFAULT_MIN_WIN_RATE,
        min_signals: int = DEFAULT_MIN_SIGNALS,
    ):
        self.min_win_rate = min_win_rate
        self.min_signals = min_signals

    def calculate_signal_quality(self, pattern_signals: Dict[str, List[float]]) -> Dict[str, QualityMetrics]:
        """Calculate quality metrics from per-pattern signal returns.

        Args:
            pattern_signals: {pattern_name: [returns_per_signal]}

        Returns:
            {pattern_name: QualityMetrics}
        """
        results = {}
        for name, returns in pattern_signals.items():
            ret_arr = np.asarray(returns, dtype=np.float64)
            n = len(ret_arr)

            if n == 0:
                results[name] = QualityMetrics(
                    pattern_name=name,
                    total_signals=0,
                    win_rate=0.0,
                    avg_return=0.0,
                    std_return=0.0,
                    sharpe_estimate=0.0,
                    max_consecutive_losses=0,
                    signal_consistency=0.0,
                )
                continue

            wins = (ret_arr > 0).sum()
            win_rate = float(wins / n)
            avg_ret = float(np.mean(ret_arr))
            std_ret = float(np.std(ret_arr, ddof=1)) if n > 1 else 0.0

            sharpe = avg_ret / std_ret if std_ret > 1e-12 else 0.0
            max_cons_loss = self._max_consecutive_loss(ret_arr)
            consistency = self._signal_consistency(ret_arr)

            results[name] = QualityMetrics(
                pattern_name=name,
                total_signals=n,
                win_rate=win_rate,
                avg_return=avg_ret,
                std_return=std_ret,
                sharpe_estimate=sharpe,
                max_consecutive_losses=max_cons_loss,
                signal_consistency=consistency,
            )

        return results

    def filter_by_win_rate(
        self,
        patterns_quality: Dict[str, QualityMetrics],
        min_win_rate: Optional[float] = None,
    ) -> List[str]:
        """Filter patterns meeting minimum win rate threshold.

        Args:
            patterns_quality: {pattern_name: QualityMetrics}
            min_win_rate: Override minimum win rate.

        Returns:
            List of pattern names passing the threshold.
        """
        threshold = min_win_rate if min_win_rate is not None else self.min_win_rate
        return [
            name for name, qm in patterns_quality.items()
            if qm.win_rate >= threshold
        ]

    def filter_by_signal_count(
        self,
        patterns_quality: Dict[str, QualityMetrics],
        min_signals: Optional[int] = None,
    ) -> List[str]:
        """Filter patterns meeting minimum signal count.

        Args:
            patterns_quality: {pattern_name: QualityMetrics}
            min_signals: Override minimum signal count.

        Returns:
            List of pattern names passing the threshold.
        """
        threshold = min_signals if min_signals is not None else self.min_signals
        return [
            name for name, qm in patterns_quality.items()
            if qm.total_signals >= threshold
        ]

    def score_signal_quality(self, quality_metrics: QualityMetrics) -> float:
        """Compute composite quality score (0.0 to 1.0).

        Score = 0.4 * win_rate + 0.3 * signal_density + 0.3 * consistency.

        Args:
            quality_metrics: QualityMetrics for a pattern.

        Returns:
            Composite score in [0, 1].
        """
        wr_score = quality_metrics.win_rate

        signal_density = min(quality_metrics.total_signals / max(self.min_signals, 1), 1.0)

        consistency = quality_metrics.signal_consistency

        return 0.4 * wr_score + 0.3 * signal_density + 0.3 * consistency

    def apply_all_filters(
        self,
        pattern_signals: Dict[str, List[float]],
        min_win_rate: Optional[float] = None,
        min_signals: Optional[int] = None,
        min_composite_score: float = 0.4,
    ) -> List[str]:
        """Apply all quality filters and return passing patterns.

        Pipeline: win_rate -> signal_count -> composite_score.

        Args:
            pattern_signals: {pattern_name: [returns_per_signal]}
            min_win_rate: Override minimum win rate.
            min_signals: Override minimum signal count.
            min_composite_score: Minimum composite quality score.

        Returns:
            List of pattern names passing all filters.
        """
        quality = self.calculate_signal_quality(pattern_signals)
        wr_passed = set(self.filter_by_win_rate(quality, min_win_rate))
        sc_passed = set(self.filter_by_signal_count(quality, min_signals))

        candidate_names = wr_passed & sc_passed
        final = []
        for name in candidate_names:
            score = self.score_signal_quality(quality[name])
            if score >= min_composite_score:
                final.append(name)

        return sorted(final)

    @staticmethod
    def _max_consecutive_loss(returns: np.ndarray) -> int:
        """Count maximum consecutive losing signals."""
        max_count = 0
        current = 0
        for r in returns:
            if r <= 0:
                current += 1
                max_count = max(max_count, current)
            else:
                current = 0
        return max_count

    @staticmethod
    def _signal_consistency(returns: np.ndarray) -> float:
        """Measure signal consistency via rolling win rate stability.

        Returns a value in [0, 1] where higher means more consistent.
        """
        n = len(returns)
        if n < 10:
            return 0.0

        window = min(30, n // 2)
        rolling_wr = []
        for i in range(n - window + 1):
            wr = float(np.mean(returns[i : i + window] > 0))
            rolling_wr.append(wr)

        if not rolling_wr:
            return 0.0

        std_wr = float(np.std(rolling_wr))
        consistency = max(0.0, 1.0 - std_wr * 2)
        return consistency
