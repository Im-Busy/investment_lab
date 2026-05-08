"""Statistical Significance Filter for Pattern Selection.

Provides statistical tests to validate pattern returns significance
and filter by performance thresholds (Sharpe > 0.5, PF > 1.2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

DEFAULT_P_VALUE_THRESHOLD = 0.05
DEFAULT_MIN_SHARPE = 0.5
DEFAULT_MIN_PROFIT_FACTOR = 1.2


@dataclass
class SignificanceResult:
    """Statistical significance result for a single pattern."""

    pattern_name: str
    mean_return: float
    std_return: float
    t_statistic: float
    p_value: float
    is_significant: bool
    sample_size: int


class StatisticalSignificanceFilter:
    """Filters patterns by statistical significance and performance thresholds."""

    def __init__(
        self,
        p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD,
        min_sharpe: float = DEFAULT_MIN_SHARPE,
        min_profit_factor: float = DEFAULT_MIN_PROFIT_FACTOR,
    ):
        self.p_value_threshold = p_value_threshold
        self.min_sharpe = min_sharpe
        self.min_profit_factor = min_profit_factor

    def test_pattern_significance(
        self, pattern_returns: np.ndarray, pattern_name: str = ""
    ) -> SignificanceResult:
        """One-sample t-test on pattern returns vs zero mean.

        Args:
            pattern_returns: Array of per-trade or per-bar returns.
            pattern_name: Identifier for the pattern.

        Returns:
            SignificanceResult with t-statistic and p-value.
        """
        returns = np.asarray(pattern_returns, dtype=np.float64)
        n = len(returns)

        if n < 2:
            return SignificanceResult(
                pattern_name=pattern_name,
                mean_return=float(np.mean(returns)) if n == 1 else 0.0,
                std_return=0.0,
                t_statistic=0.0,
                p_value=1.0,
                is_significant=False,
                sample_size=n,
            )

        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns, ddof=1))

        if std_ret < 1e-12:
            t_stat = float("inf") if mean_ret > 0 else 0.0
            p_val = 0.0 if mean_ret > 0 else 1.0
        else:
            t_stat, p_val = stats.ttest_1samp(returns, 0.0, nan_policy="omit")
            t_stat = float(t_stat)
            p_val = float(p_val)

        return SignificanceResult(
            pattern_name=pattern_name,
            mean_return=mean_ret,
            std_return=std_ret,
            t_statistic=t_stat,
            p_value=p_val,
            is_significant=p_val < self.p_value_threshold,
            sample_size=n,
        )

    def filter_by_performance(
        self,
        patterns_perf: Dict[str, Dict[str, float]],
        min_sharpe: Optional[float] = None,
        min_profit_factor: Optional[float] = None,
    ) -> List[str]:
        """Filter patterns meeting Sharpe and profit factor thresholds.

        Args:
            patterns_perf: {pattern_name: {"sharpe": X, "profit_factor": Y, ...}}
            min_sharpe: Override minimum Sharpe ratio.
            min_profit_factor: Override minimum profit factor.

        Returns:
            List of pattern names passing both thresholds.
        """
        threshold_sharpe = min_sharpe if min_sharpe is not None else self.min_sharpe
        threshold_pf = (
            min_profit_factor if min_profit_factor is not None else self.min_profit_factor
        )

        passed = []
        for name, metrics in patterns_perf.items():
            sharpe = metrics.get("sharpe", metrics.get("sharpe_ratio", 0.0))
            pf = metrics.get("profit_factor", 0.0)
            if sharpe >= threshold_sharpe and pf >= threshold_pf:
                passed.append(name)
        return passed

    def calculate_p_values(self, pattern_results: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate p-value per pattern from return arrays.

        Args:
            pattern_results: {pattern_name: returns_array}

        Returns:
            {pattern_name: p_value}
        """
        p_values = {}
        for name, returns in pattern_results.items():
            result = self.test_pattern_significance(returns, pattern_name=name)
            p_values[name] = result.p_value
        return p_values

    def filter_significant_patterns(
        self,
        pattern_results: Dict[str, np.ndarray],
        patterns_perf: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[str]:
        """Filter patterns passing both statistical significance and performance.

        Args:
            pattern_results: {pattern_name: returns_array}
            patterns_perf: Optional performance metrics dict.

        Returns:
            List of pattern names passing all filters.
        """
        significant = []
        for name, returns in pattern_results.items():
            sig = self.test_pattern_significance(returns, pattern_name=name)
            if sig.is_significant:
                significant.append(name)

        if patterns_perf is None:
            return significant

        perf_passed = self.filter_by_performance(patterns_perf)
        return [n for n in significant if n in perf_passed]
