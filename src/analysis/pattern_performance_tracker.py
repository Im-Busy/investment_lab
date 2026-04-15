"""
Pattern Performance Tracker for Rolling Metrics Dashboard

Provides rolling performance metrics for pattern selection tracking.
Tracks pattern performance over time with rolling windows for
up-to-date pattern quality assessment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class RollingMetrics:
    """Rolling performance metrics for a single pattern."""

    pattern_name: str
    window_size: int

    # Rolling window metrics
    rolling_win_rate: float
    rolling_sharpe_ratio: float
    rolling_profit_factor: float
    rolling_avg_return: float
    rolling_max_drawdown: float

    # Trade counts
    total_trades: int
    rolling_trades: int

    # Time period
    start_date: pd.Timestamp
    end_date: pd.Timestamp

    # Trend detection
    win_rate_trend: str  # "improving", "deteriorating", "stable"
    sharpe_trend: str  # "improving", "deteriorating", "stable"

    # Recency score (higher = more recent performance is strong)
    recency_score: float = 0.0


@dataclass
class PerformanceTrackerConfig:
    """Configuration for the performance tracker."""

    # Rolling window sizes (in number of trades, not bars)
    short_window: int = 20
    medium_window: int = 50
    long_window: int = 100

    # Trend detection thresholds
    trend_threshold: float = 0.05  # Minimum change to detect trend

    # Minimum trades before metrics are valid
    min_trades_for_metrics: int = 10


class PatternPerformanceTracker:
    """
    Tracks pattern performance over time with rolling windows.

    Provides up-to-date metrics for pattern selection by maintaining
    rolling performance statistics across different time windows.
    """

    def __init__(self, config: Optional[PerformanceTrackerConfig] = None):
        self.config = config or PerformanceTrackerConfig()
        self._pattern_data: Dict[str, pd.DataFrame] = {}

    def add_pattern_results(
        self,
        pattern_name: str,
        trade_results: pd.DataFrame,
    ) -> None:
        """
        Add trade results for a pattern.

        Args:
            pattern_name: Name of the pattern
            trade_results: DataFrame with columns:
                - entry_time (pd.Timestamp)
                - exit_time (pd.Timestamp)
                - pnl (float)
                - pnl_pct (float)
                - exit_reason (optional)
        """
        self._pattern_data[pattern_name] = trade_results.copy()

    def get_rolling_metrics(
        self,
        pattern_name: str,
        window_size: Optional[int] = None,
    ) -> Optional[RollingMetrics]:
        """
        Get rolling performance metrics for a pattern.

        Args:
            pattern_name: Pattern to get metrics for
            window_size: Optional override for window size (number of trades)

        Returns:
            RollingMetrics or None if insufficient data
        """
        if pattern_name not in self._pattern_data:
            return None

        data = self._pattern_data[pattern_name]
        if len(data) < self.config.min_trades_for_metrics:
            return None

        window = window_size or self.config.medium_window

        # Get most recent window of trades
        recent = data.tail(window)
        all_data = data

        # Calculate rolling metrics
        wins = (recent["pnl"] > 0).sum()
        rolling_wr = wins / len(recent) if len(recent) > 0 else 0.0

        # Simplified Sharpe
        if len(recent) > 1 and recent["pnl"].std() > 0:
            rolling_sharpe = recent["pnl"].mean() / recent["pnl"].std() * np.sqrt(252)
        else:
            rolling_sharpe = 0.0

        # Profit Factor
        gross_wins = recent[recent["pnl"] > 0]["pnl"].sum()
        gross_losses = abs(recent[recent["pnl"] < 0]["pnl"].sum())
        rolling_pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        # Average return
        rolling_avg_ret = recent["pnl_pct"].mean() if "pnl_pct" in recent.columns else 0.0

        # Max drawdown (simplified)
        cumulative = recent["pnl"].cumsum()
        rolling_max_dd = (cumulative.max() - cumulative.min()) if len(cumulative) > 0 else 0.0

        # Trend detection
        mid_point = len(recent) // 2
        if mid_point > 0:
            first_half_wr = (recent.iloc[:mid_point]["pnl"] > 0).sum() / mid_point
            second_half_wr = (recent.iloc[mid_point:]["pnl"] > 0).sum() / (len(recent) - mid_point)

            if second_half_wr - first_half_wr > self.config.trend_threshold:
                wr_trend = "improving"
            elif second_half_wr - first_half_wr < -self.config.trend_threshold:
                wr_trend = "deteriorating"
            else:
                wr_trend = "stable"
        else:
            wr_trend = "stable"

        # Sharpe trend
        first_half_sharpe = self._calc_simple_sharpe(recent.iloc[:mid_point])
        second_half_sharpe = self._calc_simple_sharpe(recent.iloc[mid_point:])

        if second_half_sharpe - first_half_sharpe > self.config.trend_threshold:
            sharpe_trend = "improving"
        elif second_half_sharpe - first_half_sharpe < -self.config.trend_threshold:
            sharpe_trend = "deteriorating"
        else:
            sharpe_trend = "stable"

        # Recency score (recent window vs all-time)
        all_wr = (all_data["pnl"] > 0).sum() / len(all_data) if len(all_data) > 0 else 0.0
        recency_score = rolling_wr - all_wr

        return RollingMetrics(
            pattern_name=pattern_name,
            window_size=window,
            rolling_win_rate=rolling_wr,
            rolling_sharpe_ratio=rolling_sharpe,
            rolling_profit_factor=rolling_pf if not np.isinf(rolling_pf) else 999.0,
            rolling_avg_return=rolling_avg_ret,
            rolling_max_drawdown=rolling_max_dd,
            total_trades=len(all_data),
            rolling_trades=len(recent),
            start_date=data["entry_time"].iloc[0] if "entry_time" in data.columns else pd.NaT,
            end_date=data["entry_time"].iloc[-1] if "entry_time" in data.columns else pd.NaT,
            win_rate_trend=wr_trend,
            sharpe_trend=sharpe_trend,
            recency_score=recency_score,
        )

    def get_all_pattern_summaries(
        self,
        patterns: Optional[List[str]] = None,
        window_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get rolling performance summaries for all tracked patterns.

        Args:
            patterns: Optional list of patterns to summarize
            window_size: Optional window size override

        Returns:
            DataFrame with rolling metrics for all patterns
        """
        pattern_list = patterns or list(self._pattern_data.keys())

        summaries = []
        for pname in pattern_list:
            metrics = self.get_rolling_metrics(pname, window_size)
            if metrics:
                summaries.append(
                    {
                        "pattern_name": metrics.pattern_name,
                        "rolling_win_rate": metrics.rolling_win_rate,
                        "rolling_sharpe": metrics.rolling_sharpe_ratio,
                        "rolling_profit_factor": metrics.rolling_profit_factor,
                        "rolling_avg_return": metrics.rolling_avg_return,
                        "rolling_max_drawdown": metrics.rolling_max_drawdown,
                        "rolling_trades": metrics.rolling_trades,
                        "total_trades": metrics.total_trades,
                        "win_rate_trend": metrics.win_rate_trend,
                        "sharpe_trend": metrics.sharpe_trend,
                        "recency_score": metrics.recency_score,
                    }
                )

        df = pd.DataFrame(summaries)
        if len(df) > 0:
            df = df.sort_values("rolling_sharpe", ascending=False)

        return df

    def detect_degrading_patterns(
        self,
        patterns: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Detect patterns with deteriorating performance.

        Args:
            patterns: Optional list of patterns to check

        Returns:
            List of pattern names showing deterioration
        """
        pattern_list = patterns or list(self._pattern_data.keys())
        degrading = []

        for pname in pattern_list:
            metrics = self.get_rolling_metrics(pname)
            if metrics and (
                metrics.win_rate_trend == "deteriorating" or metrics.sharpe_trend == "deteriorating"
            ):
                degrading.append(pname)

        return degrading

    def _calc_simple_sharpe(self, data: pd.DataFrame) -> float:
        """Calculate simple Sharpe ratio."""
        if "pnl" not in data.columns or len(data) < 2:
            return 0.0

        std = data["pnl"].std()
        if std == 0:
            return 0.0

        return data["pnl"].mean() / std * np.sqrt(252)

    def get_pattern_data(
        self,
        pattern_name: str,
    ) -> Optional[pd.DataFrame]:
        """
        Get raw trade data for a pattern.

        Args:
            pattern_name: Pattern to get data for

        Returns:
            Trade DataFrame or None if not found
        """
        return self._pattern_data.get(pattern_name)

    def list_tracked_patterns(self) -> List[str]:
        """Get list of all tracked pattern names."""
        return list(self._pattern_data.keys())
