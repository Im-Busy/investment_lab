"""
Trade Attributor - Layer 2 of Contribution Analysis

After a backtest completes, match each backtesting.py trade back to the signal log
to determine which patterns contributed to each trade.

Challenge: backtesting.py manages trades internally. We get trade objects via
self.results._trades with entry_time, exit_time, entry_price, exit_price, size, pl, pl_pct.
We need to match these to our signal log entries by timestamp.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .signal_event_log import SignalEventLog


@dataclass
class AttributedTrade:
    """A trade with full pattern attribution."""

    trade_id: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    duration: str
    direction: str  # "LONG" or "SHORT"
    contributing_patterns: List[str]  # patterns that fired at entry
    confluence_count: int  # how many patterns agreed
    avg_confidence: float  # average confidence of contributing patterns
    pattern_categories: List[str]  # categories of contributing patterns
    all_detections_at_entry: int  # total detections at entry bar (including sub-threshold)

    # Computed fields
    is_winner: bool = False
    holding_days: int = 0


class TradeAttributor:
    """Matches backtesting.py trades to signal event logs."""

    def __init__(
        self,
        signal_log: SignalEventLog,
        trades_df: pd.DataFrame,
        tolerance_bars: int = 1,
    ):
        """
        Args:
            signal_log: SignalEventLog from the backtest
            trades_df: DataFrame from BacktestPyRunner.get_trades()
            tolerance_bars: Max bars between signal and trade entry for matching
        """
        self.signal_log = signal_log
        self.trades_df = trades_df
        self.tolerance_bars = tolerance_bars
        self.attributed_trades: List[AttributedTrade] = []
        self._bar_timestamps: Dict[pd.Timestamp, int] = {}

        # Build timestamp to bar index mapping
        for event in signal_log.events:
            if event.timestamp not in self._bar_timestamps:
                self._bar_timestamps[event.timestamp] = event.bar_index

    def attribute_trades(self) -> List[AttributedTrade]:
        """
        Match trades to signal events by entry timestamp.

        Returns:
            List of AttributedTrade objects
        """
        self.attributed_trades = []

        if self.trades_df.empty:
            return self.attributed_trades

        for idx, trade_row in self.trades_df.iterrows():
            attributed_trade = self._attribute_single_trade(idx, trade_row)
            self.attributed_trades.append(attributed_trade)

        return self.attributed_trades

    def _attribute_single_trade(self, trade_id: int, trade_row: pd.Series) -> AttributedTrade:
        """
        Attribute a single trade to signal events.

        Args:
            trade_id: Trade identifier
            trade_row: Row from trades_df

        Returns:
            AttributedTrade object
        """
        entry_time = trade_row.get("entry_time", pd.NaT)
        exit_time = trade_row.get("exit_time", pd.NaT)
        entry_price = trade_row.get("entry_price", 0.0)
        exit_price = trade_row.get("exit_price", 0.0)
        size = trade_row.get("size", 0.0)
        pnl = trade_row.get("pl", 0.0)
        pnl_pct = trade_row.get("pl_pct", 0.0)
        direction = "LONG" if trade_row.get("is_long", True) else "SHORT"

        # Calculate duration
        duration = self._calculate_duration(entry_time, exit_time)

        # Find matching signal event
        bar_index = self._find_matching_bar(entry_time)

        # Get detections at that bar
        contributing_patterns = []
        confluence_count = 0
        avg_confidence = 0.0
        pattern_categories = []
        all_detections_at_entry = 0

        if bar_index is not None:
            events = self.signal_log.get_detections_for_bar(bar_index)
            all_detections_at_entry = len(events)

            # Get metadata for the bar
            metadata = self.signal_log.get_bar_metadata(bar_index)
            if metadata and metadata.get("passed_threshold", False):
                active_patterns = metadata.get("active_patterns", [])
                contributing_patterns = active_patterns
                confluence_count = metadata.get("confluence_count", len(active_patterns))

                # Calculate average confidence from contributing patterns
                confidences = []
                categories = set()
                for event in events:
                    if event.pattern_name in active_patterns:
                        confidences.append(event.confidence)
                        categories.add(event.pattern_category)

                avg_confidence = np.mean(confidences) if confidences else 0.0
                pattern_categories = sorted(categories)

        # Compute is_winner and holding_days
        is_winner = pnl > 0
        holding_days = self._calculate_holding_days(entry_time, exit_time)

        return AttributedTrade(
            trade_id=trade_id,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration=duration,
            direction=direction,
            contributing_patterns=contributing_patterns,
            confluence_count=confluence_count,
            avg_confidence=avg_confidence,
            pattern_categories=pattern_categories,
            all_detections_at_entry=all_detections_at_entry,
            is_winner=is_winner,
            holding_days=holding_days,
        )

    def _find_matching_bar(self, entry_time: pd.Timestamp) -> Optional[int]:
        """
        Find the closest bar in signal_log by timestamp.

        Args:
            entry_time: Trade entry timestamp

        Returns:
            Bar index or None if no match found
        """
        if pd.isna(entry_time):
            return None

        # Direct match
        if entry_time in self._bar_timestamps:
            return self._bar_timestamps[entry_time]

        # Find closest within tolerance
        best_bar = None
        best_diff = None

        for timestamp, bar_index in self._bar_timestamps.items():
            diff_val = timestamp - entry_time
            
            # Handle numpy.float64 case
            if isinstance(diff_val, (int, float, np.integer, np.floating)):
                diff = abs(float(diff_val))
            else:
                # Try to get total_seconds
                try:
                    diff = abs(diff_val.total_seconds())
                except AttributeError:
                    diff = abs(float(diff_val))
            
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_bar = bar_index

        # Check if within tolerance (convert bars to approximate seconds)
        # Assuming daily data, 1 bar = 1 day = 86400 seconds
        max_diff_seconds = self.tolerance_bars * 86400
        if best_diff is not None and best_diff <= max_diff_seconds:
            return best_bar

        return None

    def _calculate_duration(self, entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> str:
        """
        Calculate human-readable trade duration.

        Args:
            entry_time: Trade entry timestamp
            exit_time: Trade exit timestamp

        Returns:
            Duration string (e.g., "3 days", "2 hours")
        """
        if pd.isna(entry_time) or pd.isna(exit_time):
            return "Unknown"

        delta = exit_time - entry_time
        
        # Handle numpy.float64 case
        if isinstance(delta, (int, float, np.integer, np.floating)):
            # delta is a numeric value (likely seconds or days as float)
            if delta == 0:
                return "0 seconds"
            elif abs(delta) < 1:
                # Likely in days
                days = int(delta * 86400)  # Convert to seconds
                if days == 0:
                    return "<1 day"
                elif days == 1:
                    return "1 day"
                else:
                    return f"{days} days"
            else:
                # Likely already in days or seconds
                days = int(delta)
                if days == 0:
                    return "<1 day"
                elif days == 1:
                    return "1 day"
                else:
                    return f"{days} days"
        
        # Handle timedelta case
        try:
            days = delta.days
            if days == 0:
                hours = delta.seconds // 3600
                if hours == 0:
                    minutes = delta.seconds // 60
                    return f"{minutes} minutes"
                return f"{hours} hours"
            elif days == 1:
                return "1 day"
            else:
                return f"{days} days"
        except AttributeError:
            return "Unknown"

    def _calculate_holding_days(self, entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> int:
        """
        Calculate holding period in days.

        Args:
            entry_time: Trade entry timestamp
            exit_time: Trade exit timestamp

        Returns:
            Number of days
        """
        if pd.isna(entry_time) or pd.isna(exit_time):
            return 0

        delta = exit_time - entry_time
        
        # Handle numpy.float64 case
        if isinstance(delta, (int, float, np.integer, np.floating)):
            # delta is a numeric value (likely in days)
            return max(0, int(delta))
        
        # Handle timedelta case
        try:
            return max(0, delta.days)
        except AttributeError:
            return 0

    def get_pattern_trade_stats(self) -> pd.DataFrame:
        """
        For each pattern: number of trades it participated in,
        win rate when present, avg P&L when present.

        Returns:
            DataFrame with pattern statistics
        """
        if not self.attributed_trades:
            return pd.DataFrame()

        pattern_stats: Dict[str, Dict[str, Any]] = {}

        for trade in self.attributed_trades:
            for pattern in trade.contributing_patterns:
                if pattern not in pattern_stats:
                    pattern_stats[pattern] = {
                        "pattern_name": pattern,
                        "trade_count": 0,
                        "win_count": 0,
                        "total_pnl": 0.0,
                        "total_pnl_pct": 0.0,
                    }

                stats = pattern_stats[pattern]
                stats["trade_count"] += 1
                if trade.is_winner:
                    stats["win_count"] += 1
                stats["total_pnl"] += trade.pnl
                stats["total_pnl_pct"] += trade.pnl_pct

        # Convert to DataFrame
        result = []
        for pattern, stats in pattern_stats.items():
            trade_count = stats["trade_count"]
            win_count = stats["win_count"]
            result.append(
                {
                    "pattern_name": pattern,
                    "trade_count": trade_count,
                    "win_count": win_count,
                    "win_rate": win_count / trade_count if trade_count > 0 else 0.0,
                    "avg_pnl": stats["total_pnl"] / trade_count if trade_count > 0 else 0.0,
                    "avg_pnl_pct": stats["total_pnl_pct"] / trade_count if trade_count > 0 else 0.0,
                    "total_pnl": stats["total_pnl"],
                }
            )

        df = pd.DataFrame(result)
        if not df.empty:
            df = df.sort_values("trade_count", ascending=False).reset_index(drop=True)
        return df

    def get_pattern_combination_stats(self) -> pd.DataFrame:
        """
        For each unique combination of patterns: how many trades,
        win rate, avg P&L. Sorted by frequency.

        Returns:
            DataFrame with pattern combination statistics
        """
        if not self.attributed_trades:
            return pd.DataFrame()

        combo_stats: Dict[str, Dict[str, Any]] = {}

        for trade in self.attributed_trades:
            if not trade.contributing_patterns:
                continue

            # Create sorted combination key
            combo_key = ",".join(sorted(trade.contributing_patterns))

            if combo_key not in combo_stats:
                combo_stats[combo_key] = {
                    "pattern_combo": combo_key,
                    "trade_count": 0,
                    "win_count": 0,
                    "total_pnl": 0.0,
                    "total_pnl_pct": 0.0,
                }

            stats = combo_stats[combo_key]
            stats["trade_count"] += 1
            if trade.is_winner:
                stats["win_count"] += 1
            stats["total_pnl"] += trade.pnl
            stats["total_pnl_pct"] += trade.pnl_pct

        # Convert to DataFrame
        result = []
        for combo, stats in combo_stats.items():
            trade_count = stats["trade_count"]
            win_count = stats["win_count"]
            result.append(
                {
                    "pattern_combo": combo,
                    "trade_count": trade_count,
                    "win_count": win_count,
                    "win_rate": win_count / trade_count if trade_count > 0 else 0.0,
                    "avg_pnl": stats["total_pnl"] / trade_count if trade_count > 0 else 0.0,
                    "avg_pnl_pct": stats["total_pnl_pct"] / trade_count if trade_count > 0 else 0.0,
                    "total_pnl": stats["total_pnl"],
                }
            )

        df = pd.DataFrame(result)
        if not df.empty:
            df = df.sort_values("trade_count", ascending=False).reset_index(drop=True)
        return df

    def get_best_trade_recipes(self, n: int = 10) -> pd.DataFrame:
        """
        Top N trades by P&L with their pattern recipes.

        Args:
            n: Number of top trades to return

        Returns:
            DataFrame with top trades
        """
        if not self.attributed_trades:
            return pd.DataFrame()

        # Sort by P&L descending
        sorted_trades = sorted(self.attributed_trades, key=lambda t: t.pnl, reverse=True)
        top_trades = sorted_trades[:n]

        result = []
        for trade in top_trades:
            result.append(
                {
                    "trade_id": trade.trade_id,
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "direction": trade.direction,
                    "pnl": trade.pnl,
                    "pnl_pct": trade.pnl_pct,
                    "duration": trade.duration,
                    "contributing_patterns": ", ".join(trade.contributing_patterns),
                    "confluence_count": trade.confluence_count,
                    "avg_confidence": trade.avg_confidence,
                }
            )

        return pd.DataFrame(result)

    def get_worst_trade_recipes(self, n: int = 10) -> pd.DataFrame:
        """
        Bottom N trades by P&L with their pattern recipes.

        Args:
            n: Number of worst trades to return

        Returns:
            DataFrame with worst trades
        """
        if not self.attributed_trades:
            return pd.DataFrame()

        # Sort by P&L ascending
        sorted_trades = sorted(self.attributed_trades, key=lambda t: t.pnl)
        worst_trades = sorted_trades[:n]

        result = []
        for trade in worst_trades:
            result.append(
                {
                    "trade_id": trade.trade_id,
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "direction": trade.direction,
                    "pnl": trade.pnl,
                    "pnl_pct": trade.pnl_pct,
                    "duration": trade.duration,
                    "contributing_patterns": ", ".join(trade.contributing_patterns),
                    "confluence_count": trade.confluence_count,
                    "avg_confidence": trade.avg_confidence,
                }
            )

        return pd.DataFrame(result)

    def get_confluence_vs_performance(self) -> pd.DataFrame:
        """
        Group trades by confluence_count (1, 2, 3, 4, 5+).
        For each group: count, win rate, avg P&L, avg confidence.

        Returns:
            DataFrame with confluence performance
        """
        if not self.attributed_trades:
            return pd.DataFrame()

        # Group by confluence count
        groups: Dict[int, List[AttributedTrade]] = {}
        for trade in self.attributed_trades:
            count = trade.confluence_count
            if count >= 5:
                count = 5  # Group 5+ together
            if count not in groups:
                groups[count] = []
            groups[count].append(trade)

        result = []
        for count in sorted(groups.keys()):
            trades = groups[count]
            trade_count = len(trades)
            win_count = sum(1 for t in trades if t.is_winner)
            total_pnl = sum(t.pnl for t in trades)
            total_pnl_pct = sum(t.pnl_pct for t in trades)
            avg_confidence = np.mean([t.avg_confidence for t in trades]) if trades else 0.0

            label = f"{count}+" if count == 5 else str(count)

            result.append(
                {
                    "confluence_count": label,
                    "trade_count": trade_count,
                    "win_count": win_count,
                    "win_rate": win_count / trade_count if trade_count > 0 else 0.0,
                    "avg_pnl": total_pnl / trade_count if trade_count > 0 else 0.0,
                    "avg_pnl_pct": total_pnl_pct / trade_count if trade_count > 0 else 0.0,
                    "avg_confidence": avg_confidence,
                }
            )

        return pd.DataFrame(result)

    def get_pattern_participation_rate(self) -> pd.DataFrame:
        """
        For each pattern: what % of all trades included it?
        What % of winning trades included it?
        What % of losing trades included it?

        Returns:
            DataFrame with participation rates
        """
        if not self.attributed_trades:
            return pd.DataFrame()

        total_trades = len(self.attributed_trades)
        winning_trades = [t for t in self.attributed_trades if t.is_winner]
        losing_trades = [t for t in self.attributed_trades if not t.is_winner]

        total_winners = len(winning_trades)
        total_losers = len(losing_trades)

        # Count participation
        pattern_counts: Dict[str, Dict[str, int]] = {}

        for trade in self.attributed_trades:
            for pattern in trade.contributing_patterns:
                if pattern not in pattern_counts:
                    pattern_counts[pattern] = {
                        "total": 0,
                        "winners": 0,
                        "losers": 0,
                    }

                pattern_counts[pattern]["total"] += 1
                if trade.is_winner:
                    pattern_counts[pattern]["winners"] += 1
                else:
                    pattern_counts[pattern]["losers"] += 1

        result = []
        for pattern, counts in pattern_counts.items():
            result.append(
                {
                    "pattern_name": pattern,
                    "participation_rate": counts["total"] / total_trades if total_trades > 0 else 0.0,
                    "winner_participation_rate": counts["winners"] / total_winners if total_winners > 0 else 0.0,
                    "loser_participation_rate": counts["losers"] / total_losers if total_losers > 0 else 0.0,
                    "total_trades": counts["total"],
                    "winner_trades": counts["winners"],
                    "loser_trades": counts["losers"],
                }
            )

        df = pd.DataFrame(result)
        if not df.empty:
            df = df.sort_values("participation_rate", ascending=False).reset_index(drop=True)
        return df

    def get_unattributed_trades(self) -> pd.DataFrame:
        """
        Get trades that couldn't be attributed to any signal.

        Returns:
            DataFrame with unattributed trades
        """
        if not self.attributed_trades:
            return pd.DataFrame()

        unattributed = [t for t in self.attributed_trades if not t.contributing_patterns]

        result = []
        for trade in unattributed:
            result.append(
                {
                    "trade_id": trade.trade_id,
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "direction": trade.direction,
                    "pnl": trade.pnl,
                    "pnl_pct": trade.pnl_pct,
                    "duration": trade.duration,
                }
            )

        return pd.DataFrame(result)

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for trade attribution.

        Returns:
            Dictionary with summary statistics
        """
        if not self.attributed_trades:
            return {
                "total_trades": 0,
                "attributed_trades": 0,
                "unattributed_trades": 0,
                "attribution_rate": 0.0,
            }

        total = len(self.attributed_trades)
        attributed = sum(1 for t in self.attributed_trades if t.contributing_patterns)
        unattributed = total - attributed

        return {
            "total_trades": total,
            "attributed_trades": attributed,
            "unattributed_trades": unattributed,
            "attribution_rate": attributed / total if total > 0 else 0.0,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert attributed trades to DataFrame.

        Returns:
            DataFrame with all attributed trades
        """
        if not self.attributed_trades:
            return pd.DataFrame()

        data = []
        for trade in self.attributed_trades:
            data.append(
                {
                    "trade_id": trade.trade_id,
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "size": trade.size,
                    "pnl": trade.pnl,
                    "pnl_pct": trade.pnl_pct,
                    "duration": trade.duration,
                    "direction": trade.direction,
                    "contributing_patterns": ", ".join(trade.contributing_patterns),
                    "confluence_count": trade.confluence_count,
                    "avg_confidence": trade.avg_confidence,
                    "pattern_categories": ", ".join(trade.pattern_categories),
                    "all_detections_at_entry": trade.all_detections_at_entry,
                    "is_winner": trade.is_winner,
                    "holding_days": trade.holding_days,
                }
            )

        return pd.DataFrame(data)

    def __len__(self) -> int:
        """Return number of attributed trades."""
        return len(self.attributed_trades)

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_summary_stats()
        return (
            f"TradeAttributor("
            f"total={stats['total_trades']}, "
            f"attributed={stats['attributed_trades']}, "
            f"rate={stats['attribution_rate']:.1%})"
        )
