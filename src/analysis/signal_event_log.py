"""
Signal Event Log - Layer 1 of Contribution Analysis

Records every pattern detection event across every bar - not just the ones that passed the confluence threshold.
This is the foundation for all subsequent analysis.

Key Insight: The _pattern_signals_cache in MultiPatternStrategyOptimized already contains pre-computed
signals for all patterns across all bars. We just need to read it out. Zero additional computation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class SignalEvent:
    """A single pattern detection event on a single bar."""

    bar_index: int
    timestamp: pd.Timestamp
    pattern_name: str
    pattern_category: str  # basic, harmonic, complex, classic, continuation, breakout, candlestick
    direction: str  # "Long", "Short"
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    pattern_type: str  # Reversal, Continuation, Breakout, etc.

    # Post-backtest fields (filled later by TradeAttributor)
    led_to_trade: bool = False
    trade_pnl: Optional[float] = None
    trade_pnl_pct: Optional[float] = None


class SignalEventLog:
    """Comprehensive log of ALL pattern detection events."""

    def __init__(self):
        self.events: List[SignalEvent] = []
        self._bar_timestamps: Optional[pd.DatetimeIndex] = None
        self._events_by_bar: Dict[int, List[SignalEvent]] = {}
        self._events_by_pattern: Dict[str, List[SignalEvent]] = {}
        self._bar_metadata: Dict[int, Dict[str, Any]] = {}

    def record_bar_detections(
        self,
        bar_index: int,
        timestamp: pd.Timestamp,
        all_detections: List[Dict[str, Any]],
        active_patterns: Optional[List[str]] = None,
        confluence_count: int = 0,
        passed_threshold: bool = False,
    ) -> None:
        """
        Record all pattern detections for a single bar.

        Args:
            bar_index: Index of the bar in the dataset
            timestamp: Timestamp of the bar
            all_detections: List of detection dictionaries from pattern detectors
            active_patterns: List of pattern names that passed threshold (if any)
            confluence_count: Number of patterns that agreed on direction
            passed_threshold: Whether this bar passed the confluence threshold
        """
        # Store bar metadata
        self._bar_metadata[bar_index] = {
            "timestamp": timestamp,
            "confluence_count": confluence_count,
            "passed_threshold": passed_threshold,
            "active_patterns": active_patterns or [],
        }

        # Record each detection as a SignalEvent
        for detection in all_detections:
            event = SignalEvent(
                bar_index=bar_index,
                timestamp=timestamp,
                pattern_name=detection.get("pattern_name", "Unknown"),
                pattern_category=detection.get("pattern_category", "unknown"),
                direction=detection.get("direction", "Unknown"),
                confidence=detection.get("confidence", 0.0),
                entry_price=detection.get("entry_price", 0.0),
                stop_loss=detection.get("stop_loss", 0.0),
                take_profit_1=detection.get("take_profit_1", 0.0),
                pattern_type=detection.get("pattern_type", "Unknown"),
            )

            self.events.append(event)

            # Index by bar
            if bar_index not in self._events_by_bar:
                self._events_by_bar[bar_index] = []
            self._events_by_bar[bar_index].append(event)

            # Index by pattern
            if event.pattern_name not in self._events_by_pattern:
                self._events_by_pattern[event.pattern_name] = []
            self._events_by_pattern[event.pattern_name].append(event)

    def update_bar_passed_threshold(
        self,
        bar_index: int,
        active_patterns: List[str],
        confluence_count: int,
    ) -> None:
        """
        Update bar metadata after trade execution to mark which patterns passed threshold.

        Args:
            bar_index: Index of the bar
            active_patterns: List of pattern names that contributed to the trade
            confluence_count: Number of patterns that agreed
        """
        if bar_index in self._bar_metadata:
            self._bar_metadata[bar_index]["passed_threshold"] = True
            self._bar_metadata[bar_index]["active_patterns"] = active_patterns
            self._bar_metadata[bar_index]["confluence_count"] = confluence_count

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to DataFrame for analysis.

        Returns:
            DataFrame with all signal events
        """
        if not self.events:
            return pd.DataFrame()

        data = []
        for event in self.events:
            data.append(
                {
                    "bar_index": event.bar_index,
                    "timestamp": event.timestamp,
                    "pattern_name": event.pattern_name,
                    "pattern_category": event.pattern_category,
                    "direction": event.direction,
                    "confidence": event.confidence,
                    "entry_price": event.entry_price,
                    "stop_loss": event.stop_loss,
                    "take_profit_1": event.take_profit_1,
                    "pattern_type": event.pattern_type,
                    "led_to_trade": event.led_to_trade,
                    "trade_pnl": event.trade_pnl,
                    "trade_pnl_pct": event.trade_pnl_pct,
                }
            )

        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values(["bar_index", "pattern_name"]).reset_index(drop=True)
        return df

    def get_detections_for_bar(self, bar_index: int) -> List[SignalEvent]:
        """
        Get all detections at a specific bar.

        Args:
            bar_index: Index of the bar

        Returns:
            List of SignalEvent objects for that bar
        """
        return self._events_by_bar.get(bar_index, [])

    def get_detections_for_pattern(self, pattern_name: str) -> List[SignalEvent]:
        """
        Get all detections for a specific pattern.

        Args:
            pattern_name: Name of the pattern

        Returns:
            List of SignalEvent objects for that pattern
        """
        return self._events_by_pattern.get(pattern_name, [])

    def get_detection_frequency(self) -> pd.DataFrame:
        """
        How often each pattern fires (as % of bars).

        Returns:
            DataFrame with pattern_name, detection_count, detection_frequency_pct
        """
        if not self.events:
            return pd.DataFrame()

        # Get total number of unique bars
        total_bars = len(self._events_by_bar)

        # Count detections per pattern
        pattern_counts: Dict[str, int] = {}
        for event in self.events:
            pattern_counts[event.pattern_name] = pattern_counts.get(event.pattern_name, 0) + 1

        # Build result
        result = []
        for pattern_name, count in pattern_counts.items():
            result.append(
                {
                    "pattern_name": pattern_name,
                    "detection_count": count,
                    "detection_frequency_pct": (count / total_bars * 100) if total_bars > 0 else 0,
                }
            )

        df = pd.DataFrame(result)
        if not df.empty:
            df = df.sort_values("detection_count", ascending=False).reset_index(drop=True)
        return df

    def get_co_occurrence_matrix(self) -> pd.DataFrame:
        """
        NxN matrix: how often patterns fire on the same bar.

        Returns:
            DataFrame where cell [i,j] = number of bars where both pattern i and j fired
        """
        if not self.events:
            return pd.DataFrame()

        # Get unique patterns
        patterns = sorted(set(event.pattern_name for event in self.events))
        n = len(patterns)

        # Initialize matrix
        matrix = np.zeros((n, n), dtype=int)

        # Build pattern name to index mapping
        pattern_to_idx = {name: idx for idx, name in enumerate(patterns)}

        # Count co-occurrences
        for bar_index, events in self._events_by_bar.items():
            # Get patterns that fired on this bar
            bar_patterns = set(event.pattern_name for event in events)

            # Update matrix
            for p1 in bar_patterns:
                for p2 in bar_patterns:
                    matrix[pattern_to_idx[p1], pattern_to_idx[p2]] += 1

        # Convert to DataFrame
        df = pd.DataFrame(matrix, index=patterns, columns=patterns)
        return df

    def get_bar_metadata(self, bar_index: int) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific bar.

        Args:
            bar_index: Index of the bar

        Returns:
            Dictionary with bar metadata or None if not found
        """
        return self._bar_metadata.get(bar_index)

    def get_all_bar_metadata(self) -> Dict[int, Dict[str, Any]]:
        """
        Get metadata for all bars.

        Returns:
            Dictionary mapping bar_index to metadata
        """
        return self._bar_metadata.copy()

    def get_pattern_categories(self) -> List[str]:
        """
        Get unique pattern categories.

        Returns:
            List of unique pattern categories
        """
        return sorted(set(event.pattern_category for event in self.events))

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the signal log.

        Returns:
            Dictionary with summary statistics
        """
        if not self.events:
            return {
                "total_events": 0,
                "unique_patterns": 0,
                "unique_bars": 0,
                "avg_events_per_bar": 0,
            }

        unique_patterns = len(set(event.pattern_name for event in self.events))
        unique_bars = len(self._events_by_bar)

        return {
            "total_events": len(self.events),
            "unique_patterns": unique_patterns,
            "unique_bars": unique_bars,
            "avg_events_per_bar": len(self.events) / unique_bars if unique_bars > 0 else 0,
        }

    def export_csv(self, filepath: str) -> None:
        """
        Export all signal events to a CSV file.

        Args:
            filepath: Path to save the CSV file
        """
        df = self.to_dataframe()
        if not df.empty:
            df.to_csv(filepath, index=False)

    def clear(self) -> None:
        """Clear all events and metadata."""
        self.events.clear()
        self._events_by_bar.clear()
        self._events_by_pattern.clear()
        self._bar_metadata.clear()

    def __len__(self) -> int:
        """Return number of events."""
        return len(self.events)

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_summary_stats()
        return (
            f"SignalEventLog("
            f"events={stats['total_events']}, "
            f"patterns={stats['unique_patterns']}, "
            f"bars={stats['unique_bars']})"
        )
