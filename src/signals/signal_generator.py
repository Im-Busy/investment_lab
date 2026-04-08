"""
Signal Generator

Aggregates signals from multiple pattern detectors and filters
based on confidence, direction, and risk parameters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from ..patterns.base import BasePattern, PatternResult, SignalDirection, TradeSignal


@dataclass
class AggregatedSignal:
    """
    Aggregated trading signal from multiple pattern detectors.

    Attributes:
        timestamp: Signal timestamp
        direction: Long or Short
        entry_price: Suggested entry price
        stop_loss: Stop loss price
        take_profit_1: First take profit target
        take_profit_2: Second take profit target (optional)
        take_profit_3: Third take profit target (optional)
        confidence: Aggregated confidence score
        patterns: List of contributing patterns
        pattern_count: Number of patterns agreeing
        metadata: Additional signal information
    """

    timestamp: pd.Timestamp
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    confidence: float = 0.5
    patterns: List[str] = field(default_factory=list)
    pattern_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary format."""
        return {
            "timestamp": str(self.timestamp),
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "take_profit_3": self.take_profit_3,
            "confidence": self.confidence,
            "patterns": self.patterns,
            "pattern_count": self.pattern_count,
            "metadata": self.metadata,
        }


class SignalGenerator:
    """
    Signal Generator

    Aggregates signals from multiple pattern detectors, filters
    based on confidence thresholds, and combines overlapping signals.
    """

    def __init__(
        self,
        patterns: List[BasePattern],
        min_confidence: float = 0.5,
        max_signals_per_bar: int = 3,
        combine_same_direction: bool = True,
        conflict_resolution: str = "highest_confidence",
    ):
        """
        Initialize Signal Generator.

        Args:
            patterns: List of pattern detectors
            min_confidence: Minimum confidence threshold for signals
            max_signals_per_bar: Maximum signals to generate per bar
            combine_same_direction: Combine signals with same direction
            conflict_resolution: How to handle conflicting signals
                - 'highest_confidence': Keep highest confidence signal
                - 'majority': Use direction with most signals
                - 'none': Don't generate signal on conflict
        """
        self.patterns = patterns
        self.min_confidence = min_confidence
        self.max_signals_per_bar = max_signals_per_bar
        self.combine_same_direction = combine_same_direction
        self.conflict_resolution = conflict_resolution

    def detect_all_patterns(self, df: pd.DataFrame, i: int) -> List[PatternResult]:
        """
        Run all pattern detectors on a specific bar.

        Args:
            df: DataFrame with OHLCV data
            i: Bar index to analyze

        Returns:
            List of PatternResult objects for detected patterns
        """
        results = []

        for pattern in self.patterns:
            try:
                result = pattern.detect(df, i)
                if result.detected and result.signal is not None:
                    if result.signal.confidence >= self.min_confidence:
                        results.append(result)
            except Exception as e:
                # Log error but continue with other patterns
                print(f"Error detecting {pattern.name}: {str(e)}")
                continue

        return results

    def generate_signals(self, df: pd.DataFrame, i: int) -> List[AggregatedSignal]:
        """
        Generate trading signals for a specific bar.

        Args:
            df: DataFrame with OHLCV data
            i: Bar index to analyze

        Returns:
            List of AggregatedSignal objects
        """
        # Detect all patterns
        results = self.detect_all_patterns(df, i)

        if not results:
            return []

        # Separate by direction and extract non-None signals
        long_signals = [
            r
            for r in results
            if r.signal is not None and r.signal.direction == SignalDirection.LONG
        ]
        short_signals = [
            r
            for r in results
            if r.signal is not None and r.signal.direction == SignalDirection.SHORT
        ]

        signals: List[AggregatedSignal] = []

        # Handle conflicts
        if long_signals and short_signals:
            if self.conflict_resolution == "highest_confidence":
                # Explicit extraction to narrow types (signal is non-None by filter above)
                _long_sigs: List[TradeSignal] = [
                    r.signal for r in long_signals if r.signal is not None
                ]  # type: ignore[assignment]
                _short_sigs: List[TradeSignal] = [
                    r.signal for r in short_signals if r.signal is not None
                ]  # type: ignore[assignment]
                best_long_conf = max(s.confidence for s in _long_sigs) if _long_sigs else 0
                best_short_conf = max(s.confidence for s in _short_sigs) if _short_sigs else 0

                if best_long_conf >= best_short_conf:
                    short_signals = []
                else:
                    long_signals = []

            elif self.conflict_resolution == "majority":
                # Use direction with more signals
                if len(long_signals) >= len(short_signals):
                    short_signals = []
                else:
                    long_signals = []

            elif self.conflict_resolution == "none":
                # Don't generate signal on conflict
                return []

        # Aggregate long signals
        if long_signals:
            signal = self._aggregate_signals(long_signals, SignalDirection.LONG, df, i)
            if signal:
                signals.append(signal)

        # Aggregate short signals
        if short_signals:
            signal = self._aggregate_signals(short_signals, SignalDirection.SHORT, df, i)
            if signal:
                signals.append(signal)

        # Limit signals per bar
        return signals[: self.max_signals_per_bar]

    def _aggregate_signals(
        self, results: List[PatternResult], direction: SignalDirection, df: pd.DataFrame, i: int
    ) -> Optional[AggregatedSignal]:
        """
        Aggregate multiple signals of the same direction.

        Args:
            results: List of PatternResult objects
            direction: Signal direction
            df: DataFrame with OHLCV data
            i: Bar index

        Returns:
            AggregatedSignal or None
        """
        if not results:
            return None

        if self.combine_same_direction and len(results) > 1:
            # Combine signals (signal is guaranteed non-None by detect_all_patterns filter)
            signals = [r.signal for r in results if r.signal is not None]

            if not signals:
                return None

            # Weighted average entry price by confidence
            total_confidence = sum(s.confidence for s in signals)
            if total_confidence == 0:
                total_confidence = 1

            entry_price = sum(s.entry_price * s.confidence for s in signals) / total_confidence

            # Use most aggressive stop loss (furthest from entry)
            if direction == SignalDirection.LONG:
                stop_loss = min(s.stop_loss for s in signals)
            else:
                stop_loss = max(s.stop_loss for s in signals)

            # Use most conservative take profit (closest to entry)
            tp1_values = [s.take_profit_1 for s in signals if s.take_profit_1 is not None]
            take_profit_1 = min(tp1_values) if tp1_values else signals[0].take_profit_1

            # Combine confidence with bonus for multiple patterns
            base_confidence = total_confidence / len(signals)
            pattern_bonus = min(0.15, len(signals) * 0.05)
            confidence = min(1.0, base_confidence + pattern_bonus)

            pattern_names = [r.pattern_name for r in results]

            ts_combined: pd.Timestamp = df.index[i]

            return AggregatedSignal(
                timestamp=ts_combined,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=signals[0].take_profit_2,
                take_profit_3=signals[0].take_profit_3,
                confidence=confidence,
                patterns=pattern_names,
                pattern_count=len(results),
                metadata={
                    "combined": True,
                    "individual_confidences": [s.confidence for s in signals],  # noqa: PGH003
                },
            )
        else:
            # Return single best signal
            best_result = max(results, key=lambda r: r.signal.confidence if r.signal else 0)
            signal = best_result.signal
            if signal is None:
                return None

            ts: pd.Timestamp = signal.timestamp if signal.timestamp is not None else df.index[i]

            return AggregatedSignal(
                timestamp=ts,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit_1=signal.take_profit_1,
                take_profit_2=signal.take_profit_2,
                take_profit_3=signal.take_profit_3,
                confidence=signal.confidence,
                patterns=[best_result.pattern_name],
                pattern_count=1,
                metadata=signal.metadata,
            )

    def scan_dataframe(
        self, df: pd.DataFrame, start_index: Optional[int] = None, end_index: Optional[int] = None
    ) -> List[AggregatedSignal]:
        """
        Scan entire DataFrame for signals.

        Args:
            df: DataFrame with OHLCV data
            start_index: Starting bar index (default: first bar)
            end_index: Ending bar index (default: last bar)

        Returns:
            List of all AggregatedSignal objects found
        """
        if start_index is None:
            start_index = max(p.min_bars_required for p in self.patterns)

        if end_index is None:
            end_index = len(df)

        all_signals = []

        for i in range(start_index, end_index):
            signals = self.generate_signals(df, i)
            all_signals.extend(signals)

        return all_signals

    def get_pattern_summary(self, df: pd.DataFrame, i: int) -> Dict[str, Any]:
        """
        Get a summary of all pattern detections at a bar.

        Args:
            df: DataFrame with OHLCV data
            i: Bar index

        Returns:
            Dictionary with pattern detection summary
        """
        results = self.detect_all_patterns(df, i)

        long_patterns: List[Dict[str, Any]] = []
        short_patterns: List[Dict[str, Any]] = []

        for result in results:
            s = result.signal
            if s is None:
                continue
            pattern_info: Dict[str, Any] = {
                "name": result.pattern_name,
                "confidence": s.confidence,
                "entry": s.entry_price,
            }

            if s.direction == SignalDirection.LONG:
                long_patterns.append(pattern_info)
            else:
                short_patterns.append(pattern_info)

        signals = self.generate_signals(df, i)

        summary: Dict[str, Any] = {
            "timestamp": str(df.iloc[i].name) if hasattr(df.iloc[i], "name") else i,
            "total_patterns": len(results),
            "long_patterns": long_patterns,
            "short_patterns": short_patterns,
            "signals": [s.to_dict() for s in signals],
        }

        return summary
