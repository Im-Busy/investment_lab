"""
Descending Triangle Pattern

Detection Logic:
- Upper bound: downward-sloping resistance line (lower highs - at least 2 touches with negative slope)
- Lower bound: horizontal support line (flat lows - at least 2 touches within tolerance)
- Price must touch each trendline at least 2 times
- Pattern typically has bearish bias (downward breakout more common)

Entry Rules:
- Short Entry: Sell Stop = low[breakdown_bar] - filter (Close below support)
- Long Entry: Buy Stop = high[breakout_bar] + filter (only on confirmed upside break)
- Confirmation: Close must be outside trendline by at least filter amount

Stop Loss Rules:
- Short Stop: Above the downward-sloping resistance line or pattern high + filter
- Alternative Stop: Above most recent swing high

Take Profit Rules:
- Target 1: Entry - 0.62 * Triangle_Depth (62% of pattern height)
- Target 2: Entry - 1.00 * Triangle_Depth (100% of pattern height)
- Triangle_Depth = Highest peak in pattern - Support level
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...indicators.pivots import find_swing_highs, find_swing_lows
from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class DescendingTriangle(BasePattern):
    """
    Descending Triangle Pattern Detector

    A bearish continuation pattern characterized by a downward-sloping resistance line
    and a horizontal support line (lower highs).
    """

    def __init__(
        self,
        lookback: int = 5,
        support_tolerance: float = 0.02,
        min_touches: int = 2,
        min_pattern_bars: int = 10,
        max_pattern_bars: int = 60,
        max_slope: float = -0.001,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        confirmation_filter: float = 0.005,
    ):
        """
        Initialize Descending Triangle pattern detector.

        Args:
            lookback: Lookback period for pivot detection
            support_tolerance: Maximum price deviation for horizontal support (default 2%)
            min_touches: Minimum touches on each trendline (default 2)
            min_pattern_bars: Minimum bars for pattern formation
            max_pattern_bars: Maximum bars for pattern formation
            max_slope: Maximum slope for resistance line (negative for descending)
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            confirmation_filter: Minimum breakout percentage for confirmation
        """
        super().__init__(
            name="Descending Triangle", pattern_type=PatternType.CONTINUATION, min_bars_required=min_pattern_bars
        )
        self.lookback = lookback
        self.support_tolerance = support_tolerance
        self.min_touches = min_touches
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        self.max_slope = max_slope
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.confirmation_filter = confirmation_filter

    def _find_pivots(
        self, df: pd.DataFrame, i: int
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Find swing highs and lows for pattern detection."""
        swing_highs = find_swing_highs(df, self.lookback)
        swing_lows = find_swing_lows(df, self.lookback)

        peaks = []
        troughs = []

        lookback = min(self.max_pattern_bars * 2, i)

        for j in range(i - lookback, i + 1):
            if j < 0:
                continue
            if pd.notna(swing_highs.iloc[j]):
                peaks.append((j, self._safe_float(swing_highs.iloc[j])))
            if pd.notna(swing_lows.iloc[j]):
                troughs.append((j, self._safe_float(swing_lows.iloc[j])))

        return peaks, troughs

    def _calculate_slope(self, points: List[Tuple[int, float]]) -> float:
        """Calculate linear regression slope of price points."""
        if len(points) < 2:
            return 0.0

        indices = np.array([p[0] for p in points], dtype=np.float64)
        prices = np.array([p[1] for p in points], dtype=np.float64)

        # Linear regression: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
        n = len(points)
        sum_x = np.sum(indices)
        sum_y = np.sum(prices)
        sum_xy = np.sum(indices * prices)
        sum_x2 = np.sum(indices * indices)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return float(slope)

    def _find_horizontal_support(
        self, troughs: List[Tuple[int, float]]
    ) -> Optional[Tuple[List[Tuple[int, float]], float]]:
        """
        Find horizontal support level from troughs.

        Returns:
            Tuple of (qualifying troughs, support level) or None
        """
        if len(troughs) < self.min_touches:
            return None

        # Sort troughs by price
        sorted_troughs = sorted(troughs, key=lambda x: x[1])

        # Try to find clusters of troughs at similar levels
        for start_idx in range(len(sorted_troughs) - self.min_touches + 1):
            candidate_troughs = [sorted_troughs[start_idx]]
            support_level = sorted_troughs[start_idx][1]

            for idx in range(start_idx + 1, len(sorted_troughs)):
                trough = sorted_troughs[idx]
                price_diff = abs(trough[1] - support_level) / support_level

                if price_diff <= self.support_tolerance:
                    candidate_troughs.append(trough)

            if len(candidate_troughs) >= self.min_touches:
                # Calculate average support level
                avg_support = float(np.mean([t[1] for t in candidate_troughs]))
                return (candidate_troughs, avg_support)

        return None

    def _find_falling_resistance(
        self, peaks: List[Tuple[int, float]], max_slope: float
    ) -> Optional[Tuple[List[Tuple[int, float]], float]]:
        """
        Find falling resistance line from peaks.

        Returns:
            Tuple of (qualifying peaks, slope) or None
        """
        if len(peaks) < self.min_touches:
            return None

        # Sort peaks by index (chronological order)
        sorted_peaks = sorted(peaks, key=lambda x: x[0])

        # Calculate slope of all peaks
        slope = self._calculate_slope(sorted_peaks)

        # Check if slope is negative (falling)
        if slope > max_slope:
            return None

        return (sorted_peaks, slope)

    def _find_pattern(
        self, arrays: dict, peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]], i: int
    ) -> Optional[Dict]:
        """
        Find Descending Triangle pattern.

        Args:
            arrays: Dictionary with NumPy arrays
            peaks: List of (index, price) for swing highs
            troughs: List of (index, price) for swing lows
            i: Current bar index

        Returns:
            Dictionary with pattern details or None
        """
        close_arr = arrays["close"]
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        # Find horizontal support
        support_result = self._find_horizontal_support(troughs)
        if support_result is None:
            return None

        support_troughs, support_level = support_result

        # Find falling resistance
        resistance_result = self._find_falling_resistance(peaks, self.max_slope)
        if resistance_result is None:
            return None

        resistance_peaks, resistance_slope = resistance_result

        # Check pattern duration
        all_indices = [p[0] for p in resistance_peaks] + [t[0] for t in support_troughs]
        pattern_start = min(all_indices)
        pattern_end = max(all_indices)
        pattern_duration = pattern_end - pattern_start

        if pattern_duration < self.min_pattern_bars or pattern_duration > self.max_pattern_bars:
            return None

        # Get pattern boundaries
        highest_peak = max(p[1] for p in resistance_peaks)
        pattern_depth = highest_peak - support_level

        if pattern_depth <= 0:
            return None

        # Check for downside breakout
        current_close = float(close_arr[i])
        current_low = float(low_arr[i])

        # Downside breakout: close below support with confirmation
        downside_breakout = current_close < support_level * (1 - self.confirmation_filter)

        # Upside breakout: close above resistance line (less common for descending triangle)
        # Calculate resistance line value at current bar
        resistance_start_idx = resistance_peaks[0][0]
        resistance_start_price = resistance_peaks[0][1]
        resistance_at_current = resistance_start_price + resistance_slope * (i - resistance_start_idx)
        upside_breakout = current_close > resistance_at_current * (1 + self.confirmation_filter)

        if downside_breakout:
            return {
                "direction": "bearish",
                "support_level": support_level,
                "support_troughs": support_troughs,
                "resistance_peaks": resistance_peaks,
                "resistance_slope": resistance_slope,
                "highest_peak": highest_peak,
                "pattern_depth": pattern_depth,
                "pattern_start": pattern_start,
                "pattern_end": i,
                "breakout_bar": i,
            }
        elif upside_breakout:
            return {
                "direction": "bullish",
                "support_level": support_level,
                "support_troughs": support_troughs,
                "resistance_peaks": resistance_peaks,
                "resistance_slope": resistance_slope,
                "highest_peak": highest_peak,
                "pattern_depth": pattern_depth,
                "pattern_start": pattern_start,
                "pattern_end": i,
                "breakout_bar": i,
            }

        return None

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Descending Triangle pattern at bar index i.

        Args:
            df: DataFrame with OHLCV data
            i: Current bar index
            window_start: Optional window start for bounds checking

        Returns:
            PatternResult with detection status and signal
        """
        if not self._validate_data(df, i, window_start):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        arrays = self._extract_arrays(df)
        peaks, troughs = self._find_pivots(df, i)
        pattern = self._find_pattern(arrays, peaks, troughs, i)

        if pattern is None:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        signal = self._generate_signal(df, i, pattern, arrays)

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                "support_level": pattern["support_level"],
                "resistance_slope": pattern["resistance_slope"],
                "highest_peak": pattern["highest_peak"],
                "pattern_depth": pattern["pattern_depth"],
                "direction": pattern["direction"],
            },
            bars_since_detection=0,
            start_index=pattern["pattern_start"],
            end_index=i,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal."""
        result = self.detect(df, i)
        return result.signal if result.detected else None

    def _generate_signal(
        self, df: pd.DataFrame, i: int, pattern: Dict, arrays: dict
    ) -> Optional[TradeSignal]:
        """Generate Descending Triangle signal."""
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        current_high = float(high_arr[i])
        current_low = float(low_arr[i])
        support_level = pattern["support_level"]
        highest_peak = pattern["highest_peak"]
        pattern_depth = pattern["pattern_depth"]
        direction = pattern["direction"]

        if direction == "bearish":
            # Downside breakout - SHORT signal
            entry_price = current_low - self.entry_offset
            stop_loss = highest_peak + self.stop_offset
            take_profit_1 = entry_price - (pattern_depth * 0.62)
            take_profit_2 = entry_price - pattern_depth
            take_profit_3 = entry_price - (pattern_depth * 1.27)

            confidence = 0.65  # Higher confidence for downside breakout in descending triangle

            return TradeSignal(
                pattern_name=self.name,
                direction=SignalDirection.SHORT,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                take_profit_3=take_profit_3,
                confidence=confidence,
                timestamp=df.index[i] if hasattr(df, "index") else None,
                metadata={
                    "breakout_direction": "down",
                    "support_level": support_level,
                    "pattern_depth": pattern_depth,
                    "entry_type": "sell_stop",
                },
            )
        else:
            # Upside breakout - LONG signal (less common)
            entry_price = current_high + self.entry_offset
            stop_loss = support_level - self.stop_offset
            take_profit_1 = entry_price + (pattern_depth * 0.62)
            take_profit_2 = entry_price + pattern_depth
            take_profit_3 = entry_price + (pattern_depth * 1.27)

            confidence = 0.50  # Lower confidence for upside breakout in descending triangle

            return TradeSignal(
                pattern_name=self.name,
                direction=SignalDirection.LONG,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                take_profit_3=take_profit_3,
                confidence=confidence,
                timestamp=df.index[i] if hasattr(df, "index") else None,
                metadata={
                    "breakout_direction": "up",
                    "support_level": support_level,
                    "pattern_depth": pattern_depth,
                    "entry_type": "buy_stop",
                },
            )
