"""
Rectangle Pattern (Horizontal Channel)

Detection Logic:
- Price oscillates between parallel horizontal support and resistance
- Minimum: 2 touches of support + 2 touches of resistance
- Support level: at least 2 lows within tolerance
- Resistance level: at least 2 highs within tolerance
- Can be continuation or reversal depending on breakout direction

Entry Rules:
- Long Entry: Buy Stop = high[breakout_bar] + filter (Close above resistance)
- Short Entry: Sell Stop = low[breakdown_bar] - filter (Close below support)
- Warning: High false breakout rate → require strong confirmation

Stop Loss Rules:
- Long Stop: Support level - filter
- Short Stop: Resistance level + filter

Take Profit Rules:
- Long Target: Entry + (Resistance - Support) = pattern height
- Short Target: Entry - (Resistance - Support) = pattern height
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...indicators.pivots import find_swing_highs, find_swing_lows
from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class Rectangle(BasePattern):
    """
    Rectangle Pattern Detector

    A continuation/reversal pattern characterized by price oscillating
    between horizontal support and resistance levels.
    """

    def __init__(
        self,
        lookback: int = 5,
        level_tolerance: float = 0.02,
        min_touches: int = 2,
        min_pattern_bars: int = 10,
        max_pattern_bars: int = 60,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        confirmation_filter: float = 0.005,
        volume_filter: bool = False,
    ):
        """
        Initialize Rectangle pattern detector.

        Args:
            lookback: Lookback period for pivot detection
            level_tolerance: Maximum price deviation for horizontal levels (default 2%)
            min_touches: Minimum touches on each level (default 2)
            min_pattern_bars: Minimum bars for pattern formation
            max_pattern_bars: Maximum bars for pattern formation
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            confirmation_filter: Minimum breakout percentage for confirmation
            volume_filter: Require volume confirmation on breakout
        """
        super().__init__(
            name="Rectangle", pattern_type=PatternType.CONTINUATION, min_bars_required=min_pattern_bars
        )
        self.lookback = lookback
        self.level_tolerance = level_tolerance
        self.min_touches = min_touches
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.confirmation_filter = confirmation_filter
        self.volume_filter = volume_filter

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

    def _find_horizontal_level(
        self, points: List[Tuple[int, float]], tolerance: float, min_touches: int
    ) -> Optional[Tuple[List[Tuple[int, float]], float]]:
        """
        Find a horizontal price level from a list of points.

        Args:
            points: List of (index, price) tuples
            tolerance: Maximum price deviation as percentage
            min_touches: Minimum number of touches required

        Returns:
            Tuple of (qualifying points, level price) or None
        """
        if len(points) < min_touches:
            return None

        # Sort points by price
        sorted_points = sorted(points, key=lambda x: x[1])

        # Try to find clusters of points at similar levels
        best_cluster = None
        best_level = None

        for start_idx in range(len(sorted_points) - min_touches + 1):
            candidate_points = [sorted_points[start_idx]]
            level_price = sorted_points[start_idx][1]

            for idx in range(start_idx + 1, len(sorted_points)):
                point = sorted_points[idx]
                price_diff = abs(point[1] - level_price) / level_price if level_price > 0 else 0

                if price_diff <= tolerance:
                    candidate_points.append(point)

            if len(candidate_points) >= min_touches:
                    # Calculate average level price
                    avg_level = float(np.mean([p[1] for p in candidate_points]))
                    
                    if best_cluster is None or len(candidate_points) > len(best_cluster):
                        best_cluster = candidate_points
                        best_level = avg_level
    
            if best_cluster is not None and best_level is not None:
                return (best_cluster, best_level)

        return None

    def _check_volume_breakout(
        self, arrays: dict, i: int, lookback: int = 20
    ) -> Tuple[bool, bool]:
        """
        Check if volume confirms breakout.

        Returns:
            Tuple of (volume_above_average, volume_spike)
        """
        volume_arr = arrays["volume"]
        if i < lookback:
            return (False, False)

        current_volume = float(volume_arr[i])
        avg_volume = float(np.mean(volume_arr[i - lookback:i]))

        volume_above_average = current_volume > avg_volume
        volume_spike = current_volume > avg_volume * 1.5  # 50% above average

        return (volume_above_average, volume_spike)

    def _find_pattern(
        self, arrays: dict, peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]], i: int
    ) -> Optional[Dict]:
        """
        Find Rectangle pattern.

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

        # Find horizontal resistance level
        resistance_result = self._find_horizontal_level(peaks, self.level_tolerance, self.min_touches)
        if resistance_result is None:
            return None

        resistance_points, resistance_level = resistance_result

        # Find horizontal support level
        support_result = self._find_horizontal_level(troughs, self.level_tolerance, self.min_touches)
        if support_result is None:
            return None

        support_points, support_level = support_result

        # Ensure resistance is above support
        if resistance_level <= support_level:
            return None

        # Check pattern height is reasonable (not too narrow, not too wide)
        pattern_height = resistance_level - support_level
        mid_price = (resistance_level + support_level) / 2
        height_pct = pattern_height / mid_price if mid_price > 0 else 0

        # Pattern height should be between 1% and 20% of price
        if height_pct < 0.01 or height_pct > 0.20:
            return None

        # Check pattern duration
        all_indices = [p[0] for p in resistance_points] + [t[0] for t in support_points]
        pattern_start = min(all_indices)
        pattern_end = max(all_indices)
        pattern_duration = pattern_end - pattern_start

        if pattern_duration < self.min_pattern_bars or pattern_duration > self.max_pattern_bars:
            return None

        # Check for breakout
        current_close = float(close_arr[i])
        current_high = float(high_arr[i])
        current_low = float(low_arr[i])

        # Upside breakout: close above resistance with confirmation
        upside_breakout = current_close > resistance_level * (1 + self.confirmation_filter)

        # Downside breakout: close below support with confirmation
        downside_breakout = current_close < support_level * (1 - self.confirmation_filter)

        # Volume check
        volume_confirmed = False
        volume_spike = False
        if self.volume_filter:
            volume_confirmed, volume_spike = self._check_volume_breakout(arrays, i)

        if upside_breakout:
            return {
                "direction": "bullish",
                "resistance_level": resistance_level,
                "resistance_points": resistance_points,
                "support_level": support_level,
                "support_points": support_points,
                "pattern_height": pattern_height,
                "pattern_start": pattern_start,
                "pattern_end": i,
                "breakout_bar": i,
                "volume_confirmed": volume_confirmed,
                "volume_spike": volume_spike,
            }
        elif downside_breakout:
            return {
                "direction": "bearish",
                "resistance_level": resistance_level,
                "resistance_points": resistance_points,
                "support_level": support_level,
                "support_points": support_points,
                "pattern_height": pattern_height,
                "pattern_start": pattern_start,
                "pattern_end": i,
                "breakout_bar": i,
                "volume_confirmed": volume_confirmed,
                "volume_spike": volume_spike,
            }

        return None

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Rectangle pattern at bar index i.

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
                "resistance_level": pattern["resistance_level"],
                "support_level": pattern["support_level"],
                "pattern_height": pattern["pattern_height"],
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
        """Generate Rectangle signal."""
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        current_high = float(high_arr[i])
        current_low = float(low_arr[i])
        resistance_level = pattern["resistance_level"]
        support_level = pattern["support_level"]
        pattern_height = pattern["pattern_height"]
        direction = pattern["direction"]

        # Base confidence - rectangles have high false breakout rate
        confidence = 0.55

        # Boost confidence if volume confirms
        if self.volume_filter and pattern.get("volume_confirmed", False):
            confidence += 0.10
        if self.volume_filter and pattern.get("volume_spike", False):
            confidence += 0.05

        if direction == "bullish":
            # Upside breakout - LONG signal
            entry_price = current_high + self.entry_offset
            stop_loss = support_level - self.stop_offset
            take_profit_1 = entry_price + (pattern_height * 0.62)
            take_profit_2 = entry_price + pattern_height
            take_profit_3 = entry_price + (pattern_height * 1.27)

            return TradeSignal(
                pattern_name=self.name,
                direction=SignalDirection.LONG,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                take_profit_3=take_profit_3,
                confidence=min(confidence, 1.0),
                timestamp=df.index[i] if hasattr(df, "index") else None,
                metadata={
                    "breakout_direction": "up",
                    "resistance_level": resistance_level,
                    "support_level": support_level,
                    "pattern_height": pattern_height,
                    "entry_type": "buy_stop",
                    "volume_confirmed": pattern.get("volume_confirmed", False),
                },
            )
        else:
            # Downside breakout - SHORT signal
            entry_price = current_low - self.entry_offset
            stop_loss = resistance_level + self.stop_offset
            take_profit_1 = entry_price - (pattern_height * 0.62)
            take_profit_2 = entry_price - pattern_height
            take_profit_3 = entry_price - (pattern_height * 1.27)

            return TradeSignal(
                pattern_name=self.name,
                direction=SignalDirection.SHORT,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                take_profit_3=take_profit_3,
                confidence=min(confidence, 1.0),
                timestamp=df.index[i] if hasattr(df, "index") else None,
                metadata={
                    "breakout_direction": "down",
                    "resistance_level": resistance_level,
                    "support_level": support_level,
                    "pattern_height": pattern_height,
                    "entry_type": "sell_stop",
                    "volume_confirmed": pattern.get("volume_confirmed", False),
                },
            )
