"""
Wedge Pattern (Rising Wedge / Falling Wedge)

Detection Logic:
- Both trendlines slope in SAME direction
- Rising Wedge: both lines slope up (bearish reversal)
- Falling Wedge: both lines slope down (bullish reversal)
- Price must touch trendlines at least 5 times total (min 3 on one side, 2 on other)
- Converging toward apex

Entry Rules:
- Rising Wedge breakout DOWN → SELL
- Falling Wedge breakout UP → BUY
- Confirmation: Close must be outside trendline by filter amount

Stop Loss Rules:
- For shorts: Above the upper trendline or most recent swing high
- For longs: Below the lower trendline or most recent swing low

Take Profit Rules:
- Downward breakout: target = lowest_trough_in_pattern
- Upward breakout: target = breakout_price + (highest_peak - lowest_trough)
- Warning: High retracement rate → use wider stops
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...indicators.pivots import find_swing_highs, find_swing_lows
from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class Wedge(BasePattern):
    """
    Wedge Pattern Detector (Rising and Falling)

    A reversal pattern where both trendlines slope in the same direction:
    - Rising Wedge: Bearish reversal (both lines slope up, breakout down)
    - Falling Wedge: Bullish reversal (both lines slope down, breakout up)
    """

    def __init__(
        self,
        lookback: int = 5,
        min_touches: int = 5,
        min_touches_per_side: int = 2,
        min_pattern_bars: int = 10,
        max_pattern_bars: int = 60,
        min_slope_magnitude: float = 0.0005,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        confirmation_filter: float = 0.005,
    ):
        """
        Initialize Wedge pattern detector.

        Args:
            lookback: Lookback period for pivot detection
            min_touches: Minimum total touches on both trendlines (default 5)
            min_touches_per_side: Minimum touches on one side (default 2)
            min_pattern_bars: Minimum bars for pattern formation
            max_pattern_bars: Maximum bars for pattern formation
            min_slope_magnitude: Minimum absolute slope for trendlines
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            confirmation_filter: Minimum breakout percentage for confirmation
        """
        super().__init__(
            name="Wedge", pattern_type=PatternType.REVERSAL, min_bars_required=min_pattern_bars
        )
        self.lookback = lookback
        self.min_touches = min_touches
        self.min_touches_per_side = min_touches_per_side
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        self.min_slope_magnitude = min_slope_magnitude
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

    def _calculate_trendline_value(
        self, start_idx: int, start_price: float, slope: float, current_idx: int
    ) -> float:
        """Calculate trendline value at a given index."""
        return start_price + slope * (current_idx - start_idx)

    def _find_wedge_pattern(
        self, arrays: dict, peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]], i: int
    ) -> Optional[Dict]:
        """
        Find Wedge pattern (rising or falling).

        Args:
            arrays: Dictionary with NumPy arrays
            peaks: List of (index, price) for swing highs
            troughs: List of (index, price) for swing lows
            i: Current bar index

        Returns:
            Dictionary with pattern details or None
        """
        close_arr = arrays["close"]

        # Need enough points for both trendlines
        if len(peaks) < self.min_touches_per_side or len(troughs) < self.min_touches_per_side:
            return None

        # Sort by index (chronological order)
        sorted_peaks = sorted(peaks, key=lambda x: x[0])
        sorted_troughs = sorted(troughs, key=lambda x: x[0])

        # Calculate slopes
        upper_slope = self._calculate_slope(sorted_peaks)
        lower_slope = self._calculate_slope(sorted_troughs)

        # Check for wedge conditions
        # Rising wedge: both slopes positive (upper slope steeper)
        # Falling wedge: both slopes negative (lower slope steeper)

        total_touches = len(peaks) + len(troughs)
        if total_touches < self.min_touches:
            return None

        # Check pattern duration
        all_indices = [p[0] for p in peaks] + [t[0] for t in troughs]
        pattern_start = min(all_indices)
        pattern_end = max(all_indices)
        pattern_duration = pattern_end - pattern_start

        if pattern_duration < self.min_pattern_bars or pattern_duration > self.max_pattern_bars:
            return None

        # Get pattern boundaries
        highest_peak = max(p[1] for p in peaks)
        lowest_trough = min(t[1] for t in troughs)
        pattern_height = highest_peak - lowest_trough

        if pattern_height <= 0:
            return None

        current_close = float(close_arr[i])

        # Rising Wedge: both slopes positive
        if upper_slope > self.min_slope_magnitude and lower_slope > self.min_slope_magnitude:
            # Upper slope should be steeper (converging)
            if upper_slope <= lower_slope:
                return None

            # Calculate trendline values at current bar
            upper_start_idx = sorted_peaks[0][0]
            upper_start_price = sorted_peaks[0][1]
            upper_value = self._calculate_trendline_value(upper_start_idx, upper_start_price, upper_slope, i)

            lower_start_idx = sorted_troughs[0][0]
            lower_start_price = sorted_troughs[0][1]
            lower_value = self._calculate_trendline_value(lower_start_idx, lower_start_price, lower_slope, i)

            # Check for downside breakout (expected for rising wedge)
            downside_breakout = current_close < lower_value * (1 - self.confirmation_filter)

            if downside_breakout:
                return {
                    "wedge_type": "rising",
                    "direction": "bearish",
                    "upper_slope": upper_slope,
                    "lower_slope": lower_slope,
                    "upper_trendline": upper_value,
                    "lower_trendline": lower_value,
                    "highest_peak": highest_peak,
                    "lowest_trough": lowest_trough,
                    "pattern_height": pattern_height,
                    "pattern_start": pattern_start,
                    "pattern_end": i,
                    "breakout_bar": i,
                    "peaks": sorted_peaks,
                    "troughs": sorted_troughs,
                }

        # Falling Wedge: both slopes negative
        elif upper_slope < -self.min_slope_magnitude and lower_slope < -self.min_slope_magnitude:
            # Lower slope should be steeper (more negative, converging)
            if lower_slope >= upper_slope:
                return None

            # Calculate trendline values at current bar
            upper_start_idx = sorted_peaks[0][0]
            upper_start_price = sorted_peaks[0][1]
            upper_value = self._calculate_trendline_value(upper_start_idx, upper_start_price, upper_slope, i)

            lower_start_idx = sorted_troughs[0][0]
            lower_start_price = sorted_troughs[0][1]
            lower_value = self._calculate_trendline_value(lower_start_idx, lower_start_price, lower_slope, i)

            # Check for upside breakout (expected for falling wedge)
            upside_breakout = current_close > upper_value * (1 + self.confirmation_filter)

            if upside_breakout:
                return {
                    "wedge_type": "falling",
                    "direction": "bullish",
                    "upper_slope": upper_slope,
                    "lower_slope": lower_slope,
                    "upper_trendline": upper_value,
                    "lower_trendline": lower_value,
                    "highest_peak": highest_peak,
                    "lowest_trough": lowest_trough,
                    "pattern_height": pattern_height,
                    "pattern_start": pattern_start,
                    "pattern_end": i,
                    "breakout_bar": i,
                    "peaks": sorted_peaks,
                    "troughs": sorted_troughs,
                }

        return None

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Wedge pattern at bar index i.

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
        pattern = self._find_wedge_pattern(arrays, peaks, troughs, i)

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
                "wedge_type": pattern["wedge_type"],
                "upper_slope": pattern["upper_slope"],
                "lower_slope": pattern["lower_slope"],
                "highest_peak": pattern["highest_peak"],
                "lowest_trough": pattern["lowest_trough"],
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
        """Generate Wedge signal."""
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        current_high = float(high_arr[i])
        current_low = float(low_arr[i])
        highest_peak = pattern["highest_peak"]
        lowest_trough = pattern["lowest_trough"]
        pattern_height = pattern["pattern_height"]
        direction = pattern["direction"]
        wedge_type = pattern["wedge_type"]

        # Base confidence - wedges have high retracement rate
        confidence = 0.55

        # Higher confidence for falling wedge (bullish)
        if wedge_type == "falling":
            confidence += 0.05

        if direction == "bullish":
            # Upside breakout from falling wedge - LONG signal
            entry_price = current_high + self.entry_offset
            stop_loss = lowest_trough - self.stop_offset
            take_profit_1 = entry_price + (pattern_height * 0.62)
            take_profit_2 = entry_price + pattern_height
            take_profit_3 = entry_price + (pattern_height * 1.27)

            return TradeSignal(
                pattern_name=f"{wedge_type.title()} Wedge",
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
                    "wedge_type": wedge_type,
                    "highest_peak": highest_peak,
                    "lowest_trough": lowest_trough,
                    "pattern_height": pattern_height,
                    "entry_type": "buy_stop",
                },
            )
        else:
            # Downside breakout from rising wedge - SHORT signal
            entry_price = current_low - self.entry_offset
            stop_loss = highest_peak + self.stop_offset
            # For rising wedge breakdown, target is the lowest trough
            take_profit_1 = entry_price - (pattern_height * 0.62)
            take_profit_2 = lowest_trough  # Target the lowest point
            take_profit_3 = entry_price - (pattern_height * 1.27)

            return TradeSignal(
                pattern_name=f"{wedge_type.title()} Wedge",
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
                    "wedge_type": wedge_type,
                    "highest_peak": highest_peak,
                    "lowest_trough": lowest_trough,
                    "pattern_height": pattern_height,
                    "entry_type": "sell_stop",
                },
            )
