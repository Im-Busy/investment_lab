"""
Ascending Triangle Pattern

Detection Logic:
- Upper bound: horizontal resistance line (flat highs - at least 2 touches within tolerance)
- Lower bound: upward-sloping support line (higher lows - at least 2 touches with positive slope)
- Price must touch each trendline at least 2 times
- Pattern typically has bullish bias (upward breakout more common)

Entry Rules:
- Long Entry: Buy Stop = high[breakout_bar] + filter (Close above resistance)
- Short Entry: Sell Stop = low[breakdown_bar] - filter (only on confirmed downside break)
- Confirmation: Close must be outside trendline by at least filter amount

Stop Loss Rules:
- Long Stop: Below the upward-sloping support line or pattern low - filter
- Alternative Stop: Below most recent swing low

Take Profit Rules:
- Target 1: Entry + 0.62 * Triangle_Depth (62% of pattern height)
- Target 2: Entry + 1.00 * Triangle_Depth (100% of pattern height)
- Triangle_Depth = Resistance level - Lowest trough in pattern
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...indicators.pivots import find_swing_highs, find_swing_lows
from ...indicators.technical import volume_sma
from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class AscendingTriangle(BasePattern):
    """
    Ascending Triangle Pattern Detector

    A bullish continuation pattern characterized by a horizontal resistance line
    and an upward-sloping support line (higher lows).
    """

    def __init__(
        self,
        lookback: int = 5,
        resistance_tolerance: float = 0.02,
        min_touches: int = 2,
        min_pattern_bars: int = 10,
        max_pattern_bars: int = 60,
        min_slope: float = 0.001,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        confirmation_filter: float = 0.005,
    ):
        """
        Initialize Ascending Triangle pattern detector.

        Args:
            lookback: Lookback period for pivot detection
            resistance_tolerance: Maximum price deviation for horizontal resistance (default 2%)
            min_touches: Minimum touches on each trendline (default 2)
            min_pattern_bars: Minimum bars for pattern formation
            max_pattern_bars: Maximum bars for pattern formation
            min_slope: Minimum slope for support line (positive for ascending)
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            confirmation_filter: Minimum breakout percentage for confirmation
        """
        super().__init__(
            name="Ascending Triangle",
            pattern_type=PatternType.CONTINUATION,
            min_bars_required=min_pattern_bars,
        )
        self.lookback = lookback
        self.resistance_tolerance = resistance_tolerance
        self.min_touches = min_touches
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        self.min_slope = min_slope
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.confirmation_filter = confirmation_filter

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized Ascending Triangle detection across all bars.

        Returns np.int8 array: 0=no signal, 1=long, -1=short.
        """
        n = len(df)
        signals = np.zeros(n, dtype=np.int8)
        close = df["Close"].to_numpy(dtype=np.float64)

        swing_highs = find_swing_highs(df, self.lookback)
        swing_lows = find_swing_lows(df, self.lookback)

        sh_arr = swing_highs.to_numpy(dtype=np.float64)
        sl_arr = swing_lows.to_numpy(dtype=np.float64)
        n_swing_highs = pd.notna(swing_highs).to_numpy()
        n_swing_lows = pd.notna(swing_lows).to_numpy()

        for i in range(max(self.lookback, self.min_pattern_bars), n):
            lookback = min(self.max_pattern_bars * 2, i)

            peaks: list = []
            troughs: list = []
            for j in range(i - lookback, i + 1):
                if j < 0:
                    continue
                if n_swing_highs[j]:
                    peaks.append((j, float(sh_arr[j])))
                if n_swing_lows[j]:
                    troughs.append((j, float(sl_arr[j])))

            if len(peaks) < self.min_touches or len(troughs) < self.min_touches:
                continue

            resistance_result = self._find_horizontal_resistance(peaks)
            if resistance_result is None:
                continue
            resistance_peaks, resistance_level = resistance_result

            support_result = self._find_rising_support(troughs, self.min_slope)
            if support_result is None:
                continue
            support_troughs, support_slope = support_result

            all_indices = [p[0] for p in resistance_peaks] + [t[0] for t in support_troughs]
            p_start = min(all_indices)
            p_end = max(all_indices)
            if p_end - p_start < self.min_pattern_bars or p_end - p_start > self.max_pattern_bars:
                continue

            lowest_trough = min(t[1] for t in support_troughs)
            if resistance_level <= lowest_trough:
                continue

            curr_close = float(close[i])

            upside = curr_close > resistance_level * (1.0 + self.confirmation_filter)
            sup_start_idx = support_troughs[0][0]
            sup_start_price = support_troughs[0][1]
            sup_at_curr = sup_start_price + support_slope * (i - sup_start_idx)
            downside = curr_close < sup_at_curr * (1.0 - self.confirmation_filter)

            if upside:
                signals[i] = 1
            elif downside:
                signals[i] = -1

        return signals

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
        return slope

    def _find_horizontal_resistance(
        self, peaks: List[Tuple[int, float]]
    ) -> Optional[Tuple[List[Tuple[int, float]], float]]:
        """
        Find horizontal resistance level from peaks.

        Returns:
            Tuple of (qualifying peaks, resistance level) or None
        """
        if len(peaks) < self.min_touches:
            return None

        # Sort peaks by price
        sorted_peaks = sorted(peaks, key=lambda x: x[1])

        # Try to find clusters of peaks at similar levels
        for start_idx in range(len(sorted_peaks) - self.min_touches + 1):
            candidate_peaks = [sorted_peaks[start_idx]]
            resistance_level = sorted_peaks[start_idx][1]

            for idx in range(start_idx + 1, len(sorted_peaks)):
                peak = sorted_peaks[idx]
                price_diff = abs(peak[1] - resistance_level) / resistance_level

                if price_diff <= self.resistance_tolerance:
                    candidate_peaks.append(peak)

            if len(candidate_peaks) >= self.min_touches:
                # Calculate average resistance level
                avg_resistance = float(np.mean([p[1] for p in candidate_peaks]))
                return (candidate_peaks, avg_resistance)

        return None

    def _find_rising_support(
        self, troughs: List[Tuple[int, float]], min_slope: float
    ) -> Optional[Tuple[List[Tuple[int, float]], float]]:
        """
        Find rising support line from troughs.

        Returns:
            Tuple of (qualifying troughs, slope) or None
        """
        if len(troughs) < self.min_touches:
            return None

        # Sort troughs by index (chronological order)
        sorted_troughs = sorted(troughs, key=lambda x: x[0])

        # Calculate slope of all troughs
        slope = self._calculate_slope(sorted_troughs)

        # Check if slope is positive (rising)
        if slope < min_slope:
            return None

        return (sorted_troughs, slope)

    def _find_pattern(
        self, arrays: dict, peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]], i: int
    ) -> Optional[Dict]:
        """
        Find Ascending Triangle pattern.

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

        # Find horizontal resistance
        resistance_result = self._find_horizontal_resistance(peaks)
        if resistance_result is None:
            return None

        resistance_peaks, resistance_level = resistance_result

        # Find rising support
        support_result = self._find_rising_support(troughs, self.min_slope)
        if support_result is None:
            return None

        support_troughs, support_slope = support_result

        # Check pattern duration
        all_indices = [p[0] for p in resistance_peaks] + [t[0] for t in support_troughs]
        pattern_start = min(all_indices)
        pattern_end = max(all_indices)
        pattern_duration = pattern_end - pattern_start

        if pattern_duration < self.min_pattern_bars or pattern_duration > self.max_pattern_bars:
            return None

        # Get pattern boundaries
        lowest_trough = min(t[1] for t in support_troughs)
        pattern_depth = resistance_level - lowest_trough

        if pattern_depth <= 0:
            return None

        # Check for upside breakout
        current_close = float(close_arr[i])
        current_high = float(high_arr[i])

        # Upside breakout: close above resistance with confirmation
        upside_breakout = current_close > resistance_level * (1 + self.confirmation_filter)

        # Downside breakout: close below support line (less common for ascending triangle)
        # Calculate support line value at current bar
        support_start_idx = support_troughs[0][0]
        support_start_price = support_troughs[0][1]
        support_at_current = support_start_price + support_slope * (i - support_start_idx)
        downside_breakout = current_close < support_at_current * (1 - self.confirmation_filter)

        if upside_breakout:
            return {
                "direction": "bullish",
                "resistance_level": resistance_level,
                "resistance_peaks": resistance_peaks,
                "support_troughs": support_troughs,
                "support_slope": support_slope,
                "lowest_trough": lowest_trough,
                "pattern_depth": pattern_depth,
                "pattern_start": pattern_start,
                "pattern_end": i,
                "breakout_bar": i,
            }
        elif downside_breakout:
            return {
                "direction": "bearish",
                "resistance_level": resistance_level,
                "resistance_peaks": resistance_peaks,
                "support_troughs": support_troughs,
                "support_slope": support_slope,
                "lowest_trough": lowest_trough,
                "pattern_depth": pattern_depth,
                "pattern_start": pattern_start,
                "pattern_end": i,
                "breakout_bar": i,
            }

        return None

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Ascending Triangle pattern at bar index i.

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
                "support_slope": pattern["support_slope"],
                "lowest_trough": pattern["lowest_trough"],
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
        """Generate Ascending Triangle signal."""
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        current_high = float(high_arr[i])
        current_low = float(low_arr[i])
        resistance_level = pattern["resistance_level"]
        lowest_trough = pattern["lowest_trough"]
        pattern_depth = pattern["pattern_depth"]
        direction = pattern["direction"]
        pattern_start = pattern["pattern_start"]
        pattern_end = pattern["pattern_end"]

        # Volume validation: high volume on breakout confirms pattern
        vol_arr = arrays.get("volume", None)
        volume_confirmed = False
        if vol_arr is not None and i < len(vol_arr):
            vol_sma20 = volume_sma(pd.Series(vol_arr[: i + 1], index=range(i + 1)), 20)
            current_vol = float(vol_arr[i]) if vol_arr[i] is not None else 0.0
            avg_vol = (
                float(vol_sma20.iloc[-1])
                if len(vol_sma20) > 0 and not pd.isna(vol_sma20.iloc[-1])
                else 0.0
            )
            if avg_vol > 0:
                volume_confirmed = current_vol > avg_vol

        if direction == "bullish":
            # Upside breakout - LONG signal
            entry_price = current_high + self.entry_offset
            stop_loss = lowest_trough - self.stop_offset
            take_profit_1 = entry_price + (pattern_depth * 0.62)
            take_profit_2 = entry_price + pattern_depth
            take_profit_3 = entry_price + (pattern_depth * 1.27)

            confidence = 0.65
            if volume_confirmed:
                confidence += 0.10

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
                    "pattern_depth": pattern_depth,
                    "entry_type": "buy_stop",
                },
            )
        else:
            # Downside breakout - SHORT signal (less common)
            entry_price = current_low - self.entry_offset
            stop_loss = resistance_level + self.stop_offset
            take_profit_1 = entry_price - (pattern_depth * 0.62)
            take_profit_2 = entry_price - pattern_depth
            take_profit_3 = entry_price - (pattern_depth * 1.27)

            confidence = 0.50
            if volume_confirmed:
                confidence += 0.10

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
                    "pattern_depth": pattern_depth,
                    "entry_type": "sell_stop",
                },
            )
