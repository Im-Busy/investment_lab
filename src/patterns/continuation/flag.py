"""
Flag Pattern (Bullish/Bearish)

Detection Logic:
- Flag Pole: Sharp, steep price move (up or down), slope > threshold over N bars
- Flag: Small parallel channel sloping against the trend
- Duration: Typically 1-4 weeks
- Consolidation slopes opposite to the pole direction
- Breakout occurs in the original trend direction

Entry Rules:
- Long Entry: Buy Stop = high[breakout_bar] + filter (for bullish flag)
- Short Entry: Sell Stop = low[breakdown_bar] - filter (for bearish flag)
- Trade in direction of flag pole

Stop Loss Rules:
- Long Stop: Below the flag low or pole start
- Short Stop: Above the flag high or pole start

Take Profit Rules:
- Target: breakout_price + flag_pole_height
- Flag Pole Height = pole_end_price - pole_start_price
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...indicators.pivots import find_swing_highs, find_swing_lows
from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class Flag(BasePattern):
    """
    Flag Pattern Detector

    A continuation pattern consisting of a sharp price move (flag pole)
    followed by a short consolidation (flag) that slopes against the trend.
    """

    def __init__(
        self,
        lookback: int = 5,
        pole_bars: int = 4,
        pole_min_slope: float = 0.02,
        flag_bars_min: int = 5,
        flag_bars_max: int = 20,
        flag_slope_threshold: float = 0.005,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        confirmation_filter: float = 0.005,
    ):
        """
        Initialize Flag pattern detector.

        Args:
            lookback: Lookback period for pivot detection
            pole_bars: Minimum bars for flag pole
            pole_min_slope: Minimum slope percentage for pole (default 2%)
            flag_bars_min: Minimum bars for flag formation
            flag_bars_max: Maximum bars for flag formation
            flag_slope_threshold: Maximum slope for flag (should be opposite to pole)
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            confirmation_filter: Minimum breakout percentage for confirmation
        """
        super().__init__(
            name="Flag", pattern_type=PatternType.CONTINUATION, min_bars_required=pole_bars + flag_bars_min
        )
        self.lookback = lookback
        self.pole_bars = pole_bars
        self.pole_min_slope = pole_min_slope
        self.flag_bars_min = flag_bars_min
        self.flag_bars_max = flag_bars_max
        self.flag_slope_threshold = flag_slope_threshold
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

        lookback = min(self.flag_bars_max + self.pole_bars + 20, i)

        for j in range(i - lookback, i + 1):
            if j < 0:
                continue
            if pd.notna(swing_highs.iloc[j]):
                peaks.append((j, self._safe_float(swing_highs.iloc[j])))
            if pd.notna(swing_lows.iloc[j]):
                troughs.append((j, self._safe_float(swing_lows.iloc[j])))

        return peaks, troughs

    def _calculate_slope(self, start_idx: int, start_price: float, end_idx: int, end_price: float) -> float:
        """Calculate slope between two points."""
        if end_idx == start_idx:
            return 0.0
        return (end_price - start_price) / (end_idx - start_idx)

    def _find_pattern(
        self, arrays: dict, peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]], i: int
    ) -> Optional[Dict]:
        """
        Find Flag pattern.

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

        # Need enough history
        if i < self.pole_bars + self.flag_bars_min:
            return None

        # Look for bullish flag (pole up, flag down)
        bullish_flag = self._find_bullish_flag(arrays, peaks, troughs, i)
        if bullish_flag:
            return bullish_flag

        # Look for bearish flag (pole down, flag up)
        bearish_flag = self._find_bearish_flag(arrays, peaks, troughs, i)
        if bearish_flag:
            return bearish_flag

        return None

    def _find_bullish_flag(
        self, arrays: dict, peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]], i: int
    ) -> Optional[Dict]:
        """Find bullish flag pattern (pole up, flag slopes down)."""
        close_arr = arrays["close"]
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        # Search for pole within valid range
        for pole_start in range(max(0, i - self.flag_bars_max - self.pole_bars - 10), i - self.flag_bars_min - self.pole_bars + 1):
            pole_end = pole_start + self.pole_bars

            # Check if pole is within range
            if pole_end >= i - self.flag_bars_min:
                continue

            # Calculate pole characteristics
            pole_start_price = float(close_arr[pole_start])
            pole_end_price = float(high_arr[pole_end])  # Use high for bullish pole

            # Pole should be going up
            pole_height = pole_end_price - pole_start_price
            if pole_height <= 0:
                continue

            pole_slope = pole_height / (pole_end - pole_start)
            pole_pct = pole_height / pole_start_price if pole_start_price > 0 else 0

            # Check minimum pole slope
            if pole_pct < self.pole_min_slope:
                continue

            # Find flag consolidation
            flag_start = pole_end
            flag_end = i

            if flag_end - flag_start < self.flag_bars_min or flag_end - flag_start > self.flag_bars_max:
                continue

            # Calculate flag slope (should be negative/flat for bullish flag)
            flag_highs = [float(high_arr[j]) for j in range(flag_start, flag_end + 1)]
            flag_lows = [float(low_arr[j]) for j in range(flag_start, flag_end + 1)]

            flag_high_slope = self._calculate_slope(flag_start, flag_highs[0], flag_end, flag_highs[-1])
            flag_low_slope = self._calculate_slope(flag_start, flag_lows[0], flag_end, flag_lows[-1])

            # Flag should slope down or be flat (opposite to pole)
            if flag_high_slope > self.flag_slope_threshold or flag_low_slope > self.flag_slope_threshold:
                continue

            # Check for breakout above flag
            current_close = float(close_arr[i])
            current_high = float(high_arr[i])
            flag_high = max(flag_highs)

            if current_close > flag_high * (1 + self.confirmation_filter):
                return {
                    "direction": "bullish",
                    "pole_start": pole_start,
                    "pole_end": pole_end,
                    "pole_height": pole_height,
                    "pole_start_price": pole_start_price,
                    "pole_end_price": pole_end_price,
                    "flag_start": flag_start,
                    "flag_end": flag_end,
                    "flag_high": flag_high,
                    "flag_low": min(flag_lows),
                    "breakout_bar": i,
                }

        return None

    def _find_bearish_flag(
        self, arrays: dict, peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]], i: int
    ) -> Optional[Dict]:
        """Find bearish flag pattern (pole down, flag slopes up)."""
        close_arr = arrays["close"]
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        # Search for pole within valid range
        for pole_start in range(max(0, i - self.flag_bars_max - self.pole_bars - 10), i - self.flag_bars_min - self.pole_bars + 1):
            pole_end = pole_start + self.pole_bars

            # Check if pole is within range
            if pole_end >= i - self.flag_bars_min:
                continue

            # Calculate pole characteristics
            pole_start_price = float(close_arr[pole_start])
            pole_end_price = float(low_arr[pole_end])  # Use low for bearish pole

            # Pole should be going down
            pole_height = pole_start_price - pole_end_price
            if pole_height <= 0:
                continue

            pole_slope = pole_height / (pole_end - pole_start)
            pole_pct = pole_height / pole_start_price if pole_start_price > 0 else 0

            # Check minimum pole slope
            if pole_pct < self.pole_min_slope:
                continue

            # Find flag consolidation
            flag_start = pole_end
            flag_end = i

            if flag_end - flag_start < self.flag_bars_min or flag_end - flag_start > self.flag_bars_max:
                continue

            # Calculate flag slope (should be positive/flat for bearish flag)
            flag_highs = [float(high_arr[j]) for j in range(flag_start, flag_end + 1)]
            flag_lows = [float(low_arr[j]) for j in range(flag_start, flag_end + 1)]

            flag_high_slope = self._calculate_slope(flag_start, flag_highs[0], flag_end, flag_highs[-1])
            flag_low_slope = self._calculate_slope(flag_start, flag_lows[0], flag_end, flag_lows[-1])

            # Flag should slope up or be flat (opposite to pole)
            if flag_high_slope < -self.flag_slope_threshold or flag_low_slope < -self.flag_slope_threshold:
                continue

            # Check for breakout below flag
            current_close = float(close_arr[i])
            current_low = float(low_arr[i])
            flag_low = min(flag_lows)

            if current_close < flag_low * (1 - self.confirmation_filter):
                return {
                    "direction": "bearish",
                    "pole_start": pole_start,
                    "pole_end": pole_end,
                    "pole_height": pole_height,
                    "pole_start_price": pole_start_price,
                    "pole_end_price": pole_end_price,
                    "flag_start": flag_start,
                    "flag_end": flag_end,
                    "flag_high": max(flag_highs),
                    "flag_low": flag_low,
                    "breakout_bar": i,
                }

        return None

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Flag pattern at bar index i.

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
                "pole_start": pattern["pole_start"],
                "pole_end": pattern["pole_end"],
                "pole_height": pattern["pole_height"],
                "flag_start": pattern["flag_start"],
                "flag_end": pattern["flag_end"],
                "flag_high": pattern["flag_high"],
                "flag_low": pattern["flag_low"],
                "direction": pattern["direction"],
            },
            bars_since_detection=0,
            start_index=pattern["pole_start"],
            end_index=i,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal."""
        result = self.detect(df, i)
        return result.signal if result.detected else None

    def _generate_signal(
        self, df: pd.DataFrame, i: int, pattern: Dict, arrays: dict
    ) -> Optional[TradeSignal]:
        """Generate Flag signal."""
        high_arr = arrays["high"]
        low_arr = arrays["close"]

        current_high = float(high_arr[i])
        current_low = float(low_arr[i])
        pole_height = pattern["pole_height"]
        direction = pattern["direction"]

        if direction == "bullish":
            # Bullish flag breakout - LONG signal
            entry_price = current_high + self.entry_offset
            stop_loss = pattern["flag_low"] - self.stop_offset
            take_profit_1 = entry_price + (pole_height * 0.62)
            take_profit_2 = entry_price + pole_height
            take_profit_3 = entry_price + (pole_height * 1.27)

            confidence = 0.65

            return TradeSignal(
                pattern_name="Bullish Flag",
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
                    "pole_height": pole_height,
                    "flag_bars": pattern["flag_end"] - pattern["flag_start"],
                    "entry_type": "buy_stop",
                },
            )
        else:
            # Bearish flag breakout - SHORT signal
            entry_price = current_low - self.entry_offset
            stop_loss = pattern["flag_high"] + self.stop_offset
            take_profit_1 = entry_price - (pole_height * 0.62)
            take_profit_2 = entry_price - pole_height
            take_profit_3 = entry_price - (pole_height * 1.27)

            confidence = 0.65

            return TradeSignal(
                pattern_name="Bearish Flag",
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
                    "pole_height": pole_height,
                    "flag_bars": pattern["flag_end"] - pattern["flag_start"],
                    "entry_type": "sell_stop",
                },
            )
