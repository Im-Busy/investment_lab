"""
Pennant Pattern (Bullish/Bearish)

Detection Logic:
- Flag Pole: Sharp, steep price move (up or down), slope > threshold over N bars
- Pennant: Small symmetrical triangle sloping against the trend
- Duration: Typically 1-3 weeks
- Converging trendlines (opposite slopes)
- Breakout occurs in the original trend direction

Entry Rules:
- Long Entry: Buy Stop = high[breakout_bar] + filter (for bullish pennant)
- Short Entry: Sell Stop = low[breakdown_bar] - filter (for bearish pennant)
- Trade in direction of flag pole

Stop Loss Rules:
- Long Stop: Below the pennant low or pole start
- Short Stop: Above the pennant high or pole start

Take Profit Rules:
- Target: breakout_price + flag_pole_height
- Flag Pole Height = pole_end_price - pole_start_price
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...indicators.pivots import find_swing_highs, find_swing_lows
from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class Pennant(BasePattern):
    """
    Pennant Pattern Detector

    A continuation pattern consisting of a sharp price move (flag pole)
    followed by a short symmetrical triangle consolidation (pennant).
    """

    def __init__(
        self,
        lookback: int = 5,
        pole_bars: int = 4,
        pole_min_slope: float = 0.02,
        pennant_bars_min: int = 5,
        pennant_bars_max: int = 20,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        confirmation_filter: float = 0.005,
    ):
        """
        Initialize Pennant pattern detector.

        Args:
            lookback: Lookback period for pivot detection
            pole_bars: Minimum bars for flag pole
            pole_min_slope: Minimum slope percentage for pole (default 2%)
            pennant_bars_min: Minimum bars for pennant formation
            pennant_bars_max: Maximum bars for pennant formation
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            confirmation_filter: Minimum breakout percentage for confirmation
        """
        super().__init__(
            name="Pennant",
            pattern_type=PatternType.CONTINUATION,
            min_bars_required=pole_bars + pennant_bars_min,
        )
        self.lookback = lookback
        self.pole_bars = pole_bars
        self.pole_min_slope = pole_min_slope
        self.pennant_bars_min = pennant_bars_min
        self.pennant_bars_max = pennant_bars_max
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.confirmation_filter = confirmation_filter

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized detection of Pennant patterns across the entire DataFrame.

        Returns:
            np.ndarray of np.int8: 0=no signal, 1=LONG (bullish pennant), -1=SHORT (bearish pennant)
        """
        n = len(df)
        result = np.zeros(n, dtype=np.int8)
        min_bars = self.pole_bars + self.pennant_bars_min
        if n < min_bars:
            return result

        close_a = df["Close"].to_numpy()
        high_a = df["High"].to_numpy()
        low_a = df["Low"].to_numpy()
        pb = self.pole_bars
        pmin = self.pole_min_slope
        bmin = self.pennant_bars_min
        bmax = self.pennant_bars_max
        confirm = self.confirmation_filter

        for i in range(min_bars, n):
            search_start = max(0, i - bmax - pb - 10)

            # Bullish pennant: pole up, converging triangle
            for pole_start in range(search_start, i - bmin - pb + 1):
                pole_end = pole_start + pb
                if pole_end >= i - bmin:
                    continue
                pole_start_price = close_a[pole_start]
                pole_end_price = high_a[pole_end]
                pole_height = pole_end_price - pole_start_price
                if pole_height <= 0:
                    continue
                pole_pct = pole_height / pole_start_price if pole_start_price > 0 else 0.0
                if pole_pct < pmin:
                    continue

                pennant_start = pole_end
                pennant_end = i
                penn_len = pennant_end - pennant_start
                if penn_len < bmin or penn_len > bmax:
                    continue

                penn_highs = high_a[pennant_start : pennant_end + 1]
                penn_lows = low_a[pennant_start : pennant_end + 1]

                upper_slope = (penn_highs[-1] - penn_highs[0]) / penn_len
                lower_slope = (penn_lows[-1] - penn_lows[0]) / penn_len

                if upper_slope >= 0.0 or lower_slope <= 0.0:
                    continue

                pennant_high = float(np.max(penn_highs))
                if close_a[i] > pennant_high * (1.0 + confirm):
                    result[i] = 1
                    break

            if result[i] != 0:
                continue

            # Bearish pennant: pole down, converging triangle
            for pole_start in range(search_start, i - bmin - pb + 1):
                pole_end = pole_start + pb
                if pole_end >= i - bmin:
                    continue
                pole_start_price = close_a[pole_start]
                pole_end_price = low_a[pole_end]
                pole_height = pole_start_price - pole_end_price
                if pole_height <= 0:
                    continue
                pole_pct = pole_height / pole_start_price if pole_start_price > 0 else 0.0
                if pole_pct < pmin:
                    continue

                pennant_start = pole_end
                pennant_end = i
                penn_len = pennant_end - pennant_start
                if penn_len < bmin or penn_len > bmax:
                    continue

                penn_highs = high_a[pennant_start : pennant_end + 1]
                penn_lows = low_a[pennant_start : pennant_end + 1]

                upper_slope = (penn_highs[-1] - penn_highs[0]) / penn_len
                lower_slope = (penn_lows[-1] - penn_lows[0]) / penn_len

                if upper_slope >= 0.0 or lower_slope <= 0.0:
                    continue

                pennant_low = float(np.min(penn_lows))
                if close_a[i] < pennant_low * (1.0 - confirm):
                    result[i] = -1
                    break

        return result

    def _find_pivots(
        self, df: pd.DataFrame, i: int
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Find swing highs and lows for pattern detection."""
        swing_highs = find_swing_highs(df, self.lookback)
        swing_lows = find_swing_lows(df, self.lookback)

        peaks = []
        troughs = []

        lookback = min(self.pennant_bars_max + self.pole_bars + 20, i)

        for j in range(i - lookback, i + 1):
            if j < 0:
                continue
            if pd.notna(swing_highs.iloc[j]):
                peaks.append((j, self._safe_float(swing_highs.iloc[j])))
            if pd.notna(swing_lows.iloc[j]):
                troughs.append((j, self._safe_float(swing_lows.iloc[j])))

        return peaks, troughs

    def _calculate_slope(
        self, start_idx: int, start_price: float, end_idx: int, end_price: float
    ) -> float:
        """Calculate slope between two points."""
        if end_idx == start_idx:
            return 0.0
        return (end_price - start_price) / (end_idx - start_idx)

    def _find_pattern(
        self, arrays: dict, peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]], i: int
    ) -> Optional[Dict]:
        """
        Find Pennant pattern.

        Args:
            arrays: Dictionary with NumPy arrays
            peaks: List of (index, price) for swing highs
            troughs: List of (index, price) for swing lows
            i: Current bar index

        Returns:
            Dictionary with pattern details or None
        """
        close_arr = arrays["close"]

        # Need enough history
        if i < self.pole_bars + self.pennant_bars_min:
            return None

        # Look for bullish pennant (pole up, pennant converges)
        bullish_pennant = self._find_bullish_pennant(arrays, peaks, troughs, i)
        if bullish_pennant:
            return bullish_pennant

        # Look for bearish pennant (pole down, pennant converges)
        bearish_pennant = self._find_bearish_pennant(arrays, peaks, troughs, i)
        if bearish_pennant:
            return bearish_pennant

        return None

    def _find_bullish_pennant(
        self, arrays: dict, peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]], i: int
    ) -> Optional[Dict]:
        """Find bullish pennant pattern (pole up, converging triangle)."""
        close_arr = arrays["close"]
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        # Search for pole within valid range
        for pole_start in range(
            max(0, i - self.pennant_bars_max - self.pole_bars - 10),
            i - self.pennant_bars_min - self.pole_bars + 1,
        ):
            pole_end = pole_start + self.pole_bars

            # Check if pole is within range
            if pole_end >= i - self.pennant_bars_min:
                continue

            # Calculate pole characteristics
            pole_start_price = float(close_arr[pole_start])
            pole_end_price = float(high_arr[pole_end])  # Use high for bullish pole

            # Pole should be going up
            pole_height = pole_end_price - pole_start_price
            if pole_height <= 0:
                continue

            pole_pct = pole_height / pole_start_price if pole_start_price > 0 else 0

            # Check minimum pole slope
            if pole_pct < self.pole_min_slope:
                continue

            # Find pennant consolidation
            pennant_start = pole_end
            pennant_end = i

            if (
                pennant_end - pennant_start < self.pennant_bars_min
                or pennant_end - pennant_start > self.pennant_bars_max
            ):
                continue

            # Calculate pennant trendlines (converging)
            pennant_highs = [float(high_arr[j]) for j in range(pennant_start, pennant_end + 1)]
            pennant_lows = [float(low_arr[j]) for j in range(pennant_start, pennant_end + 1)]

            # Upper trendline should slope down
            upper_slope = self._calculate_slope(
                pennant_start, pennant_highs[0], pennant_end, pennant_highs[-1]
            )

            # Lower trendline should slope up
            lower_slope = self._calculate_slope(
                pennant_start, pennant_lows[0], pennant_end, pennant_lows[-1]
            )

            # Pennant should be converging (upper down, lower up)
            if upper_slope >= 0 or lower_slope <= 0:
                continue

            # Check for breakout above pennant
            current_close = float(close_arr[i])
            current_high = float(high_arr[i])
            pennant_high = max(pennant_highs)

            if current_close > pennant_high * (1 + self.confirmation_filter):
                return {
                    "direction": "bullish",
                    "pole_start": pole_start,
                    "pole_end": pole_end,
                    "pole_height": pole_height,
                    "pole_start_price": pole_start_price,
                    "pole_end_price": pole_end_price,
                    "pennant_start": pennant_start,
                    "pennant_end": pennant_end,
                    "pennant_high": pennant_high,
                    "pennant_low": min(pennant_lows),
                    "upper_slope": upper_slope,
                    "lower_slope": lower_slope,
                    "breakout_bar": i,
                }

        return None

    def _find_bearish_pennant(
        self, arrays: dict, peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]], i: int
    ) -> Optional[Dict]:
        """Find bearish pennant pattern (pole down, converging triangle)."""
        close_arr = arrays["close"]
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        # Search for pole within valid range
        for pole_start in range(
            max(0, i - self.pennant_bars_max - self.pole_bars - 10),
            i - self.pennant_bars_min - self.pole_bars + 1,
        ):
            pole_end = pole_start + self.pole_bars

            # Check if pole is within range
            if pole_end >= i - self.pennant_bars_min:
                continue

            # Calculate pole characteristics
            pole_start_price = float(close_arr[pole_start])
            pole_end_price = float(low_arr[pole_end])  # Use low for bearish pole

            # Pole should be going down
            pole_height = pole_start_price - pole_end_price
            if pole_height <= 0:
                continue

            pole_pct = pole_height / pole_start_price if pole_start_price > 0 else 0

            # Check minimum pole slope
            if pole_pct < self.pole_min_slope:
                continue

            # Find pennant consolidation
            pennant_start = pole_end
            pennant_end = i

            if (
                pennant_end - pennant_start < self.pennant_bars_min
                or pennant_end - pennant_start > self.pennant_bars_max
            ):
                continue

            # Calculate pennant trendlines (converging)
            pennant_highs = [float(high_arr[j]) for j in range(pennant_start, pennant_end + 1)]
            pennant_lows = [float(low_arr[j]) for j in range(pennant_start, pennant_end + 1)]

            # Upper trendline should slope down
            upper_slope = self._calculate_slope(
                pennant_start, pennant_highs[0], pennant_end, pennant_highs[-1]
            )

            # Lower trendline should slope up
            lower_slope = self._calculate_slope(
                pennant_start, pennant_lows[0], pennant_end, pennant_lows[-1]
            )

            # Pennant should be converging (upper down, lower up)
            if upper_slope >= 0 or lower_slope <= 0:
                continue

            # Check for breakout below pennant
            current_close = float(close_arr[i])
            current_low = float(low_arr[i])
            pennant_low = min(pennant_lows)

            if current_close < pennant_low * (1 - self.confirmation_filter):
                return {
                    "direction": "bearish",
                    "pole_start": pole_start,
                    "pole_end": pole_end,
                    "pole_height": pole_height,
                    "pole_start_price": pole_start_price,
                    "pole_end_price": pole_end_price,
                    "pennant_start": pennant_start,
                    "pennant_end": pennant_end,
                    "pennant_high": max(pennant_highs),
                    "pennant_low": pennant_low,
                    "upper_slope": upper_slope,
                    "lower_slope": lower_slope,
                    "breakout_bar": i,
                }

        return None

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Pennant pattern at bar index i.

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
                "pennant_start": pattern["pennant_start"],
                "pennant_end": pattern["pennant_end"],
                "pennant_high": pattern["pennant_high"],
                "pennant_low": pattern["pennant_low"],
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
        """Generate Pennant signal."""
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        current_high = float(high_arr[i])
        current_low = float(low_arr[i])
        pole_height = pattern["pole_height"]
        direction = pattern["direction"]

        if direction == "bullish":
            # Bullish pennant breakout - LONG signal
            entry_price = current_high + self.entry_offset
            stop_loss = pattern["pennant_low"] - self.stop_offset
            take_profit_1 = entry_price + (pole_height * 0.62)
            take_profit_2 = entry_price + pole_height
            take_profit_3 = entry_price + (pole_height * 1.27)

            confidence = 0.65

            return TradeSignal(
                pattern_name="Bullish Pennant",
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
                    "pennant_bars": pattern["pennant_end"] - pattern["pennant_start"],
                    "entry_type": "buy_stop",
                },
            )
        else:
            # Bearish pennant breakout - SHORT signal
            entry_price = current_low - self.entry_offset
            stop_loss = pattern["pennant_high"] + self.stop_offset
            take_profit_1 = entry_price - (pole_height * 0.62)
            take_profit_2 = entry_price - pole_height
            take_profit_3 = entry_price - (pole_height * 1.27)

            confidence = 0.65

            return TradeSignal(
                pattern_name="Bearish Pennant",
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
                    "pennant_bars": pattern["pennant_end"] - pattern["pennant_start"],
                    "entry_type": "sell_stop",
                },
            )
