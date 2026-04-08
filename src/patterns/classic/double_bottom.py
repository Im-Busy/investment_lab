"""
Double Bottom Pattern

Detection Logic:
- Trough1 = Local Min where low[i] < low[i-5:i-1] and low[i] < low[i+1:i+5]
- Trough2 = Local Min where abs(Trough1 - Trough2) < 0.05 * Trough1 (within 5%)
- Peak = Local Max between Trough1 and Trough2 (neckline level)
- Volume[Trough1] > Volume[Trough2] (volume dissipation on second trough)
- Pattern confirmed when close[i] > Peak (neckline breakout)

Entry Rules:
- Long Entry: Buy Stop = high[breakout_bar] + 0.01
- Entry triggered when close[i] > Peak level
- Entry valid for next 3 to 5 bars after breakout confirmation

Stop Loss Rules:
- Stop = Midpoint of Double Bottom pattern - 0.01
- Midpoint = (Min(Trough1, Trough2) + Peak) / 2
- Alternative Stop = Min(Trough1, Trough2) - 0.01 (conservative)

Take Profit Rules:
- Target1 = Entry + (Peak - Min(Trough1, Trough2))
- Target2 = Entry + 1.27 * (Peak - Min(Trough1, Trough2))
- Target3 = Entry + 1.62 * (Peak - Min(Trough1, Trough2))
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...indicators.pivots import find_swing_highs, find_swing_lows
from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class DoubleBottom(BasePattern):
    """
    Double Bottom Pattern Detector

    A bullish reversal pattern consisting of two troughs at approximately
    the same level, followed by a breakout above the neckline.
    """

    def __init__(
        self,
        lookback: int = 5,
        trough_tolerance: float = 0.05,
        min_pattern_bars: int = 20,
        max_pattern_bars: int = 120,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        volume_filter: bool = False,
    ):
        """
        Initialize Double Bottom pattern detector.

        FIXED: Increased max_pattern_bars from 60 to 120 for daily timeframe.
        Double bottoms can take 3-6 months to form on daily data.

        Args:
            lookback: Lookback period for pivot detection
            trough_tolerance: Maximum trough depth difference (default 5%)
            min_pattern_bars: Minimum bars between troughs
            max_pattern_bars: Maximum bars between troughs (default 120)
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            volume_filter: Require volume dissipation
        """
        super().__init__(
            name="Double Bottom",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=min_pattern_bars,
        )
        self.lookback = lookback
        self.trough_tolerance = trough_tolerance
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.volume_filter = volume_filter

    def _find_troughs_and_peaks(
        self, df: pd.DataFrame, i: int
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Find swing lows and highs for pattern detection."""
        swing_highs = find_swing_highs(df, self.lookback)
        swing_lows = find_swing_lows(df, self.lookback)

        troughs = []
        peaks = []

        lookback = min(self.max_pattern_bars * 2, i)

        for j in range(i - lookback, i + 1):
            if j < 0:
                continue
            if pd.notna(swing_lows.iloc[j]):
                troughs.append((j, self._safe_float(swing_lows.iloc[j])))
            if pd.notna(swing_highs.iloc[j]):
                peaks.append((j, self._safe_float(swing_highs.iloc[j])))

        return troughs, peaks

    def _find_pattern(
        self, troughs: List[Tuple[int, float]], peaks: List[Tuple[int, float]]
    ) -> Optional[Dict]:
        """
        Find Double Bottom pattern.

        Args:
            troughs: List of (index, price) for swing lows
            peaks: List of (index, price) for swing highs

        Returns:
            Dictionary with pattern details or None
        """
        if len(troughs) < 2 or len(peaks) < 1:
            return None

        troughs = sorted(troughs, key=lambda x: x[0])

        # Find two troughs at similar levels
        for i in range(len(troughs) - 1):
            trough1 = troughs[i]

            for j in range(i + 1, len(troughs)):
                trough2 = troughs[j]

                # Check trough depth similarity
                trough_diff = abs(trough1[1] - trough2[1]) / min(trough1[1], trough2[1])

                if trough_diff > self.trough_tolerance:
                    continue

                # Check pattern duration
                pattern_duration = trough2[0] - trough1[0]
                if (
                    pattern_duration < self.min_pattern_bars
                    or pattern_duration > self.max_pattern_bars
                ):
                    continue

                # Find peak between troughs
                neckline_peaks = []
                for peak_idx, peak_price in peaks:
                    if trough1[0] < peak_idx < trough2[0]:
                        neckline_peaks.append((peak_idx, peak_price))

                if not neckline_peaks:
                    continue

                # Find the highest peak (neckline)
                neckline = max(neckline_peaks, key=lambda x: x[1])

                # Calculate pattern depth
                pattern_depth = neckline[1] - min(trough1[1], trough2[1])

                if pattern_depth <= 0:
                    continue

                return {
                    "trough1": trough1,
                    "trough2": trough2,
                    "neckline": neckline,
                    "pattern_depth": pattern_depth,
                    "trough_diff": trough_diff,
                    "pattern_duration": pattern_duration,
                }

        return None

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Double Bottom pattern at bar index i.

        Args:
            df: DataFrame with OHLCV data
            i: Current bar index
            window_start: Optional window start for bounds checking (avoids DataFrame slicing)

        Returns:
            PatternResult with detection status and signal
        """
        if not self._validate_data(df, i, window_start):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # PERFORMANCE OPTIMIZATION: Use NumPy arrays for faster access
        arrays = self._extract_arrays(df)

        # Find troughs and peaks
        troughs, peaks = self._find_troughs_and_peaks(df, i)

        # Find pattern
        pattern = self._find_pattern(troughs, peaks)

        if pattern is None:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Check for volume dissipation
        volume_arr = arrays["volume"]
        trough1_vol = (
            float(volume_arr[pattern["trough1"][0]])
            if len(volume_arr) > pattern["trough1"][0]
            else 0.0
        )
        trough2_vol = (
            float(volume_arr[pattern["trough2"][0]])
            if len(volume_arr) > pattern["trough2"][0]
            else 0.0
        )
        volume_dissipating = trough2_vol < trough1_vol

        # Check for breakout
        close_arr = arrays["close"]
        current_close = float(close_arr[i])
        neckline_level = pattern["neckline"][1]

        breakout = current_close > neckline_level

        if not breakout:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                pivot_points={
                    "pattern_detected": True,
                    "awaiting_breakout": True,
                    "neckline_level": neckline_level,
                    "trough1": pattern["trough1"][1],
                    "trough2": pattern["trough2"][1],
                },
            )

        # Generate signal
        signal = self._generate_signal(df, i, pattern, volume_dissipating, arrays)

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                "trough1_idx": pattern["trough1"][0],
                "trough1": pattern["trough1"][1],
                "trough2_idx": pattern["trough2"][0],
                "trough2": pattern["trough2"][1],
                "neckline_idx": pattern["neckline"][0],
                "neckline": pattern["neckline"][1],
                "pattern_depth": pattern["pattern_depth"],
                "volume_dissipating": volume_dissipating,
            },
            bars_since_detection=0,
            start_index=pattern["trough1"][0],
            end_index=i,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal."""
        result = self.detect(df, i)
        return result.signal if result.detected else None

    def _generate_signal(
        self, df: pd.DataFrame, i: int, pattern: Dict, volume_dissipating: bool, arrays: dict
    ) -> Optional[TradeSignal]:
        """Generate trade signal for Double Bottom breakout."""
        high_arr = arrays["high"]

        current_high = float(high_arr[i])
        neckline = pattern["neckline"][1]
        trough_low = min(pattern["trough1"][1], pattern["trough2"][1])
        pattern_depth = pattern["pattern_depth"]

        # Entry above breakout bar high
        entry_price = current_high + self.entry_offset

        # Stop at midpoint of pattern or below troughs
        midpoint = (trough_low + neckline) / 2
        stop_loss = midpoint - self.stop_offset

        # Targets based on pattern depth
        take_profit_1 = entry_price + pattern_depth
        take_profit_2 = entry_price + (pattern_depth * 1.27)
        take_profit_3 = entry_price + (pattern_depth * 1.62)

        # Confidence
        confidence = 0.55
        if volume_dissipating:
            confidence += 0.1

        # Check for prior downtrend
        if pattern["trough1"][0] > 10:
            high_slice = high_arr[pattern["trough1"][0] - 10 : pattern["trough1"][0]]
            prior_high = float(np.max(high_slice))
            if trough_low < prior_high * 0.9:  # 10% downtrend
                confidence += 0.05

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
                "pattern_depth": pattern_depth,
                "neckline": neckline,
                "trough_diff": pattern["trough_diff"],
                "volume_dissipating": volume_dissipating,
                "entry_type": "buy_stop",
            },
        )
