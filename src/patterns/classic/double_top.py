"""
Double Top Pattern

Detection Logic:
- Peak1 = Local Max where high[i] > high[i-5:i-1] and high[i] > high[i+1:i+5]
- Peak2 = Local Max where abs(Peak1 - Peak2) < 0.05 * Peak1 (within 5%)
- Trough = Local Min between Peak1 and Peak2 (neckline level)
- Volume[Peak1] > Volume[Peak2] (volume dissipation on second peak)
- Pattern confirmed when close[i] < Trough (neckline breakdown)

Entry Rules:
- Short Entry: Sell Stop = low[breakdown_bar] - 0.01
- Entry triggered when close[i] < Trough level
- Entry valid for next 3 to 5 bars after breakdown confirmation

Stop Loss Rules:
- Stop = Midpoint of Double Top pattern + 0.01
- Midpoint = (Max(Peak1, Peak2) + Trough) / 2
- Alternative Stop = Max(Peak1, Peak2) + 0.01 (conservative)

Take Profit Rules:
- Target = Entry - (Max(Peak1, Peak2) - Trough)
- Target = 100% of pattern depth from breakdown level
- Pattern Depth = Max(Peak1, Peak2) - Trough
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...indicators.pivots import find_swing_highs, find_swing_lows
from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class DoubleTop(BasePattern):
    """
    Double Top Pattern Detector

    A bearish reversal pattern consisting of two peaks at approximately
    the same level, followed by a breakdown below the neckline.
    """

    def __init__(
        self,
        lookback: int = 5,
        peak_tolerance: float = 0.05,
        min_pattern_bars: int = 20,
        max_pattern_bars: int = 120,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        volume_filter: bool = True,
    ):
        """
        Initialize Double Top pattern detector.

        FIXED: Increased max_pattern_bars from 60 to 120 for daily timeframe.
        Double tops can take 3-6 months to form on daily data.

        Args:
            lookback: Lookback period for pivot detection
            peak_tolerance: Maximum peak height difference (default 5%)
            min_pattern_bars: Minimum bars between peaks
            max_pattern_bars: Maximum bars between peaks (default 120)
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            volume_filter: Require volume dissipation
        """
        super().__init__(
            name="Double Top", pattern_type=PatternType.REVERSAL, min_bars_required=min_pattern_bars
        )
        self.lookback = lookback
        self.peak_tolerance = peak_tolerance
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.volume_filter = volume_filter

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized Double Top detection across all bars.

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

            if len(peaks) < 2 or len(troughs) < 1:
                continue

            peaks_sorted = sorted(peaks, key=lambda x: x[0])

            pattern_found = False
            neckline_idx = 0
            neckline_price = 0.0

            for ip in range(len(peaks_sorted) - 1):
                p1 = peaks_sorted[ip]
                for jp in range(ip + 1, len(peaks_sorted)):
                    p2 = peaks_sorted[jp]
                    peak_diff = abs(p1[1] - p2[1]) / max(p1[1], p2[1])
                    if peak_diff > self.peak_tolerance:
                        continue
                    duration = p2[0] - p1[0]
                    if duration < self.min_pattern_bars or duration > self.max_pattern_bars:
                        continue
                    neckline_troughs = [(ti, tp) for ti, tp in troughs if p1[0] < ti < p2[0]]
                    if not neckline_troughs:
                        continue
                    nl = min(neckline_troughs, key=lambda x: x[1])
                    depth = max(p1[1], p2[1]) - nl[1]
                    if depth <= 0:
                        continue
                    neckline_idx = nl[0]
                    neckline_price = nl[1]
                    pattern_found = True
                    break
                if pattern_found:
                    break

            if not pattern_found:
                continue

            if i > neckline_idx and float(close[i]) < neckline_price:
                signals[i] = -1

        return signals

    def _find_peaks_and_troughs(
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

    def _find_pattern(
        self, peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]]
    ) -> Optional[Dict]:
        """
        Find Double Top pattern.

        Args:
            peaks: List of (index, price) for swing highs
            troughs: List of (index, price) for swing lows

        Returns:
            Dictionary with pattern details or None
        """
        if len(peaks) < 2 or len(troughs) < 1:
            return None

        peaks = sorted(peaks, key=lambda x: x[0])

        # Find two peaks at similar levels
        for i in range(len(peaks) - 1):
            peak1 = peaks[i]

            for j in range(i + 1, len(peaks)):
                peak2 = peaks[j]

                # Check peak height similarity
                peak_diff = abs(peak1[1] - peak2[1]) / max(peak1[1], peak2[1])

                if peak_diff > self.peak_tolerance:
                    continue

                # Check pattern duration
                pattern_duration = peak2[0] - peak1[0]
                if (
                    pattern_duration < self.min_pattern_bars
                    or pattern_duration > self.max_pattern_bars
                ):
                    continue

                # Find trough between peaks
                neckline_troughs = []
                for trough_idx, trough_price in troughs:
                    if peak1[0] < trough_idx < peak2[0]:
                        neckline_troughs.append((trough_idx, trough_price))

                if not neckline_troughs:
                    continue

                # Find the lowest trough (neckline)
                neckline = min(neckline_troughs, key=lambda x: x[1])

                # Calculate pattern depth
                pattern_depth = max(peak1[1], peak2[1]) - neckline[1]

                if pattern_depth <= 0:
                    continue

                return {
                    "peak1": peak1,
                    "peak2": peak2,
                    "neckline": neckline,
                    "pattern_depth": pattern_depth,
                    "peak_diff": peak_diff,
                    "pattern_duration": pattern_duration,
                }

        return None

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Double Top pattern at bar index i.

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

        # Find peaks and troughs
        peaks, troughs = self._find_peaks_and_troughs(df, i)

        # Find pattern
        pattern = self._find_pattern(peaks, troughs)

        if pattern is None:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Check for volume dissipation
        volume_arr = arrays["volume"]
        peak1_vol = (
            float(volume_arr[pattern["peak1"][0]]) if len(volume_arr) > pattern["peak1"][0] else 0.0
        )
        peak2_vol = (
            float(volume_arr[pattern["peak2"][0]]) if len(volume_arr) > pattern["peak2"][0] else 0.0
        )
        volume_dissipating = peak2_vol < peak1_vol

        # Check for breakdown
        close_arr = arrays["close"]
        current_close = float(close_arr[i])
        neckline_level = pattern["neckline"][1]

        breakdown = current_close < neckline_level

        if not breakdown:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                pivot_points={
                    "pattern_detected": True,
                    "awaiting_breakdown": True,
                    "neckline_level": neckline_level,
                    "peak1": pattern["peak1"][1],
                    "peak2": pattern["peak2"][1],
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
                "peak1_idx": pattern["peak1"][0],
                "peak1": pattern["peak1"][1],
                "peak2_idx": pattern["peak2"][0],
                "peak2": pattern["peak2"][1],
                "neckline_idx": pattern["neckline"][0],
                "neckline": pattern["neckline"][1],
                "pattern_depth": pattern["pattern_depth"],
                "volume_dissipating": volume_dissipating,
            },
            bars_since_detection=0,
            start_index=pattern["peak1"][0],
            end_index=i,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal."""
        result = self.detect(df, i)
        return result.signal if result.detected else None

    def _generate_signal(
        self, df: pd.DataFrame, i: int, pattern: Dict, volume_dissipating: bool, arrays: dict
    ) -> Optional[TradeSignal]:
        """Generate trade signal for Double Top breakdown."""
        low_arr = arrays["low"]

        current_low = float(low_arr[i])
        neckline = pattern["neckline"][1]
        peak_high = max(pattern["peak1"][1], pattern["peak2"][1])
        pattern_depth = pattern["pattern_depth"]

        # Entry below breakdown bar low
        entry_price = current_low - self.entry_offset

        # Stop at midpoint of pattern or above peaks
        midpoint = (peak_high + neckline) / 2
        stop_loss = midpoint + self.stop_offset

        # Target based on pattern depth
        take_profit_1 = entry_price - pattern_depth
        take_profit_2 = entry_price - (pattern_depth * 1.27)

        # Confidence
        confidence = 0.55
        if volume_dissipating:
            confidence += 0.1

        # Check for prior uptrend
        if pattern["peak1"][0] > 10:
            low_slice = low_arr[pattern["peak1"][0] - 10 : pattern["peak1"][0]]
            prior_low = float(np.min(low_slice))
            if peak_high > prior_low * 1.1:  # 10% uptrend
                confidence += 0.05

        return TradeSignal(
            pattern_name=self.name,
            direction=SignalDirection.SHORT,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=None,
            confidence=min(confidence, 1.0),
            timestamp=df.index[i] if hasattr(df, "index") else None,
            metadata={
                "pattern_depth": pattern_depth,
                "neckline": neckline,
                "peak_diff": pattern["peak_diff"],
                "volume_dissipating": volume_dissipating,
                "entry_type": "sell_stop",
            },
        )
