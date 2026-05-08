"""
Triple Top Pattern

Detection Logic:
- Peak1, Peak2, Peak3 = Three local maxima at similar price levels (within 3%)
- Trough1, Trough2 = Two troughs between the three peaks (neckline levels)
- Volume[Peak1] > Volume[Peak2] > Volume[Peak3] (diminishing volume)
- Pattern confirmed when close[i] < min(Trough1, Trough2) (neckline breakdown)

Entry Rules:
- Short Entry: Sell Stop = low[breakdown_bar] - 0.01
- Entry triggered when close[i] < neckline level
- Entry valid for next 3 to 5 bars after breakdown confirmation

Stop Loss Rules:
- Stop = Highest peak + 0.01
- Alternative Stop = Pattern midpoint + 0.01 (conservative)

Take Profit Rules:
- Target = Entry - (Highest Peak - Neckline)
- Target = 100% of pattern depth from breakdown level
- Pattern Depth = Highest Peak - Neckline
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...indicators.pivots import find_swing_highs, find_swing_lows
from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class TripleTop(BasePattern):
    """
    Triple Top Pattern Detector

    A bearish reversal pattern consisting of three peaks at approximately
    the same level, followed by a breakdown below the neckline.
    """

    def __init__(
        self,
        lookback: int = 5,
        peak_tolerance: float = 0.03,
        min_pattern_bars: int = 15,
        max_pattern_bars: int = 100,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        volume_filter: bool = False,
    ):
        """
        Initialize Triple Top pattern detector.

        Args:
            lookback: Lookback period for pivot detection
            peak_tolerance: Maximum peak height difference (default 3%)
            min_pattern_bars: Minimum bars between first and last peak
            max_pattern_bars: Maximum bars between first and last peak
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            volume_filter: Require volume dissipation across peaks
        """
        super().__init__(
            name="Triple Top", pattern_type=PatternType.REVERSAL, min_bars_required=min_pattern_bars
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
        Vectorized Triple Top detection across all bars.

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

            if len(peaks) < 3 or len(troughs) < 2:
                continue

            peaks_sorted = sorted(peaks, key=lambda x: x[0])

            pattern_found = False
            neckline = 0.0

            for ip1 in range(len(peaks_sorted) - 2):
                p1 = peaks_sorted[ip1]
                for ip2 in range(ip1 + 1, len(peaks_sorted) - 1):
                    p2 = peaks_sorted[ip2]
                    for ip3 in range(ip2 + 1, len(peaks_sorted)):
                        p3 = peaks_sorted[ip3]
                        duration = p3[0] - p1[0]
                        if duration < self.min_pattern_bars or duration > self.max_pattern_bars:
                            continue
                        max_p = max(p1[1], p2[1], p3[1])
                        min_p = min(p1[1], p2[1], p3[1])
                        if max_p == 0:
                            continue
                        if (max_p - min_p) / max_p > self.peak_tolerance:
                            continue

                        t1_candidates = [(ti, tp) for ti, tp in troughs if p1[0] < ti < p2[0]]
                        t2_candidates = [(ti, tp) for ti, tp in troughs if p2[0] < ti < p3[0]]
                        if not t1_candidates or not t2_candidates:
                            continue
                        t1 = min(t1_candidates, key=lambda x: x[1])
                        t2 = min(t2_candidates, key=lambda x: x[1])
                        neckline = min(t1[1], t2[1])

                        if float(close[i]) < neckline:
                            pattern_found = True
                            break
                    if pattern_found:
                        break
                if pattern_found:
                    break

            if pattern_found:
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

    def _check_peaks_similar(
        self, peak1: Tuple[int, float], peak2: Tuple[int, float], peak3: Tuple[int, float]
    ) -> bool:
        """Check if three peaks are at similar price levels."""
        max_price = max(peak1[1], peak2[1], peak3[1])
        min_price = min(peak1[1], peak2[1], peak3[1])

        if max_price == 0:
            return False

        price_range = (max_price - min_price) / max_price
        return price_range <= self.peak_tolerance

    def _check_volume_diminishing(
        self, arrays: dict, peak1_idx: int, peak2_idx: int, peak3_idx: int
    ) -> bool:
        """Check if volume is diminishing across peaks."""
        volume_arr = arrays["volume"]
        if len(volume_arr) <= max(peak1_idx, peak2_idx, peak3_idx):
            return False

        vol1 = float(volume_arr[peak1_idx])
        vol2 = float(volume_arr[peak2_idx])
        vol3 = float(volume_arr[peak3_idx])

        return vol1 > vol2 > vol3

    def _find_pattern(
        self, arrays: dict, peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]], i: int
    ) -> Optional[Dict]:
        """
        Find Triple Top pattern.

        Args:
            arrays: Dictionary with NumPy arrays (open, high, low, close, volume)
            peaks: List of (index, price) for swing highs
            troughs: List of (index, price) for swing lows
            i: Current bar index

        Returns:
            Dictionary with pattern details or None
        """
        close_arr = arrays["close"]

        if len(peaks) < 3 or len(troughs) < 2:
            return None

        peaks = sorted(peaks, key=lambda x: x[0])

        # Find three peaks at similar levels
        for idx1 in range(len(peaks) - 2):
            peak1 = peaks[idx1]

            for idx2 in range(idx1 + 1, len(peaks) - 1):
                peak2 = peaks[idx2]

                for idx3 in range(idx2 + 1, len(peaks)):
                    peak3 = peaks[idx3]

                    # Check pattern duration
                    pattern_duration = peak3[0] - peak1[0]
                    if (
                        pattern_duration < self.min_pattern_bars
                        or pattern_duration > self.max_pattern_bars
                    ):
                        continue

                    # Check if peaks are at similar levels
                    if not self._check_peaks_similar(peak1, peak2, peak3):
                        continue

                    # Find troughs between peaks
                    trough1_candidates = []
                    trough2_candidates = []

                    for trough_idx, trough_price in troughs:
                        if peak1[0] < trough_idx < peak2[0]:
                            trough1_candidates.append((trough_idx, trough_price))
                        elif peak2[0] < trough_idx < peak3[0]:
                            trough2_candidates.append((trough_idx, trough_price))

                    if not trough1_candidates or not trough2_candidates:
                        continue

                    # Find the lowest troughs (neckline)
                    trough1 = min(trough1_candidates, key=lambda x: x[1])
                    trough2 = min(trough2_candidates, key=lambda x: x[1])
                    neckline = min(trough1[1], trough2[1])

                    # Check for breakdown
                    current_close = float(close_arr[i])

                    if current_close < neckline:
                        # Volume filter check
                        volume_confirmed = True
                        if self.volume_filter:
                            volume_confirmed = self._check_volume_diminishing(
                                arrays, peak1[0], peak2[0], peak3[0]
                            )

                        highest_peak = max(peak1[1], peak2[1], peak3[1])

                        return {
                            "peak1": peak1,
                            "peak2": peak2,
                            "peak3": peak3,
                            "trough1": trough1,
                            "trough2": trough2,
                            "neckline": neckline,
                            "highest_peak": highest_peak,
                            "pattern_depth": highest_peak - neckline,
                            "volume_confirmed": volume_confirmed,
                            "breakdown_bar": i,
                        }

        return None

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Triple Top pattern at bar index i.

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

        peaks, troughs = self._find_peaks_and_troughs(df, i)
        pattern = self._find_pattern(arrays, peaks, troughs, i)

        if pattern is None:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Generate signal
        signal = self._generate_signal(df, i, pattern, arrays)

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                "peak1_idx": pattern["peak1"][0],
                "peak1_price": pattern["peak1"][1],
                "peak2_idx": pattern["peak2"][0],
                "peak2_price": pattern["peak2"][1],
                "peak3_idx": pattern["peak3"][0],
                "peak3_price": pattern["peak3"][1],
                "neckline": pattern["neckline"],
                "highest_peak": pattern["highest_peak"],
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
        self, df: pd.DataFrame, i: int, pattern: Dict, arrays: dict
    ) -> Optional[TradeSignal]:
        """Generate Triple Top short signal."""
        low_arr = arrays["low"]
        close_arr = arrays["close"]

        current_low = float(low_arr[i])
        neckline = pattern["neckline"]
        highest_peak = pattern["highest_peak"]

        # Entry below breakdown bar low
        entry_price = current_low - self.entry_offset

        # Stop above highest peak
        stop_loss = highest_peak + self.stop_offset

        # Target: 100% pattern depth
        pattern_depth = pattern["pattern_depth"]
        take_profit_1 = entry_price - pattern_depth
        take_profit_2 = entry_price - (pattern_depth * 1.27)
        take_profit_3 = entry_price - (pattern_depth * 1.62)

        # Calculate confidence
        confidence = 0.60

        # Boost confidence if volume is diminishing
        if pattern["volume_confirmed"]:
            confidence += 0.1

        # Check for close below neckline (stronger signal)
        current_close = float(close_arr[i])
        if current_close < neckline:
            confidence += 0.05

        # Check pattern symmetry (peaks evenly spaced)
        peak1_peak2_dist = pattern["peak2"][0] - pattern["peak1"][0]
        peak2_peak3_dist = pattern["peak3"][0] - pattern["peak2"][0]
        if peak1_peak2_dist > 0 and peak2_peak3_dist > 0:
            symmetry = min(peak1_peak2_dist, peak2_peak3_dist) / max(
                peak1_peak2_dist, peak2_peak3_dist
            )
            if symmetry > 0.7:
                confidence += 0.05

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
                "neckline": neckline,
                "highest_peak": highest_peak,
                "pattern_depth": pattern_depth,
                "peak1_idx": pattern["peak1"][0],
                "peak2_idx": pattern["peak2"][0],
                "peak3_idx": pattern["peak3"][0],
                "volume_confirmed": pattern["volume_confirmed"],
                "entry_type": "sell_stop",
            },
        )
