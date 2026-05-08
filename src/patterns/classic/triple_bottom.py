"""
Triple Bottom Pattern

Detection Logic:
- Trough1, Trough2, Trough3 = Three local minima at similar price levels (within 3%)
- Peak1, Peak2 = Two peaks between the three troughs (neckline levels)
- Volume[Trough1] > Volume[Trough2] > Volume[Trough3] (diminishing volume on successive troughs)
- Pattern confirmed when close[i] > max(Peak1, Peak2) (neckline breakout)

Entry Rules:
- Long Entry: Buy Stop = high[breakout_bar] + 0.01
- Entry triggered when close[i] > neckline level
- Entry valid for next 3 to 5 bars after breakout confirmation

Stop Loss Rules:
- Stop = Lowest trough - 0.01
- Alternative Stop = Pattern midpoint - 0.01 (conservative)

Take Profit Rules:
- Target = Entry + (Neckline - Lowest Trough)
- Target = 100% of pattern depth from breakout level
- Pattern Depth = Neckline - Lowest Trough
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...indicators.pivots import find_swing_highs, find_swing_lows
from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class TripleBottom(BasePattern):
    """
    Triple Bottom Pattern Detector

    A bullish reversal pattern consisting of three troughs at approximately
    the same level, followed by a breakout above the neckline.
    """

    def __init__(
        self,
        lookback: int = 5,
        trough_tolerance: float = 0.03,
        min_pattern_bars: int = 15,
        max_pattern_bars: int = 100,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        volume_filter: bool = False,
    ):
        """
        Initialize Triple Bottom pattern detector.

        Args:
            lookback: Lookback period for pivot detection
            trough_tolerance: Maximum trough price difference (default 3%)
            min_pattern_bars: Minimum bars between first and last trough
            max_pattern_bars: Maximum bars between first and last trough
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            volume_filter: Require volume dissipation across troughs
        """
        super().__init__(
            name="Triple Bottom",
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

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized Triple Bottom detection across all bars.

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

            if len(troughs) < 3 or len(peaks) < 2:
                continue

            troughs_sorted = sorted(troughs, key=lambda x: x[0])

            pattern_found = False
            neckline = 0.0

            for it1 in range(len(troughs_sorted) - 2):
                t1 = troughs_sorted[it1]
                for it2 in range(it1 + 1, len(troughs_sorted) - 1):
                    t2 = troughs_sorted[it2]
                    for it3 in range(it2 + 1, len(troughs_sorted)):
                        t3 = troughs_sorted[it3]
                        duration = t3[0] - t1[0]
                        if duration < self.min_pattern_bars or duration > self.max_pattern_bars:
                            continue
                        max_p = max(t1[1], t2[1], t3[1])
                        min_p = min(t1[1], t2[1], t3[1])
                        if max_p == 0:
                            continue
                        if (max_p - min_p) / max_p > self.trough_tolerance:
                            continue

                        p1_candidates = [(pi, pp) for pi, pp in peaks if t1[0] < pi < t2[0]]
                        p2_candidates = [(pi, pp) for pi, pp in peaks if t2[0] < pi < t3[0]]
                        if not p1_candidates or not p2_candidates:
                            continue
                        pk1 = max(p1_candidates, key=lambda x: x[1])
                        pk2 = max(p2_candidates, key=lambda x: x[1])
                        neckline = max(pk1[1], pk2[1])

                        if float(close[i]) > neckline:
                            pattern_found = True
                            break
                    if pattern_found:
                        break
                if pattern_found:
                    break

            if pattern_found:
                signals[i] = 1

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

    def _check_troughs_similar(
        self, trough1: Tuple[int, float], trough2: Tuple[int, float], trough3: Tuple[int, float]
    ) -> bool:
        """Check if three troughs are at similar price levels."""
        max_price = max(trough1[1], trough2[1], trough3[1])
        min_price = min(trough1[1], trough2[1], trough3[1])

        if max_price == 0:
            return False

        price_range = (max_price - min_price) / max_price
        return price_range <= self.trough_tolerance

    def _check_volume_diminishing(
        self, arrays: dict, trough1_idx: int, trough2_idx: int, trough3_idx: int
    ) -> bool:
        """Check if volume is diminishing across troughs."""
        volume_arr = arrays["volume"]
        if len(volume_arr) <= max(trough1_idx, trough2_idx, trough3_idx):
            return False

        vol1 = float(volume_arr[trough1_idx])
        vol2 = float(volume_arr[trough2_idx])
        vol3 = float(volume_arr[trough3_idx])

        return vol1 > vol2 > vol3

    def _find_pattern(
        self, arrays: dict, peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]], i: int
    ) -> Optional[Dict]:
        """
        Find Triple Bottom pattern.

        Args:
            arrays: Dictionary with NumPy arrays (open, high, low, close, volume)
            peaks: List of (index, price) for swing highs
            troughs: List of (index, price) for swing lows
            i: Current bar index

        Returns:
            Dictionary with pattern details or None
        """
        close_arr = arrays["close"]

        if len(troughs) < 3 or len(peaks) < 2:
            return None

        troughs = sorted(troughs, key=lambda x: x[0])

        # Find three troughs at similar levels
        for idx1 in range(len(troughs) - 2):
            trough1 = troughs[idx1]

            for idx2 in range(idx1 + 1, len(troughs) - 1):
                trough2 = troughs[idx2]

                for idx3 in range(idx2 + 1, len(troughs)):
                    trough3 = troughs[idx3]

                    # Check pattern duration
                    pattern_duration = trough3[0] - trough1[0]
                    if (
                        pattern_duration < self.min_pattern_bars
                        or pattern_duration > self.max_pattern_bars
                    ):
                        continue

                    # Check if troughs are at similar levels
                    if not self._check_troughs_similar(trough1, trough2, trough3):
                        continue

                    # Find peaks between troughs
                    peak1_candidates = []
                    peak2_candidates = []

                    for peak_idx, peak_price in peaks:
                        if trough1[0] < peak_idx < trough2[0]:
                            peak1_candidates.append((peak_idx, peak_price))
                        elif trough2[0] < peak_idx < trough3[0]:
                            peak2_candidates.append((peak_idx, peak_price))

                    if not peak1_candidates or not peak2_candidates:
                        continue

                    # Find the highest peaks (neckline)
                    peak1 = max(peak1_candidates, key=lambda x: x[1])
                    peak2 = max(peak2_candidates, key=lambda x: x[1])
                    neckline = max(peak1[1], peak2[1])

                    # Check for breakout
                    current_close = float(close_arr[i])

                    if current_close > neckline:
                        # Volume filter check
                        volume_confirmed = True
                        if self.volume_filter:
                            volume_confirmed = self._check_volume_diminishing(
                                arrays, trough1[0], trough2[0], trough3[0]
                            )

                        lowest_trough = min(trough1[1], trough2[1], trough3[1])

                        return {
                            "trough1": trough1,
                            "trough2": trough2,
                            "trough3": trough3,
                            "peak1": peak1,
                            "peak2": peak2,
                            "neckline": neckline,
                            "lowest_trough": lowest_trough,
                            "pattern_depth": neckline - lowest_trough,
                            "volume_confirmed": volume_confirmed,
                            "breakout_bar": i,
                        }

        return None

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Triple Bottom pattern at bar index i.

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
                "trough1_idx": pattern["trough1"][0],
                "trough1_price": pattern["trough1"][1],
                "trough2_idx": pattern["trough2"][0],
                "trough2_price": pattern["trough2"][1],
                "trough3_idx": pattern["trough3"][0],
                "trough3_price": pattern["trough3"][1],
                "neckline": pattern["neckline"],
                "lowest_trough": pattern["lowest_trough"],
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
        self, df: pd.DataFrame, i: int, pattern: Dict, arrays: dict
    ) -> Optional[TradeSignal]:
        """Generate Triple Bottom long signal."""
        high_arr = arrays["high"]
        close_arr = arrays["close"]

        current_high = float(high_arr[i])
        neckline = pattern["neckline"]
        lowest_trough = pattern["lowest_trough"]

        # Entry above breakout bar high
        entry_price = current_high + self.entry_offset

        # Stop below lowest trough
        stop_loss = lowest_trough - self.stop_offset

        # Target: 100% pattern depth
        pattern_depth = pattern["pattern_depth"]
        take_profit_1 = entry_price + pattern_depth
        take_profit_2 = entry_price + (pattern_depth * 1.27)
        take_profit_3 = entry_price + (pattern_depth * 1.62)

        # Calculate confidence
        confidence = 0.60

        # Boost confidence if volume is diminishing
        if pattern["volume_confirmed"]:
            confidence += 0.1

        # Check for close above neckline (stronger signal)
        current_close = float(close_arr[i])
        if current_close > neckline:
            confidence += 0.05

        # Check pattern symmetry (troughs evenly spaced)
        trough1_trough2_dist = pattern["trough2"][0] - pattern["trough1"][0]
        trough2_trough3_dist = pattern["trough3"][0] - pattern["trough2"][0]
        if trough1_trough2_dist > 0 and trough2_trough3_dist > 0:
            symmetry = min(trough1_trough2_dist, trough2_trough3_dist) / max(
                trough1_trough2_dist, trough2_trough3_dist
            )
            if symmetry > 0.7:
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
                "neckline": neckline,
                "lowest_trough": lowest_trough,
                "pattern_depth": pattern_depth,
                "trough1_idx": pattern["trough1"][0],
                "trough2_idx": pattern["trough2"][0],
                "trough3_idx": pattern["trough3"][0],
                "volume_confirmed": pattern["volume_confirmed"],
                "entry_type": "buy_stop",
            },
        )
