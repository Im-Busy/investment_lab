"""
Head and Shoulders Pattern

Detection Logic:
- Identify 3 Peaks: Left_Shoulder, Head, Right_Shoulder using Local Extrema (lookback n=5)
- Condition 1 (Head): High[Head] > High[Left_Shoulder] AND High[Head] > High[Right_Shoulder]
- Condition 2 (Shoulders): abs(High[Left_Shoulder] - High[Right_Shoulder]) < 0.10 * High[Head] (within 10%)
- Condition 3 (Neckline): Line connecting Low between Left_Shoulder/Head AND Low between Head/Right_Shoulder
- Neckline_Slope: Can be horizontal or angled (slanted)
- Pattern Complete when price approaches neckline after Right_Shoulder formation

Entry Rules:
- Short Entry: Sell Stop = Low[Breakdown_Bar] - 0.01
- Entry triggered when close[i] < Neckline_Level
- Breakdown bar must close below neckline

Stop Loss Rules:
- Stop Loss: Neckline_Level + 0.01
- Alternative Stop: High[Right_Shoulder] + 0.01

Take Profit Rules:
- Target 1: Entry - 0.62 * Pattern_Depth
- Target 2: Entry - 1.00 * Pattern_Depth
- Pattern_Depth = High[Head] - Neckline_Level
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...indicators.pivots import find_swing_highs, find_swing_lows
from ...indicators.technical import volume_sma
from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class HeadAndShoulders(BasePattern):
    """
    Head and Shoulders Pattern Detector

    A bearish reversal pattern consisting of three peaks with the
    middle peak (head) higher than the two shoulders.
    """

    def __init__(
        self,
        lookback: int = 5,
        shoulder_tolerance: float = 0.10,
        min_pattern_bars: int = 30,
        max_pattern_bars: int = 180,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        volume_filter: bool = False,
    ):
        """
        Initialize Head and Shoulders pattern detector.

        FIXED: Increased max_pattern_bars from 120 to 180 for daily timeframe.
        Head and shoulders can take 4-6 months to form on daily data.

        Args:
            lookback: Lookback period for pivot detection
            shoulder_tolerance: Maximum shoulder height difference (default 10%)
            min_pattern_bars: Minimum bars for pattern formation
            max_pattern_bars: Maximum bars for pattern formation (default 180)
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            volume_filter: Require volume confirmation
        """
        super().__init__(
            name="Head and Shoulders",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=min_pattern_bars,
        )
        self.lookback = lookback
        self.shoulder_tolerance = shoulder_tolerance
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.volume_filter = volume_filter

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized Head and Shoulders detection across all bars.

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

        min_bars = max(self.lookback, self.min_pattern_bars)

        for i in range(min_bars, n):
            lookback = min(self.max_pattern_bars, i)

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

            peaks_s = sorted(peaks, key=lambda x: x[0])

            pattern_found = False
            neckline_val = 0.0

            for ip in range(len(peaks_s) - 2):
                ls = peaks_s[ip]
                for jp in range(ip + 1, len(peaks_s) - 1):
                    head = peaks_s[jp]
                    if head[1] <= ls[1]:
                        continue
                    for kp in range(jp + 1, len(peaks_s)):
                        rs = peaks_s[kp]
                        if head[1] <= rs[1]:
                            continue
                        shoulder_diff = abs(ls[1] - rs[1]) / head[1]
                        if shoulder_diff > self.shoulder_tolerance:
                            continue
                        nl_troughs = [(ti, tp) for ti, tp in troughs if ls[0] < ti < rs[0]]
                        if len(nl_troughs) < 2:
                            continue
                        nl_troughs_s = sorted(nl_troughs, key=lambda x: x[0])
                        t1 = None
                        t2 = None
                        for t in nl_troughs_s:
                            if ls[0] < t[0] < head[0]:
                                if t1 is None or t[1] < t1[1]:
                                    t1 = t
                            elif head[0] < t[0] < rs[0]:
                                if t2 is None or t[1] < t2[1]:
                                    t2 = t
                        if t1 is None or t2 is None:
                            continue
                        nl_slope = (t2[1] - t1[1]) / (t2[0] - t1[0]) if t2[0] != t1[0] else 0.0
                        nl_intercept = t1[1] - nl_slope * t1[0]
                        depth = head[1] - (nl_slope * rs[0] + nl_intercept)
                        if depth <= 0:
                            continue
                        neckline_val = nl_slope * i + nl_intercept
                        pattern_found = True
                        break
                    if pattern_found:
                        break
                if pattern_found:
                    break

            if pattern_found:
                curr_close = float(close[i])
                if curr_close < neckline_val:
                    signals[i] = -1

        return signals

    def _find_peaks_and_troughs(
        self, df: pd.DataFrame, i: int
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """
        Find swing highs and lows for pattern detection.

        Args:
            df: DataFrame with OHLC data
            i: Current bar index

        Returns:
            Tuple of (peaks, troughs) lists
        """
        swing_highs = find_swing_highs(df, self.lookback)
        swing_lows = find_swing_lows(df, self.lookback)

        peaks = []
        troughs = []

        lookback = min(self.max_pattern_bars, i)

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
        Find Head and Shoulders pattern from peaks and troughs.

        Args:
            peaks: List of (index, price) for swing highs
            troughs: List of (index, price) for swing lows

        Returns:
            Dictionary with pattern details or None
        """
        if len(peaks) < 3 or len(troughs) < 2:
            return None

        # Sort peaks by index (chronological)
        peaks = sorted(peaks, key=lambda x: x[0])

        # Try to find three consecutive peaks where middle is highest
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]

            for j in range(i + 1, len(peaks) - 1):
                head = peaks[j]

                # Head must be higher than left shoulder
                if head[1] <= left_shoulder[1]:
                    continue

                for k in range(j + 1, len(peaks)):
                    right_shoulder = peaks[k]

                    # Head must be higher than right shoulder
                    if head[1] <= right_shoulder[1]:
                        continue

                    # Shoulders should be similar height (within tolerance)
                    shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / head[1]
                    if shoulder_diff > self.shoulder_tolerance:
                        continue

                    # Find troughs between peaks for neckline
                    neckline_troughs = []
                    for trough_idx, trough_price in troughs:
                        if left_shoulder[0] < trough_idx < right_shoulder[0]:
                            neckline_troughs.append((trough_idx, trough_price))

                    if len(neckline_troughs) < 2:
                        continue

                    # Sort troughs by index
                    neckline_troughs = sorted(neckline_troughs, key=lambda x: x[0])

                    # Find the two main troughs (between left shoulder-head and head-right shoulder)
                    trough1 = None  # Between left shoulder and head
                    trough2 = None  # Between head and right shoulder

                    for trough in neckline_troughs:
                        if left_shoulder[0] < trough[0] < head[0]:
                            if trough1 is None or trough[1] < trough1[1]:
                                trough1 = trough
                        elif head[0] < trough[0] < right_shoulder[0]:
                            if trough2 is None or trough[1] < trough2[1]:
                                trough2 = trough

                    if trough1 is None or trough2 is None:
                        continue

                    # Calculate neckline
                    neckline_slope = (
                        (trough2[1] - trough1[1]) / (trough2[0] - trough1[0])
                        if trough2[0] != trough1[0]
                        else 0
                    )
                    neckline_intercept = trough1[1] - (neckline_slope * trough1[0])

                    # Pattern depth
                    pattern_depth = head[1] - (
                        (neckline_slope * right_shoulder[0]) + neckline_intercept
                    )

                    if pattern_depth <= 0:
                        continue

                    return {
                        "left_shoulder": left_shoulder,
                        "head": head,
                        "right_shoulder": right_shoulder,
                        "neckline_trough1": trough1,
                        "neckline_trough2": trough2,
                        "neckline_slope": neckline_slope,
                        "neckline_intercept": neckline_intercept,
                        "pattern_depth": pattern_depth,
                        "shoulder_diff": shoulder_diff,
                    }

        return None

    def _get_neckline_value(self, pattern: Dict, idx: int) -> float:
        """
        Calculate neckline value at a given index.

        Args:
            pattern: Pattern dictionary
            idx: Bar index

        Returns:
            Neckline price at index
        """
        return float((pattern["neckline_slope"] * idx) + pattern["neckline_intercept"])

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Head and Shoulders pattern at bar index i.

        Args:
            df: DataFrame with OHLCV data
            i: Current bar index

        Returns:
            PatternResult with detection status and signal
        """
        if not self._validate_data(df, i):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Find peaks and troughs
        peaks, troughs = self._find_peaks_and_troughs(df, i)

        # Find pattern
        pattern = self._find_pattern(peaks, troughs)

        if pattern is None:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Check pattern duration
        pattern_duration = pattern["right_shoulder"][0] - pattern["left_shoulder"][0]
        if pattern_duration < self.min_pattern_bars or pattern_duration > self.max_pattern_bars:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Check for breakdown below neckline
        current_close = self._safe_float(df.iloc[i]["Close"])
        neckline_value = self._get_neckline_value(pattern, i)

        breakdown = current_close < neckline_value

        if not breakdown:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                pivot_points={
                    "pattern_detected": True,
                    "awaiting_breakdown": True,
                    "neckline_value": neckline_value,
                    "current_close": current_close,
                },
            )

        # Generate signal
        signal = self._generate_signal(df, i, pattern, neckline_value)

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                "left_shoulder_idx": pattern["left_shoulder"][0],
                "left_shoulder": pattern["left_shoulder"][1],
                "head_idx": pattern["head"][0],
                "head": pattern["head"][1],
                "right_shoulder_idx": pattern["right_shoulder"][0],
                "right_shoulder": pattern["right_shoulder"][1],
                "neckline_value": neckline_value,
                "pattern_depth": pattern["pattern_depth"],
                "neckline_slope": pattern["neckline_slope"],
            },
            bars_since_detection=0,
            start_index=pattern["left_shoulder"][0],
            end_index=i,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal."""
        result = self.detect(df, i)
        return result.signal if result.detected else None

    def _generate_signal(
        self, df: pd.DataFrame, i: int, pattern: Dict, neckline_value: float
    ) -> Optional[TradeSignal]:
        """Generate trade signal for Head and Shoulders breakdown."""

        current_low = self._safe_float(df.iloc[i]["Low"])
        head_high = pattern["head"][1]
        right_shoulder_high = pattern["right_shoulder"][1]
        pattern_depth = pattern["pattern_depth"]

        # Entry below breakdown bar low
        entry_price = current_low - self.entry_offset

        # Stop loss above neckline or right shoulder
        stop_loss = max(neckline_value, right_shoulder_high) + self.stop_offset

        # Targets based on pattern depth
        take_profit_1 = entry_price - (pattern_depth * 0.62)
        take_profit_2 = entry_price - pattern_depth

        # Volume confirmation
        volume_confirmed = True
        volume_dissipation = True

        if self.volume_filter:
            vol_sma = volume_sma(df["Volume"], 20)
            if i < len(vol_sma):
                current_vol = self._safe_float(df.iloc[i]["Volume"])
                avg_vol = self._safe_float(vol_sma.iloc[i])
                volume_confirmed = current_vol > avg_vol

            # Check volume dissipation from left to right shoulder
            left_shoulder_vol = self._safe_float(df.iloc[pattern["left_shoulder"][0]]["Volume"])
            head_vol = self._safe_float(df.iloc[pattern["head"][0]]["Volume"])
            right_shoulder_vol = self._safe_float(df.iloc[pattern["right_shoulder"][0]]["Volume"])

            if left_shoulder_vol > 0 and right_shoulder_vol > 0:
                volume_dissipation = right_shoulder_vol < left_shoulder_vol

        # Confidence
        confidence = 0.55
        if volume_confirmed:
            confidence += 0.1
        if volume_dissipation:
            confidence += 0.1

        return TradeSignal(
            pattern_name=self.name,
            direction=SignalDirection.SHORT,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=None,
            confidence=min(confidence, 1.0),
            timestamp=df.index[i] if hasattr(df, "index") and i < len(df.index) else None,
            metadata={
                "pattern_depth": pattern_depth,
                "neckline_value": neckline_value,
                "volume_confirmed": volume_confirmed,
                "volume_dissipation": volume_dissipation,
                "entry_type": "sell_stop",
            },
        )


class InverseHeadAndShoulders(BasePattern):
    """
    Inverse Head and Shoulders Pattern Detector

    A bullish reversal pattern consisting of three troughs with the
    middle trough (head) lower than the two shoulders.
    """

    def __init__(
        self,
        lookback: int = 5,
        shoulder_tolerance: float = 0.10,
        min_pattern_bars: int = 30,
        max_pattern_bars: int = 180,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        volume_filter: bool = False,
    ):
        """
        FIXED: Increased max_pattern_bars from 120 to 180 for daily timeframe.
        """
        super().__init__(
            name="Inverse Head and Shoulders",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=min_pattern_bars,
        )
        self.lookback = lookback
        self.shoulder_tolerance = shoulder_tolerance
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.volume_filter = volume_filter

    def _find_peaks_and_troughs(
        self, df: pd.DataFrame, i: int
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Find swing highs and lows for pattern detection."""
        swing_highs = find_swing_highs(df, self.lookback)
        swing_lows = find_swing_lows(df, self.lookback)

        peaks = []
        troughs = []

        lookback = min(self.max_pattern_bars, i)

        for j in range(i - lookback, i + 1):
            if j < 0:
                continue
            if pd.notna(swing_highs.iloc[j]):
                peaks.append((j, self._safe_float(swing_highs.iloc[j])))
            if pd.notna(swing_lows.iloc[j]):
                troughs.append((j, self._safe_float(swing_lows.iloc[j])))

        return peaks, troughs

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized Inverse Head and Shoulders detection across all bars.

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

        min_bars = max(self.lookback, self.min_pattern_bars)

        for i in range(min_bars, n):
            lookback = min(self.max_pattern_bars, i)

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

            troughs_s = sorted(troughs, key=lambda x: x[0])

            pattern_found = False
            neckline_val = 0.0

            for it in range(len(troughs_s) - 2):
                ls = troughs_s[it]
                for jt in range(it + 1, len(troughs_s) - 1):
                    head = troughs_s[jt]
                    if head[1] >= ls[1]:
                        continue
                    for kt in range(jt + 1, len(troughs_s)):
                        rs = troughs_s[kt]
                        if head[1] >= rs[1]:
                            continue
                        shoulder_diff = abs(ls[1] - rs[1]) / abs(head[1])
                        if shoulder_diff > self.shoulder_tolerance:
                            continue
                        nl_peaks = [(pi, pp) for pi, pp in peaks if ls[0] < pi < rs[0]]
                        if len(nl_peaks) < 2:
                            continue
                        nl_peaks_s = sorted(nl_peaks, key=lambda x: x[0])
                        p1 = None
                        p2 = None
                        for p in nl_peaks_s:
                            if ls[0] < p[0] < head[0]:
                                if p1 is None or p[1] > p1[1]:
                                    p1 = p
                            elif head[0] < p[0] < rs[0]:
                                if p2 is None or p[1] > p2[1]:
                                    p2 = p
                        if p1 is None or p2 is None:
                            continue
                        nl_slope = (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] != p1[0] else 0.0
                        nl_intercept = p1[1] - nl_slope * p1[0]
                        depth = (nl_slope * rs[0] + nl_intercept) - head[1]
                        if depth <= 0:
                            continue
                        neckline_val = nl_slope * i + nl_intercept
                        pattern_found = True
                        break
                    if pattern_found:
                        break
                if pattern_found:
                    break

            if pattern_found:
                curr_close = float(close[i])
                if curr_close > neckline_val:
                    signals[i] = 1

        return signals

    def _find_pattern(
        self, peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]]
    ) -> Optional[Dict]:
        """Find Inverse Head and Shoulders pattern."""
        if len(troughs) < 3 or len(peaks) < 2:
            return None

        troughs = sorted(troughs, key=lambda x: x[0])

        # Find three troughs where middle is lowest
        for i in range(len(troughs) - 2):
            left_shoulder = troughs[i]

            for j in range(i + 1, len(troughs) - 1):
                head = troughs[j]

                # Head must be lower than left shoulder
                if head[1] >= left_shoulder[1]:
                    continue

                for k in range(j + 1, len(troughs)):
                    right_shoulder = troughs[k]

                    # Head must be lower than right shoulder
                    if head[1] >= right_shoulder[1]:
                        continue

                    # Shoulders should be similar depth
                    shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / abs(head[1])
                    if shoulder_diff > self.shoulder_tolerance:
                        continue

                    # Find peaks between troughs for neckline
                    neckline_peaks = []
                    for peak_idx, peak_price in peaks:
                        if left_shoulder[0] < peak_idx < right_shoulder[0]:
                            neckline_peaks.append((peak_idx, peak_price))

                    if len(neckline_peaks) < 2:
                        continue

                    neckline_peaks = sorted(neckline_peaks, key=lambda x: x[0])

                    # Find the two main peaks
                    peak1 = None
                    peak2 = None

                    for peak in neckline_peaks:
                        if left_shoulder[0] < peak[0] < head[0]:
                            if peak1 is None or peak[1] > peak1[1]:
                                peak1 = peak
                        elif head[0] < peak[0] < right_shoulder[0]:
                            if peak2 is None or peak[1] > peak2[1]:
                                peak2 = peak

                    if peak1 is None or peak2 is None:
                        continue

                    # Calculate neckline
                    neckline_slope = (
                        (peak2[1] - peak1[1]) / (peak2[0] - peak1[0]) if peak2[0] != peak1[0] else 0
                    )
                    neckline_intercept = peak1[1] - (neckline_slope * peak1[0])

                    # Pattern depth
                    pattern_depth = (
                        (neckline_slope * right_shoulder[0]) + neckline_intercept
                    ) - head[1]

                    if pattern_depth <= 0:
                        continue

                    return {
                        "left_shoulder": left_shoulder,
                        "head": head,
                        "right_shoulder": right_shoulder,
                        "neckline_peak1": peak1,
                        "neckline_peak2": peak2,
                        "neckline_slope": neckline_slope,
                        "neckline_intercept": neckline_intercept,
                        "pattern_depth": pattern_depth,
                        "shoulder_diff": shoulder_diff,
                    }

        return None

    def _get_neckline_value(self, pattern: Dict, idx: int) -> float:
        """Calculate neckline value at a given index."""
        return float((pattern["neckline_slope"] * idx) + pattern["neckline_intercept"])

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """Detect Inverse Head and Shoulders pattern at bar index i."""
        if not self._validate_data(df, i):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        peaks, troughs = self._find_peaks_and_troughs(df, i)
        pattern = self._find_pattern(peaks, troughs)

        if pattern is None:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        pattern_duration = pattern["right_shoulder"][0] - pattern["left_shoulder"][0]
        if pattern_duration < self.min_pattern_bars or pattern_duration > self.max_pattern_bars:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        current_close = self._safe_float(df.iloc[i]["Close"])
        neckline_value = self._get_neckline_value(pattern, i)

        breakout = current_close > neckline_value

        if not breakout:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                pivot_points={
                    "pattern_detected": True,
                    "awaiting_breakout": True,
                    "neckline_value": neckline_value,
                },
            )

        signal = self._generate_signal(df, i, pattern, neckline_value)

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                "left_shoulder_idx": pattern["left_shoulder"][0],
                "left_shoulder": pattern["left_shoulder"][1],
                "head_idx": pattern["head"][0],
                "head": pattern["head"][1],
                "right_shoulder_idx": pattern["right_shoulder"][0],
                "right_shoulder": pattern["right_shoulder"][1],
                "neckline_value": neckline_value,
                "pattern_depth": pattern["pattern_depth"],
            },
            bars_since_detection=0,
            start_index=pattern["left_shoulder"][0],
            end_index=i,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal."""
        result = self.detect(df, i)
        return result.signal if result.detected else None

    def _generate_signal(
        self, df: pd.DataFrame, i: int, pattern: Dict, neckline_value: float
    ) -> Optional[TradeSignal]:
        """Generate trade signal for Inverse Head and Shoulders breakout."""

        current_high = self._safe_float(df.iloc[i]["High"])
        head_low = pattern["head"][1]
        right_shoulder_low = pattern["right_shoulder"][1]
        pattern_depth = pattern["pattern_depth"]

        entry_price = current_high + self.entry_offset
        stop_loss = min(neckline_value, right_shoulder_low) - self.stop_offset

        take_profit_1 = entry_price + (pattern_depth * 0.62)
        take_profit_2 = entry_price + pattern_depth

        confidence = 0.55

        return TradeSignal(
            pattern_name=self.name,
            direction=SignalDirection.LONG,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=None,
            confidence=min(confidence, 1.0),
            timestamp=df.index[i] if hasattr(df, "index") and i < len(df.index) else None,
            metadata={
                "pattern_depth": pattern_depth,
                "neckline_value": neckline_value,
                "entry_type": "buy_stop",
            },
        )
