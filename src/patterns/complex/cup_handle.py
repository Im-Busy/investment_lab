"""
Cup and Handle Pattern

Detection Logic:
- Identify Cup Formation: Price rallies from left rim to right rim with rounded bottom
- Cup Depth = Max(High during cup formation) - Min(Low during cup formation)
- Handle Formation: Price correction after right rim of cup
- Handle Retracement: 0.25 * Cup_Depth <= (Right_Rim_High - Handle_Low) <= 0.38 * Cup_Depth
- Handle Duration: Minimum 5 bars, Maximum 20 bars (tunable parameter)
- Pattern Complete when price approaches right rim level again

Entry Rules:
- Long Entry: Buy Stop = High[Cup_Right_Rim] + 0.01
- Entry triggered when close[i] > High[Cup_Right_Rim]
- Entry valid for next 3 to 5 bars after breakout

Stop Loss Rules:
- Stop Loss: Low[Handle] - 0.01
- Alternative Stop: Low[Cup] - 0.01 (for wider stops)

Take Profit Rules:
- Target 1: Entry + 0.62 * Cup_Depth
- Target 2: Entry + 1.00 * Cup_Depth
- Cup_Depth = High[Cup_Right_Rim] - Low[Cup_Bottom]
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ...indicators.pivots import find_swing_highs, find_swing_lows
from ...indicators.technical import volume_sma
from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class CupAndHandle(BasePattern):
    """
    Cup and Handle Pattern Detector

    A bullish continuation pattern consisting of a cup-shaped price
    formation followed by a smaller handle consolidation.
    """

    def __init__(
        self,
        min_cup_bars: int = 20,
        max_cup_bars: int = 100,
        min_handle_bars: int = 5,
        max_handle_bars: int = 20,
        handle_retracement_min: float = 0.25,
        handle_retracement_max: float = 0.38,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        volume_filter: bool = False,
    ):
        """
        Initialize Cup and Handle pattern detector.

        Args:
            min_cup_bars: Minimum bars for cup formation
            max_cup_bars: Maximum bars for cup formation
            min_handle_bars: Minimum bars for handle formation
            max_handle_bars: Maximum bars for handle formation
            handle_retracement_min: Minimum handle retracement (default 25%)
            handle_retracement_max: Maximum handle retracement (default 38%)
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            volume_filter: Require volume confirmation
        """
        super().__init__(
            name="Cup and Handle",
            pattern_type=PatternType.CONTINUATION,
            min_bars_required=min_cup_bars + min_handle_bars,
        )
        self.min_cup_bars = min_cup_bars
        self.max_cup_bars = max_cup_bars
        self.min_handle_bars = min_handle_bars
        self.max_handle_bars = max_handle_bars
        self.handle_retracement_min = handle_retracement_min
        self.handle_retracement_max = handle_retracement_max
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.volume_filter = volume_filter

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized Cup and Handle detection across all bars.

        Returns np.int8 array: 0=no signal, 1=long, -1=short.
        """
        n = len(df)
        signals = np.zeros(n, dtype=np.int8)
        close = df["Close"].to_numpy(dtype=np.float64)
        high = df["High"].to_numpy(dtype=np.float64)
        low = df["Low"].to_numpy(dtype=np.float64)

        swing_highs = find_swing_highs(df, 5)
        swing_lows = find_swing_lows(df, 5)

        sh_arr = swing_highs.to_numpy(dtype=np.float64)
        sl_arr = swing_lows.to_numpy(dtype=np.float64)
        n_sh = pd.notna(swing_highs).to_numpy()
        n_sl = pd.notna(swing_lows).to_numpy()

        min_bars = max(self.min_cup_bars, self.min_handle_bars)

        for i in range(min_bars, n):
            lookback = min(self.max_cup_bars, i)

            highs: list = []
            lows: list = []
            for j in range(i - lookback, i + 1):
                if j < 0:
                    continue
                if n_sh[j]:
                    highs.append((j, float(sh_arr[j])))
                if n_sl[j]:
                    lows.append((j, float(sl_arr[j])))

            if len(highs) < 2 or len(lows) < 1:
                continue

            # Find cup
            cup_found = False
            right_rim = None
            cup_bottom = None
            cup_depth_val = 0.0

            for ridx in range(len(highs) - 1, 0, -1):
                rr = highs[ridx]
                for lidx in range(ridx - 1, -1, -1):
                    lr = highs[lidx]
                    rim_diff = abs(lr[1] - rr[1]) / lr[1]
                    if rim_diff > 0.05:
                        continue
                    cb = None
                    for li, lp in lows:
                        if lr[0] < li < rr[0]:
                            if cb is None or lp < cb[1]:
                                cb = (li, lp)
                    if cb is None:
                        continue
                    cup_d = min(lr[1], rr[1]) - cb[1]
                    avg_rim = (lr[1] + rr[1]) / 2.0
                    if cup_d <= 0 or cup_d > avg_rim * 0.5:
                        continue
                    duration = rr[0] - lr[0]
                    if duration < self.min_cup_bars or duration > self.max_cup_bars:
                        continue
                    cup_found = True
                    right_rim = rr
                    cup_bottom = cb
                    cup_depth_val = cup_d
                    break
                if cup_found:
                    break

            if not cup_found:
                continue

            # Find handle
            h_start = right_rim[0]
            if i - h_start < self.min_handle_bars or i - h_start > self.max_handle_bars:
                continue

            handle_high = right_rim[1]
            handle_low = float("inf")
            for j in range(h_start, i + 1):
                h_val = float(high[j])
                l_val = float(low[j])
                if h_val > handle_high:
                    handle_high = h_val
                if l_val < handle_low:
                    handle_low = l_val

            if handle_low < cup_bottom[1]:
                continue

            h_retrace = (handle_high - handle_low) / cup_depth_val if cup_depth_val > 0 else 0.0
            if not (self.handle_retracement_min <= h_retrace <= self.handle_retracement_max):
                continue

            # Breakout check
            curr_close = float(close[i])
            if curr_close > right_rim[1]:
                signals[i] = 1

        return signals

    def _find_cup_formation(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        """
        Find cup formation in price data.

        Args:
            df: DataFrame with OHLCV data
            i: Current bar index

        Returns:
            Dictionary with cup formation details or None
        """
        if i < self.min_cup_bars:
            return None

        # Find swing highs and lows
        swing_highs = find_swing_highs(df, 5)
        swing_lows = find_swing_lows(df, 5)

        # Look for left rim (high), bottom (low), right rim (high)
        # The cup should have a rounded bottom

        # Get recent swing points
        highs = []
        lows = []

        lookback = min(self.max_cup_bars, i)

        for j in range(i - lookback, i + 1):
            if j < 0:
                continue
            if pd.notna(swing_highs.iloc[j]):
                highs.append((j, self._safe_float(swing_highs.iloc[j])))
            if pd.notna(swing_lows.iloc[j]):
                lows.append((j, self._safe_float(swing_lows.iloc[j])))

        if len(highs) < 2 or len(lows) < 1:
            return None

        # Find the most recent high (potential right rim)
        # and a prior high of similar level (potential left rim)
        # with a low in between (cup bottom)

        for right_idx in range(len(highs) - 1, 0, -1):
            right_rim = highs[right_idx]

            for left_idx in range(right_idx - 1, -1, -1):
                left_rim = highs[left_idx]

                # Check rim similarity (within 5%)
                rim_diff = abs(left_rim[1] - right_rim[1]) / left_rim[1]
                if rim_diff > 0.05:
                    continue

                # Find lowest point between rims
                cup_bottom = None
                cup_bottom_idx = None

                for low_idx, low_price in lows:
                    if left_rim[0] < low_idx < right_rim[0]:
                        if cup_bottom is None or low_price < cup_bottom:
                            cup_bottom = low_price
                            cup_bottom_idx = low_idx

                if cup_bottom is None or cup_bottom_idx is None:
                    continue

                assert cup_bottom_idx is not None, "cup_bottom_idx must be set"

                # Check cup depth (should be meaningful but not too deep)
                cup_depth = min(left_rim[1], right_rim[1]) - cup_bottom
                avg_rim = (left_rim[1] + right_rim[1]) / 2

                if cup_depth <= 0 or cup_depth > avg_rim * 0.5:
                    continue

                # Check cup duration
                cup_duration = right_rim[0] - left_rim[0]
                if cup_duration < self.min_cup_bars or cup_duration > self.max_cup_bars:
                    continue

                # Check for rounded bottom (series of higher lows after bottom)
                arrays = self._extract_arrays(df)
                rounded = self._check_rounded_bottom(arrays, cup_bottom_idx, right_rim[0])

                return {
                    "left_rim": left_rim,
                    "right_rim": right_rim,
                    "cup_bottom": (cup_bottom_idx, cup_bottom),
                    "cup_depth": cup_depth,
                    "cup_duration": cup_duration,
                    "rounded_bottom": rounded,
                }

        return None

    def _check_rounded_bottom(self, arrays: dict, bottom_idx: int, right_rim_idx: int) -> bool:
        """
        Check if the cup has a rounded bottom (higher lows after bottom).

        Args:
            arrays: Dictionary with NumPy arrays (open, high, low, close, volume)
            bottom_idx: Index of cup bottom
            right_rim_idx: Index of right rim

        Returns:
            True if rounded bottom detected
        """
        if bottom_idx >= right_rim_idx:
            return False

        low_arr = arrays["low"]

        # Check for series of higher lows
        higher_lows_count = 0
        prev_low = float(low_arr[bottom_idx])

        for j in range(bottom_idx + 1, right_rim_idx):
            curr_low = float(low_arr[j])
            if curr_low > prev_low:
                higher_lows_count += 1
            prev_low = min(prev_low, curr_low)

        # At least 50% of bars should have higher lows
        total_bars = right_rim_idx - bottom_idx - 1
        return total_bars > 0 and higher_lows_count / total_bars >= 0.5

    def _find_handle_formation(self, arrays: dict, i: int, cup: Dict) -> Optional[Dict]:
        """
        Find handle formation after cup.

        Args:
            arrays: Dictionary with NumPy arrays (open, high, low, close, volume)
            i: Current bar index
            cup: Cup formation dictionary

        Returns:
            Dictionary with handle formation details or None
        """
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        right_rim = cup["right_rim"]
        cup_depth = cup["cup_depth"]

        # Handle should form after right rim
        handle_start = right_rim[0]

        if i - handle_start < self.min_handle_bars:
            return None

        if i - handle_start > self.max_handle_bars:
            return None

        # Find handle high and low
        handle_high = right_rim[1]  # Start from right rim
        handle_low = float("inf")
        handle_low_idx = None

        for j in range(handle_start, i + 1):
            high = float(high_arr[j])
            low = float(low_arr[j])

            if high > handle_high:
                handle_high = high
            if low < handle_low:
                handle_low = low
                handle_low_idx = j

        if handle_low_idx is None:
            return None

        # Calculate handle retracement
        handle_retracement = (handle_high - handle_low) / cup_depth if cup_depth > 0 else 0

        # Check handle retracement bounds
        if not (self.handle_retracement_min <= handle_retracement <= self.handle_retracement_max):
            return None

        # Handle should be a consolidation (not a new downtrend)
        # Check if price is still above cup bottom
        if handle_low < cup["cup_bottom"][1]:
            return None

        return {
            "handle_start": handle_start,
            "handle_high": handle_high,
            "handle_low": handle_low,
            "handle_low_idx": handle_low_idx,
            "handle_retracement": handle_retracement,
            "handle_duration": i - handle_start,
        }

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Cup and Handle pattern at bar index i.

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

        # Find cup formation
        cup = self._find_cup_formation(df, i)

        if cup is None:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Find handle formation
        handle = self._find_handle_formation(arrays, i, cup)

        if handle is None:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                pivot_points={"cup_detected": True, "cup": cup},
            )

        # Check for breakout
        close_arr = arrays["close"]
        current_close = float(close_arr[i])
        right_rim_high = cup["right_rim"][1]

        breakout = current_close > right_rim_high

        if not breakout:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                pivot_points={
                    "cup_detected": True,
                    "handle_detected": True,
                    "cup": cup,
                    "handle": handle,
                    "awaiting_breakout": True,
                },
            )

        # Generate signal
        signal = self._generate_signal(df, i, cup, handle, arrays)

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                "left_rim_idx": cup["left_rim"][0],
                "left_rim": cup["left_rim"][1],
                "right_rim_idx": cup["right_rim"][0],
                "right_rim": cup["right_rim"][1],
                "cup_bottom_idx": cup["cup_bottom"][0],
                "cup_bottom": cup["cup_bottom"][1],
                "cup_depth": cup["cup_depth"],
                "handle_low": handle["handle_low"],
                "handle_retracement": handle["handle_retracement"],
            },
            bars_since_detection=0,
            start_index=cup["left_rim"][0],
            end_index=i,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal."""
        result = self.detect(df, i)
        return result.signal if result.detected else None

    def _generate_signal(
        self, df: pd.DataFrame, i: int, cup: Dict, handle: Dict, arrays: dict
    ) -> Optional[TradeSignal]:
        """Generate trade signal for Cup and Handle breakout."""
        volume_arr = arrays["volume"]

        right_rim_high = cup["right_rim"][1]
        cup_depth = cup["cup_depth"]
        handle_low = handle["handle_low"]

        # Entry above right rim
        entry_price = right_rim_high + self.entry_offset

        # Stop loss below handle
        stop_loss = handle_low - self.stop_offset

        # Targets based on cup depth
        take_profit_1 = entry_price + (cup_depth * 0.62)
        take_profit_2 = entry_price + cup_depth

        # Volume confirmation
        volume_confirmed = True
        if self.volume_filter:
            vol_sma = volume_sma(df["Volume"], 20)
            if i < len(vol_sma):
                current_vol = float(volume_arr[i]) if len(volume_arr) > i else 0.0
                avg_vol = self._safe_float(vol_sma.iloc[i])
                volume_confirmed = current_vol > avg_vol

        # Check volume during handle (should be lower than cup volume)
        cup_start = cup["left_rim"][0]
        cup_end = cup["right_rim"][0]
        handle_start = handle["handle_start"]

        if handle_start > cup_end and cup_end > cup_start and len(volume_arr) > i:
            cup_volume = float(np.mean(volume_arr[cup_start:cup_end]))
            handle_volume = float(np.mean(volume_arr[handle_start : i + 1]))
            volume_decreasing = handle_volume < cup_volume
        else:
            volume_decreasing = True

        # Confidence
        confidence = 0.6
        if volume_confirmed:
            confidence += 0.1
        if volume_decreasing:
            confidence += 0.05
        if cup["rounded_bottom"]:
            confidence += 0.1

        return TradeSignal(
            pattern_name=self.name,
            direction=SignalDirection.LONG,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=None,
            confidence=min(confidence, 1.0),
            timestamp=df.index[i] if hasattr(df, "index") else None,
            metadata={
                "cup_depth": cup_depth,
                "handle_retracement": handle["handle_retracement"],
                "handle_duration": handle["handle_duration"],
                "rounded_bottom": cup["rounded_bottom"],
                "volume_confirmed": volume_confirmed,
                "volume_decreasing": volume_decreasing,
                "entry_type": "buy_stop",
            },
        )
