"""
Dead Cat Bounce Pattern

Detection Logic:
- Event Day = Bar with 15%+ decline (close[i] < close[i-1] * 0.85)
- Event Day High = High of the event day
- Event Day Low = Low of the event day
- Bounce = Price retraces 50-62% of the event day decline
- Pattern confirmed when price resumes decline from the bounce level

Entry Rules:
- Short Entry at 50-62% Fibonacci retracement of event day range
- Entry triggered when price bounces to Fib level and shows reversal
- Entry valid for next 3 to 5 bars after confirmation

Stop Loss Rules:
- Stop = Event Day High + 0.01
- Stop placed above the event day's extreme

Take Profit Rules:
- Target = 100% of the event day's range (gap fill)
- Target 2 = 127% extension
- Target 3 = 162% extension
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ...indicators.fibonacci import fibonacci_retracement
from ...indicators.technical import volume_sma
from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class DeadCatBounce(BasePattern):
    """
    Dead Cat Bounce Pattern Detector

    A bearish continuation pattern that occurs after a significant
    downward move (15%+), followed by a partial recovery that fails.
    """

    def __init__(
        self,
        event_decline_pct: float = 0.15,
        min_bounce_pct: float = 0.38,
        max_bounce_pct: float = 0.62,
        max_bars_after_event: int = 20,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        volume_filter: bool = False,
    ):
        """
        Initialize Dead Cat Bounce pattern detector.

        Args:
            event_decline_pct: Minimum decline percentage for event day (default 15%)
            min_bounce_pct: Minimum bounce retracement (default 38%)
            max_bounce_pct: Maximum bounce retracement (default 62%)
            max_bars_after_event: Maximum bars to look for bounce after event
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            volume_filter: Require volume confirmation
        """
        super().__init__(
            name="Dead Cat Bounce",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=max_bars_after_event + 5,
        )
        self.event_decline_pct = event_decline_pct
        self.min_bounce_pct = min_bounce_pct
        self.max_bounce_pct = max_bounce_pct
        self.max_bars_after_event = max_bars_after_event
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.volume_filter = volume_filter

    def _find_event_day(self, arrays: dict, i: int) -> Optional[Dict]:
        """
        Find a significant decline event day.

        Args:
            arrays: Dictionary with NumPy arrays (open, high, low, close, volume)
            i: Current bar index

        Returns:
            Dictionary with event day details or None
        """
        close_arr = arrays["close"]
        high_arr = arrays["high"]
        low_arr = arrays["low"]
        open_arr = arrays["open"]
        volume_arr = arrays["volume"]

        # Look back for event day within max_bars_after_event
        lookback_start = max(1, i - self.max_bars_after_event)

        for event_idx in range(lookback_start, i):
            prev_close = float(close_arr[event_idx - 1])
            event_close = float(close_arr[event_idx])
            event_high = float(high_arr[event_idx])
            event_low = float(low_arr[event_idx])
            event_open = float(open_arr[event_idx])
            event_volume = float(volume_arr[event_idx]) if len(volume_arr) > 0 else 0.0

            # Calculate decline percentage
            if prev_close == 0:
                continue

            decline_pct = (prev_close - event_close) / prev_close

            # Check for significant decline
            if decline_pct >= self.event_decline_pct:
                return {
                    "event_idx": event_idx,
                    "event_open": event_open,
                    "event_high": event_high,
                    "event_low": event_low,
                    "event_close": event_close,
                    "prev_close": prev_close,
                    "decline_pct": decline_pct,
                    "event_range": event_high - event_low,
                    "event_volume": event_volume,
                }

        return None

    def _check_bounce(self, arrays: dict, i: int, event: Dict) -> Optional[Dict]:
        """
        Check for a valid bounce after the event day.

        Args:
            arrays: Dictionary with NumPy arrays (open, high, low, close, volume)
            i: Current bar index
            event: Event day details

        Returns:
            Dictionary with bounce details or None
        """
        high_arr = arrays["high"]
        low_arr = arrays["low"]
        close_arr = arrays["close"]

        event_idx = event["event_idx"]
        event_low = event["event_low"]
        event_high = event["event_high"]
        prev_close = event["prev_close"]

        # Calculate Fibonacci levels from the decline
        decline_start = prev_close
        decline_end = event_low

        fib_levels = fibonacci_retracement(decline_start, decline_end)

        # Find the highest point after event day (the bounce)
        bounce_high = event_low
        bounce_high_idx = event_idx

        for j in range(event_idx + 1, i + 1):
            high = float(high_arr[j])
            if high > bounce_high:
                bounce_high = high
                bounce_high_idx = j

        # Calculate bounce retracement
        decline_range = decline_start - decline_end
        if decline_range == 0:
            return None

        bounce_range = bounce_high - decline_end
        retracement_pct = bounce_range / decline_range

        # Check if bounce is within valid range
        if retracement_pct < self.min_bounce_pct or retracement_pct > self.max_bounce_pct:
            return None

        # Check for reversal from bounce level
        current_close = float(close_arr[i])
        current_high = float(high_arr[i])

        # Check if price has started to decline from bounce
        if current_close >= bounce_high:
            return None

        # Check for bearish reversal signal
        prev_close_bar = float(close_arr[i - 1])

        # Price should be declining from the bounce
        if current_close >= prev_close_bar:
            return None

        return {
            "bounce_high": bounce_high,
            "bounce_high_idx": bounce_high_idx,
            "retracement_pct": retracement_pct,
            "fib_382": fib_levels[0.382],
            "fib_500": fib_levels[0.5],
            "fib_618": fib_levels[0.618],
            "decline_range": decline_range,
        }

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized detection of Dead Cat Bounce patterns across the entire DataFrame.

        Returns:
            np.ndarray of np.int8: 0=no signal, -1=SHORT
        """
        n = len(df)
        result = np.zeros(n, dtype=np.int8)
        if n < self.max_bars_after_event + 5:
            return result

        close_a = df["Close"].to_numpy()
        high_a = df["High"].to_numpy()
        low_a = df["Low"].to_numpy()

        decl_pct = self.event_decline_pct
        min_bounce = self.min_bounce_pct
        max_bounce = self.max_bounce_pct
        max_after = self.max_bars_after_event

        for i in range(max_after + 1, n):
            lookback_start = max(1, i - max_after)
            event = None

            for event_idx in range(lookback_start, i):
                prev_close = close_a[event_idx - 1]
                event_close = close_a[event_idx]
                if prev_close == 0:
                    continue
                decline = (prev_close - event_close) / prev_close
                if decline >= decl_pct:
                    event = {
                        "idx": event_idx,
                        "high": high_a[event_idx],
                        "low": low_a[event_idx],
                        "close": event_close,
                        "prev_close": prev_close,
                        "decline_pct": decline,
                        "range": high_a[event_idx] - low_a[event_idx],
                    }
                    break

            if event is None:
                continue

            event_idx = event["idx"]
            event_low = event["low"]
            decl_start = event["prev_close"]
            decl_end = event_low
            decl_range = decl_start - decl_end
            if decl_range == 0:
                continue

            # Find highest point after event (the bounce)
            bounce_high = float(np.max(high_a[event_idx + 1 : i + 1]))
            bounce_high_idx = int(np.argmax(high_a[event_idx + 1 : i + 1])) + event_idx + 1
            retrace = (bounce_high - decl_end) / decl_range

            if retrace < min_bounce or retrace > max_bounce:
                continue

            # Check for reversal from bounce
            cur_close = close_a[i]
            if cur_close >= bounce_high:
                continue
            if i > 0 and cur_close >= close_a[i - 1]:
                continue

            result[i] = -1

        return result

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Dead Cat Bounce pattern at bar index i.

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

        # Find event day
        event = self._find_event_day(arrays, i)

        if event is None:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Check for valid bounce
        bounce = self._check_bounce(arrays, i, event)

        if bounce is None:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Generate signal
        signal = self._generate_signal(df, i, event, bounce, arrays)

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                "event_idx": event["event_idx"],
                "event_high": event["event_high"],
                "event_low": event["event_low"],
                "event_close": event["event_close"],
                "decline_pct": event["decline_pct"],
                "bounce_high": bounce["bounce_high"],
                "retracement_pct": bounce["retracement_pct"],
            },
            bars_since_detection=0,
            start_index=event["event_idx"],
            end_index=i,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal."""
        result = self.detect(df, i)
        return result.signal if result.detected else None

    def _generate_signal(
        self, df: pd.DataFrame, i: int, event: Dict, bounce: Dict, arrays: dict
    ) -> Optional[TradeSignal]:
        """Generate Dead Cat Bounce short signal."""
        low_arr = arrays["low"]
        close_arr = arrays["close"]

        current_low = float(low_arr[i])
        current_close = float(close_arr[i])
        event_high = event["event_high"]
        event_low = event["event_low"]

        # Entry below current bar's low
        entry_price = current_low - self.entry_offset

        # Stop above event day high
        stop_loss = event_high + self.stop_offset

        # Target 1: 100% of event day range (full gap fill)
        event_range = event["event_range"]
        take_profit_1 = entry_price - event_range

        # Target 2: 127% extension
        take_profit_2 = entry_price - (event_range * 1.27)

        # Target 3: 162% extension
        take_profit_3 = entry_price - (event_range * 1.62)

        # Calculate confidence
        confidence = 0.55

        # Boost confidence based on event magnitude
        if event["decline_pct"] > 0.20:
            confidence += 0.1
        elif event["decline_pct"] > 0.15:
            confidence += 0.05

        # Boost confidence for bounce near 50% retracement (optimal)
        retracement = bounce["retracement_pct"]
        if 0.45 <= retracement <= 0.55:
            confidence += 0.1
        elif 0.38 <= retracement <= 0.62:
            confidence += 0.05

        # Check volume on event day
        if self.volume_filter and i > event["event_idx"]:
            avg_volume = self._safe_float(volume_sma(df["Volume"], 20).iloc[i])
            if avg_volume > 0 and event["event_volume"] > avg_volume * 1.5:
                confidence += 0.05

        # Check for bearish momentum
        if current_close < event_low:
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
                "event_idx": event["event_idx"],
                "event_high": event_high,
                "event_low": event_low,
                "decline_pct": event["decline_pct"],
                "bounce_high": bounce["bounce_high"],
                "retracement_pct": bounce["retracement_pct"],
                "fib_382": bounce["fib_382"],
                "fib_500": bounce["fib_500"],
                "fib_618": bounce["fib_618"],
                "entry_type": "sell_stop",
            },
        )
