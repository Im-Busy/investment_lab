"""
Two-Bar Reversal Pattern (Pipe Bottom / Pipe Top)

Detection Logic:
- Pipe Bottom (bullish reversal):
  - Occurs after extended downtrend (N bars declining)
  - Bar 1: Long bearish candle, closes at or near low
  - Bar 2: Long candle (any color), closes in upper 50% of its range
  - Both bars have larger range than preceding 3-5 bars

- Pipe Top (bearish reversal):
  - Occurs after extended uptrend
  - Bar 1: Long bullish candle, closes at or near high
  - Bar 2: Long candle (any color), closes in lower 50% of its range
  - Both bars have larger range than preceding 3-5 bars

Entry Rules:
- Long Entry: Buy Stop = high[bar2] + filter (breakout above bar 2 high)
- Short Entry: Sell Stop = low[bar2] - filter (breakdown below bar 2 low)
- Entry triggered on confirmation breakout

Stop Loss Rules:
- Long Stop: Below bar 2 low - filter
- Short Stop: Above bar 2 high + filter

Take Profit Rules:
- Target: bar2.high + max(bar1.range, bar2.range) for long
- Target: bar2.low - max(bar1.range, bar2.range) for short
- Note: More reliable on weekly data -> adjust confidence by timeframe
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class TwoBarReversal(BasePattern):
    """
    Two-Bar Reversal Pattern Detector (Pipe Bottom / Pipe Top)

    A short-term reversal pattern consisting of two bars with specific
    characteristics following a trend.
    """

    def __init__(
        self,
        trend_bars: int = 5,
        min_range_multiplier: float = 1.2,
        close_threshold: float = 0.3,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        confirmation_filter: float = 0.005,
    ):
        """
        Initialize Two-Bar Reversal pattern detector.

        Args:
            trend_bars: Number of bars to check for prior trend
            min_range_multiplier: Minimum range multiplier vs average (default 1.2)
            close_threshold: Threshold for close position (0.3 = 30% from extreme)
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            confirmation_filter: Minimum breakout percentage for confirmation
        """
        super().__init__(
            name="Two-Bar Reversal",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=trend_bars + 2,
        )
        self.trend_bars = trend_bars
        self.min_range_multiplier = min_range_multiplier
        self.close_threshold = close_threshold
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.confirmation_filter = confirmation_filter

    def _calculate_range(self, arrays: dict, i: int) -> float:
        """Calculate bar range."""
        high = float(arrays["high"][i])
        low = float(arrays["low"][i])
        return high - low

    def _calculate_body(self, arrays: dict, i: int) -> tuple:
        """Calculate bar body (open, close, body, direction)."""
        open_price = float(arrays["open"][i])
        close = float(arrays["close"][i])
        high = float(arrays["high"][i])
        low = float(arrays["low"][i])

        body = abs(close - open_price)
        is_bullish = close > open_price

        return open_price, close, body, is_bullish

    def _is_downtrend(self, arrays: dict, end_bar: int, n_bars: int) -> bool:
        """Check if price is in downtrend over n_bars."""
        if end_bar < n_bars:
            return False

        close_arr = arrays["close"]
        closes = [float(close_arr[i]) for i in range(end_bar - n_bars, end_bar)]

        # Check if majority of closes are declining
        declining = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i - 1])

        # Also check overall direction
        overall_decline = closes[0] > closes[-1]

        return declining >= n_bars * 0.6 and overall_decline

    def _is_uptrend(self, arrays: dict, end_bar: int, n_bars: int) -> bool:
        """Check if price is in uptrend over n_bars."""
        if end_bar < n_bars:
            return False

        close_arr = arrays["close"]
        closes = [float(close_arr[i]) for i in range(end_bar - n_bars, end_bar)]

        # Check if majority of closes are rising
        rising = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])

        # Also check overall direction
        overall_rise = closes[0] < closes[-1]

        return rising >= n_bars * 0.6 and overall_rise

    def _find_pattern(self, arrays: dict, i: int) -> Optional[Dict]:
        """
        Find Two-Bar Reversal pattern.

        Args:
            arrays: Dictionary with NumPy arrays
            i: Current bar index

        Returns:
            Dictionary with pattern details or None
        """
        if i < self.trend_bars + 2:
            return None

        # Get bar 1 and bar 2 (bar 1 is more recent, bar 2 is current)
        bar1_idx = i - 1
        bar2_idx = i

        # Calculate ranges
        bar1_range = self._calculate_range(arrays, bar1_idx)
        bar2_range = self._calculate_range(arrays, bar2_idx)

        # Calculate average range of preceding bars
        preceding_ranges = [self._calculate_range(arrays, j) for j in range(bar1_idx - 3, bar1_idx)]
        avg_range = np.mean(preceding_ranges) if preceding_ranges else bar1_range

        # Both bars should have larger range than average
        if bar1_range < avg_range * self.min_range_multiplier:
            return None
        if bar2_range < avg_range * self.min_range_multiplier:
            return None

        # Get bar details
        bar1_open, bar1_close, bar1_body, bar1_bullish = self._calculate_body(arrays, bar1_idx)
        bar2_open, bar2_close, bar2_body, bar2_bullish = self._calculate_body(arrays, bar2_idx)

        bar1_high = float(arrays["high"][bar1_idx])
        bar1_low = float(arrays["low"][bar1_idx])
        bar2_high = float(arrays["high"][bar2_idx])
        bar2_low = float(arrays["low"][bar2_idx])

        max_range = max(bar1_range, bar2_range)

        # Check for Pipe Bottom (bullish reversal)
        if self._is_downtrend(arrays, bar1_idx, self.trend_bars):
            # Bar 1: Long bearish candle, closes at or near low
            bar1_close_position = (bar1_close - bar1_low) / bar1_range if bar1_range > 0 else 0

            # Bar 2: Long candle, closes in upper 50% of range
            bar2_close_position = (bar2_close - bar2_low) / bar2_range if bar2_range > 0 else 0

            if (
                not bar1_bullish
                and bar1_close_position <= self.close_threshold
                and bar2_close_position >= 0.5
            ):
                # Check for breakout above bar 2 high
                current_close = float(arrays["close"][i])
                if current_close > bar2_high * (1 + self.confirmation_filter):
                    return {
                        "direction": "bullish",
                        "bar1_idx": bar1_idx,
                        "bar2_idx": bar2_idx,
                        "bar1_range": bar1_range,
                        "bar2_range": bar2_range,
                        "bar1_high": bar1_high,
                        "bar1_low": bar1_low,
                        "bar2_high": bar2_high,
                        "bar2_low": bar2_low,
                        "max_range": max_range,
                        "breakout_bar": i,
                    }

        # Check for Pipe Top (bearish reversal)
        if self._is_uptrend(arrays, bar1_idx, self.trend_bars):
            # Bar 1: Long bullish candle, closes at or near high
            bar1_close_position = (bar1_high - bar1_close) / bar1_range if bar1_range > 0 else 0

            # Bar 2: Long candle, closes in lower 50% of range
            bar2_close_position = (bar2_high - bar2_close) / bar2_range if bar2_range > 0 else 0

            if (
                bar1_bullish
                and bar1_close_position <= self.close_threshold
                and bar2_close_position >= 0.5
            ):
                # Check for breakdown below bar 2 low
                current_close = float(arrays["close"][i])
                if current_close < bar2_low * (1 - self.confirmation_filter):
                    return {
                        "direction": "bearish",
                        "bar1_idx": bar1_idx,
                        "bar2_idx": bar2_idx,
                        "bar1_range": bar1_range,
                        "bar2_range": bar2_range,
                        "bar1_high": bar1_high,
                        "bar1_low": bar1_low,
                        "bar2_high": bar2_high,
                        "bar2_low": bar2_low,
                        "max_range": max_range,
                        "breakout_bar": i,
                    }

        return None

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized detection of Two-Bar Reversal signals across the entire DataFrame.

        Args:
            df: DataFrame with 'Open', 'High', 'Low', 'Close' columns

        Returns:
            np.ndarray of np.int8: 0=no signal, 1=LONG, -1=SHORT
        """
        n = len(df)
        result = np.zeros(n, dtype=np.int8)
        if n < self.trend_bars + 3:
            return result

        arrays = self._extract_arrays(df)
        high_a = arrays["high"]
        low_a = arrays["low"]
        open_a = arrays["open"]
        close_a = arrays["close"]

        range_a = high_a - low_a
        is_bullish_a = close_a > open_a

        for i in range(self.trend_bars + 2, n):
            bar1_idx = i - 1
            bar2_idx = i

            bar1_range = range_a[bar1_idx]
            bar2_range = range_a[bar2_idx]

            if bar1_range == 0 or bar2_range == 0:
                continue

            preceding_ranges = range_a[bar1_idx - 3 : bar1_idx]
            avg_range = np.mean(preceding_ranges) if len(preceding_ranges) > 0 else bar1_range

            if bar1_range < avg_range * self.min_range_multiplier:
                continue
            if bar2_range < avg_range * self.min_range_multiplier:
                continue

            bar1_close = close_a[bar1_idx]
            bar1_low = low_a[bar1_idx]
            bar1_high = high_a[bar1_idx]
            bar2_close = close_a[bar2_idx]
            bar2_low = low_a[bar2_idx]
            bar2_high = high_a[bar2_idx]

            closes_dt = close_a[bar1_idx - self.trend_bars : bar1_idx]
            if len(closes_dt) < 2:
                continue
            declining = np.sum(np.diff(closes_dt) < 0)
            is_dt = declining >= self.trend_bars * 0.6 and closes_dt[0] > closes_dt[-1]

            is_ut = False
            if not is_dt:
                rising = np.sum(np.diff(closes_dt) > 0)
                is_ut = rising >= self.trend_bars * 0.6 and closes_dt[0] < closes_dt[-1]

            if is_dt:
                bar1_close_pos = (bar1_close - bar1_low) / bar1_range
                bar2_close_pos = (bar2_close - bar2_low) / bar2_range

                if (
                    not is_bullish_a[bar1_idx]
                    and bar1_close_pos <= self.close_threshold
                    and bar2_close_pos >= 0.5
                ):
                    if close_a[i] > bar2_high * (1.0 + self.confirmation_filter):
                        result[i] = 1
            elif is_ut:
                bar1_close_pos = (bar1_high - bar1_close) / bar1_range
                bar2_close_pos = (bar2_high - bar2_close) / bar2_range

                if (
                    is_bullish_a[bar1_idx]
                    and bar1_close_pos <= self.close_threshold
                    and bar2_close_pos >= 0.5
                ):
                    if close_a[i] < bar2_low * (1.0 - self.confirmation_filter):
                        result[i] = -1

        return result

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Two-Bar Reversal pattern at bar index i.

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
        pattern = self._find_pattern(arrays, i)

        if pattern is None:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        signal = self._generate_signal(df, i, pattern, arrays)

        pattern_name = "Pipe Bottom" if pattern["direction"] == "bullish" else "Pipe Top"

        return PatternResult(
            detected=True,
            pattern_name=pattern_name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                "direction": pattern["direction"],
                "bar1_idx": pattern["bar1_idx"],
                "bar2_idx": pattern["bar2_idx"],
                "bar1_range": pattern["bar1_range"],
                "bar2_range": pattern["bar2_range"],
                "bar2_high": pattern["bar2_high"],
                "bar2_low": pattern["bar2_low"],
            },
            bars_since_detection=0,
            start_index=pattern["bar1_idx"],
            end_index=i,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal."""
        result = self.detect(df, i)
        return result.signal if result.detected else None

    def _generate_signal(
        self, df: pd.DataFrame, i: int, pattern: Dict, arrays: dict
    ) -> Optional[TradeSignal]:
        """Generate Two-Bar Reversal signal."""
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        current_high = float(high_arr[i])
        current_low = float(low_arr[i])
        direction = pattern["direction"]
        max_range = pattern["max_range"]
        bar2_high = pattern["bar2_high"]
        bar2_low = pattern["bar2_low"]

        if direction == "bullish":
            # Pipe Bottom - LONG signal
            entry_price = current_high + self.entry_offset
            stop_loss = bar2_low - self.stop_offset
            take_profit_1 = entry_price + max_range
            take_profit_2 = entry_price + (max_range * 1.5)
            take_profit_3 = entry_price + (max_range * 2.0)

            confidence = 0.55

            return TradeSignal(
                pattern_name="Pipe Bottom",
                direction=SignalDirection.LONG,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                take_profit_3=take_profit_3,
                confidence=confidence,
                timestamp=df.index[i] if hasattr(df, "index") else None,
                metadata={
                    "direction": "bullish",
                    "bar1_range": pattern["bar1_range"],
                    "bar2_range": pattern["bar2_range"],
                    "max_range": max_range,
                    "entry_type": "buy_stop",
                },
            )
        else:
            # Pipe Top - SHORT signal
            entry_price = current_low - self.entry_offset
            stop_loss = bar2_high + self.stop_offset
            take_profit_1 = entry_price - max_range
            take_profit_2 = entry_price - (max_range * 1.5)
            take_profit_3 = entry_price - (max_range * 2.0)

            confidence = 0.55

            return TradeSignal(
                pattern_name="Pipe Top",
                direction=SignalDirection.SHORT,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                take_profit_3=take_profit_3,
                confidence=confidence,
                timestamp=df.index[i] if hasattr(df, "index") else None,
                metadata={
                    "direction": "bearish",
                    "bar1_range": pattern["bar1_range"],
                    "bar2_range": pattern["bar2_range"],
                    "max_range": max_range,
                    "entry_type": "sell_stop",
                },
            )
