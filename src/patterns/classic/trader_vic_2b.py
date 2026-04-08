"""
Trader Vic's 2B Pattern

Detection Logic:
For Bearish 2B:
- NewHigh = high[i] > Max(high[i-21:i-1]) (21-bar new high)
- Pullback = low[i+1:i+5] < high[i] (retracement after new high)
- FailedTest = high[i+5:i+10] > high[i] but close[i+5:i+10] < high[i]
- Breakdown = close[j] < low[NewHigh_bar] (close below breakout bar low)

For Bullish 2B:
- NewLow = low[i] < Min(low[i-21:i-1]) (21-bar new low)
- Pullback = high[i+1:i+5] > low[i] (retracement after new low)
- FailedTest = low[i+5:i+10] < low[i] but close[i+5:i+10] > low[i]
- Breakout = close[j] > high[NewLow_bar] (close above breakout bar high)

Entry Rules:
- Bearish 2B: Short Entry = low[FailedTest_bar] - 0.01
- Bullish 2B: Long Entry = high[FailedTest_bar] + 0.01
- Entry triggered on close below/above the breakout bar's low/high
- Entry valid for next 3 to 5 bars after confirmation

Stop Loss Rules:
- Bearish 2B Stop = high[FailedTest_bar] + 0.01
- Bullish 2B Stop = low[FailedTest_bar] - 0.01
- Stop placed above recent high (short) or below recent low (long)

Take Profit Rules:
- Bearish 2B Target = Swing Low prior to NewHigh bar
- Bullish 2B Target = Swing High prior to NewLow bar
- Target = 100% retracement of the rally/decline that made the new high/low
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ...indicators.pivots import get_recent_swing_high, get_recent_swing_low
from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class TraderVic2B(BasePattern):
    """
    Trader Vic's 2B Pattern Detector

    Identifies failed breakouts where price makes a new high/low
    but fails to sustain it, indicating a potential reversal.
    """

    def __init__(
        self,
        lookback: int = 21,
        pullback_bars: int = 5,
        test_bars: int = 10,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        volume_filter: bool = False,
    ):
        """
        Initialize 2B pattern detector.

        Args:
            lookback: Lookback period for new high/low (default 21)
            pullback_bars: Bars for pullback after new high/low
            test_bars: Bars for failed test window
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            volume_filter: Require volume confirmation
        """
        super().__init__(
            name="Trader Vic's 2B",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=lookback + test_bars + pullback_bars,
        )
        self.lookback = lookback
        self.pullback_bars = pullback_bars
        self.test_bars = test_bars
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.volume_filter = volume_filter

    def _find_bearish_2b(self, arrays: dict, i: int) -> Optional[Dict]:
        """
        Find bearish 2B pattern (failed new high).

        Args:
            arrays: Dictionary with NumPy arrays (open, high, low, close, volume)
            i: Current bar index

        Returns:
            Dictionary with pattern details or None
        """
        if i < self.lookback + self.test_bars:
            return None

        high_arr = arrays["high"]
        low_arr = arrays["low"]
        close_arr = arrays["close"]

        # Look for new high in the past
        for new_high_idx in range(i - self.test_bars, i - self.pullback_bars, -1):
            if new_high_idx < self.lookback:
                continue

            # Check for 21-bar new high
            new_high = float(high_arr[new_high_idx])
            prev_highs = high_arr[new_high_idx - self.lookback : new_high_idx]

            if len(prev_highs) == 0 or new_high <= float(np.max(prev_highs)):
                continue

            # Check for pullback after new high
            pullback_low = float("inf")
            pullback_low_idx = None
            for j in range(new_high_idx + 1, min(new_high_idx + self.pullback_bars + 1, i + 1)):
                low = float(low_arr[j])
                if low < pullback_low:
                    pullback_low = low
                    pullback_low_idx = j

            if pullback_low >= new_high:
                continue

            # Check for failed test (price approaches new high but closes below)
            for j in range(
                new_high_idx + self.pullback_bars, min(new_high_idx + self.test_bars + 1, i + 1)
            ):
                high = float(high_arr[j])
                close = float(close_arr[j])

                # Failed test: makes new high or near it, but closes below
                if high >= new_high and close < new_high:
                    # Check for breakdown
                    current_close = float(close_arr[i])
                    new_high_bar_low = float(low_arr[new_high_idx])

                    if current_close < new_high_bar_low:
                        return {
                            "new_high_idx": new_high_idx,
                            "new_high": new_high,
                            "failed_test_idx": j,
                            "failed_test_high": high,
                            "pullback_low": pullback_low,
                            "breakdown": True,
                            "direction": "bearish",
                        }

        return None

    def _find_bullish_2b(self, arrays: dict, i: int) -> Optional[Dict]:
        """
        Find bullish 2B pattern (failed new low).

        Args:
            arrays: Dictionary with NumPy arrays (open, high, low, close, volume)
            i: Current bar index

        Returns:
            Dictionary with pattern details or None
        """
        if i < self.lookback + self.test_bars:
            return None

        high_arr = arrays["high"]
        low_arr = arrays["low"]
        close_arr = arrays["close"]

        # Look for new low in the past
        for new_low_idx in range(i - self.test_bars, i - self.pullback_bars, -1):
            if new_low_idx < self.lookback:
                continue

            # Check for 21-bar new low
            new_low = float(low_arr[new_low_idx])
            prev_lows = low_arr[new_low_idx - self.lookback : new_low_idx]

            if len(prev_lows) == 0 or new_low >= float(np.min(prev_lows)):
                continue

            # Check for pullback after new low
            pullback_high = float("-inf")
            pullback_high_idx = None
            for j in range(new_low_idx + 1, min(new_low_idx + self.pullback_bars + 1, i + 1)):
                high = float(high_arr[j])
                if high > pullback_high:
                    pullback_high = high
                    pullback_high_idx = j

            if pullback_high <= new_low:
                continue

            # Check for failed test (price approaches new low but closes above)
            for j in range(
                new_low_idx + self.pullback_bars, min(new_low_idx + self.test_bars + 1, i + 1)
            ):
                low = float(low_arr[j])
                close = float(close_arr[j])

                # Failed test: makes new low or near it, but closes above
                if low <= new_low and close > new_low:
                    # Check for breakout
                    current_close = float(close_arr[i])
                    new_low_bar_high = float(high_arr[new_low_idx])

                    if current_close > new_low_bar_high:
                        return {
                            "new_low_idx": new_low_idx,
                            "new_low": new_low,
                            "failed_test_idx": j,
                            "failed_test_low": low,
                            "pullback_high": pullback_high,
                            "breakout": True,
                            "direction": "bullish",
                        }

        return None

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect 2B pattern at bar index i.

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

        # Try bearish 2B first
        bearish = self._find_bearish_2b(arrays, i)

        # Try bullish 2B
        bullish = self._find_bullish_2b(arrays, i)

        if bearish is None and bullish is None:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Generate signal
        if bearish is not None:
            signal = self._generate_bearish_signal(df, i, bearish, arrays)
            direction = "bearish"
            pattern = bearish
        elif bullish is not None:
            signal = self._generate_bullish_signal(df, i, bullish, arrays)
            direction = "bullish"
            pattern = bullish
        else:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
            )

        return PatternResult(
            detected=True,
            pattern_name=f"{self.name} ({direction.title()})",
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={k: float(v) for k, v in pattern.items() if isinstance(v, (int, float))},
            bars_since_detection=0,
            start_index=pattern.get("new_high_idx") or pattern.get("new_low_idx"),
            end_index=i,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal."""
        result = self.detect(df, i)
        return result.signal if result.detected else None

    def _generate_bearish_signal(
        self, df: pd.DataFrame, i: int, pattern: Dict, arrays: dict
    ) -> Optional[TradeSignal]:
        """Generate bearish 2B signal."""
        low_arr = arrays["low"]

        current_low = float(low_arr[i])
        failed_test_high = pattern["failed_test_high"]

        # Entry below failed test low
        entry_price = current_low - self.entry_offset

        # Stop above failed test high
        stop_loss = failed_test_high + self.stop_offset

        # Target prior swing low
        swing_low = get_recent_swing_low(df, pattern["new_high_idx"], lookback=50)
        if swing_low:
            take_profit_1 = swing_low[1]
        else:
            risk = stop_loss - entry_price
            take_profit_1 = entry_price - (risk * 2)

        take_profit_2 = entry_price - (
            (pattern["new_high"] - pattern.get("pullback_low", 0)) * 0.62
        )

        # Volume check
        volume_arr = arrays["volume"]
        volume_confirmed = True
        if self.volume_filter and len(volume_arr) > 0:
            new_high_vol = float(volume_arr[pattern["new_high_idx"]])
            failed_test_vol = float(volume_arr[pattern["failed_test_idx"]])
            volume_confirmed = failed_test_vol < new_high_vol

        confidence = 0.55
        if volume_confirmed:
            confidence += 0.1

        return TradeSignal(
            pattern_name=f"{self.name} (Bearish)",
            direction=SignalDirection.SHORT,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=None,
            confidence=min(confidence, 1.0),
            timestamp=df.index[i] if hasattr(df, "index") else None,
            metadata={
                "new_high": pattern["new_high"],
                "failed_test_idx": pattern["failed_test_idx"],
                "volume_confirmed": volume_confirmed,
                "entry_type": "sell_stop",
            },
        )

    def _generate_bullish_signal(
        self, df: pd.DataFrame, i: int, pattern: Dict, arrays: dict
    ) -> Optional[TradeSignal]:
        """Generate bullish 2B signal."""
        high_arr = arrays["high"]

        current_high = float(high_arr[i])
        failed_test_low = pattern["failed_test_low"]

        # Entry above failed test high
        entry_price = current_high + self.entry_offset

        # Stop below failed test low
        stop_loss = failed_test_low - self.stop_offset

        # Target prior swing high
        swing_high = get_recent_swing_high(df, pattern["new_low_idx"], lookback=50)
        if swing_high:
            take_profit_1 = swing_high[1]
        else:
            risk = entry_price - stop_loss
            take_profit_1 = entry_price + (risk * 2)

        take_profit_2 = entry_price + (
            (pattern.get("pullback_high", 0) - pattern["new_low"]) * 0.62
        )

        # Volume check
        volume_arr = arrays["volume"]
        volume_confirmed = True
        if self.volume_filter and len(volume_arr) > 0:
            new_low_vol = float(volume_arr[pattern["new_low_idx"]])
            failed_test_vol = float(volume_arr[pattern["failed_test_idx"]])
            volume_confirmed = failed_test_vol < new_low_vol

        confidence = 0.55
        if volume_confirmed:
            confidence += 0.1

        return TradeSignal(
            pattern_name=f"{self.name} (Bullish)",
            direction=SignalDirection.LONG,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=None,
            confidence=min(confidence, 1.0),
            timestamp=df.index[i] if hasattr(df, "index") else None,
            metadata={
                "new_low": pattern["new_low"],
                "failed_test_idx": pattern["failed_test_idx"],
                "volume_confirmed": volume_confirmed,
                "entry_type": "buy_stop",
            },
        )
