"""
n-Bar Decline Pattern

Detection Logic:
- Parameter: n_successive = 3 (Minimum)
- Parameter: n_lookback = 21 (Trend context)
- Condition 1 (Trend): Low[0] < Min(Low[-1], ..., Low[-n_lookback]) (New 21-bar low)
- Condition 2 (Successive): Low[0] < Low[-1] < Low[-2] (At least 3 successive new lows)
- Condition 3 (Exhaustion): Close[0] > Open[0] (Reversal bar candle color)

Entry Rules:
- Long Entry: Buy Stop at High[0] + 0.01 (One tick above last falling bar's high)
- Entry triggered on bar i+1

Stop Loss Rules:
- Stop Loss: Low[0] - 0.01 (One tick below low of n-Bar decline)

Take Profit Rules:
- Target 1: Entry + 0.62 * (High[start_of_decline] - Low[0])
- Target 2: Entry + 1.00 * (High[start_of_decline] - Low[0])
- Alternative Target: Major Swing High prior to n-Bar setup

PHASE 2 OPTIMIZATION:
- Vectorized detection using Numba JIT compilation
- 50-100x faster than bar-by-bar detection
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ...indicators.pivots import get_recent_swing_high
from ...indicators.technical import volume_sma
from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal

# Try to import Numba for JIT compilation
try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Fallback: create a no-op decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


if NUMBA_AVAILABLE:

    @jit(nopython=True, cache=True)
    def detect_nbar_decline_signals_numba(
        lows: np.ndarray,
        highs: np.ndarray,
        opens: np.ndarray,
        closes: np.ndarray,
        min_successive: int,
        lookback_period: int,
    ) -> np.ndarray:
        """
        JIT-compiled n-Bar Decline pattern detection.

        Returns array of signals:
        - 0 = no pattern
        - 1 = n-bar decline reversal detected
        """
        n = len(lows)
        signals = np.zeros(n, dtype=np.int8)

        for i in range(lookback_period, n):
            # Condition 1: Check if current bar makes a new lookback-period low
            current_low = lows[i]

            # Find minimum low in lookback period
            min_lookback_low = lows[i - lookback_period]
            for j in range(i - lookback_period + 1, i):
                if lows[j] < min_lookback_low:
                    min_lookback_low = lows[j]

            is_new_low = current_low < min_lookback_low

            if not is_new_low:
                continue

            # Condition 2: Count successive new lows going backward
            successive_count = 0
            for j in range(i, 0, -1):
                if lows[j] < lows[j - 1]:
                    successive_count += 1
                else:
                    break

            if successive_count < min_successive:
                continue

            # Condition 3: Check for reversal bar (bullish candle)
            is_reversal = closes[i] > opens[i]

            if is_reversal:
                signals[i] = 1

        return signals

else:

    def detect_nbar_decline_signals_numba(
        lows: np.ndarray,
        highs: np.ndarray,
        opens: np.ndarray,
        closes: np.ndarray,
        min_successive: int,
        lookback_period: int,
    ) -> np.ndarray:
        """Fallback implementation without Numba."""
        n = len(lows)
        signals = np.zeros(n, dtype=np.int8)

        for i in range(lookback_period, n):
            current_low = lows[i]

            # Find minimum low in lookback period
            min_lookback_low = np.min(lows[i - lookback_period : i])
            is_new_low = current_low < min_lookback_low

            if not is_new_low:
                continue

            # Count successive new lows
            successive_count = 0
            for j in range(i, 0, -1):
                if lows[j] < lows[j - 1]:
                    successive_count += 1
                else:
                    break

            if successive_count < min_successive:
                continue

            # Check for reversal bar
            is_reversal = closes[i] > opens[i]

            if is_reversal:
                signals[i] = 1

        return signals


class NBarDecline(BasePattern):
    """
    n-Bar Decline Pattern Detector

    A counter-trend reversal pattern that identifies exhaustion after
    a series of consecutive new lows followed by a reversal bar.
    """

    def __init__(
        self,
        min_successive: int = 3,
        lookback_period: int = 21,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        require_volume_spike: bool = False,
        volume_threshold: float = 1.5,
    ):
        """
        Initialize n-Bar Decline pattern detector.

        Args:
            min_successive: Minimum number of successive new lows
            lookback_period: Lookback period for trend context
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            require_volume_spike: Require volume spike on reversal bar
            volume_threshold: Volume multiplier threshold (e.g., 1.5x average)
        """
        super().__init__(
            name="n-Bar Decline",
            pattern_type=PatternType.COUNTER_TREND,
            min_bars_required=lookback_period + 1,
        )
        self.min_successive = min_successive
        self.lookback_period = lookback_period
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.require_volume_spike = require_volume_spike
        self.volume_threshold = volume_threshold

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized n-Bar Decline pattern detection using Numba JIT compilation.

        PHASE 2 OPTIMIZATION: 50-100x faster than bar-by-bar detection.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            numpy array of signal values:
            - 0 = no signal
            - 1 = n-bar decline reversal detected
        """
        # Extract arrays
        lows = df["Low"].values.astype(np.float64)
        highs = df["High"].values.astype(np.float64)
        opens = df["Open"].values.astype(np.float64)
        closes = df["Close"].values.astype(np.float64)

        # Run vectorized detection
        return detect_nbar_decline_signals_numba(
            lows,
            highs,
            opens,
            closes,
            self.min_successive,
            self.lookback_period,
        )

    def _find_decline_start(self, arrays: dict, i: int) -> Optional[Tuple[int, int]]:
        """
        Find the start of the n-bar decline.

        Args:
            arrays: Dictionary with NumPy arrays (open, high, low, close, volume)
            i: Current bar index (reversal bar)

        Returns:
            Tuple of (start_index, number_of_decline_bars) or None
        """
        if i < self.min_successive:
            return None

        # Count successive new lows going backward
        successive_count = 0
        start_idx = i
        low_arr = arrays["low"]

        for j in range(i, 0, -1):
            current_low = float(low_arr[j])
            prev_low = float(low_arr[j - 1])

            if current_low < prev_low:
                successive_count += 1
                start_idx = j
            else:
                break

        if successive_count >= self.min_successive:
            return (start_idx, successive_count)

        return None

    def _is_new_lookback_low(self, arrays: dict, i: int) -> bool:
        """
        Check if current bar makes a new lookback-period low.

        Args:
            arrays: Dictionary with NumPy arrays (open, high, low, close, volume)
            i: Current bar index

        Returns:
            True if new low
        """
        if i < self.lookback_period:
            return False

        low_arr = arrays["low"]
        current_low = float(low_arr[i])

        # Check if current low is the lowest in lookback period
        lookback_lows = low_arr[i - self.lookback_period : i]
        return current_low < np.min(lookback_lows)

    def _is_reversal_bar(self, arrays: dict, i: int) -> bool:
        """
        Check if bar i is a bullish reversal bar.

        Reversal bar: Close > Open (bullish candle)

        Args:
            arrays: Dictionary with NumPy arrays (open, high, low, close, volume)
            i: Current bar index

        Returns:
            True if reversal bar
        """
        open_arr = arrays["open"]
        close_arr = arrays["close"]

        open_price = float(open_arr[i])
        close = float(close_arr[i])

        return close > open_price

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect n-Bar Decline pattern at bar index i.

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

        # Condition 1: New lookback-period low
        is_new_low = self._is_new_lookback_low(arrays, i)

        # Find decline start and count
        decline_info = self._find_decline_start(arrays, i)

        if decline_info is None:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        start_idx, successive_count = decline_info

        # Condition 2: At least min_successive new lows
        condition_2 = successive_count >= self.min_successive

        # Condition 3: Reversal bar (bullish candle)
        is_reversal = self._is_reversal_bar(arrays, i)

        # All conditions must be met
        detected = is_new_low and condition_2 and is_reversal

        if not detected:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                pivot_points={
                    "is_new_low": is_new_low,
                    "successive_count": successive_count,
                    "is_reversal_bar": is_reversal,
                },
            )

        # Generate signal
        signal = self.generate_signal(df, i, start_idx, successive_count)

        # Get the high of decline start for targets
        high_arr = arrays["high"]
        low_arr = arrays["low"]
        decline_start_high = float(high_arr[start_idx])
        current_low = float(low_arr[i])

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                "decline_start_idx": start_idx,
                "decline_start_high": decline_start_high,
                "decline_low": current_low,
                "successive_count": successive_count,
                "decline_range": decline_start_high - current_low,
            },
            bars_since_detection=0,
            start_index=start_idx,
            end_index=i,
        )

    def generate_signal(
        self, df: pd.DataFrame, i: int, start_idx: int, successive_count: int
    ) -> Optional[TradeSignal]:
        """
        Generate trade signal for n-Bar Decline pattern.

        Args:
            df: DataFrame with OHLCV data
            i: Current bar index (reversal bar)
            start_idx: Index where decline started
            successive_count: Number of successive decline bars

        Returns:
            TradeSignal if valid entry point, None otherwise
        """
        arrays = self._extract_arrays(df)
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        current_high = float(high_arr[i])
        current_low = float(low_arr[i])
        decline_start_high = float(high_arr[start_idx])

        # Entry: Above reversal bar high
        entry_price = current_high + self.entry_offset

        # Stop loss: Below reversal bar low
        stop_loss = current_low - self.stop_offset

        # Decline range for targets
        decline_range = decline_start_high - current_low

        # Targets based on Fibonacci retracement
        take_profit_1 = entry_price + (decline_range * 0.62)  # 62% retracement
        take_profit_2 = entry_price + decline_range  # 100% retracement

        # Find prior swing high for alternative target
        swing_high = get_recent_swing_high(df, start_idx, lookback=50)
        if swing_high:
            take_profit_3 = swing_high[1]
        else:
            take_profit_3 = entry_price + (decline_range * 1.27)

        # Volume confirmation
        volume_confirmed = True
        if self.require_volume_spike:
            vol_sma = volume_sma(df["Volume"], 20)
            if i < len(vol_sma):
                current_vol = float(arrays["volume"][i]) if len(arrays["volume"]) > 0 else 0.0
                avg_vol = self._safe_float(vol_sma.iloc[i])
                volume_confirmed = current_vol > (avg_vol * self.volume_threshold)

        # Calculate confidence
        confidence = 0.55
        if successive_count >= 5:
            confidence += 0.1  # More exhaustion
        if volume_confirmed:
            confidence += 0.1
        if self.require_volume_spike and volume_confirmed:
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
                "successive_count": successive_count,
                "decline_start_idx": start_idx,
                "decline_range": decline_range,
                "volume_confirmed": volume_confirmed,
                "entry_type": "buy_stop",
            },
        )


class NBarRally(BasePattern):
    """
    n-Bar Rally Pattern Detector

    The opposite of n-Bar Decline - identifies exhaustion after
    a series of consecutive new highs followed by a reversal bar.
    """

    def __init__(
        self,
        min_successive: int = 3,
        lookback_period: int = 21,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        require_volume_spike: bool = False,
        volume_threshold: float = 1.5,
    ):
        super().__init__(
            name="n-Bar Rally",
            pattern_type=PatternType.COUNTER_TREND,
            min_bars_required=lookback_period + 1,
        )
        self.min_successive = min_successive
        self.lookback_period = lookback_period
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.require_volume_spike = require_volume_spike
        self.volume_threshold = volume_threshold

    def _find_rally_start(self, arrays: dict, i: int) -> Optional[Tuple[int, int]]:
        """Find the start of the n-bar rally."""
        if i < self.min_successive:
            return None

        successive_count = 0
        start_idx = i
        high_arr = arrays["high"]

        for j in range(i, 0, -1):
            current_high = float(high_arr[j])
            prev_high = float(high_arr[j - 1])

            if current_high > prev_high:
                successive_count += 1
                start_idx = j
            else:
                break

        if successive_count >= self.min_successive:
            return (start_idx, successive_count)

        return None

    def _is_new_lookback_high(self, arrays: dict, i: int) -> bool:
        """Check if current bar makes a new lookback-period high."""
        if i < self.lookback_period:
            return False

        high_arr = arrays["high"]
        current_high = float(high_arr[i])

        # Check if current high is the highest in lookback period
        lookback_highs = high_arr[i - self.lookback_period : i]
        return current_high > np.max(lookback_highs)

    def _is_reversal_bar(self, arrays: dict, i: int) -> bool:
        """Check if bar i is a bearish reversal bar."""
        open_arr = arrays["open"]
        close_arr = arrays["close"]

        open_price = float(open_arr[i])
        close = float(close_arr[i])

        return close < open_price

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """Detect n-Bar Rally pattern at bar index i."""
        if not self._validate_data(df, i, window_start):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # PERFORMANCE OPTIMIZATION: Use NumPy arrays for faster access
        arrays = self._extract_arrays(df)

        is_new_high = self._is_new_lookback_high(arrays, i)
        rally_info = self._find_rally_start(arrays, i)

        if rally_info is None:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        start_idx, successive_count = rally_info
        condition_2 = successive_count >= self.min_successive
        is_reversal = self._is_reversal_bar(arrays, i)

        detected = is_new_high and condition_2 and is_reversal

        if not detected:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        signal = self.generate_signal(df, i, start_idx, successive_count)

        high_arr = arrays["high"]
        low_arr = arrays["low"]
        rally_start_low = float(low_arr[start_idx])
        current_high = float(high_arr[i])

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                "rally_start_idx": start_idx,
                "rally_start_low": rally_start_low,
                "rally_high": current_high,
                "successive_count": successive_count,
                "rally_range": current_high - rally_start_low,
            },
            bars_since_detection=0,
            start_index=start_idx,
            end_index=i,
        )

    def generate_signal(
        self, df: pd.DataFrame, i: int, start_idx: int, successive_count: int
    ) -> Optional[TradeSignal]:
        """Generate short trade signal for n-Bar Rally pattern."""
        arrays = self._extract_arrays(df)
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        current_high = float(high_arr[i])
        current_low = float(low_arr[i])
        rally_start_low = float(low_arr[start_idx])

        entry_price = current_low - self.entry_offset
        stop_loss = current_high + self.stop_offset

        rally_range = current_high - rally_start_low

        take_profit_1 = entry_price - (rally_range * 0.62)
        take_profit_2 = entry_price - rally_range
        take_profit_3 = entry_price - (rally_range * 1.27)

        confidence = 0.55
        if successive_count >= 5:
            confidence += 0.1

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
                "successive_count": successive_count,
                "rally_start_idx": start_idx,
                "rally_range": rally_range,
                "entry_type": "sell_stop",
            },
        )
