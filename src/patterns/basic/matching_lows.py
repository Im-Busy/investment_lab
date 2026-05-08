"""
Matching Lows Pattern

Detection Logic:
- Define Support Level: L_support = Low[-2]
- Condition 1 (Test 1): Abs(Low[-1] - L_support) < epsilon (e.g., 0.05 * ATR)
- Condition 2 (Test 2): Abs(Low[0] - L_support) < epsilon
- Condition 3 (Duration): Minimum 3 consecutive bars testing this level without closing significantly below
- Condition 4 (Breakout): High[0] > High[-1] (Breakout bar formation)

Entry Rules:
- Long Entry: Buy Stop at High[0] + 0.01 (Above breakout bar high)
- Entry valid only if Close[0] > Open[0] (Bullish breakout bar)

Stop Loss Rules:
- Stop Loss: Min(Low[-2], Low[-1], Low[0]) - 0.01

Take Profit Rules:
- Target 1: Entry + (High[0] - Low[0]) (1x Breakout Bar Length)
- Target 2: Entry + 2 * (High[0] - Low[0]) (2x Breakout Bar Length)

PHASE 2 OPTIMIZATION:
- Vectorized detection using Numba JIT compilation
- 50-100x faster than bar-by-bar detection
"""

from typing import Optional

import numpy as np
import pandas as pd

from ...indicators.technical import atr, is_bullish_candle, volume_sma
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
    def detect_matching_lows_signals_numba(
        lows: np.ndarray,
        highs: np.ndarray,
        closes: np.ndarray,
        opens: np.ndarray,
        atr_values: np.ndarray,
        epsilon_atr_multiplier: float,
        min_test_bars: int,
        max_test_bars: int,
        require_bullish_breakout: bool,
    ) -> np.ndarray:
        """
        JIT-compiled Matching Lows pattern detection.

        Returns array of signals:
        - 0 = no pattern
        - 1 = long signal (matching lows with breakout)
        """
        n = len(lows)
        signals = np.zeros(n, dtype=np.int8)

        for i in range(min_test_bars, n):
            # Skip if ATR is not available
            if np.isnan(atr_values[i]):
                continue

            epsilon = atr_values[i] * epsilon_atr_multiplier

            # Look for matching lows in recent bars
            start_idx = max(0, i - max_test_bars)

            # Find potential support level from the lowest low
            support_level = lows[start_idx]
            for j in range(start_idx + 1, i + 1):
                if lows[j] < support_level:
                    support_level = lows[j]

            # Count bars that test this support level
            test_count = 0
            for j in range(start_idx, i + 1):
                low = lows[j]
                close = closes[j]

                # Check if low is within epsilon of support
                if abs(low - support_level) <= epsilon:
                    # Check that close is not significantly below support
                    if close >= support_level - epsilon:
                        test_count += 1

            # Need minimum test bars
            if test_count < min_test_bars:
                continue

            # Check for breakout (current high > previous high)
            if i > 0 and highs[i] > highs[i - 1]:
                # Check for bullish breakout bar if required
                if require_bullish_breakout:
                    if closes[i] > opens[i]:
                        signals[i] = 1
                else:
                    signals[i] = 1

        return signals

    @jit(nopython=True, cache=True)
    def detect_matching_highs_signals_numba(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        opens: np.ndarray,
        atr_values: np.ndarray,
        epsilon_atr_multiplier: float,
        min_test_bars: int,
        max_test_bars: int,
        require_bearish_breakout: bool,
    ) -> np.ndarray:
        """
        JIT-compiled Matching Highs pattern detection.

        Returns array of signals:
        - 0 = no pattern
        - -1 = short signal (matching highs with breakdown)
        """
        n = len(highs)
        signals = np.zeros(n, dtype=np.int8)

        for i in range(min_test_bars, n):
            # Skip if ATR is not available
            if np.isnan(atr_values[i]):
                continue

            epsilon = atr_values[i] * epsilon_atr_multiplier

            # Look for matching highs in recent bars
            start_idx = max(0, i - max_test_bars)

            # Find potential resistance level from the highest high
            resistance_level = highs[start_idx]
            for j in range(start_idx + 1, i + 1):
                if highs[j] > resistance_level:
                    resistance_level = highs[j]

            # Count bars that test this resistance level
            test_count = 0
            for j in range(start_idx, i + 1):
                high = highs[j]
                close = closes[j]

                # Check if high is within epsilon of resistance
                if abs(high - resistance_level) <= epsilon:
                    # Check that close is not significantly above resistance
                    if close <= resistance_level + epsilon:
                        test_count += 1

            # Need minimum test bars
            if test_count < min_test_bars:
                continue

            # Check for breakdown (current low < previous low)
            if i > 0 and lows[i] < lows[i - 1]:
                # Check for bearish breakdown bar if required
                if require_bearish_breakout:
                    if closes[i] < opens[i]:
                        signals[i] = -1
                else:
                    signals[i] = -1

        return signals

else:

    def detect_matching_lows_signals_numba(
        lows: np.ndarray,
        highs: np.ndarray,
        closes: np.ndarray,
        opens: np.ndarray,
        atr_values: np.ndarray,
        epsilon_atr_multiplier: float,
        min_test_bars: int,
        max_test_bars: int,
        require_bullish_breakout: bool,
    ) -> np.ndarray:
        """Fallback implementation without Numba."""
        n = len(lows)
        signals = np.zeros(n, dtype=np.int8)

        for i in range(min_test_bars, n):
            if np.isnan(atr_values[i]):
                continue

            epsilon = atr_values[i] * epsilon_atr_multiplier
            start_idx = max(0, i - max_test_bars)

            support_level = np.min(lows[start_idx : i + 1])

            test_count = 0
            for j in range(start_idx, i + 1):
                if abs(lows[j] - support_level) <= epsilon:
                    if closes[j] >= support_level - epsilon:
                        test_count += 1

            if test_count < min_test_bars:
                continue

            if i > 0 and highs[i] > highs[i - 1]:
                if require_bullish_breakout:
                    if closes[i] > opens[i]:
                        signals[i] = 1
                else:
                    signals[i] = 1

        return signals

    def detect_matching_highs_signals_numba(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        opens: np.ndarray,
        atr_values: np.ndarray,
        epsilon_atr_multiplier: float,
        min_test_bars: int,
        max_test_bars: int,
        require_bearish_breakout: bool,
    ) -> np.ndarray:
        """Fallback implementation without Numba."""
        n = len(highs)
        signals = np.zeros(n, dtype=np.int8)

        for i in range(min_test_bars, n):
            if np.isnan(atr_values[i]):
                continue

            epsilon = atr_values[i] * epsilon_atr_multiplier
            start_idx = max(0, i - max_test_bars)

            resistance_level = np.max(highs[start_idx : i + 1])

            test_count = 0
            for j in range(start_idx, i + 1):
                if abs(highs[j] - resistance_level) <= epsilon:
                    if closes[j] <= resistance_level + epsilon:
                        test_count += 1

            if test_count < min_test_bars:
                continue

            if i > 0 and lows[i] < lows[i - 1]:
                if require_bearish_breakout:
                    if closes[i] < opens[i]:
                        signals[i] = -1
                else:
                    signals[i] = -1

        return signals


class MatchingLows(BasePattern):
    """
    Matching Lows Pattern Detector

    A reversal pattern where multiple bars test the same support level
    and then break out to the upside.
    """

    def __init__(
        self,
        epsilon_atr_multiplier: float = 0.20,
        min_test_bars: int = 2,
        max_test_bars: int = 20,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        require_bullish_breakout: bool = True,
        volume_filter: bool = False,
    ):
        """
        Initialize Matching Lows pattern detector.

        FIXED: Relaxed parameters for daily timeframe compatibility.
        - epsilon_atr_multiplier: 0.05 -> 0.20 (was too strict, pattern rarely triggered)
        - min_test_bars: 3 -> 2 (reduced for daily data where finding 3+ tests is rare)

        Args:
            epsilon_atr_multiplier: ATR multiplier for level tolerance (default 0.20)
            min_test_bars: Minimum bars testing support level (default 2)
            max_test_bars: Maximum bars to look back for matching lows
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            require_bullish_breakout: Require bullish candle for entry
            volume_filter: Whether to require volume confirmation
        """
        super().__init__(
            name="Matching Lows", pattern_type=PatternType.REVERSAL, min_bars_required=10
        )
        self.epsilon_atr_multiplier = epsilon_atr_multiplier
        self.min_test_bars = min_test_bars
        self.max_test_bars = max_test_bars
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.require_bullish_breakout = require_bullish_breakout
        self.volume_filter = volume_filter

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized Matching Lows pattern detection using Numba JIT compilation.

        PHASE 2 OPTIMIZATION: 50-100x faster than bar-by-bar detection.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            numpy array of signal values:
            - 0 = no signal
            - 1 = long signal
        """
        # Extract arrays
        lows: np.ndarray = np.asarray(df["Low"].values, dtype=np.float64)
        highs: np.ndarray = np.asarray(df["High"].values, dtype=np.float64)
        closes: np.ndarray = np.asarray(df["Close"].values, dtype=np.float64)
        opens: np.ndarray = np.asarray(df["Open"].values, dtype=np.float64)

        # Pre-compute ATR
        atr_series = atr(df, period=14)
        atr_values: np.ndarray = np.asarray(atr_series.values, dtype=np.float64)

        # Run vectorized detection
        result = detect_matching_lows_signals_numba(
            lows,
            highs,
            closes,
            opens,
            atr_values,
            self.epsilon_atr_multiplier,
            self.min_test_bars,
            self.max_test_bars,
            self.require_bullish_breakout,
        )
        return result  # type: ignore[no-any-return]

    def _find_matching_lows(self, arrays: dict, i: int, epsilon: float) -> Optional[dict]:
        """
        Find bars with matching lows within epsilon tolerance.

        Args:
            arrays: Dictionary with NumPy arrays (open, high, low, close, volume)
            i: Current bar index
            epsilon: Price tolerance

        Returns:
            Dictionary with matching low info or None
        """
        if i < self.min_test_bars:
            return None

        # Look for matching lows in recent bars
        start_idx = max(0, i - self.max_test_bars)

        # Find potential support level from the lowest low
        low_arr = arrays["low"]
        close_arr = arrays["close"]

        lows_slice = low_arr[start_idx : i + 1]
        support_level = float(np.min(lows_slice))

        # Count bars that test this support level
        matching_bars = []
        for j in range(start_idx, i + 1):
            low = float(low_arr[j])
            close = float(close_arr[j])

            # Check if low is within epsilon of support
            if abs(low - support_level) <= epsilon:
                # Check that close is not significantly below support
                if close >= support_level - epsilon:
                    matching_bars.append(j)

        if len(matching_bars) >= self.min_test_bars:
            return {
                "support_level": support_level,
                "matching_bars": matching_bars,
                "num_tests": len(matching_bars),
            }

        return None

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Matching Lows pattern at bar index i.

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

        # Calculate ATR for epsilon tolerance
        atr_series = atr(df, period=14)
        if i >= len(atr_series) or pd.isna(atr_series.iloc[i]):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        current_atr = self._safe_float(atr_series.iloc[i])
        epsilon = current_atr * self.epsilon_atr_multiplier

        # Find matching lows
        match_info = self._find_matching_lows(arrays, i, epsilon)

        if match_info is None:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Check for breakout
        high_arr = arrays["high"]
        current_high = float(high_arr[i])
        prev_high = float(high_arr[i - 1]) if i > 0 else current_high

        # Condition 4: Breakout bar formation
        is_breakout = current_high > prev_high

        if not is_breakout:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                pivot_points={
                    "support_level": match_info["support_level"],
                    "num_tests": match_info["num_tests"],
                },
            )

        # Check for bullish breakout bar
        open_arr = arrays["open"]
        close_arr = arrays["close"]
        current_open = float(open_arr[i])
        current_close = float(close_arr[i])
        is_bullish = is_bullish_candle(current_open, current_close)

        # Generate signal
        signal = None
        if not self.require_bullish_breakout or is_bullish:
            signal = self.generate_signal(df, i, match_info)

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                "support_level": match_info["support_level"],
                "num_tests": match_info["num_tests"],
                "matching_bars": match_info["matching_bars"],
                "is_bullish_breakout": is_bullish,
            },
            bars_since_detection=0,
            start_index=match_info["matching_bars"][0] if match_info["matching_bars"] else i,
            end_index=i,
        )

    def generate_signal(
        self, df: pd.DataFrame, i: int, match_info: Optional[dict] = None
    ) -> Optional[TradeSignal]:  # type: ignore[override]
        """
        Generate trade signal for Matching Lows pattern.

        Args:
            df: DataFrame with OHLCV data
            i: Current bar index
            match_info: Dictionary with matching low information

        Returns:
            TradeSignal if valid entry point, None otherwise
        """
        assert match_info is not None, "match_info is required"
        arrays = self._extract_arrays(df)
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        current_high = float(high_arr[i])
        current_low = float(low_arr[i])

        # Entry: Above breakout bar high
        entry_price = current_high + self.entry_offset

        # Stop loss: Below lowest matching low
        matching_bars = match_info["matching_bars"]
        min_low = float(np.min(low_arr[matching_bars]))
        stop_loss = min_low - self.stop_offset

        # Breakout bar range for targets
        breakout_range = current_high - current_low

        # Targets: 1x and 2x breakout bar range
        take_profit_1 = entry_price + breakout_range
        take_profit_2 = entry_price + (2 * breakout_range)

        # Volume confirmation
        volume_confirmed = True
        if self.volume_filter:
            vol_sma = volume_sma(df["Volume"], 20)
            if i < len(vol_sma):
                current_vol = float(arrays["volume"][i]) if len(arrays["volume"]) > 0 else 0.0
                avg_vol = self._safe_float(vol_sma.iloc[i])
                # Volume on breakout should be higher than average
                volume_confirmed = current_vol > avg_vol

        # Calculate confidence
        confidence = 0.55
        if match_info["num_tests"] >= 4:
            confidence += 0.1  # More tests = stronger support
        if volume_confirmed:
            confidence += 0.1
        if self.require_bullish_breakout:
            confidence += 0.05

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
                "support_level": match_info["support_level"],
                "num_tests": match_info["num_tests"],
                "breakout_range": breakout_range,
                "volume_confirmed": volume_confirmed,
                "entry_type": "buy_stop",
            },
        )


class MatchingHighs(BasePattern):
    """
    Matching Highs Pattern Detector

    The opposite of Matching Lows - multiple bars test the same resistance
    level and then break down.
    """

    def __init__(
        self,
        epsilon_atr_multiplier: float = 0.05,
        min_test_bars: int = 3,
        max_test_bars: int = 20,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        require_bearish_breakout: bool = True,
        volume_filter: bool = False,
    ):
        super().__init__(
            name="Matching Highs", pattern_type=PatternType.REVERSAL, min_bars_required=10
        )
        self.epsilon_atr_multiplier = epsilon_atr_multiplier
        self.min_test_bars = min_test_bars
        self.max_test_bars = max_test_bars
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.require_bearish_breakout = require_bearish_breakout
        self.volume_filter = volume_filter

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized Matching Highs pattern detection using Numba JIT compilation.

        PHASE 2 OPTIMIZATION: 50-100x faster than bar-by-bar detection.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            numpy array of signal values:
            - 0 = no signal
            - -1 = short signal
        """
        # Extract arrays
        highs: np.ndarray = np.asarray(df["High"].values, dtype=np.float64)
        lows: np.ndarray = np.asarray(df["Low"].values, dtype=np.float64)
        closes: np.ndarray = np.asarray(df["Close"].values, dtype=np.float64)
        opens: np.ndarray = np.asarray(df["Open"].values, dtype=np.float64)

        # Pre-compute ATR
        atr_series = atr(df, period=14)
        atr_values: np.ndarray = np.asarray(atr_series.values, dtype=np.float64)

        # Run vectorized detection
        result = detect_matching_highs_signals_numba(
            highs,
            lows,
            closes,
            opens,
            atr_values,
            self.epsilon_atr_multiplier,
            self.min_test_bars,
            self.max_test_bars,
            self.require_bearish_breakout,
        )
        return result  # type: ignore[no-any-return]

    def _find_matching_highs(self, arrays: dict, i: int, epsilon: float) -> Optional[dict]:
        """Find bars with matching highs within epsilon tolerance."""
        if i < self.min_test_bars:
            return None

        start_idx = max(0, i - self.max_test_bars)

        # Find potential resistance level from the highest high
        high_arr = arrays["high"]
        close_arr = arrays["close"]

        highs_slice = high_arr[start_idx : i + 1]
        resistance_level = float(np.max(highs_slice))

        # Count bars that test this resistance level
        matching_bars = []
        for j in range(start_idx, i + 1):
            high = float(high_arr[j])
            close = float(close_arr[j])

            # Check if high is within epsilon of resistance
            if abs(high - resistance_level) <= epsilon:
                # Check that close is not significantly above resistance
                if close <= resistance_level + epsilon:
                    matching_bars.append(j)

        if len(matching_bars) >= self.min_test_bars:
            return {
                "resistance_level": resistance_level,
                "matching_bars": matching_bars,
                "num_tests": len(matching_bars),
            }

        return None

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """Detect Matching Highs pattern at bar index i."""
        if not self._validate_data(df, i, window_start):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # PERFORMANCE OPTIMIZATION: Use NumPy arrays for faster access
        arrays = self._extract_arrays(df)

        atr_series = atr(df, period=14)
        if i >= len(atr_series) or pd.isna(atr_series.iloc[i]):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        current_atr = self._safe_float(atr_series.iloc[i])
        epsilon = current_atr * self.epsilon_atr_multiplier

        match_info = self._find_matching_highs(arrays, i, epsilon)

        if match_info is None:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Check for breakdown
        low_arr = arrays["low"]
        current_low = float(low_arr[i])
        prev_low = float(low_arr[i - 1]) if i > 0 else current_low

        is_breakdown = current_low < prev_low

        if not is_breakdown:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                pivot_points={
                    "resistance_level": match_info["resistance_level"],
                    "num_tests": match_info["num_tests"],
                },
            )

        open_arr = arrays["open"]
        close_arr = arrays["close"]
        current_open = float(open_arr[i])
        current_close = float(close_arr[i])
        is_bearish = current_close < current_open

        signal = None
        if not self.require_bearish_breakout or is_bearish:
            signal = self.generate_signal(df, i, match_info)

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                "resistance_level": match_info["resistance_level"],
                "num_tests": match_info["num_tests"],
                "matching_bars": match_info["matching_bars"],
                "is_bearish_breakout": is_bearish,
            },
            bars_since_detection=0,
            start_index=match_info["matching_bars"][0] if match_info["matching_bars"] else i,
            end_index=i,
        )

    def generate_signal(
        self, df: pd.DataFrame, i: int, match_info: Optional[dict] = None
    ) -> Optional[TradeSignal]:  # type: ignore[override]
        """Generate short trade signal for Matching Highs pattern."""
        assert match_info is not None, "match_info is required"
        arrays = self._extract_arrays(df)
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        current_high = float(high_arr[i])
        current_low = float(low_arr[i])

        entry_price = current_low - self.entry_offset

        matching_bars = match_info["matching_bars"]
        max_high = float(np.max(high_arr[matching_bars]))
        stop_loss = max_high + self.stop_offset

        breakout_range = current_high - current_low

        take_profit_1 = entry_price - breakout_range
        take_profit_2 = entry_price - (2 * breakout_range)

        confidence = 0.55
        if match_info["num_tests"] >= 4:
            confidence += 0.1
        if self.require_bearish_breakout:
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
                "resistance_level": match_info["resistance_level"],
                "num_tests": match_info["num_tests"],
                "breakout_range": breakout_range,
                "entry_type": "sell_stop",
            },
        )
