"""
Market Structure Low (MSL) Pattern

Detection Logic:
- Define Close Sequence: C[-2], C[-1], C[0] (where 0 is current bar)
- Condition 1 (Down Move): C[-1] < C[-2]
- Condition 2 (Higher Low of Close): C[0] > C[-1] AND C[0] < C[-2]
- Condition 3 (Confirmation): Close[i+1] > Max(C[-2], C[-1], C[0])

Entry Rules:
- Long Entry: Buy Stop order at Max(C[-2], C[-1], C[0]) + 0.01
- Valid for next 3 to 5 bars after confirmation

Stop Loss Rules:
- Stop Loss: Low[0] - 0.01 (Low of the MSL formation bar)
- Alternative Stop: Major Swing Low prior to formation

Take Profit Rules:
- Target 1: Market Structure High (MSH) formation
- Target 2: Close[i] < Low[i-1] (Exit on reversal signal)

PHASE 2 OPTIMIZATION:
- Vectorized detection using Numba JIT compilation
- 50-100x faster than bar-by-bar detection
"""

from typing import Optional

import numpy as np
import pandas as pd

from ...indicators.pivots import get_recent_swing_low
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
    def detect_msl_signals_numba(closes: np.ndarray, lows: np.ndarray) -> np.ndarray:
        """
        JIT-compiled MSL pattern detection.

        FIXED: Removed lookahead bias. Signal is generated at confirmation bar (i),
        not at pattern formation bar. Pattern forms at bars i-3, i-2, i-1 and
        is confirmed by bar i's close.

        Returns array of signals:
        - 0 = no pattern
        - 1 = long signal (confirmed MSL)
        """
        n = len(closes)
        signals = np.zeros(n, dtype=np.int8)

        # Start from bar 3 to have enough history for pattern (i-3, i-2, i-1)
        for i in range(3, n):
            # Pattern formation bars: i-3, i-2, i-1
            c_minus_3 = closes[i - 3]
            c_minus_2 = closes[i - 2]
            c_minus_1 = closes[i - 1]
            # Confirmation bar: i (current)
            c_0 = closes[i]

            # Condition 1: Down Move - C[-2] < C[-3]
            cond1 = c_minus_2 < c_minus_3

            # Condition 2: Higher Low of Close - C[-1] > C[-2] AND C[-1] < C[-3]
            cond2 = (c_minus_1 > c_minus_2) and (c_minus_1 < c_minus_3)

            # Condition 3: Confirmation - Close[i] > Max(C[-3], C[-2], C[-1])
            max_close = max(c_minus_3, c_minus_2, c_minus_1)
            cond3 = c_0 > max_close

            if cond1 and cond2 and cond3:
                signals[i] = 1

        return signals
else:

    def detect_msl_signals_numba(closes: np.ndarray, lows: np.ndarray) -> np.ndarray:
        """Fallback implementation without Numba."""
        n = len(closes)
        signals = np.zeros(n, dtype=np.int8)

        # Start from bar 3 to have enough history for pattern (i-3, i-2, i-1)
        for i in range(3, n):
            # Pattern formation bars: i-3, i-2, i-1
            c_minus_3 = closes[i - 3]
            c_minus_2 = closes[i - 2]
            c_minus_1 = closes[i - 1]
            # Confirmation bar: i (current)
            c_0 = closes[i]

            # Condition 1: Down Move - C[-2] < C[-3]
            cond1 = c_minus_2 < c_minus_3

            # Condition 2: Higher Low of Close - C[-1] > C[-2] AND C[-1] < C[-3]
            cond2 = (c_minus_1 > c_minus_2) and (c_minus_1 < c_minus_3)

            # Condition 3: Confirmation - Close[i] > Max(C[-3], C[-2], C[-1])
            max_close = max(c_minus_3, c_minus_2, c_minus_1)
            cond3 = c_0 > max_close

            if cond1 and cond2 and cond3:
                signals[i] = 1

        return signals


class MarketStructureLow(BasePattern):
    """
    Market Structure Low (MSL) Pattern Detector

    A reversal pattern that identifies a potential bottom formation
    based on a specific sequence of closing prices.
    """

    def __init__(
        self,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        confirmation_bars: int = 5,
        volume_filter: bool = False,
    ):
        """
        Initialize MSL pattern detector.

        Args:
            entry_offset: Price offset for entry orders
            stop_offset: Price offset for stop loss
            confirmation_bars: Maximum bars after confirmation for valid entry
            volume_filter: Whether to require volume confirmation
        """
        super().__init__(
            name="Market Structure Low", pattern_type=PatternType.REVERSAL, min_bars_required=5
        )
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.confirmation_bars = confirmation_bars
        self.volume_filter = volume_filter

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized MSL pattern detection using Numba JIT compilation.

        PHASE 2 OPTIMIZATION: 50-100x faster than bar-by-bar detection.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            numpy array of signal values: 0=none, 1=long
        """
        closes: np.ndarray = np.asarray(df["Close"].values, dtype=np.float64)
        lows: np.ndarray = np.asarray(df["Low"].values, dtype=np.float64)
        return detect_msl_signals_numba(closes, lows)  # type: ignore[no-any-return]

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect MSL pattern at bar index i.

        FIXED: Removed lookahead bias. At bar i (confirmation bar), we check
        if the pattern formed at bars i-3, i-2, i-1 and is confirmed by bar i.

        Args:
            df: DataFrame with OHLCV data
            i: Current bar index (confirmation bar)
            window_start: Optional window start for bounds checking (avoids DataFrame slicing)

        Returns:
            PatternResult with detection status and signal
        """
        if not self._validate_data(df, i, window_start):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Need at least 4 bars: 3 for pattern formation + 1 for confirmation
        if i < 3:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # PERFORMANCE OPTIMIZATION: Use NumPy arrays for faster access
        arrays = self._extract_arrays(df)

        # Pattern formation bars: i-3, i-2, i-1
        # Confirmation bar: i (current)
        c_minus_3 = float(arrays["close"][i - 3])
        c_minus_2 = float(arrays["close"][i - 2])
        c_minus_1 = float(arrays["close"][i - 1])
        c_0 = float(arrays["close"][i])

        # Condition 1: Down Move - C[-2] < C[-3]
        condition_1 = c_minus_2 < c_minus_3

        # Condition 2: Higher Low of Close - C[-1] > C[-2] AND C[-1] < C[-3]
        condition_2 = (c_minus_1 > c_minus_2) and (c_minus_1 < c_minus_3)

        # Check if pattern is forming (conditions 1 and 2 met)
        pattern_forming = condition_1 and condition_2

        if not pattern_forming:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Condition 3: Confirmation - Close[i] > Max(C[-3], C[-2], C[-1])
        max_close = max(c_minus_3, c_minus_2, c_minus_1)
        confirmed = c_0 > max_close

        if not confirmed:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                pivot_points={
                    "msl_high_close": max_close,
                    "msl_low": float(arrays["low"][i - 1]),  # Low of the pattern formation bar
                },
            )

        # Pattern is confirmed - generate signal
        signal = self.generate_signal(df, i - 1)  # Signal based on pattern formation bar

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                "msl_high_close": max_close,
                "msl_low": float(arrays["low"][i - 1]),
                "c_minus_3": c_minus_3,
                "c_minus_2": c_minus_2,
                "c_minus_1": c_minus_1,
                "c_0": c_0,
            },
            bars_since_detection=0,
            start_index=i - 3,
            end_index=i,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """
        Generate trade signal for MSL pattern.

        Args:
            df: DataFrame with OHLCV data
            i: Current bar index (MSL formation bar)

        Returns:
            TradeSignal if valid entry point, None otherwise
        """
        if i < 2:
            return None

        # Get the close sequence
        c_minus_2 = self._safe_float(df.iloc[i - 2]["Close"])
        c_minus_1 = self._safe_float(df.iloc[i - 1]["Close"])
        c_0 = self._safe_float(df.iloc[i]["Close"])

        # Entry price: Max of the three closes + offset
        max_close = max(c_minus_2, c_minus_1, c_0)
        entry_price = max_close + self.entry_offset

        # Stop loss: Low of MSL bar - offset
        msl_low = self._safe_float(df.iloc[i]["Low"])
        stop_loss = msl_low - self.stop_offset

        # Get swing low for alternative target calculation
        swing_low = get_recent_swing_low(df, i, lookback=50)

        # Calculate targets
        # Target 1: Risk-reward based (conservative)
        risk = entry_price - stop_loss
        take_profit_1 = entry_price + (risk * 2)  # 2:1 R:R

        # Target 2: 3:1 R:R
        take_profit_2 = entry_price + (risk * 3)

        # Volume confirmation
        volume_confirmed = True
        if self.volume_filter:
            vol_sma = volume_sma(df["Volume"], 20)
            if i < len(vol_sma):
                current_vol = self._safe_float(df.iloc[i]["Volume"])
                avg_vol = self._safe_float(vol_sma.iloc[i])
                volume_confirmed = current_vol > avg_vol

        # Confidence based on pattern quality
        confidence = 0.6
        if volume_confirmed:
            confidence += 0.1

        # Check for downtrend context (pattern should occur after downtrend)
        if i >= 10:
            current_low = self._safe_float(df.iloc[i]["Low"])
            prior_low = self._safe_float(df.iloc[i - 10]["Low"])
            if current_low < prior_low:
                confidence += 0.1  # Pattern occurs after downtrend

        return TradeSignal(
            pattern_name=self.name,
            direction=SignalDirection.LONG,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=None,
            confidence=min(confidence, 1.0),
            timestamp=pd.Timestamp(df.iloc[i].name)
            if hasattr(df.iloc[i], "name") and df.iloc[i].name is not None
            else None,  # type: ignore[arg-type]
            metadata={
                "msl_high_close": max_close,
                "msl_low": msl_low,
                "risk_reward_ratio": 2.0,
                "volume_confirmed": volume_confirmed,
                "entry_type": "buy_stop",
            },
        )


class MarketStructureHigh(BasePattern):
    """
    Market Structure High (MSH) Pattern Detector

    The opposite of MSL - identifies potential top formations.
    """

    def __init__(
        self,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        confirmation_bars: int = 5,
        volume_filter: bool = False,
    ):
        super().__init__(
            name="Market Structure High", pattern_type=PatternType.REVERSAL, min_bars_required=5
        )
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset
        self.confirmation_bars = confirmation_bars
        self.volume_filter = volume_filter

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect MSH pattern at bar index i.

        FIXED: Removed lookahead bias. At bar i (confirmation bar), we check
        if the pattern formed at bars i-3, i-2, i-1 and is confirmed by bar i.

        MSH Logic (opposite of MSL):
        - Condition 1 (Up Move): C[-2] > C[-3]
        - Condition 2 (Lower High of Close): C[-1] < C[-2] AND C[-1] > C[-3]
        - Condition 3 (Confirmation): Close[i] < Min(C[-3], C[-2], C[-1])
        """
        if not self._validate_data(df, i, window_start):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        if i < 3:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Pattern formation bars: i-3, i-2, i-1
        # Confirmation bar: i (current)
        c_minus_3 = self._safe_float(df.iloc[i - 3]["Close"])
        c_minus_2 = self._safe_float(df.iloc[i - 2]["Close"])
        c_minus_1 = self._safe_float(df.iloc[i - 1]["Close"])
        c_0 = self._safe_float(df.iloc[i]["Close"])

        # Condition 1: Up Move - C[-2] > C[-3]
        condition_1 = c_minus_2 > c_minus_3

        # Condition 2: Lower High of Close - C[-1] < C[-2] AND C[-1] > C[-3]
        condition_2 = (c_minus_1 < c_minus_2) and (c_minus_1 > c_minus_3)

        pattern_forming = condition_1 and condition_2

        if not pattern_forming:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Condition 3: Confirmation - Close[i] < Min(C[-3], C[-2], C[-1])
        min_close = min(c_minus_3, c_minus_2, c_minus_1)
        confirmed = c_0 < min_close

        if not confirmed:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        signal = self.generate_signal(df, i - 1)  # Signal based on pattern formation bar

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                "msh_low_close": min_close,
                "msh_high": self._safe_float(df.iloc[i - 1]["High"]),
                "c_minus_3": c_minus_3,
                "c_minus_2": c_minus_2,
                "c_minus_1": c_minus_1,
                "c_0": c_0,
            },
            bars_since_detection=0,
            start_index=i - 3,
            end_index=i,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate short trade signal for MSH pattern."""
        if i < 2:
            return None

        c_minus_2 = self._safe_float(df.iloc[i - 2]["Close"])
        c_minus_1 = self._safe_float(df.iloc[i - 1]["Close"])
        c_0 = self._safe_float(df.iloc[i]["Close"])

        min_close = min(c_minus_2, c_minus_1, c_0)
        entry_price = min_close - self.entry_offset

        msh_high = self._safe_float(df.iloc[i]["High"])
        stop_loss = msh_high + self.stop_offset

        risk = stop_loss - entry_price
        take_profit_1 = entry_price - (risk * 2)
        take_profit_2 = entry_price - (risk * 3)

        confidence = 0.6

        return TradeSignal(
            pattern_name=self.name,
            direction=SignalDirection.SHORT,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=None,
            confidence=confidence,
            timestamp=pd.Timestamp(df.iloc[i].name)
            if hasattr(df.iloc[i], "name") and df.iloc[i].name is not None
            else None,  # type: ignore[arg-type]
            metadata={"msh_low_close": min_close, "msh_high": msh_high, "entry_type": "sell_stop"},
        )
