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

        Returns array of signals:
        - 0 = no pattern
        - 1 = long signal (confirmed MSL)
        """
        n = len(closes)
        signals = np.zeros(n, dtype=np.int8)

        for i in range(2, n - 1):
            c_minus_2 = closes[i - 2]
            c_minus_1 = closes[i - 1]
            c_0 = closes[i]
            c_plus_1 = closes[i + 1]

            # Condition 1: Down Move
            cond1 = c_minus_1 < c_minus_2

            # Condition 2: Higher Low of Close
            cond2 = (c_0 > c_minus_1) and (c_0 < c_minus_2)

            # Condition 3: Confirmation
            max_close = max(c_minus_2, c_minus_1, c_0)
            cond3 = c_plus_1 > max_close

            if cond1 and cond2 and cond3:
                signals[i] = 1

        return signals
else:

    def detect_msl_signals_numba(closes: np.ndarray, lows: np.ndarray) -> np.ndarray:
        """Fallback implementation without Numba."""
        n = len(closes)
        signals = np.zeros(n, dtype=np.int8)

        for i in range(2, n - 1):
            c_minus_2 = closes[i - 2]
            c_minus_1 = closes[i - 1]
            c_0 = closes[i]
            c_plus_1 = closes[i + 1]

            cond1 = c_minus_1 < c_minus_2
            cond2 = (c_0 > c_minus_1) and (c_0 < c_minus_2)
            max_close = max(c_minus_2, c_minus_1, c_0)
            cond3 = c_plus_1 > max_close

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
        closes = df["Close"].values.astype(np.float64)
        lows = df["Low"].values.astype(np.float64)
        return detect_msl_signals_numba(closes, lows)

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect MSL pattern at bar index i.

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

        # Need at least 3 bars for the pattern + 1 for confirmation
        if i < 3:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # PERFORMANCE OPTIMIZATION: Use NumPy arrays for faster access
        arrays = self._extract_arrays(df)

        # Get the close sequence: C[-2], C[-1], C[0]
        # At bar i, we're looking at bars i-2, i-1, i
        c_minus_2 = float(arrays["close"][i - 2])
        c_minus_1 = float(arrays["close"][i - 1])
        c_0 = float(arrays["close"][i])

        # Condition 1: Down Move - C[-1] < C[-2]
        condition_1 = c_minus_1 < c_minus_2

        # Condition 2: Higher Low of Close - C[0] > C[-1] AND C[0] < C[-2]
        condition_2 = (c_0 > c_minus_1) and (c_0 < c_minus_2)

        # Check if pattern is forming (conditions 1 and 2 met)
        pattern_forming = condition_1 and condition_2

        if not pattern_forming:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Condition 3: Confirmation - Check next bar
        # We need to check if bar i+1 confirms the pattern
        confirmed = False
        confirmation_index = None

        if i + 1 < len(arrays["close"]):
            c_plus_1 = float(arrays["close"][i + 1])
            max_close = max(c_minus_2, c_minus_1, c_0)
            if c_plus_1 > max_close:
                confirmed = True
                confirmation_index = i + 1

        # For real-time detection, we also allow checking if current bar
        # is within confirmation window of a previously formed pattern
        if not confirmed:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                pivot_points={
                    "msl_high_close": max(c_minus_2, c_minus_1, c_0),
                    "msl_low": float(arrays["low"][i]),
                },
            )

        # Pattern is confirmed - generate signal
        signal = self.generate_signal(df, i)

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                "msl_high_close": max(c_minus_2, c_minus_1, c_0),
                "msl_low": float(arrays["low"][i]),
                "c_minus_2": c_minus_2,
                "c_minus_1": c_minus_1,
                "c_0": c_0,
            },
            bars_since_detection=0,
            start_index=i - 2,
            end_index=confirmation_index if confirmation_index else i,
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
            timestamp=df.iloc[i].name if hasattr(df.iloc[i], "name") else None,
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

    def detect(self, df: pd.DataFrame, i: int) -> PatternResult:
        """
        Detect MSH pattern at bar index i.

        MSH Logic (opposite of MSL):
        - Condition 1 (Up Move): C[-1] > C[-2]
        - Condition 2 (Lower High of Close): C[0] < C[-1] AND C[0] > C[-2]
        - Condition 3 (Confirmation): Close[i+1] < Min(C[-2], C[-1], C[0])
        """
        if not self._validate_data(df, i):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        if i < 3:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        c_minus_2 = self._safe_float(df.iloc[i - 2]["Close"])
        c_minus_1 = self._safe_float(df.iloc[i - 1]["Close"])
        c_0 = self._safe_float(df.iloc[i]["Close"])

        # Condition 1: Up Move
        condition_1 = c_minus_1 > c_minus_2

        # Condition 2: Lower High of Close
        condition_2 = (c_0 < c_minus_1) and (c_0 > c_minus_2)

        pattern_forming = condition_1 and condition_2

        if not pattern_forming:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Condition 3: Confirmation
        confirmed = False
        confirmation_index = None

        if i + 1 < len(df):
            c_plus_1 = self._safe_float(df.iloc[i + 1]["Close"])
            min_close = min(c_minus_2, c_minus_1, c_0)
            if c_plus_1 < min_close:
                confirmed = True
                confirmation_index = i + 1

        if not confirmed:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        signal = self.generate_signal(df, i)

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                "msh_low_close": min(c_minus_2, c_minus_1, c_0),
                "msh_high": self._safe_float(df.iloc[i]["High"]),
            },
            bars_since_detection=0,
            start_index=i - 2,
            end_index=confirmation_index if confirmation_index else i,
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
            timestamp=df.iloc[i].name if hasattr(df.iloc[i], "name") else None,
            metadata={"msh_low_close": min_close, "msh_high": msh_high, "entry_type": "sell_stop"},
        )
