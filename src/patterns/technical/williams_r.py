"""
Williams %R Pattern Detector

Momentum oscillator that measures overbought/oversold conditions relative
to the highest high and lowest low over a lookback period.

Detection Logic:
- %R = (Highest High - Close) / (Highest High - Lowest Low) × -100
- Oversold: %R < -80 → price near bottom of range → potential LONG reversal
- Overbought: %R > -20 → price near top of range → potential SHORT reversal

Entry Rules:
- Long: %R crosses ABOVE -80 (exit oversold) with bullish confirmation
- Short: %R crosses BELOW -20 (exit overbought) with bearish confirmation

Stop Loss: 1.5× ATR from entry
Take Profit: 2.5× ATR from entry

Confidence: 0.52
"""

from typing import Optional

import numpy as np
import pandas as pd

from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal
from ...indicators.technical import atr


class WilliamsRPattern(BasePattern):
    def __init__(self, period: int = 14, oversold: float = -80.0, overbought: float = -20.0):
        super().__init__(
            name="Williams %R",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=period + 2,
        )
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def _compute_williams_r(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> np.ndarray:
        n = len(highs)
        result = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            highest = np.max(highs[i - self.period + 1 : i + 1])
            lowest = np.min(lows[i - self.period + 1 : i + 1])
            if highest != lowest:
                result[i] = (highest - closes[i]) / (highest - lowest) * -100.0
        return result

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        result = np.zeros(n, dtype=np.int8)

        high = df["High"].to_numpy(dtype=np.float64)
        low = df["Low"].to_numpy(dtype=np.float64)
        close = df["Close"].to_numpy(dtype=np.float64)

        wr = self._compute_williams_r(high, low, close)

        for i in range(self.min_bars_required, n):
            if np.isnan(wr[i]) or np.isnan(wr[i - 1]):
                continue

            exiting_oversold = wr[i - 1] <= self.oversold and wr[i] > self.oversold
            exiting_overbought = wr[i - 1] >= self.overbought and wr[i] < self.overbought

            if exiting_oversold:
                if close[i] > close[i - 1]:
                    result[i] = 1
            elif exiting_overbought:
                if close[i] < close[i - 1]:
                    result[i] = -1

        return result

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        if i < self.min_bars_required:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        signals = self.detect_vectorized(df)
        if i < len(signals) and signals[i] != 0:
            direction = SignalDirection.LONG if signals[i] == 1 else SignalDirection.SHORT
            atr_val = atr(df.iloc[: i + 1], 14)[-1]
            entry_price = df["Close"].iloc[i]

            if direction == SignalDirection.LONG:
                sl = entry_price - 1.5 * atr_val
                tp = entry_price + 2.5 * atr_val
            else:
                sl = entry_price + 1.5 * atr_val
                tp = entry_price - 2.5 * atr_val

            signal = TradeSignal(
                pattern_name=self.name,
                direction=direction,
                entry_price=entry_price,
                stop_loss=sl,
                take_profit_1=tp,
                confidence=0.52,
                timestamp=df.index[i],
                metadata={"atr": atr_val},
            )
            return PatternResult(
                detected=True,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                signal=signal,
            )
        return PatternResult(detected=False, pattern_name=self.name, pattern_type=self.pattern_type)

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        result = self.detect(df, i)
        return result.signal if result.detected else None
