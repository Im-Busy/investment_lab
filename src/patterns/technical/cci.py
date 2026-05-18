"""
Commodity Channel Index (CCI) Pattern Detector

Measures deviation of typical price from its SMA, normalized by mean absolute
deviation. Identifies extreme conditions that precede reversals.

Detection Logic:
- TP = (High + Low + Close) / 3
- CCI = (TP - SMA(TP, period)) / (0.015 × MAD)
- Oversold: CCI < -100 → potential LONG reversal
- Overbought: CCI > +100 → potential SHORT reversal

Entry Rules:
- Long: CCI crosses ABOVE -100 (exit oversold) + bullish bar
- Short: CCI crosses BELOW +100 (exit overbought) + bearish bar

Stop Loss: 1.5× ATR from entry
Take Profit: 2.5× ATR from entry

Confidence: 0.50
"""

from typing import Optional

import numpy as np
import pandas as pd

from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal
from ...indicators.technical import atr


class CCIPattern(BasePattern):
    def __init__(self, period: int = 20, oversold: float = -100.0, overbought: float = 100.0):
        super().__init__(
            name="CCI",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=period + 2,
            preferred_regimes=[],
            incompatible_regimes=[],
        )
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def _compute_cci(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
        n = len(highs)
        tp = (highs + lows + closes) / 3.0
        result = np.full(n, np.nan)

        for i in range(self.period - 1, n):
            window = tp[i - self.period + 1 : i + 1]
            sma = np.mean(window)
            mad = np.mean(np.abs(window - sma))
            if mad > 0:
                result[i] = (tp[i] - sma) / (0.015 * mad)
        return result

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        result = np.zeros(n, dtype=np.int8)

        high = df["High"].to_numpy(dtype=np.float64)
        low = df["Low"].to_numpy(dtype=np.float64)
        close = df["Close"].to_numpy(dtype=np.float64)

        cci = self._compute_cci(high, low, close)

        for i in range(self.min_bars_required, n):
            if np.isnan(cci[i]) or np.isnan(cci[i - 1]):
                continue

            exiting_bearish = cci[i - 1] <= self.oversold and cci[i] > self.oversold
            exiting_bullish = cci[i - 1] >= self.overbought and cci[i] < self.overbought

            if exiting_bearish:
                if close[i] > close[i - 1]:
                    result[i] = 1
            elif exiting_bullish:
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
                confidence=0.50,
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
