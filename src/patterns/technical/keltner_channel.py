"""
Keltner Channel Pattern Detector

Volatility-based channel system using EMA and ATR. Identifies:
- Pullbacks to lower band in uptrend → LONG continuation
- Rejections from upper band in downtrend → SHORT continuation

Detection Logic:
- Compute EMA(20) basis + ATR(10) × multiplier bands
- Uptrend: price > 50 EMA AND low touches/hovers near lower band → LONG
- Downtrend: price < 50 EMA AND high touches/hovers near upper band → SHORT

Entry Rules:
- Long: Price within lower band zone + bullish candle close
- Short: Price within upper band zone + bearish candle close

Stop Loss: 2× ATR from entry
Take Profit: 3× ATR from entry

Confidence: 0.60
"""

from typing import Optional

import numpy as np
import pandas as pd

from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal
from ...indicators.technical import atr


class KeltnerChannelPattern(BasePattern):
    def __init__(
        self,
        ema_length: int = 20,
        atr_length: int = 10,
        atr_multiplier: float = 2.0,
        trend_ema: int = 50,
        band_zone_pct: float = 0.10,
    ):
        super().__init__(
            name="Keltner Channel",
            pattern_type=PatternType.CONTINUATION,
            min_bars_required=max(ema_length, atr_length, trend_ema) + 1,
        )
        self.ema_length = ema_length
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.trend_ema = trend_ema
        self.band_zone_pct = band_zone_pct

    def _compute_ema(self, series: np.ndarray, period: int) -> np.ndarray:
        result = np.full(len(series), np.nan)
        alpha = 2.0 / (period + 1)
        for i in range(len(series)):
            if np.isnan(series[i]):
                continue
            if i == 0 or np.isnan(result[i - 1]):
                result[i] = series[i]
            else:
                result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
        return result

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        result = np.zeros(n, dtype=np.int8)

        close = df["Close"].to_numpy(dtype=np.float64)
        high = df["High"].to_numpy(dtype=np.float64)
        low = df["Low"].to_numpy(dtype=np.float64)
        vol = df["Volume"].to_numpy(dtype=np.float64) if "Volume" in df.columns else np.ones(n)

        atr_arr = atr(df, self.atr_length)
        if isinstance(atr_arr, pd.Series):
            atr_arr = atr_arr.to_numpy(dtype=np.float64)
        basis = self._compute_ema(close, self.ema_length)
        trend = self._compute_ema(close, self.trend_ema)
        upper = basis + atr_arr * self.atr_multiplier
        lower = basis - atr_arr * self.atr_multiplier

        start = max(self.ema_length, self.atr_length, self.trend_ema)
        for i in range(start, n):
            if np.isnan(basis[i]) or np.isnan(atr_arr[i]) or np.isnan(trend[i]):
                continue

            band_width = upper[i] - lower[i]
            if band_width <= 0:
                continue
            zone = band_width * self.band_zone_pct

            in_uptrend = close[i] > trend[i]
            in_downtrend = close[i] < trend[i]

            if in_uptrend:
                dist_from_lower = (low[i] - lower[i]) / band_width if band_width > 0 else 999
                if dist_from_lower <= self.band_zone_pct:
                    if close[i] > close[i - 1] if i > 0 else True:
                        result[i] = 1

            elif in_downtrend:
                dist_from_upper = (upper[i] - high[i]) / band_width if band_width > 0 else 999
                if dist_from_upper <= self.band_zone_pct:
                    if close[i] < close[i - 1] if i > 0 else True:
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
            atr_val = atr(df.iloc[: i + 1], self.atr_length)[-1]
            entry_price = df["Close"].iloc[i]

            if direction == SignalDirection.LONG:
                sl = entry_price - 2.0 * atr_val
                tp = entry_price + 3.0 * atr_val
            else:
                sl = entry_price + 2.0 * atr_val
                tp = entry_price - 3.0 * atr_val

            signal = TradeSignal(
                pattern_name=self.name,
                direction=direction,
                entry_price=entry_price,
                stop_loss=sl,
                take_profit_1=tp,
                confidence=0.60,
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
