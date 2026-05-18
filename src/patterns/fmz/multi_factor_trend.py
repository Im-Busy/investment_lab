"""
Multi-Factor Trend Following — Quad Confirmation System

Origin: FMZ strategies repo, PineScript v5, Author: ianzeng123

Detection Logic:
- Parabolic SAR: price crosses above/below SAR
- EMA(2): fast trend confirmation (close > EMA for long)
- RSI(6): momentum direction (RSI > 60 long, RSI < 40 short)
- ADX(14) >= 30: trend strength filter

Entry Rules:
- Long: crossover(close, SAR) AND close > EMA(2) AND RSI(6) > 60 AND ADX >= 30
- Short: crossunder(close, SAR) AND close < EMA(2) AND RSI(6) < 40 AND ADX >= 30

Stop Loss Rules:
- Long SL: close - ATR(14) × 1.5
- Short SL: close + ATR(14) × 1.5

Take Profit Rules:
- Long TP: close + ATR × 1.5 × 2 (2:1 R:R)
- Short TP: close - ATR × 1.5 × 2
"""

from typing import Optional

import numpy as np
import pandas as pd

from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal
from ...indicators.pinescript_helpers import crossover, crossunder, dmi, nz, sar
from ...indicators.technical import ema, rsi


class MultiFactorTrend(BasePattern):
    """Quad-confirmation trend following: SAR + EMA + RSI + ADX."""

    def __init__(
        self,
        sar_start: float = 0.02,
        sar_increment: float = 0.02,
        sar_max: float = 0.2,
        rsi_length: int = 6,
        ema_fast_length: int = 2,
        adx_threshold: float = 30.0,
        atr_multiplier: float = 1.5,
    ):
        super().__init__(
            name="Multi-Factor Trend",
            pattern_type=PatternType.CONTINUATION,
            min_bars_required=max(30, rsi_length + 5, 14 + 5),
        )
        self.sar_start = sar_start
        self.sar_increment = sar_increment
        self.sar_max = sar_max
        self.rsi_length = rsi_length
        self.ema_fast_length = ema_fast_length
        self.adx_threshold = adx_threshold
        self.atr_multiplier = atr_multiplier

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        if not self._validate_data(df, i, window_start):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        arrays = self._extract_arrays(df)
        close_arr = arrays["close"]
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        sar_vals = sar(high_arr, low_arr, self.sar_start, self.sar_increment, self.sar_max)
        ema_fast = ema(pd.Series(close_arr), self.ema_fast_length).to_numpy()
        rsi_vals = rsi(pd.Series(close_arr), self.rsi_length).to_numpy()
        _, _, adx_vals = dmi(high_arr, low_arr, close_arr, 14, 14)

        if i < self.min_bars_required:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        if np.isnan(sar_vals[i]) or np.isnan(rsi_vals[i]) or np.isnan(adx_vals[i]):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        long_cond = (
            crossover(close_arr[i - 1 : i + 1], sar_vals[i - 1 : i + 1])[-1]
            and close_arr[i] > ema_fast[i]
            and rsi_vals[i] > 60
            and adx_vals[i] >= self.adx_threshold
        )

        short_cond = (
            crossunder(close_arr[i - 1 : i + 1], sar_vals[i - 1 : i + 1])[-1]
            and close_arr[i] < ema_fast[i]
            and rsi_vals[i] < 40
            and adx_vals[i] >= self.adx_threshold
        )

        if not long_cond and not short_cond:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        signal = self._generate_signal(arrays, i, rsi_vals[i], adx_vals[i], long_cond)
        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            start_index=i,
            end_index=i,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        result = self.detect(df, i)
        return result.signal if result.detected else None

    def _generate_signal(
        self, arrays: dict, i: int, rsi_val: float, adx_val: float, is_long: bool
    ) -> TradeSignal:
        close_price = float(arrays["close"][i])
        high_arr = arrays["high"]
        low_arr = arrays["low"]
        atr_val = self._calc_atr_value(high_arr, low_arr, arrays["close"], i, 14)
        sl_distance = atr_val * self.atr_multiplier

        if is_long:
            entry_price = close_price
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + sl_distance * 2.0
            confidence = min(
                0.50 + (adx_val - self.adx_threshold) * 0.005 + (rsi_val - 60) * 0.002, 0.85
            )
            return TradeSignal(
                pattern_name="Multi-Factor Trend Long",
                direction=SignalDirection.LONG,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit,
                confidence=confidence,
                metadata={
                    "entry_type": "market",
                    "rsi": round(rsi_val, 1),
                    "adx": round(adx_val, 1),
                    "atr": round(atr_val, 4),
                },
            )
        else:
            entry_price = close_price
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - sl_distance * 2.0
            confidence = min(
                0.50 + (adx_val - self.adx_threshold) * 0.005 + (40 - rsi_val) * 0.002, 0.85
            )
            return TradeSignal(
                pattern_name="Multi-Factor Trend Short",
                direction=SignalDirection.SHORT,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit,
                confidence=confidence,
                metadata={
                    "entry_type": "market",
                    "rsi": round(rsi_val, 1),
                    "adx": round(adx_val, 1),
                    "atr": round(atr_val, 4),
                },
            )

    def _calc_atr_value(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, i: int, period: int
    ) -> float:
        start = max(0, i - period * 2)
        tr = []
        for j in range(start + 1, i + 1):
            tr.append(
                max(
                    high[j] - low[j],
                    abs(high[j] - close[j - 1]),
                    abs(low[j] - close[j - 1]),
                )
            )
        if not tr:
            return 0.0
        atr_prev = np.mean(tr[:period]) if len(tr) >= period else np.mean(tr)
        for j in range(min(period, len(tr)), len(tr)):
            atr_prev = (atr_prev * (period - 1) + tr[j]) / period
        return atr_prev
