"""
Ichimoku Cloud Pattern Detector

Multi-component Japanese trend-following system. Uses five lines to assess
trend direction, momentum, and dynamic support/resistance.

Components:
- Tenkan-sen (Conversion): (highest_high + lowest_low) / 2 over 9 periods
- Kijun-sen (Base): (highest_high + lowest_low) / 2 over 26 periods
- Senkou Span A (Leading 1): (Tenkan + Kijun) / 2, displaced 26 periods forward
- Senkou Span B (Leading 2): (highest_high + lowest_low) / 2 over 52 periods, displaced 26
- Chikou Span (Lagging): Close displaced 26 periods backward

Detection Logic:
- LONG: Tenkan crosses ABOVE Kijun + price above cloud + bullish cloud color
- SHORT: Tenkan crosses BELOW Kijun + price below cloud + bearish cloud color

Entry Rules:
- Long: TK bullish cross + price above cloud + Senkou A > Senkou B
- Short: TK bearish cross + price below cloud + Senkou A < Senkou B

Stop Loss: Below/above Kijun-sen for long/short
Take Profit: 2× ATR from entry

Confidence: 0.58
"""

from typing import Optional

import numpy as np
import pandas as pd

from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal
from ...indicators.technical import atr


class IchimokuPattern(BasePattern):
    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26,
    ):
        super().__init__(
            name="Ichimoku Cloud",
            pattern_type=PatternType.CONTINUATION,
            min_bars_required=senkou_b_period + displacement + 2,
        )
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement

    def _donchian_mid(self, high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
        n = len(high)
        result = np.full(n, np.nan)
        for i in range(period - 1, n):
            result[i] = (
                np.max(high[i - period + 1 : i + 1]) + np.min(low[i - period + 1 : i + 1])
            ) / 2
        return result

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        result = np.zeros(n, dtype=np.int8)

        high = df["High"].to_numpy(dtype=np.float64)
        low = df["Low"].to_numpy(dtype=np.float64)
        close = df["Close"].to_numpy(dtype=np.float64)

        tenkan = self._donchian_mid(high, low, self.tenkan_period)
        kijun = self._donchian_mid(high, low, self.kijun_period)
        senkou_a = (tenkan + kijun) / 2.0
        senkou_b = self._donchian_mid(high, low, self.senkou_b_period)

        cloud_upper = np.full(n, np.nan)
        cloud_lower = np.full(n, np.nan)
        for i in range(n):
            if not np.isnan(senkou_a[i]) and not np.isnan(senkou_b[i]):
                cloud_upper[i] = max(senkou_a[i], senkou_b[i])
                cloud_lower[i] = min(senkou_a[i], senkou_b[i])

        start = self.min_bars_required
        for i in range(start, n):
            if i < 1:
                continue
            disp_idx = i - self.displacement
            if disp_idx < 0:
                continue

            cloud_up = cloud_upper[disp_idx]
            cloud_lo = cloud_lower[disp_idx]

            if (
                np.isnan(tenkan[i])
                or np.isnan(kijun[i])
                or np.isnan(tenkan[i - 1])
                or np.isnan(kijun[i - 1])
            ):
                continue
            if np.isnan(cloud_up) or np.isnan(cloud_lo):
                continue

            bullish_tk = tenkan[i] > kijun[i] and tenkan[i - 1] <= kijun[i - 1]
            bearish_tk = tenkan[i] < kijun[i] and tenkan[i - 1] >= kijun[i - 1]

            price_above_cloud = close[i] > cloud_up
            price_below_cloud = close[i] < cloud_lo
            bullish_cloud = cloud_up > cloud_lo

            if bullish_tk and price_above_cloud and bullish_cloud:
                result[i] = 1
            elif bearish_tk and price_below_cloud and not bullish_cloud:
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
                tp = entry_price + 3.0 * atr_val
            else:
                sl = entry_price + 1.5 * atr_val
                tp = entry_price - 3.0 * atr_val

            signal = TradeSignal(
                pattern_name=self.name,
                direction=direction,
                entry_price=entry_price,
                stop_loss=sl,
                take_profit_1=tp,
                confidence=0.58,
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
