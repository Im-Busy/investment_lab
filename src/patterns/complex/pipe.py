"""
Pipe Pattern Detector

Two-bar mechanical reversal pattern with zero parameters. Non-Fibonacci, purely
price-structure based. From Insight #8 (Duddella pipe pattern).

Detection Logic:
- Pipe Up (bullish reversal): two consecutive red candles (close < open),
  second fully engulfs first (high > prior_high AND low < prior_low).
- Pipe Down (bearish reversal): two consecutive green candles (close > open),
  second fully engulfs first.

Entry: beyond extreme of both pipes (bullish=above high[i], bearish=below low[i])
Stop: opposite extreme (bullish=below low[i], bearish=above high[i])
Target: L = body size of larger candle, T1 = L, T2 = 2L
Confidence: 0.55 base, +0.05 if second bar volume > first
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class PipePattern(BasePattern):
    """Two-bar pipe reversal pattern detector."""

    def __init__(self, entry_offset: float = 0.01, stop_offset: float = 0.01):
        super().__init__(
            name="Pipe Pattern",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=2,
        )
        self.entry_offset = entry_offset
        self.stop_offset = stop_offset

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        result = np.zeros(n, dtype=np.int8)
        if n < self.min_bars_required:
            return result

        open_a = df["Open"].to_numpy()
        high_a = df["High"].to_numpy()
        low_a = df["Low"].to_numpy()
        close_a = df["Close"].to_numpy()
        vol_a = df["Volume"].to_numpy()

        for i in range(2, n):
            body_c0 = close_a[i - 1] - open_a[i - 1]
            body_c1 = close_a[i] - open_a[i]
            is_red0 = body_c0 < 0
            is_red1 = body_c1 < 0
            is_green0 = body_c0 > 0
            is_green1 = body_c1 > 0

            engulfs = high_a[i] > high_a[i - 1] and low_a[i] < low_a[i - 1]
            if not engulfs:
                continue

            if is_red0 and is_red1:
                result[i] = 1
            elif is_green0 and is_green1:
                result[i] = -1

        return result

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        if not self._validate_data(df, i, window_start=window_start):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        open0 = float(df.iloc[i - 1]["Open"])
        high0 = float(df.iloc[i - 1]["High"])
        low0 = float(df.iloc[i - 1]["Low"])
        close0 = float(df.iloc[i - 1]["Close"])
        open1 = float(df.iloc[i]["Open"])
        high1 = float(df.iloc[i]["High"])
        low1 = float(df.iloc[i]["Low"])
        close1 = float(df.iloc[i]["Close"])
        vol1 = float(df.iloc[i].get("Volume", 0))
        vol0 = float(df.iloc[i - 1].get("Volume", 0))

        body0 = close0 - open0
        body1 = close1 - open1
        is_red0 = body0 < 0
        is_red1 = body1 < 0
        is_green0 = body0 > 0
        is_green1 = body1 > 0

        engulfs = high1 > high0 and low1 < low0
        if not engulfs:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        if is_red0 and is_red1:
            direction = "long"
        elif is_green0 and is_green1:
            direction = "short"
        else:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        signal = self._generate_signal(
            open0, high0, low0, close0, open1, high1, low1, close1, vol0, vol1, direction
        )

        return PatternResult(
            detected=True,
            pattern_name=f"{self.name} ({direction.title()})",
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                "candle0_body": abs(body0),
                "candle1_body": abs(body1),
                "larger_body": max(abs(body0), abs(body1)),
            },
            start_index=i - 1,
            end_index=i,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        result = self.detect(df, i)
        return result.signal if result.detected else None

    def _generate_signal(
        self,
        open0: float,
        high0: float,
        low0: float,
        close0: float,
        open1: float,
        high1: float,
        low1: float,
        close1: float,
        vol0: float,
        vol1: float,
        direction: str,
    ) -> TradeSignal:
        body0 = abs(close0 - open0)
        body1 = abs(close1 - open1)
        body_max = max(body0, body1)

        confidence = 0.55
        if vol1 > vol0:
            confidence += 0.05

        if direction == "long":
            entry_price = high1 + self.entry_offset
            stop_loss = low1 - self.stop_offset
            take_profit_1 = entry_price + body_max
            take_profit_2 = entry_price + 2 * body_max
            sig_dir = SignalDirection.LONG
        else:
            entry_price = low1 - self.entry_offset
            stop_loss = high1 + self.stop_offset
            take_profit_1 = entry_price - body_max
            take_profit_2 = entry_price - 2 * body_max
            sig_dir = SignalDirection.SHORT

        return TradeSignal(
            pattern_name=f"{self.name} ({direction.title()})",
            direction=sig_dir,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=None,
            confidence=min(confidence, 1.0),
            metadata={
                "body_c0": body0,
                "body_c1": body1,
                "target_l": body_max,
                "vol_greater": vol1 > vol0,
            },
        )
