"""
EMA-MACD High-Frequency Strategy

Origin: FMZ strategies repo, PineScript v5, Author: ChaoZhang

Detection Logic:
- EMA(9/21) crossover with MACD confirmation
- Ultra-short MACD parameters (6, 13, 4) for high-frequency signals

Entry Rules:
- Long: crossover(EMA9, EMA21) AND MACD line > signal line
- Short: crossunder(EMA9, EMA21) AND MACD line < signal line

Stop Loss Rules:
- Long SL: entry - ATR(14) × 1.5
- Short SL: entry + ATR(14) × 1.5

Take Profit Rules:
- Long TP: entry + (entry - SL) × 2.0 (1:2 R:R)
- Short TP: entry - (SL - entry) × 2.0
"""

from typing import Optional

import numpy as np
import pandas as pd

from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal
from ...indicators.pinescript_helpers import crossover, crossunder, macd, nz
from ...indicators.technical import ema


class EMAMACDHF(BasePattern):
    """EMA(9/21) crossover + MACD(6,13,4) high-frequency confirmation."""

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        macd_fast: int = 6,
        macd_slow: int = 13,
        macd_signal: int = 4,
        atr_period: int = 14,
        sl_atr_mult: float = 1.5,
        rr_ratio: float = 2.0,
    ):
        super().__init__(
            name="EMA-MACD HF",
            pattern_type=PatternType.CONTINUATION,
            min_bars_required=max(ema_slow + 5, macd_slow + macd_signal + 5, atr_period + 5),
        )
        self.ema_fast_period = ema_fast
        self.ema_slow_period = ema_slow
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal_period = macd_signal
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.rr_ratio = rr_ratio

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        if not self._validate_data(df, i, window_start):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        arrays = self._extract_arrays(df)
        close_arr = arrays["close"]

        ema_fast_vals = ema(pd.Series(close_arr), self.ema_fast_period).to_numpy()
        ema_slow_vals = ema(pd.Series(close_arr), self.ema_slow_period).to_numpy()
        macd_line, signal_line, _ = macd(
            close_arr, self.macd_fast, self.macd_slow, self.macd_signal_period
        )

        if i < self.min_bars_required or i < 2:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        if np.isnan(ema_fast_vals[i]) or np.isnan(ema_slow_vals[i]) or np.isnan(macd_line[i]):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        long_cond = (
            crossover_value(
                ema_fast_vals[i], ema_slow_vals[i], ema_fast_vals[i - 1], ema_slow_vals[i - 1]
            )
            and macd_line[i] > signal_line[i]
        )
        short_cond = (
            crossunder_value(
                ema_fast_vals[i], ema_slow_vals[i], ema_fast_vals[i - 1], ema_slow_vals[i - 1]
            )
            and macd_line[i] < signal_line[i]
        )

        if not long_cond and not short_cond:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        signal = self._generate_signal(arrays, i, long_cond)
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

    def _generate_signal(self, arrays: dict, i: int, is_long: bool) -> TradeSignal:
        close_price = float(arrays["close"][i])
        high_arr = arrays["high"]
        low_arr = arrays["low"]
        atr_val = self._calc_atr_value(high_arr, low_arr, arrays["close"], i, self.atr_period)

        if is_long:
            entry_price = close_price
            sl_distance = atr_val * self.sl_atr_mult
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + sl_distance * self.rr_ratio
            return TradeSignal(
                pattern_name="EMA-MACD HF Long",
                direction=SignalDirection.LONG,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit,
                confidence=0.55,
                metadata={
                    "entry_type": "market",
                    "atr": round(atr_val, 4),
                    "ema9_21_crossover": True,
                },
            )
        else:
            entry_price = close_price
            sl_distance = atr_val * self.sl_atr_mult
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - sl_distance * self.rr_ratio
            return TradeSignal(
                pattern_name="EMA-MACD HF Short",
                direction=SignalDirection.SHORT,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit,
                confidence=0.55,
                metadata={
                    "entry_type": "market",
                    "atr": round(atr_val, 4),
                    "ema9_21_crossunder": True,
                },
            )

    def _calc_atr_value(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, i: int, period: int
    ) -> float:
        start = max(0, i - period * 2)
        tr_list = []
        for j in range(start + 1, i + 1):
            tr_list.append(
                max(
                    high[j] - low[j],
                    abs(high[j] - close[j - 1]),
                    abs(low[j] - close[j - 1]),
                )
            )
        if not tr_list:
            return 0.0
        atr_prev = np.mean(tr_list[: min(period, len(tr_list))])
        for j in range(min(period, len(tr_list)), len(tr_list)):
            atr_prev = (atr_prev * (period - 1) + tr_list[j]) / period
        return atr_prev


def crossover_value(a_curr: float, b_curr: float, a_prev: float, b_prev: float) -> bool:
    return a_curr > b_curr and a_prev <= b_prev


def crossunder_value(a_curr: float, b_curr: float, a_prev: float, b_prev: float) -> bool:
    return a_curr < b_curr and a_prev >= b_prev
