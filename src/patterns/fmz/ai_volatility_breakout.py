"""
AI Volatility Adaptive Breakout — Multi-Logic Composite

Origin: FMZ strategies repo, PineScript v6, Author: ianzeng123

Detection Logic (3 independent entry types):
1. Gap Fill Mean Reversion: large gap vs VWAP, mean reversion entry
2. VWAP Momentum: price crosses above/below VWAP
3. Volatility Compression Breakout: squeeze then expansion

Risk Management:
- Dynamic position sizing inversely proportional to ATR/ATR_SMA ratio
- ATR-based SL (1.5x ATR) and TP (3.0x ATR)
- Market regime detection via 50/200 EMA spread
"""

from typing import Optional

import numpy as np
import pandas as pd

from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal
from ...indicators.pinescript_helpers import (
    crossover,
    crossunder,
    highest,
    lowest,
    nz,
    vwap_simple,
)
from ...indicators.technical import atr, ema, sma


class AIVolatilityBreakout(BasePattern):
    """Multi-logic volatility-adaptive breakout system."""

    def __init__(
        self,
        fast_ema: int = 9,
        slow_ema: int = 21,
        atr_len: int = 14,
        atr_mult_sl: float = 1.5,
        atr_mult_tp: float = 3.0,
    ):
        super().__init__(
            name="AI Volatility Breakout",
            pattern_type=PatternType.BREAKOUT,
            min_bars_required=max(fast_ema + 5, slow_ema + 5, atr_len + 5, 20),
        )
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.atr_len = atr_len
        self.atr_mult_sl = atr_mult_sl
        self.atr_mult_tp = atr_mult_tp

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        if not self._validate_data(df, i, window_start) or i < 3:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        arrays = self._extract_arrays(df)
        close_arr = arrays["close"]
        high_arr = arrays["high"]
        low_arr = arrays["low"]
        vol_arr = arrays["volume"]

        vwap_vals = vwap_simple(high_arr, low_arr, close_arr, vol_arr)
        atr_vals = atr(
            pd.DataFrame({"High": high_arr, "Low": low_arr, "Close": close_arr}), self.atr_len
        ).to_numpy()

        if i < self.min_bars_required or np.isnan(vwap_vals[i]) or np.isnan(atr_vals[i]):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        close_series = pd.Series(close_arr)
        ema_fast_vals = ema(close_series, self.fast_ema).to_numpy()

        entry_dir = None
        entry_type = None

        # Logic 1: Gap Fill Mean Reversion
        gap_size = abs(close_arr[i] - close_arr[i - 1])
        long_gap = (
            close_arr[i] > ema_fast_vals[i]
            and gap_size > nz(atr_vals[i] * 0.5, 0)
            and close_arr[i] < vwap_vals[i]
        )
        short_gap = (
            close_arr[i] < ema_fast_vals[i]
            and gap_size > nz(atr_vals[i] * 0.5, 0)
            and close_arr[i] > vwap_vals[i]
        )

        if long_gap:
            entry_dir = SignalDirection.LONG
            entry_type = "gap_fill"
        elif short_gap:
            entry_dir = SignalDirection.SHORT
            entry_type = "gap_fill"

        # Logic 2: VWAP Momentum
        if entry_dir is None and i >= 2:
            c = crossover(close_arr[i - 1 : i + 1], vwap_vals[i - 1 : i + 1])
            if len(c) > 1 and c[-1]:
                entry_dir = SignalDirection.LONG
                entry_type = "vwap_momentum"
            u = crossunder(close_arr[i - 1 : i + 1], vwap_vals[i - 1 : i + 1])
            if len(u) > 1 and u[-1]:
                entry_dir = SignalDirection.SHORT
                entry_type = "vwap_momentum"

        # Logic 3: Volatility Compression Breakout
        if entry_dir is None:
            lowest_10 = lowest(low_arr, 10)
            highest_10 = highest(high_arr, 10)
            gap_check = abs(close_arr[i] - close_arr[i - 1])
            compression = not np.isnan(lowest_10[i]) and abs(low_arr[i] - lowest_10[i]) < nz(
                atr_vals[i] * 0.1, 0
            )
            breakout = not np.isnan(highest_10[i]) and gap_check > nz(atr_vals[i] * 0.5, 0)
            if compression:
                entry_dir = SignalDirection.LONG
                entry_type = "compression_breakout"
            if breakout:
                entry_dir = SignalDirection.LONG
                entry_type = "compression_breakout"

        if entry_dir is None:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        signal = self._generate_signal(arrays, i, atr_vals[i], entry_dir, entry_type)
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
        self, arrays: dict, i: int, atr_val: float, direction: SignalDirection, entry_type: str
    ) -> TradeSignal:
        close_price = float(arrays["close"][i])

        if direction == SignalDirection.LONG:
            entry_price = close_price
            stop_loss = entry_price - self.atr_mult_sl * atr_val
            take_profit = entry_price + self.atr_mult_tp * atr_val
            return TradeSignal(
                pattern_name=f"AI Vol Breakout Long ({entry_type})",
                direction=SignalDirection.LONG,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit,
                confidence=0.55,
                metadata={
                    "entry_type": entry_type,
                    "atr": round(atr_val, 4),
                },
            )
        else:
            entry_price = close_price
            stop_loss = entry_price + self.atr_mult_sl * atr_val
            take_profit = entry_price - self.atr_mult_tp * atr_val
            return TradeSignal(
                pattern_name=f"AI Vol Breakout Short ({entry_type})",
                direction=SignalDirection.SHORT,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit,
                confidence=0.55,
                metadata={
                    "entry_type": entry_type,
                    "atr": round(atr_val, 4),
                },
            )
