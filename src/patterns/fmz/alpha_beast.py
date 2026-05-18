"""
Alpha Beast — Multi-Indicator Synergistic Dynamic Risk Control

Origin: FMZ strategies repo, PineScript v6, Author: ianzeng123

Detection Logic:
- Supertrend direction: uptrend = direction < close, downtrend = direction > close
- RSI threshold: RSI > 60 (long), RSI < 40 (short)
- Volume confirmation: volume > SMA(volume, 20) × 1.5

Entry Rules:
- Long: Supertrend uptrend AND RSI > 60 AND volume surge
- Short: Supertrend downtrend AND RSI < 40 AND volume surge

Stop Loss Rules:
- Long SL: close - ATR(14) × 1.2
- Short SL: close + ATR(14) × 1.2

Take Profit Rules:
- Long TP: close + ATR × 1.2 × 2.5 (2.5:1 R:R)
- Short TP: close - ATR × 1.2 × 2.5
"""

from typing import Optional

import numpy as np
import pandas as pd

from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal
from ...indicators.pinescript_helpers import nz, supertrend
from ...indicators.technical import ema, rsi, sma


class AlphaBeast(BasePattern):
    """Triple-confirmation strategy: Supertrend + RSI + Volume surge."""

    def __init__(
        self,
        rsi_length: int = 14,
        rsi_threshold: float = 60.0,
        atr_length: int = 14,
        atr_mult_sl: float = 1.2,
        rr_ratio: float = 2.5,
        supertrend_factor: float = 3.0,
        supertrend_length: int = 10,
        vol_mult: float = 1.5,
    ):
        super().__init__(
            name="Alpha Beast",
            pattern_type=PatternType.CONTINUATION,
            min_bars_required=max(20, rsi_length + 1, atr_length + 1, supertrend_length + 1),
        )
        self.rsi_length = rsi_length
        self.rsi_threshold = rsi_threshold
        self.atr_length = atr_length
        self.atr_mult_sl = atr_mult_sl
        self.rr_ratio = rr_ratio
        self.supertrend_factor = supertrend_factor
        self.supertrend_length = supertrend_length
        self.vol_mult = vol_mult

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        if not self._validate_data(df, i, window_start):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        arrays = self._extract_arrays(df)
        close_arr = arrays["close"]
        high_arr = arrays["high"]
        low_arr = arrays["low"]
        vol_arr = arrays["volume"]

        rsi_series = rsi(pd.Series(close_arr), self.rsi_length).to_numpy()
        _, st_direction = supertrend(
            high_arr, low_arr, close_arr, self.supertrend_factor, self.supertrend_length
        )
        vol_sma = sma(pd.Series(vol_arr), 20).to_numpy()

        if i < self.min_bars_required or np.isnan(rsi_series[i]) or st_direction[i] == 0:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        rsi_val = rsi_series[i]
        is_uptrend = st_direction[i] == 1
        is_downtrend = st_direction[i] == -1
        vol_boost = vol_arr[i] > vol_sma[i] * self.vol_mult

        long_cond = is_uptrend and rsi_val > self.rsi_threshold and vol_boost
        short_cond = is_downtrend and rsi_val < (100 - self.rsi_threshold) and vol_boost

        if not long_cond and not short_cond:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        signal = self._generate_signal(arrays, i, rsi_val, long_cond, short_cond)
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
        self,
        arrays: dict,
        i: int,
        rsi_val: float,
        long_cond: bool,
        short_cond: bool,
    ) -> Optional[TradeSignal]:
        close_price = float(arrays["close"][i])
        high_arr = arrays["high"]
        low_arr = arrays["low"]
        atr_series = self._calc_atr(high_arr, low_arr, arrays["close"], self.atr_length)
        atr_val = nz(atr_series[i], 0.0)

        if long_cond:
            entry_price = close_price
            sl_distance = atr_val * self.atr_mult_sl
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + sl_distance * self.rr_ratio
            confidence = min(0.55 + (rsi_val - self.rsi_threshold) * 0.005, 0.85)
            return TradeSignal(
                pattern_name="Alpha Beast Long",
                direction=SignalDirection.LONG,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit,
                confidence=confidence,
                metadata={
                    "entry_type": "market",
                    "rsi": round(rsi_val, 1),
                    "atr": round(atr_val, 4),
                    "supertrend_uptrend": True,
                },
            )

        if short_cond:
            entry_price = close_price
            sl_distance = atr_val * self.atr_mult_sl
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - sl_distance * self.rr_ratio
            rsi_strength = (100 - self.rsi_threshold) - rsi_val
            confidence = min(0.55 + max(rsi_strength, 0) * 0.005, 0.85)
            return TradeSignal(
                pattern_name="Alpha Beast Short",
                direction=SignalDirection.SHORT,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit,
                confidence=confidence,
                metadata={
                    "entry_type": "market",
                    "rsi": round(rsi_val, 1),
                    "atr": round(atr_val, 4),
                    "supertrend_downtrend": True,
                },
            )

        return None

    def _calc_atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> np.ndarray:
        """Internal ATR calculation matching TradingView's RMA-based ATR."""
        n = len(close)
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
        atr = np.full(n, np.nan)
        atr[period] = np.nanmean(tr[1 : period + 1])
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr
