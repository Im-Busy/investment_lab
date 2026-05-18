"""
Adaptive Bollinger Bands Trend Following — Multi-Level Risk Management

Origin: FMZ strategies repo, PineScript v5, Author: ChaoZhang

Detection Logic:
- Bollinger Bands (14, 1.5 stddev) breakout REVERSAL (fades the break)
- Optional EMA(80) trend filter
- Short: previous candle closes above upper band AND previous bullish, current bearish
- Long: previous candle closes below lower band AND current bullish

Entry Rules:
- Long: close[1] < lower_band[1] AND close > open (current bullish reversal)
- Short: close[1] > upper_band[1] AND close[1] > open[1] AND close < open (bearish reversal)

Exit Rules (4-layer):
1. Trailing stop at BB mid-line crossover
2. Fixed $ stop loss
3. Fixed $ take profit
4. Max bars in trade (time-based exit)
"""

from typing import Optional

import numpy as np
import pandas as pd

from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal
from ...indicators.technical import ema, sma, std_dev


class AdaptiveBollinger(BasePattern):
    """Bollinger Band breakout reversion with 4-layer exit logic."""

    def __init__(
        self,
        bb_length: int = 14,
        bb_stddev: float = 1.5,
        use_ema_filter: bool = True,
        ema_length: int = 80,
        sl_dollars: float = 300.0,
        tp_dollars: float = 1000.0,
        bars_till_close: int = 10,
    ):
        super().__init__(
            name="Adaptive Bollinger",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=max(bb_length + 5, ema_length + 5 if use_ema_filter else 20),
        )
        self.bb_length = bb_length
        self.bb_stddev = bb_stddev
        self.use_ema_filter = use_ema_filter
        self.ema_length = ema_length
        self.sl_dollars = sl_dollars
        self.tp_dollars = tp_dollars
        self.bars_till_close = bars_till_close

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        if not self._validate_data(df, i, window_start) or i < 2:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        arrays = self._extract_arrays(df)
        close_arr = arrays["close"]
        open_arr = arrays["open"]

        close_series = pd.Series(close_arr)
        bb_basis = sma(close_series, self.bb_length)
        bb_std = std_dev(close_series, self.bb_length)
        upper_band = bb_basis + self.bb_stddev * bb_std
        lower_band = bb_basis - self.bb_stddev * bb_std

        if i < self.min_bars_required:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        upper_prev = upper_band.iloc[i - 1] if i - 1 < len(upper_band) else np.nan
        lower_prev = lower_band.iloc[i - 1] if i - 1 < len(lower_band) else np.nan

        if np.isnan(upper_prev) or np.isnan(lower_prev):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        buy_cond = close_arr[i - 1] < lower_prev and close_arr[i] > open_arr[i]
        sell_cond = (
            close_arr[i - 1] > upper_prev
            and close_arr[i - 1] > open_arr[i - 1]
            and close_arr[i] < open_arr[i]
        )

        if self.use_ema_filter:
            ema_vals = ema(close_series, self.ema_length).to_numpy()
            if i < self.ema_length or np.isnan(ema_vals[i]) or np.isnan(ema_vals[i - 1]):
                return PatternResult(
                    detected=False, pattern_name=self.name, pattern_type=self.pattern_type
                )
            if buy_cond and not (ema_vals[i] > ema_vals[i - 1]):
                buy_cond = False
            if sell_cond and not (ema_vals[i] < ema_vals[i - 1]):
                sell_cond = False

        if not buy_cond and not sell_cond:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        signal = self._generate_signal(arrays, i, bb_basis.iloc[i], buy_cond)
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

    def _generate_signal(self, arrays: dict, i: int, bb_mid: float, is_long: bool) -> TradeSignal:
        close_price = float(arrays["close"][i])

        if is_long:
            entry_price = close_price
            stop_loss = entry_price - self.sl_dollars
            take_profit = entry_price + self.tp_dollars
            return TradeSignal(
                pattern_name="Adaptive Bollinger Long",
                direction=SignalDirection.LONG,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit,
                take_profit_2=float(bb_mid),
                confidence=0.50,
                metadata={
                    "entry_type": "market",
                    "bb_mid": round(bb_mid, 4),
                    "exit_bars": self.bars_till_close,
                    "max_bars": self.bars_till_close,
                },
            )
        else:
            entry_price = close_price
            stop_loss = entry_price + self.sl_dollars
            take_profit = entry_price - self.tp_dollars
            return TradeSignal(
                pattern_name="Adaptive Bollinger Short",
                direction=SignalDirection.SHORT,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit,
                take_profit_2=float(bb_mid),
                confidence=0.50,
                metadata={
                    "entry_type": "market",
                    "bb_mid": round(bb_mid, 4),
                    "exit_bars": self.bars_till_close,
                    "max_bars": self.bars_till_close,
                },
            )
