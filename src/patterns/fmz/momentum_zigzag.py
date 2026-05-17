"""
Momentum-based ZigZag with Force Detection

Origin: FMZ strategies repo, PineScript v5, Author: ChaoZhang

Detection Logic:
- Selectable momentum indicator (MACD/MA/QQE) determines pivot direction
- Non-repainting ZigZag draws at the exact moment of momentum reversal
- Force Detection: RSI(5) overbought/oversold distinguishes strong vs weak legs

Entry Rules:
- Long: momentumUP AND NOT force_down (prior down-leg lacked momentum)
- Short: momentumDOWN AND NOT force_up (prior up-leg lacked momentum)

Stop Loss Rules:
- Long SL: most recent swing low from ZigZag
- Short SL: most recent swing high from ZigZag

Take Profit Rules:
- Fixed TP level (absolute dollar/point target)
"""

from typing import Optional

import numpy as np
import pandas as pd

from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal
from ...indicators.pinescript_helpers import barssince, macd, nz, qqe
from ...indicators.technical import ema, rsi, sma


class MomentumZigZag(BasePattern):
    """Momentum-driven ZigZag pivot detection with force analysis."""

    MOMENTUM_MACD = "MACD"
    MOMENTUM_MA = "MovingAverage"
    MOMENTUM_QQE = "QQE"

    def __init__(
        self,
        momentum_indicator: str = "QQE",
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal_period: int = 9,
        ma_length: int = 20,
        ma_type: str = "SMA",
        qqe_factor: float = 4.238,
        rsi_length: int = 14,
        rsi_smoothing: int = 5,
        qqe_threshold: float = 10.0,
        rsi_force_period: int = 5,
        ob_level: float = 80.0,
        os_level: float = 20.0,
        take_profit_points: float = 200.0,
    ):
        super().__init__(
            name="Momentum ZigZag",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=max(
                macd_slow + macd_signal_period + 10, ma_length + 10, rsi_length + 10
            ),
        )
        self.momentum_indicator = momentum_indicator
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal_period = macd_signal_period
        self.ma_length = ma_length
        self.ma_type = ma_type
        self.qqe_factor = qqe_factor
        self.rsi_length = rsi_length
        self.rsi_smoothing = rsi_smoothing
        self.qqe_threshold = qqe_threshold
        self.rsi_force_period = rsi_force_period
        self.ob_level = ob_level
        self.os_level = os_level
        self.take_profit_points = take_profit_points

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        if not self._validate_data(df, i, window_start):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        arrays = self._extract_arrays(df)
        close_arr = arrays["close"]
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        # Compute momentum direction
        momentum_up, momentum_down = self._compute_momentum(arrays, i)

        if i < self.min_bars_required:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        if not momentum_up and not momentum_down:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Force detection via RSI(5)
        force_up, force_down = self._detect_force(arrays, i, momentum_up, momentum_down)

        long_signal = momentum_up and not force_down
        short_signal = momentum_down and not force_up

        if not long_signal and not short_signal:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        zigzag_level = self._get_zigzag_level(arrays, i, momentum_up)

        signal = self._generate_signal(arrays, i, zigzag_level, long_signal, force_up, force_down)
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

    def _compute_momentum(self, arrays: dict, i: int) -> tuple:
        """Compute momentum direction based on selected indicator."""
        close_arr = arrays["close"]
        high_arr = arrays["high"]
        low_arr = arrays["low"]
        vol_arr = arrays["volume"]

        if self.momentum_indicator == self.MOMENTUM_MACD:
            macd_line, signal_line, _ = macd(
                close_arr, self.macd_fast, self.macd_slow, self.macd_signal_period
            )
            if i < 2 or np.isnan(macd_line[i]) or np.isnan(signal_line[i]):
                return False, False
            up = macd_line[i] > signal_line[i] and macd_line[i - 1] <= signal_line[i - 1]
            down = macd_line[i] < signal_line[i] and macd_line[i - 1] >= signal_line[i - 1]

        elif self.momentum_indicator == self.MOMENTUM_MA:
            ma_vals = self._compute_ma(close_arr, self.ma_length, self.ma_type)
            if i < 3 or np.isnan(ma_vals[i]):
                return False, False
            up = ma_vals[i] > ma_vals[i - 1] and ma_vals[i - 2] > ma_vals[i - 1]
            down = ma_vals[i] < ma_vals[i - 1] and ma_vals[i - 2] < ma_vals[i - 1]

        elif self.momentum_indicator == self.MOMENTUM_QQE:
            rsi_vals = rsi(pd.Series(close_arr), self.rsi_length).to_numpy()
            qqe_crosses = qqe(
                rsi_vals, self.qqe_factor, self.rsi_smoothing, self.qqe_threshold, self.rsi_length
            )
            if i >= len(qqe_crosses):
                return False, False
            up = qqe_crosses[i] == 1
            down = qqe_crosses[i] == -1

        else:
            return False, False

        return up, down

    def _compute_ma(self, source: np.ndarray, length: int, ma_type: str) -> np.ndarray:
        """Compute moving average based on type."""
        series = pd.Series(source)
        if ma_type == "EMA":
            return ema(series, length).to_numpy()
        elif ma_type == "WMA":
            weights = np.arange(1, length + 1)
            return (
                series.rolling(length)
                .apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
                .to_numpy()
            )
        elif ma_type == "HMA":
            half = int(length / 2)
            sqrt = int(np.sqrt(length))
            wma_half = series.rolling(half).apply(
                lambda x: np.dot(x, np.arange(1, half + 1)) / np.arange(1, half + 1).sum(), raw=True
            )
            wma_full = series.rolling(length).apply(
                lambda x: np.dot(x, np.arange(1, length + 1)) / np.arange(1, length + 1).sum(),
                raw=True,
            )
            hma_raw = 2 * wma_half - wma_full
            return (
                hma_raw.rolling(sqrt)
                .apply(
                    lambda x: np.dot(x, np.arange(1, sqrt + 1)) / np.arange(1, sqrt + 1).sum(),
                    raw=True,
                )
                .to_numpy()
            )
        else:
            return sma(series, length).to_numpy()

    def _detect_force(self, arrays: dict, i: int, momentum_up: bool, momentum_down: bool) -> tuple:
        """Detect whether prior leg had momentum force via RSI(5) overbought/oversold."""
        close_arr = arrays["close"]
        rsi5 = rsi(pd.Series(close_arr), self.rsi_force_period).to_numpy()

        rsi5_ob = rsi5 > self.ob_level
        rsi5_os = rsi5 < self.os_level
        rsi5_ob[: self.rsi_force_period] = False
        rsi5_os[: self.rsi_force_period] = False

        bars_since_ob = barssince(rsi5_ob)
        bars_since_os = barssince(rsi5_os)

        # Create momentum arrays for barssince calculation
        momentum_up_arr = np.zeros(len(close_arr), dtype=bool)
        momentum_down_arr = np.zeros(len(close_arr), dtype=bool)

        for j in range(len(close_arr)):
            mu, md = self._compute_momentum(arrays, j)
            momentum_up_arr[j] = mu
            momentum_down_arr[j] = md

        bars_since_up = barssince(momentum_up_arr)
        bars_since_down = barssince(momentum_down_arr)

        if i < 1:
            return False, False

        prev = i - 1
        if np.isnan(bars_since_up[prev]) or np.isnan(bars_since_ob[prev]):
            force_up = False
        else:
            force_up = momentum_down and bars_since_up[prev] >= bars_since_ob[prev]

        if np.isnan(bars_since_down[prev]) or np.isnan(bars_since_os[prev]):
            force_down = False
        else:
            force_down = momentum_up and bars_since_down[prev] >= bars_since_os[prev]

        return force_up, force_down

    def _get_zigzag_level(self, arrays: dict, i: int, momentum_up: bool) -> float:
        """Get the ZigZag pivot level at current bar."""
        high_arr = arrays["high"]
        low_arr = arrays["low"]
        close_arr = arrays["close"]

        if momentum_up:
            return float(low_arr[i])
        else:
            return float(high_arr[i])

    def _generate_signal(
        self,
        arrays: dict,
        i: int,
        zigzag_level: float,
        is_long: bool,
        force_up: bool,
        force_down: bool,
    ) -> TradeSignal:
        close_price = float(arrays["close"][i])
        high_arr = arrays["high"]
        low_arr = arrays["low"]

        if is_long:
            entry_price = close_price
            stop_loss = min(float(low_arr[i]), zigzag_level)
            take_profit = entry_price + self.take_profit_points
            return TradeSignal(
                pattern_name="Momentum ZigZag Long",
                direction=SignalDirection.LONG,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit,
                confidence=0.60,
                metadata={
                    "entry_type": "market",
                    "zigzag_level": round(zigzag_level, 4),
                    "force_down": force_down,
                    "indicator": self.momentum_indicator,
                },
            )
        else:
            entry_price = close_price
            stop_loss = max(float(high_arr[i]), zigzag_level)
            take_profit = entry_price - self.take_profit_points
            return TradeSignal(
                pattern_name="Momentum ZigZag Short",
                direction=SignalDirection.SHORT,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit,
                confidence=0.60,
                metadata={
                    "entry_type": "market",
                    "zigzag_level": round(zigzag_level, 4),
                    "force_up": force_up,
                    "indicator": self.momentum_indicator,
                },
            )
