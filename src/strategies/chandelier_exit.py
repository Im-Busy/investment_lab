"""
Chandelier Exit Strategy for backtesting.py

Volatility-based trailing stop system. Enters when price breaks above/below
the Chandelier lines (highest high - ATR×multiplier / lowest low + ATR×multiplier).
Exits when price crosses back through the respective line.

Pine Script source: strategies/trend-following/chandelier-exit/strategy.pine
from EternaHybridExchange/tradingview-strategies repo.
"""

import numpy as np
from backtesting import Strategy


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    """Calculate Average True Range (Wilder's smoothing)."""
    tr = np.maximum(high, np.roll(close, 1)) - np.minimum(low, np.roll(close, 1))
    tr[0] = np.nan
    result = np.full_like(close, np.nan)
    for i in range(length - 1, len(tr)):
        if i == length - 1:
            result[i] = np.nanmean(tr[1 : i + 1])
        else:
            result[i] = (result[i - 1] * (length - 1) + tr[i]) / length
    return result


class ChandelierExitStrategy(Strategy):
    """
    Chandelier Exit Strategy

    Entry logic:
    - Long: Price crosses above Chandelier Long (highest high - ATR×mult) + uptrend
    - Short: Price crosses below Chandelier Short (lowest low + ATR×mult) + downtrend

    Exit logic:
    - Long exit: Price crosses back below Chandelier Long
    - Short exit: Price crosses back above Chandelier Short

    Parameters:
        atr_length: ATR period (default 22)
        atr_multiplier: ATR multiplier (default 3.0)
        trend_ema: Trend filter EMA (default 50)
        use_trend_filter: Enable trend filter (default True)
    """

    atr_length = 22
    atr_multiplier = 3.0
    trend_ema = 50
    use_trend_filter = True

    def init(self) -> None:
        """Initialize Chandelier Exit indicators."""
        self.atr_value = self.I(
            lambda: atr(self.data.High, self.data.Low, self.data.Close, self.atr_length),
            name="ATR",
            color="gray",
        )
        self.chandelier_long = self.I(
            lambda: self._calc_chandelier_long(),
            name="Chandelier Long",
            color="green",
        )
        self.chandelier_short = self.I(
            lambda: self._calc_chandelier_short(),
            name="Chandelier Short",
            color="red",
        )
        if self.use_trend_filter:
            self.ema_filter = self.I(
                lambda: self.data.Close.s.ewm(span=self.trend_ema, adjust=False).mean().values,
                name=f"EMA{self.trend_ema}",
                color="orange",
            )

    def _calc_chandelier_long(self) -> np.ndarray:
        """Chandelier Long = highest high - ATR × multiplier."""
        highest_high = self.data.High.s.rolling(window=self.atr_length).max().values
        atr = self.atr_value
        result = np.full_like(highest_high, np.nan)
        for i in range(len(highest_high)):
            if not np.isnan(highest_high[i]) and not np.isnan(atr[i]):
                result[i] = highest_high[i] - atr[i] * self.atr_multiplier
        return result

    def _calc_chandelier_short(self) -> np.ndarray:
        """Chandelier Short = lowest low + ATR × multiplier."""
        lowest_low = self.data.Low.s.rolling(window=self.atr_length).min().values
        atr = self.atr_value
        result = np.full_like(lowest_low, np.nan)
        for i in range(len(lowest_low)):
            if not np.isnan(lowest_low[i]) and not np.isnan(atr[i]):
                result[i] = lowest_low[i] + atr[i] * self.atr_multiplier
        return result

    def next(self) -> None:
        """Execute strategy logic on each bar."""
        close = self.data.Close[-1]
        close_prev = self.data.Close[-2]
        cl_long = self.chandelier_long[-1]
        cl_short = self.chandelier_short[-1]
        cl_long_prev = self.chandelier_long[-2]
        cl_short_prev = self.chandelier_short[-2]

        # Trend filter
        if self.use_trend_filter:
            ema_val = self.ema_filter[-1]
            uptrend = close > ema_val
            downtrend = close < ema_val
        else:
            uptrend = True
            downtrend = True

        # Long entry: price crosses above Chandelier Long
        long_breakout = close > cl_long and close_prev <= cl_long_prev
        long_condition = uptrend and long_breakout

        # Short entry: price crosses below Chandelier Short
        short_breakdown = close < cl_short and close_prev >= cl_short_prev
        short_condition = downtrend and short_breakdown

        # Exit conditions
        exit_long = close < cl_long
        exit_short = close > cl_short

        # Execute trades
        if long_condition and not self.position:
            self.buy()

        if short_condition and not self.position:
            self.sell()

        if exit_long and self.position.is_long:
            self.position.close()

        if exit_short and self.position.is_short:
            self.position.close()
