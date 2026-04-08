"""
RSI Divergence Strategy for backtesting.py

Detects bullish/bearish divergences between price and RSI.
Bullish divergence: price makes lower low, RSI makes higher low.
Bearish divergence: price makes higher high, RSI makes lower high.

Pine Script source: strategies/momentum/rsi_divergence.pine
from EternaHybridExchange/tradingview-strategies repo.
"""

import numpy as np
from backtesting import Strategy


def calc_pivot_low(data: np.ndarray, left: int, right: int) -> np.ndarray:
    """Find pivot lows."""
    result = np.full_like(data, np.nan)
    for i in range(left, len(data) - right):
        window = data[i - left : i + right + 1]
        if data[i] == np.min(window) and np.sum(window == data[i]) == 1:
            result[i + right] = data[i]
    return result


def calc_pivot_high(data: np.ndarray, left: int, right: int) -> np.ndarray:
    """Find pivot highs."""
    result = np.full_like(data, np.nan)
    for i in range(left, len(data) - right):
        window = data[i - left : i + right + 1]
        if data[i] == np.max(window) and np.sum(window == data[i]) == 1:
            result[i + right] = data[i]
    return result


class RSIDivergenceStrategy(Strategy):
    """
    RSI Divergence Strategy

    Entry logic:
    - Long: Bullish divergence (price lower low, RSI higher low) + RSI < 30
    - Short: Bearish divergence (price higher high, RSI lower high) + RSI > 70

    Exit logic:
    - Long exit: Bearish divergence detected
    - Short exit: Bullish divergence detected

    Parameters:
        rsi_length: RSI period (default 14)
        rsi_overbought: RSI overbought level (default 70)
        rsi_oversold: RSI oversold level (default 30)
        pivot_lookback: Pivot detection lookback (default 5)
    """

    rsi_length = 14
    rsi_overbought = 70
    rsi_oversold = 30
    pivot_lookback = 5

    def init(self) -> None:
        """Initialize RSI indicator."""
        self.rsi = self.I(
            lambda: (
                self.data.Close.s.rolling(window=self.rsi_length)
                .apply(
                    lambda x: 100 - 100 / (1 + (x[x > 0].sum() / max(1, -x[x < 0].sum()))), raw=True
                )
                .values
            ),
            name="RSI",
            color="blue",
        )

    def next(self) -> None:
        """Execute strategy logic on each bar."""
        # Simplified divergence detection using rolling windows
        n = max(20, self.pivot_lookback * 4)
        if len(self.data) < n:
            return

        rsi_vals = self.rsi[-n:]
        low_vals = self.data.Low.s.values[-n:]
        high_vals = self.data.High.s.values[-n:]
        rsi_current = rsi_vals[-1]

        # Find recent lows and highs
        rsi_min = np.nanmin(rsi_vals[:-1])
        rsi_max = np.nanmax(rsi_vals[:-1])
        low_min = np.nanmin(low_vals[:-1])
        high_max = np.nanmax(high_vals[:-1])

        # Simplified divergence: RSI higher while price lower
        recent_rsi_low = np.nanmin(rsi_vals[-self.pivot_lookback * 2 :])
        recent_low = np.nanmin(low_vals[-self.pivot_lookback * 2 :])
        prev_rsi_low = np.nanmin(rsi_vals[-self.pivot_lookback * 4 : -self.pivot_lookback * 2])
        prev_low = np.nanmin(low_vals[-self.pivot_lookback * 4 : -self.pivot_lookback * 2])

        # Bullish divergence: price making lower low, RSI making higher low
        bullish_div = (
            recent_low < prev_low
            and recent_rsi_low > prev_rsi_low
            and rsi_current < self.rsi_oversold
        )

        # Bearish divergence: price making higher high, RSI making lower high
        recent_rsi_high = np.nanmax(rsi_vals[-self.pivot_lookback * 2 :])
        recent_high = np.nanmax(high_vals[-self.pivot_lookback * 2 :])
        prev_rsi_high = np.nanmax(rsi_vals[-self.pivot_lookback * 4 : -self.pivot_lookback * 2])
        prev_high = np.nanmax(high_vals[-self.pivot_lookback * 4 : -self.pivot_lookback * 2])

        bearish_div = (
            recent_high > prev_high
            and recent_rsi_high < prev_rsi_high
            and rsi_current > self.rsi_overbought
        )

        # Execute trades
        if bullish_div and not self.position:
            self.buy()

        if bearish_div and not self.position:
            self.sell()

        if bearish_div and self.position.is_long:
            self.position.close()

        if bullish_div and self.position.is_short:
            self.position.close()
