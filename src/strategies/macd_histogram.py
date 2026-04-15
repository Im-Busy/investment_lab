"""
MACD Histogram Strategy for backtesting.py

Trades MACD histogram direction changes and zero-line crosses.
"""

import numpy as np
from backtesting import Strategy


class MACDHistogramStrategy(Strategy):
    """MACD Histogram Strategy - trades histogram sign changes."""

    fast = 12
    slow = 26
    signal = 9

    def init(self) -> None:
        """Initialize MACD indicators."""
        fast, slow, sig = self.fast, self.slow, self.signal

        def calc_macd():
            ema_fast = self.data.Close.s.ewm(span=fast, adjust=False).mean()
            ema_slow = self.data.Close.s.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=sig, adjust=False).mean()
            histogram = macd - signal
            return histogram.values

        self.histogram = self.I(calc_macd, name="MACD Histogram", color="orange")

    def next(self) -> None:
        """Execute strategy logic."""
        hist = self.histogram[-1]
        hist_prev = self.histogram[-2]

        bullish = hist_prev < 0 and hist > hist_prev
        bearish = hist_prev > 0 and hist < hist_prev

        if bullish and not self.position:
            self.buy()

        if bearish and not self.position:
            self.sell()

        if bearish and self.position.is_long:
            self.position.close()

        if bullish and self.position.is_short:
            self.position.close()
