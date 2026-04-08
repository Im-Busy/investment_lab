"""
CCI Strategy for backtesting.py

Commodity Channel Index - trades threshold crosses and extreme reversals.
"""

import numpy as np
from backtesting import Strategy


class CCIStrategy(Strategy):
    """CCI Strategy - threshold crosses and extreme reversals."""

    cci_length = 20
    overbought = 100
    oversold = -100

    def init(self) -> None:
        """Initialize CCI indicator."""
        length = self.cci_length

        def calc_cci():
            tp = (self.data.High.s + self.data.Low.s + self.data.Close.s) / 3
            sma = tp.rolling(window=length).mean()
            mad = tp.rolling(window=length).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            cci = (tp - sma) / (0.015 * mad)
            return cci.values

        self.cci = self.I(calc_cci, name="CCI", color="blue")

    def next(self) -> None:
        """Execute strategy logic."""
        cci = self.cci[-1]
        cci_prev = self.cci[-2]

        bullish = cci > self.oversold and cci_prev <= self.oversold
        bearish = cci < self.overbought and cci_prev >= self.overbought

        if bullish and not self.position:
            self.buy()

        if bearish and not self.position:
            self.sell()

        if bearish and self.position.is_long:
            self.position.close()

        if bullish and self.position.is_short:
            self.position.close()
