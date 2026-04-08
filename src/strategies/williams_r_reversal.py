"""
Williams %R Reversal Strategy for backtesting.py

Overbought/oversold reversals using Williams Percent Range.
"""

from backtesting import Strategy


class WilliamsRReversalStrategy(Strategy):
    """Williams %R Reversal Strategy."""

    wr_length = 14
    overbought = -20
    oversold = -80

    def init(self) -> None:
        """Initialize Williams %R indicator."""
        length = self.wr_length

        def calc_wr():
            highest = self.data.High.s.rolling(window=length).max()
            lowest = self.data.Low.s.rolling(window=length).min()
            wr = -100 * (highest - self.data.Close.s) / (highest - lowest)
            return wr.values

        self.wr = self.I(calc_wr, name="Williams %R", color="blue")

    def next(self) -> None:
        """Execute strategy logic."""
        wr = self.wr[-1]
        wr_prev = self.wr[-2]

        bullish = wr > self.oversold and wr_prev <= self.oversold
        bearish = wr < self.overbought and wr_prev >= self.overbought

        if bullish and not self.position:
            self.buy()

        if bearish and not self.position:
            self.sell()

        if bearish and self.position.is_long:
            self.position.close()

        if bullish and self.position.is_short:
            self.position.close()
