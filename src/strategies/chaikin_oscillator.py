"""
Chaikin Oscillator Strategy for backtesting.py

Accumulation/Distribution momentum. EMA(3) - EMA(10) of A/D line.
"""

import numpy as np
from backtesting import Strategy


class ChaikinOscillatorStrategy(Strategy):
    """Chaikin Oscillator Strategy."""

    fast_length = 3
    slow_length = 10

    def init(self) -> None:
        """Initialize Chaikin Oscillator."""
        fl, sl = self.fast_length, self.slow_length

        def calc_chaikin():
            close_s = self.data.Close.s
            high_s = self.data.High.s
            low_s = self.data.Low.s
            vol_s = self.data.Volume.s
            mfm = ((close_s - low_s) - (high_s - close_s)) / (high_s - low_s).replace(0, np.nan)
            mfv = mfm * vol_s
            ad = mfv.cumsum()
            chaikin = ad.ewm(span=fl, adjust=False).mean() - ad.ewm(span=sl, adjust=False).mean()
            return chaikin.values

        self.chaikin = self.I(calc_chaikin, name="Chaikin", color="blue")

    def next(self) -> None:
        """Execute strategy logic."""
        c = self.chaikin[-1]
        c_prev = self.chaikin[-2]

        bullish = c > 0 and c_prev <= 0
        bearish = c < 0 and c_prev >= 0

        if bullish and not self.position:
            self.buy()

        if bearish and not self.position:
            self.sell()

        if bearish and self.position.is_long:
            self.position.close()

        if bullish and self.position.is_short:
            self.position.close()
