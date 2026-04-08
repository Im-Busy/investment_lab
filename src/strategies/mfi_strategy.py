"""
MFI Strategy for backtesting.py

Money Flow Index - volume-weighted RSI with oversold/overbought signals.
"""

import numpy as np
from backtesting import Strategy


class MFIStrategy(Strategy):
    """MFI Strategy - oversold/overbought with volume confirmation."""

    mfi_length = 14
    overbought = 80
    oversold = 20

    def init(self) -> None:
        """Initialize MFI indicator."""
        length = self.mfi_length

        def calc_mfi():
            tp = (self.data.High.s + self.data.Low.s + self.data.Close.s) / 3
            raw_mf = tp * self.data.Volume.s
            tp_shifted = tp.shift(1)
            direction = np.where(tp > tp_shifted, 1, np.where(tp < tp_shifted, -1, 0))
            pmf = np.where(direction > 0, raw_mf, 0)
            nmf = np.where(direction < 0, raw_mf, 0)
            pmf_sum = np.convolve(pmf, np.ones(length), mode="valid")
            nmf_sum = np.convolve(nmf, np.ones(length), mode="valid")
            mfi = np.full(len(tp), 50.0)
            for i in range(len(pmf_sum)):
                if nmf_sum[i] == 0:
                    mfi[i + length - 1] = 100
                else:
                    mfi[i + length - 1] = 100 - 100 / (1 + pmf_sum[i] / nmf_sum[i])
            return mfi

        self.mfi = self.I(calc_mfi, name="MFI", color="blue")

    def next(self) -> None:
        """Execute strategy logic."""
        mfi = self.mfi[-1]
        mfi_prev = self.mfi[-2]

        bullish = mfi > self.oversold and mfi_prev <= self.oversold
        bearish = mfi < self.overbought and mfi_prev >= self.overbought

        if bullish and not self.position:
            self.buy()

        if bearish and not self.position:
            self.sell()

        if bearish and self.position.is_long:
            self.position.close()

        if bullish and self.position.is_short:
            self.position.close()
