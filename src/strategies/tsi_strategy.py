"""
TSI Strategy for backtesting.py

True Strength Index - double-smoothed momentum oscillator.
"""

import numpy as np
from backtesting import Strategy


class TSIStrategy(Strategy):
    """TSI Strategy - double-smoothed momentum crossovers."""

    long_length = 25
    short_length = 13
    signal_length = 13
    overbought = 25
    oversold = -25

    def init(self) -> None:
        """Initialize TSI indicator."""
        ll = self.long_length
        sl = self.short_length
        sig = self.signal_length

        def calc_tsi():
            closes = self.data.Close.s
            pc = closes.diff()
            abs_pc = pc.abs()

            def double_ema(series, l1, l2):
                return series.ewm(span=l1, adjust=False).mean().ewm(span=l2, adjust=False).mean()

            dspc = double_ema(pc, ll, sl)
            dsapc = double_ema(abs_pc, ll, sl)
            tsi = 100 * dspc / dsapc.replace(0, np.nan)
            signal = tsi.ewm(span=sig, adjust=False).mean()
            return tsi.values, signal.values

        self.tsi_vals, self.signal_vals = calc_tsi()
        self.tsi = self.I(lambda: self.tsi_vals, name="TSI", color="blue")
        self.signal = self.I(lambda: self.signal_vals, name="Signal", color="orange")

    def next(self) -> None:
        """Execute strategy logic."""
        tsi = self.tsi[-1]
        sig = self.signal[-1]
        tsi_prev = self.tsi[-2]
        sig_prev = self.signal[-2]

        bullish_cross = tsi > sig and tsi_prev <= sig_prev
        bearish_cross = tsi < sig and tsi_prev >= sig_prev

        buy_signal = (bullish_cross and tsi < 0) or (
            tsi < self.oversold and tsi > tsi_prev and bullish_cross
        )
        sell_signal = (bearish_cross and tsi > 0) or (
            tsi > self.overbought and tsi < tsi_prev and bearish_cross
        )

        if buy_signal and not self.position:
            self.buy()

        if sell_signal and not self.position:
            self.sell()

        if sell_signal and self.position.is_long:
            self.position.close()

        if buy_signal and self.position.is_short:
            self.position.close()
