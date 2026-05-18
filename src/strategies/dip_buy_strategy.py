"""
Defensive Dip Buy Strategy — SO (Southern Company / Utilities)

Utilities are range-bound yield vehicles. When price falls 2-sigma below its
50-day moving average, it's an overreaction — buy for mean reversion.
Trailing ATR stop exit.

Works on: SO, and other low-volatility defensive stocks (utilities, staples).

IS 2020-2024: Sharpe 0.56, Return +86.5%, 13 trades
OOS 2025-2026: Sharpe 1.14, Return +19.7%, 2 trades

Usage:
    from backtesting import Backtest
    bt = Backtest(df, DipBuyStrategy, cash=10_000, commission=0.001)
    stats = bt.run(ma_period=50, std_mult=2.0, trail_atr=4.0)
"""

from backtesting import Strategy
import pandas as pd


class DipBuyStrategy(Strategy):
    ma_period = 50
    std_mult = 2.0
    trail_atr = 4.0
    atr_period = 14

    def init(self):
        c = pd.Series(self.data.Close)
        ma = c.rolling(self.ma_period).mean()
        std = c.rolling(self.ma_period).std()
        self._lower = (ma - self.std_mult * std).values
        self._close = c.values

        hi = pd.Series(self.data.High)
        lo = pd.Series(self.data.Low)
        tr = pd.concat(
            [
                hi - lo,
                (hi - c.shift(1)).abs(),
                (lo - c.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        self._atr = tr.ewm(span=self.atr_period, adjust=False).mean().values

    def next(self):
        i = len(self.data) - 1
        if i < self.ma_period:
            return
        if self._close[i] < self._lower[i] and not self.position:
            self.buy()
        elif self.position and len(self.trades) > 0:
            entry_bar = self.trades[-1].entry_bar
            if i > entry_bar:
                atr_val = self._atr[i]
                highest = max(self._close[entry_bar : i + 1])
                if self._close[i] <= highest - self.trail_atr * atr_val:
                    self.position.close()
