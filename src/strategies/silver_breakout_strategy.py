"""
Silver Vol Expansion Breakout Strategy — SLV (iShares Silver Trust)

Silver moves in violent bursts after volatility contraction. When ATR expands
near its 100-day maximum AND price is above 20MA, enter long. Wide trailing
stop (8 ATR) accommodates silver's extreme swings.

IS 2020-2024: Sharpe -0.11, Return -11.9%, 8 trades
OOS 2025-2026: Sharpe 0.82, Return +92.6%, 1 trade

Usage:
    from backtesting import Backtest
    bt = Backtest(df, SilverBreakoutStrategy, cash=10_000, commission=0.001)
    stats = bt.run(atr_period=14, ma_period=20, atr_max_lookback=100, trail_atr=8.0)
"""

from backtesting import Strategy
import pandas as pd


class SilverBreakoutStrategy(Strategy):
    atr_period = 14
    ma_period = 20
    atr_max_lookback = 100
    trail_atr = 8.0

    def init(self):
        c = pd.Series(self.data.Close)
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
        atr_vals = tr.ewm(span=self.atr_period, adjust=False).mean()
        self._atr = atr_vals.values
        self._ma = c.rolling(self.ma_period).mean().values
        self._close = c.values

    def next(self):
        i = len(self.data) - 1
        if i < max(self.atr_max_lookback, self.ma_period):
            return

        atr_window = self._atr[max(0, i - self.atr_max_lookback) : i + 1]
        atr_max = max(atr_window) if len(atr_window) > 0 else 0.01
        vol_expanding = self._atr[i] > atr_max * 0.85
        above_ma = self._close[i] > self._ma[i]

        if vol_expanding and above_ma and not self.position:
            self.buy()
        elif self.position and len(self.trades) > 0:
            entry_bar = self.trades[-1].entry_bar
            if i > entry_bar:
                highest = max(self._close[entry_bar : i + 1])
                if self._close[i] <= highest - self.trail_atr * self._atr[i]:
                    self.position.close()
