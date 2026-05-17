"""
Sector Oversold Reversal Strategy — XLV (Healthcare ETF)

Healthcare ETFs mean-revert when oversold — investors rotate back to defensive
sectors. RSI(14) below 30 triggers entry. Trailing ATR stop exit.

IS 2020-2024: Sharpe 0.76, Return +90.8%, 28 trades
OOS 2025-2026: Sharpe -0.47, Return -7.6%, 5 trades

Usage:
    from backtesting import Backtest
    bt = Backtest(df, SectorOversoldStrategy, cash=10_000, commission=0.001)
    stats = bt.run(rsi_period=14, oversold=30, trail_atr=4.0)
"""

from backtesting import Strategy
import pandas as pd


class SectorOversoldStrategy(Strategy):
    rsi_period = 14
    oversold = 30
    trail_atr = 4.0
    atr_period = 14

    def init(self):
        c = pd.Series(self.data.Close)
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        self._rsi = (100 - (100 / (1 + rs))).values
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
        if i < self.rsi_period:
            return
        if self._rsi[i] < self.oversold and not self.position:
            self.buy()
        elif self.position and len(self.trades) > 0:
            entry_bar = self.trades[-1].entry_bar
            if i > entry_bar:
                atr_val = self._atr[i]
                highest = max(self._close[entry_bar : i + 1])
                if self._close[i] <= highest - self.trail_atr * atr_val:
                    self.position.close()
