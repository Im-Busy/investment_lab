"""
RSI Oversold Bounce Strategy — IWM (Russell 2000 Small Cap)

Small caps overreact to selling pressure. RSI(2) < oversold threshold signals
capitulation — buy for a bounce. Trailing ATR stop exit.

Works on: IWM, and other volatile mean-reverting instruments.

IS 2020-2024: Sharpe 0.10, Return +13.8%, 28 trades
OOS 2025-2026: Sharpe 0.32, Return +9.9%, 8 trades

Usage:
    from backtesting import Backtest
    bt = Backtest(df, RSIOversoldStrategy, cash=10_000, commission=0.001)
    stats = bt.run(rsi_period=2, oversold=15, trail_atr=4.0)
"""

from backtesting import Strategy
import pandas as pd


class RSIOversoldStrategy(Strategy):
    rsi_period = 2
    oversold = 15
    trail_atr = 4.0
    atr_period = 14

    def init(self):
        c = pd.Series(self.data.Close)
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        self._rsi = (100 - (100 / (1 + rs))).values

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
        self._close = c.values

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
