"""
Bond Rate Spike Fade Strategy — TLT (20+ Year Treasury Bond ETF)

Bond selloffs (yield spikes) are frequently overdone. When TLT falls 1.5-sigma
below its 50MA AND RSI(14) is below 30, it signals capitulation selling — buy
for the bounce. Trailing ATR stop exit.

IS 2020-2024: Sharpe -0.56, Return -30.8%, 24 trades (brutal rate-hiking cycle)
OOS 2025-2026: Sharpe 0.44, Return +5.4%, 4 trades (rates stabilized)

Usage:
    from backtesting import Backtest
    bt = Backtest(df, BondFadeStrategy, cash=10_000, commission=0.001)
    stats = bt.run(ma_period=50, std_mult=1.5, rsi_period=14, rsi_oversold=30, trail_atr=4.0)
"""

from backtesting import Strategy
import pandas as pd


class BondFadeStrategy(Strategy):
    ma_period = 50
    std_mult = 1.5
    rsi_period = 14
    rsi_oversold = 30
    trail_atr = 4.0
    atr_period = 14

    def init(self):
        c = pd.Series(self.data.Close)
        hi = pd.Series(self.data.High)
        lo = pd.Series(self.data.Low)

        ma = c.rolling(self.ma_period).mean()
        std = c.rolling(self.ma_period).std()
        self._lower = (ma - self.std_mult * std).values
        self._close = c.values

        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        self._rsi = (100 - (100 / (1 + rs))).values

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
        if i < max(self.ma_period, self.rsi_period):
            return

        capitulation = self._close[i] < self._lower[i] and self._rsi[i] < self.rsi_oversold

        if capitulation and not self.position:
            self.buy()
        elif self.position and len(self.trades) > 0:
            entry_bar = self.trades[-1].entry_bar
            if i > entry_bar:
                highest = max(self._close[entry_bar : i + 1])
                if self._close[i] <= highest - self.trail_atr * self._atr[i]:
                    self.position.close()
