"""
BB Squeeze Breakout Strategy — QQQ (NASDAQ 100)

Tech trends strongly after volatility contraction. When Bollinger Band width
is at its N-day low (squeeze) and price breaks above upper band, enter long.
Trailing ATR stop exit.

IS 2020-2024: Sharpe 0.45, Return +19.1%, 7 trades
OOS 2025-2026: No signals (squeeze threshold too strict — lower squeeze_pct for OOS)

Usage:
    from backtesting import Backtest
    bt = Backtest(df, BBSqueezeStrategy, cash=10_000, commission=0.001)
    stats = bt.run(bb_period=20, bb_std=2.0, squeeze_pct=20, trail_atr=4.0)
"""

from backtesting import Strategy
import pandas as pd


class BBSqueezeStrategy(Strategy):
    bb_period = 20
    bb_std = 2.0
    squeeze_pct = 20
    trail_atr = 4.0
    atr_period = 14

    def init(self):
        c = pd.Series(self.data.Close)
        sma = c.rolling(self.bb_period).mean()
        std = c.rolling(self.bb_period).std()
        upper = sma + self.bb_std * std
        lower = sma - self.bb_std * std
        width = (upper - lower) / sma
        self._width = width.values
        self._upper = upper.values
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
        if i < self.bb_period + 252:
            return

        width_window = self._width[max(0, i - 251) : i + 1]
        squeeze_thresh = pd.Series(width_window).quantile(self.squeeze_pct / 100)

        squeeze = self._width[i] < squeeze_thresh
        breakout = self._close[i] > self._upper[i]

        if squeeze and breakout and not self.position:
            self.buy()
        elif self.position and len(self.trades) > 0:
            entry_bar = self.trades[-1].entry_bar
            if i > entry_bar:
                atr_val = self._atr[i]
                highest = max(self._close[entry_bar : i + 1])
                if self._close[i] <= highest - self.trail_atr * atr_val:
                    self.position.close()
