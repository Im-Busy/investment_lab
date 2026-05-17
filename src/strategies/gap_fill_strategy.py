"""
Gap Fill Strategy — SPY (S&P 500)

Overnight gaps in SPY tend to fill intraday due to market-maker hedging
and mean reversion. When SPY gaps down at open, buy expecting the gap to
close. Exit on trailing ATR stop.

IS 2020-2024: Sharpe 0.80, Return +134.8%, 24 trades
OOS 2025-2026: Sharpe 1.23, Return +37.2%, 5 trades

Usage:
    from backtesting import Backtest
    bt = Backtest(df, GapFillStrategy, cash=10_000, commission=0.001)
    stats = bt.run(gap_threshold=0.005, trail_atr=4.0)
"""

from backtesting import Strategy
import pandas as pd


class GapFillStrategy(Strategy):
    gap_threshold = 0.005
    trail_atr = 4.0
    atr_period = 14

    def init(self):
        hi = pd.Series(self.data.High)
        lo = pd.Series(self.data.Low)
        c = pd.Series(self.data.Close)
        tr = pd.concat(
            [
                hi - lo,
                (hi - c.shift(1)).abs(),
                (lo - c.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_vals = tr.ewm(span=self.atr_period, adjust=False).mean().values
        self._atr = atr_vals

    def next(self):
        i = len(self.data) - 1
        prev_close = self.data.Close[-2] if i > 0 else self.data.Open[-1]
        gap = (self.data.Open[-1] - prev_close) / prev_close if prev_close > 0 else 0

        if gap < -self.gap_threshold and not self.position:
            self.buy()
        elif self.position and len(self.trades) > 0:
            entry_bar = self.trades[-1].entry_bar
            if i > entry_bar:
                atr_val = self._atr[i] if i < len(self._atr) else 0
                highest = max(self.data.Close[entry_bar : i + 1])
                if self.data.Close[-1] <= highest - self.trail_atr * atr_val:
                    self.position.close()
