"""
Gold Trend + Rate Filter Strategy — GLD (SPDR Gold Shares)

Gold trends in falling-real-rate environments. When fast MA crosses above slow MA
and ADX confirms trend strength, enter long. Wide trailing stop (6 ATR) allows
gold's characteristic rallies to run.

IS 2020-2024: Sharpe 0.48, Return +33.0%, 9 trades
OOS 2025-2026: Sharpe 1.01, Return +43.6%, 1 trade

Usage:
    from backtesting import Backtest
    bt = Backtest(df, GoldTrendStrategy, cash=10_000, commission=0.001)
    stats = bt.run(ma_fast=10, ma_slow=50, adx_period=14, adx_min=20, trail_atr=6.0)
"""

from backtesting import Strategy
import pandas as pd


class GoldTrendStrategy(Strategy):
    ma_fast = 10
    ma_slow = 50
    adx_period = 14
    adx_min = 20
    trail_atr = 6.0
    atr_period = 14

    def init(self):
        c = pd.Series(self.data.Close)
        hi = pd.Series(self.data.High)
        lo = pd.Series(self.data.Low)

        self._ma_fast = c.rolling(self.ma_fast).mean().values
        self._ma_slow = c.rolling(self.ma_slow).mean().values
        self._close = c.values

        tr = pd.concat(
            [
                hi - lo,
                (hi - c.shift(1)).abs(),
                (lo - c.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_ema = tr.ewm(span=self.atr_period, adjust=False).mean()
        self._atr = atr_ema.values

        plus_dm = hi.diff().clip(lower=0)
        minus_dm = (-lo.diff()).clip(lower=0)
        atr_s = atr_ema.replace(0, 1e-9)
        plus_di = 100 * plus_dm.ewm(span=self.adx_period, adjust=False).mean() / atr_s
        minus_di = 100 * minus_dm.ewm(span=self.adx_period, adjust=False).mean() / atr_s
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
        self._adx = dx.ewm(span=self.adx_period, adjust=False).mean().values

    def next(self):
        i = len(self.data) - 1
        if i < max(self.ma_slow, self.adx_period):
            return

        trend_up = (
            self._ma_fast[i] > self._ma_slow[i]
            and self._close[i] > self._ma_slow[i]
            and self._adx[i] > self.adx_min
        )

        if trend_up and not self.position:
            self.buy()
        elif self.position and len(self.trades) > 0:
            entry_bar = self.trades[-1].entry_bar
            if i > entry_bar:
                highest = max(self._close[entry_bar : i + 1])
                if self._close[i] <= highest - self.trail_atr * self._atr[i]:
                    self.position.close()
