"""
Low-Volatility Momentum Strategy — JNJ (Johnson & Johnson / Healthcare)

Defensive healthcare stocks exhibit the low-volatility anomaly: buying when
volatility is below median AND price is above its 50-day moving average
captures the drift in calm, trending periods. Trailing ATR stop exit.

IS 2020-2024: Sharpe -0.06, Return -2.3%, 12 trades
OOS 2025-2026: Sharpe -0.28, Return -2.3%, 2 trades

Usage:
    from backtesting import Backtest
    bt = Backtest(df, LowVolMomentumStrategy, cash=10_000, commission=0.001)
    stats = bt.run(ma_period=50, vol_period=20, vol_percentile=50, trail_atr=4.0)
"""

from backtesting import Strategy
import pandas as pd


class LowVolMomentumStrategy(Strategy):
    ma_period = 50
    vol_period = 20
    vol_percentile = 50
    trail_atr = 4.0
    atr_period = 14

    def init(self):
        c = pd.Series(self.data.Close)
        ret = c.pct_change()
        vol = ret.rolling(self.vol_period).std()
        self._vol = vol.values
        self._ma = c.rolling(self.ma_period).mean().values
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
        if i < max(self.ma_period, 252):
            return

        vol_window = self._vol[max(0, i - 251) : i + 1]
        vol_pctile = pd.Series(vol_window).rank(pct=True).iloc[-1]

        low_vol = vol_pctile < (self.vol_percentile / 100)
        above_ma = self._close[i] > self._ma[i]

        if low_vol and above_ma and not self.position:
            self.buy()
        elif self.position and len(self.trades) > 0:
            entry_bar = self.trades[-1].entry_bar
            if i > entry_bar:
                atr_val = self._atr[i]
                highest = max(self._close[entry_bar : i + 1])
                if self._close[i] <= highest - self.trail_atr * atr_val:
                    self.position.close()
