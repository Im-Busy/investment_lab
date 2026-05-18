"""
Crypto Momentum + Volume Strategy — BTC_USD / ETH_USD

Cryptocurrencies trend on high-volume breakouts. When price is above 20MA,
volume exceeds 1.5x 20-day average, and daily return exceeds minimum, enter long.
Wide trailing stop (10 ATR) accommodates crypto's extreme volatility.

Works on: BTC_USD, ETH_USD, and other trending crypto assets.

BTC IS 2020-2024: Sharpe 0.67, Return +949%, 7 trades (cash=100k)
BTC OOS 2025-2026: Sharpe -0.67, Return -18.4%, 2 trades (crypto drawdown)
ETH IS 2020-2024: Sharpe 0.41, Return +1142%, 12 trades (cash=50k)
ETH OOS 2025-2026: Sharpe -0.67, Return -43.3%, 4 trades

Usage:
    from backtesting import Backtest
    bt = Backtest(df, CryptoMomentumStrategy, cash=100_000, commission=0.001)
    stats = bt.run(ma_period=20, vol_mult=1.5, return_min=0.02, trail_atr=10.0)
"""

from backtesting import Strategy
import pandas as pd


class CryptoMomentumStrategy(Strategy):
    ma_period = 20
    vol_mult = 1.5
    return_min = 0.02
    trail_atr = 10.0
    atr_period = 14

    def init(self):
        c = pd.Series(self.data.Close)
        hi = pd.Series(self.data.High)
        lo = pd.Series(self.data.Low)
        v = pd.Series(self.data.Volume)

        self._ma = c.rolling(self.ma_period).mean().values
        self._vol_ma = v.rolling(20).mean().values
        self._ret = c.pct_change().values
        self._close = c.values
        self._volume = v.values

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
        if i < max(self.ma_period, 20):
            return
        if i >= len(self._ret) or self._ret[i] is None:
            return

        above_ma = self._close[i] > self._ma[i]
        vol_surge = self._volume[i] > self._vol_ma[i] * self.vol_mult
        momentum = bool(self._ret[i] > self.return_min)

        if above_ma and vol_surge and momentum and not self.position:
            self.buy()
        elif self.position and len(self.trades) > 0:
            entry_bar = self.trades[-1].entry_bar
            if i > entry_bar:
                highest = max(self._close[entry_bar : i + 1])
                if self._close[i] <= highest - self.trail_atr * self._atr[i]:
                    self.position.close()
