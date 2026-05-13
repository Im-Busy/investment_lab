"""
EMA Ribbon (9/21/55) Strategy for backtesting.py

Multi-EMA trend-following system that uses the alignment of three exponential
moving averages. Entries are taken when price bounces off the fast EMA while
all three EMAs are properly aligned, indicating strong trend momentum.

Pine Script source: strategies/trend-following/ema-ribbon/strategy.pine
from EternaHybridExchange/tradingview-strategies repo.
"""

import pandas as pd
from backtesting import Strategy


class EMARibbonStrategy(Strategy):
    """
    EMA Ribbon 9/21/55 Strategy

    Entry logic:
    - Long: EMA9 > EMA21 > EMA55 (bullish alignment) AND price crosses above EMA9
    - Short: EMA9 < EMA21 < EMA55 (bearish alignment) AND price crosses below EMA9

    Exit logic:
    - Long exit: EMA9 crosses below EMA21 (alignment breaks)
    - Short exit: EMA9 crosses above EMA21 (alignment breaks)

    Parameters:
        ema_fast: Fast EMA period (default 9)
        ema_mid: Mid EMA period (default 21)
        ema_slow: Slow EMA period (default 55)
        risk_pct: Risk per trade as percentage of equity (0=use full equity)
        sl_atr_mult: Stop-loss distance in ATR multiples
    """

    ema_fast = 9
    ema_mid = 21
    ema_slow = 55
    risk_pct = 2.0
    sl_atr_mult = 1.5

    def init(self) -> None:
        self.ema9 = self.I(
            lambda: self.data.Close.s.ewm(span=self.ema_fast, adjust=False).mean().values,
            name=f"EMA{self.ema_fast}",
            color="green",
        )
        self.ema21 = self.I(
            lambda: self.data.Close.s.ewm(span=self.ema_mid, adjust=False).mean().values,
            name=f"EMA{self.ema_mid}",
            color="orange",
        )
        self.ema55 = self.I(
            lambda: self.data.Close.s.ewm(span=self.ema_slow, adjust=False).mean().values,
            name=f"EMA{self.ema_slow}",
            color="red",
        )

        high = self.data.df.High
        low = self.data.df.Low
        close = self.data.df.Close
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.atr14 = self.I(lambda: tr.rolling(14).mean().values, name="ATR(14)")

    def _position_size(self) -> float:
        if self.risk_pct <= 0:
            return 1.0
        risk_amount = self.equity * (self.risk_pct / 100)
        stop_distance = self.atr14[-1] * self.sl_atr_mult
        if stop_distance <= 0:
            return 1.0
        size = risk_amount / stop_distance
        return max(1, int(size))

    def next(self) -> None:
        ema9 = self.ema9[-1]
        ema21 = self.ema21[-1]
        ema9_prev = self.ema9[-2]
        ema21_prev = self.ema21[-2]
        close = self.data.Close[-1]
        close_prev = self.data.Close[-2]

        if self.position:
            exit_long = ema9 < ema21 and ema9_prev >= ema21_prev
            exit_short = ema9 > ema21 and ema9_prev <= ema21_prev
            if exit_long and self.position.is_long:
                self.position.close()
            if exit_short and self.position.is_short:
                self.position.close()
            return

        ema55 = self.ema55[-1]
        bullish_alignment = ema9 > ema21 and ema21 > ema55
        bearish_alignment = ema9 < ema21 and ema21 < ema55

        long_cross = close > ema9 and close_prev <= ema9_prev
        short_cross = close < ema9 and close_prev >= ema9_prev

        if bullish_alignment and long_cross:
            self.buy(size=self._position_size())
        elif bearish_alignment and short_cross:
            self.sell(size=self._position_size())
