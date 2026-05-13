"""
SMA Crossover 50/200 Strategy for backtesting.py

Classic trend-following strategy using Simple Moving Average crossovers.
The Golden Cross (fast SMA crosses above slow SMA) signals bullish trends,
while the Death Cross (fast crosses below slow) signals bearish trends.

Pine Script source: strategies/trend-following/sma-crossover/strategy.pine
from EternaHybridExchange/tradingview-strategies repo.
"""

import pandas as pd
from backtesting import Strategy


class SMACrossoverStrategy(Strategy):
    """
    SMA Crossover 50/200 Strategy

    Entry logic:
    - Long: 50 SMA crosses above 200 SMA (Golden Cross)
    - Short: 50 SMA crosses below 200 SMA (Death Cross)

    Exit logic:
    - Long exit: Death Cross (50 SMA crosses below 200 SMA)
    - Short exit: Golden Cross (50 SMA crosses above 200 SMA)

    Parameters:
        fast_sma: Fast SMA period (default 50)
        slow_sma: Slow SMA period (default 200)
        risk_pct: Risk per trade as percentage of equity (0=use full equity)
        sl_atr_mult: Stop-loss distance in ATR multiples
    """

    fast_sma = 50
    slow_sma = 200
    risk_pct = 2.0
    sl_atr_mult = 1.5

    def init(self) -> None:
        self.sma_fast = self.I(
            lambda: self.data.Close.s.rolling(window=self.fast_sma).mean().values,
            name=f"SMA{self.fast_sma}",
            color="blue",
        )
        self.sma_slow = self.I(
            lambda: self.data.Close.s.rolling(window=self.slow_sma).mean().values,
            name=f"SMA{self.slow_sma}",
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
        fast = self.sma_fast[-1]
        slow = self.sma_slow[-1]
        fast_prev = self.sma_fast[-2]
        slow_prev = self.sma_slow[-2]

        golden_cross = fast > slow and fast_prev <= slow_prev
        death_cross = fast < slow and fast_prev >= slow_prev

        if self.position:
            if death_cross and self.position.is_long:
                self.position.close()
            if golden_cross and self.position.is_short:
                self.position.close()
            return

        if golden_cross:
            self.buy(size=self._position_size())
        elif death_cross:
            self.sell(size=self._position_size())
