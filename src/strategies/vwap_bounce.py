"""
VWAP Bounce Strategy for backtesting.py

Institutional-grade mean reversion within a trend. Uses Volume Weighted Average Price
(VWAP) as dynamic support/resistance. Enters when price bounces off VWAP bands in the
direction of the main trend (determined by 50 EMA).

Pine Script source: strategies/trend-following/vwap-bounce/strategy.pine
from EternaHybridExchange/tradingview-strategies repo.
"""

import pandas as pd
from backtesting import Strategy

from src.indicators.vwap import compute_vwap


class VWAPBounceStrategy(Strategy):
    """
    VWAP Bounce Strategy

    Entry logic:
    - Long: Price above 50 EMA + bounces from VWAP lower band
      (previous bar low <= vwap_lower, current bar close crosses above vwap_lower)
    - Short: Price below 50 EMA + rejects from VWAP upper band
      (previous bar high >= vwap_upper, current bar close crosses below vwap_upper)

    Exit logic:
    - Long exit: Close crosses below center VWAP line
    - Short exit: Close crosses above center VWAP line

    Parameters:
        vwap_deviation_pct: Band width as percentage of VWAP (default 0.5%)
        trend_ema: EMA period for trend filter (default 50)
        risk_pct: Risk per trade as percentage of equity (0=use full equity)
        sl_atr_mult: Stop-loss distance in ATR multiples
    """

    vwap_deviation_pct = 0.5
    trend_ema = 50
    risk_pct = 2.0
    sl_atr_mult = 1.5

    def init(self) -> None:
        vwap_df = compute_vwap(self.data.df, deviation_pct=self.vwap_deviation_pct)
        self.vwap = self.I(lambda: vwap_df["vwap"].values, name="VWAP", color="blue")
        self.vwap_upper = self.I(
            lambda: vwap_df["vwap_upper"].values, name="VWAP Upper", color="lightblue"
        )
        self.vwap_lower = self.I(
            lambda: vwap_df["vwap_lower"].values, name="VWAP Lower", color="lightblue"
        )

        self.ema_trend = self.I(
            lambda: self.data.Close.s.ewm(span=self.trend_ema, adjust=False).mean().values,
            name=f"EMA{self.trend_ema}",
            color="orange",
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
            return 1.0  # full position
        risk_amount = self.equity * (self.risk_pct / 100)
        stop_distance = self.atr14[-1] * self.sl_atr_mult
        if stop_distance <= 0:
            return 1.0
        size = risk_amount / stop_distance
        return max(1, int(size))

    def next(self) -> None:
        if self.position:
            close = self.data.Close[-1]
            vwap_val = self.vwap[-1]
            exit_long = close < vwap_val
            exit_short = close > vwap_val
            if exit_long and self.position.is_long:
                self.position.close()
            if exit_short and self.position.is_short:
                self.position.close()
            return

        close = self.data.Close[-1]
        close_prev = self.data.Close[-2]
        low = self.data.Low[-1]
        high = self.data.High[-1]

        vwap_val = self.vwap[-1]
        vwap_lower_val = self.vwap_lower[-1]
        vwap_lower_prev = self.vwap_lower[-2]
        vwap_upper_val = self.vwap_upper[-1]
        vwap_upper_prev = self.vwap_upper[-2]
        ema_val = self.ema_trend[-1]

        uptrend = close > ema_val
        downtrend = close < ema_val

        long_crossover = close > vwap_lower_val and close_prev <= vwap_lower_prev
        long_condition = uptrend and long_crossover and low <= vwap_lower_val

        short_crossover = close < vwap_upper_val and close_prev >= vwap_upper_prev
        short_condition = downtrend and short_crossover and high >= vwap_upper_val

        if long_condition:
            self.buy(size=self._position_size())
        elif short_condition:
            self.sell(size=self._position_size())
