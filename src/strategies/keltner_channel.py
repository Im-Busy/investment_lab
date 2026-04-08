"""
Keltner Channel Trend Strategy for backtesting.py

Volatility-based channel system using EMA and Average True Range (ATR).
Identifies trending markets and enters on pullbacks to the channel edges.

Pine Script source: strategies/trend-following/keltner-channel/strategy.pine
from EternaHybridExchange/tradingview-strategies repo.
"""

import numpy as np
from backtesting import Strategy


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    """Calculate Average True Range."""
    tr = np.maximum(high, np.roll(close, 1)) - np.minimum(low, np.roll(close, 1))
    tr[0] = np.nan
    result = np.full_like(close, np.nan)
    # Use simple moving average of TR like ta.atr in TradingView
    for i in range(length - 1, len(tr)):
        if i == length - 1:
            result[i] = np.nanmean(tr[1 : i + 1])  # skip first NaN
        else:
            result[i] = (result[i - 1] * (length - 1) + tr[i]) / length
    return result


class KeltnerChannelStrategy(Strategy):
    """
    Keltner Channel Trend Strategy

    Entry logic:
    - Long: Price above 50 EMA + bounces off lower Keltner band
    - Short: Price below 50 EMA + rejects from upper Keltner band

    Exit logic:
    - Long exit: Close crosses below basis (middle line)
    - Short exit: Close crosses above basis (middle line)

    Parameters:
        ema_length: Basis EMA period (default 20)
        atr_length: ATR period (default 10)
        atr_multiplier: ATR multiplier for bands (default 2.0)
        trend_ema: Trend filter EMA (default 50)
    """

    ema_length = 20
    atr_length = 10
    atr_multiplier = 2.0
    trend_ema = 50

    def init(self) -> None:
        """Initialize Keltner Channel indicators."""
        # Basis EMA
        self.basis = self.I(
            lambda: self.data.Close.s.ewm(span=self.ema_length, adjust=False).mean().values,
            name="Basis",
            color="blue",
        )
        # ATR
        self.atr_value = self.I(
            lambda: atr(self.data.High, self.data.Low, self.data.Close, self.atr_length),
            name="ATR",
            color="gray",
        )
        # Upper and Lower bands
        self.upper_band = self.I(
            lambda: self.basis + self.atr_value * self.atr_multiplier,
            name="Upper Band",
            color="green",
        )
        self.lower_band = self.I(
            lambda: self.basis - self.atr_value * self.atr_multiplier,
            name="Lower Band",
            color="red",
        )
        # Trend EMA
        self.trend_ema = self.I(
            lambda: self.data.Close.s.ewm(span=self.trend_ema, adjust=False).mean().values,
            name=f"Trend EMA{self.trend_ema}",
            color="orange",
        )

    def next(self) -> None:
        """Execute strategy logic on each bar."""
        close = self.data.Close[-1]
        close_prev = self.data.Close[-2]
        low = self.data.Low[-1]
        high = self.data.High[-1]

        basis_val = self.basis[-1]
        upper_val = self.upper_band[-1]
        lower_val = self.lower_band[-1]
        trend_val = self.trend_ema[-1]

        # Trend filter
        uptrend = close > trend_val
        downtrend = close < trend_val

        # Long entry: uptrend + crossover(close, lowerBand) + low <= lowerBand
        long_cross = close > lower_val and close_prev <= lower_val
        long_bounce = long_cross and low <= lower_val
        long_condition = uptrend and long_bounce

        # Short entry: downtrend + crossunder(close, upperBand) + high >= upperBand
        short_cross = close < upper_val and close_prev >= upper_val
        short_reject = short_cross and high >= upper_val
        short_condition = downtrend and short_reject

        # Exit conditions
        exit_long = close < basis_val
        exit_short = close > basis_val

        # Execute trades
        if long_condition and not self.position:
            self.buy()

        if short_condition and not self.position:
            self.sell()

        if exit_long and self.position.is_long:
            self.position.close()

        if exit_short and self.position.is_short:
            self.position.close()
