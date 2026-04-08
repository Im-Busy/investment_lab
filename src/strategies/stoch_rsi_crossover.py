"""
Stochastic RSI Crossover Strategy for backtesting.py

K crosses above D in oversold zone for long, K crosses below D in overbought for short.

Pine Script source: strategies/momentum/stoch_rsi_crossover.pine
from EternaHybridExchange/tradingview-strategies repo.
"""

from backtesting import Strategy


class StochRSICrossoverStrategy(Strategy):
    """
    Stochastic RSI Crossover Strategy

    Entry logic:
    - Long: K crosses above D while in oversold zone (< 20)
    - Short: K crosses below D while in overbought zone (> 80)

    Exit: Opposite crossover
    """

    rsi_length = 14
    stoch_length = 14
    k_smoothing = 3
    d_smoothing = 3
    upper_level = 80
    lower_level = 20

    def init(self) -> None:
        """Initialize StochRSI indicators."""
        closes = self.data.Close.s

        def calc_stoch_rsi():
            rsi = closes.rolling(window=self.rsi_length).apply(
                lambda x: 100 - 100 / (1 + (x[x > 0].sum() / max(1, -x[x < 0].sum()))), raw=True
            )
            stoch_rsi = (
                100
                * (rsi - rsi.rolling(window=self.stoch_length).min())
                / (
                    rsi.rolling(window=self.stoch_length).max()
                    - rsi.rolling(window=self.stoch_length).min()
                )
            )
            k = stoch_rsi.rolling(window=self.k_smoothing).mean()
            d = k.rolling(window=self.d_smoothing).mean()
            return k.values, d.values

        self.k_vals, self.d_vals = calc_stoch_rsi()
        self.k = self.I(lambda: self.k_vals, name="K", color="blue")
        self.d = self.I(lambda: self.d_vals, name="D", color="orange")

    def next(self) -> None:
        """Execute strategy logic."""
        k = self.k[-1]
        d = self.d[-1]
        k_prev = self.k[-2]
        d_prev = self.d[-2]

        bullish_cross = k > d and k_prev <= d_prev and k < self.lower_level
        bearish_cross = k < d and k_prev >= d_prev and k > self.upper_level

        if bullish_cross and not self.position:
            self.buy()

        if bearish_cross and not self.position:
            self.sell()

        if bearish_cross and self.position.is_long:
            self.position.close()

        if bullish_cross and self.position.is_short:
            self.position.close()
