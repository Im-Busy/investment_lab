"""
EMA Ribbon (9/21/55) Strategy for backtesting.py

Multi-EMA trend-following system that uses the alignment of three exponential
moving averages. Entries are taken when price bounces off the fast EMA while
all three EMAs are properly aligned, indicating strong trend momentum.

Pine Script source: strategies/trend-following/ema-ribbon/strategy.pine
from EternaHybridExchange/tradingview-strategies repo.
"""

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
    """

    ema_fast = 9
    ema_mid = 21
    ema_slow = 55

    def init(self) -> None:
        """Initialize EMA indicators."""
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

    def next(self) -> None:
        """Execute strategy logic on each bar."""
        ema9 = self.ema9[-1]
        ema21 = self.ema21[-1]
        ema55 = self.ema55[-1]
        ema9_prev = self.ema9[-2]
        ema21_prev = self.ema21[-2]
        close = self.data.Close[-1]
        close_prev = self.data.Close[-2]

        # Bullish alignment: EMA9 > EMA21 > EMA55
        bullish_alignment = ema9 > ema21 and ema21 > ema55

        # Bearish alignment: EMA9 < EMA21 < EMA55
        bearish_alignment = ema9 < ema21 and ema21 < ema55

        # Long entry: bullish alignment + price crosses above EMA9
        long_cross = close > ema9 and close_prev <= ema9_prev
        long_condition = bullish_alignment and long_cross

        # Short entry: bearish alignment + price crosses below EMA9
        short_cross = close < ema9 and close_prev >= ema9_prev
        short_condition = bearish_alignment and short_cross

        # Exit when alignment breaks (EMA9 crosses EMA21)
        exit_long = ema9 < ema21 and ema9_prev >= ema21_prev
        exit_short = ema9 > ema21 and ema9_prev <= ema21_prev

        # Execute trades
        if long_condition and not self.position:
            self.buy()

        if short_condition and not self.position:
            self.sell()

        if exit_long and self.position.is_long:
            self.position.close()

        if exit_short and self.position.is_short:
            self.position.close()
