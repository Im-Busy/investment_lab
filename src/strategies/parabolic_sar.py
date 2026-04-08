"""
Parabolic SAR Strategy for backtesting.py

"Stop and Reverse" indicator. Places dots above/below price that act as
trailing stops. When price crosses the SAR, it signals a trend reversal.

Pine Script source: strategies/trend-following/parabolic-sar/strategy.pine
from EternaHybridExchange/tradingview-strategies repo.
"""

import numpy as np
from backtesting import Strategy


def calc_sar(
    high: np.ndarray,
    low: np.ndarray,
    start: float = 0.02,
    increment: float = 0.02,
    maximum: float = 0.2,
) -> np.ndarray:
    """Calculate Parabolic SAR values."""
    n = len(high)
    sar = np.full(n, np.nan)
    trend = np.full(n, 1)  # 1 = uptrend, -1 = downtrend
    ep = np.full(n, np.nan)  # Extreme Point
    af = np.full(n, start)  # Acceleration Factor

    # Initialize: assume uptrend, SAR starts at first low
    sar[0] = low[0]
    ep[0] = high[0]
    af[0] = start

    for i in range(1, n):
        prev_sar = sar[i - 1]
        prev_af = af[i - 1]
        prev_ep = ep[i - 1]
        prev_trend = trend[i - 1]

        # Calculate SAR
        sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)

        if prev_trend == 1:  # Uptrend
            # SAR cannot go above previous two bars' lows
            if i >= 2:
                min_low = min(low[i - 1], low[i - 2])
                sar[i] = min(sar[i], min_low)

            if high[i] > prev_ep:
                ep[i] = high[i]
                af[i] = min(prev_af + increment, maximum)
            else:
                ep[i] = prev_ep
                af[i] = prev_af

            if low[i] < sar[i]:
                # Trend reversal
                trend[i] = -1
                sar[i] = prev_ep
                ep[i] = low[i]
                af[i] = start
            else:
                trend[i] = 1
        else:  # Downtrend
            # SAR cannot go below previous two bars' highs
            if i >= 2:
                max_high = max(high[i - 1], high[i - 2])
                sar[i] = max(sar[i], max_high)

            if low[i] < prev_ep:
                ep[i] = low[i]
                af[i] = min(prev_af + increment, maximum)
            else:
                ep[i] = prev_ep
                af[i] = prev_af

            if high[i] > sar[i]:
                # Trend reversal
                trend[i] = 1
                sar[i] = prev_ep
                ep[i] = high[i]
                af[i] = start
            else:
                trend[i] = -1

    return sar


class ParabolicSARStrategy(Strategy):
    """
    Parabolic SAR Strategy

    Entry logic:
    - Long: SAR flips up (close crosses above SAR) AND close > 50 EMA
    - Short: SAR flips down (close crosses below SAR) AND close < 50 EMA

    Exit: On opposite SAR flip (always in market)

    Parameters:
        sar_start: Initial acceleration factor (default 0.02)
        sar_increment: Step increment (default 0.02)
        sar_maximum: Maximum AF (default 0.2)
        trend_ema: Trend filter EMA (default 50)
        use_trend_filter: Enable trend filter (default True)
    """

    sar_start = 0.02
    sar_increment = 0.02
    sar_maximum = 0.2
    trend_ema = 50
    use_trend_filter = True

    def init(self) -> None:
        """Initialize Parabolic SAR indicator."""
        self.sar_value = self.I(
            lambda: calc_sar(
                self.data.High, self.data.Low, self.sar_start, self.sar_increment, self.sar_maximum
            ),
            name="SAR",
            color="red",
        )
        if self.use_trend_filter:
            self.ema_filter = self.I(
                lambda: self.data.Close.s.ewm(span=self.trend_ema, adjust=False).mean().values,
                name=f"EMA{self.trend_ema}",
                color="orange",
            )

    def next(self) -> None:
        """Execute strategy logic on each bar."""
        close = self.data.Close[-1]
        close_prev = self.data.Close[-2]
        sar = self.sar_value[-1]
        sar_prev = self.sar_value[-2]

        # SAR flip detection
        sar_flip_up = close > sar and close_prev <= sar_prev
        sar_flip_down = close < sar and close_prev >= sar_prev

        # Trend filter
        if self.use_trend_filter:
            ema_val = self.ema_filter[-1]
            long_condition = sar_flip_up and close > ema_val
            short_condition = sar_flip_down and close < ema_val
        else:
            long_condition = sar_flip_up
            short_condition = sar_flip_down

        # Execute trades
        if long_condition and not self.position:
            self.buy()

        if short_condition and not self.position:
            self.sell()

        # Exit on opposite flip
        if sar_flip_down and self.position.is_long:
            self.position.close()

        if sar_flip_up and self.position.is_short:
            self.position.close()
