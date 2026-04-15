"""
Donchian Channel Breakout Strategy for backtesting.py

Classic breakout system based on Richard Donchian's trend-following method
(used in the famous Turtle Trading system). Enters when price breaks above/below
the highest high/lowest low of the past N periods.

How It Works:
- Long Entry: Price closes above N-period high (upper band breakout)
- Short Entry: Price closes below N-period low (lower band breakdown)
- Exit: Price crosses middle line or opposite band

Components:
- Upper Band: Highest high of last N periods
- Lower Band: Lowest low of last N periods
- Middle Line: Average of upper and lower bands

Best Timeframes:
- Primary: 1H, 4H, 1D
- Alternative: 15m for scalping
- Original Turtle: Daily (20-day/55-day channels)

Risk/Reward Profile:
- Win Rate: 35-45%
- Risk/Reward: 1:3-5 (few large wins)
- Max Drawdown: 20-30%
- Trading Style: Breakout/momentum
"""

import numpy as np
from backtesting import Strategy


class DonchianChannelStrategy(Strategy):
    """
    Donchian Channel Breakout Strategy

    Entry logic:
    - Long: Close crosses above N-period high (upper band)
    - Short: Close crosses below N-period low (lower band)

    Exit logic:
    - Long exit: Close crosses below middle line
    - Short exit: Close crosses above middle line

    Parameters:
        channel_period: Lookback period for channel (default 20)
        exit_period: Lookback for exit channel (default 10)
    """

    channel_period = 20
    exit_period = 10

    def init(self) -> None:
        """Initialize Donchian channel indicators."""
        p = self.channel_period

        self.upper = self.I(
            lambda: self.data.High.s.rolling(window=p, min_periods=p).max().values,
            name=f"Donchian Upper ({p})",
            color="green",
        )
        self.lower = self.I(
            lambda: self.data.Low.s.rolling(window=p, min_periods=p).min().values,
            name=f"Donchian Lower ({p})",
            color="red",
        )

    def next(self) -> None:
        """Execute strategy logic on each bar."""
        close = self.data.Close[-1]
        prev_close = self.data.Close[-2]
        upper = self.upper[-1]
        lower = self.lower[-1]

        # Skip if indicators are NaN (warmup period)
        if np.isnan(upper) or np.isnan(lower):
            return

        # Compute middle line on the fly
        middle = (upper + lower) / 2

        # Previous bar channel values for breakout detection
        upper_prev = self.upper[-2]
        lower_prev = self.lower[-2]

        if np.isnan(upper_prev) or np.isnan(lower_prev):
            return

        middle_prev = (upper_prev + lower_prev) / 2

        # Long breakout: close crosses above upper channel
        long_breakout = close > upper_prev and prev_close <= upper_prev

        # Short breakdown: close crosses below lower channel
        short_breakout = close < lower_prev and prev_close >= lower_prev

        # Long exit: close crosses below middle line
        long_exit = close < middle_prev and prev_close >= middle_prev

        # Short exit: close crosses above middle line
        short_exit = close > middle_prev and prev_close <= middle_prev

        # Execute trades
        if long_breakout and not self.position:
            self.buy()

        if short_breakout and not self.position:
            self.sell()

        # Exit on opposite signal or middle line cross
        if long_exit and self.position.is_long:
            self.position.close()

        if short_exit and self.position.is_short:
            self.position.close()

        # Also exit long on short breakout (reversal)
        if short_breakout and self.position.is_long:
            self.position.close()

        # Also exit short on long breakout (reversal)
        if long_breakout and self.position.is_short:
            self.position.close()
