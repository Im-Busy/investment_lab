"""
Ichimoku Cloud Strategy for backtesting.py

Comprehensive Japanese trend-following system. The "cloud" (Kumo) acts as
dynamic support/resistance, while Tenkan/Kijun crosses signal entries.

Pine Script source: strategies/trend-following/ichimoku-cloud/strategy.pine
from EternaHybridExchange/tradingview-strategies repo.
"""

import numpy as np
from backtesting import Strategy


def donchian(high: np.ndarray, low: np.ndarray, length: int) -> np.ndarray:
    """Calculate Donchian channel midpoint."""
    result = np.full_like(high, np.nan)
    for i in range(length - 1, len(high)):
        result[i] = (np.max(high[i - length + 1 : i + 1]) + np.min(low[i - length + 1 : i + 1])) / 2
    return result


class IchimokuCloudStrategy(Strategy):
    """
    Ichimoku Cloud Strategy

    Entry logic:
    - Long: Price above cloud + Tenkan crosses above Kijun + bullish cloud
    - Short: Price below cloud + Tenkan crosses below Kijun + bearish cloud

    Exit logic:
    - Long exit: Tenkan crosses below Kijun
    - Short exit: Tenkan crosses above Kijun

    Parameters:
        conversion_period: Tenkan-sen period (default 9)
        base_period: Kijun-sen period (default 26)
        lagging_span_period: Senkou Span B period (default 52)
        displacement: Cloud forward displacement (default 26)
    """

    conversion_period = 9
    base_period = 26
    lagging_span_period = 52
    displacement = 26

    def init(self) -> None:
        """Initialize Ichimoku indicators."""
        self.conversion = self.I(
            lambda: donchian(self.data.High, self.data.Low, self.conversion_period),
            name="Tenkan",
            color="blue",
        )
        self.base = self.I(
            lambda: donchian(self.data.High, self.data.Low, self.base_period),
            name="Kijun",
            color="red",
        )
        self.lead1 = self.I(
            lambda: (self.conversion + self.base) / 2,
            name="Senkou A",
            color="green",
        )
        self.lead2 = self.I(
            lambda: donchian(self.data.High, self.data.Low, self.lagging_span_period),
            name="Senkou B",
            color="red",
        )
        # Displaced cloud (shifted forward)
        self.cloud_upper = self.I(
            lambda: self._shift(np.maximum(self.lead1, self.lead2)),
            name="Cloud Upper",
            color="lightgreen",
        )
        self.cloud_lower = self.I(
            lambda: self._shift(np.minimum(self.lead1, self.lead2)),
            name="Cloud Lower",
            color="lightred",
        )

    def _shift(self, arr: np.ndarray) -> np.ndarray:
        """Shift array forward by displacement."""
        result = np.full_like(arr, np.nan)
        for i in range(self.displacement, len(arr)):
            result[i] = arr[i - self.displacement]
        return result

    def next(self) -> None:
        """Execute strategy logic on each bar."""
        conv = self.conversion[-1]
        base = self.base[-1]
        conv_prev = self.conversion[-2]
        base_prev = self.base[-2]
        close = self.data.Close[-1]
        cloud_up = self.cloud_upper[-1]
        cloud_lo = self.cloud_lower[-1]

        # Cloud conditions
        bullish_cloud = cloud_up > cloud_lo
        bearish_cloud = cloud_up < cloud_lo

        # Price vs cloud
        price_above = (
            close > max(cloud_up, cloud_lo)
            if not (np.isnan(cloud_up) or np.isnan(cloud_lo))
            else False
        )
        price_below = (
            close < min(cloud_up, cloud_lo)
            if not (np.isnan(cloud_up) or np.isnan(cloud_lo))
            else False
        )

        # TK crosses
        bullish_tk = conv > base and conv_prev <= base_prev
        bearish_tk = conv < base and conv_prev >= base_prev

        # Entry conditions
        long_condition = price_above and bullish_tk and bullish_cloud
        short_condition = price_below and bearish_tk and bearish_cloud

        # Exit conditions
        exit_long = conv < base
        exit_short = conv > base

        # Execute trades
        if long_condition and not self.position:
            self.buy()

        if short_condition and not self.position:
            self.sell()

        if exit_long and self.position.is_long:
            self.position.close()

        if exit_short and self.position.is_short:
            self.position.close()
