"""
Pair Trading Strategy with Kalman Filter

Implements a spread trading strategy using Kalman filter for
dynamic hedge ratio estimation.

Strategy Logic:
1. Calculate spread = price_A - hedge_ratio * price_B
2. Enter long spread when spread < -entry_z * spread_std
3. Enter short spread when spread > entry_z * spread_std
4. Exit when spread crosses back to mean (z-score near 0)
5. Stop loss if spread exceeds ±stop_z * spread_std

Parameters:
    entry_z: Z-score threshold for entry (default: 2.0)
    exit_z: Z-score threshold for exit (default: 0.5)
    stop_z: Z-score threshold for stop loss (default: 3.0)
    lookback: Rolling window for spread statistics (default: 60)
"""

from backtesting import Strategy
import numpy as np


class PairTradingStrategy(Strategy):
    """
    Pair trading strategy using spread mean reversion.

    Parameters:
        entry_z: Entry z-score threshold (default: 2.0)
        exit_z: Exit z-score threshold (default: 0.5)
        stop_z: Stop loss z-score threshold (default: 3.0)
        lookback: Rolling window for spread statistics (default: 60)
        symbol_a: Name of first asset (for logging)
        symbol_b: Name of second asset (for logging)
    """

    entry_z = 2.0
    exit_z = 0.5
    stop_z = 3.0
    lookback = 60
    symbol_a = "A"
    symbol_b = "B"

    def init(self):
        """Initialize indicators."""
        # Assume data has columns: Close, Close_2 (for second asset)
        price_a = self.data.Close
        price_b = self.data.Close_2

        # Calculate spread using rolling OLS hedge ratio
        self.spread = self.I(self._calculate_spread, price_a, price_b, self.lookback)
        self.spread_mean = self.I(self._rolling_mean, self.spread, self.lookback)
        self.spread_std = self.I(self._rolling_std, self.spread, self.lookback)
        self.z_score = self.I(
            self._calculate_zscore, self.spread, self.spread_mean, self.spread_std
        )

    @staticmethod
    def _calculate_spread(price_a, price_b, lookback):
        """Calculate spread using rolling hedge ratio."""
        spread = np.empty_like(price_a)
        spread[:] = np.nan

        for i in range(lookback, len(price_a)):
            y = price_a[i - lookback : i]
            x = price_b[i - lookback : i]

            # OLS regression: y = beta * x + alpha
            beta = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 0 else 1.0
            alpha = np.mean(y) - beta * np.mean(x)

            spread[i] = price_a[i] - (beta * price_b[i] + alpha)

        return spread

    @staticmethod
    def _rolling_mean(values, window):
        """Calculate rolling mean."""
        result = np.empty_like(values)
        result[:] = np.nan
        for i in range(window - 1, len(values)):
            if not np.all(np.isnan(values[i - window + 1 : i + 1])):
                result[i] = np.nanmean(values[i - window + 1 : i + 1])
        return result

    @staticmethod
    def _rolling_std(values, window):
        """Calculate rolling standard deviation."""
        result = np.empty_like(values)
        result[:] = np.nan
        for i in range(window - 1, len(values)):
            if not np.all(np.isnan(values[i - window + 1 : i + 1])):
                result[i] = np.nanstd(values[i - window + 1 : i + 1], ddof=1)
        return result

    @staticmethod
    def _calculate_zscore(spread, mean, std):
        """Calculate z-score of spread."""
        zscore = np.empty_like(spread)
        zscore[:] = np.nan
        for i in range(len(spread)):
            if (
                not np.isnan(spread[i])
                and not np.isnan(mean[i])
                and not np.isnan(std[i])
                and std[i] > 0
            ):
                zscore[i] = (spread[i] - mean[i]) / std[i]
        return zscore

    def next(self):
        """Execute trading logic."""
        # Skip if indicators not ready
        if np.isnan(self.z_score[-1]) or np.isnan(self.spread_std[-1]):
            return

        z = self.z_score[-1]
        std = self.spread_std[-1]

        if std <= 0:
            return

        if self.position:
            # Check exit conditions
            if self.position.is_long:
                # Long spread: exit when z-score crosses above exit threshold
                if z > -self.exit_z:
                    self.sell(size=1)  # Close long spread position
                # Stop loss
                elif z < -self.stop_z:
                    self.sell(size=1)
            else:
                # Short spread: exit when z-score crosses below exit threshold
                if z < self.exit_z:
                    self.buy(size=1)  # Close short spread position
                # Stop loss
                elif z > self.stop_z:
                    self.buy(size=1)
        else:
            # Entry signals
            if z < -self.entry_z:
                # Spread is too low - go long spread (buy A, sell B)
                self.buy(size=0.5)  # Simplified: just go long
            elif z > self.entry_z:
                # Spread is too high - go short spread (sell A, buy B)
                self.sell(size=0.5)  # Simplified: just go short
