"""
Connors RSI Mean Reversion Strategy

Based on research by Larry Connors from "Short Term Trading Strategies That Work"
This is a proven mean-reversion strategy with documented edge.

Strategy Logic:
1. Stock must be in long-term uptrend (Close > 200-day SMA)
2. 2-period RSI drops below 10 (oversold)
3. Buy on close
4. Exit when Close > 5-day SMA

Risk Management:
- Position sizing: Equal weight across signals
- No stop loss (mean reversion relies on staying in trade)
- Only long positions (bull market bias)

References:
- Connors, L. (2009). Short Term Trading Strategies That Work
- Connors, L. & Katsenelson, C. (2012). High Probability Trading Strategies

"""

import numpy as np
from backtesting import Strategy


class ConnorsRSIMeanReversion(Strategy):
    """
    Connors RSI 2-period mean reversion strategy.

    Parameters:
        rsi_period: RSI calculation period (default: 2)
        rsi_entry_threshold: RSI level to enter (default: 10)
        exit_sma_period: SMA period for exit signal (default: 5)
        trend_sma_period: Long-term trend SMA period (default: 200)
    """

    rsi_period = 2
    rsi_entry_threshold = 10
    exit_sma_period = 5
    trend_sma_period = 200

    def init(self):
        """Calculate indicators."""
        # 2-period RSI
        self.rsi = self.I(self._calculate_rsi, self.data.Close, self.rsi_period)

        # 5-day SMA for exit
        self.exit_sma = self.I(self._sma, self.data.Close, self.exit_sma_period)

        # 200-day SMA for trend filter
        self.trend_sma = self.I(self._sma, self.data.Close, self.trend_sma_period)

    @staticmethod
    def _sma(close, period):
        import numpy as np

        result = np.empty_like(close)
        result[:] = np.nan
        for i in range(period - 1, len(close)):
            result[i] = np.mean(close[i - period + 1 : i + 1])
        return result

    @staticmethod
    def _calculate_rsi(close, period):
        import numpy as np

        rsi = np.empty_like(close)
        rsi[:] = np.nan

        if len(close) < period + 1:
            return rsi

        # Calculate price changes
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        # Initial average gain/loss
        avg_gain = np.mean(gain[:period])
        avg_loss = np.mean(loss[:period])

        # Set initial RSI
        if avg_loss == 0:
            rsi[period] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[period] = 100 - (100 / (1 + rs))

        # Smoothed RSI for remaining values
        for i in range(period, len(close) - 1):
            avg_gain = (avg_gain * (period - 1) + gain[i]) / period
            avg_loss = (avg_loss * (period - 1) + loss[i]) / period

            if avg_loss == 0:
                rsi[i + 1] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100 - (100 / (1 + rs))

        return rsi

    def next(self):
        """Execute trading logic."""
        # Skip if indicators not ready
        if np.isnan(self.rsi[-1]) or np.isnan(self.trend_sma[-1]):
            return

        in_uptrend = self.data.Close[-1] > self.trend_sma[-1]

        if self.position:
            # Exit when close > 5-day SMA
            if self.data.Close[-1] > self.exit_sma[-1]:
                if self.position.is_long:
                    self.sell(size=1)
        else:
            # Enter when RSI < threshold AND in uptrend
            if self.rsi[-1] < self.rsi_entry_threshold and in_uptrend:
                self.buy(size=0.95)
