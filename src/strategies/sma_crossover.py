"""
SMA Crossover 50/200 Strategy for backtesting.py

Classic trend-following strategy using Simple Moving Average crossovers.
The Golden Cross (fast SMA crosses above slow SMA) signals bullish trends,
while the Death Cross (fast crosses below slow) signals bearish trends.

Pine Script source: strategies/trend-following/sma-crossover/strategy.pine
from EternaHybridExchange/tradingview-strategies repo.
"""

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
    """

    fast_sma = 50
    slow_sma = 200

    def init(self) -> None:
        """Initialize SMA indicators."""
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

    def next(self) -> None:
        """Execute strategy logic on each bar."""
        fast = self.sma_fast[-1]
        slow = self.sma_slow[-1]
        fast_prev = self.sma_fast[-2]
        slow_prev = self.sma_slow[-2]

        # Golden Cross: fast SMA crosses above slow SMA
        golden_cross = fast > slow and fast_prev <= slow_prev

        # Death Cross: fast SMA crosses below slow SMA
        death_cross = fast < slow and fast_prev >= slow_prev

        # Execute trades
        if golden_cross and not self.position:
            self.buy()

        if death_cross and not self.position:
            self.sell()

        # Exit on opposite signal
        if death_cross and self.position.is_long:
            self.position.close()

        if golden_cross and self.position.is_short:
            self.position.close()
