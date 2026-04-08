"""
Awesome Oscillator Strategy for backtesting.py

Bill Williams momentum oscillator. SMA(5) - SMA(34) of median price.
Signals: zero-line cross, twin peaks divergence, saucer pattern.
"""

from backtesting import Strategy


class AwesomeOscillatorStrategy(Strategy):
    """Awesome Oscillator Strategy."""

    fast_length = 5
    slow_length = 34

    def init(self) -> None:
        """Initialize AO indicator."""
        fl, sl = self.fast_length, self.slow_length

        def calc_ao():
            median = (self.data.High.s + self.data.Low.s) / 2
            ao = median.rolling(window=fl).mean() - median.rolling(window=sl).mean()
            return ao.values

        self.ao = self.I(calc_ao, name="AO", color="blue")

    def next(self) -> None:
        """Execute strategy logic."""
        ao = self.ao[-1]
        ao1 = self.ao[-2]
        ao2 = self.ao[-3]
        ao3 = self.ao[-4]

        # Zero-line cross
        bullish_zero = ao > 0 and ao1 <= 0
        bearish_zero = ao < 0 and ao1 >= 0

        # Twin Peaks (simplified)
        twin_bullish = ao1 < 0 and ao > ao1 and ao2 < ao1 and ao2 < ao3
        twin_bearish = ao1 > 0 and ao < ao1 and ao2 > ao1 and ao2 > ao3

        # Saucer
        saucer_bull = ao > 0 and ao1 < 0 and ao1 < ao2
        saucer_bear = ao < 0 and ao1 > 0 and ao1 > ao2

        buy = bullish_zero or twin_bullish or saucer_bull
        sell = bearish_zero or twin_bearish or saucer_bear

        if buy and not self.position:
            self.buy()

        if sell and not self.position:
            self.sell()

        if sell and self.position.is_long:
            self.position.close()

        if buy and self.position.is_short:
            self.position.close()
