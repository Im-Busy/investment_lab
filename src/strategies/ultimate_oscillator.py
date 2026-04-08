"""
Ultimate Oscillator Strategy for backtesting.py

Multi-timeframe momentum oscillator by Larry Williams.
Combines three timeframes (7, 14, 28) to reduce false signals.
"""

from backtesting import Strategy


class UltimateOscillatorStrategy(Strategy):
    """Ultimate Oscillator Strategy."""

    fast_length = 7
    medium_length = 14
    slow_length = 28
    overbought = 70
    oversold = 30

    def init(self) -> None:
        """Initialize Ultimate Oscillator."""
        l1, l2, l3 = self.fast_length, self.medium_length, self.slow_length

        def calc_uo():
            close_s = self.data.Close.s
            high_s = self.data.High.s
            low_s = self.data.Low.s
            bp = close_s - low_s.clip(lower=close_s.shift(1))
            tr = high_s.clip(upper=close_s.shift(1)) - low_s.clip(lower=close_s.shift(1))

            def avg(length):
                return bp.rolling(window=length).sum() / tr.rolling(window=length).sum()

            uo = 100 * (4 * avg(l1) + 2 * avg(l2) + avg(l3)) / 7
            return uo.values

        self.uo = self.I(calc_uo, name="UO", color="blue")

    def next(self) -> None:
        """Execute strategy logic."""
        uo = self.uo[-1]
        uo_prev = self.uo[-2]

        bullish = uo > self.oversold and uo_prev <= self.oversold and uo < 50
        bearish = uo < self.overbought and uo_prev >= self.overbought and uo > 50

        if bullish and not self.position:
            self.buy()

        if bearish and not self.position:
            self.sell()

        if bearish and self.position.is_long:
            self.position.close()

        if bullish and self.position.is_short:
            self.position.close()
