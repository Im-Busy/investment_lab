"""
Linear Regression Channel Strategy for backtesting.py

Statistical trend-following system based on linear regression with
standard deviation bands.

Pine Script source: strategies/trend-following/linear-regression-channel/strategy.pine
from EternaHybridExchange/tradingview-strategies repo.
"""

import numpy as np
from backtesting import Strategy


class LinearRegressionChannelStrategy(Strategy):
    """Linear Regression Channel Strategy."""

    length_lr = 100
    deviation_multiplier = 2.0
    trend_ema = 50

    def init(self) -> None:
        """Initialize Linear Regression Channel."""
        length = self.length_lr
        dev_mult = self.deviation_multiplier
        ema_len = self.trend_ema

        def calc_lr_channel():
            close_s = self.data.Close.s
            n = len(close_s)
            lin_reg = np.full(n, np.nan)
            upper = np.full(n, np.nan)
            lower = np.full(n, np.nan)

            for i in range(length - 1, n):
                y = close_s.iloc[i - length + 1 : i + 1].values
                x = np.arange(length)
                # Linear regression: y = mx + b
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                ss_xy = np.sum((x - x_mean) * (y - y_mean))
                ss_xx = np.sum((x - x_mean) ** 2)
                if ss_xx == 0:
                    continue
                m = ss_xy / ss_xx
                b = y_mean - m * x_mean
                # Predicted value at last point
                lin_reg[i] = m * (length - 1) + b
                # Standard deviation of residuals
                predicted = m * x + b
                residuals = y - predicted
                std_dev = np.sqrt(np.sum(residuals**2) / length)
                upper[i] = lin_reg[i] + std_dev * dev_mult
                lower[i] = lin_reg[i] - std_dev * dev_mult

            return lin_reg, upper, lower

        self.lr_vals, self.upper_vals, self.lower_vals = calc_lr_channel()
        self.lin_reg = self.I(lambda: self.lr_vals, name="LinReg", color="blue")
        self.upper = self.I(lambda: self.upper_vals, name="Upper", color="green")
        self.lower = self.I(lambda: self.lower_vals, name="Lower", color="red")
        self.ema_filter = self.I(
            lambda: self.data.Close.s.ewm(span=ema_len, adjust=False).mean().values,
            name=f"EMA{ema_len}",
            color="orange",
        )

    def next(self) -> None:
        """Execute strategy logic."""
        close = self.data.Close[-1]
        close_prev = self.data.Close[-2]
        lr = self.lin_reg[-1]
        lr_prev = self.lin_reg[-2]
        lr_5 = self.lin_reg[-6] if len(self.data) > 5 else lr
        upper = self.upper[-1]
        lower = self.lower[-1]
        ema = self.ema_filter[-1]

        if np.isnan(lr) or np.isnan(upper) or np.isnan(lower):
            return

        uptrend = close > ema and lr > lr_5
        downtrend = close < ema and lr < lr_5

        long_bounce = close > lower and close_prev <= lower and self.data.Low[-1] <= lower
        short_reject = close < upper and close_prev >= upper and self.data.High[-1] >= upper

        long_condition = uptrend and long_bounce
        short_condition = downtrend and short_reject

        exit_long = close < lr
        exit_short = close > lr

        if long_condition and not self.position:
            self.buy()

        if short_condition and not self.position:
            self.sell()

        if exit_long and self.position.is_long:
            self.position.close()

        if exit_short and self.position.is_short:
            self.position.close()
