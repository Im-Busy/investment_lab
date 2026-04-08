"""
VWAP Bounce Strategy for backtesting.py

Institutional-grade mean reversion within a trend. Uses Volume Weighted Average Price
(VWAP) as dynamic support/resistance. Enters when price bounces off VWAP bands in the
direction of the main trend (determined by 50 EMA).

Pine Script source: strategies/trend-following/vwap-bounce/strategy.pine
from EternaHybridExchange/tradingview-strategies repo.
"""

from backtesting import Strategy

from src.indicators.vwap import compute_vwap


class VWAPBounceStrategy(Strategy):
    """
    VWAP Bounce Strategy

    Entry logic:
    - Long: Price above 50 EMA + bounces from VWAP lower band
      (previous bar low <= vwap_lower, current bar close crosses above vwap_lower)
    - Short: Price below 50 EMA + rejects from VWAP upper band
      (previous bar high >= vwap_upper, current bar close crosses below vwap_upper)

    Exit logic:
    - Long exit: Close crosses below center VWAP line
    - Short exit: Close crosses above center VWAP line

    Parameters:
        vwap_deviation_pct: Band width as percentage of VWAP (default 0.5%)
        trend_ema: EMA period for trend filter (default 50)
    """

    vwap_deviation_pct = 0.5
    trend_ema = 50

    def init(self) -> None:
        """Initialize indicators."""
        # Compute VWAP with bands
        vwap_df = compute_vwap(self.data.df, deviation_pct=self.vwap_deviation_pct)
        self.vwap = self.I(lambda: vwap_df["vwap"].values, name="VWAP", color="blue")
        self.vwap_upper = self.I(
            lambda: vwap_df["vwap_upper"].values, name="VWAP Upper", color="lightblue"
        )
        self.vwap_lower = self.I(
            lambda: vwap_df["vwap_lower"].values, name="VWAP Lower", color="lightblue"
        )

        # Trend EMA
        self.ema_trend = self.I(
            lambda: self.data.Close.s.ewm(span=self.trend_ema, adjust=False).mean().values,
            name=f"EMA{self.trend_ema}",
            color="orange",
        )

    def next(self) -> None:
        """Execute strategy logic on each bar."""
        close = self.data.Close[-1]
        close_prev = self.data.Close[-2]
        low = self.data.Low[-1]
        high = self.data.High[-1]

        vwap_val = self.vwap[-1]
        vwap_upper_val = self.vwap_upper[-1]
        vwap_lower_val = self.vwap_lower[-1]
        vwap_lower_prev = self.vwap_lower[-2]
        vwap_upper_prev = self.vwap_upper[-2]
        ema_val = self.ema_trend[-1]

        # Trend filter
        uptrend = close > ema_val
        downtrend = close < ema_val

        # Long entry: uptrend + crossover(close, vwapLower) + low <= vwapLower
        # ta.crossover(close, vwapLower) = close[-1] > vwap_lower[-1] AND close[-2] <= vwap_lower[-2]
        long_crossover = close > vwap_lower_val and close_prev <= vwap_lower_prev
        long_bounce = long_crossover and low <= vwap_lower_val
        long_condition = uptrend and long_bounce

        # Short entry: downtrend + crossunder(close, vwapUpper) + high >= vwapUpper
        # ta.crossunder(close, vwapUpper) = close[-1] < vwap_upper[-1] AND close[-2] >= vwap_upper[-2]
        short_crossover = close < vwap_upper_val and close_prev >= vwap_upper_prev
        short_reject = short_crossover and high >= vwap_upper_val
        short_condition = downtrend and short_reject

        # Exit conditions
        exit_long = close < vwap_val
        exit_short = close > vwap_val

        # Execute trades
        if long_condition and not self.position:
            self.buy()

        if short_condition and not self.position:
            self.sell()

        if exit_long and self.position.is_long:
            self.position.close()

        if exit_short and self.position.is_short:
            self.position.close()
