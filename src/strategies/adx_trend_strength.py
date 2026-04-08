"""
ADX Trend Strength Strategy for backtesting.py

Uses Average Directional Index (ADX) to measure trend strength and
Directional Indicators (DI+/DI-) to determine direction. Only trades
when ADX confirms strong trending conditions.

Pine Script source: strategies/trend-following/adx-trend-strength/strategy.pine
from EternaHybridExchange/tradingview-strategies repo.
"""

import numpy as np
from backtesting import Strategy


def calc_dmi(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, di_length: int, adx_length: int
) -> tuple:
    """
    Calculate DI+, DI-, and ADX using Wilder's smoothing.
    Returns (di_plus, di_minus, adx) arrays.
    """
    n = len(high)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # Wilder's smoothing
    atr_tr = np.maximum(high, np.roll(close, 1)) - np.minimum(low, np.roll(close, 1))
    atr_tr[0] = np.nan

    def wilder_smooth(data: np.ndarray, length: int) -> np.ndarray:
        result = np.full_like(data, np.nan)
        for i in range(length - 1, len(data)):
            if i == length - 1:
                valid = data[1 : i + 1]  # skip first element
                result[i] = np.nanmean(valid)
            else:
                result[i] = (result[i - 1] * (length - 1) + data[i]) / length
        return result

    smoothed_plus_dm = wilder_smooth(plus_dm, di_length)
    smoothed_minus_dm = wilder_smooth(minus_dm, di_length)
    smoothed_tr = wilder_smooth(atr_tr, di_length)

    di_plus = np.where(smoothed_tr > 0, 100 * smoothed_plus_dm / smoothed_tr, 0)
    di_minus = np.where(smoothed_tr > 0, 100 * smoothed_minus_dm / smoothed_tr, 0)

    # ADX calculation
    dx = np.where(
        (di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0
    )
    adx = wilder_smooth(dx.astype(float), adx_length)

    return di_plus, di_minus, adx


class ADXTrendStrengthStrategy(Strategy):
    """
    ADX Trend Strength Strategy

    Entry logic:
    - Long: DI+ crosses above DI- AND ADX > 25 AND price > 50 EMA
    - Short: DI- crosses above DI+ AND ADX > 25 AND price < 50 EMA

    Exit logic:
    - Long exit: DI- crosses above DI+ OR ADX drops below 70% of threshold
    - Short exit: DI+ crosses above DI- OR ADX drops below 70% of threshold

    Parameters:
        adx_length: ADX smoothing period (default 14)
        adx_threshold: Minimum ADX for strong trend (default 25)
        di_length: DI calculation period (default 14)
        ema_length: Trend EMA period (default 50)
    """

    adx_length = 14
    adx_threshold = 25
    di_length = 14
    ema_length = 50

    def init(self) -> None:
        """Initialize ADX/DI indicators."""
        di_plus, di_minus, adx = calc_dmi(
            self.data.High, self.data.Low, self.data.Close, self.di_length, self.adx_length
        )
        self.di_plus = self.I(lambda: di_plus, name="DI+", color="green")
        self.di_minus = self.I(lambda: di_minus, name="DI-", color="red")
        self.adx = self.I(lambda: adx, name="ADX", color="blue")
        self.trend_ema = self.I(
            lambda: self.data.Close.s.ewm(span=self.ema_length, adjust=False).mean().values,
            name=f"EMA{self.ema_length}",
            color="orange",
        )

    def next(self) -> None:
        """Execute strategy logic on each bar."""
        di_p = self.di_plus[-1]
        di_m = self.di_minus[-1]
        di_p_prev = self.di_plus[-2]
        di_m_prev = self.di_minus[-2]
        adx_val = self.adx[-1]
        close = self.data.Close[-1]
        ema_val = self.trend_ema[-1]

        # Strong trend condition
        strong_trend = adx_val > self.adx_threshold

        # Entry conditions
        bullish_di = di_p > di_m and di_p_prev <= di_m_prev
        bearish_di = di_m > di_p and di_m_prev <= di_p_prev

        long_condition = bullish_di and strong_trend and close > ema_val
        short_condition = bearish_di and strong_trend and close < ema_val

        # Exit conditions
        exit_long = (di_m > di_p) or (adx_val < self.adx_threshold * 0.7)
        exit_short = (di_p > di_m) or (adx_val < self.adx_threshold * 0.7)

        # Execute trades
        if long_condition and not self.position:
            self.buy()

        if short_condition and not self.position:
            self.sell()

        if exit_long and self.position.is_long:
            self.position.close()

        if exit_short and self.position.is_short:
            self.position.close()
