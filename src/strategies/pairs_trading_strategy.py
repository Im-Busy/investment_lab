"""Pairs trading strategy with cointegration and z-score mean reversion.

Uses instrument A's OHLCV as the primary tradable asset. Instrument B's close
is provided as extra 'Close_B' column. Spread signals from both instruments
drive entries/exits on A. This approximates pairs P&L via the primary leg.

Usage:
    from backtesting import Backtest
    bt = Backtest(df_A_with_CloseB, PairsTradingStrategy, cash=10_000, commission=.001)
    stats = bt.run()
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from backtesting import Strategy

logger = logging.getLogger(__name__)


class PairsTradingStrategy(Strategy):
    """Mean-reversion pairs strategy using instrument A for execution.

    Spread = log(P_A) - hedge_ratio * log(P_B), computed from:
      - Close_A = self.data.Close (primary instrument)
      - Close_B = self.data.df['Close_B'] (secondary instrument)

    Parameters:
        lookback: Rolling window for hedge and z-score estimation (default 252).
        entry_z: Z-score threshold for entry (default 2.0).
        exit_z: Z-score threshold for exit (default 0.0).
        stop_z: Z-score threshold for stop loss (default 3.0).
        min_correlation: Minimum rolling correlation to allow trades (default 0.7).
        atr_exit_mult: ATR multiplier for trailing stop (default 2.0; 0 = disabled).
    """

    lookback: int = 252
    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_z: float = 3.0
    min_correlation: float = 0.7
    atr_exit_mult: float = 0.0

    def init(self) -> None:
        df = self.data.df

        if "Close_B" not in df.columns:
            raise ValueError(
                "DataFrame must contain 'Close_B' column. "
                "Use scripts/backtest_pairs.py to load data."
            )

        close_a = pd.Series(self.data.Close, index=df.index)
        close_b = pd.Series(df["Close_B"].values.astype(float), index=df.index)

        log_a = np.log(close_a.values)
        log_b = np.log(np.maximum(close_b.values, 1e-10))

        self._log_a = log_a
        self._log_b = log_b

        self._hedge = self.I(self._rolling_hedge_ratio, log_a, log_b, self.lookback)
        self._spread = self.I(self._compute_spread, log_a, log_b, self._hedge)
        self._spread_mean = self.I(self._rolling_mean, self._spread, self.lookback)
        self._spread_std = self.I(self._rolling_std, self._spread, self.lookback)
        self._zscore = self.I(
            self._compute_zscore, self._spread, self._spread_mean, self._spread_std
        )
        self._correlation = self.I(self._rolling_corr, log_a, log_b, self.lookback)

        if self.atr_exit_mult > 0:
            high = pd.Series(self.data.High, index=df.index)
            low = pd.Series(self.data.Low, index=df.index)
            self._atr = self.I(self._compute_atr, high.values, low.values, close_a.values, 14)
        else:
            self._atr = None

    @staticmethod
    def _rolling_hedge_ratio(log_a: np.ndarray, log_b: np.ndarray, window: int) -> np.ndarray:
        result = np.full_like(log_a, np.nan)
        for i in range(window, len(log_a)):
            x = log_b[i - window : i]
            y = log_a[i - window : i]
            vx = np.var(x)
            if vx > 1e-10:
                result[i] = np.cov(x, y)[0, 1] / vx
            else:
                result[i] = 1.0
        return result

    @staticmethod
    def _compute_spread(log_a: np.ndarray, log_b: np.ndarray, hedge: np.ndarray) -> np.ndarray:
        result = np.full_like(log_a, np.nan)
        mask = ~np.isnan(hedge)
        result[mask] = log_a[mask] - hedge[mask] * log_b[mask]
        return result

    @staticmethod
    def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
        result = np.full_like(values, np.nan)
        for i in range(window - 1, len(values)):
            seg = values[i - window + 1 : i + 1]
            mask = ~np.isnan(seg)
            if mask.sum() > window // 2:
                result[i] = np.mean(seg[mask])
        return result

    @staticmethod
    def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
        result = np.full_like(values, np.nan)
        for i in range(window - 1, len(values)):
            seg = values[i - window + 1 : i + 1]
            mask = ~np.isnan(seg)
            if mask.sum() > window // 2:
                result[i] = np.std(seg[mask], ddof=1)
        return result

    @staticmethod
    def _compute_zscore(spread: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        result = np.full_like(spread, np.nan)
        for i in range(len(spread)):
            if (
                not np.isnan(spread[i])
                and not np.isnan(mean[i])
                and not np.isnan(std[i])
                and std[i] > 1e-10
            ):
                result[i] = (spread[i] - mean[i]) / std[i]
        return result

    @staticmethod
    def _rolling_corr(log_a: np.ndarray, log_b: np.ndarray, window: int) -> np.ndarray:
        result = np.full_like(log_a, np.nan)
        for i in range(window, len(log_a)):
            x = log_a[i - window : i]
            y = log_b[i - window : i]
            sx = np.std(x)
            sy = np.std(y)
            if sx > 0 and sy > 0:
                result[i] = np.corrcoef(x, y)[0, 1]
        return result

    @staticmethod
    def _compute_atr(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> np.ndarray:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
        )
        tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
        tr = np.insert(tr, 0, high[0] - low[0])
        atr = np.full_like(close, np.nan)
        if len(tr) >= period:
            atr[period - 1] = np.mean(tr[:period])
            for i in range(period, len(tr)):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

    def next(self) -> None:
        z = self._zscore[-1]
        corr = self._correlation[-1]

        if np.isnan(z) or np.isnan(corr):
            return

        if corr < self.min_correlation:
            if self.position:
                self.position.close()
            return

        if self.position:
            if self.position.is_long:
                if z > -self.exit_z or z < -self.stop_z:
                    self.position.close()
                elif self.atr_exit_mult > 0 and self._atr is not None:
                    atr_val = self._atr[-1]
                    if not np.isnan(atr_val) and atr_val > 0:
                        stop_price = self.data.Close[-1] - self.atr_exit_mult * atr_val
                        if self.data.Low[-1] <= stop_price:
                            self.position.close()
            else:
                if z < self.exit_z or z > self.stop_z:
                    self.position.close()
                elif self.atr_exit_mult > 0 and self._atr is not None:
                    atr_val = self._atr[-1]
                    if not np.isnan(atr_val) and atr_val > 0:
                        stop_price = self.data.Close[-1] + self.atr_exit_mult * atr_val
                        if self.data.High[-1] >= stop_price:
                            self.position.close()
        else:
            if z < -self.entry_z:
                self.buy()
            elif z > self.entry_z:
                self.sell()
