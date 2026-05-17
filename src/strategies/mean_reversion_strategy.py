"""Dedicated mean-reversion strategy (backtesting.py Strategy).

Separates mean-reversion from trend-following: different stop logic,
different entry criteria, different regime routing.

Course insight: \"King of ranging markets.\" Trend following loses here,
mean reversion profits. But mean reversion gets steamrolled in trends:
so regime routing is CRITICAL.

Indicators (computed inline, no external dependencies):
- RSI(14) oversold/overbought
- Williams %R(14) oversold/overbought
- StochRSI(14) oversold/overbought
- CCI(20) oversold/overbought
- MFI(14) oversold/overbought
- Bollinger Band touch/rejection

Key differences from trend-following:
- Fixed take-profit at ATR target, NOT trailing stop
- Wider initial stop (reversion can overshoot before snapping back)
- Time-based exit (if hasn't reverted in N bars, exit)
- Requires regime gate: MUST be in Ranging regime (ADX < 20)

Usage:
    from backtesting import Backtest
    bt = Backtest(df, MeanReversionStrategy, cash=10_000, commission=0.001)
    stats = bt.run()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy


class MeanReversionStrategy(Strategy):
    """Pure mean-reversion strategy for ranging/sideways markets.

    Entry: When 2+ indicators agree on oversold/overbought condition
           AND ADX confirms ranging (ADX < adx_threshold).

    Exit: Fixed take-profit at tp_atr_mult * ATR above/below entry,
          wider stop-loss at sl_atr_mult * ATR,
          time-based exit after max_hold_bars.
    """

    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    williams_r_period: int = 14
    williams_oversold: int = -80
    williams_overbought: int = -20
    cci_period: int = 20
    cci_oversold: int = -100
    cci_overbought: int = 100
    mfi_period: int = 14
    mfi_oversold: int = 20
    mfi_overbought: int = 80
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_oversold: int = 20
    stoch_overbought: int = 80
    bb_period: int = 20
    bb_std: float = 2.0
    require_ranging: bool = True
    adx_period: int = 14
    adx_threshold: int = 20
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 3.0
    max_hold_bars: int = 10
    atr_period: int = 14
    risk_pct: float = 0.01
    min_confluence: int = 2

    def init(self) -> None:
        c = pd.Series(self.data.Close)
        h = pd.Series(self.data.High)
        lo = pd.Series(self.data.Low)
        v = pd.Series(self.data.Volume)
        n = len(c)

        tr = pd.concat(
            [
                h - lo,
                (h - c.shift(1)).abs(),
                (lo - c.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        self._atr = tr.ewm(span=self.atr_period, adjust=False).mean().values
        self._close = c.values

        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        avg_loss_raw = loss.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        avg_loss = avg_loss_raw.replace(0, 1e-9)
        rs = avg_gain / avg_loss
        self._rsi = (100 - (100 / (1 + rs))).values

        hh = h.rolling(self.williams_r_period).max()
        ll = lo.rolling(self.williams_r_period).min()
        denom = (hh - ll).replace(0, 1e-9)
        self._williams_r = (-100 * (hh - c) / denom).values

        tp = (h + lo + c) / 3.0
        sma_tp = tp.rolling(self.cci_period).mean()
        mad = (
            tp.rolling(self.cci_period)
            .apply(lambda x: (np.abs(x - x.mean())).mean(), raw=True)
            .replace(0, 1e-9)
        )
        self._cci = ((tp - sma_tp) / (0.015 * mad)).values

        typical = (h + lo + c) / 3.0
        raw_mf = typical * v
        pos_flow = raw_mf.where(typical > typical.shift(1), 0.0)
        neg_flow = raw_mf.where(typical < typical.shift(1), 0.0)
        pos_sum = pos_flow.rolling(self.mfi_period).sum()
        neg_sum = neg_flow.rolling(self.mfi_period).sum()
        mfr = pos_sum / neg_sum.replace(0, 1e-9)
        self._mfi = (100 - (100 / (1 + mfr))).values

        rsi_series = pd.Series(self._rsi)
        rsi_low = rsi_series.rolling(self.stoch_k_period).min()
        rsi_high = rsi_series.rolling(self.stoch_k_period).max()
        denom_sr = (rsi_high - rsi_low).replace(0, 1e-9)
        fast_k = (rsi_series - rsi_low) / denom_sr * 100
        self._stoch_rsi = fast_k.ewm(span=self.stoch_d_period, adjust=False).mean().values

        bb_mid = c.rolling(self.bb_period).mean()
        bb_stdv = c.rolling(self.bb_period).std()
        self._bb_lower = (bb_mid - self.bb_std * bb_stdv).values
        self._bb_upper = (bb_mid + self.bb_std * bb_stdv).values

        dm_plus = h.diff()
        dm_minus = -lo.diff()
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
        atr_adx = tr.ewm(span=self.adx_period, adjust=False).mean()
        di_plus = (
            100 * dm_plus.ewm(span=self.adx_period, adjust=False).mean() / atr_adx.replace(0, 1e-9)
        )
        di_minus = (
            100 * dm_minus.ewm(span=self.adx_period, adjust=False).mean() / atr_adx.replace(0, 1e-9)
        )
        dx_raw = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, 1e-9)
        self._adx = dx_raw.ewm(span=self.adx_period, adjust=False).mean().values

        min_len = max(
            self.rsi_period,
            self.williams_r_period,
            self.cci_period,
            self.mfi_period,
            self.stoch_k_period,
            self.bb_period,
            self.adx_period,
        )
        self._signal_start = min_len

        self._rsi_signal = np.zeros(n, dtype=np.int8)
        self._wr_signal = np.zeros(n, dtype=np.int8)
        self._cci_signal = np.zeros(n, dtype=np.int8)
        self._mfi_signal = np.zeros(n, dtype=np.int8)
        self._stoch_signal = np.zeros(n, dtype=np.int8)
        self._bb_signal = np.zeros(n, dtype=np.int8)

        for i in range(min_len, n):
            if not np.isnan(self._rsi[i]):
                if self._rsi[i] < self.rsi_oversold:
                    self._rsi_signal[i] = 1
                elif self._rsi[i] > self.rsi_overbought:
                    self._rsi_signal[i] = -1
            if not np.isnan(self._williams_r[i]):
                if self._williams_r[i] < self.williams_oversold:
                    self._wr_signal[i] = 1
                elif self._williams_r[i] > self.williams_overbought:
                    self._wr_signal[i] = -1
            if not np.isnan(self._cci[i]):
                if self._cci[i] < self.cci_oversold:
                    self._cci_signal[i] = 1
                elif self._cci[i] > self.cci_overbought:
                    self._cci_signal[i] = -1
            if not np.isnan(self._mfi[i]):
                if self._mfi[i] < self.mfi_oversold:
                    self._mfi_signal[i] = 1
                elif self._mfi[i] > self.mfi_overbought:
                    self._mfi_signal[i] = -1
            if not np.isnan(self._stoch_rsi[i]):
                if self._stoch_rsi[i] < self.stoch_oversold:
                    self._stoch_signal[i] = 1
                elif self._stoch_rsi[i] > self.stoch_overbought:
                    self._stoch_signal[i] = -1
            if not np.isnan(self._bb_lower[i]) and not np.isnan(self._bb_upper[i]):
                if self._close[i] <= self._bb_lower[i]:
                    self._bb_signal[i] = 1
                elif self._close[i] >= self._bb_upper[i]:
                    self._bb_signal[i] = -1

    def _count_bullish(self, i: int) -> int:
        return sum(
            [
                self._rsi_signal[i] == 1,
                self._wr_signal[i] == 1,
                self._cci_signal[i] == 1,
                self._mfi_signal[i] == 1,
                self._stoch_signal[i] == 1,
                self._bb_signal[i] == 1,
            ]
        )

    def _count_bearish(self, i: int) -> int:
        return sum(
            [
                self._rsi_signal[i] == -1,
                self._wr_signal[i] == -1,
                self._cci_signal[i] == -1,
                self._mfi_signal[i] == -1,
                self._stoch_signal[i] == -1,
                self._bb_signal[i] == -1,
            ]
        )

    def _in_ranging_regime(self, i: int) -> bool:
        if not self.require_ranging:
            return True
        return not np.isnan(self._adx[i]) and self._adx[i] < self.adx_threshold

    def next(self) -> None:
        i = len(self.data) - 1
        if i < self._signal_start:
            return
        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return
        price = self._close[i]

        if self.position and self.position.is_long:
            entry_price = self.trades[-1].entry_price
            bars_held = i - self.trades[-1].entry_bar
            tp_price = entry_price + self.tp_atr_mult * atr_val
            sl_price = entry_price - self.sl_atr_mult * atr_val
            if price >= tp_price or price <= sl_price or bars_held >= self.max_hold_bars:
                self.position.close()
            return

        if not self._in_ranging_regime(i):
            return

        bull_count = self._count_bullish(i)
        bear_count = self._count_bearish(i)

        if bull_count >= self.min_confluence:
            self.buy()
        elif bear_count >= self.min_confluence:
            self.sell()
