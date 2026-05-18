"""Regime-based strategy router.

Routes between trend-following and mean-reversion strategies based on
ADX regime detection. Uses precomputed signals from both sub-strategies
and selects the appropriate one based on the current market regime.

Regime mapping:
- Trending  (ADX > 25): Route to trend-following signals
- Ranging   (ADX < 20): Route to mean-reversion signals
- Transition (ADX 20-25): Hold existing positions, skip new entries

Usage:
    from backtesting import Backtest
    bt = Backtest(df, RegimeRouterStrategy, cash=10_000, commission=0.001)
    stats = bt.run()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy


class RegimeRouterStrategy(Strategy):
    """Routes entry signals between trend-following and mean-reversion.

    Trend-following (ADX > 25): Uses a composite signal from RSI(14),
    Williams %R(14), CCI(20), and price relative to 50MA. ATR trailing stop.

    Mean-reversion (ADX < 20): Uses composite signal from RSI(14),
    Williams %R(14), CCI(20), MFI(14), StochRSI, Bollinger Bands.
    Fixed TP, wider SL, time-based exit.

    Both signal sets are precomputed at init. The router selects which
    signal path to use based on current ADX value.
    """

    # ── Common ──
    atr_period: int = 14
    adx_period: int = 14
    adx_trending: int = 25
    adx_ranging: int = 20

    # ── Trend-following params ──
    trend_rsi_period: int = 14
    trend_rsi_oversold: int = 30
    trend_rsi_overbought: int = 70
    trend_ma_period: int = 50
    trend_trail_atr: float = 3.0

    # ── Mean-reversion params ──
    mr_rsi_period: int = 14
    mr_rsi_oversold: int = 30
    mr_rsi_overbought: int = 70
    mr_wr_period: int = 14
    mr_wr_oversold: int = -80
    mr_wr_overbought: int = -20
    mr_cci_period: int = 20
    mr_cci_oversold: int = -100
    mr_cci_overbought: int = 100
    mr_mfi_period: int = 14
    mr_mfi_oversold: int = 20
    mr_mfi_overbought: int = 80
    mr_stoch_k: int = 14
    mr_stoch_d: int = 3
    mr_stoch_os: int = 20
    mr_stoch_ob: int = 80
    mr_bb_period: int = 20
    mr_bb_std: float = 2.0
    mr_tp_atr: float = 2.0
    mr_sl_atr: float = 3.0
    mr_max_hold: int = 10
    mr_min_confluence: int = 2

    risk_pct: float = 0.01

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

        # ── ADX for regime detection ──
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

        # ── RSI(14) shared ──
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean().replace(0, 1e-9)
        rs = avg_gain / avg_loss
        self._rsi = (100 - (100 / (1 + rs))).values

        # ── Williams %R(14) ──
        hh = h.rolling(self.mr_wr_period).max()
        ll = lo.rolling(self.mr_wr_period).min()
        self._wr = (-100 * (hh - c) / (hh - ll).replace(0, 1e-9)).values

        # ── CCI(20) ──
        tp = (h + lo + c) / 3.0
        sma_tp = tp.rolling(self.mr_cci_period).mean()
        mad = (
            tp.rolling(self.mr_cci_period)
            .apply(lambda x: (np.abs(x - x.mean())).mean(), raw=True)
            .replace(0, 1e-9)
        )
        self._cci = ((tp - sma_tp) / (0.015 * mad)).values

        # ── MFI(14) ──
        typical = (h + lo + c) / 3.0
        raw_mf = typical * v
        pos_flow = raw_mf.where(typical > typical.shift(1), 0.0)
        neg_flow = raw_mf.where(typical < typical.shift(1), 0.0)
        pos_sum = pos_flow.rolling(self.mr_mfi_period).sum()
        neg_sum = neg_flow.rolling(self.mr_mfi_period).sum()
        mfr = pos_sum / neg_sum.replace(0, 1e-9)
        self._mfi = (100 - (100 / (1 + mfr))).values

        # ── StochRSI ──
        rsi_s = pd.Series(self._rsi)
        rsi_low = rsi_s.rolling(self.mr_stoch_k).min()
        rsi_high = rsi_s.rolling(self.mr_stoch_k).max()
        fast_k = (rsi_s - rsi_low) / (rsi_high - rsi_low).replace(0, 1e-9) * 100
        self._stoch_rsi = fast_k.ewm(span=self.mr_stoch_d, adjust=False).mean().values

        # ── Bollinger Bands ──
        bb_mid = c.rolling(self.mr_bb_period).mean()
        bb_stdv = c.rolling(self.mr_bb_period).std()
        self._bb_lo = (bb_mid - self.mr_bb_std * bb_stdv).values
        self._bb_hi = (bb_mid + self.mr_bb_std * bb_stdv).values

        # ── 50MA for trend ──
        self._ma50 = c.rolling(self.trend_ma_period).mean().values

        # ── Trail stop state ──
        self._trail_high = 0.0

        self._start_idx = max(50, self.trend_ma_period, self.mr_bb_period)
        self._regime: list[str] = [""] * n

    def _mr_bull_count(self, i: int) -> int:
        c = 0
        if not np.isnan(self._rsi[i]) and self._rsi[i] < self.mr_rsi_oversold:
            c += 1
        if not np.isnan(self._wr[i]) and self._wr[i] < self.mr_wr_oversold:
            c += 1
        if not np.isnan(self._cci[i]) and self._cci[i] < self.mr_cci_oversold:
            c += 1
        if not np.isnan(self._mfi[i]) and self._mfi[i] < self.mr_mfi_oversold:
            c += 1
        if not np.isnan(self._stoch_rsi[i]) and self._stoch_rsi[i] < self.mr_stoch_os:
            c += 1
        if not np.isnan(self._bb_lo[i]) and self._close[i] <= self._bb_lo[i]:
            c += 1
        return c

    def _trend_bull_strength(self, i: int) -> int:
        s = 0
        if not np.isnan(self._rsi[i]) and self._rsi[i] < self.trend_rsi_oversold:
            s += 1
        if not np.isnan(self._ma50[i]) and self._close[i] > self._ma50[i]:
            s += 1
        if not np.isnan(self._wr[i]) and self._wr[i] < -80:
            s += 1
        if not np.isnan(self._cci[i]) and self._cci[i] < -100:
            s += 1
        return s

    def next(self) -> None:
        i = len(self.data) - 1
        if i < self._start_idx:
            return
        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return
        price = self._close[i]
        adx_val = self._adx[i]

        regime = "transition"
        if not np.isnan(adx_val):
            if adx_val > self.adx_trending:
                regime = "trending"
            elif adx_val < self.adx_ranging:
                regime = "ranging"
        self._regime[i] = regime

        # ── Position management ──
        if self.position and self.position.is_long:
            if regime == "ranging":
                entry_price = self.trades[-1].entry_price
                bars_held = i - self.trades[-1].entry_bar
                tp_price = entry_price + self.mr_tp_atr * atr_val
                sl_price = entry_price - self.mr_sl_atr * atr_val
                if price >= tp_price or price <= sl_price or bars_held >= self.mr_max_hold:
                    self.position.close()
            else:
                if price > self._trail_high:
                    self._trail_high = price
                trail_sl = self._trail_high - self.trend_trail_atr * atr_val
                if price <= trail_sl:
                    self.position.close()
            return

        # ── Entry logic ──
        if regime == "trending":
            bull_strength = self._trend_bull_strength(i)
            if bull_strength >= 2:
                self._trail_high = price
                self.buy()
        elif regime == "ranging":
            bull_count = self._mr_bull_count(i)
            if bull_count >= self.mr_min_confluence:
                self.buy()
