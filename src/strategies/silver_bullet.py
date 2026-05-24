# -*- coding: utf-8 -*-
"""
ICT Silver Bullet Strategy

Time-based algorithmic trading model using kill zones and Fair Value Gaps.
Operates during three specific one-hour windows when institutional activity peaks:
- London Open Kill Zone: 02:00-05:00 EST (06:00-09:00 UTC)
- New York AM Kill Zone: 08:00-11:00 EST (12:00-15:00 UTC)
- London Close Kill Zone: 10:00-12:00 EST (14:00-16:00 UTC)

Strategy flow:
1. Wait for a kill zone window to start.
2. Identify liquidity sweep (stops taken).
3. Wait for FVG formation after sweep.
4. Enter on FVG during the kill zone.
5. Target the next liquidity pool.

Integrates with backtesting.py and the plugin registry.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from backtesting import Strategy

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Kill zone definitions in UTC
KILL_ZONES = {
    "asian": {"start": 0, "end": 4},  # 00:00-04:00 UTC
    "london_open": {"start": 6, "end": 9},  # 06:00-09:00 UTC
    "new_york_am": {"start": 12, "end": 15},  # 12:00-15:00 UTC
    "london_close": {"start": 14, "end": 16},  # 14:00-16:00 UTC
}


class SilverBulletStrategy(Strategy):
    """
    ICT Silver Bullet - time-based FVG entry strategy.

    Enters only during kill zone windows when a liquidity sweep is followed
    by a Fair Value Gap formation.

    Parameters:
        kill_zone: Which kill zone to trade (default 'london_open')
        entry_threshold: Minimum score for entry (default 0.5)
        trail_stop_atr: ATR multiplier for trailing stop (default 2.0)
        atr_period: ATR calculation period (default 14)
        sweep_lookback: Bars to look back for sweep detection (default 12)
        use_short: Enable short entries (default True)
    """

    kill_zone = "london_open"
    entry_threshold = 0.5
    trail_stop_atr = 2.0
    atr_period = 14
    sweep_lookback = 12
    use_short = True
    # ── Phase 21 new-tech gates (opt-in only — gates OFF by default) ──
    use_vix_gate: bool = False
    vix_gate_stress_mult: float = 0.30
    vix_gate_elevated_mult: float = 0.75
    use_yield_curve_gate: bool = False
    yield_curve_inversion_mult: float = 0.50
    yield_curve_near_inversion_mult: float = 0.75
    # ── Phase 20 Multi-TP ──
    use_multi_tp: bool = False
    tp1_atr: float = 1.5
    tp1_size: float = 0.5
    move_sl_to_be: bool = True

    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        # ATR calculation
        tr = pd.DataFrame(
            {
                "hl": high - low,
                "hc": abs(high - close.shift(1)),
                "lc": abs(low - close.shift(1)),
            }
        ).max(axis=1)
        atr_series = tr.rolling(self.atr_period).mean()
        self.atr = self.I(lambda: atr_series, name="ATR")

        # Session range (approximate Asia range using lookback)
        n = len(close)
        session_high = np.full(n, np.nan)
        session_low = np.full(n, np.nan)
        fvg_signal = np.zeros(n)

        for i in range(self.sweep_lookback, n):
            s = max(0, i - self.sweep_lookback)
            session_high[i] = float(np.max(high[s:i]))
            session_low[i] = float(np.min(low[s:i]))

            if i >= 2:
                atr_val = atr_series.iloc[i] if not pd.isna(atr_series.iloc[i]) else 0.001
                bar1_high = float(high.iloc[i - 2])
                bar1_low = float(low.iloc[i - 2])
                bar2_close = float(close.iloc[i - 1])
                bar2_open = float(self.data.Open[i - 1])
                bar3_high = float(high.iloc[i])
                bar3_low = float(low.iloc[i])

                bullish_fvg = bar1_high < bar3_low and bar2_close > bar2_open
                bearish_fvg = bar1_low > bar3_high and bar2_close < bar2_open

                if bullish_fvg:
                    fvg_signal[i] = 1
                elif bearish_fvg:
                    fvg_signal[i] = -1

        self.session_high = self.I(
            lambda: pd.Series(session_high, index=close.index), name="SessionHigh"
        )
        self.session_low = self.I(
            lambda: pd.Series(session_low, index=close.index), name="SessionLow"
        )
        self.fvg_signal = self.I(lambda: pd.Series(fvg_signal, index=close.index), name="FVG")

        self._trail_high: float = 0.0
        self._trail_low: float = float("inf")
        self._tp1_hit: bool = False
        self._entry_price: float = 0.0

        # ── Phase 21: New-tech gates ──
        self._init_new_tech_gates()

    def _in_kill_zone(self, idx: int) -> bool:
        """Check if current bar is within the configured kill zone."""
        try:
            bar_time = self.data.index[idx]
            hour = bar_time.hour if hasattr(bar_time, "hour") else bar_time.hour
        except Exception:
            return False

        zone = KILL_ZONES.get(self.kill_zone)
        if zone is None:
            return False

        if zone["start"] <= zone["end"]:
            return zone["start"] <= hour < zone["end"]
        else:
            return hour >= zone["start"] or hour < zone["end"]

    def _detect_sweep(self, idx: int) -> int:
        """Detect liquidity sweep. Returns 1 (bullish), -1 (bearish), 0 (none)."""
        if idx < self.sweep_lookback + 1:
            return 0

        high_i = float(self.data.High[idx])
        low_i = float(self.data.Low[idx])
        close_i = float(self.data.Close[idx])

        sh = float(self.session_high[idx])
        sl = float(self.session_low[idx])

        if np.isnan(sh) or np.isnan(sl):
            return 0

        atr_val = self.atr[idx]
        buffer = 0.3 * atr_val if not np.isnan(atr_val) else 0.0

        # Bullish sweep: breaks above session high then closes below
        if high_i > sh + buffer and close_i < sh:
            return -1  # Bearish reversal after high sweep
        # Bearish sweep: breaks below session low then closes above
        if low_i < sl - buffer and close_i > sl:
            return 1  # Bullish reversal after low sweep

        return 0

    def _init_new_tech_gates(self) -> None:
        """Initialize VIX and yield curve macro regime gates."""
        import logging

        _log = logging.getLogger(__name__)
        n = len(self.data.Close)
        self._vix_mults = np.ones(n, dtype=np.float64)
        self._yield_curve_mults = np.ones(n, dtype=np.float64)

        if self.use_vix_gate:
            try:
                from src.signals.vix_regime_gate import VixRegimeGate

                gate = VixRegimeGate(
                    stress_mult=self.vix_gate_stress_mult,
                    elevated_mult=self.vix_gate_elevated_mult,
                )
                gate.fit(start=str(self.data.index[0].date()))
                for i in range(n):
                    d = self.data.index[i]
                    if hasattr(d, "date"):
                        d = d.date()
                    self._vix_mults[i] = gate.multiplier(date=d)
            except Exception:
                _log.debug("VIX gate init failed", exc_info=True)

        if self.use_yield_curve_gate:
            try:
                from src.signals.yield_curve_gate import YieldCurveGate

                gate = YieldCurveGate(
                    inversion_mult=self.yield_curve_inversion_mult,
                    near_inversion_mult=self.yield_curve_near_inversion_mult,
                )
                gate.fit(start=str(self.data.index[0].date()))
                for i in range(n):
                    d = self.data.index[i]
                    if hasattr(d, "date"):
                        d = d.date()
                    self._yield_curve_mults[i] = gate.multiplier(date=d)
            except Exception:
                _log.debug("Yield curve gate init failed", exc_info=True)

    def next(self):
        idx = len(self.data) - 1

        if self.position:
            close = self.data.Close[idx]
            atr_val = self.atr[idx]
            if not np.isnan(atr_val) and atr_val > 0:
                if self.position.is_long:
                    self._trail_high = max(self._trail_high, close)
                    trail_sl = self._trail_high - self.trail_stop_atr * atr_val
                    if self.use_multi_tp and not self._tp1_hit:
                        tp1_price = self._entry_price + self.tp1_atr * atr_val
                        if close >= tp1_price:
                            self.position.close(portion=self.tp1_size)
                            self._tp1_hit = True
                            if self.move_sl_to_be:
                                self._trail_high = self._entry_price
                                trail_sl = self._entry_price
                        elif close <= trail_sl:
                            self.position.close()
                            self._trail_high = 0.0
                    elif close <= trail_sl:
                        self.position.close()
                        self._trail_high = 0.0
                elif self.position.is_short:
                    self._trail_low = min(self._trail_low, close)
                    trail_sl = self._trail_low + self.trail_stop_atr * atr_val
                    if self.use_multi_tp and not self._tp1_hit:
                        tp1_price = self._entry_price - self.tp1_atr * atr_val
                        if close <= tp1_price:
                            self.position.close(portion=self.tp1_size)
                            self._tp1_hit = True
                            if self.move_sl_to_be:
                                self._trail_low = self._entry_price
                                trail_sl = self._entry_price
                        elif close >= trail_sl:
                            self.position.close()
                            self._trail_low = float("inf")
                    elif close >= trail_sl:
                        self.position.close()
                        self._trail_low = float("inf")
            return

        # Only trade during kill zone
        if not self._in_kill_zone(idx):
            return

        # Entry logic
        sweep = self._detect_sweep(idx)
        fvg = self.fvg_signal[idx]

        sweep_recent = False
        for i in range(max(idx - 3, 0), idx + 1):
            s = self._detect_sweep(i)
            if s == -1:
                sweep_recent = True
                sweep_dir = -1
                break
            elif s == 1:
                sweep_recent = True
                sweep_dir = 1
                break

        if not sweep_recent:
            return

        close = self.data.Close[idx]
        # ── Phase 21: Apply macro regime gates ──
        entry_mult = 1.0
        if idx < len(self._vix_mults):
            entry_mult *= self._vix_mults[idx]
        if idx < len(self._yield_curve_mults):
            entry_mult *= self._yield_curve_mults[idx]
        if entry_mult < 0.15:
            return  # Block entry only in combined stress+inversion (0.30x0.50=0.15)
        if sweep_dir == 1 and fvg > 0:
            self._trail_high = close
            self._entry_price = close
            self._tp1_hit = False
            self.buy(size=1.0)
        elif sweep_dir == -1 and fvg < 0 and self.use_short:
            self._trail_low = close
            self._entry_price = close
            self._tp1_hit = False
            self.sell(size=1.0)
