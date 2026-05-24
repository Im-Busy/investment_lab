# -*- coding: utf-8 -*-
"""
ICT Cameron's Model Strategy

Three-component strategy based on:
1. Draw on Liquidity: Where price is heading to collect liquidity.
2. Stop Rate: Level where stop losses cluster (liquidity pool to sweep).
3. Entry: FVG or order block after the sweep.

Strategy flow:
1. Identify the draw on liquidity (next significant level).
2. Locate the stop rate (liquidity pool to be swept).
3. Wait for the stop rate to be swept.
4. Enter on FVG/order block after the sweep.
5. Target the draw on liquidity.

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


class CameronModelStrategy(Strategy):
    """
    Cameron's Model - Draw on Liquidity + Stop Rate + FVG Entry.

    Parameters:
        swing_lookback: Bars for swing high/low detection (default 50)
        atr_period: ATR period (default 14)
        sweep_buffer_atr: ATR multiple for sweep confirmation (default 0.3)
        trail_stop_atr: ATR multiplier for trailing stop (default 2.0)
        min_swing_range_atr: Minimum swing range as ATR multiple (default 0.5)
        use_short: Enable short entries (default True)
    """

    swing_lookback = 50
    atr_period = 14
    sweep_buffer_atr = 0.3
    trail_stop_atr = 2.0
    min_swing_range_atr = 0.5
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

        # ATR
        tr = pd.DataFrame(
            {
                "hl": high - low,
                "hc": abs(high - close.shift(1)),
                "lc": abs(low - close.shift(1)),
            }
        ).max(axis=1)
        self.atr = self.I(lambda: tr.rolling(self.atr_period).mean(), name="ATR")

        n = len(close)

        # Swing highs and lows
        sw_len = self.swing_lookback
        swing_high = np.zeros(n)
        swing_low = np.zeros(n)
        swing_high_level = np.full(n, np.nan)
        swing_low_level = np.full(n, np.nan)

        for i in range(sw_len, n - sw_len):
            window = high.iloc[i - sw_len : i + sw_len + 1]
            if high.iloc[i] == window.max():
                swing_high[i] = 1
                swing_high_level[i] = float(high.iloc[i])

        for i in range(sw_len, n - sw_len):
            window = low.iloc[i - sw_len : i + sw_len + 1]
            if low.iloc[i] == window.min():
                swing_low[i] = 1
                swing_low_level[i] = float(low.iloc[i])

        # FVG detection
        fvg_signal = np.zeros(n)
        fvg_top = np.full(n, np.nan)
        fvg_bottom = np.full(n, np.nan)

        for i in range(2, n):
            bar1_high = float(high.iloc[i - 2])
            bar1_low = float(low.iloc[i - 2])
            bar2_close = float(close.iloc[i - 1])
            bar2_open = float(self.data.Open[i - 1])
            bar3_high = float(high.iloc[i])
            bar3_low = float(low.iloc[i])

            if bar1_high < bar3_low and bar2_close > bar2_open:
                fvg_signal[i] = 1
                fvg_top[i] = bar3_low
                fvg_bottom[i] = bar1_high
            elif bar1_low > bar3_high and bar2_close < bar2_open:
                fvg_signal[i] = -1
                fvg_top[i] = bar1_low
                fvg_bottom[i] = bar3_high

        self.swing_high = self.I(lambda: pd.Series(swing_high, index=close.index), name="SwingHigh")
        self.swing_low = self.I(lambda: pd.Series(swing_low, index=close.index), name="SwingLow")
        self.swing_high_level = self.I(
            lambda: pd.Series(swing_high_level, index=close.index), name="SwingHighLevel"
        )
        self.swing_low_level = self.I(
            lambda: pd.Series(swing_low_level, index=close.index), name="SwingLowLevel"
        )
        self.fvg = self.I(lambda: pd.Series(fvg_signal, index=close.index), name="FVG")
        self.fvg_top = self.I(lambda: pd.Series(fvg_top, index=close.index), name="FVGTop")
        self.fvg_bottom = self.I(lambda: pd.Series(fvg_bottom, index=close.index), name="FVGBottom")

        self._trail_high: float = 0.0
        self._trail_low: float = float("inf")
        self._tp: float = 0.0
        self._tp1_hit: bool = False
        self._entry_price: float = 0.0
        self._init_new_tech_gates()

    def _init_new_tech_gates(self) -> None:
        import logging

        _log = logging.getLogger(__name__)
        n = len(self.data.Close)
        self._vix_mults = np.ones(n, dtype=np.float64)
        self._yield_curve_mults = np.ones(n, dtype=np.float64)
        if self.use_vix_gate:
            try:
                from src.signals.vix_regime_gate import VixRegimeGate

                gate = VixRegimeGate(
                    stress_mult=self.vix_gate_stress_mult, elevated_mult=self.vix_gate_elevated_mult
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

    def _find_draw_on_liquidity(self, idx: int) -> tuple:
        """Find the draw on liquidity (target) and stop rate.

        Returns (draw_level, stop_rate, direction).
        direction: 1 for bullish (draw above, sweep below), -1 for bearish.
        """
        if idx < self.swing_lookback:
            return None, None, 0

        atr_val = self.atr[idx]
        if np.isnan(atr_val) or atr_val <= 0:
            atr_val = float(self.data.Close[idx]) * 0.005

        current_price = float(self.data.Close[idx])

        # Find recent swing highs and lows
        recent_swing_high = None
        recent_swing_low = None
        recent_swing_high_idx = -1
        recent_swing_low_idx = -1

        for i in range(max(0, idx - 200), idx):
            if self.swing_high[i] == 1:
                level = self.swing_high_level[i]
                if not np.isnan(level):
                    recent_swing_high = level
                    recent_swing_high_idx = i
            if self.swing_low[i] == 1:
                level = self.swing_low_level[i]
                if not np.isnan(level):
                    recent_swing_low = level
                    recent_swing_low_idx = i

        if recent_swing_high is None or recent_swing_low is None:
            return None, None, 0

        # Determine draw direction
        if recent_swing_high_idx > recent_swing_low_idx:
            # Most recent is swing high → draw is lower (bearish)
            return recent_swing_low, recent_swing_high, -1
        else:
            # Most recent is swing low → draw is higher (bullish)
            return recent_swing_high, recent_swing_low, 1

    def _detect_sweep_of_stop_rate(self, idx: int, stop_rate: float, direction: int) -> bool:
        """Check if stop rate has been swept in the last few bars."""
        if stop_rate is None:
            return False

        atr_val = self.atr[idx]
        if np.isnan(atr_val) or atr_val <= 0:
            atr_val = float(self.data.Close[idx]) * 0.005
        buffer = self.sweep_buffer_atr * atr_val

        for i in range(max(0, idx - 5), idx + 1):
            high_i = float(self.data.High[i])
            low_i = float(self.data.Low[i])
            close_i = float(self.data.Close[i])

            if direction == 1:
                # Bullish: sweep below stop_rate then reverse
                if low_i < stop_rate - buffer and close_i > stop_rate:
                    return True
            else:
                # Bearish: sweep above stop_rate then reverse
                if high_i > stop_rate + buffer and close_i < stop_rate:
                    return True

        return False

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
                        elif self._tp > 0 and close >= self._tp:
                            self.position.close()
                            self._trail_high = 0.0
                            self._tp = 0.0
                        elif close <= trail_sl:
                            self.position.close()
                            self._trail_high = 0.0
                            self._tp = 0.0
                    elif self._tp > 0 and close >= self._tp:
                        self.position.close()
                        self._trail_high = 0.0
                        self._tp = 0.0
                    elif close <= trail_sl:
                        self.position.close()
                        self._trail_high = 0.0
                        self._tp = 0.0
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
                        elif self._tp > 0 and close <= self._tp:
                            self.position.close()
                            self._trail_low = float("inf")
                            self._tp = 0.0
                        elif close >= trail_sl:
                            self.position.close()
                            self._trail_low = float("inf")
                            self._tp = 0.0
                    elif self._tp > 0 and close <= self._tp:
                        self.position.close()
                        self._trail_low = float("inf")
                        self._tp = 0.0
                    elif close >= trail_sl:
                        self.position.close()
                        self._trail_low = float("inf")
                        self._tp = 0.0
            return

        draw_level, stop_rate, direction = self._find_draw_on_liquidity(idx)

        if draw_level is None or direction == 0:
            return

        sweep = self._detect_sweep_of_stop_rate(idx, stop_rate, direction)
        if not sweep:
            return

        fvg = self.fvg[idx]
        close = self.data.Close[idx]

        entry_mult = 1.0
        if idx < len(self._vix_mults):
            entry_mult *= self._vix_mults[idx]
        if idx < len(self._yield_curve_mults):
            entry_mult *= self._yield_curve_mults[idx]
        if entry_mult < 0.15:
            return

        if direction == 1 and fvg > 0:
            self._trail_high = close
            self._tp = draw_level
            self._entry_price = close
            self._tp1_hit = False
            self.buy(size=1.0)
        elif direction == -1 and fvg < 0 and self.use_short:
            self._trail_low = close
            self._tp = draw_level
            self._entry_price = close
            self._tp1_hit = False
            self.sell(size=1.0)
