# -*- coding: utf-8 -*-
"""
ICT Turtle Soup Strategy

Stop-hunt and false-breakout strategy that exploits failed breakouts.
When price spikes through key support/resistance but fails to sustain,
it traps breakout traders (turtles) and reverses.

Strategy flow:
1. Identify key liquidity levels (session highs/lows, swing points).
2. Wait for price to sweep the liquidity (breakout).
3. Confirm the breakout fails (price reverses back inside range).
4. Enter on the reversal using FVG or order block.
5. Target liquidity on the opposite side.

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


class TurtleSoupStrategy(Strategy):
    """
    ICT Turtle Soup - false breakout trap strategy.

    Enters when price breaks a key level, fails to sustain, and reverses
    back inside the range. Uses FVG for precise entry timing.

    Parameters:
        session_bars: Bars per session for range calculation (default 24)
        breakout_buffer_atr: ATR multiple for breakout confirmation (default 0.3)
        reversal_confirm_bars: Bars to wait for reversal confirmation (default 2)
        trail_stop_atr: ATR multiplier for trailing stop (default 2.0)
        atr_period: ATR period (default 14)
        use_short: Enable short entries (default True)
    """

    session_bars = 24
    breakout_buffer_atr = 0.3
    reversal_confirm_bars = 2
    trail_stop_atr = 2.0
    atr_period = 14
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
        session_high = np.full(n, np.nan)
        session_low = np.full(n, np.nan)
        fvg_signal = np.zeros(n)

        for i in range(n):
            s = max(0, i - self.session_bars + 1)
            session_high[i] = float(np.max(high[s : i + 1]))
            session_low[i] = float(np.min(low[s : i + 1]))

            if i >= 2:
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
        self.fvg = self.I(lambda: pd.Series(fvg_signal, index=close.index), name="FVG")

        self._trail_high: float = 0.0
        self._trail_low: float = float("inf")
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

    def _detect_false_breakout(self, idx: int) -> int:
        """
        Detect false breakout (Turtle Soup).

        Returns 1 (bullish turtle soup = false break low), -1 (bearish turtle
        soup = false break high), or 0 (none).
        """
        if idx < self.session_bars + self.reversal_confirm_bars + 1:
            return 0

        atr_val = self.atr[idx]
        if np.isnan(atr_val) or atr_val <= 0:
            atr_val = float(self.data.Close[idx]) * 0.002

        buffer = self.breakout_buffer_atr * atr_val
        sh = float(self.session_high[idx - self.reversal_confirm_bars])
        sl = float(self.session_low[idx - self.reversal_confirm_bars])

        if np.isnan(sh) or np.isnan(sl):
            return 0

        # Check bars between breakout attempt and now
        broke_high = False
        broke_low = False
        reversed_high = False
        reversed_low = False

        for i in range(max(idx - self.reversal_confirm_bars, 0), idx + 1):
            high_i = float(self.data.High[i])
            low_i = float(self.data.Low[i])
            close_i = float(self.data.Close[i])

            if not broke_high and high_i > sh + buffer:
                broke_high = True
            if not broke_low and low_i < sl - buffer:
                broke_low = True

            # Reversal confirmation: close back inside range after breakout
            if broke_high and close_i < sh:
                reversed_high = True
            if broke_low and close_i > sl:
                reversed_low = True

        if reversed_high and broke_high:
            return -1  # Bearish turtle soup
        if reversed_low and broke_low:
            return 1  # Bullish turtle soup

        return 0

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

        turtle = self._detect_false_breakout(idx)
        fvg = self.fvg[idx]
        close = self.data.Close[idx]

        entry_mult = 1.0
        if idx < len(self._vix_mults):
            entry_mult *= self._vix_mults[idx]
        if idx < len(self._yield_curve_mults):
            entry_mult *= self._yield_curve_mults[idx]
        if entry_mult < 0.15:
            return

        if turtle == 1 and fvg > 0:
            self._trail_high = close
            self._entry_price = close
            self._tp1_hit = False
            self.buy(size=1.0)
        elif turtle == -1 and fvg < 0 and self.use_short:
            self._trail_low = close
            self._entry_price = close
            self._tp1_hit = False
            self.sell(size=1.0)
