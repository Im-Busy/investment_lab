"""
SMC Sweep-Only Strategy — minimal, no composite scoring.

Based on the finding that raw liquidity sweeps have +1.89% standalone PnL
(42.4% WR, 1.39 W/L) and all composite scoring additions degrade this edge.

Architecture:
  - Detects liquidity sweeps on swing pivot levels (not session ranges)
  - Enter on sweep bar close, exit on ATR trailing stop
  - No MSL/MSH, no BOS, no FVG, no Order Blocks, no gates
  - Volume confirmation as optional boost only
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


class SMCSweepOnlyStrategy(Strategy):
    """SMC sweep-only strategy: enter on sweep bar, exit on ATR trail.

    Parameters:
        entry_threshold: Minimum sweep score to enter (default 0.30)
        trail_stop_atr: ATR multiplier for trailing stop (default 2.0)
        atr_period: ATR calculation period (default 14)
        sweep_lookback: Swing pivot lookback for liquidity levels (default 10)
        sweep_buffer_mult: ATR buffer multiplier for sweep detection (default 0.3)
        use_multi_tp: Enable multi-TP exit (default True)
        tp1_atr: TP1 distance in ATR multiples (default 1.5)
        tp1_size: Portion to close at TP1 (default 0.5)
        move_sl_to_be: Move SL to breakeven after TP1 (default True)
        volume_min: Min relative volume (vs 20-bar avg) for entry (default 0.0=off)
        use_short: Enable short entries (default False)
    """

    entry_threshold: float = 0.30
    trail_stop_atr: float = 2.0
    atr_period: int = 14
    sweep_lookback: int = 10
    sweep_buffer_mult: float = 0.3
    use_multi_tp: bool = True
    tp1_atr: float = 1.5
    tp1_size: float = 0.5
    move_sl_to_be: bool = True
    volume_min: float = 0.0
    use_short: bool = False

    _strategy_ref: Optional[list] = None

    # ── init() ──────────────────────────────────────────────────

    def init(self) -> None:
        if self._strategy_ref is not None:
            self._strategy_ref.append(self)

        self._df = self._build_df()
        self._n_bars = len(self._df)

        self._atr = self._compute_atr()
        self._sweep_signals = self._detect_sweeps()
        self._vol_mult = self._compute_relative_volume()

        self._trail_high: float = 0.0
        self._trail_low: float = float("inf")
        self._entry_price: float = 0.0
        self._tp1_hit: bool = False

        n_sweeps = int(np.sum(self._sweep_signals != 0))
        n_bull = int(np.sum(self._sweep_signals > 0))
        n_bear = int(np.sum(self._sweep_signals < 0))
        logger.info(
            "SMCSweepOnly: %d bars, sweeps=%d (bull=%d, bear=%d)",
            self._n_bars,
            n_sweeps,
            n_bull,
            n_bear,
        )

    def _build_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Open": self.data.Open.s,
                "High": self.data.High.s,
                "Low": self.data.Low.s,
                "Close": self.data.Close.s,
                "Volume": self.data.Volume.s,
            },
            index=self.data.index,
        )

    def _compute_atr(self) -> np.ndarray:
        high = self._df["High"].values
        low = self._df["Low"].values
        close = self._df["Close"].values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - prev_close),
                np.abs(low - prev_close),
            ),
        )

        n = len(close)
        atr = np.zeros(n)
        period = self.atr_period
        atr[:period] = np.mean(tr[:period]) if period <= n else np.mean(tr)
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

    def _detect_sweeps(self) -> np.ndarray:
        """Detect liquidity sweeps on swing pivot levels.

        Uses rolling swing high/low (pivot_lookback on each side) as
        liquidity levels. A sweep occurs when price breaks beyond the
        level by >buffer*ATR and closes back inside.

        Returns:
            Array: +1=bullish sweep, -1=bearish sweep, 0=no sweep.
        """
        high = self._df["High"].values
        low = self._df["Low"].values
        close = self._df["Close"].values
        n = len(close)
        lb = self.sweep_lookback

        signals = np.zeros(n, dtype=np.int8)

        # Find swing pivots (higher high / lower low vs lb bars on each side)
        swing_highs = np.zeros(n, dtype=bool)
        swing_lows = np.zeros(n, dtype=bool)

        for i in range(lb, n - lb):
            if high[i] == np.max(high[i - lb : i + lb + 1]):
                swing_highs[i] = True
            if low[i] == np.min(low[i - lb : i + lb + 1]):
                swing_lows[i] = True

        # Detect sweeps: price breaks pivot by buffer*ATR, then reverses
        for i in range(lb * 2, n):
            atr_val = self._atr[i] if self._atr[i] > 0 else close[i] * 0.01
            buffer = self.sweep_buffer_mult * atr_val

            # Look back for nearest swing high/low (within lb*2 bars)
            for j in range(i - lb * 2, i - 1):
                if swing_highs[j] and high[i] > high[j] + buffer:
                    # Bearish sweep: price broke above swing high, closed below
                    if close[i] < high[j]:
                        signals[i] = -1
                    break
                if swing_lows[j] and low[i] < low[j] - buffer:
                    # Bullish sweep: price broke below swing low, closed above
                    if close[i] > low[j]:
                        signals[i] = 1
                    break

            if signals[i] != 0:
                continue

        return signals

    def _compute_relative_volume(self) -> np.ndarray:
        vol = self._df["Volume"].values
        n = len(vol)
        rel_vol = np.ones(n)
        window = 20
        for i in range(window, n):
            avg = np.mean(vol[i - window : i])
            if avg > 0:
                rel_vol[i] = min(max(vol[i] / avg, 0.5), 3.0)
        return rel_vol

    # ── Scoring ────────────────────────────────────────

    def _compute_score(self, idx: int) -> float:
        """Simple score: 1.0 per sweep, volume boost, tanh normalized."""
        if idx < self.sweep_lookback * 2:
            return 0.0

        sweep = self._sweep_signals[idx]
        if sweep == 0:
            return 0.0

        score = float(sweep)

        if self.volume_min > 0 and idx < self._n_bars:
            if self._vol_mult[idx] < self.volume_min:
                score *= 0.5

        return float(np.tanh(score * 2.0))

    # ── Sizing ─────────────────────────────────────────

    def _calculate_size(self, price: float, atr: float) -> float:
        risk_pct = 0.01
        risk_capital = float(self.equity) * risk_pct
        risk_distance = self.trail_stop_atr * atr
        if risk_distance <= 0:
            risk_distance = price * 0.01
        pos_size = risk_capital / risk_distance
        min_size = float(self.equity) * 0.001 / max(price, 1e-10)
        return max(pos_size, min_size)

    # ── Bar-by-bar ─────────────────────────────────────

    def next(self) -> None:
        idx = len(self.data) - 1
        if idx < self.sweep_lookback * 2 or idx >= self._n_bars:
            return

        score = self._compute_score(idx)
        current_close = float(self.data.Close[-1])
        atr = float(self._atr[idx]) if idx < self._n_bars else current_close * 0.02
        if atr <= 0:
            atr = current_close * 0.02

        if self.position:
            if self.position.is_long:
                self._trail_high = max(self._trail_high, current_close)
                trail_sl = self._trail_high - self.trail_stop_atr * atr

                if self.use_multi_tp and not self._tp1_hit:
                    tp1_price = self._entry_price + self.tp1_atr * atr
                    if current_close >= tp1_price:
                        self.position.close(portion=self.tp1_size)
                        self._tp1_hit = True
                        if self.move_sl_to_be:
                            self._trail_high = self._entry_price
                            trail_sl = self._entry_price
                        return
                    elif current_close <= trail_sl:
                        self.position.close()
                        self._trail_high = 0.0
                        return
                else:
                    if current_close <= trail_sl:
                        self.position.close()
                        self._trail_high = 0.0
                        return

                if score < -self.entry_threshold:
                    self.position.close()
                    self._trail_high = 0.0
            else:
                self._trail_low = min(self._trail_low, current_close)
                trail_sl = self._trail_low + self.trail_stop_atr * atr

                if self.use_multi_tp and not self._tp1_hit:
                    tp1_price = self._entry_price - self.tp1_atr * atr
                    if current_close <= tp1_price:
                        self.position.close(portion=self.tp1_size)
                        self._tp1_hit = True
                        if self.move_sl_to_be:
                            self._trail_low = self._entry_price
                            trail_sl = self._entry_price
                        return
                    elif current_close >= trail_sl:
                        self.position.close()
                        self._trail_low = float("inf")
                        return
                else:
                    if current_close >= trail_sl:
                        self.position.close()
                        self._trail_low = float("inf")
                        return

                if score > self.entry_threshold:
                    self.position.close()
                    self._trail_low = float("inf")
        else:
            if score >= self.entry_threshold:
                size = self._calculate_size(current_close, atr)
                self.buy(size=min(size / current_close, 0.95))
                self._trail_high = current_close
                self._trail_low = float("inf")
                self._entry_price = current_close
                self._tp1_hit = False
            elif self.use_short and score <= -self.entry_threshold:
                size = self._calculate_size(current_close, atr)
                self.sell(size=min(size / current_close, 0.95))
                self._trail_low = current_close
                self._trail_high = 0.0
                self._entry_price = current_close
                self._tp1_hit = False
