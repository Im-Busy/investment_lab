"""
SMC Intraday Reversal Strategy for backtesting.py

Unified Smart Money Concepts strategy merging signal detection from
smc_reversal.py and backtesting.py integration from smc_reversal_bt.py.

Architecture:
  - All signals precomputed in init() (no per-bar detection)
  - Component scoring: sweep, MSL/MSH, BOS/CHOCH, FVG proximity, order blocks
  - Multiplicative gate chain: vol gate, session time gate, crash gate
  - Volume pressure features: BOP, up/down/rup/rdown pressure
  - Intraday crash factors: DTURN, NCSKEW, TVOL
  - ATR trailing stop + optional multi-TP exit (from RulesFirst)
  - Tanh-normalized composite score in [-1, +1]

Phase 5 additions (2026-05-19):
  T5.1: Multiplicative gate chain (vol gate, session time gate)
  T5.2: Tick direction volume pressure features (BOP, up/down pressure)
  T5.3: Intraday crash/risk factors (DTURN, NCSKEW, TVOL)
  T5.4: IR weighting infrastructure for SMC components
  Order block detection + scoring

Phase 6 additions (2026-05-20):
  T6.1: Breaker Block detector (failed OB → opposite polarity zone)
  T6.2: Mitigation Block detector (mitigated swing → reverse polarity)
  T6.3: Rejection Block detector (structural rejection at OB/FVG)
  T6.4: All three detectors integrated into scoring with toggle params

Usage:
  from backtesting import Backtest
  from src.strategies.smc_strategy import SMCStrategy

  bt = Backtest(df, SMCStrategy, cash=10_000, commission=0.001)
  stats = bt.run(entry_threshold=0.55, trail_stop_atr=3.0)
  bt.plot()
"""

from __future__ import annotations

import logging
import os
import sys
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from backtesting import Strategy

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def _import_smc_quietly():
    """Import smartmoneyconcepts without printing the star banner."""
    import sys
    import io

    old_stdout = sys.stdout
    try:
        sys.stdout = io.StringIO()
        from smartmoneyconcepts import smc as smc_lib
    finally:
        sys.stdout = old_stdout
    return smc_lib


SMC_COMPONENT_WEIGHTS: dict[str, float] = {
    "sweep_reversal": 1.5,
    "breaker_block": 0.85,
    "mss_bos_choch": 0.65,
    "mitigation_block": 0.45,
    "fvg_proximity": 0.55,
    "rejection_block": 0.55,
    "msl_msh": 0.15,
    "bos_choch": 0.05,
    "order_block": 0.10,
}


class SMCStrategy(Strategy):
    """SMC intraday reversal strategy with precomputed vectorized signals.

    Parameters:
        entry_threshold: Minimum score to enter long (default 0.55)
        exit_threshold: Score drop threshold to exit (default 0.30)
        trail_stop_atr: ATR multiplier for trailing stop (default 3.0)
        session_bars: Bars per session range (default 24 for hourly)
        atr_period: ATR calculation period (default 14)
        confluence_bonus: Boost for multi-component agreement (default 0.10)
        volume_confirm: Enable volume confirmation (default True)
        use_multi_tp: Enable multi-TP exit (default True)
        tp1_atr: TP1 distance in ATR multiples (default 1.5)
        tp1_size: Portion to close at TP1 (default 0.5)
        move_sl_to_be: Move SL to breakeven after TP1 (default True)
        use_short: Enable short entries (default False)
        sweep_buffer_mult: ATR multiplier for sweep detection (default 0.5)
        fvg_proximity_mult: ATR multiplier for FVG entry zone (default 2.0)
        use_vol_gate: Enable volatility gate (default False)
        use_session_gate: Enable session time gate (default False)
        use_crash_gate: Enable crash factor gate (default False)
        use_volume_pressure: Enable tick direction volume features (default False)

        use_order_blocks: Enable order block detection (default False)

        use_breaker_blocks: Enable breaker block detection (default False)

        use_mitigation_blocks: Enable mitigation block detection (default False)

        use_rejection_blocks: Enable rejection block detection (default False)
        use_dow_gate: Enable day-of-week gate (default False)
        use_90min_cycle: Enable 90-minute cycle sensitivity (default False)
        use_frankfurt_gate: Enable Frankfurt fake move gate (default False)
        use_smc_sessions: Enable smartmoneyconcepts sessions() killzone detection (default False)
        use_smc_retrace: Enable smartmoneyconcepts retracements() tracking (default False)
        session_trade_start: UTC hour to start trading (default 0=00:00)
        session_trade_end: UTC hour to stop trading (default 24=always)
        use_killzone_gate: Enable killzone-only trading (default True)
        min_confluence: Minimum components agreeing for entry (default 0 = no gating)
        htf_ema_period: HTF daily EMA period for trend bias (default 50)
        crypto_mode: Auto-adjust gate thresholds for crypto volatility (default False)
        use_judas_swing: Enable Judas Swing gate (default False)
        judas_swing_atr_mult: ATR mult for sweep depth (default 0.3)
        judas_swing_displacement_mult: ATR mult for reversal (default 1.0)
        use_po3_gate: Enable PO3/AMD phase gate (default False)
        use_ote_confluence: Enable OTE Fibonacci confluence (default False)
        use_cisd: Enable CISD confirmation (default False)
        use_crt: Enable Candle Range Theory (default False)
        use_smt: Enable SMT divergence (default False)
        use_sd_zones: Enable S&D zone patterns (default False)
        use_ob_fvg_colocation: Enable OB+FVG colocation boost (default False)
        use_unicorn: Enable Unicorn pattern detection (default False)
        use_poi_grading: Enable POI 4-criteria grading (default False)
        use_daily_loss_limit: Enable daily loss limit (default False)
        daily_loss_limit: Max daily loss as fraction (default 0.03)
        max_risk_pct: Max risk per trade (default 0.01)
        use_trade_plan: Trade plan type (""/smc/ict/hybrid) (default "")
        trade_plan_strictness: Fraction of checks needed (default 1.0)
        instrument_class: Market type (auto/forex/crypto/stocks/gold) (default "auto")
        use_sfp: Enable Swing Failure Pattern (default False)
        use_vix_gate: Enable VIX regime gate (default True)
        vix_gate_stress_mult: VIX stress regime score multiplier (default 0.30)
        vix_gate_elevated_mult: VIX elevated regime score multiplier (default 0.75)
        use_yield_curve_gate: Enable yield curve macro gate (default True)
        yield_curve_inversion_mult: Inverted yield curve score multiplier (default 0.50)
        yield_curve_near_inversion_mult: Near-inversion score multiplier (default 0.75)
    """

    entry_threshold: float = 0.55
    exit_threshold: float = 0.30
    trail_stop_atr: float = 3.0
    session_bars: int = 24
    atr_period: int = 14
    confluence_bonus: float = 0.10
    volume_confirm: bool = True
    use_multi_tp: bool = True
    tp1_atr: float = 1.5
    tp1_size: float = 0.5
    move_sl_to_be: bool = True
    use_short: bool = False
    sweep_buffer_mult: float = 0.5
    fvg_proximity_mult: float = 2.0
    use_vol_gate: bool = False
    use_session_gate: bool = False
    use_crash_gate: bool = False
    use_volume_pressure: bool = False
    use_order_blocks: bool = False
    use_breaker_blocks: bool = False
    use_mitigation_blocks: bool = False
    use_rejection_blocks: bool = False
    use_dow_gate: bool = False
    use_90min_cycle: bool = False
    use_frankfurt_gate: bool = False
    use_smc_sessions: bool = True
    use_smc_retrace: bool = False
    session_trade_start: int = 0
    session_trade_end: int = 24
    crp_lookback: int = 20
    ob_lookback: int = 5
    min_confluence: int = 0
    htf_ema_period: int = 50
    crypto_mode: bool = False
    # ── Phase G2-G10 new params ──
    use_judas_swing: bool = False
    judas_swing_atr_mult: float = 0.3
    judas_swing_displacement_mult: float = 1.0
    judas_swing_reversal_bars: int = 5
    use_po3_gate: bool = False
    use_ote_confluence: bool = False
    use_cisd: bool = False
    use_crt: bool = False
    use_smt: bool = False
    use_sd_zones: bool = False
    use_ob_fvg_colocation: bool = False
    use_unicorn: bool = False
    use_poi_grading: bool = False
    use_daily_loss_limit: bool = False
    daily_loss_limit: float = 0.03
    max_risk_pct: float = 0.01
    use_trade_plan: str = ""
    trade_plan_strictness: float = 1.0
    instrument_class: str = "auto"
    use_sfp: bool = False
    # ── Phase 21 new-tech gates (opt-in only — gates OFF by default) ──
    use_vix_gate: bool = False
    vix_gate_stress_mult: float = 0.30
    vix_gate_elevated_mult: float = 0.75
    use_yield_curve_gate: bool = False
    yield_curve_inversion_mult: float = 0.50
    yield_curve_near_inversion_mult: float = 0.75
    # ── Phase 22 fixes: Quality gates ──
    use_htf_gate: bool = False
    use_killzone_gate: bool = False
    use_swing_points: bool = False
    swing_point_weight: float = 0.20
    use_ict_patterns: bool = False
    ict_pattern_weight: float = 0.15

    _strategy_ref: Optional[list] = None

    # ── init() ──────────────────────────────────────────────────

    def init(self) -> None:
        if self._strategy_ref is not None:
            self._strategy_ref.append(self)

        self._df = self._build_df()
        self._n_bars = len(self._df)

        self._detect_instrument_class()

        self._precompute_atr()
        self._precompute_indicators()
        if self.use_vol_gate or self.use_session_gate or self.use_crash_gate:
            self._precompute_gate_arrays()
        else:
            self._vol_gate_mults = np.ones(self._n_bars, dtype=np.float64)
            self._session_mults = np.ones(self._n_bars, dtype=np.float64)
            self._crash_mults = np.ones(self._n_bars, dtype=np.float64)
        if self.use_volume_pressure:
            self._precompute_volume_pressure()
        else:
            self._bop = np.zeros(self._n_bars, dtype=np.float64)
            self._up_pressure = np.zeros(self._n_bars, dtype=np.float64)
            self._down_pressure = np.zeros(self._n_bars, dtype=np.float64)
        if self.use_crash_gate:
            self._precompute_crash_factors()
        else:
            self._dturn = np.ones(self._n_bars, dtype=np.float64)
            self._ncs_kew = np.zeros(self._n_bars, dtype=np.float64)
            self._tvol = np.zeros(self._n_bars, dtype=np.float64)
        if self.use_order_blocks:
            self._precompute_order_blocks()
        else:
            self._ob_signals = np.zeros(self._n_bars, dtype=np.int8)
            self._ob_proximity = np.zeros(self._n_bars, dtype=np.float64)

        if self.use_breaker_blocks or self.use_mitigation_blocks or self.use_rejection_blocks:
            self._precompute_phase6_signals()
        else:
            self._breaker_bull = np.zeros(self._n_bars, dtype=np.int8)
            self._breaker_bear = np.zeros(self._n_bars, dtype=np.int8)
            self._mitigation_bull = np.zeros(self._n_bars, dtype=np.int8)
            self._mitigation_bear = np.zeros(self._n_bars, dtype=np.int8)
            self._rejection_bull = np.zeros(self._n_bars, dtype=np.int8)
            self._rejection_bear = np.zeros(self._n_bars, dtype=np.int8)

        if self.use_smc_sessions:
            self._precompute_smc_sessions()
            self._precompute_market_killzones()
        else:
            self._in_london_kz = np.zeros(self._n_bars, dtype=bool)
            self._in_ny_kz = np.zeros(self._n_bars, dtype=bool)
            self._in_asian_kz = np.zeros(self._n_bars, dtype=bool)
            if self.use_killzone_gate:
                self._precompute_market_killzones()
            else:
                self._market_kz_active = np.zeros(self._n_bars, dtype=bool)

        if self.use_smc_retrace:
            self._precompute_smc_retracements()
        else:
            self._retrace_direction = np.zeros(self._n_bars, dtype=np.int8)
            self._retrace_current = np.zeros(self._n_bars, dtype=np.float64)
            self._retrace_deepest = np.zeros(self._n_bars, dtype=np.float64)

        # ── Phase G2-G10: Initialize all new arrays ──
        self._precompute_all_new()

        # ── Phase 21: New-tech gates (VIX + yield curve) ──
        self._vix_mults: Optional[np.ndarray] = None
        if self.use_vix_gate:
            self._init_vix_gate()
        else:
            self._vix_mults = np.ones(self._n_bars, dtype=np.float64)

        self._yield_curve_mults: Optional[np.ndarray] = None
        if self.use_yield_curve_gate:
            self._init_yield_curve_gate()
        else:
            self._yield_curve_mults = np.ones(self._n_bars, dtype=np.float64)

        self._precompute_time_gates()

        self._precompute_htf_bias()

        self._active_weights = dict(SMC_COMPONENT_WEIGHTS)

        self._post_init_precompute()

        if self.use_swing_points:
            self._init_swing_points()
        else:
            self._swing_high = np.zeros(self._n_bars, dtype=np.int8)
            self._swing_low = np.zeros(self._n_bars, dtype=np.int8)
        if self.use_ict_patterns:
            self._init_ict_patterns()
        else:
            self._ict_bull = np.zeros(self._n_bars, dtype=np.int8)
            self._ict_bear = np.zeros(self._n_bars, dtype=np.int8)

        self._trail_high: float = 0.0
        self._trail_low: float = float("inf")
        self._entry_price: float = 0.0
        self._tp1_hit: bool = False

        logger.info(
            "SMCStrategy: %d bars, sweeps=%d, msl=%d, msh=%d, bos=%d, fvg=%d, ob=%d, "
            "breaker_bull=%d, breaker_bear=%d, mit_bull=%d, mit_bear=%d, rej_bull=%d, rej_bear=%d",
            self._n_bars,
            int(np.sum(self._sweep_signals != 0)),
            int(np.sum(self._msl_signals > 0)),
            int(np.sum(self._msh_signals < 0)),
            int(np.sum(self._bos_signals != 0)),
            int(np.sum(self._fvg_proximity > 0)),
            int(np.sum(self._ob_signals != 0)),
            int(np.sum(self._breaker_bull)),
            int(np.sum(self._breaker_bear)),
            int(np.sum(self._mitigation_bull)),
            int(np.sum(self._mitigation_bear)),
            int(np.sum(self._rejection_bull)),
            int(np.sum(self._rejection_bear)),
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

    def _precompute_atr(self) -> None:
        high = self._df["High"]
        low = self._df["Low"]
        close = self._df["Close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        atr_vals = tr.rolling(self.atr_period).mean().bfill().fillna(close * 0.02)
        self._atr = atr_vals.to_numpy(dtype=np.float64)

    def _precompute_indicators(self) -> None:
        """Precompute all SMC signals as numpy arrays."""
        n = self._n_bars

        self._sweep_signals = np.zeros(n, dtype=np.int8)
        self._msl_signals = np.zeros(n, dtype=np.int8)
        self._msh_signals = np.zeros(n, dtype=np.int8)
        self._bos_signals = np.zeros(n, dtype=np.int8)
        self._fvg_proximity = np.zeros(n, dtype=np.float64)
        self._vol_mult = np.ones(n, dtype=np.float64)
        self._session_highs = np.zeros(n, dtype=np.float64)
        self._session_lows = np.zeros(n, dtype=np.float64)

        self._compute_session_ranges()
        self._compute_sweep_signals()
        self._compute_msl_msh()
        self._compute_bos_choch()
        self._compute_fvg_proximity()
        self._compute_volume_multipliers()

    # ── Session range detection ─────────────────────────────────

    def _compute_session_ranges(self) -> None:
        """Rolling session high/low over `session_bars`."""
        high = self._df["High"].to_numpy(dtype=np.float64)
        low = self._df["Low"].to_numpy(dtype=np.float64)
        n = self._n_bars

        for i in range(n):
            start = max(0, i - self.session_bars + 1)
            self._session_highs[i] = float(np.max(high[start : i + 1]))
            self._session_lows[i] = float(np.min(low[start : i + 1]))

    # ── Liquidity sweep detection ───────────────────────────────

    def _compute_sweep_signals(self) -> None:
        """Detect sweep-and-reverse signals WITH conviction grading.

        Bullish: price breaks below session low then reverses above.
        Bearish: price breaks above session high then reverses below.

        Conviction grades sweep quality continuously:
          - sweep_depth: how far beyond the session level (ATR-normalized, 0-3 range)
          - reversal_strength: how strongly the candle reversed (close within candle range)
          - conviction = clip(depth * reversal_strength, 0.1, 2.0)
        """
        close = self._df["Close"].to_numpy(dtype=np.float64)
        high = self._df["High"].to_numpy(dtype=np.float64)
        low = self._df["Low"].to_numpy(dtype=np.float64)
        n = self._n_bars

        self._sweep_conviction = np.zeros(n, dtype=np.float64)

        for i in range(2, n):
            atr = self._atr[i] if self._atr[i] > 0 else close[i] * 0.01
            buffer = self.sweep_buffer_mult * atr
            session_high = self._session_highs[i - 1] + buffer
            session_low = self._session_lows[i - 1] - buffer

            if high[i] > session_high and close[i] < close[i - 1]:
                sweep_depth = min((high[i] - session_high) / max(atr, 1e-10), 3.0)
                candle_range = max(high[i] - low[i], 1e-10)
                reversal = (high[i] - close[i]) / candle_range
                conviction = float(np.clip(sweep_depth * max(reversal, 0.2), 0.1, 2.0))
                self._sweep_signals[i] = -1
                self._sweep_conviction[i] = -conviction
            elif low[i] < session_low and close[i] > close[i - 1]:
                sweep_depth = min((session_low - low[i]) / max(atr, 1e-10), 3.0)
                candle_range = max(high[i] - low[i], 1e-10)
                reversal = (close[i] - low[i]) / candle_range
                conviction = float(np.clip(sweep_depth * max(reversal, 0.2), 0.1, 2.0))
                self._sweep_signals[i] = 1
                self._sweep_conviction[i] = conviction

    # ── Duddella MSL/MSH (3-bar close pattern) ────────────────

    def _compute_msl_msh(self) -> None:
        """Duddella 2007: 3-bar close pattern for structure levels.

        MSL: close[t-1] lower low, close[t] breaks above formation → bullish.
        MSH: close[t-1] higher high, close[t] breaks below formation → bearish.
        """
        close = self._df["Close"].to_numpy(dtype=np.float64)
        high = self._df["High"].to_numpy(dtype=np.float64)
        low = self._df["Low"].to_numpy(dtype=np.float64)
        n = self._n_bars

        for i in range(2, n):
            prev_close = close[i - 1]
            prev_prev_close = close[i - 2]
            curr_close = close[i]
            atr = self._atr[i] if self._atr[i] > 0 else curr_close * 0.01
            min_dist = 0.5 * atr

            msl_dist = prev_prev_close - prev_close
            if msl_dist > min_dist and curr_close > max(prev_close, prev_prev_close):
                if low[i] > low[i - 1]:
                    self._msl_signals[i] = 1

            msh_dist = prev_close - prev_prev_close
            if msh_dist > min_dist and curr_close < min(prev_close, prev_prev_close):
                if high[i] < high[i - 1]:
                    self._msh_signals[i] = -1

    # ── BOS/CHOCH via smartmoneyconcepts ─────────────────────

    def _compute_bos_choch(self) -> None:
        try:
            self._compute_bos_choch_smc()
        except Exception:
            logger.debug("smartmoneyconcepts BOS/CHOCH failed, using custom", exc_info=True)
            self._compute_bos_custom()

    def _compute_bos_choch_smc(self) -> None:
        smc = _import_smc_quietly()

        close_arr = self._df["Close"].to_numpy(dtype=np.float64)
        high_arr = self._df["High"].to_numpy(dtype=np.float64)
        low_arr = self._df["Low"].to_numpy(dtype=np.float64)

        swing_data = smc.swing_highs_lows(
            pd.DataFrame({"high": high_arr, "low": low_arr, "close": close_arr})
        )
        if swing_data is None or len(swing_data) == 0:
            self._compute_bos_custom()
            return

        bos_data = smc.bos_choch(
            pd.DataFrame({"high": high_arr, "low": low_arr, "close": close_arr}),
            swing_data,
            close_arr,
        )
        if bos_data is None or len(bos_data) == 0:
            self._compute_bos_custom()
            return

        self._bos_signals = np.zeros(self._n_bars, dtype=np.int8)
        bos_col = None
        for col in ["BOS", "bos", "Bos"]:
            if col in bos_data.columns:
                bos_col = col
                break
        choch_col = None
        for col in ["CHOCH", "choch", "Choch"]:
            if col in bos_data.columns:
                choch_col = col
                break

        bos_arr = np.zeros(self._n_bars, dtype=np.int8)
        if bos_col:
            for i, v in enumerate(bos_data[bos_col].values):
                if i < self._n_bars and not pd.isna(v):
                    val = int(v)
                    if val != 0 and self._atr[i] > 0:
                        move = abs(close_arr[i] - close_arr[i - 1]) if i > 0 else 0.0
                        if move > 0.5 * self._atr[i]:
                            bos_arr[i] = val
        if choch_col:
            for i, v in enumerate(bos_data[choch_col].values):
                if i < self._n_bars and pd.notna(v):
                    val = int(v)
                    if val != 0 and self._atr[i] > 0:
                        move = abs(close_arr[i] - close_arr[i - 1]) if i > 0 else 0.0
                        if move > 0.5 * self._atr[i]:
                            bos_arr[i] = val
        self._bos_signals = bos_arr

    def _compute_bos_custom(self) -> None:
        """Custom BOS detection: break of swing high/low with ATR filter."""
        close = self._df["Close"].to_numpy(dtype=np.float64)
        high = self._df["High"].to_numpy(dtype=np.float64)
        low = self._df["Low"].to_numpy(dtype=np.float64)
        n = self._n_bars
        lookback = 10

        self._bos_signals = np.zeros(n, dtype=np.int8)

        for i in range(lookback * 2 + 1, n):
            window_high = np.max(high[i - lookback : i])
            window_low = np.min(low[i - lookback : i])
            atr = self._atr[i] if self._atr[i] > 0 else close[i] * 0.01

            if close[i] > window_high and (close[i] - window_high) > 0.5 * atr:
                self._bos_signals[i] = 1
            elif close[i] < window_low and (window_low - close[i]) > 0.5 * atr:
                self._bos_signals[i] = -1

    # ── FVG proximity ────────────────────────────────────────

    def _compute_fvg_proximity(self) -> None:
        """Compute directional proximity to unfilled Fair Value Gaps.

        Positive = proximity to bullish FVG (buy support), negative = bearish.
        Uses strict ICT zero-wick-overlap rule and CE proximity boost.
        """
        close = self._df["Close"].to_numpy(dtype=np.float64)
        n = self._n_bars

        for i in range(3, n):
            atr = self._atr[i] if self._atr[i] > 0 else close[i] * 0.01

            prev_high = float(self._df["High"].iloc[i - 1])
            prev_low = float(self._df["Low"].iloc[i - 1])
            two_high = float(self._df["High"].iloc[i - 2])
            two_low = float(self._df["Low"].iloc[i - 2])
            c1_high = two_high
            c1_low = two_low
            c3_high = float(self._df["High"].iloc[i])
            c3_low = float(self._df["Low"].iloc[i])

            bullish_fvg = c1_high < c3_low and two_high < prev_low
            bearish_fvg = c1_low > c3_high and two_low > prev_high

            threshold = self.fvg_proximity_mult * atr

            if bullish_fvg and abs(close[i] - prev_low) < threshold:
                score = float(np.clip(1.0 - abs(close[i] - prev_low) / threshold, 0.1, 1.0))
                ce = (c1_high + c3_low) / 2.0
                if abs(close[i] - ce) < abs(close[i] - prev_low) and abs(close[i] - ce) < threshold:
                    score *= 1.3
                self._fvg_proximity[i] = score
            elif bearish_fvg and abs(close[i] - prev_high) < threshold:
                score = -float(np.clip(1.0 - abs(close[i] - prev_high) / threshold, 0.1, 1.0))
                ce = (c1_low + c3_high) / 2.0
                if (
                    abs(close[i] - ce) < abs(close[i] - prev_high)
                    and abs(close[i] - ce) < threshold
                ):
                    score *= 1.3
                self._fvg_proximity[i] = score

    # ── Order block detection ─────────────────────────────────

    def _precompute_order_blocks(self) -> None:
        """Detect order blocks (supply/demand zones).

        Bullish OB: last down candle before sequence of up candles.
        Bearish OB: last up candle before sequence of down candles.
        Proximity scoring: how close is current price to a recent OB zone.
        """
        if not self.use_order_blocks:
            self._ob_signals = np.zeros(self._n_bars, dtype=np.int8)
            self._ob_proximity = np.zeros(self._n_bars, dtype=np.float64)
            return

        close = self._df["Close"].to_numpy(dtype=np.float64)
        open_ = self._df["Open"].to_numpy(dtype=np.float64)
        high = self._df["High"].to_numpy(dtype=np.float64)
        low = self._df["Low"].to_numpy(dtype=np.float64)
        n = self._n_bars

        self._ob_signals = np.zeros(n, dtype=np.int8)
        self._ob_proximity = np.zeros(n, dtype=np.float64)

        lookback = self.ob_lookback

        for i in range(lookback * 2 + 1, n):
            atr = self._atr[i] if self._atr[i] > 0 else close[i] * 0.01

            bullish_ob = True
            last_down_idx = -1
            for j in range(i - lookback, i):
                if close[j] <= open_[j]:
                    last_down_idx = j
                else:
                    if last_down_idx >= 0 and j - last_down_idx >= 2:
                        break
            else:
                bullish_ob = False

            if bullish_ob and last_down_idx >= 0:
                zone_high = high[last_down_idx]
                zone_low = low[last_down_idx]
                zone_mid = (zone_high + zone_low) / 2.0
                dist = abs(close[i] - zone_mid)
                threshold = 2.0 * atr
                if dist < threshold and close[i] > zone_mid:
                    self._ob_signals[i] = 1
                    self._ob_proximity[i] = float(np.clip(1.0 - dist / threshold, 0.1, 1.0))

            bearish_ob = True
            last_up_idx = -1
            for j in range(i - lookback, i):
                if close[j] > open_[j]:
                    last_up_idx = j
                else:
                    if last_up_idx >= 0 and j - last_up_idx >= 2:
                        break
            else:
                bearish_ob = False

            if bearish_ob and last_up_idx >= 0:
                zone_high = high[last_up_idx]
                zone_low = low[last_up_idx]
                zone_mid = (zone_high + zone_low) / 2.0
                dist = abs(close[i] - zone_mid)
                threshold = 2.0 * atr
                if dist < threshold and close[i] < zone_mid:
                    self._ob_signals[i] = -1
                    self._ob_proximity[i] = float(np.clip(1.0 - dist / threshold, 0.1, 1.0))

    # ── Volume confirmation ──────────────────────────────────

    def _compute_volume_multipliers(self) -> None:
        vol = self._df["Volume"].to_numpy(dtype=np.float64)
        n = self._n_bars

        for i in range(n):
            if i < 20:
                continue
            avg_vol = np.mean(vol[i - 20 : i])
            if avg_vol <= 0:
                continue
            rel_vol = vol[i] / avg_vol
            self._vol_mult[i] = float(np.clip(rel_vol, 0.5, 2.0))

    # ── Higher-timeframe bias filter ─────────────────────────

    def _precompute_htf_bias(self) -> None:
        """Compute daily EMA trend bias mapped to hourly bars.

        Resamples hourly close to daily, computes EMA, then maps the
        position relative to EMA onto each hourly bar index.
        Bias = +1 when price > EMA (bullish), -1 when below (bearish).
        """
        n = self._n_bars
        self._htf_bias = np.zeros(n, dtype=np.float64)

        close = self._df["Close"]
        if not isinstance(close.index, pd.DatetimeIndex):
            return
        close_daily = close.resample("D").last().dropna()
        if len(close_daily) < self.htf_ema_period:
            self._htf_bias[:] = 0.0
            return

        ema = close_daily.ewm(span=self.htf_ema_period).mean()
        daily_bias = (close_daily > ema).astype(np.float64) * 2.0 - 1.0

        for i in range(n):
            bar_date = close.index[i].floor("D")
            if bar_date in daily_bias.index:
                self._htf_bias[i] = float(daily_bias.loc[bar_date])
            elif bar_date > daily_bias.index[0]:
                self._htf_bias[i] = float(daily_bias.iloc[-1])

    # ── T5.1: Multiplicative gate chain ──────────────────

    def _precompute_gate_arrays(self) -> None:
        """Precompute per-bar multiplier arrays for the gate chain.

        Vol gate: reduces signals during high volatility (ATR-based).
        Session gate: reduces signals outside desired trading hours.
        Crash gate: reduces signals when crash risk is elevated.
        """
        n = self._n_bars
        close = self._df["Close"].to_numpy(dtype=np.float64)

        self._vol_gate_mults = np.ones(n, dtype=np.float64)
        if self.use_vol_gate:
            atr_pct = self._atr / np.maximum(close, 1e-10)
            atr_med = float(np.median(atr_pct[atr_pct > 0])) if np.any(atr_pct > 0) else 0.01
            if self.crypto_mode:
                atr_med = (
                    float(np.percentile(atr_pct[atr_pct > 0], 85)) if np.any(atr_pct > 0) else 0.01
                )
            if atr_med > 0:
                vol_ratio = atr_pct / (atr_med + 1e-10)
                if self.crypto_mode:
                    self._vol_gate_mults = np.clip(2.0 - vol_ratio, 0.5, 2.0)
                else:
                    self._vol_gate_mults = np.clip(1.2 - vol_ratio, 0.3, 1.5)

        self._session_mults = np.ones(n, dtype=np.float64)
        if self.use_session_gate and hasattr(self._df.index, "hour"):
            hours = self._df.index.hour
            for i in range(n):
                h = int(hours[i])
                if h < self.session_trade_start or h >= self.session_trade_end:
                    self._session_mults[i] = 0.5

        self._crash_mults = np.ones(n, dtype=np.float64)

    # ── T5.2: Tick direction volume pressure ──────────────

    def _precompute_volume_pressure(self) -> None:
        """Compute tick direction volume features (approximation from OHLCV).

        Based on ML4T Ch12. Uses OHLCV to approximate order flow pressure:
          - BOP (Balance of Power): (Close - Open) / (High - Low)
          - up_pressure: fraction of volume on up bars (smoothed)
          - down_pressure: fraction of volume on down bars (smoothed)
        """
        n = self._n_bars
        close = self._df["Close"].to_numpy(dtype=np.float64)
        open_ = self._df["Open"].to_numpy(dtype=np.float64)
        high = self._df["High"].to_numpy(dtype=np.float64)
        low = self._df["Low"].to_numpy(dtype=np.float64)
        volume = self._df["Volume"].to_numpy(dtype=np.float64)

        self._bop = np.zeros(n, dtype=np.float64)
        hl_range = high - low
        nonzero = hl_range > 0
        self._bop[nonzero] = (close[nonzero] - open_[nonzero]) / hl_range[nonzero]

        self._up_pressure = np.zeros(n, dtype=np.float64)
        self._down_pressure = np.zeros(n, dtype=np.float64)

        up = (close > open_).astype(np.float64)
        down = (close < open_).astype(np.float64)

        window = 10
        for i in range(n):
            start = max(0, i - window + 1)
            vol_sum = float(np.sum(volume[start : i + 1]))
            if vol_sum > 0:
                self._up_pressure[i] = (
                    float(np.sum(volume[start : i + 1] * up[start : i + 1])) / vol_sum
                )
                self._down_pressure[i] = (
                    float(np.sum(volume[start : i + 1] * down[start : i + 1])) / vol_sum
                )

    # ── T5.3: Intraday crash/risk factors ────────────────

    def _precompute_crash_factors(self) -> None:
        """Compute intraday crash indicators on rolling windows.

        Adapted from crash-based trading paper:
          - DTURN: Detrended turnover (current vol / rolling avg)
          - NCSKEW: Negative return skewness (20-bar rolling)
          - TVOL: Rolling volatility (20-bar std dev of returns)
        """
        n = self._n_bars
        close = self._df["Close"].to_numpy(dtype=np.float64)
        volume = self._df["Volume"].to_numpy(dtype=np.float64)
        window = self.crp_lookback

        returns = np.zeros(n, dtype=np.float64)
        returns[1:] = np.diff(np.log(close))

        self._dturn = np.ones(n, dtype=np.float64)
        self._ncs_kew = np.zeros(n, dtype=np.float64)
        self._tvol = np.zeros(n, dtype=np.float64)

        for i in range(window, n):
            r = returns[i - window : i]
            self._tvol[i] = float(np.std(r)) if np.std(r) > 0 else 0.0
            skew = float(pd.Series(r).skew())
            self._ncs_kew[i] = -skew if not pd.isna(skew) else 0.0

            vol_slice = volume[i - window : i]
            avg_v = float(np.mean(vol_slice)) if np.mean(vol_slice) > 0 else 1.0
            self._dturn[i] = float(volume[i]) / avg_v

        if self.use_crash_gate:
            tvol_cutoff = (
                float(np.percentile(self._tvol[self._tvol > 0], 80))
                if np.any(self._tvol > 0)
                else 1.0
            )
            for i in range(n):
                if self._tvol[i] <= 0:
                    continue
                vol_ratio = self._tvol[i] / max(tvol_cutoff, 1e-10)
                dturn_ratio = self._dturn[i]
                mult = 1.0
                if vol_ratio > 1.0:
                    mult -= 0.25 * min(vol_ratio - 1.0, 2.0)
                if dturn_ratio > 1.5:
                    mult -= 0.15 * min(dturn_ratio - 1.5, 3.0)
                if self._ncs_kew[i] > 0.5:
                    mult -= 0.10 * min(self._ncs_kew[i] - 0.5, 1.5)
                self._crash_mults[i] = float(np.clip(mult, 0.25, 1.2))

    # ── Phase 6: Breaker, Mitigation, Rejection blocks ─────

    def _precompute_phase6_signals(self) -> None:
        """Precompute Phase 6 SMC detectors: Breaker, Mitigation, Rejection blocks."""
        n = self._n_bars

        smc_lib = _import_smc_quietly()

        from src.patterns.smc.breaker import detect_breaker_blocks
        from src.patterns.smc.mitigation import detect_mitigation_blocks
        from src.patterns.smc.rejection import detect_rejection_blocks

        vol_sma = self._df["Volume"].rolling(20, min_periods=1).mean().to_numpy(dtype=np.float64)

        session_arr = np.ones(n, dtype=bool)
        if self.use_session_gate and hasattr(self._df.index, "hour"):
            hours = self._df.index.hour
            for i in range(n):
                h = int(hours[i])
                if h < self.session_trade_start or h >= self.session_trade_end:
                    session_arr[i] = False

        swing_hl = smc_lib.swing_highs_lows(self._df, swing_length=20)

        if self.use_breaker_blocks:
            self._breaker_bull, self._breaker_bear, self._breakers = detect_breaker_blocks(
                self._df,
                swing_hl,
                self._atr,
                vol_sma,
                session_active=session_arr,
                sweep_signals=self._sweep_signals,
            )
        else:
            self._breaker_bull = np.zeros(n, dtype=np.int8)
            self._breaker_bear = np.zeros(n, dtype=np.int8)

        if self.use_mitigation_blocks:
            self._mitigation_bull, self._mitigation_bear, self._mitigations = (
                detect_mitigation_blocks(
                    self._df,
                    swing_hl,
                    self._atr,
                    session_active=session_arr,
                )
            )
        else:
            self._mitigation_bull = np.zeros(n, dtype=np.int8)
            self._mitigation_bear = np.zeros(n, dtype=np.int8)

        if self.use_rejection_blocks:
            ob_result = smc_lib.ob(self._df, swing_hl, close_mitigation=False)
            fvg_result = smc_lib.fvg(self._df, join_consecutive=True)
            self._rejection_bull, self._rejection_bear, self._rejections = detect_rejection_blocks(
                self._df,
                fvg_result,
                ob_result,
                self._atr,
                vol_sma,
                session_active=session_arr,
            )
        else:
            self._rejection_bull = np.zeros(n, dtype=np.int8)
            self._rejection_bear = np.zeros(n, dtype=np.int8)

    # ── Phase 9: smartmoneyconcepts library full integration ─

    def _precompute_smc_sessions(self) -> None:
        """Use smartmoneyconcepts sessions() for 9 predefined killzone detections."""
        smc_lib = _import_smc_quietly()
        n = self._n_bars
        self._in_london_kz = np.zeros(n, dtype=bool)
        self._in_ny_kz = np.zeros(n, dtype=bool)
        self._in_asian_kz = np.zeros(n, dtype=bool)

        if not isinstance(self._df.index, pd.DatetimeIndex):
            return

        try:
            london = smc_lib.sessions(self._df, "London open kill zone", time_zone="UTC+0")
            ny = smc_lib.sessions(self._df, "New York kill zone", time_zone="UTC+0")
            asian = smc_lib.sessions(self._df, "Asian kill zone", time_zone="UTC+0")
            self._in_london_kz = london["Active"].to_numpy(dtype=np.int8).astype(bool)[:n]
            self._in_ny_kz = ny["Active"].to_numpy(dtype=np.int8).astype(bool)[:n]
            self._in_asian_kz = asian["Active"].to_numpy(dtype=np.int8).astype(bool)[:n]
        except Exception:
            logger.debug("smc.sessions() failed — killzone gates disabled", exc_info=True)

    def _precompute_smc_retracements(self) -> None:
        """Direction-aware Fibonacci retracement tracking."""
        smc_lib = _import_smc_quietly()
        n = self._n_bars
        self._retrace_direction = np.zeros(n, dtype=np.int8)
        self._retrace_current = np.zeros(n, dtype=np.float64)
        self._retrace_deepest = np.zeros(n, dtype=np.float64)

        try:
            swing_hl = smc_lib.swing_highs_lows(self._df, swing_length=20)
            ret = smc_lib.retracements(self._df, swing_hl)
            self._retrace_direction = ret["Direction"].to_numpy(dtype=np.int8)[:n]
            self._retrace_current = ret["CurrentRetracement%"].to_numpy(dtype=np.float64)[:n]
            self._retrace_deepest = ret["DeepestRetracement%"].to_numpy(dtype=np.float64)[:n]
        except Exception:
            logger.debug("smc.retracements() failed", exc_info=True)

    # ── G2-G10: All new feature precomputation ─────────────

    def _precompute_all_new(self) -> None:
        """Initialize and precompute all Phase G2-G10 arrays."""
        n = self._n_bars

        default_bool = np.zeros(n, dtype=bool)
        default_int8 = np.zeros(n, dtype=np.int8)
        default_float = np.zeros(n, dtype=np.float64)

        self._bos_signals = (
            np.zeros(n, dtype=np.int8) if not hasattr(self, "_bos_signals") else self._bos_signals
        )
        self._choch_signals = np.zeros(n, dtype=np.int8)
        self._fake_choch = np.zeros(n, dtype=bool)
        self._judas_swing_dir = np.zeros(n, dtype=np.int8)
        self._po3_phase = np.zeros(n, dtype=np.int8)
        self._in_ote_zone = np.zeros(n, dtype=bool)
        self._ote_retracement_pct = np.zeros(n, dtype=np.float64)
        self._cisd_dir = np.zeros(n, dtype=np.int8)
        self._crt_dir = np.zeros(n, dtype=np.int8)
        self._smt_dir = np.zeros(n, dtype=np.int8)
        self._sd_zone_dir = np.zeros(n, dtype=np.int8)
        self._sd_zone_strength = np.zeros(n, dtype=np.float64)
        self._unicorn_signal = np.zeros(n, dtype=int)
        self._sfp_dir = np.zeros(n, dtype=np.int8)

        self._daily_pnl = 0.0
        self._last_trade_date = None
        self._consecutive_losses = 0

        any_g4 = (
            self.use_po3_gate
            or self.use_ote_confluence
            or self.use_cisd
            or self.use_crt
            or self.use_smt
        )
        any_g5 = self.use_sd_zones or self.use_ob_fvg_colocation or self.use_unicorn
        any_g3 = self.use_judas_swing
        any_new = any_g3 or any_g4 or any_g5 or self.use_sfp or self.use_poi_grading

        if not any_new:
            return

        try:
            smc_lib = _import_smc_quietly()
            swing_hl = smc_lib.swing_highs_lows(self._df, swing_length=20)
        except Exception:
            swing_hl = None

        # G2: BOS/CHOCH split
        if self._bos_signals is not None and len(self._bos_signals) == n:
            bos_arr = self._bos_signals
        else:
            bos_arr = np.zeros(n, dtype=np.int8)

        try:
            from src.indicators.mss import detect_bos, detect_fake_choch

            for i in range(10, n, 5):
                bos_info = detect_bos(self._df, self._atr, lookback=10, end_bar=i)
                if bos_info.detected and bos_info.direction == "bullish" and bos_info.is_valid:
                    self._bos_signals[i] = 1
                elif bos_info.detected and bos_info.direction == "bearish" and bos_info.is_valid:
                    self._bos_signals[i] = -1

                fake = detect_fake_choch(self._df, self._atr, htf_bias=self._htf_bias, end_bar=i)
                if fake.detected:
                    self._fake_choch[i] = True
                    self._choch_signals[i] = 1 if fake.direction == "bullish" else -1
        except Exception:
            logger.debug("BOS/CHOCH detection skipped", exc_info=True)

        # G3: Judas Swing
        if self.use_judas_swing and isinstance(self._df.index, pd.DatetimeIndex):
            try:
                from src.indicators.judas_swing import detect_judas_swing
                from src.indicators.asian_range import get_asian_range_for_bar

                for i in range(96, n, 24):
                    ah, al, _ = get_asian_range_for_bar(self._df, i)
                    if ah <= 0 or al <= 0:
                        continue
                    js = detect_judas_swing(
                        self._df,
                        ah,
                        al,
                        i,
                        min(i + 24, n - 1),
                        self._atr,
                        atr_mult=self.judas_swing_atr_mult,
                        displacement_mult=self.judas_swing_displacement_mult,
                        reversal_bars=self.judas_swing_reversal_bars,
                    )
                    if js is not None and js.detected:
                        for j in range(js.reversal_bar, min(js.reversal_bar + 24, n)):
                            self._judas_swing_dir[j] = 1 if js.direction == "bullish" else -1
            except Exception:
                logger.debug("Judas Swing detection failed", exc_info=True)

        # G4: PO3/AMD gate
        if self.use_po3_gate:
            try:
                from src.indicators.power_of_3 import detect_po3_daily

                po3_df = detect_po3_daily(self._df)
                phase_map = {"accumulation": 1, "manipulation": 2, "distribution": 3, "none": 0}
                for i in range(min(n, len(po3_df))):
                    self._po3_phase[i] = phase_map.get(str(po3_df["phase"].iloc[i]), 0)
            except Exception:
                logger.debug("PO3 gate skipped", exc_info=True)

        # G4: OTE Fibonacci confluence
        if self.use_ote_confluence:
            try:
                for i in range(2, n):
                    recent_high = float(np.max(self._df["High"].iloc[max(0, i - 20) : i + 1]))
                    recent_low = float(np.min(self._df["Low"].iloc[max(0, i - 20) : i + 1]))
                    fib_range = recent_high - recent_low
                    if fib_range <= 0:
                        continue
                    retracement = (recent_high - self._df["Close"].iloc[i]) / fib_range
                    retracement = max(0.0, min(1.0, retracement))
                    if 0.62 <= retracement <= 0.79:
                        self._in_ote_zone[i] = True
                        self._ote_retracement_pct[i] = retracement * 100
            except Exception:
                pass

        # G4: CISD
        if self.use_cisd:
            try:
                from src.indicators.cisd import detect_cisd_vectorized

                cisd_df = detect_cisd_vectorized(self._df)
                cisd_bull = cisd_df["bullish_cisd"].to_numpy(dtype=bool)
                cisd_bear = cisd_df["bearish_cisd"].to_numpy(dtype=bool)
                for i in range(min(n, len(cisd_bull))):
                    if cisd_bull[i]:
                        self._cisd_dir[i] = 1
                    elif cisd_bear[i]:
                        self._cisd_dir[i] = -1
            except Exception:
                pass

        # G4: CRT
        if self.use_crt:
            try:
                from src.indicators.crt import find_crt_setups

                atr_series = pd.Series(self._atr, index=self._df.index)
                setups = find_crt_setups(self._df, atr_series)
                for setup in setups:
                    if setup.detected and setup.reversal_bar is not None and setup.reversal_bar < n:
                        self._crt_dir[setup.reversal_bar] = (
                            1 if setup.target_direction == "bullish" else -1
                        )
            except Exception:
                pass

        # G4: SMT divergence
        if self.use_smt and hasattr(self, "_smt_secondary_data"):
            try:
                from src.signals.smc_divergence import detect_smt_divergence

                smt_signals, _ = detect_smt_divergence(self._df, self._smt_secondary_data)
                self._smt_dir = smt_signals[:n].astype(np.int8)
            except Exception:
                pass

        # G5: S&D zones
        if self.use_sd_zones:
            try:
                from src.patterns.smc.sd_zones import detect_sd_zones

                sd_zones = detect_sd_zones(self._df, self._atr)
                for zone in sd_zones:
                    if zone.detection_bar < n:
                        self._sd_zone_dir[zone.detection_bar] = zone.direction
                        self._sd_zone_strength[zone.detection_bar] = zone.strength
            except Exception:
                pass

        # G5: Unicorn (Breaker + FVG overlap)
        if self.use_unicorn:
            for i in range(3, n):
                brk_dir = int(self._breaker_bull[i]) - int(self._breaker_bear[i])
                if brk_dir == 0:
                    continue
                prev_high = float(self._df["High"].iloc[i - 1])
                prev_low = float(self._df["Low"].iloc[i - 1])
                two_high = float(self._df["High"].iloc[i - 2])
                two_low = float(self._df["Low"].iloc[i - 2])
                has_fvg = (two_high < prev_low) or (two_low > prev_high)
                if has_fvg:
                    self._unicorn_signal[i] = 1

        # G10: SFP
        if self.use_sfp:
            try:
                from src.patterns.smc.sfp import detect_sfp

                sfps = detect_sfp(self._df, atr=self._atr)
                for sfp in sfps:
                    if sfp.detected and sfp.reversal_bar is not None and sfp.reversal_bar < n:
                        self._sfp_dir[sfp.reversal_bar] = 1 if sfp.direction == "bullish" else -1
            except Exception:
                pass

    # ── G10: Market-specific kill zone detection ────────────

    # ── Phase 21: New-tech gate helpers ─────────────────────

    def _init_vix_gate(self) -> None:
        """Initialize VIX regime gate for per-bar score multipliers."""
        try:
            from src.signals.vix_regime_gate import VixRegimeGate

            gate = VixRegimeGate(
                stress_mult=self.vix_gate_stress_mult,
                elevated_mult=self.vix_gate_elevated_mult,
            )
            gate.fit(start=str(self._df.index[0].date()))
            n = self._n_bars
            self._vix_mults = np.ones(n, dtype=np.float64)
            for i in range(n):
                d = self._df.index[i]
                if hasattr(d, "date"):
                    d = d.date()
                self._vix_mults[i] = gate.multiplier(date=d)
        except Exception:
            logger.debug("VIX gate init failed, using defaults", exc_info=True)
            self._vix_mults = np.ones(self._n_bars, dtype=np.float64)

    def _init_yield_curve_gate(self) -> None:
        """Initialize yield curve macro gate for per-bar score multipliers."""
        try:
            from src.signals.yield_curve_gate import YieldCurveGate

            gate = YieldCurveGate(
                inversion_mult=self.yield_curve_inversion_mult,
                near_inversion_mult=self.yield_curve_near_inversion_mult,
            )
            gate.fit(start=str(self._df.index[0].date()))
            n = self._n_bars
            self._yield_curve_mults = np.ones(n, dtype=np.float64)
            for i in range(n):
                d = self._df.index[i]
                if hasattr(d, "date"):
                    d = d.date()
                self._yield_curve_mults[i] = gate.multiplier(date=d)
        except Exception:
            logger.debug("Yield curve gate init failed, using defaults", exc_info=True)
            self._yield_curve_mults = np.ones(self._n_bars, dtype=np.float64)

    # ── G10: Market-specific kill zone detection ────────────

    def _detect_instrument_class(self) -> None:
        """Auto-detect instrument class from symbol prefix.

        Rules:
          - XAU* → gold
          - BTC* / ETH* → crypto
          - ES* / NQ* / SPY* / QQQ* → stocks
          - else → forex
        """
        if self.instrument_class != "auto":
            self._instrument_type = self.instrument_class
            return

        symbol = ""
        if hasattr(self.data, "df") and hasattr(self.data.df, "name") and self.data.df.name:
            symbol = str(self.data.df.name).upper()
        if not symbol and hasattr(self._df, "name") and self._df.name:
            symbol = str(self._df.name).upper()
        if not symbol and hasattr(self._df.index, "name") and self._df.index.name:
            symbol = str(self._df.index.name).upper()

        if symbol.startswith("XAU"):
            self._instrument_type = "gold"
        elif symbol.startswith(("BTC", "ETH")):
            self._instrument_type = "crypto"
        elif symbol.startswith(("ES", "NQ", "SPY", "QQQ")):
            self._instrument_type = "stocks"
        elif symbol:
            self._instrument_type = "forex"
        else:
            self._instrument_type = "forex"

    def _precompute_market_killzones(self) -> None:
        """Precompute per-bar boolean array for market-specific kill zones.

        Time windows (UTC, EST-based):
          - forex: London 06-09 UTC (2-5am EST), NY 11-14 UTC (7-10am EST)
          - crypto: NYSE open 13:30-15:30 UTC (9:30-11:30am EST)
          - stocks/indices: Market open 13:30-15:00 UTC (9:30-11am EST)
          - gold: London-NY overlap 12-16 UTC (8am-12pm EST)
        """
        n = self._n_bars
        self._market_kz_active = np.zeros(n, dtype=bool)

        if not isinstance(self._df.index, pd.DatetimeIndex):
            return

        inst = self._instrument_type
        hours = self._df.index.hour

        if inst == "forex":
            for i in range(n):
                h = hours[i]
                self._market_kz_active[i] = (6 <= h < 9) or (11 <= h < 14)
        elif inst == "crypto":
            for i in range(n):
                h = hours[i]
                m = self._df.index[i].minute
                total_mins = h * 60 + m
                self._market_kz_active[i] = 810 <= total_mins < 930
        elif inst in ("stocks", "indices"):
            for i in range(n):
                h = hours[i]
                m = self._df.index[i].minute
                total_mins = h * 60 + m
                self._market_kz_active[i] = 810 <= total_mins < 900
        else:
            for i in range(n):
                h = hours[i]
                self._market_kz_active[i] = 12 <= h < 16

    # ── Phase 12: Time-based gates ──────────────────────────

    def _precompute_time_gates(self) -> None:
        """Precompute all Phase 12 time-based multiplier arrays."""
        n = self._n_bars
        self._dow_mults = np.ones(n, dtype=np.float64)
        self._cycle_mults = np.ones(n, dtype=np.float64)
        self._frankfurt_mults = np.ones(n, dtype=np.float64)

        if self.use_dow_gate:
            self._precompute_dow_gate()
        if self.use_90min_cycle:
            self._precompute_90min_cycle()
        if self.use_frankfurt_gate:
            self._precompute_frankfurt_gate()

    def _precompute_dow_gate(self) -> None:
        """Day-of-week bias from David Woods weekly profile.

        Mon: MANIPULATION — 0.70x (caution)
        Tue: CONTINUATION — 1.0x
        Wed: REACCUMULATION/REVERSAL — 0.85x
        Thu: COMPLETE Wed move — 1.0x
        Fri: DISTRIBUTION — 0.60x
        """
        if not isinstance(self._df.index, pd.DatetimeIndex):
            return
        weekday_map = {0: 0.70, 1: 1.00, 2: 0.85, 3: 1.00, 4: 0.60, 5: 0.50, 6: 0.50}
        for i in range(self._n_bars):
            wd = self._df.index[i].weekday()
            self._dow_mults[i] = weekday_map.get(wd, 1.0)

    def _precompute_90min_cycle(self) -> None:
        """90-minute cycle awareness — higher sensitivity near cycle boundaries.

        David Woods: Risk of significant price change every 90 minutes from 00:00 NY.
        Amplifies signal sensitivity at 90-min markers rather than blocking.
        """
        if not isinstance(self._df.index, pd.DatetimeIndex):
            return
        for i in range(self._n_bars):
            t = self._df.index[i]
            mins = t.hour * 60 + t.minute
            cycle_pos = mins % 90
            if cycle_pos <= 5 or cycle_pos >= 85:
                self._cycle_mults[i] = 1.3

    def _precompute_frankfurt_gate(self) -> None:
        """Frankfurt fake move gate — reduce confidence during Frankfurt session.

        David Woods: 'Frankfurt always makes a fake move' — 02:00-03:00 EST.
        Approx 07:00-08:00 UTC.
        """
        if not isinstance(self._df.index, pd.DatetimeIndex):
            return
        for i in range(self._n_bars):
            hour = self._df.index[i].hour
            if 7 <= hour < 8:
                self._frankfurt_mults[i] = 0.70

    # ── Signal scoring ───────────────────────────────────────

    def _compute_smc_score(self, idx: int) -> float:
        """Compute composite SMC score using precomputed signal arrays.

        G2-G10 enhancements: BOS/CHOCH split, Judas Swing gate, PO3/OTE/CISD/CRT/SMT,
        S&D zones, Unicorn, POI grading, SFP, trade plan check.
        """
        if idx < 2:
            return 0.0

        sweep_sig = float(self._sweep_signals[idx]) if idx < self._n_bars else 0.0
        if sweep_sig == 0:
            return 0.0

        sweep_conv = 0.0
        if hasattr(self, "_sweep_conviction") and idx < len(self._sweep_conviction):
            sweep_conv = float(self._sweep_conviction[idx])
        if sweep_conv == 0.0:
            sweep_conv = sweep_sig

        sweep_dir = 1.0 if sweep_conv > 0 else -1.0

        msl_sig = float(self._msl_signals[idx]) if idx < self._n_bars else 0.0
        msh_sig = float(self._msh_signals[idx]) if idx < self._n_bars else 0.0
        bos_sig = float(self._bos_signals[idx]) if idx < self._n_bars else 0.0
        fvg_sig = float(self._fvg_proximity[idx]) if idx < self._n_bars else 0.0

        mss_sig = msl_sig + msh_sig

        ob_sig = 0.0
        if self.use_order_blocks and hasattr(self, "_ob_signals"):
            ob_sig = (
                float(self._ob_signals[idx]) * float(self._ob_proximity[idx])
                if idx < len(self._ob_signals)
                else 0.0
            )

        w = self._active_weights
        base_score = sweep_conv * w["sweep_reversal"]

        # ── Phase 22 Fix: HTF trend gate (hard filter when enabled — opt-in) ──
        if self.use_htf_gate and idx < len(self._htf_bias):
            htf = self._htf_bias[idx]
            if htf != 0 and np.sign(htf) != sweep_dir:
                return 0.0

        # ── Phase 22 Fix: Killzone gate (hard filter — ON by default) ──
        if (
            self.use_killzone_gate
            and hasattr(self, "_market_kz_active")
            and idx < len(self._market_kz_active)
        ):
            in_smc = getattr(self, "_in_london_kz", None)
            if in_smc is not None and idx < len(in_smc):
                in_smc_kz = in_smc[idx] or (
                    hasattr(self, "_in_ny_kz") and idx < len(self._in_ny_kz) and self._in_ny_kz[idx]
                )
            else:
                in_smc_kz = False
            in_market_kz = self._market_kz_active[idx]
            if not in_smc_kz and not in_market_kz:
                return 0.0

        confirmations = [
            s * sweep_dir for s in (mss_sig, bos_sig, fvg_sig, ob_sig) if s * sweep_dir > 0
        ]

        if self.min_confluence > 0 and len(confirmations) < self.min_confluence:
            return 0.0

        # ── Scoring: BOS vs CHOCH differentiated ──
        base_score += sum(
            s * w.get(k, w.get("bos_choch", 0.05))
            for s, k in zip(
                (mss_sig, bos_sig, fvg_sig, ob_sig),
                ("msl_msh", "bos_choch", "fvg_proximity", "order_block"),
            )
        )

        # G2: CHOCH vs BOS differentiation
        choch_sig = (
            float(self._choch_signals[idx])
            if hasattr(self, "_choch_signals") and idx < len(self._choch_signals)
            else 0.0
        )
        if choch_sig != 0 and choch_sig == np.sign(sweep_sig):
            base_score += choch_sig * 0.40
        elif bos_sig != 0 and bos_sig == np.sign(sweep_sig):
            base_score += bos_sig * 0.15

        if hasattr(self, "_fake_choch") and idx < len(self._fake_choch) and self._fake_choch[idx]:
            base_score *= 0.3

        # G4: PO3 gate
        if self.use_po3_gate and hasattr(self, "_po3_phase"):
            phase = self._po3_phase[idx]
            if phase == 1:
                return 0.0
            elif phase == 2:
                base_score *= 0.5
            elif phase == 3:
                base_score *= 1.0

        # G4: OTE confluence
        if (
            self.use_ote_confluence
            and hasattr(self, "_in_ote_zone")
            and idx < len(self._in_ote_zone)
        ):
            if self._in_ote_zone[idx]:
                if (
                    hasattr(self, "_ote_retracement_pct")
                    and abs(self._ote_retracement_pct[idx] - 70.5) < 3
                ):
                    base_score *= 1.35
                else:
                    base_score *= 1.25

        # G4: CISD confirmation
        if self.use_cisd and hasattr(self, "_cisd_dir") and idx < len(self._cisd_dir):
            cisd = self._cisd_dir[idx]
            if cisd != 0 and np.sign(cisd) == np.sign(sweep_sig):
                base_score += 0.15 * cisd

        # G4: CRT confirmation
        if self.use_crt and hasattr(self, "_crt_dir") and idx < len(self._crt_dir):
            crt = self._crt_dir[idx]
            if crt != 0 and np.sign(crt) == np.sign(sweep_sig):
                base_score += 0.10 * crt

        # G4: SMT divergence
        if self.use_smt and hasattr(self, "_smt_dir") and idx < len(self._smt_dir):
            smt = self._smt_dir[idx]
            if smt != 0 and np.sign(smt) == np.sign(sweep_sig):
                base_score += 0.20 * smt

        # G5: S&D zone
        if self.use_sd_zones and hasattr(self, "_sd_zone_dir") and idx < len(self._sd_zone_dir):
            sd = self._sd_zone_dir[idx]
            if sd != 0 and np.sign(sd) == np.sign(sweep_sig):
                sd_strength = (
                    self._sd_zone_strength[idx] if idx < len(self._sd_zone_strength) else 0.5
                )
                base_score += 0.25 * sd * sd_strength

        # Phase 6 components
        if self.use_breaker_blocks:
            breaker_bull = float(self._breaker_bull[idx]) if idx < self._n_bars else 0.0
            breaker_bear = float(self._breaker_bear[idx]) if idx < self._n_bars else 0.0
            breaker_dir = breaker_bull - breaker_bear
            if breaker_dir != 0:
                base_score += (
                    breaker_dir
                    * w.get("breaker_block", 0.70)
                    * (1.2 if abs(breaker_dir) > 0 else 1.0)
                )

        if self.use_mitigation_blocks:
            mit_bull = float(self._mitigation_bull[idx]) if idx < self._n_bars else 0.0
            mit_bear = float(self._mitigation_bear[idx]) if idx < self._n_bars else 0.0
            mit_dir = mit_bull - mit_bear
            if mit_dir != 0:
                base_score += mit_dir * w.get("mitigation_block", 0.45)

        if self.use_rejection_blocks:
            rej_bull = float(self._rejection_bull[idx]) if idx < self._n_bars else 0.0
            rej_bear = float(self._rejection_bear[idx]) if idx < self._n_bars else 0.0
            rej_dir = rej_bull - rej_bear
            if rej_dir != 0:
                base_score += rej_dir * w.get("rejection_block", 0.55)

        # G5: Unicorn
        if (
            self.use_unicorn
            and hasattr(self, "_unicorn_signal")
            and idx < len(self._unicorn_signal)
        ):
            if self._unicorn_signal[idx] > 0:
                base_score += 0.30 * sweep_dir

        # G10: SFP
        if self.use_sfp and hasattr(self, "_sfp_dir") and idx < len(self._sfp_dir):
            sfp = self._sfp_dir[idx]
            if sfp != 0 and np.sign(sfp) == np.sign(sweep_sig):
                base_score += 0.20 * sfp

        # Confluence bonuses
        if len(confirmations) >= 1:
            base_score += self.confluence_bonus * sweep_dir
        if len(confirmations) >= 2:
            base_score += self.confluence_bonus * sweep_dir

        # Volume multiplier
        volume_mult = 1.0
        if self.volume_confirm and idx < self._n_bars:
            volume_mult = self._vol_mult[idx]

        # Pressure multiplier
        pressure_mult = 1.0
        if self.use_volume_pressure and hasattr(self, "_up_pressure") and idx < self._n_bars:
            up_p = self._up_pressure[idx]
            down_p = self._down_pressure[idx]
            bop = self._bop[idx]
            if sweep_dir > 0 and up_p > down_p:
                pressure_mult = 1.0 + 0.2 * (up_p - down_p)
            elif sweep_dir < 0 and down_p > up_p:
                pressure_mult = 1.0 + 0.2 * (down_p - up_p)
            elif abs(bop) > 0.3:
                pressure_mult = 1.0 + 0.1 * abs(bop)

        # Gate chain
        gate_mults = 1.0
        if (
            self.use_vol_gate
            and hasattr(self, "_vol_gate_mults")
            and idx < len(self._vol_gate_mults)
        ):
            gate_mults *= self._vol_gate_mults[idx]
        if (
            self.use_session_gate
            and hasattr(self, "_session_mults")
            and idx < len(self._session_mults)
        ):
            gate_mults *= self._session_mults[idx]
        if self.use_crash_gate and hasattr(self, "_crash_mults") and idx < len(self._crash_mults):
            gate_mults *= self._crash_mults[idx]
        if self.use_dow_gate and hasattr(self, "_dow_mults") and idx < len(self._dow_mults):
            gate_mults *= self._dow_mults[idx]
        if (
            self.use_frankfurt_gate
            and hasattr(self, "_frankfurt_mults")
            and idx < len(self._frankfurt_mults)
        ):
            gate_mults *= self._frankfurt_mults[idx]

        # G3: Judas Swing gate
        if (
            self.use_judas_swing
            and hasattr(self, "_judas_swing_dir")
            and idx < len(self._judas_swing_dir)
        ):
            js_dir = self._judas_swing_dir[idx]
            if js_dir == 0:
                base_score *= 0.4
            elif js_dir == int(sweep_dir):
                base_score *= 1.3
            else:
                base_score *= 0.0

        # A2: Swing point alignment bonus
        if self.use_swing_points and idx < self._n_bars:
            if sweep_dir > 0 and self._swing_low[idx]:
                base_score += self.swing_point_weight * sweep_dir
            elif sweep_dir < 0 and self._swing_high[idx]:
                base_score += self.swing_point_weight * sweep_dir

        # A1: ICT candlestick pattern bonus
        if self.use_ict_patterns and idx < self._n_bars:
            if sweep_dir > 0 and self._ict_bull[idx]:
                base_score += self.ict_pattern_weight * sweep_dir
            elif sweep_dir < 0 and self._ict_bear[idx]:
                base_score += self.ict_pattern_weight * sweep_dir

        # Miscellaneous multipliers
        cycle_mult = 1.0
        if self.use_90min_cycle and hasattr(self, "_cycle_mults") and idx < len(self._cycle_mults):
            cycle_mult = self._cycle_mults[idx]

        killzone_mult = 1.0
        if self.use_smc_sessions and hasattr(self, "_in_london_kz") and idx < self._n_bars:
            in_kz = self._in_london_kz[idx] or self._in_ny_kz[idx]
            killzone_mult = 1.25 if in_kz else 1.0

        market_kz_mult = 1.0
        if (
            self.use_smc_sessions
            and hasattr(self, "_market_kz_active")
            and idx < len(self._market_kz_active)
        ):
            if self._market_kz_active[idx]:
                market_kz_mult = 1.2

        deep_retrace_mult = 1.0
        if self.use_smc_retrace and hasattr(self, "_retrace_deepest") and idx < self._n_bars:
            if self._retrace_deepest[idx] > 78.6:
                deep_retrace_mult = 1.3

        htf_mult = 1.0
        if self.use_htf_gate and self._htf_bias[idx] != 0.0 and sweep_dir == self._htf_bias[idx]:
            htf_mult = 1.15
        elif self._htf_bias[idx] != 0.0 and sweep_dir == self._htf_bias[idx]:
            htf_mult = 1.15

        score = (
            base_score
            * volume_mult
            * pressure_mult
            * float(np.clip(gate_mults, 0.5, 2.0))
            * htf_mult
        )
        score *= cycle_mult * killzone_mult * market_kz_mult * deep_retrace_mult

        # G6: POI grading
        if self.use_poi_grading and hasattr(self, "_poi_grades") and idx < len(self._poi_grades):
            grade_score = self._poi_grades[idx]
            if grade_score == 4:
                score *= 1.15
            elif grade_score < 4:
                score *= 0.3

        # G9: Trade plan check
        if (
            self.use_trade_plan
            and hasattr(self, "_trade_plan_passes")
            and idx < len(self._trade_plan_passes)
        ):
            if not self._trade_plan_passes[idx]:
                return 0.0

        # ── Phase 21: Macro regime gates (final score modifiers) ──
        if self.use_vix_gate and self._vix_mults is not None and idx < len(self._vix_mults):
            score *= self._vix_mults[idx]
        if (
            self.use_yield_curve_gate
            and self._yield_curve_mults is not None
            and idx < len(self._yield_curve_mults)
        ):
            score *= self._yield_curve_mults[idx]

        return float(np.tanh(np.clip(score, -5.0, 5.0)))

    # ── Post-init precompute (needs all signals ready) ───────

    def _post_init_precompute(self) -> None:
        """Precompute arrays that depend on all signals being initialized."""
        n = self._n_bars
        self._poi_grades = np.zeros(n, dtype=np.int32)
        self._trade_plan_passes = np.ones(n, dtype=bool)

        if self.use_poi_grading:
            for i in range(3, n):
                zone_info = {
                    "level": float(self._df["Close"].iloc[i]),
                    "direction": int(
                        np.sign(
                            float(self._sweep_signals[i]) if i < len(self._sweep_signals) else 0
                        )
                    )
                    or 1,
                    "bar_index": i,
                    "zone_type": "sweep",
                }
                bos_ok = i < len(self._bos_signals) and self._bos_signals[i] != 0
                liq_ok = i > 20
                retest_ok = True
                dist_ok = True
                self._poi_grades[i] = sum([bos_ok, liq_ok, retest_ok, dist_ok])

        if self.use_trade_plan:
            try:
                from src.strategies.smc_trade_plan import run_trade_plan_check

                for i in range(10, n, 10):
                    precomputed = {
                        "htf_bias": self._htf_bias,
                        "bos_signals": self._bos_signals,
                        "sweep_signals": self._sweep_signals,
                        "choch_signals": self._choch_signals
                        if hasattr(self, "_choch_signals")
                        else self._bos_signals,
                        "fvg_proximity": self._fvg_proximity,
                        "judas_swing_signals": self._judas_swing_dir
                        if hasattr(self, "_judas_swing_dir")
                        else np.zeros(n, dtype=np.int8),
                        "mss_signals": self._msl_signals + self._msh_signals,
                        "in_killzone": np.zeros(n, dtype=bool),
                        "poi_grade_score": int(self._poi_grades[i])
                        if i < len(self._poi_grades)
                        else 0,
                        "current_price": float(self._df["Close"].iloc[i]) if i < n else 0.0,
                    }
                    passed, _, _ = run_trade_plan_check(
                        self.use_trade_plan,
                        i,
                        precomputed,
                        self.trade_plan_strictness,
                    )
                    self._trade_plan_passes[i : min(i + 10, n)] = passed
            except Exception:
                self._trade_plan_passes[:] = True

    def _calculate_size(self, price: float, atr: float) -> float:
        risk_capital = float(self.equity) * self.max_risk_pct
        risk_distance = self.trail_stop_atr * atr
        if risk_distance <= 0:
            risk_distance = price * 0.01
        pos_size = risk_capital / risk_distance
        min_size = float(self.equity) * 0.001 / max(price, 1e-10)
        return max(pos_size, min_size)

    def _init_swing_points(self) -> None:
        """A2: Precompute swing high/low arrays from price data."""
        try:
            from src.indicators.swing_point_detector import detect_swing_points

            high = self._df["High"].to_numpy(dtype=np.float64)
            low = self._df["Low"].to_numpy(dtype=np.float64)
            sh, sl = detect_swing_points(high, low, threshold=0.005)
            self._swing_high = sh
            self._swing_low = sl
            n_high = int(np.sum(sh > 0))
            n_low = int(np.sum(sl < 0))
            logger.info(
                "Swing points: %d highs, %d lows, weight=%.2f",
                n_high,
                n_low,
                self.swing_point_weight,
            )
        except Exception:
            logger.warning("Swing points init failed, disabling", exc_info=True)
            self._swing_high = np.zeros(self._n_bars, dtype=np.int8)
            self._swing_low = np.zeros(self._n_bars, dtype=np.int8)

    def _init_ict_patterns(self) -> None:
        """A1: Precompute 12 ICT candlestick pattern arrays."""
        try:
            from src.patterns.candlestick.ict_single_patterns import detect_all_twelve

            o = self._df["Open"].to_numpy(dtype=np.float64)
            h = self._df["High"].to_numpy(dtype=np.float64)
            lo = self._df["Low"].to_numpy(dtype=np.float64)
            c = self._df["Close"].to_numpy(dtype=np.float64)
            patterns = detect_all_twelve(o, h, lo, c)
            bullish_names = [
                "white_marubozu",
                "closing_white_marubozu",
                "opening_white_marubozu",
                "white_dragonfly_doji",
                "white_paper_umbrella",
                "white_spinning_top",
            ]
            bearish_names = [
                "black_marubozu",
                "closing_black_marubozu",
                "opening_black_marubozu",
                "black_gravestone_doji",
                "black_paper_umbrella",
                "black_spinning_top",
            ]
            bull = np.zeros(self._n_bars, dtype=np.int8)
            bear = np.zeros(self._n_bars, dtype=np.int8)
            for name in bullish_names:
                bull = np.where(patterns.get(name, 0) > 0, 1, bull).astype(np.int8)
            for name in bearish_names:
                bear = np.where(patterns.get(name, 0) > 0, 1, bear).astype(np.int8)
            self._ict_bull = bull
            self._ict_bear = bear
            n_bull = int(np.sum(bull))
            n_bear = int(np.sum(bear))
            logger.info(
                "ICT patterns: %d bullish, %d bearish, weight=%.2f",
                n_bull,
                n_bear,
                self.ict_pattern_weight,
            )
        except Exception:
            logger.warning("ICT patterns init failed, disabling", exc_info=True)
            self._ict_bull = np.zeros(self._n_bars, dtype=np.int8)
            self._ict_bear = np.zeros(self._n_bars, dtype=np.int8)

    def next(self) -> None:
        idx = len(self.data) - 1
        if idx < 20 or idx >= self._n_bars:
            return

        # G7: Daily loss limit check
        if self.use_daily_loss_limit:
            if hasattr(self.data.index[idx], "date"):
                current_date = self.data.index[idx].date()
            else:
                current_date = None
            if self._last_trade_date is not None and current_date is not None:
                if current_date != self._last_trade_date:
                    self._daily_pnl = 0.0
                    self._consecutive_losses = 0
            self._last_trade_date = current_date
            starting_equity = float(self.equity) + self._daily_pnl
            if starting_equity > 0 and self._daily_pnl < -self.daily_loss_limit * starting_equity:
                return

        score = self._compute_smc_score(idx)
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
                    elif current_close <= trail_sl:
                        self.position.close()
                        self._trail_high = 0.0
                        self._daily_pnl += (
                            (current_close - self._entry_price) * self.position.size
                            if self.position
                            else 0.0
                        )
                    elif score < self.exit_threshold:
                        self.position.close()
                        self._trail_high = 0.0
                        self._daily_pnl += (
                            (current_close - self._entry_price) * self.position.size
                            if self.position
                            else 0.0
                        )
                else:
                    if current_close <= trail_sl:
                        self.position.close()
                        self._trail_high = 0.0
                        self._daily_pnl += (
                            (current_close - self._entry_price) * self.position.size
                            if self.position
                            else 0.0
                        )
                    elif score < self.exit_threshold:
                        self.position.close()
                        self._trail_high = 0.0
                        self._daily_pnl += (
                            (current_close - self._entry_price) * self.position.size
                            if self.position
                            else 0.0
                        )
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
                    elif current_close >= trail_sl:
                        self.position.close()
                        self._trail_low = float("inf")
                        self._daily_pnl += (
                            (self._entry_price - current_close) * self.position.size
                            if self.position
                            else 0.0
                        )
                    elif score > -self.exit_threshold:
                        self.position.close()
                        self._trail_low = float("inf")
                        self._daily_pnl += (
                            (self._entry_price - current_close) * self.position.size
                            if self.position
                            else 0.0
                        )
                else:
                    if current_close >= trail_sl:
                        self.position.close()
                        self._trail_low = float("inf")
                        self._daily_pnl += (
                            (self._entry_price - current_close) * self.position.size
                            if self.position
                            else 0.0
                        )
                    elif score > -self.exit_threshold:
                        self.position.close()
                        self._trail_low = float("inf")
                        self._daily_pnl += (
                            (self._entry_price - current_close) * self.position.size
                            if self.position
                            else 0.0
                        )
        else:
            if score >= self.entry_threshold:
                order_size = min(self._calculate_size(current_close, atr), 0.95)
                self.buy(size=order_size)
                self._trail_high = current_close
                self._trail_low = float("inf")
                self._entry_price = current_close
                self._tp1_hit = False
            elif self.use_short and score <= -self.entry_threshold:
                order_size = min(self._calculate_size(current_close, atr), 0.95)
                self.sell(size=order_size)
                self._trail_low = current_close
                self._trail_high = 0.0
                self._entry_price = current_close
                self._tp1_hit = False
