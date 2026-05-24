# -*- coding: utf-8 -*-
"""
Asian Liquidity Sweep Trading System

Full backtest-capable strategy based on Asian session liquidity sweeps +
MSS + FVG/IFVG zones. Tracks Asia range, detects sweep, waits for MSS,
then enters at limit orders on nearby FVG/IFVG zones with HTF bias
confirmation. Includes risk-managed exits (ATR-based SL, R:R targets).

Origin: TradingView strategy, PineScript v5
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal

import numpy as np
import pandas as pd

from ..indicators.asian_range import get_asian_range_for_bar
from ..indicators.mss import find_pivot_high, find_pivot_low


@dataclass
class Zone:
    """FVG/IFVG zone."""

    top: float
    bottom: float
    is_bull: bool
    is_mitigated: bool = False
    is_inverted: bool = False
    bar_index: int = 0
    active: bool = True


@dataclass
class AsianSweepTrade:
    """A completed or pending trade."""

    direction: str  # 'long' or 'short'
    entry_bar: int
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


@dataclass
class AsianSweepStrategyState:
    """State of the Asian sweep strategy."""

    asian_high: float = 0.0
    asian_low: float = 0.0
    in_session: bool = False
    swept_side: str = "None"
    sweep_confirmed: bool = False
    mss_bull: bool = False
    mss_bear: bool = False
    htf_bias: str = "Neutral"
    last_swing_high: float = float("nan")
    last_swing_low: float = float("nan")
    zones: List[Zone] = field(default_factory=list)
    pending_long: bool = False
    pending_short: bool = False


def run_asian_sweep_strategy(
    df: pd.DataFrame,
    pivot_len: int = 5,
    fvg_min_atr: float = 0.5,
    fvg_atr_len: int = 14,
    max_zone_lookback: int = 50,
    require_htf_bias: bool = True,
    require_ifvg: bool = True,
    entry_type: str = "Limit",
    risk_pct: float = 1.0,
    sl_padding_atr: float = 1.0,
    rr_target: float = 2.0,
    session_start_hour: int = 0,
    session_end_hour: int = 8,
    sweep_mode: str = "Wick",
    sweep_buffer: float = 0.0,
) -> pd.DataFrame:
    """
    Run the Asian Liquidity Sweep Trading System.

    Args:
        df: OHLCV DataFrame with datetime index
        pivot_len: Pivot left/right bars
        fvg_min_atr: Minimum FVG size as ATR multiplier
        fvg_atr_len: ATR length for FVG
        max_zone_lookback: Max bars to look back for zones
        require_htf_bias: Require HTF bias match
        require_ifvg: Require valid FVG/IFVG for entry
        entry_type: 'Market' or 'Limit'
        risk_pct: Risk % per trade
        sl_padding_atr: SL padding as ATR multiplier
        rr_target: Risk:Reward target
        session_start_hour: Asian session start hour
        session_end_hour: Asian session end hour
        sweep_mode: 'Wick' or 'Close'
        sweep_buffer: Sweep buffer in pips

    Returns:
        DataFrame with trade signals
    """
    n = len(df)
    if n < 100:
        return pd.DataFrame()

    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)
    close = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64) if "Volume" in df.columns else np.ones(n)

    atr = _compute_atr(high, low, close, fvg_atr_len)
    htf_bias = _compute_htf_bias_simple(close, 240, 20)

    state = AsianSweepStrategyState()
    trades: List[AsianSweepTrade] = []
    active_trade: Optional[AsianSweepTrade] = None

    signals = np.zeros(n, dtype=np.int8)
    entry_long = np.zeros(n, dtype=bool)
    entry_short = np.zeros(n, dtype=bool)
    exit_long = np.zeros(n, dtype=bool)
    exit_short = np.zeros(n, dtype=bool)
    stop_levels = np.full(n, np.nan)
    tp_levels = np.full(n, np.nan)

    for i in range(n):
        if hasattr(df.index[i], "hour"):
            hour = df.index[i].hour
        else:
            hour = i % 24

        atr_val = max(atr[i], 0.0001)

        if hour == session_start_hour and not state.in_session:
            state.in_session = True
            state.asian_high = float(high[i])
            state.asian_low = float(low[i])
            state.swept_side = "None"
            state.sweep_confirmed = False
            state.mss_bull = False
            state.mss_bear = False
            state.pending_long = False
            state.pending_short = False

        if state.in_session:
            state.asian_high = max(state.asian_high, float(high[i]))
            state.asian_low = min(state.asian_low, float(low[i]))

        if hour == session_end_hour and state.in_session:
            state.in_session = False

        if not state.in_session and not state.sweep_confirmed and state.asian_high > 0:
            low_check = float(low[i]) if sweep_mode == "Wick" else float(close[i])
            high_check = float(high[i]) if sweep_mode == "Wick" else float(close[i])

            if low_check < state.asian_low - sweep_buffer:
                state.swept_side = "Low"
                state.sweep_confirmed = True
            elif high_check > state.asian_high + sweep_buffer:
                state.swept_side = "High"
                state.sweep_confirmed = True

        if i >= pivot_len:
            ph = find_pivot_high(df, i, pivot_len)
            pl = find_pivot_low(df, i, pivot_len)
            if ph is not None:
                state.last_swing_high = ph.price
            if pl is not None:
                state.last_swing_low = pl.price

        state.htf_bias = (
            "Bullish" if htf_bias[i] > 0 else "Bearish" if htf_bias[i] < 0 else "Neutral"
        )

        if state.sweep_confirmed:
            if state.swept_side == "Low" and not state.mss_bull:
                if not np.isnan(state.last_swing_high) and close[i] > state.last_swing_high:
                    state.mss_bull = True
            if state.swept_side == "High" and not state.mss_bear:
                if not np.isnan(state.last_swing_low) and close[i] < state.last_swing_low:
                    state.mss_bear = True

        _detect_and_update_zones(
            high, low, close, i, atr_val, fvg_min_atr, max_zone_lookback, state
        )

        bias_ok = not require_htf_bias or (
            (state.mss_bull and state.htf_bias == "Bullish")
            or (state.mss_bear and state.htf_bias == "Bearish")
        )

        if state.mss_bull and bias_ok and active_trade is None:
            bull_zone = _find_nearest_zone(state.zones, "bullish")
            if not require_ifvg or bull_zone is not None:
                if entry_type == "Limit" and bull_zone is not None:
                    entry_p = bull_zone.bottom
                else:
                    entry_p = float(close[i])

                stop_dist = max(entry_p - state.asian_low, atr_val * 2)
                sl = entry_p - stop_dist
                if sl_padding_atr > 0:
                    sl = min(sl, entry_p - atr_val * sl_padding_atr)
                tp = entry_p + (entry_p - sl) * rr_target

                active_trade = AsianSweepTrade(
                    direction="long",
                    entry_bar=i,
                    entry_price=entry_p,
                    stop_loss=sl,
                    take_profit=tp,
                )
                entry_long[i] = True
                stop_levels[i] = sl
                tp_levels[i] = tp
                state.mss_bull = False

        if state.mss_bear and bias_ok and active_trade is None:
            bear_zone = _find_nearest_zone(state.zones, "bearish")
            if not require_ifvg or bear_zone is not None:
                if entry_type == "Limit" and bear_zone is not None:
                    entry_p = bear_zone.top
                else:
                    entry_p = float(close[i])

                stop_dist = max(state.asian_high - entry_p, atr_val * 2)
                sl = entry_p + stop_dist
                if sl_padding_atr > 0:
                    sl = max(sl, entry_p + atr_val * sl_padding_atr)
                tp = entry_p - (sl - entry_p) * rr_target

                active_trade = AsianSweepTrade(
                    direction="short",
                    entry_bar=i,
                    entry_price=entry_p,
                    stop_loss=sl,
                    take_profit=tp,
                )
                entry_short[i] = True
                stop_levels[i] = sl
                tp_levels[i] = tp
                state.mss_bear = False

        if active_trade is not None:
            if active_trade.direction == "long":
                if low[i] <= active_trade.stop_loss:
                    active_trade.exit_bar = i
                    active_trade.exit_price = active_trade.stop_loss
                    active_trade.pnl = active_trade.stop_loss - active_trade.entry_price
                    active_trade.pnl_pct = active_trade.pnl / active_trade.entry_price * 100.0
                    trades.append(active_trade)
                    exit_long[i] = True
                    active_trade = None
                elif high[i] >= active_trade.take_profit:
                    active_trade.exit_bar = i
                    active_trade.exit_price = active_trade.take_profit
                    active_trade.pnl = active_trade.take_profit - active_trade.entry_price
                    active_trade.pnl_pct = active_trade.pnl / active_trade.entry_price * 100.0
                    trades.append(active_trade)
                    exit_long[i] = True
                    active_trade = None
            else:
                if high[i] >= active_trade.stop_loss:
                    active_trade.exit_bar = i
                    active_trade.exit_price = active_trade.stop_loss
                    active_trade.pnl = active_trade.entry_price - active_trade.stop_loss
                    active_trade.pnl_pct = active_trade.pnl / active_trade.entry_price * 100.0
                    trades.append(active_trade)
                    exit_short[i] = True
                    active_trade = None
                elif low[i] <= active_trade.take_profit:
                    active_trade.exit_bar = i
                    active_trade.exit_price = active_trade.take_profit
                    active_trade.pnl = active_trade.entry_price - active_trade.take_profit
                    active_trade.pnl_pct = active_trade.pnl / active_trade.entry_price * 100.0
                    trades.append(active_trade)
                    exit_short[i] = True
                    active_trade = None

        signals[i] = 1 if entry_long[i] else -1 if entry_short[i] else 0

    result = pd.DataFrame(
        {
            "signal": signals,
            "entry_long": entry_long,
            "entry_short": entry_short,
            "exit_long": exit_long,
            "exit_short": exit_short,
            "stop_loss": stop_levels,
            "take_profit": tp_levels,
        },
        index=df.index,
    )

    return result


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    atr = np.zeros(n)
    if n > 1:
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
        atr[0] = float(high[0] - low[0]) if n > 0 else 1.0
        alpha = 1.0 / period
        for i in range(1, n):
            atr[i] = alpha * tr[i - 1] + (1 - alpha) * atr[i - 1]
    return atr


def _compute_htf_bias_simple(close: np.ndarray, period: int, sma_len: int) -> np.ndarray:
    n = len(close)
    bias = np.zeros(n, dtype=np.int8)
    for i in range(sma_len, n):
        sma = float(np.mean(close[i - sma_len + 1 : i + 1]))
        if close[i] > sma:
            bias[i] = 1
        elif close[i] < sma:
            bias[i] = -1
    return bias


def _detect_and_update_zones(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    i: int,
    atr_val: float,
    fvg_min_atr: float,
    max_lookback: int,
    state: AsianSweepStrategyState,
) -> None:
    if i < 3:
        return

    is_bull_fvg = low[i] > high[i - 2] and (low[i] - high[i - 2] > atr_val * fvg_min_atr)
    is_bear_fvg = high[i] < low[i - 2] and (low[i - 2] - high[i] > atr_val * fvg_min_atr)

    if is_bull_fvg:
        state.zones.append(Zone(top=high[i - 2], bottom=low[i], is_bull=True, bar_index=i - 2))
    if is_bear_fvg:
        state.zones.append(Zone(top=low[i - 2], bottom=high[i], is_bull=False, bar_index=i - 2))

    for z in state.zones:
        if z.active and not z.is_mitigated:
            if z.is_bull and low[i] <= z.top and high[i] >= z.bottom:
                z.is_mitigated = True
            elif not z.is_bull and high[i] >= z.bottom and low[i] <= z.top:
                z.is_mitigated = True

    state.zones = [z for z in state.zones if i - z.bar_index < max_lookback * 4]


def _find_nearest_zone(zones: List[Zone], direction: str) -> Optional[Zone]:
    active = [z for z in zones if z.active and not z.is_mitigated]
    if direction == "bullish":
        active = [z for z in active if z.is_bull]
    else:
        active = [z for z in active if not z.is_bull]
    return active[-1] if active else None
