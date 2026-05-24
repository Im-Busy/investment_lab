# -*- coding: utf-8 -*-
"""
Asian Sweep + MSS + IFVG + HTF Bias Composite Indicator

Combines Asian session range, liquidity sweep, Market Structure Shift (MSS),
Fair Value Gap (FVG), Inverse FVG (IFVG), and Higher Timeframe (HTF) bias
into one composite trading signal.

Origin: TradingView indicators - AsianSweep_MSS_IFVG_HTF_Bias (v5) and
AsianLiquiditySweepTradingSystem (v5)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal

import numpy as np
import pandas as pd

from .asian_range import detect_asian_range, get_asian_range_for_bar
from .mss import detect_mss, MSSInfo
from .ifvg import detect_ifvg, IFVG, update_ifvg_status, get_active_ifvgs
from .liquidity_sweep import detect_liquidity_sweep, SweepInfo


@dataclass
class AsianSweepSignal:
    """Composite Asian sweep trading signal."""

    date: pd.Timestamp
    asian_high: float
    asian_low: float
    asian_range: float
    sweep_detected: bool
    sweep_direction: Optional[str]  # 'bullish', 'bearish'
    mss_detected: bool
    mss_direction: Optional[str]
    htf_bias: str  # 'Bullish', 'Bearish', 'Neutral'
    fvg_present: bool
    ifvg_present: bool
    ifvg_direction: Optional[str]
    entry_signal: str  # 'LONG', 'SHORT', 'NONE'
    confidence: float  # 0.0 - 1.0
    entry_price: float
    stop_loss: float
    take_profit: float


def _compute_htf_bias(
    df: pd.DataFrame,
    htf_period: int = 240,
    sma_length: int = 20,
) -> np.ndarray:
    """Compute higher timeframe bias as a numeric array.

    +1 = bullish, -1 = bearish, 0 = neutral
    """
    n = len(df)
    bias = np.zeros(n, dtype=np.int8)

    for i in range(1, n):
        close_current = float(df.iloc[i]["Close"])

        lookback = min(i, sma_length + htf_period)
        if lookback >= sma_length:
            recent_closes = df.iloc[max(0, i - lookback) : i + 1]["Close"].to_numpy(
                dtype=np.float64
            )
            sma_val = float(np.mean(recent_closes[-min(sma_length, len(recent_closes)) :]))
            if close_current > sma_val:
                bias[i] = 1
            elif close_current < sma_val:
                bias[i] = -1

    return bias


def _detect_fvg_at_bar(
    high: np.ndarray, low: np.ndarray, i: int, atr_val: float, fvg_threshold: float
) -> Optional[IFVG]:
    """Detect FVG at a specific bar index."""
    if i < 2:
        return None

    bull_fvg = low[i - 1] > high[i - 2] and high[i - 1] < low[i]
    bear_fvg = high[i - 1] < low[i - 2] and low[i - 1] > high[i]

    if not bull_fvg and not bear_fvg:
        return None

    if fvg_threshold > 0 and atr_val > 0:
        if bull_fvg:
            gap = low[i - 1] - high[i - 2]
            if gap < fvg_threshold * atr_val:
                return None
        else:
            gap = low[i - 2] - high[i - 1]
            if gap < fvg_threshold * atr_val:
                return None

    if bull_fvg:
        return IFVG(
            high=low[i - 1],
            low=high[i - 2],
            direction="bullish",
            start_bar=i - 2,
            end_bar=i,
            created_at=None,
        )
    else:
        return IFVG(
            high=low[i - 2],
            low=high[i - 1],
            direction="bearish",
            start_bar=i - 2,
            end_bar=i,
            created_at=None,
        )


def _detect_ifvg(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, i: int
) -> Optional[Tuple[str, float]]:
    """Detect Inverse FVG at bar i.

    Returns (direction, level_price) or None.
    """
    if i < 3:
        return None

    bull_fvg = low[i - 1] > high[i - 2] and high[i - 1] < low[i]
    bear_fvg = high[i - 1] < low[i - 2] and low[i - 1] > high[i]

    if bull_fvg and close[i] < low[i - 1] and close[i - 1] > high[i - 2]:
        return ("bullish", low[i - 1])
    if bear_fvg and close[i] > high[i - 1] and close[i - 1] < low[i - 2]:
        return ("bearish", high[i - 1])

    return None


def detect_asian_sweep_setup(
    df: pd.DataFrame,
    pivot_len: int = 5,
    fvg_min_atr: float = 0.0,
    fvg_lookback: int = 20,
    htf_period: int = 240,
    htf_sma_len: int = 20,
    sweep_buffer: float = 0.0,
    atr_period: int = 14,
) -> List[AsianSweepSignal]:
    """
    Detect Asian sweep + MSS + IFVG + HTF bias setups.

    Args:
        df: OHLCV DataFrame with datetime index
        pivot_len: Bars for pivot detection
        fvg_min_atr: Minimum FVG size as ATR multiplier
        fvg_lookback: Bars to look back for FVG/IFVG
        htf_period: Higher timeframe period (bars)
        htf_sma_len: SMA length for HTF bias
        sweep_buffer: Buffer pips for sweep detection
        atr_period: ATR period

    Returns:
        List of AsianSweepSignal entries
    """
    if len(df) < 100:
        return []

    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)
    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(df)

    htf_bias = _compute_htf_bias(df, htf_period, htf_sma_len)

    signals: List[AsianSweepSignal] = []

    asia_high = 0.0
    asia_low = 0.0
    swept_up = False
    swept_dn = False
    last_ph = np.nan
    last_pl = np.nan
    session_active = False

    for i in range(1, n):
        bar_time = df.index[i]
        hour = bar_time.hour if hasattr(bar_time, "hour") else 0

        if hour >= 0 and hour < 8:
            if not session_active:
                session_active = True
                asia_high = float(high[i])
                asia_low = float(low[i])
                swept_up = False
                swept_dn = False
            else:
                asia_high = max(asia_high, float(high[i]))
                asia_low = min(asia_low, float(low[i]))
        elif session_active and hour >= 8:
            session_active = False

            for j in range(i, min(n, i + fvg_lookback)):
                if not swept_up and not swept_dn:
                    if float(high[j]) > asia_high + sweep_buffer:
                        swept_up = True
                    if float(low[j]) < asia_low - sweep_buffer:
                        swept_dn = True

                if swept_up or swept_dn:
                    piv_high = _find_pivot_high(high, j, pivot_len)
                    piv_low = _find_pivot_low(low, j, pivot_len)

                    if piv_high is not None:
                        last_ph = piv_high
                    if piv_low is not None:
                        last_pl = piv_low

                    bull_mss = swept_dn and close[j] > last_ph if not np.isnan(last_ph) else False
                    bear_mss = swept_up and close[j] < last_pl if not np.isnan(last_pl) else False

                    bias_ok = False
                    if bull_mss and htf_bias[j] == 1:
                        bias_ok = True
                    elif bear_mss and htf_bias[j] == -1:
                        bias_ok = True

                    if bias_ok:
                        atr_val = 0.01 * float(close[j])
                        if j >= atr_period:
                            true_range = np.maximum(
                                high[max(0, j - atr_period) : j + 1]
                                - low[max(0, j - atr_period) : j + 1],
                                np.abs(
                                    high[max(0, j - atr_period) : j + 1]
                                    - np.roll(close[max(0, j - atr_period) : j + 1], 1)
                                ),
                            )
                            atr_val = float(np.mean(true_range[-atr_period:]))

                        ifvg = _detect_ifvg(high, low, close, j)

                        if bull_mss:
                            sl = asia_low - atr_val
                            tp = close[j] + (close[j] - sl) * 2.0
                            signals.append(
                                AsianSweepSignal(
                                    date=bar_time,
                                    asian_high=asia_high,
                                    asian_low=asia_low,
                                    asian_range=asia_high - asia_low,
                                    sweep_detected=swept_dn,
                                    sweep_direction="bearish" if swept_dn else "bullish",
                                    mss_detected=True,
                                    mss_direction="bullish",
                                    htf_bias="Bullish" if htf_bias[j] > 0 else "Bearish",
                                    fvg_present=ifvg is not None,
                                    ifvg_present=ifvg is not None,
                                    ifvg_direction=ifvg[0] if ifvg else None,
                                    entry_signal="LONG",
                                    confidence=0.7 if ifvg else 0.5,
                                    entry_price=float(close[j]),
                                    stop_loss=sl,
                                    take_profit=tp,
                                )
                            )
                        elif bear_mss:
                            sl = asia_high + atr_val
                            tp = close[j] - (sl - close[j]) * 2.0
                            signals.append(
                                AsianSweepSignal(
                                    date=bar_time,
                                    asian_high=asia_high,
                                    asian_low=asia_low,
                                    asian_range=asia_high - asia_low,
                                    sweep_detected=swept_up,
                                    sweep_direction="bullish" if swept_up else "bearish",
                                    mss_detected=True,
                                    mss_direction="bearish",
                                    htf_bias="Bullish" if htf_bias[j] > 0 else "Bearish",
                                    fvg_present=ifvg is not None,
                                    ifvg_present=ifvg is not None,
                                    ifvg_direction=ifvg[0] if ifvg else None,
                                    entry_signal="SHORT",
                                    confidence=0.7 if ifvg else 0.5,
                                    entry_price=float(close[j]),
                                    stop_loss=sl,
                                    take_profit=tp,
                                )
                            )

    return signals


def _find_pivot_high(arr: np.ndarray, idx: int, length: int) -> Optional[float]:
    """Find most recent pivot high."""
    if idx < length:
        return None
    for i in range(idx - length, length, -1):
        is_pivot = True
        pivot_val = arr[i]
        for j in range(1, length + 1):
            if arr[i - j] >= pivot_val or arr[i + j] >= pivot_val:
                is_pivot = False
                break
        if is_pivot:
            return float(pivot_val)
    return None


def _find_pivot_low(arr: np.ndarray, idx: int, length: int) -> Optional[float]:
    """Find most recent pivot low."""
    if idx < length:
        return None
    for i in range(idx - length, length, -1):
        is_pivot = True
        pivot_val = arr[i]
        for j in range(1, length + 1):
            if arr[i - j] <= pivot_val or arr[i + j] <= pivot_val:
                is_pivot = False
                break
        if is_pivot:
            return float(pivot_val)
    return None
