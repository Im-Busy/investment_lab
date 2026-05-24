# -*- coding: utf-8 -*-
"""
Swing Structure Forecast

Detects confirmed swing pivots, builds ATR-proportioned S/R zones, and
projects a probabilistic forecast beam for the next swing leg using
statistical aggregation of historical swing magnitudes.

Origin: TradingView indicator by BOSWaves, PineScript v6
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np


@dataclass
class SwingPivot:
    """A confirmed swing pivot."""

    price: float
    bar_index: int
    is_high: bool


@dataclass
class SRZone:
    """Support/Resistance zone."""

    price: float
    start_idx: int
    is_resistance: bool
    broken: bool = False
    width: float = 0.0


@dataclass
class ForecastResult:
    """Forecast result for the current swing."""

    origin_price: float
    origin_idx: int
    target_price: float
    target_idx: int
    projected_pct: float
    projected_bars: float
    std_dev: float
    band_half: float
    is_bearish: bool
    method: str


def _find_swing_high(high: np.ndarray, length: int) -> np.ndarray:
    """Find rolling swing highs using highest over window."""
    n = len(high)
    result = np.full(n, np.nan)
    for i in range(length - 1, n):
        window = high[i - length + 1 : i + 1]
        if high[i] == np.max(window):
            result[i] = high[i]
    return result


def _find_swing_low(low: np.ndarray, length: int) -> np.ndarray:
    """Find rolling swing lows using lowest over window."""
    n = len(low)
    result = np.full(n, np.nan)
    for i in range(length - 1, n):
        window = low[i - length + 1 : i + 1]
        if low[i] == np.min(window):
            result[i] = low[i]
    return result


def detect_swings(
    high: np.ndarray,
    low: np.ndarray,
    swing_length: int = 16,
) -> Tuple[List[SwingPivot], List[float], List[float]]:
    """
    Detect confirmed swing pivots and record swing statistics.

    Args:
        high: High price array
        low: Low price array
        swing_length: Number of bars for swing detection

    Returns:
        Tuple of (swings, percentages, durations)
    """
    n = len(high)
    hi_markers = _find_swing_high(high, swing_length)
    lo_markers = _find_swing_low(low, swing_length)

    swings: List[SwingPivot] = []
    pcts: List[float] = []
    durs: List[float] = []

    dir_up = False
    last_hi = SwingPivot(0.0, 0, True)
    last_lo = SwingPivot(0.0, 0, False)

    for i in range(1, n):
        if not np.isnan(hi_markers[i]):
            dir_up = True
        if not np.isnan(lo_markers[i]):
            dir_up = False

        if not np.isnan(hi_markers[i - 1]) and high[i] < hi_markers[i - 1]:
            last_hi = SwingPivot(float(high[i - 1]), i - 1, True)

        if not np.isnan(lo_markers[i - 1]) and low[i] > lo_markers[i - 1]:
            last_lo = SwingPivot(float(low[i - 1]), i - 1, False)

        if i > 1:
            prev_hi = not np.isnan(hi_markers[i - 1])
            cur_hi = not np.isnan(hi_markers[i])
            prev_lo = not np.isnan(lo_markers[i - 1])
            cur_lo = not np.isnan(lo_markers[i])

            direction_change = (cur_hi and prev_lo and not dir_up) or (
                cur_lo and prev_hi and dir_up
            )

            if direction_change and last_hi.price > 0 and last_lo.price > 0:
                if dir_up:
                    pct = abs((last_hi.price - last_lo.price) / last_lo.price * 100.0)
                else:
                    pct = abs((last_lo.price - last_hi.price) / last_hi.price * 100.0)
                bars = abs(last_hi.bar_index - last_lo.bar_index)
                pcts.append(pct)
                durs.append(float(bars))

    return swings, pcts, durs


def forecast_next_swing(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    swing_length: int = 16,
    samples: int = 20,
    method: str = "Weighted",
    atr_period: int = 200,
    zone_width_atr: float = 0.3,
) -> Optional[ForecastResult]:
    """
    Forecast the next swing leg using historical swing statistics.

    Args:
        high: High price array
        low: Low price array
        close: Close price array
        swing_length: Bars for swing detection
        samples: Number of recent swings to use
        method: 'Weighted', 'Average', or 'Median'
        atr_period: ATR period for zone width
        zone_width_atr: Zone width as ATR multiplier

    Returns:
        ForecastResult or None if insufficient data
    """
    _, pcts, durs = detect_swings(high, low, swing_length)

    if len(pcts) < 2:
        return None

    recent_pcts = pcts[-samples:]
    recent_durs = durs[-samples:]

    if method == "Weighted":
        weights = np.arange(1, len(recent_pcts) + 1, dtype=np.float64)
        fPct = float(np.average(recent_pcts, weights=weights))
        fBars = float(np.average(recent_durs, weights=weights))
    elif method == "Median":
        fPct = float(np.median(recent_pcts))
        fBars = float(np.median(recent_durs))
    else:
        fPct = float(np.mean(recent_pcts))
        fBars = float(np.mean(recent_durs))

    variance = float(np.var(recent_pcts, ddof=0))
    std_dev = float(np.sqrt(variance))

    hi_markers = _find_swing_high(high, swing_length)
    lo_markers = _find_swing_low(low, swing_length)
    n = len(high)

    is_bear = (
        not np.isnan(hi_markers[-1]) if not np.isnan(hi_markers[-1]) else np.isnan(lo_markers[-1])
    )
    last_hi_idx = n - 1
    last_lo_idx = n - 1

    for i in range(n - 1, 0, -1):
        if not np.isnan(hi_markers[i]) and last_hi_idx == n - 1:
            last_hi_idx = i
        if not np.isnan(lo_markers[i]) and last_lo_idx == n - 1:
            last_lo_idx = i

    atr_val = _compute_atr(high, low, close, atr_period, n - 1)

    if is_bear:
        origin = float(high[last_hi_idx])
        origin_idx = last_hi_idx
        target = origin * (1.0 - fPct / 100.0)
    else:
        origin = float(low[last_lo_idx])
        origin_idx = last_lo_idx
        target = origin * (1.0 + fPct / 100.0)

    target_idx = n - 1 + 5
    band_half = max(origin * std_dev / 100.0, atr_val * 0.1)

    return ForecastResult(
        origin_price=origin,
        origin_idx=origin_idx,
        target_price=target,
        target_idx=target_idx,
        projected_pct=fPct,
        projected_bars=fBars,
        std_dev=std_dev,
        band_half=band_half,
        is_bearish=is_bear,
        method=method,
    )


def _compute_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, idx: int
) -> float:
    """Compute ATR at a given index."""
    if idx < period:
        return float(np.mean(high[1:] - low[1:])) if idx > 1 else 0.01

    tr = np.zeros(idx + 1)
    for i in range(1, idx + 1):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    return float(np.mean(tr[max(0, idx - period) : idx + 1]))


def compute_fib_levels(
    origin: float,
    target: float,
    fib_levels: List[float] = [1.0, 1.272, 1.618, 2.0, 2.618],
) -> List[float]:
    """Compute Fibonacci extension levels from origin and target."""
    full_move = target - origin
    return [origin + full_move * level for level in fib_levels]
