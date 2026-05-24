# -*- coding: utf-8 -*-
"""
TASC 2026.05 The AutoTune Filter (John F. Ehlers)

Computes the dominant cycle from rolling autocorrelation of high-pass filtered
price data. Uses the detected cycle as the center period of an adaptive band-pass
filter. Dynamically adjusts as the cycle changes.

Origin: TASC May 2026, PineScript v6 by PineCoders
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np


class DisplayMode(Enum):
    HP_FILTER = "F"
    MIN_CORR = "M"
    DOMINANT_CYCLE = "D"
    BAND_PASS = "B"


@dataclass
class AutoTuneResult:
    """AutoTune filter output for one evaluation point."""

    hp_filter: float
    min_correlation: float
    dominant_cycle: float
    band_pass: float
    display_value: float


def _hpf(src: np.ndarray, period: int, i: int) -> float:
    """Ehlers high-pass filter."""
    w = 1.414 * np.pi / period
    q = np.exp(-w)
    c1 = 2.0 * q * np.cos(w)
    c2 = q * q
    a0 = 0.25 * (1.0 + c1 + c2)

    if i < 4:
        return 0.0

    v1 = _hpf_memo.get((i - 1, period))
    v2 = _hpf_memo.get((i - 2, period))
    v1_val = v1 if v1 is not None else 0.0
    v2_val = v2 if v2 is not None else 0.0
    res = a0 * (src[i] - 2.0 * src[i - 1] + src[i - 2]) + c1 * v1_val - c2 * v2_val
    _hpf_memo[(i, period)] = res
    return float(res)


_hpf_memo: dict = {}


def _bpf(src: np.ndarray, period: float, bw: float, i: int) -> float:
    """Ehlers band-pass filter."""
    w0 = 2.0 * np.pi / max(period, 1)
    l1 = np.cos(w0)
    g1 = np.cos(w0 * bw)
    if g1 == 0:
        g1 = 1e-10
    s1 = 1.0 / g1 - np.sqrt(max(1.0 / (g1 * g1) - 1.0, 0.0))

    if i < 3:
        return 0.0

    v1 = _bpf_memo.get((i - 1, period, bw))
    v2 = _bpf_memo.get((i - 2, period, bw))
    v1_val = v1 if v1 is not None else 0.0
    v2_val = v2 if v2 is not None else 0.0
    res = 0.5 * (1.0 - s1) * (src[i] - src[i - 2]) + l1 * (1.0 + s1) * v1_val - s1 * v2_val
    _bpf_memo[(i, period, bw)] = res
    return float(res)


_bpf_memo: dict = {}


def auto_tune_filter(
    src: np.ndarray,
    window: int = 20,
    bw: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the AutoTune Filter (rolling autocorrelation cycle + band-pass).

    Args:
        src: Source price series (e.g. close prices)
        window: Window length for HP filter and ACF computation
        bw: Bandwidth for the passband

    Returns:
        Tuple of (hp, min_corr, dominant_cycle, band_pass) arrays
    """
    global _hpf_memo, _bpf_memo
    _hpf_memo = {}
    _bpf_memo = {}

    n = len(src)
    hp = np.full(n, np.nan)
    min_corr = np.full(n, np.nan)
    dominant_cycle = np.full(n, np.nan)
    band_pass = np.full(n, np.nan)

    hp_buffer = []

    for t in range(n):
        hp_val = _hpf(src, window, t)
        hp[t] = hp_val
        hp_buffer.append(hp_val)

        if len(hp_buffer) > window:
            hp_buffer = hp_buffer[-window:]

        if len(hp_buffer) >= window:
            buf = np.array(hp_buffer, dtype=np.float64)
            sx = np.sum(buf)
            sxx = np.sum(buf * buf)

            acf_vals = np.zeros(window)
            for lag in range(1, window + 1):
                lag_start = max(0, t - window + 1 - lag)
                lag_end = t + 1 - lag
                lag_len = lag_end - lag_start
                if lag_len >= window and t >= lag:
                    sy = float(np.sum(hp[lag_start:lag_end]))
                    syy = float(np.sum(hp[lag_start:lag_end] ** 2))
                    sxy = float(np.sum(buf * hp[lag_start:lag_end]))

                    cov = float(window) * sxy - sx * sy
                    vx = float(window) * sxx - sx * sx
                    vy = float(window) * syy - sy * sy

                    denom = np.sqrt(max(vx * vy, 1e-20))
                    acf_vals[lag - 1] = cov / denom if denom > 1e-15 else 1.0
                else:
                    acf_vals[lag - 1] = 1.0

            min_corr_val = float(np.min(acf_vals))
            min_idx = int(np.argmin(acf_vals))
            dc = (min_idx + 1) * 2

            if t > 0 and not np.isnan(dominant_cycle[t - 1]):
                prev_dc = int(dominant_cycle[t - 1])
                dc = max(min(dc, prev_dc + 2), prev_dc - 2)

            min_corr[t] = min_corr_val
            dominant_cycle[t] = float(dc)

            bp = _bpf(src, float(dc), bw, t)
            band_pass[t] = bp
        else:
            min_corr[t] = 1.0
            dominant_cycle[t] = float(window * 2)
            band_pass[t] = 0.0

    return hp, min_corr, dominant_cycle, band_pass


def compute_display_series(
    src: np.ndarray,
    window: int = 20,
    bw: float = 0.25,
    mode: DisplayMode = DisplayMode.BAND_PASS,
) -> np.ndarray:
    """
    Compute the display series for a given mode.

    Args:
        src: Source price series
        window: Window length
        bw: Bandwidth
        mode: Display mode (F=HP, M=MinCorr, D=DomCycle, B=BandPass)

    Returns:
        Display series array
    """
    hp, min_corr, dc, bp = auto_tune_filter(src, window, bw)

    if mode == DisplayMode.HP_FILTER:
        return hp
    elif mode == DisplayMode.MIN_CORR:
        return min_corr
    elif mode == DisplayMode.DOMINANT_CYCLE:
        return dc
    else:
        return bp


def rate_of_change(bp: np.ndarray) -> np.ndarray:
    """Compute rate of change of band-pass filter (zero-cross = mean-reversion signal)."""
    roc = np.zeros_like(bp)
    roc[1:] = bp[1:] - bp[:-1]
    return roc
