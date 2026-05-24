# -*- coding: utf-8 -*-
"""
Fair Value Gap Profile + Rolling POC

Builds a price-level histogram from Fair Value Gap (FVG) occurrences over a
lookback period. Shows where bullish/bearish FVGs cluster, their relative
strength, delta volume, and a rolling Point of Control.

Origin: TradingView indicator by BigBeluga, PineScript v6
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FVGProfileResult:
    """FVG profile analysis result for one evaluation point."""

    bins: np.ndarray
    bin_edges: np.ndarray
    bull_counts: np.ndarray
    bear_counts: np.ndarray
    bull_volumes: np.ndarray
    bear_volumes: np.ndarray
    strength_pct: np.ndarray
    delta_volume: np.ndarray
    poc_price: float
    poc_index: int
    total_gap_count: int
    max_activity: int


def detect_bull_fvg(high: np.ndarray, low: np.ndarray, i: int) -> bool:
    """Detect bullish FVG: bar i+2 high < bar i low."""
    if i + 2 >= len(high):
        return False
    return float(high[i + 2]) < float(low[i])


def detect_bear_fvg(high: np.ndarray, low: np.ndarray, i: int) -> bool:
    """Detect bearish FVG: bar i+2 low > bar i high."""
    if i + 2 >= len(high):
        return False
    return float(low[i + 2]) > float(high[i])


def compute_fvg_profile(
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    period: int = 500,
    bin_count: int = 40,
) -> FVGProfileResult:
    """
    Compute FVG profile over a lookback window.

    Args:
        high: High price array
        low: Low price array
        volume: Volume array
        period: Lookback window in bars
        bin_count: Number of price bins

    Returns:
        FVGProfileResult with profile data
    """
    n = len(high)
    if n == 0:
        return FVGProfileResult(
            bins=np.array([], dtype=int),
            bin_edges=np.array([]),
            bull_counts=np.array([], dtype=int),
            bear_counts=np.array([], dtype=int),
            bull_volumes=np.array([], dtype=float),
            bear_volumes=np.array([], dtype=float),
            strength_pct=np.array([], dtype=float),
            delta_volume=np.array([], dtype=float),
            total_gap_count=0,
            max_activity=0,
            poc_index=0,
            poc_price=0.0,
        )
    start = max(0, n - period)
    window_high = high[start:n]
    window_low = low[start:n]
    window_vol = volume[start:n]

    price_min = float(np.min(window_low))
    price_max = float(np.max(window_high))
    step = (price_max - price_min) / bin_count if bin_count > 0 else 0.001

    bull_counts = np.zeros(bin_count, dtype=int)
    bear_counts = np.zeros(bin_count, dtype=int)
    bull_volumes = np.zeros(bin_count, dtype=float)
    bear_volumes = np.zeros(bin_count, dtype=float)

    for i in range(len(window_high) - 2):
        actual_i = start + i
        if detect_bull_fvg(high, low, actual_i):
            avg = (float(high[actual_i + 2]) + float(low[actual_i])) / 2.0
            bin_idx = int(np.floor((avg - price_min) / step))
            if 0 <= bin_idx < bin_count:
                bull_counts[bin_idx] += 1
                bull_volumes[bin_idx] += float(window_vol[i + 1])

        if detect_bear_fvg(high, low, actual_i):
            avg = (float(low[actual_i + 2]) + float(high[actual_i])) / 2.0
            bin_idx = int(np.floor((avg - price_min) / step))
            if 0 <= bin_idx < bin_count:
                bear_counts[bin_idx] += 1
                bear_volumes[bin_idx] += float(window_vol[i + 1])

    max_activity = int(np.max(bull_counts + bear_counts))
    total_gaps = int(np.sum(bull_counts) + np.sum(bear_counts))

    if total_gaps > 0 and max_activity > 0:
        strength_pct = (bull_counts + bear_counts) / max_activity * 100.0
    else:
        strength_pct = np.zeros(bin_count)

    delta_volume = bull_volumes - bear_volumes

    if max_activity > 0:
        poc_idx = int(np.argmax(bull_counts + bear_counts))
        poc_price = price_min + step * poc_idx + step / 2.0
    else:
        poc_idx = 0
        poc_price = 0.0

    bin_edges = np.linspace(price_min, price_max, bin_count + 1)

    return FVGProfileResult(
        bins=np.arange(bin_count),
        bin_edges=bin_edges,
        bull_counts=bull_counts,
        bear_counts=bear_counts,
        bull_volumes=bull_volumes,
        bear_volumes=bear_volumes,
        strength_pct=strength_pct,
        delta_volume=delta_volume,
        poc_price=poc_price,
        poc_index=poc_idx,
        total_gap_count=total_gaps,
        max_activity=max_activity,
    )


def compute_rolling_poc(
    high: np.ndarray,
    low: np.ndarray,
    period: int = 500,
    bin_count: int = 40,
    smoothing: int = 10,
) -> np.ndarray:
    """
    Compute rolling Point of Control using a SMA of per-bar POC values.

    Args:
        high: High price array
        low: Low price array
        period: Lookback window
        bin_count: Number of price bins
        smoothing: SMA period for smoothing

    Returns:
        Array of rolling POC values (same length as input)
    """
    n = len(high)
    poc_raw = np.full(n, np.nan)

    for t in range(period, n):
        profile = compute_fvg_profile(
            high[: t + 1], low[: t + 1], np.ones(t + 1), period, bin_count
        )
        if profile.max_activity > 0:
            poc_raw[t] = profile.poc_price

    poc_series = pd.Series(poc_raw).rolling(window=smoothing, min_periods=1).mean().to_numpy()
    return poc_series


def compute_fvg_profile_df(
    df: pd.DataFrame,
    period: int = 500,
    bin_count: int = 40,
) -> FVGProfileResult:
    """
    Compute FVG profile from a DataFrame.

    Args:
        df: DataFrame with High, Low, Volume columns
        period: Lookback window
        bin_count: Number of price bins

    Returns:
        FVGProfileResult
    """
    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64) if "Volume" in df.columns else np.ones(len(df))
    return compute_fvg_profile(high, low, volume, period, bin_count)
