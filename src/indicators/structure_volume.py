# -*- coding: utf-8 -*-
"""
Market Structure Volume Profiles

Anchors volume profiles to market structure events (BoS / CHoCH) instead of
session time. Each completed structural leg gets its own volume profile,
CVD, POC, and Value Area.

Origin: TradingView indicator by KioseffTrading, PineScript v6
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class StructureProfile:
    """Volume profile for one structural leg."""

    highest: float
    lowest: float
    start_price: float
    end_price: float
    buy_volume: float
    sell_volume: float
    cvd: float
    poc_price: float
    level_arr: np.ndarray
    vol_at_level_buy: np.ndarray
    vol_at_level_sell: np.ndarray


def compute_structure_cvd(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    atr_period: int = 14,
    reset_mode: str = "CHoCH",
) -> Tuple[np.ndarray, np.ndarray, List[StructureProfile]]:
    """
    Compute Cumulative Volume Delta anchored to market structure.

    Args:
        high, low, close, volume: OHLCV arrays
        atr_period: ATR period for structure detection
        reset_mode: 'CHoCH', 'BoS+CHoCH', 'Day', 'Week'

    Returns:
        Tuple of (CVD array, price array, list of StructureProfile)
    """
    n = len(close)
    cvd = np.zeros(n)
    profiles: List[StructureProfile] = []

    buy_vol = np.where(close > np.roll(close, 1), volume, 0)
    sell_vol = np.where(close < np.roll(close, 1), volume, 0)
    buy_vol[0] = 0
    sell_vol[0] = 0

    atr = _compute_atr(high, low, close, atr_period)

    swing_len = max(3, atr_period // 2)
    swings_high = _find_swing_highs(high, swing_len)
    swings_low = _find_swing_lows(low, swing_len)

    current_cvd = 0.0
    profile_start = 0
    profile_high = float(high[0])
    profile_low = float(low[0])
    profile_start_price = float(close[0])
    profile_vol_buy = 0.0

    for i in range(n):
        current_cvd += buy_vol[i] - sell_vol[i]
        cvd[i] = current_cvd

        if i > 0:
            profile_high = max(profile_high, float(high[i]))
            profile_low = min(profile_low, float(low[i]))

        is_structure_end = False
        if swing_len <= i < n - swing_len:
            if swings_high[i] and i > profile_start:
                is_structure_end = True
            elif swings_low[i] and i > profile_start:
                is_structure_end = True

        if is_structure_end and i > profile_start + swing_len:
            n_bins = 20
            price_range = max(profile_high - profile_low, 0.0001)
            step = price_range / n_bins
            level_arr = np.linspace(profile_low, profile_high, n_bins + 1)
            vol_buy_at_level = np.zeros(n_bins)
            vol_sell_at_level = np.zeros(n_bins)

            for j in range(profile_start, i):
                bin_idx = int((close[j] - profile_low) / step)
                bin_idx = max(0, min(n_bins - 1, bin_idx))
                if close[j] > close[j - 1] if j > profile_start else True:
                    vol_buy_at_level[bin_idx] += volume[j]
                else:
                    vol_sell_at_level[bin_idx] += volume[j]

            total_vol = vol_buy_at_level + vol_sell_at_level
            poc_idx = int(np.argmax(total_vol)) if np.sum(total_vol) > 0 else 0
            poc_price = float(level_arr[poc_idx] + step / 2.0)

            profiles.append(
                StructureProfile(
                    highest=profile_high,
                    lowest=profile_low,
                    start_price=profile_start_price,
                    end_price=float(close[i]),
                    buy_volume=float(np.sum(vol_buy_at_level)),
                    sell_volume=float(np.sum(vol_sell_at_level)),
                    cvd=current_cvd,
                    poc_price=poc_price,
                    level_arr=level_arr,
                    vol_at_level_buy=vol_buy_at_level,
                    vol_at_level_sell=vol_sell_at_level,
                )
            )

            if reset_mode in ("CHoCH", "BoS+CHoCH"):
                current_cvd = 0.0

            profile_start = i
            profile_high = float(high[i])
            profile_low = float(low[i])
            profile_start_price = float(close[i])

    return cvd, atr, profiles


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Compute ATR."""
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


def _find_swing_highs(high: np.ndarray, length: int) -> np.ndarray:
    """Find confirmed swing highs."""
    n = len(high)
    result = np.zeros(n, dtype=bool)
    for i in range(length, n - length):
        pivot_val = high[i]
        is_pivot = True
        for j in range(1, length + 1):
            if high[i - j] >= pivot_val or high[i + j] >= pivot_val:
                is_pivot = False
                break
        result[i] = is_pivot
    return result


def _find_swing_lows(low: np.ndarray, length: int) -> np.ndarray:
    """Find confirmed swing lows."""
    n = len(low)
    result = np.zeros(n, dtype=bool)
    for i in range(length, n - length):
        pivot_val = low[i]
        is_pivot = True
        for j in range(1, length + 1):
            if low[i - j] <= pivot_val or low[i + j] <= pivot_val:
                is_pivot = False
                break
        result[i] = is_pivot
    return result
