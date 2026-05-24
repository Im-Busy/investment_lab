# -*- coding: utf-8 -*-
"""
Whale Liquidity and Absorption Profile

Samples intrabar volume data to reconstruct buying/selling, delta, and
absorption activity. Groups activity into horizontal price bins with
strong/weak separation via percentile filtering.

Origin: TradingView indicator by AlgoAlpha, PineScript v6
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class WhaleProfileResult:
    """Whale liquidity profile analysis result."""

    bin_edges: np.ndarray
    strong_bull: np.ndarray
    weak_bull: np.ndarray
    weak_bear: np.ndarray
    strong_bear: np.ndarray
    absorption: np.ndarray
    delta: np.ndarray
    total_volume: np.ndarray
    poc_index: int
    poc_volume: float
    value_area_bins: np.ndarray
    strength_threshold: float
    max_bin_volume: float


def _estimate_intrabar_data(
    df_rows: np.ndarray,
    granularity: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate intrabar direction, volume, and close from OHLC bars."""
    open_p = df_rows[:, 0]
    high_p = df_rows[:, 1]
    low_p = df_rows[:, 2]
    close_p = df_rows[:, 3]
    volume_p = df_rows[:, 4]

    n = len(open_p)
    all_dirs = []
    all_vols = []
    all_prices = []

    for i in range(n):
        o = float(open_p[i])
        h = float(high_p[i])
        lo = float(low_p[i])
        c = float(close_p[i])
        v = float(volume_p[i])

        seg_vol = v / granularity if granularity > 0 else v
        is_green = c > o

        if is_green:
            body_range = max(c - o, 0)
            lower_wick = max(o - lo, 0)
            upper_wick = max(h - c, 0)
        else:
            body_range = max(o - c, 0)
            lower_wick = max(c - lo, 0)
            upper_wick = max(h - o, 0)

        total_range = max(h - lo, 0.0001)
        body_segs = max(1, int(granularity * body_range / total_range))
        lower_segs = max(0, int(granularity * lower_wick / total_range))
        upper_segs = max(0, int(granularity * upper_wick / total_range))
        remaining = granularity - body_segs - lower_segs - upper_segs
        body_segs += max(0, remaining)

        for _ in range(lower_segs):
            price = lo + (o - lo) * 0.5 if is_green else lo + (c - lo) * 0.5
            all_dirs.append(1 if is_green else -1)
            all_vols.append(seg_vol)
            all_prices.append(price)

        for _ in range(body_segs):
            price = o + (c - o) * 0.5 if is_green else o - (o - c) * 0.5
            all_dirs.append(1 if is_green else -1)
            all_vols.append(seg_vol)
            all_prices.append(price)

        for _ in range(upper_segs):
            price = c + (h - c) * 0.5 if is_green else o + (h - o) * 0.5
            all_dirs.append(1 if is_green else -1)
            all_vols.append(seg_vol)
            all_prices.append(price)

    return np.array(all_dirs), np.array(all_vols), np.array(all_prices)


def compute_whale_profile(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    open_: np.ndarray,
    lookback: int = 200,
    profile_bins: int = 35,
    strength_filter: int = 97,
    strong_only: bool = False,
    detect_value_area: bool = False,
    value_area_pct: int = 30,
) -> WhaleProfileResult:
    """
    Compute whale liquidity and absorption profile.

    Args:
        high: High price array
        low: Low price array
        close: Close price array
        volume: Volume array
        open_: Open price array
        lookback: Lookback window for profile
        profile_bins: Number of price bins
        strength_filter: Percentile threshold for strong classification
        strong_only: If True, hide weak volume segments
        detect_value_area: Enable value area detection
        value_area_pct: Value area percentage

    Returns:
        WhaleProfileResult
    """
    start = max(0, len(high) - lookback)
    window_data = np.column_stack(
        [
            open_[start:],
            high[start:],
            low[start:],
            close[start:],
            volume[start:],
        ]
    )

    dirs, vols, prices = _estimate_intrabar_data(window_data, granularity=10)

    if len(dirs) == 0:
        return WhaleProfileResult(
            bin_edges=np.linspace(0, 1, profile_bins + 1),
            strong_bull=np.zeros(profile_bins),
            weak_bull=np.zeros(profile_bins),
            weak_bear=np.zeros(profile_bins),
            strong_bear=np.zeros(profile_bins),
            absorption=np.zeros(profile_bins),
            delta=np.zeros(profile_bins),
            total_volume=np.zeros(profile_bins),
            poc_index=0,
            poc_volume=0.0,
            value_area_bins=np.zeros(profile_bins, dtype=bool),
            strength_threshold=0.0,
            max_bin_volume=0.0,
        )

    if len(vols) > 0:
        sorted_vols = np.sort(vols)
        pct_idx = min(len(sorted_vols) - 1, max(0, int(len(sorted_vols) * strength_filter / 100.0)))
        strength_threshold = float(sorted_vols[pct_idx])
    else:
        strength_threshold = 0.0

    min_price = float(np.min(prices))
    max_price = float(np.max(prices))
    price_range = max(max_price - min_price, 0.0001)
    step = price_range / profile_bins

    bin_edges = np.linspace(min_price, max_price, profile_bins + 1)

    strong_bull = np.zeros(profile_bins, dtype=float)
    weak_bull = np.zeros(profile_bins, dtype=float)
    weak_bear = np.zeros(profile_bins, dtype=float)
    strong_bear = np.zeros(profile_bins, dtype=float)
    absorption = np.zeros(profile_bins, dtype=float)

    for i in range(len(dirs)):
        d = dirs[i]
        v = vols[i]
        p = prices[i]

        bin_idx = int(np.floor((p - min_price) / step))
        bin_idx = max(0, min(profile_bins - 1, bin_idx))
        is_strong = v >= strength_threshold

        if d > 0 and is_strong:
            strong_bull[bin_idx] += v
        elif d > 0:
            weak_bull[bin_idx] += v
        elif is_strong:
            strong_bear[bin_idx] += v
        else:
            weak_bear[bin_idx] += v

    if strong_only:
        weak_bull.fill(0)
        weak_bear.fill(0)

    buy_vol = strong_bull + weak_bull
    sell_vol = strong_bear + weak_bear
    delta = buy_vol - sell_vol
    total_vol = strong_bull + weak_bull + weak_bear + strong_bear

    max_bin_vol = float(np.max(total_vol))
    poc_idx = int(np.argmax(total_vol))
    poc_vol = float(total_vol[poc_idx])

    value_area_bins = np.zeros(profile_bins, dtype=bool)
    if detect_value_area and np.sum(total_vol) > 0:
        target_va_vol = np.sum(total_vol) * value_area_pct / 100.0
        value_area_bins[poc_idx] = True
        acc_vol = total_vol[poc_idx]
        lower = poc_idx
        upper = poc_idx

        while acc_vol < target_va_vol and (lower > 0 or upper < profile_bins - 1):
            lower_cand = total_vol[lower - 1] if lower > 0 else -1.0
            upper_cand = total_vol[upper + 1] if upper < profile_bins - 1 else -1.0

            if upper_cand >= lower_cand:
                upper += 1
                acc_vol += upper_cand
                value_area_bins[upper] = True
            else:
                lower -= 1
                acc_vol += lower_cand
                value_area_bins[lower] = True

    return WhaleProfileResult(
        bin_edges=bin_edges,
        strong_bull=strong_bull,
        weak_bull=weak_bull,
        weak_bear=weak_bear,
        strong_bear=strong_bear,
        absorption=absorption,
        delta=delta,
        total_volume=total_vol,
        poc_index=poc_idx,
        poc_volume=poc_vol,
        value_area_bins=value_area_bins,
        strength_threshold=strength_threshold,
        max_bin_volume=max_bin_vol,
    )
