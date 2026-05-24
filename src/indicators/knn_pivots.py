# -*- coding: utf-8 -*-
"""
Machine Learning Pivot Points (KNN)

Uses K-Nearest Neighbors to classify current price slope against historical
pivot signatures. When the current linear regression slope matches past
pivot-high slopes, it flags an approaching pivot.

Origin: TradingView indicator by Steversteves, PineScript v6
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np


@dataclass
class KNNPivotResult:
    """KNN pivot classification result."""

    classification: str  # 'Neutral', 'Approaching Pivot High', 'Approaching Pivot Low'
    confidence: float  # 0-100
    distance_to_high: float
    distance_to_low: float
    transition: bool  # True when classification changed from Neutral to signal


def _get_historical_slope(src: np.ndarray, start_idx: int, window_len: int) -> float:
    """Compute linear regression slope over window."""
    if start_idx + window_len > len(src):
        return 0.0

    y = src[start_idx : start_idx + window_len].astype(np.float64)
    x = np.arange(window_len, dtype=np.float64)

    sum_x = float(np.sum(x))
    sum_y = float(np.sum(y))
    sum_xy = float(np.sum(x * y))
    sum_x2 = float(np.sum(x * x))

    denom = window_len * sum_x2 - sum_x**2
    if denom == 0:
        return 0.0

    return (window_len * sum_xy - sum_x * sum_y) / denom


def _rolling_slope(src: np.ndarray, length: int) -> np.ndarray:
    """Rolling linear regression slope."""
    n = len(src)
    result = np.full(n, np.nan)

    for i in range(length, n):
        y = src[i - length + 1 : i + 1].astype(np.float64)
        x = np.arange(length, dtype=np.float64)

        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        num = float(np.sum((x - x_mean) * (y - y_mean)))
        den = float(np.sum((x - x_mean) ** 2))

        if den > 0:
            result[i] = num / den

    return result


def _find_pivots(arr: np.ndarray, left: int, right: int, mode: str) -> List[Tuple[int, float]]:
    """Find confirmed pivot points."""
    pivots = []
    n = len(arr)

    for i in range(left, n - right):
        is_pivot = True
        pivot_val = arr[i]

        if mode == "high":
            for j in range(1, left + 1):
                if arr[i - j] >= pivot_val:
                    is_pivot = False
                    break
            if is_pivot:
                for j in range(1, right + 1):
                    if arr[i + j] >= pivot_val:
                        is_pivot = False
                        break
        else:
            for j in range(1, left + 1):
                if arr[i - j] <= pivot_val:
                    is_pivot = False
                    break
            if is_pivot:
                for j in range(1, right + 1):
                    if arr[i + j] <= pivot_val:
                        is_pivot = False
                        break

        if is_pivot:
            pivots.append((i, float(pivot_val)))

    return pivots


def _knn_distance(current_val: float, history: np.ndarray, k: int) -> float:
    """Compute average distance to k nearest neighbors."""
    if len(history) == 0:
        return float("inf")
    if len(history) < k:
        k = len(history)

    distances = np.abs(history - current_val)
    nearest = np.sort(distances)[:k]
    return float(np.mean(nearest))


def compute_knn_pivots(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    pivot_bars: int = 10,
    knn_clusters: int = 2,
    pivot_window: int = 20,
) -> List[KNNPivotResult]:
    """
    Compute KNN pivot classification over the time series.

    Args:
        high: High price array
        low: Low price array
        close: Close price array
        pivot_bars: Bars left/right for pivot detection
        knn_clusters: KNN clusters (k)
        pivot_window: Pivot window length for slope lookback

    Returns:
        List of KNNPivotResult per bar
    """
    n = len(high)
    results: List[KNNPivotResult] = []

    piv_hi_slopes: List[float] = []
    piv_lo_slopes: List[float] = []

    piv_highs = _find_pivots(high, pivot_bars, pivot_bars, "high")
    piv_lows = _find_pivots(low, pivot_bars, pivot_bars, "low")

    for idx, pval in piv_highs:
        slope = _get_historical_slope(high, idx, pivot_window)
        piv_hi_slopes.append(slope)

    for idx, pval in piv_lows:
        slope = _get_historical_slope(low, idx, pivot_window)
        piv_lo_slopes.append(slope)

    hi_arr = np.array(piv_hi_slopes, dtype=np.float64)
    lo_arr = np.array(piv_lo_slopes, dtype=np.float64)

    slopes = _rolling_slope(close, pivot_window)
    prev_classification = "Neutral"

    for i in range(n):
        current_slope = slopes[i]
        if np.isnan(current_slope):
            results.append(
                KNNPivotResult(
                    classification="Neutral",
                    confidence=0.0,
                    distance_to_high=float("inf"),
                    distance_to_low=float("inf"),
                    transition=False,
                )
            )
            continue

        dist_hi = _knn_distance(current_slope, hi_arr, knn_clusters)
        dist_lo = _knn_distance(current_slope, lo_arr, knn_clusters)

        total = dist_hi + dist_lo
        if total > 0 and dist_hi < dist_lo:
            confidence = 100.0 * (1.0 - dist_hi / total)
            classification = "Approaching Pivot High"
        elif total > 0 and dist_lo < dist_hi:
            confidence = 100.0 * (1.0 - dist_lo / total)
            classification = "Approaching Pivot Low"
        else:
            classification = "Neutral"
            confidence = 0.0

        transition = prev_classification == "Neutral" and classification != "Neutral"
        prev_classification = classification

        results.append(
            KNNPivotResult(
                classification=classification,
                confidence=confidence,
                distance_to_high=dist_hi,
                distance_to_low=dist_lo,
                transition=transition,
            )
        )

    return results


def compute_backtest_stats(
    results: List[KNNPivotResult],
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
) -> dict:
    """Compute backtest statistics from KNN pivot predictions."""
    pivot_high_moves: List[float] = []
    pivot_low_moves: List[float] = []
    piv_hi_pass = 0
    piv_lo_pass = 0

    for i, r in enumerate(results):
        if r.transition:
            if r.classification == "Approaching Pivot High":
                if i > 0:
                    move = high[i] - low[i - 1]
                    pivot_low_moves.append(max(0, move))
            elif r.classification == "Approaching Pivot Low":
                if i > 0:
                    move = (low[i] - high[i - 1]) / high[i - 1] * 100.0
                    pivot_high_moves.append(max(0, move))

    return {
        "low_success_rate": 100.0 * piv_lo_pass / max(1, len(pivot_low_moves)),
        "high_success_rate": 100.0 * piv_hi_pass / max(1, len(pivot_high_moves)),
        "avg_low_move": float(np.mean(pivot_low_moves)) if pivot_low_moves else 0.0,
        "avg_high_move": float(np.mean(pivot_high_moves)) if pivot_high_moves else 0.0,
        "pivot_low_events": len(pivot_low_moves),
        "pivot_high_events": len(pivot_high_moves),
    }
