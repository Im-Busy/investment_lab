"""Divergence detection — regular and hidden bullish/bearish with multiple indicator support.

Paper: "Divergence Detection Algorithms" (B10 in MASTER_COMPARISON_REPORT)
Works with: RSI, MFI, DeMarker, Ultimate Oscillator, StochRSI, CCI, MACD.

Four types detected:
  Regular Bullish:  price lower low  + indicator higher low   → reversal up
  Regular Bearish:  price higher high + indicator lower high   → reversal down
  Hidden Bullish:   price higher low  + indicator lower low    → continuation up
  Hidden Bearish:   price lower high  + indicator higher high  → continuation down
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Optional


def _find_swings(
    series: np.ndarray,
    lookback: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Find local swing highs and lows in a series.

    A bar is a swing high if its value > all values in [i-lookback, i+lookback].
    A bar is a swing low if its value < all values in [i-lookback, i+lookback].
    """
    n = len(series)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        window = series[i - lookback : i + lookback + 1]
        if series[i] == window.max() and np.sum(window == series[i]) == 1:
            swing_high[i] = True
        if series[i] == window.min() and np.sum(window == series[i]) == 1:
            swing_low[i] = True

    return swing_high, swing_low


def detect_divergences(
    price: np.ndarray,
    indicator: np.ndarray,
    lookback: int = 5,
    max_bars_between: int = 30,
) -> dict[str, np.ndarray]:
    """Detect all four divergence types between price and an indicator.

    Args:
        price: Array of close prices.
        indicator: Array of indicator values (RSI, MFI, etc.).
        lookback: Bars to look each side for swing detection (default 5).
        max_bars_between: Maximum bars between swing points for a pair (default 30).

    Returns:
        Dict with four int8 arrays: reg_bull, reg_bear, hidden_bull, hidden_bear.
    """
    n = len(price)
    p_high, p_low = _find_swings(price, lookback)
    i_high, i_low = _find_swings(indicator, lookback)

    reg_bull = np.zeros(n, dtype=np.int8)
    reg_bear = np.zeros(n, dtype=np.int8)
    hidden_bull = np.zeros(n, dtype=np.int8)
    hidden_bear = np.zeros(n, dtype=np.int8)

    p_high_idxs = np.where(p_high)[0]
    p_low_idxs = np.where(p_low)[0]
    i_high_idxs = np.where(i_high)[0]
    i_low_idxs = np.where(i_low)[0]

    for pi in range(1, len(p_low_idxs)):
        prev_p = p_low_idxs[pi - 1]
        cur_p = p_low_idxs[pi]
        if cur_p - prev_p > max_bars_between:
            continue

        for ii in range(1, len(i_low_idxs)):
            prev_i = i_low_idxs[ii - 1]
            cur_i = i_low_idxs[ii]
            if cur_i - prev_i > max_bars_between:
                continue

            if abs(prev_p - prev_i) <= lookback and abs(cur_p - cur_i) <= lookback:
                if price[cur_p] < price[prev_p] and indicator[cur_i] > indicator[prev_i]:
                    reg_bull[cur_p] = 1
                elif price[cur_p] > price[prev_p] and indicator[cur_i] < indicator[prev_i]:
                    hidden_bull[cur_p] = 1

    for pi in range(1, len(p_high_idxs)):
        prev_p = p_high_idxs[pi - 1]
        cur_p = p_high_idxs[pi]
        if cur_p - prev_p > max_bars_between:
            continue

        for ii in range(1, len(i_high_idxs)):
            prev_i = i_high_idxs[ii - 1]
            cur_i = i_high_idxs[ii]
            if cur_i - prev_i > max_bars_between:
                continue

            if abs(prev_p - prev_i) <= lookback and abs(cur_p - cur_i) <= lookback:
                if price[cur_p] > price[prev_p] and indicator[cur_i] < indicator[prev_i]:
                    reg_bear[cur_p] = -1
                elif price[cur_p] < price[prev_p] and indicator[cur_i] > indicator[prev_i]:
                    hidden_bear[cur_p] = -1

    return {
        "regular_bullish": reg_bull,
        "regular_bearish": reg_bear,
        "hidden_bullish": hidden_bull,
        "hidden_bearish": hidden_bear,
    }


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    if len(close) > period:
        avg_gain[period - 1] = np.mean(gain[:period])
        avg_loss[period - 1] = np.mean(loss[:period])
    for i in range(period, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return np.where(np.isnan(rsi), 50.0, rsi)


def compute_mfi(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Compute Money Flow Index."""
    typical = (high + low + close) / 3.0
    money_flow = typical * volume
    pos_flow = np.where(typical > np.roll(typical, 1), money_flow, 0.0)
    neg_flow = np.where(typical < np.roll(typical, 1), money_flow, 0.0)
    pos_sum = np.full_like(close, np.nan)
    neg_sum = np.full_like(close, np.nan)
    if len(close) > period:
        pos_sum[period - 1] = np.sum(pos_flow[:period])
        neg_sum[period - 1] = np.sum(neg_flow[:period])
    for i in range(period, len(close)):
        pos_sum[i] = pos_sum[i - 1] - pos_flow[i - period] + pos_flow[i]
        neg_sum[i] = neg_sum[i - 1] - neg_flow[i - period] + neg_flow[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        mr = np.where(neg_sum > 0, pos_sum / neg_sum, 1.0)
        mfi = 100.0 - (100.0 / (1.0 + mr))
    return np.where(np.isnan(mfi), 50.0, mfi)


def detect_all_divergences(
    df: pd.DataFrame,
    lookback: int = 5,
    max_bars_between: int = 30,
) -> dict[str, np.ndarray]:
    """Detect divergences using RSI, MFI, and raw price as indicators.

    Args:
        df: DataFrame with OHLCV columns.
        lookback: Swing detection window.
        max_bars_between: Max bars between paired swing points.

    Returns:
        Dict with combined divergence signal arrays per type.
    """
    close = df["Close"].to_numpy(dtype=np.float64)
    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)

    rsi = compute_rsi(close)
    mfi = compute_mfi(high, low, close, volume)

    all_reg_bull = np.zeros(len(close), dtype=np.int8)
    all_reg_bear = np.zeros(len(close), dtype=np.int8)
    all_hidden_bull = np.zeros(len(close), dtype=np.int8)
    all_hidden_bear = np.zeros(len(close), dtype=np.int8)

    for ind_name, ind_vals in [("RSI", rsi), ("MFI", mfi)]:
        d = detect_divergences(close, ind_vals, lookback, max_bars_between)
        all_reg_bull = np.where(d["regular_bullish"] != 0, d["regular_bullish"], all_reg_bull)
        all_reg_bear = np.where(d["regular_bearish"] != 0, d["regular_bearish"], all_reg_bear)
        all_hidden_bull = np.where(d["hidden_bullish"] != 0, d["hidden_bullish"], all_hidden_bull)
        all_hidden_bear = np.where(d["hidden_bearish"] != 0, d["hidden_bearish"], all_hidden_bear)

    return {
        "regular_bullish": all_reg_bull,
        "regular_bearish": all_reg_bear,
        "hidden_bullish": all_hidden_bull,
        "hidden_bearish": all_hidden_bear,
    }
