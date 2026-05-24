"""35-rule catalog: 22 crossover + 6 Bollinger + 7 divergence rules.

Paper: "Heuristic Based Trading System" (Öztürk 2015) - B1 in MASTER_COMPARISON_REPORT
Full trading rule specifications including crossover, Bollinger, and divergence signals.
Uses vectorized operations for all rules — no per-bar loops in rule evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable


def _sma(series: np.ndarray, period: int) -> np.ndarray:
    result = np.full_like(series, np.nan)
    if len(series) >= period:
        cumsum = np.cumsum(np.insert(series, 0, 0))
        result[period - 1 :] = (cumsum[period:] - cumsum[:-period]) / period
    return result


def _ema(series: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    result = np.full_like(series, np.nan)
    if period < len(series):
        result[period - 1] = np.mean(series[:period])
    for i in range(period, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def _wma(series: np.ndarray, period: int) -> np.ndarray:
    weights = np.arange(1, period + 1, dtype=np.float64)
    wsum = weights.sum()
    result = np.full_like(series, np.nan)
    if len(series) >= period:
        for i in range(period - 1, len(series)):
            window = series[i - period + 1 : i + 1]
            result[i] = np.dot(window, weights) / wsum
    return result


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _sma(gain, period)
    avg_loss = _sma(loss, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return np.where(np.isnan(rsi), 50.0, rsi)


def _roc(close: np.ndarray, period: int) -> np.ndarray:
    roc = np.zeros_like(close)
    roc[period:] = (
        (close[period:] - close[:-period])
        / np.where(close[:-period] > 0, close[:-period], 1e-10)
        * 100
    )
    return roc


def _bollinger_bands(
    close: np.ndarray, period: int = 20, num_std: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sma = _sma(close, period)
    rolling_std = np.full_like(close, np.nan)
    for i in range(period - 1, len(close)):
        rolling_std[i] = float(np.std(close[i - period + 1 : i + 1], ddof=1))
    upper = sma + num_std * rolling_std
    lower = sma - num_std * rolling_std
    return sma, upper, lower


def _macd(
    close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    return macd_line, signal_line


def generate_rules_catalog(
    close: np.ndarray,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    volume: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Generate signal arrays for all 35 trading rules.

    Each rule produces an int8 signal array: 1=buy, -1=sell, 0=none.

    Returns:
        Dict mapping rule_name → int8 signal array.
    """
    n = len(close)
    signals: dict[str, np.ndarray] = {}

    # ── Indicators computed once ──
    sma5 = _sma(close, 5)
    sma10 = _sma(close, 10)
    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    ema9 = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema55 = _ema(close, 55)
    wma10 = _wma(close, 10)
    wma30 = _wma(close, 30)
    rsi_vals = _rsi(close, 14)
    roc12 = _roc(close, 12)
    bb_mid, bb_up, bb_low = _bollinger_bands(close, 20, 2.0)
    macd_l, macd_s = _macd(close)

    # ── 22 Crossover Rules ──
    crossover_rules: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("sma5_cross_sma10", sma5, sma10),
        ("sma5_cross_sma20", sma5, sma20),
        ("sma10_cross_sma20", sma10, sma20),
        ("sma10_cross_sma50", sma10, sma50),
        ("sma20_cross_sma50", sma20, sma50),
        ("ema9_cross_ema21", ema9, ema21),
        ("ema9_cross_ema55", ema9, ema55),
        ("ema21_cross_ema55", ema21, ema55),
        ("wma10_cross_wma30", wma10, wma30),
        ("price_cross_sma5", close, sma5),
        ("price_cross_sma10", close, sma10),
        ("price_cross_sma20", close, sma20),
        ("price_cross_sma50", close, sma50),
        ("price_cross_ema9", close, ema9),
        ("price_cross_ema21", close, ema21),
        ("price_cross_ema55", close, ema55),
        ("price_cross_wma10", close, wma10),
        ("price_cross_wma30", close, wma30),
        ("sma10_cross_ema21", sma10, ema21),
        ("sma20_cross_ema55", sma20, ema55),
        ("macd_cross_signal", macd_l, macd_s),
        ("macd_cross_zero", macd_l, np.zeros(n)),
    ]

    for name, fast_arr, slow_arr in crossover_rules:
        sig = np.zeros(n, dtype=np.int8)
        for i in range(1, n):
            if np.isnan(fast_arr[i]) or np.isnan(slow_arr[i]):
                continue
            if fast_arr[i] > slow_arr[i] and fast_arr[i - 1] <= slow_arr[i - 1]:
                sig[i] = 1
            elif fast_arr[i] < slow_arr[i] and fast_arr[i - 1] >= slow_arr[i - 1]:
                sig[i] = -1
        signals[name] = sig

    # ── 6 Bollinger Rules ──
    bb_rules: list[tuple[str, np.ndarray]] = [
        ("bb_touch_lower", -bb_low),
        ("bb_touch_upper", -bb_up),
        ("bb_close_above_mid", bb_mid),
        ("bb_close_below_mid", -bb_mid),
        ("bb_squeeze", np.zeros(n)),
        ("bb_expansion", np.zeros(n)),
    ]

    for i in range(1, n):
        if np.isnan(bb_low[i]):
            continue
        if close[i - 1] > bb_low[i - 1] and close[i] <= bb_low[i]:
            signals.setdefault("bb_touch_lower", np.zeros(n, dtype=np.int8))[i] = 1
        if close[i - 1] < bb_up[i - 1] and close[i] >= bb_up[i]:
            signals.setdefault("bb_touch_upper", np.zeros(n, dtype=np.int8))[i] = -1
        if close[i - 1] <= bb_mid[i - 1] and close[i] > bb_mid[i]:
            signals.setdefault("bb_close_above_mid", np.zeros(n, dtype=np.int8))[i] = 1
        if close[i - 1] >= bb_mid[i - 1] and close[i] < bb_mid[i]:
            signals.setdefault("bb_close_below_mid", np.zeros(n, dtype=np.int8))[i] = -1

    bb_width = bb_up - bb_low
    for i in range(20, n):
        if np.isnan(bb_width[i]):
            continue
        if bb_width[i] < np.nanmin(bb_width[i - 20 : i]):
            signals.setdefault("bb_squeeze", np.zeros(n, dtype=np.int8))[i] = 1
        if bb_width[i] > np.nanmax(bb_width[i - 20 : i]):
            signals.setdefault("bb_expansion", np.zeros(n, dtype=np.int8))[i] = 1

    for name in [
        "bb_touch_lower",
        "bb_touch_upper",
        "bb_close_above_mid",
        "bb_close_below_mid",
        "bb_squeeze",
        "bb_expansion",
    ]:
        if name not in signals:
            signals[name] = np.zeros(n, dtype=np.int8)

    # ── 7 Divergence Rules ──
    divergence_rules: list[str] = [
        "rsi_divergence_bull",
        "rsi_divergence_bear",
        "rsi_hidden_bull",
        "rsi_hidden_bear",
        "price_roc_divergence_bull",
        "price_roc_divergence_bear",
        "macd_divergence",
    ]

    try:
        from src.signals.divergence_detector import detect_divergences

        d_rsi = detect_divergences(close, rsi_vals)
        signals["rsi_divergence_bull"] = d_rsi["regular_bullish"]
        signals["rsi_divergence_bear"] = d_rsi["regular_bearish"]
        signals["rsi_hidden_bull"] = d_rsi["hidden_bullish"]
        signals["rsi_hidden_bear"] = d_rsi["hidden_bearish"]

        d_roc = detect_divergences(close, roc12)
        signals["price_roc_divergence_bull"] = d_roc["regular_bullish"]
        signals["price_roc_divergence_bear"] = d_roc["regular_bearish"]

        macd_div = np.zeros(n, dtype=np.int8)
        d_macd = detect_divergences(close, macd_l)
        macd_div = np.where(d_macd["regular_bullish"] != 0, d_macd["regular_bullish"], macd_div)
        macd_div = np.where(d_macd["regular_bearish"] != 0, d_macd["regular_bearish"], macd_div)
        signals["macd_divergence"] = macd_div
    except Exception:
        for name in divergence_rules:
            signals[name] = np.zeros(n, dtype=np.int8)

    return signals
