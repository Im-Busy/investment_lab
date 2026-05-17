"""
PineScript to Python Helper Functions

Implements common PineScript built-in functions in Python/NumPy for
converting TradingView PineScript strategies to Python pattern detectors.

All functions operate on NumPy arrays for performance.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def crossover(series1: np.ndarray, series2: np.ndarray) -> np.ndarray:
    """
    Detect where series1 crosses ABOVE series2.

    True when: series1[i] > series2[i] AND series1[i-1] <= series2[i-1]

    Args:
        series1: First series
        series2: Second series

    Returns:
        Boolean array of crossover points
    """
    result = np.zeros(len(series1), dtype=bool)
    for i in range(1, len(series1)):
        if (
            np.isnan(series1[i])
            or np.isnan(series2[i])
            or np.isnan(series1[i - 1])
            or np.isnan(series2[i - 1])
        ):
            continue
        result[i] = series1[i] > series2[i] and series1[i - 1] <= series2[i - 1]
    return result


def crossunder(series1: np.ndarray, series2: np.ndarray) -> np.ndarray:
    """
    Detect where series1 crosses BELOW series2.

    True when: series1[i] < series2[i] AND series1[i-1] >= series2[i-1]

    Args:
        series1: First series
        series2: Second series

    Returns:
        Boolean array of crossunder points
    """
    result = np.zeros(len(series1), dtype=bool)
    for i in range(1, len(series1)):
        if (
            np.isnan(series1[i])
            or np.isnan(series2[i])
            or np.isnan(series1[i - 1])
            or np.isnan(series2[i - 1])
        ):
            continue
        result[i] = series1[i] < series2[i] and series1[i - 1] >= series2[i - 1]
    return result


def nz(value: float, replacement: float = 0.0) -> float:
    """
    Replace NaN with replacement value (PineScript's nz()).

    Args:
        value: Input value
        replacement: Value to use if input is NaN

    Returns:
        value if not NaN, else replacement
    """
    if value is None or np.isnan(value):
        return replacement
    return value


def highest(source: np.ndarray, length: int) -> np.ndarray:
    """
    Rolling maximum over `length` bars (PineScript's ta.highest()).

    Args:
        source: Input array
        length: Lookback period

    Returns:
        Array of rolling maxima
    """
    result = np.full(len(source), np.nan)
    for i in range(length - 1, len(source)):
        result[i] = np.nanmax(source[i - length + 1 : i + 1])
    return result


def lowest(source: np.ndarray, length: int) -> np.ndarray:
    """
    Rolling minimum over `length` bars (PineScript's ta.lowest()).

    Args:
        source: Input array
        length: Lookback period

    Returns:
        Array of rolling minima
    """
    result = np.full(len(source), np.nan)
    for i in range(length - 1, len(source)):
        result[i] = np.nanmin(source[i - length + 1 : i + 1])
    return result


def barssince(condition: np.ndarray) -> np.ndarray:
    """
    Number of bars since condition was true (PineScript's ta.barssince()).

    Args:
        condition: Boolean array

    Returns:
        Array of bar counts since last true condition
    """
    result = np.full(len(condition), np.nan)
    last_true = -1
    for i in range(len(condition)):
        if condition[i]:
            last_true = i
            result[i] = 0
        elif last_true >= 0:
            result[i] = i - last_true
    return result


def change(source: np.ndarray, length: int = 1) -> np.ndarray:
    """
    Difference between current value and value `length` bars ago
    (PineScript's ta.change()).

    Args:
        source: Input array
        length: Lookback period

    Returns:
        Array of differences
    """
    result = np.full(len(source), np.nan)
    for i in range(length, len(source)):
        result[i] = source[i] - source[i - length]
    return result


def valuewhen(condition: np.ndarray, source: np.ndarray, occurrence: int = 0) -> np.ndarray:
    """
    Get source value at the `occurrence`-th most recent true condition
    (PineScript's ta.valuewhen()).

    Args:
        condition: Boolean array
        source: Value array
        occurrence: 0 = most recent, 1 = second most recent, etc.

    Returns:
        Array of values when condition was true
    """
    result = np.full(len(source), np.nan)
    true_indices = []
    for i in range(len(source)):
        if condition[i]:
            true_indices.append(i)
        if len(true_indices) > occurrence:
            result[i] = source[true_indices[-(occurrence + 1)]]
    return result


def supertrend(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, factor: float = 3.0, period: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supertrend indicator (PineScript's ta.supertrend()).

    Returns (supertrend_line, direction) where:
    - supertrend_line: the Supertrend value
    - direction: +1 = uptrend, -1 = downtrend

    Implementation matches TradingView's Supertrend.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        factor: ATR multiplier (default 3.0)
        period: ATR period (default 10)

    Returns:
        Tuple of (supertrend_line, direction) arrays
    """
    n = len(close)
    tr = np.zeros(n)
    atr = np.zeros(n)
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend_line = np.zeros(n)
    direction = np.zeros(n, dtype=np.int8)

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    tr[0] = np.nan
    atr[period - 1] = np.nanmean(tr[1:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    for i in range(period - 1, n):
        src = (high[i] + low[i]) / 2.0
        upper_band[i] = src + factor * atr[i]
        lower_band[i] = src - factor * atr[i]

    for i in range(period, n):
        upper_band[i] = (
            upper_band[i]
            if upper_band[i] < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]
            else upper_band[i - 1]
        )
        lower_band[i] = (
            lower_band[i]
            if lower_band[i] > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]
            else lower_band[i - 1]
        )

    for i in range(period, n):
        direction[i] = (
            1
            if close[i] > upper_band[i - 1]
            else -1
            if close[i] < lower_band[i - 1]
            else direction[i - 1]
        )
        supertrend_line[i] = lower_band[i] if direction[i] == 1 else upper_band[i]

    return supertrend_line, direction


def sar(
    high: np.ndarray,
    low: np.ndarray,
    start: float = 0.02,
    increment: float = 0.02,
    maximum: float = 0.2,
) -> np.ndarray:
    """
    Parabolic SAR indicator (PineScript's ta.sar()).

    Implementation matches TradingView's Parabolic SAR.

    Args:
        high: High prices
        low: Low prices
        start: Starting acceleration factor (default 0.02)
        increment: Acceleration increment (default 0.02)
        maximum: Maximum acceleration factor (default 0.2)

    Returns:
        Array of SAR values
    """
    n = len(high)
    sar_values = np.full(n, np.nan)

    sar_values[0] = low[0]
    is_uptrend = True
    ep = high[0]
    af = start

    for i in range(1, n):
        prev_sar = sar_values[i - 1]
        if is_uptrend:
            sar_values[i] = prev_sar + af * (ep - prev_sar)
            if low[i] < sar_values[i]:
                is_uptrend = False
                sar_values[i] = ep
                ep = low[i]
                af = start
            elif high[i] > ep:
                ep = high[i]
                af = min(af + increment, maximum)
        else:
            sar_values[i] = prev_sar + af * (ep - prev_sar)
            if high[i] > sar_values[i]:
                is_uptrend = True
                sar_values[i] = ep
                ep = high[i]
                af = start
            elif low[i] < ep:
                ep = low[i]
                af = min(af + increment, maximum)

    return sar_values


def dmi(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14, adx_smoothing: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Directional Movement Index (PineScript's ta.dmi()).

    Returns (di_plus, di_minus, adx).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: DI period (default 14)
        adx_smoothing: ADX smoothing period (default 14)

    Returns:
        Tuple of (di_plus, di_minus, adx) arrays
    """
    n = len(close)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    def _rma(arr: np.ndarray, p: int) -> np.ndarray:
        result = np.full(n, np.nan)
        valid = arr[1 : p + 1]
        result[p] = np.nanmean(valid) if len(valid) > 0 else np.nan
        for i in range(p + 1, n):
            result[i] = (result[i - 1] * (p - 1) + arr[i]) / p
        return result

    tr_rma = _rma(tr, period)
    plus_dm_rma = _rma(plus_dm, period)
    minus_dm_rma = _rma(minus_dm, period)

    di_plus = 100.0 * plus_dm_rma / tr_rma
    di_minus = 100.0 * minus_dm_rma / tr_rma

    dx = 100.0 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = _rma(dx, adx_smoothing)

    return di_plus, di_minus, adx


def macd(
    source: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD indicator (PineScript's ta.macd()).

    Returns (macd_line, signal_line, histogram).

    Args:
        source: Price series (typically close)
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram) arrays
    """
    fast_ema = _ema(source, fast)
    slow_ema = _ema(source, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def vwap_simple(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    """
    Cumulative VWAP (PineScript's ta.vwap() - cumulative, not session-reset).

    VWAP = cumsum(typical_price * volume) / cumsum(volume)
    typical_price = (high + low + close) / 3

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume

    Returns:
        Array of cumulative VWAP values
    """
    typical = (high + low + close) / 3.0
    tp_vol = typical * volume
    cum_tp_vol = np.cumsum(tp_vol)
    cum_vol = np.cumsum(volume)
    vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, np.nan)
    return vwap


def stdev(source: np.ndarray, length: int) -> np.ndarray:
    """
    Rolling standard deviation (PineScript's ta.stdev()).

    Uses sample std (ddof=1) like TradingView.

    Args:
        source: Input array
        length: Lookback period

    Returns:
        Array of rolling standard deviations
    """
    result = np.full(len(source), np.nan)
    for i in range(length - 1, len(source)):
        result[i] = np.std(source[i - length + 1 : i + 1], ddof=1)
    return result


def qqe(
    rsi_values: np.ndarray,
    qqe_factor: float = 4.238,
    rsi_smoothing: int = 5,
    threshold: float = 10.0,
    rsi_period: int = 14,
) -> np.ndarray:
    """
    QQE (Quantitative Qualitative Estimation) indicator.

    Returns trend direction: 1 = bullish, -1 = bearish.

    Implementation matches the QQE logic from the Momentum ZigZag strategy.

    Args:
        rsi_values: Pre-computed RSI array
        qqe_factor: QQE smoothing factor (default 4.238)
        rsi_smoothing: RSI smoothing period (default 5)
        threshold: Threshold for signal (default 10)
        rsi_period: RSI period used to compute wilders_period (default 14)

    Returns:
        Array of direction values: 1 (bull), -1 (bear), 0 (neutral)
    """
    n = len(rsi_values)
    wilders_period = rsi_period * 2 - 1

    rsi_ma = _ema(rsi_values, rsi_smoothing)
    atr_rsi = np.abs(np.roll(rsi_ma, 1) - rsi_ma)
    atr_rsi[0] = 0.0
    ma_atr_rsi = _ema(atr_rsi, wilders_period)
    dar = _ema(ma_atr_rsi, wilders_period) * qqe_factor

    result = np.zeros(n, dtype=np.int8)
    trend = np.zeros(n, dtype=np.int8)
    longband = np.full(n, np.nan)
    shortband = np.full(n, np.nan)

    for i in range(1, n):
        newlongband = rsi_ma[i] - dar[i] if not np.isnan(dar[i]) else np.nan
        newshortband = rsi_ma[i] + dar[i] if not np.isnan(dar[i]) else np.nan

        if not np.isnan(rsi_ma[i - 1]) and not np.isnan(longband[i - 1]):
            longband[i] = (
                max(longband[i - 1], newlongband)
                if (rsi_ma[i - 1] > longband[i - 1] and rsi_ma[i] > longband[i - 1])
                else newlongband
            )
        else:
            longband[i] = newlongband

        if not np.isnan(rsi_ma[i - 1]) and not np.isnan(shortband[i - 1]):
            shortband[i] = (
                min(shortband[i - 1], newshortband)
                if (rsi_ma[i - 1] < shortband[i - 1] and rsi_ma[i] < shortband[i - 1])
                else newshortband
            )
        else:
            shortband[i] = newshortband

        # Trend detection
        trend[i] = trend[i - 1] if trend[i - 1] != 0 else 1
        if i > 0 and not np.isnan(rsi_ma[i]) and not np.isnan(shortband[i - 1]):
            if rsi_ma[i] > shortband[i - 1] and rsi_ma[i - 1] <= shortband[i - 1]:
                trend[i] = 1
        if i > 0 and not np.isnan(rsi_ma[i]) and not np.isnan(longband[i - 1]):
            if rsi_ma[i] < longband[i - 1] and rsi_ma[i - 1] >= longband[i - 1]:
                trend[i] = -1

    # QQE crosses
    qqe_crosses = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if trend[i] == 1 and trend[i - 1] == -1:
            qqe_crosses[i] = 1
        elif trend[i] == -1 and trend[i - 1] == 1:
            qqe_crosses[i] = -1

    return qqe_crosses


def _ema(source: np.ndarray, period: int) -> np.ndarray:
    """Internal EMA calculation."""
    result = np.full(len(source), np.nan)
    if len(source) < period:
        return result
    multiplier = 2.0 / (period + 1.0)
    seed_slice = source[:period]
    valid = seed_slice[~np.isnan(seed_slice)]
    if len(valid) > 0:
        result[period - 1] = np.mean(valid)
    for i in range(period, len(source)):
        result[i] = (source[i] - result[i - 1]) * multiplier + result[i - 1]
    return result


def crossover_value(series1: float, series2: float, prev1: float, prev2: float) -> bool:
    """Single-point crossover check."""
    return series1 > series2 and prev1 <= prev2


def crossunder_value(series1: float, series2: float, prev1: float, prev2: float) -> bool:
    """Single-point crossunder check."""
    return series1 < series2 and prev1 >= prev2
