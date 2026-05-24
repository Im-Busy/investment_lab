"""Six-indicator voting system for trade signal generation.

Paper: "Heuristic Based Trading System" (Öztürk 2015) - B6 in MASTER_COMPARISON_REPORT
Indicators: RSI, ROC, SMA, EMA, WMA, MACD
Majority threshold K determines minimum votes needed for a signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def _ema(series: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    result = np.empty_like(series)
    result[:period] = np.nan
    if period < len(series):
        result[period - 1] = np.mean(series[:period])
    for i in range(period, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def _sma(series: np.ndarray, period: int) -> np.ndarray:
    result = np.full_like(series, np.nan)
    if len(series) >= period:
        cumsum = np.cumsum(np.insert(series, 0, 0))
        result[period - 1 :] = (cumsum[period:] - cumsum[:-period]) / period
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


def compute_voting_signals(
    close: np.ndarray,
    majority_threshold: int = 4,
    rsi_period: int = 14,
    roc_period: int = 12,
    sma_period: int = 20,
    ema_period: int = 9,
    wma_period: int = 10,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> np.ndarray:
    """Generate buy/sell signals via 6-indicator voting.

    Six indicators each vote +1 (bullish) or -1 (bearish):
      RSI: RSI < 30 = bullish, RSI > 70 = bearish
      ROC: Rate-of-Change > 0 = bullish, < 0 = bearish
      SMA: Close > SMA = bullish, < SMA = bearish
      EMA: Close > EMA = bullish, < EMA = bearish
      WMA: Close > WMA = bullish, < WMA = bearish
      MACD: MACD line > Signal line = bullish, < Signal line = bearish

    A signal is generated when |votes| >= majority_threshold.

    Args:
        close: Array of close prices.
        majority_threshold: Minimum net votes required (default 4 of 6).
        rsi_period: RSI calculation period.
        roc_period: ROC calculation period.
        sma_period: SMA calculation period.
        ema_period: EMA calculation period.
        wma_period: WMA calculation period.
        macd_fast: MACD fast EMA period.
        macd_slow: MACD slow EMA period.
        macd_signal: MACD signal line EMA period.

    Returns:
        Signal array: +1 = buy, -1 = sell, 0 = no trade.
    """
    n = len(close)
    votes = np.zeros(n, dtype=np.int8)

    # RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _sma(gain, rsi_period)
    avg_loss = _sma(loss, rsi_period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = np.where(np.isnan(rsi), 50.0, rsi)
    votes += np.where(rsi < 30, 1, 0).astype(np.int8)
    votes -= np.where(rsi > 70, 1, 0).astype(np.int8)

    # ROC
    roc = np.zeros_like(close)
    roc[roc_period:] = (close[roc_period:] - close[:-roc_period]) / np.where(
        close[:-roc_period] > 0, close[:-roc_period], 1e-10
    )
    votes += np.where(roc > 0, 1, 0).astype(np.int8)
    votes -= np.where(roc < 0, 1, 0).astype(np.int8)

    # SMA
    sma = _sma(close, sma_period)
    votes += np.where(close > sma, 1, 0).astype(np.int8)
    votes -= np.where(close < sma, 1, 0).astype(np.int8)

    # EMA
    ema = _ema(close, ema_period)
    votes += np.where(close > ema, 1, 0).astype(np.int8)
    votes -= np.where(close < ema, 1, 0).astype(np.int8)

    # WMA
    wma = _wma(close, wma_period)
    votes += np.where(close > wma, 1, 0).astype(np.int8)
    votes -= np.where(close < wma, 1, 0).astype(np.int8)

    # MACD
    ema_fast = _ema(close, macd_fast)
    ema_slow = _ema(close, macd_slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, macd_signal)
    votes += np.where(macd_line > signal_line, 1, 0).astype(np.int8)
    votes -= np.where(macd_line < signal_line, 1, 0).astype(np.int8)

    signal = np.zeros(n, dtype=np.int8)
    signal[votes >= majority_threshold] = 1
    signal[votes <= -majority_threshold] = -1

    return signal


def compute_voting_signals_df(
    df: pd.DataFrame,
    majority_threshold: int = 4,
    rsi_period: int = 14,
    roc_period: int = 12,
    sma_period: int = 20,
    ema_period: int = 9,
    wma_period: int = 10,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
    """Compute voting signals on a DataFrame with a 'Close' column.

    Args:
        df: DataFrame with 'Close' column.
        majority_threshold: Minimum net votes required (default 4).

    Returns:
        DataFrame with added 'vote_signal' column.
    """
    result = df.copy()
    close_arr = df["Close"].to_numpy(dtype=np.float64)
    result["vote_signal"] = compute_voting_signals(
        close_arr,
        majority_threshold=majority_threshold,
        rsi_period=rsi_period,
        roc_period=roc_period,
        sma_period=sma_period,
        ema_period=ema_period,
        wma_period=wma_period,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
    )
    return result
