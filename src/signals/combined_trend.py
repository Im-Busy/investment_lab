"""P24-15: Combined 4-indicator trend confirmation signal.

RSI > 50 AND CCI ≥ +100 AND MACD_Line > Signal AND ATR_rising → trend confirmed.
All four must agree — eliminates false breakouts during ranging conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from src.indicators.technical import rsi, atr, ema
from src.indicators.pinescript_helpers import macd


@dataclass
class TrendConfirmation:
    """Result of 4-indicator trend check."""

    timestamp: pd.Timestamp
    rsi_bull: bool
    cci_bull: bool
    macd_bull: bool
    atr_rising: bool
    confirmed: bool
    direction: int  # +1 bull confirmed, -1 bear confirmed, 0 no confirmation


def compute_cci(data: pd.DataFrame, period: int = 20) -> np.ndarray:
    """Commodity Channel Index — (TP − SMA(TP)) / (0.015 × MAD(TP))."""
    tp = (data["High"] + data["Low"] + data["Close"]) / 3.0
    tp_sma = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    cci = (tp - tp_sma) / (0.015 * mad + 1e-10)
    return cci.values


def four_indicator_trend(
    data: pd.DataFrame,
    idx: int,
    rsi_period: int = 14,
    cci_period: int = 20,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    atr_period: int = 14,
) -> TrendConfirmation:
    """Check all four conditions at bar index.

    Returns:
        TrendConfirmation: confirmed=True only if all 4 agree on direction.
    """
    ts = data.index[idx] if hasattr(data.index, "__getitem__") else pd.Timestamp.now()

    if idx < max(rsi_period, cci_period, macd_slow + macd_signal, atr_period + 2) + 5:
        return TrendConfirmation(
            timestamp=ts,
            rsi_bull=False,
            cci_bull=False,
            macd_bull=False,
            atr_rising=False,
            confirmed=False,
            direction=0,
        )

    rsi_vals = rsi(data["Close"].values, rsi_period)
    atr_vals = atr(data["High"].values, data["Low"].values, data["Close"].values, atr_period)
    ema12 = ema(data["Close"].values, macd_fast)
    ema26 = ema(data["Close"].values, macd_slow)
    macd_line = ema12 - ema26
    sig_line = ema(macd_line, macd_signal)

    cci_vals = compute_cci(data, cci_period)

    rsi_bull = float(rsi_vals[idx]) > 50.0
    rsi_bear = float(rsi_vals[idx]) < 50.0
    cci_bull = float(cci_vals[idx]) >= 100.0
    cci_bear = float(cci_vals[idx]) <= -100.0
    macd_bull = float(macd_line[idx]) > float(sig_line[idx])
    macd_bear = float(macd_line[idx]) < float(sig_line[idx])
    atr_rising = float(atr_vals[idx]) > float(atr_vals[idx - 1])

    bull_confirmed = rsi_bull and cci_bull and macd_bull and atr_rising
    bear_confirmed = rsi_bear and cci_bear and macd_bear and atr_rising

    if bull_confirmed:
        direction = 1
    elif bear_confirmed:
        direction = -1
    else:
        direction = 0

    return TrendConfirmation(
        timestamp=ts,
        rsi_bull=rsi_bull if direction >= 0 else False,
        cci_bull=cci_bull if direction >= 0 else False,
        macd_bull=macd_bull if direction >= 0 else False,
        atr_rising=atr_rising,
        confirmed=bull_confirmed or bear_confirmed,
        direction=direction,
    )


def trend_multiplication_factor(trend: TrendConfirmation) -> float:
    """Convert trend confirmation into a signal multiplier.

    Bull confirmed → 1.3x, Bear confirmed → 0.7x (attenuate), None → 0.9x.
    """
    if not trend.confirmed:
        return 0.9
    if trend.direction > 0:
        return 1.3
    return 0.7
