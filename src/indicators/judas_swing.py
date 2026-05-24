"""Judas Swing detector — ICT core entry filter.

Detects London session false move sweeping Asian range then reversing.
Bullish Judas: price drops below Asian low (false move), then reverses above.
Bearish Judas: price rises above Asian high (false move), then reverses below.

Reference: ICT MMXM Model, Day 9/16 tutorial.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class JudasSwingInfo:
    """Judas Swing detection result."""

    detected: bool
    direction: str  # 'bullish' (fake drop then up) or 'bearish' (fake rise then down)
    asian_high: float
    asian_low: float
    swing_extreme: float  # price extreme of the false move
    swing_bar: int  # bar where false move completed
    reversal_bar: int  # bar where reversal confirmed
    reversal_confirmed: bool
    displacement_ratio: float  # reversal move size / ATR
    session: str  # 'london' typically

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "direction": self.direction,
            "asian_high": self.asian_high,
            "asian_low": self.asian_low,
            "swing_extreme": self.swing_extreme,
            "swing_bar": self.swing_bar,
            "reversal_bar": self.reversal_bar,
            "reversal_confirmed": self.reversal_confirmed,
            "displacement_ratio": self.displacement_ratio,
            "session": self.session,
        }


def detect_judas_swing(
    df: pd.DataFrame,
    asian_high: float,
    asian_low: float,
    session_start_idx: int,
    session_end_idx: int,
    atr_series: "np.ndarray | pd.Series",
    atr_mult: float = 0.3,
    displacement_mult: float = 1.0,
    reversal_bars: int = 5,
) -> Optional[JudasSwingInfo]:
    """Detect Judas Swing pattern during a session window.

    Algorithm:
    1. Use pre-computed Asian session high/low.
    2. During session window:
       - Bullish Judas: price drops below Asian low by >= atr_mult * ATR (false move down),
         then closes ABOVE Asian low within reversal_bars (reversal)
       - Bearish Judas: price rises above Asian high by >= atr_mult * ATR (false move up),
         then closes BELOW Asian high within reversal_bars
    3. Reversal must be energetic: displacement ratio >= displacement_mult * ATR.

    Args:
        df: OHLCV DataFrame.
        asian_high: Asian session high price.
        asian_low: Asian session low price.
        session_start_idx: Start bar index of session.
        session_end_idx: End bar index of session.
        atr_series: ATR values aligned with df.
        atr_mult: ATR multiplier for sweep depth threshold.
        displacement_mult: ATR multiplier for reversal displacement.
        reversal_bars: Max bars for reversal confirmation.

    Returns:
        JudasSwingInfo if pattern detected, None otherwise.
    """
    if session_start_idx >= session_end_idx or session_end_idx >= len(df):
        return None

    high = df["High"].to_numpy(dtype=np.float64)[session_start_idx : session_end_idx + 1]
    low = df["Low"].to_numpy(dtype=np.float64)[session_start_idx : session_end_idx + 1]
    close = df["Close"].to_numpy(dtype=np.float64)[session_start_idx : session_end_idx + 1]

    if isinstance(atr_series, pd.Series):
        atr_vals = atr_series.to_numpy(dtype=np.float64)
    else:
        atr_vals = atr_series

    n_session = len(close)
    if n_session < 3:
        return None

    for i in range(n_session):
        abs_idx = session_start_idx + i
        atr = (
            float(atr_vals[abs_idx])
            if abs_idx < len(atr_vals) and not np.isnan(atr_vals[abs_idx])
            else 0.01
        )
        if atr <= 0:
            atr = float(close[i]) * 0.01

        if low[i] < asian_low - atr_mult * atr:
            for j in range(i + 1, min(i + 1 + reversal_bars, n_session)):
                if close[j] > asian_low:
                    displacement = close[j] - asian_low
                    disp_ratio = displacement / atr
                    if disp_ratio >= displacement_mult:
                        return JudasSwingInfo(
                            detected=True,
                            direction="bullish",
                            asian_high=asian_high,
                            asian_low=asian_low,
                            swing_extreme=float(low[i]),
                            swing_bar=abs_idx + i - session_start_idx
                            if i < len(close)
                            else abs_idx,
                            reversal_bar=session_start_idx + j,
                            reversal_confirmed=True,
                            displacement_ratio=disp_ratio,
                            session="london",
                        )

        if high[i] > asian_high + atr_mult * atr:
            for j in range(i + 1, min(i + 1 + reversal_bars, n_session)):
                if close[j] < asian_high:
                    displacement = asian_high - close[j]
                    disp_ratio = displacement / atr
                    if disp_ratio >= displacement_mult:
                        return JudasSwingInfo(
                            detected=True,
                            direction="bearish",
                            asian_high=asian_high,
                            asian_low=asian_low,
                            swing_extreme=float(high[i]),
                            swing_bar=abs_idx + i - session_start_idx
                            if i < len(close)
                            else abs_idx,
                            reversal_bar=session_start_idx + j,
                            reversal_confirmed=True,
                            displacement_ratio=disp_ratio,
                            session="london",
                        )

    return None


def detect_judas_swing_vectorized(
    df: pd.DataFrame,
    asian_high_series: np.ndarray,
    asian_low_series: np.ndarray,
    atr_series: np.ndarray,
    session_mask: np.ndarray,
    atr_mult: float = 0.3,
    displacement_mult: float = 1.0,
    reversal_bars: int = 5,
) -> np.ndarray:
    """Vectorized detection returning per-bar Judas Swing direction signal.

    Returns:
        Array of +1 (bullish Judas expected), -1 (bearish), 0 (none).
    """
    n = len(df)
    signals = np.zeros(n, dtype=np.int8)

    if n < reversal_bars + 2:
        return signals

    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)
    close = df["Close"].to_numpy(dtype=np.float64)

    in_session = False
    session_start = 0

    for i in range(n):
        atr = atr_series[i] if not np.isnan(atr_series[i]) else close[i] * 0.01
        if atr <= 0:
            atr = close[i] * 0.01

        if session_mask[i]:
            if not in_session:
                session_start = i
                in_session = True

            ah = asian_high_series[i]
            al = asian_low_series[i]

            if low[i] < al - atr_mult * atr:
                for j in range(i + 1, min(i + 1 + reversal_bars, n)):
                    if close[j] > al and (close[j] - al) >= displacement_mult * atr:
                        signals[i + 1 : j + 1] = 1
                        break

            if high[i] > ah + atr_mult * atr:
                for j in range(i + 1, min(i + 1 + reversal_bars, n)):
                    if close[j] < ah and (ah - close[j]) >= displacement_mult * atr:
                        signals[i + 1 : j + 1] = -1
                        break
        else:
            in_session = False

    return signals
