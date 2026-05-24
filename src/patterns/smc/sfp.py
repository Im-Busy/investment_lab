"""Swing Failure Pattern (SFP) detector.

SFP: Price breaks a swing high/low (by wick) but fails to close beyond it,
then reverses sharply. Similar to false breakout but specific to swing pivot levels.

Reference: ICT SMC terminology, referenced in smc __init__.py docstring.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class SFPInfo:
    """Swing Failure Pattern detection result."""

    detected: bool
    direction: Optional[str]  # 'bullish' or 'bearish'
    swing_level: Optional[float]  # the pivot level that was wick-broken
    swing_bar: Optional[int]  # bar where the pivot was established
    break_bar: Optional[int]  # bar where wick break occurred
    reversal_bar: Optional[int]  # bar where reversal confirmed
    strength: float  # 0.0-1.0

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "direction": self.direction,
            "swing_level": self.swing_level,
            "swing_bar": self.swing_bar,
            "break_bar": self.break_bar,
            "reversal_bar": self.reversal_bar,
            "strength": self.strength,
        }


def detect_sfp(
    df: pd.DataFrame,
    pivot_lookback: int = 5,
    reversal_bars: int = 3,
    min_reversal_pct: float = 0.2,
    atr: "np.ndarray | None" = None,
) -> List[SFPInfo]:
    """Detect Swing Failure Patterns.

    SFP occurs when:
    - Price breaks a recent swing high/low by WICK (not body close)
    - Fails to close beyond the level
    - Reverses sharply

    Args:
        df: OHLCV DataFrame.
        pivot_lookback: Bars for pivot detection.
        reversal_bars: Maximum bars for reversal confirmation.
        min_reversal_pct: Minimum reversal as fraction of range.
        atr: Optional ATR array for strength normalization.

    Returns:
        List of SFPInfo objects.
    """
    sfps: List[SFPInfo] = []
    n = len(df)

    if n < pivot_lookback * 2 + reversal_bars:
        return sfps

    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)
    close = df["Close"].to_numpy(dtype=np.float64)
    open_ = df["Open"].to_numpy(dtype=np.float64)

    for i in range(pivot_lookback, n - reversal_bars - 1):
        # Find swing high
        is_pivot_high = True
        pivot_high_val = high[i]
        for j in range(1, pivot_lookback + 1):
            if i - j >= 0 and high[i - j] >= pivot_high_val:
                is_pivot_high = False
                break
            if i + j < n and high[i + j] > pivot_high_val:
                is_pivot_high = False
                break

        if is_pivot_high:
            # Look for SFP: wick breaks above swing high but closes below
            for k in range(i + 1, min(i + reversal_bars + 3, n)):
                if high[k] > pivot_high_val and close[k] < pivot_high_val:
                    range_at_break = high[k] - low[k]
                    reversal_dist = pivot_high_val - close[k]
                    if range_at_break > 0 and reversal_dist > min_reversal_pct * range_at_break:
                        atr_val = (
                            float(atr[k]) if atr is not None and k < len(atr) else range_at_break
                        )
                        strength = min(1.0, reversal_dist / max(atr_val, 1e-10))
                        sfps.append(
                            SFPInfo(
                                detected=True,
                                direction="bearish",
                                swing_level=pivot_high_val,
                                swing_bar=i,
                                break_bar=k,
                                reversal_bar=k,
                                strength=strength,
                            )
                        )
                    break

        # Find swing low
        is_pivot_low = True
        pivot_low_val = low[i]
        for j in range(1, pivot_lookback + 1):
            if i - j >= 0 and low[i - j] <= pivot_low_val:
                is_pivot_low = False
                break
            if i + j < n and low[i + j] < pivot_low_val:
                is_pivot_low = False
                break

        if is_pivot_low:
            # Look for SFP: wick breaks below swing low but closes above
            for k in range(i + 1, min(i + reversal_bars + 3, n)):
                if low[k] < pivot_low_val and close[k] > pivot_low_val:
                    range_at_break = high[k] - low[k]
                    reversal_dist = close[k] - pivot_low_val
                    if range_at_break > 0 and reversal_dist > min_reversal_pct * range_at_break:
                        atr_val = (
                            float(atr[k]) if atr is not None and k < len(atr) else range_at_break
                        )
                        strength = min(1.0, reversal_dist / max(atr_val, 1e-10))
                        sfps.append(
                            SFPInfo(
                                detected=True,
                                direction="bullish",
                                swing_level=pivot_low_val,
                                swing_bar=i,
                                break_bar=k,
                                reversal_bar=k,
                                strength=strength,
                            )
                        )
                    break

    return sfps


def get_sfp_signal_at_bar(sfps: List[SFPInfo], bar_idx: int) -> int:
    """Get SFP direction signal at a specific bar (+1 bullish, -1 bearish, 0 none)."""
    for sfp in sfps:
        if sfp.reversal_bar == bar_idx and sfp.detected:
            return 1 if sfp.direction == "bullish" else -1
    return 0
