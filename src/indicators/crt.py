# -*- coding: utf-8 -*-
"""
Candle Range Theory (CRT)

ICT framework that applies Power of 3 on individual candles.
Each candle acts as a microcosm of the full AMD cycle.

Key concepts:
- CRT High (CRH): Candle high = premium pricing
- CRT Low (CRL): Candle low = discount pricing
- Price sweeps one extreme (liquidity grab) then reverses toward the other.

This module detects CRT setups: reference candles with subsequent
sweep + reversal patterns indicating directional intent.

Reference: ICT Candle Range Theory methodology.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class CRTSetup:
    """CRT setup detection result.

    Attributes:
        detected: Whether CRT setup is valid.
        ref_bar: Index of reference candle.
        ref_high: CRH - candle range high.
        ref_low: CRL - candle range low.
        ref_range: CRH - CRL.
        sweep_direction: Which extreme was swept ('high_sweep' or 'low_sweep').
        sweep_bar: Bar index where sweep occurred.
        reversal_bar: Bar index where reversal was confirmed.
        target_direction: Expected direction ('bullish' or 'bearish').
        target: Price target (opposite extreme of the reference candle).
        strength: Setup strength 0.0-1.0.
        premium_level: Premium zone level (top 30% of range).
        discount_level: Discount zone level (bottom 30% of range).
    """

    detected: bool
    ref_bar: int
    ref_high: float
    ref_low: float
    ref_range: float
    sweep_direction: Optional[str]
    sweep_bar: Optional[int]
    reversal_bar: Optional[int]
    target_direction: Optional[str]
    target: Optional[float]
    strength: float
    premium_level: float
    discount_level: float

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "ref_bar": self.ref_bar,
            "ref_high": self.ref_high,
            "ref_low": self.ref_low,
            "ref_range": self.ref_range,
            "sweep_direction": self.sweep_direction,
            "sweep_bar": self.sweep_bar,
            "reversal_bar": self.reversal_bar,
            "target_direction": self.target_direction,
            "target": self.target,
            "strength": self.strength,
            "premium_level": self.premium_level,
            "discount_level": self.discount_level,
        }


def detect_crt(
    df: pd.DataFrame,
    ref_bar_idx: int,
    atr: float,
    sweep_buffer_mult: float = 0.2,
    min_ref_range_atr: float = 0.5,
    max_lookforward: int = 10,
) -> CRTSetup:
    """
    Detect a CRT setup from a reference candle.

    Algorithm:
    1. Take reference candle at ref_bar_idx.
    2. Calculate CRH, CRL, premium/discount zones.
    3. Look forward for sweep of one extreme.
    4. Confirm reversal toward the opposite extreme.
    5. Set target at the opposite extreme.

    Args:
        df: OHLCV DataFrame.
        ref_bar_idx: Index of the reference candle.
        atr: Current ATR value.
        sweep_buffer_mult: Buffer beyond extreme for sweep confirmation.
        min_ref_range_atr: Minimum reference range as ATR multiple.
        max_lookforward: Maximum bars to look ahead for setup completion.

    Returns:
        CRTSetup with detection results.

    Example:
        >>> crt = detect_crt(df, ref_bar_idx=10, atr=2.0)
        >>> if crt.detected:
        ...     print(f"CRT {crt.target_direction} target: {crt.target}")
    """
    no_setup = CRTSetup(
        detected=False,
        ref_bar=ref_bar_idx,
        ref_high=0.0,
        ref_low=0.0,
        ref_range=0.0,
        sweep_direction=None,
        sweep_bar=None,
        reversal_bar=None,
        target_direction=None,
        target=None,
        strength=0.0,
        premium_level=0.0,
        discount_level=0.0,
    )

    n = len(df)
    if ref_bar_idx >= n - 3 or ref_bar_idx < 0:
        return no_setup

    ref_high = float(df.iloc[ref_bar_idx]["High"])
    ref_low = float(df.iloc[ref_bar_idx]["Low"])
    ref_range = ref_high - ref_low

    # Validate reference range
    if ref_range < min_ref_range_atr * atr:
        return no_setup

    range_third = ref_range / 3
    premium_level = ref_high - range_third
    discount_level = ref_low + range_third

    buffer = sweep_buffer_mult * atr

    # Look forward for sweep + reversal
    end_idx = min(ref_bar_idx + max_lookforward, n)
    high_sweep = False
    low_sweep = False
    sweep_bar = None
    sweep_idx = -1

    for i in range(ref_bar_idx + 1, end_idx):
        high_i = float(df.iloc[i]["High"])
        low_i = float(df.iloc[i]["Low"])
        close_i = float(df.iloc[i]["Close"])

        # Check for high sweep (break above CRH)
        if not high_sweep and not low_sweep:
            if high_i >= ref_high + buffer:
                high_sweep = True
                sweep_bar = i
                sweep_idx = i
            elif low_i <= ref_low - buffer:
                low_sweep = True
                sweep_bar = i
                sweep_idx = i

        # Check for reversal after sweep
        if sweep_idx >= 0:
            for j in range(i + 1, end_idx):
                close_j = float(df.iloc[j]["Close"])

                if high_sweep:
                    # Swept high → should reverse down (bearish target)
                    if close_j < premium_level:
                        sweep_distance = high_i - ref_high
                        reversal_distance = premium_level - close_j
                        strength = min(
                            1.0, (sweep_distance + reversal_distance) / (ref_range + 1e-10) * 0.5
                        )

                        return CRTSetup(
                            detected=True,
                            ref_bar=ref_bar_idx,
                            ref_high=ref_high,
                            ref_low=ref_low,
                            ref_range=ref_range,
                            sweep_direction="high_sweep",
                            sweep_bar=sweep_bar,
                            reversal_bar=j,
                            target_direction="bearish",
                            target=ref_low,
                            strength=strength,
                            premium_level=premium_level,
                            discount_level=discount_level,
                        )
                elif low_sweep:
                    # Swept low → should reverse up (bullish target)
                    if close_j > discount_level:
                        sweep_distance = ref_low - low_i
                        reversal_distance = close_j - discount_level
                        strength = min(
                            1.0, (sweep_distance + reversal_distance) / (ref_range + 1e-10) * 0.5
                        )

                        return CRTSetup(
                            detected=True,
                            ref_bar=ref_bar_idx,
                            ref_high=ref_high,
                            ref_low=ref_low,
                            ref_range=ref_range,
                            sweep_direction="low_sweep",
                            sweep_bar=sweep_bar,
                            reversal_bar=j,
                            target_direction="bullish",
                            target=ref_high,
                            strength=strength,
                            premium_level=premium_level,
                            discount_level=discount_level,
                        )

        # If sweep happened but no reversal found within window, setup invalid
        if sweep_idx >= 0:
            break

    return no_setup


def find_crt_setups(
    df: pd.DataFrame,
    atr_series: pd.Series,
    min_ref_range_atr: float = 0.5,
    sweep_buffer_mult: float = 0.2,
    max_lookforward: int = 10,
    min_bar_spacing: int = 3,
) -> List[CRTSetup]:
    """
    Find all valid CRT setups in a DataFrame.

    Args:
        df: OHLCV DataFrame.
        atr_series: ATR series aligned with df.
        min_ref_range_atr: Minimum reference range as ATR multiple.
        sweep_buffer_mult: Buffer for sweep detection.
        max_lookforward: Bars to look ahead.
        min_bar_spacing: Minimum bars between reference candles.

    Returns:
        List of valid CRTSetup objects.
    """
    setups: List[CRTSetup] = []
    n = len(df)

    if n < min_bar_spacing + 3:
        return setups

    last_ref_bar = -min_bar_spacing

    for i in range(0, n - 3):
        if i - last_ref_bar < min_bar_spacing:
            continue

        atr = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0.01

        crt = detect_crt(
            df,
            ref_bar_idx=i,
            atr=atr,
            min_ref_range_atr=min_ref_range_atr,
            sweep_buffer_mult=sweep_buffer_mult,
            max_lookforward=max_lookforward,
        )

        if crt.detected:
            setups.append(crt)
            last_ref_bar = i

    return setups


def get_crt_signal_for_bar(
    crt_setups: List[CRTSetup],
    bar_idx: int,
) -> Optional[CRTSetup]:
    """
    Get the CRT setup that signals at a specific bar.

    Args:
        crt_setups: List of CRT setups.
        bar_idx: Current bar index.

    Returns:
        The CRTSetup whose reversal_bar matches bar_idx, or None.
    """
    for setup in crt_setups:
        if setup.reversal_bar == bar_idx and setup.detected:
            return setup
    return None
