# -*- coding: utf-8 -*-
"""
Power of 3 (PO3) - Accumulation, Manipulation, Distribution (AMD)

ICT concept: every price movement consists of three phases:
1. Accumulation (A): Price consolidates, builds liquidity on both sides.
2. Manipulation (M): False move to sweep liquidity, trap retail traders.
3. Distribution (D): True move in the intended direction.

This module detects AMD phases on daily and intraday candles using
range contraction/expansion, volume profile, and session context.

Reference: ICT MMXM Model, ICT Power of 3 methodology.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class PO3Phase(Enum):
    """PO3 phase classification."""

    ACCUMULATION = "accumulation"
    MANIPULATION = "manipulation"
    DISTRIBUTION = "distribution"
    NONE = "none"


@dataclass
class PO3Info:
    """PO3 cycle detection result.

    Attributes:
        phase: Current PO3 phase.
        start_bar: Bar index where phase started.
        accumulation_high: High of accumulation range.
        accumulation_low: Low of accumulation range.
        manipulation_price: Extreme price of manipulation (false move).
        manipulation_direction: Direction of the false move ('up' or 'down').
        distribution_target: Estimated distribution target.
        range_contraction_ratio: Accumulation range vs previous range ratio.
        confidence: 0.0-1.0 confidence in phase classification.
    """

    phase: PO3Phase
    start_bar: int
    accumulation_high: float
    accumulation_low: float
    manipulation_price: Optional[float]
    manipulation_direction: Optional[str]
    distribution_target: Optional[float]
    range_contraction_ratio: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "phase": self.phase.value,
            "start_bar": self.start_bar,
            "accumulation_high": self.accumulation_high,
            "accumulation_low": self.accumulation_low,
            "manipulation_price": self.manipulation_price,
            "manipulation_direction": self.manipulation_direction,
            "distribution_target": self.distribution_target,
            "range_contraction_ratio": self.range_contraction_ratio,
            "confidence": self.confidence,
        }


def detect_po3_daily(
    df: pd.DataFrame,
    min_accumulation_bars: int = 3,
    contraction_threshold: float = 0.7,
    expansion_mult: float = 1.5,
) -> pd.DataFrame:
    """
    Detect Power of 3 phases on daily candles.

    Algorithm:
    1. Accumulation: consecutive bars with decreasing range (contraction).
    2. Manipulation: bar breaks accumulation range then reverses intra-bar.
    3. Distribution: expansion bar moving opposite to manipulation.

    Args:
        df: OHLCV DataFrame (daily recommended).
        min_accumulation_bars: Minimum bars for accumulation phase.
        contraction_threshold: Range ratio below which accumulation is detected.
        expansion_mult: Range expansion multiple for distribution confirmation.

    Returns:
        DataFrame with columns: phase, accumulation_high, accumulation_low,
        manipulation_direction, confidence.
    """
    n = len(df)
    if n < min_accumulation_bars + 2:
        return pd.DataFrame(
            {
                "phase": ["none"] * n,
                "accumulation_high": [np.nan] * n,
                "accumulation_low": [np.nan] * n,
                "manipulation_direction": [None] * n,
                "confidence": [0.0] * n,
            },
            index=df.index,
        )

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    open_ = df["Open"].values

    ranges = high - low
    avg_range = np.zeros(n)
    for i in range(min_accumulation_bars, n):
        avg_range[i] = float(np.mean(ranges[i - min_accumulation_bars : i]))

    phases = ["none"] * n
    acc_highs = [np.nan] * n
    acc_lows = [np.nan] * n
    manip_dirs: List[Optional[str]] = [None] * n
    confidences = [0.0] * n

    i = min_accumulation_bars
    while i < n - 3:
        # Detect accumulation: decreasing range + contained price
        acc_bars = []
        j = i
        while j < n - 2:
            range_ratio = ranges[j] / (avg_range[j] + 1e-10)
            if range_ratio < contraction_threshold:
                acc_bars.append(j)
                j += 1
            else:
                break

        if len(acc_bars) >= min_accumulation_bars:
            acc_start = acc_bars[0]
            acc_end = acc_bars[-1]
            acc_high = float(np.max(high[acc_start : acc_end + 1]))
            acc_low = float(np.min(low[acc_start : acc_end + 1]))

            # Mark accumulation bars
            for b in acc_bars:
                phases[b] = "accumulation"
                acc_highs[b] = acc_high
                acc_lows[b] = acc_low
                confidences[b] = 0.7

            # Look for manipulation (false break + reversal)
            manip_found = False
            for k in range(acc_end + 1, min(acc_end + 5, n - 1)):
                # Bull trap: breaks above accumulation then closes below
                upper_break = high[k] > acc_high and close[k] < acc_high
                # Bear trap: breaks below accumulation then closes above
                lower_break = low[k] < acc_low and close[k] > acc_low

                if upper_break:
                    phases[k] = "manipulation"
                    acc_highs[k] = acc_high
                    acc_lows[k] = acc_low
                    manip_dirs[k] = "up"
                    confidences[k] = 0.8
                    manip_found = True

                    # Distribution: look for move opposite to manipulation
                    for d in range(k + 1, min(k + 5, n)):
                        range_expansion = ranges[d] > avg_range[d] * expansion_mult
                        if range_expansion and close[d] < close[k]:
                            phases[d] = "distribution"
                            acc_highs[d] = acc_high
                            acc_lows[d] = acc_low
                            manip_dirs[d] = "up"
                            confidences[d] = 0.85
                            i = d + 1
                            break
                    if not manip_found:
                        i = k + 1
                    break

                elif lower_break:
                    phases[k] = "manipulation"
                    acc_highs[k] = acc_high
                    acc_lows[k] = acc_low
                    manip_dirs[k] = "down"
                    confidences[k] = 0.8
                    manip_found = True

                    # Distribution: move opposite to manipulation
                    for d in range(k + 1, min(k + 5, n)):
                        range_expansion = ranges[d] > avg_range[d] * expansion_mult
                        if range_expansion and close[d] > close[k]:
                            phases[d] = "distribution"
                            acc_highs[d] = acc_high
                            acc_lows[d] = acc_low
                            manip_dirs[d] = "down"
                            confidences[d] = 0.85
                            i = d + 1
                            break
                    if not manip_found:
                        i = k + 1
                    break

            if not manip_found:
                i = acc_end + 1
        else:
            i += 1

    return pd.DataFrame(
        {
            "phase": phases,
            "accumulation_high": acc_highs,
            "accumulation_low": acc_lows,
            "manipulation_direction": manip_dirs,
            "confidence": confidences,
        },
        index=df.index,
    )


def detect_po3_intraday(
    df: pd.DataFrame,
    session_df: pd.DataFrame,
    min_accumulation_bars: int = 4,
    contraction_threshold: float = 0.7,
) -> pd.DataFrame:
    """
    Detect PO3 phases using session context (Asian/London/NY).

    Uses pre-computed session bounds to identify AMD within each session.
    Asian = Accumulation, London = Manipulation, NY = Distribution.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        session_df: Session DataFrame from smartmoneyconcepts.sessions().
        min_accumulation_bars: Minimum bars for accumulation.
        contraction_threshold: Contraction detection threshold.

    Returns:
        DataFrame with PO3 phase per bar.
    """
    n = len(df)
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    ranges = high - low

    phases = ["none"] * n
    acc_highs = [np.nan] * n
    acc_lows = [np.nan] * n
    manip_dirs: List[Optional[str]] = [None] * n
    confidences = [0.0] * n

    # Use session context: find asian→london→ny transitions
    asian_active = session_df["Active"].values if "Active" in session_df.columns else np.zeros(n)

    session_high = 0.0
    session_low = float("inf")
    session_start = -1
    in_session = False

    for i in range(n):
        if asian_active[i] == 1:
            if not in_session:
                session_start = i
                session_high = float(high[i])
                session_low = float(low[i])
                in_session = True
            else:
                session_high = max(session_high, float(high[i]))
                session_low = min(session_low, float(low[i]))

            # Mark accumulation
            if i - session_start >= min_accumulation_bars:
                phases[i] = "accumulation"
                acc_highs[i] = session_high
                acc_lows[i] = session_low
                confidences[i] = 0.6
        elif in_session and asian_active[i] == 0:
            # Post-Asian: look for manipulation then distribution
            # Manipulation: price breaks Asian range then reverts
            if high[i] > session_high and close[i] < session_high:
                phases[i] = "manipulation"
                acc_highs[i] = session_high
                acc_lows[i] = session_low
                manip_dirs[i] = "up"
                confidences[i] = 0.75
            elif low[i] < session_low and close[i] > session_low:
                phases[i] = "manipulation"
                acc_highs[i] = session_high
                acc_lows[i] = session_low
                manip_dirs[i] = "down"
                confidences[i] = 0.75

            # Distribution: expansion move
            if manip_dirs[i] == "up" and close[i] < session_low:
                phases[i] = "distribution"
                manip_dirs[i] = "up"
                confidences[i] = 0.8
            elif manip_dirs[i] == "down" and close[i] > session_high:
                phases[i] = "distribution"
                manip_dirs[i] = "down"
                confidences[i] = 0.8

            # Reset session tracking after distribution
            if phases[i] == "distribution":
                in_session = False
                session_start = -1

    return pd.DataFrame(
        {
            "phase": phases,
            "accumulation_high": acc_highs,
            "accumulation_low": acc_lows,
            "manipulation_direction": manip_dirs,
            "confidence": confidences,
        },
        index=df.index,
    )


def get_po3_phase_at_bar(po3_df: pd.DataFrame, bar_idx: int) -> PO3Info:
    """Get PO3 phase at a specific bar.

    Args:
        po3_df: DataFrame from detect_po3_daily or detect_po3_intraday.
        bar_idx: Bar index.

    Returns:
        PO3Info with phase details.
    """
    row = po3_df.iloc[bar_idx]
    phase_str = str(row["phase"])
    try:
        phase = PO3Phase(phase_str)
    except ValueError:
        phase = PO3Phase.NONE

    return PO3Info(
        phase=phase,
        start_bar=bar_idx,
        accumulation_high=float(row["accumulation_high"])
        if not np.isnan(row["accumulation_high"])
        else 0.0,
        accumulation_low=float(row["accumulation_low"])
        if not np.isnan(row["accumulation_low"])
        else 0.0,
        manipulation_price=None,
        manipulation_direction=str(row["manipulation_direction"])
        if row["manipulation_direction"] is not None
        and str(row["manipulation_direction"]) != "None"
        else None,
        distribution_target=None,
        range_contraction_ratio=0.0,
        confidence=float(row["confidence"]),
    )
