# -*- coding: utf-8 -*-
"""
Optimal Trade Entry (OTE) - Fibonacci-Based Entry Zone Detection

ICT concept: the optimal entry zone for trend continuation lies between
0.618 (61.8%) and 0.786 (78.6%) Fibonacci retracement levels.

This is where institutions load positions after retail stops are swept.
The module detects OTE zones and generates entry signals when price
reaches the zone with confluence (FVG, order block, or breaker).

Reference: ICT Optimal Trade Entry methodology.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class OTEZoneType(Enum):
    """OTE zone classification."""

    GOLDEN = "golden"  # 0.618 zone
    OPTIMAL = "optimal"  # 0.705 zone
    DEEP = "deep"  # 0.786 zone
    EXTREME = "extreme"  # 0.886 zone
    NONE = "none"


@dataclass
class OTEZone:
    """OTE Fibonacci zone.

    Attributes:
        zone_type: Type of OTE zone reached.
        retracement_pct: Actual retracement percentage.
        entry_low: Lower bound of the entry zone.
        entry_high: Upper bound of the entry zone.
        stop_loss: Stop loss level (beyond 1.0 fib).
        take_profit_1: First take profit (0.0 fib - previous extreme).
        take_profit_2: Second take profit (-0.272 extension).
        swing_high: Swing high of the trend.
        swing_low: Swing low of the trend.
        direction: Trend direction ('bullish' or 'bearish').
        confluence: Whether FVG/OB support exists at this zone.
        strength: 0.0-1.0 strength score.
    """

    zone_type: OTEZoneType
    retracement_pct: float
    entry_low: float
    entry_high: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    swing_high: float
    swing_low: float
    direction: str
    confluence: bool
    strength: float

    def to_dict(self) -> dict:
        return {
            "zone_type": self.zone_type.value,
            "retracement_pct": self.retracement_pct,
            "entry_low": self.entry_low,
            "entry_high": self.entry_high,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "swing_high": self.swing_high,
            "swing_low": self.swing_low,
            "direction": self.direction,
            "confluence": self.confluence,
            "strength": self.strength,
        }


# ICT OTE Fibonacci levels
OTE_LEVELS = {
    "golden": 0.618,
    "optimal": 0.705,
    "deep": 0.786,
    "extreme": 0.886,
}

FIB_LEVELS: list[float] = [0.0, 0.50, 0.618, 0.705, 0.79, 1.0, -0.27, -0.62, -1.0]
FIB_TP_LEVELS: list[float] = [-0.27, -0.62, -1.0]

OTE_ENTRY_BUFFER = 0.02  # 2% buffer around fib levels


def calculate_ote_zone(
    swing_high: float,
    swing_low: float,
    direction: str,
    current_price: float,
    confluence_check: Optional[callable] = None,
) -> OTEZone:
    """
    Calculate the OTE Fibonacci zone for a given swing.

    Args:
        swing_high: Swing high price.
        swing_low: Swing low price.
        direction: Trend direction ('bullish' for uptrend, 'bearish' for downtrend).
        current_price: Current price to check against zones.
        confluence_check: Optional function(current_price, zone_low, zone_high) -> bool.

    Returns:
        OTEZone with entry details.

    Example:
        >>> zone = calculate_ote_zone(105.0, 95.0, 'bullish', 98.5)
        >>> if zone.zone_type != OTEZoneType.NONE:
        ...     print(f"Entry zone: {zone.entry_low:.2f}-{zone.entry_high:.2f}")
    """
    fib_range = swing_high - swing_low
    if fib_range <= 0:
        return OTEZone(
            zone_type=OTEZoneType.NONE,
            retracement_pct=0.0,
            entry_low=0.0,
            entry_high=0.0,
            stop_loss=0.0,
            take_profit_1=0.0,
            take_profit_2=0.0,
            swing_high=swing_high,
            swing_low=swing_low,
            direction=direction,
            confluence=False,
            strength=0.0,
        )

    # Determine which OTE level price is at
    if direction == "bullish":
        # Price retraced down from swing_high toward swing_low
        retracement = (swing_high - current_price) / fib_range
        retracement = max(0.0, min(1.0, retracement))

        # Entry zone is between 0.618 and 0.786
        entry_low = swing_high - (OTE_LEVELS["deep"] * fib_range)
        entry_high = swing_high - (OTE_LEVELS["golden"] * fib_range)

        stop_loss = swing_low - (fib_range * 0.05)
        take_profit_1 = swing_high
        take_profit_2 = swing_high + (fib_range * 0.272)
    else:
        # Price retraced up from swing_low toward swing_high
        retracement = (current_price - swing_low) / fib_range
        retracement = max(0.0, min(1.0, retracement))

        entry_low = swing_low + (OTE_LEVELS["golden"] * fib_range)
        entry_high = swing_low + (OTE_LEVELS["deep"] * fib_range)

        stop_loss = swing_high + (fib_range * 0.05)
        take_profit_1 = swing_low
        take_profit_2 = swing_low - (fib_range * 0.272)

    # Classify the zone based on retracement
    zone_type = OTEZoneType.NONE
    if (
        OTE_LEVELS["golden"] - OTE_ENTRY_BUFFER
        <= retracement
        <= OTE_LEVELS["golden"] + OTE_ENTRY_BUFFER
    ):
        zone_type = OTEZoneType.GOLDEN
    elif (
        OTE_LEVELS["optimal"] - OTE_ENTRY_BUFFER
        <= retracement
        <= OTE_LEVELS["optimal"] + OTE_ENTRY_BUFFER
    ):
        zone_type = OTEZoneType.OPTIMAL
    elif (
        OTE_LEVELS["deep"] - OTE_ENTRY_BUFFER
        <= retracement
        <= OTE_LEVELS["deep"] + OTE_ENTRY_BUFFER
    ):
        zone_type = OTEZoneType.DEEP
    elif (
        OTE_LEVELS["extreme"] - OTE_ENTRY_BUFFER
        <= retracement
        <= OTE_LEVELS["extreme"] + OTE_ENTRY_BUFFER
    ):
        zone_type = OTEZoneType.EXTREME

    # Check for confluence
    confluence = False
    if confluence_check is not None and zone_type != OTEZoneType.NONE:
        try:
            confluence = confluence_check(current_price, entry_low, entry_high)
        except Exception:
            pass

    # Calculate strength
    if zone_type == OTEZoneType.GOLDEN:
        base_strength = 0.7
    elif zone_type == OTEZoneType.OPTIMAL:
        base_strength = 0.8
    elif zone_type == OTEZoneType.DEEP:
        base_strength = 0.65
    elif zone_type == OTEZoneType.EXTREME:
        base_strength = 0.5
    else:
        base_strength = 0.0

    if confluence:
        base_strength = min(1.0, base_strength + 0.15)

    return OTEZone(
        zone_type=zone_type,
        retracement_pct=retracement * 100,
        entry_low=entry_low,
        entry_high=entry_high,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        swing_high=swing_high,
        swing_low=swing_low,
        direction=direction,
        confluence=confluence,
        strength=base_strength,
    )


def detect_ote_entries(
    df: pd.DataFrame,
    swing_highs_lows: pd.DataFrame,
    fvg_df: Optional[pd.DataFrame] = None,
    ob_df: Optional[pd.DataFrame] = None,
    max_retracement_bars: int = 20,
) -> List[OTEZone]:
    """
    Scan for OTE entry zones using swing points.

    Args:
        df: OHLCV DataFrame.
        swing_highs_lows: Swing high/low DataFrame (from smartmoneyconcepts or mss).
        fvg_df: Optional FVG DataFrame for confluence.
        ob_df: Optional order block DataFrame for confluence.
        max_retracement_bars: Maximum bars since last swing for valid retracement.

    Returns:
        List of valid OTEZone entries.
    """
    entries: List[OTEZone] = []
    n = len(df)

    if n < 3:
        return entries

    if "HighLow" not in swing_highs_lows.columns:
        return entries

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values

    last_swing_high_idx = -1
    last_swing_low_idx = -1
    last_swing_high = 0.0
    last_swing_low = 0.0

    def _check_confluence(price: float, zone_low: float, zone_high: float) -> bool:
        if fvg_df is None:
            return False
        # Check if any FVG zone overlaps with OTE zone
        if "Top" in fvg_df.columns and "Bottom" in fvg_df.columns:
            fvg_zones = fvg_df.dropna(subset=["FVG"])
            for _, fvg_row in fvg_zones.iterrows():
                fvg_top = float(fvg_row["Top"])
                fvg_bottom = float(fvg_row["Bottom"])
                fvg_zone_low = min(fvg_top, fvg_bottom)
                fvg_zone_high = max(fvg_top, fvg_bottom)
                overlap = fvg_zone_low <= zone_high and fvg_zone_high >= zone_low
                if overlap:
                    return True
        return False

    for i in range(n):
        hl = swing_highs_lows["HighLow"].iloc[i]
        if pd.isna(hl):
            continue

        if hl == 1:
            last_swing_high_idx = i
            last_swing_high = float(high[i])
        elif hl == -1:
            last_swing_low_idx = i
            last_swing_low = float(low[i])

        # Check for OTE after a swing pair is established
        if last_swing_high_idx >= 0 and last_swing_low_idx >= 0:
            latest_swing = max(last_swing_high_idx, last_swing_low_idx)
            bars_since_swing = i - latest_swing

            if bars_since_swing > 0 and bars_since_swing <= max_retracement_bars:
                if last_swing_high_idx > last_swing_low_idx:
                    # Most recent was swing high → expect bearish retracement
                    direction = "bearish"
                    zone = calculate_ote_zone(
                        swing_high=last_swing_high,
                        swing_low=last_swing_low,
                        direction=direction,
                        current_price=float(close[i]),
                        confluence_check=_check_confluence if fvg_df is not None else None,
                    )
                else:
                    # Most recent was swing low → expect bullish retracement
                    direction = "bullish"
                    zone = calculate_ote_zone(
                        swing_high=last_swing_high,
                        swing_low=last_swing_low,
                        direction=direction,
                        current_price=float(close[i]),
                        confluence_check=_check_confluence if fvg_df is not None else None,
                    )

                if zone.zone_type != OTEZoneType.NONE:
                    entries.append(zone)

    return entries


def get_ote_entry_signal(
    entries: List[OTEZone],
    current_bar: int,
) -> Optional[OTEZone]:
    """
    Get the OTE entry signal for a bar.

    Args:
        entries: List of OTEZone entries.
        current_bar: Current bar index.

    Returns:
        Most recent OTEZone for this bar, or None.
    """
    if not entries:
        return None

    # Return the last entry (most recent)
    filtered = [e for e in entries if e.zone_type != OTEZoneType.NONE]
    if not filtered:
        return None

    return filtered[-1]


def fibonacci_body_to_body(
    ohlc: pd.DataFrame,
    swing_start_idx: int,
    swing_end_idx: int,
    direction: int,
) -> dict[float, float]:
    """Calculate Fibonacci levels using candle BODIES (not wicks).

    ICT Order Block & Fibonacci rule: use body highs/lows to avoid broker variance.

    Args:
        ohlc: OHLCV DataFrame
        swing_start_idx: Start index of the swing
        swing_end_idx: End index of the swing
        direction: +1 uptrend, -1 downtrend

    Returns:
        dict mapping fib level (float) to price level (float)
    """
    open_ = ohlc["Open"].to_numpy(dtype=np.float64)
    close = ohlc["Close"].to_numpy(dtype=np.float64)

    body_highs = np.maximum(open_, close)
    body_lows = np.minimum(open_, close)

    if direction == 1:
        start_body = float(np.min(body_lows[swing_start_idx : swing_end_idx + 1]))
        end_body = float(np.max(body_highs[swing_start_idx : swing_end_idx + 1]))
        price_range = end_body - start_body
    else:
        start_body = float(np.max(body_highs[swing_start_idx : swing_end_idx + 1]))
        end_body = float(np.min(body_lows[swing_start_idx : swing_end_idx + 1]))
        price_range = start_body - end_body

    levels: dict[float, float] = {}
    for level in FIB_LEVELS:
        if direction == 1:
            if level >= 0:
                levels[level] = end_body - price_range * level
            else:
                levels[level] = end_body + price_range * abs(level)
        else:
            if level >= 0:
                levels[level] = end_body + price_range * level
            else:
                levels[level] = end_body - price_range * abs(level)

    return levels


def get_ote_targets(
    fib_levels: dict[float, float],
) -> tuple[float, float, float]:
    """Get TP1 (-0.27), TP2 (-0.62), TP3 (-1.0 symmetrical swing) targets."""
    return (
        fib_levels.get(-0.27, 0.0),
        fib_levels.get(-0.62, 0.0),
        fib_levels.get(-1.0, 0.0),
    )
