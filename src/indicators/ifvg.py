# -*- coding: utf-8 -*-
"""
Inverse Fair Value Gap (IFVG) Detection

Identifies imbalance zones where price moved quickly, leaving gaps
that act as support/resistance zones for entries.

Reference: SMC-ICT-ML-Hybrid-Backtester Brief
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from loguru import logger


@dataclass
class IFVG:
    """
    Inverse Fair Value Gap container.

    Attributes:
        high: Upper boundary of the gap
        low: Lower boundary of the gap
        direction: Gap direction - 'bullish' or 'bearish'
        start_bar: Bar index where gap started
        end_bar: Bar index where gap ended
        filled: Whether the gap has been filled
        fill_bar: Bar index where gap was filled (if filled)
        atr_at_creation: ATR value when gap formed
        gap_size: Size of the gap (high - low)
        created_at: Timestamp of gap creation
        ce: Consequent Encroachment (50% midpoint of FVG)
        zero_overlap: Whether strict ICT zero-wick-overlap rule is satisfied
    """

    high: float
    low: float
    direction: str  # 'bullish' or 'bearish'
    start_bar: int
    end_bar: int
    filled: bool = False
    fill_bar: Optional[int] = None
    atr_at_creation: float = 0.0
    gap_size: float = 0.0
    created_at: Optional[pd.Timestamp] = None
    ce: float = 0.0
    zero_overlap: bool = False

    def __post_init__(self):
        """Calculate gap size and CE after initialization."""
        if self.gap_size == 0.0:
            self.gap_size = self.high - self.low
        if self.ce == 0.0:
            self.ce = (self.high + self.low) / 2.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "high": self.high,
            "low": self.low,
            "gap_size": self.gap_size,
            "direction": self.direction,
            "start_bar": self.start_bar,
            "end_bar": self.end_bar,
            "filled": self.filled,
            "fill_bar": self.fill_bar,
            "atr_at_creation": self.atr_at_creation,
            "created_at": str(self.created_at) if self.created_at else None,
            "ce": self.ce,
            "zero_overlap": self.zero_overlap,
        }

    def contains_price(self, price: float) -> bool:
        """Check if price is within the IFVG zone."""
        return self.low <= price <= self.high

    def get_edge(self, direction: str) -> float:
        """Get the edge price for entry."""
        if direction == "long":
            return self.low  # Buy at the low edge
        else:
            return self.high  # Sell at the high edge


@dataclass
class IFVGProximity:
    """
    IFVG proximity information for entry decisions.

    Attributes:
        nearest_ifvg: The nearest unfilled IFVG
        distance: Distance from current price to IFVG edge
        distance_atr_ratio: Distance normalized by ATR
        edge_price: The IFVG edge price closest to current price
        is_bullish: Whether the IFVG is bullish
        within_proximity: Whether price is within proximity threshold
        distance_to_ce: Distance from current price to CE (50% midpoint)
        distance_ce_atr_ratio: CE distance normalized by ATR
    """

    nearest_ifvg: Optional[IFVG]
    distance: float
    distance_atr_ratio: float
    edge_price: float
    is_bullish: bool
    within_proximity: bool
    distance_to_ce: float = float("inf")
    distance_ce_atr_ratio: float = float("inf")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "nearest_ifvg": self.nearest_ifvg.to_dict() if self.nearest_ifvg else None,
            "distance": self.distance,
            "distance_atr_ratio": self.distance_atr_ratio,
            "edge_price": self.edge_price,
            "is_bullish": self.is_bullish,
            "within_proximity": self.within_proximity,
        }


def detect_ifvg(
    df: pd.DataFrame, atr: pd.Series, atr_mult: float = 1.2, min_gap_bars: int = 3
) -> List[IFVG]:
    """
    Detect Inverse Fair Value Gaps in price data.

    An IFVG forms when there's an imbalance where:
    abs(close[t] - open[t+1]) > atr_mult * ATR

    The gap is considered unfilled if price hasn't revisited
    the gap zone within min_gap_bars.

    Bullish IFVG: Gap up (current close > next open)
        - Forms when price gaps up and leaves unfilled zone
        - Zone acts as support on retest

    Bearish IFVG: Gap down (current close < next open)
        - Forms when price gaps down and leaves unfilled zone
        - Zone acts as resistance on retest

    Args:
        df: OHLCV DataFrame
        atr: ATR series aligned with df
        atr_mult: Minimum gap size as ATR multiplier (default: 1.2)
        min_gap_bars: Bars to wait before considering gap unfilled

    Returns:
        List of IFVG objects

    Example:
        >>> ifvgs = detect_ifvg(df, atr, atr_mult=1.2)
        >>> for ifvg in ifvgs:
        ...     print(f"IFVG: {ifvg.low:.2f} - {ifvg.high:.2f}, filled: {ifvg.filled}")
    """
    ifvgs: list[IFVG] = []

    if len(df) < 3:
        return ifvgs

    # Ensure ATR is aligned
    if len(atr) != len(df):
        logger.warning("ATR length mismatch with DataFrame")
        atr = atr.reindex(df.index)

    for i in range(len(df) - 2):
        # Get three consecutive bars
        bar1 = df.iloc[i]
        bar2 = df.iloc[i + 1]
        bar3 = df.iloc[i + 2]

        # Get ATR value at this point
        atr_value = (
            atr.iloc[i]
            if not pd.isna(atr.iloc[i])
            else atr.iloc[i:].dropna().iloc[0]
            if not atr.iloc[i:].dropna().empty
            else 0.001
        )

        # Calculate gaps between bars
        # Bullish IFVG: bar1 low, bar3 high create the zone
        # Bearish IFVG: bar1 high, bar3 low create the zone

        # Check for Bullish IFVG (gap up)
        # bar1 high < bar3 low (gap between bar1 and bar3)
        bullish_gap = bar3["Low"] - bar1["High"]
        if bullish_gap > (atr_mult * atr_value):
            zero_overlap = _validate_fvg_zero_overlap(df, i, "bullish")
            ifvg = IFVG(
                high=bar3["Low"],
                low=bar1["High"],
                direction="bullish",
                start_bar=i,
                end_bar=i + 2,
                atr_at_creation=atr_value,
                gap_size=bullish_gap,
                created_at=df.index[i] if isinstance(df.index, pd.DatetimeIndex) else None,
                zero_overlap=zero_overlap,
            )
            ifvgs.append(ifvg)
            logger.debug(
                f"Bullish IFVG detected at bar {i}: "
                f"zone={ifvg.low:.4f}-{ifvg.high:.4f}, size={ifvg.gap_size:.4f}"
            )

        # Check for Bearish IFVG (gap down)
        # bar1 low > bar3 high (gap between bar1 and bar3)
        bearish_gap = bar1["Low"] - bar3["High"]
        if bearish_gap > (atr_mult * atr_value):
            zero_overlap = _validate_fvg_zero_overlap(df, i, "bearish")
            ifvg = IFVG(
                high=bar1["Low"],
                low=bar3["High"],
                direction="bearish",
                start_bar=i,
                end_bar=i + 2,
                atr_at_creation=atr_value,
                gap_size=bearish_gap,
                created_at=df.index[i] if isinstance(df.index, pd.DatetimeIndex) else None,
                zero_overlap=zero_overlap,
            )
            ifvgs.append(ifvg)
            logger.debug(
                f"Bearish IFVG detected at bar {i}: "
                f"zone={ifvg.low:.4f}-{ifvg.high:.4f}, size={ifvg.gap_size:.4f}"
            )

    return ifvgs


def find_nearest_unfilled_ifvg(
    df: pd.DataFrame,
    ifvgs: List[IFVG],
    current_bar: int,
    atr: float,
    proximity_threshold: float = 1.5,
    direction: Optional[str] = None,
) -> IFVGProximity:
    """
    Find the nearest unfilled IFVG for entry decisions.

    Args:
        df: OHLCV DataFrame
        ifvgs: List of detected IFVGs
        current_bar: Current bar index
        atr: Current ATR value
        proximity_threshold: Maximum distance/ATR ratio to consider
        direction: Filter by direction - 'bullish', 'bearish', or None for both

    Returns:
        IFVGProximity with nearest IFVG details
    """
    if not ifvgs or current_bar >= len(df):
        return IFVGProximity(
            nearest_ifvg=None,
            distance=float("inf"),
            distance_atr_ratio=float("inf"),
            edge_price=0.0,
            is_bullish=False,
            within_proximity=False,
            distance_to_ce=float("inf"),
            distance_ce_atr_ratio=float("inf"),
        )

    current_price = df.iloc[current_bar]["Close"]

    # Filter unfilled IFVGs
    unfilled_ifvgs = [ifvg for ifvg in ifvgs if not ifvg.filled and ifvg.end_bar < current_bar]

    # Filter by direction if specified
    if direction:
        unfilled_ifvgs = [ifvg for ifvg in unfilled_ifvgs if ifvg.direction == direction]

    if not unfilled_ifvgs:
        return IFVGProximity(
            nearest_ifvg=None,
            distance=float("inf"),
            distance_atr_ratio=float("inf"),
            edge_price=0.0,
            is_bullish=False,
            within_proximity=False,
            distance_to_ce=float("inf"),
            distance_ce_atr_ratio=float("inf"),
        )

    # Find nearest IFVG
    nearest = None
    min_distance = float("inf")

    for ifvg in unfilled_ifvgs:
        # Calculate distance to nearest edge
        if current_price > ifvg.high:
            distance = current_price - ifvg.high
            edge_price = ifvg.high
        elif current_price < ifvg.low:
            distance = ifvg.low - current_price
            edge_price = ifvg.low
        else:
            # Price is inside the IFVG
            distance = 0.0
            edge_price = ifvg.low if ifvg.direction == "bullish" else ifvg.high

        if distance < min_distance:
            min_distance = distance
            nearest = ifvg
            nearest_edge = edge_price

    if nearest is None:
        return IFVGProximity(
            nearest_ifvg=None,
            distance=float("inf"),
            distance_atr_ratio=float("inf"),
            edge_price=0.0,
            is_bullish=False,
            within_proximity=False,
            distance_to_ce=float("inf"),
            distance_ce_atr_ratio=float("inf"),
        )

    distance_atr_ratio = min_distance / atr if atr > 0 else float("inf")

    ce_dist = abs(current_price - nearest.ce) if nearest.ce > 0 else float("inf")
    ce_atr_ratio = ce_dist / atr if atr > 0 else float("inf")

    return IFVGProximity(
        nearest_ifvg=nearest,
        distance=min_distance,
        distance_atr_ratio=distance_atr_ratio,
        edge_price=nearest_edge,
        is_bullish=nearest.direction == "bullish",
        within_proximity=distance_atr_ratio <= proximity_threshold,
        distance_to_ce=ce_dist,
        distance_ce_atr_ratio=ce_atr_ratio,
    )


def update_ifvg_status(ifvgs: List[IFVG], df: pd.DataFrame, current_bar: int) -> List[IFVG]:
    """
    Update IFVG fill status based on price action.

    FIXED: An IFVG is considered filled when price ENTERS the zone, not when it
    exits. This is the correct behavior for Fair Value Gaps:
    - Bullish IFVG (support zone): Filled when price drops INTO the zone (high touches or enters)
    - Bearish IFVG (resistance zone): Filled when price rises INTO the zone (low touches or enters)

    Args:
        ifvgs: List of IFVGs to update
        df: OHLCV DataFrame
        current_bar: Current bar index

    Returns:
        Updated list of IFVGs
    """
    for ifvg in ifvgs:
        if ifvg.filled:
            continue

        # Check if price has filled the gap
        bar = df.iloc[current_bar]

        # FIXED: Gap is filled when price ENTERS the zone
        if ifvg.direction == "bullish":
            # Bullish IFVG (support zone) is filled when price drops into the zone
            # Price enters when high touches or goes below the zone top (ifvg.high)
            if bar["High"] <= ifvg.high:
                ifvg.filled = True
                ifvg.fill_bar = current_bar
                logger.debug(f"Bullish IFVG filled at bar {current_bar}")
        else:  # bearish
            # Bearish IFVG (resistance zone) is filled when price rises into the zone
            # Price enters when low touches or goes above the zone bottom (ifvg.low)
            if bar["Low"] >= ifvg.low:
                ifvg.filled = True
                ifvg.fill_bar = current_bar
                logger.debug(f"Bearish IFVG filled at bar {current_bar}")

    return ifvgs


def get_active_ifvgs(ifvgs: List[IFVG], current_bar: int, max_age: int = 50) -> List[IFVG]:
    """
    Get active (unfilled and not too old) IFVGs.

    Args:
        ifvgs: List of all IFVGs
        current_bar: Current bar index
        max_age: Maximum age in bars for an IFVG to be considered active

    Returns:
        List of active IFVGs
    """
    active = []

    for ifvg in ifvgs:
        if ifvg.filled:
            continue

        age = current_bar - ifvg.end_bar
        if age > max_age:
            continue

        active.append(ifvg)

    # Sort by distance (closest first)
    return sorted(active, key=lambda x: x.end_bar, reverse=True)


def _validate_fvg_zero_overlap(df: pd.DataFrame, i: int, direction: str) -> bool:
    """Validate strict ICT FVG zero-wick-overlap rule.

    Bullish FVG: C1 high must NOT overlap C3 low — high[i-2] < low[i].
    Bearish FVG: C1 low must NOT overlap C3 high — low[i-2] > high[i].

    Returns True only if zero overlap exists.
    """
    if i < 2 or i >= len(df):
        return False

    if direction == "bullish":
        return float(df.iloc[i - 2]["High"]) < float(df.iloc[i]["Low"])
    else:
        return float(df.iloc[i - 2]["Low"]) > float(df.iloc[i]["High"])


def _compute_fvg_quality(
    df: pd.DataFrame,
    i: int,
    fvg_direction: str,
    atr: float,
    bos_signals: "np.ndarray | None" = None,
    htf_bias: "np.ndarray | None" = None,
    half_life_bars: float = 20.0,
) -> float:
    """Compute per-bar FVG quality score 0.0-1.0 based on ICT quality filters.

    Components:
    1. BOS-triggered (0.0-0.3): FVG formed during BOS event
    2. Freshness (0.0-0.3): exponential decay, half-life=20 bars
    3. Premium/Discount alignment (0.0-0.4): FVG in correct zone for bias
    4. Size significance (0.0-0.2): gap_size / ATR ratio
    """
    score = 0.0

    if bos_signals is not None and i < len(bos_signals) and bos_signals[i] != 0:
        score += 0.3

    for lag in range(1, min(i, 50) + 1):
        if _validate_fvg_zero_overlap(df, i - lag, fvg_direction):
            freshness = float(np.exp(-lag * np.log(2) / half_life_bars))
            score += 0.3 * freshness
            break

    if htf_bias is not None and i < len(htf_bias) and htf_bias[i] != 0.0:
        bias_dir = "bullish" if htf_bias[i] > 0 else "bearish"
        if fvg_direction == bias_dir:
            score += 0.4

    gap_size = abs(float(df.iloc[i - 2]["High"]) - float(df.iloc[i]["Low"]))
    if atr > 0:
        size_ratio = gap_size / atr
        score += 0.2 * min(size_ratio / 2.0, 1.0)

    return float(np.clip(score, 0.0, 1.0))


def _compute_ce_proximity(
    df: pd.DataFrame, i: int, atr: float, proximity_mult: float = 2.0
) -> Tuple[float, float]:
    """Compute proximity to nearest FVG's Consequent Encroachment (CE) midpoint.

    Returns (ce_proximity_score, distance_ce_atr_ratio).
    """
    if i < 3:
        return 0.0, float("inf")

    close = float(df.iloc[i]["Close"])

    for lag in range(1, min(i, 30)):
        j = i - lag
        if j < 2:
            break

        bullish_fvg = _validate_fvg_zero_overlap(df, j, "bullish")
        bearish_fvg = _validate_fvg_zero_overlap(df, j, "bearish")

        fvg_ce = None
        fvg_dir = None
        if bullish_fvg:
            fvg_ce = (float(df.iloc[j - 2]["High"]) + float(df.iloc[j]["Low"])) / 2.0
            fvg_dir = "bullish"
        elif bearish_fvg:
            fvg_ce = (float(df.iloc[j - 2]["Low"]) + float(df.iloc[j]["High"])) / 2.0
            fvg_dir = "bearish"

        if fvg_ce is not None and fvg_dir is not None:
            distance = abs(close - fvg_ce)
            threshold = proximity_mult * atr
            if distance < threshold and atr > 0:
                if (fvg_dir == "bullish" and close > fvg_ce) or (
                    fvg_dir == "bearish" and close < fvg_ce
                ):
                    score = float(np.clip(1.0 - distance / threshold, 0.1, 1.0))
                    return score, distance / atr
        break

    return 0.0, float("inf")


def calculate_ifvg_entry_zones(
    df: pd.DataFrame, ifvgs: List[IFVG], current_bar: int, atr: float, buffer_mult: float = 0.1
) -> List[Tuple[float, float, str]]:
    """
    Calculate entry zones from active IFVGs.

    Args:
        df: OHLCV DataFrame
        ifvgs: List of IFVGs
        current_bar: Current bar index
        atr: Current ATR value
        buffer_mult: Buffer multiplier for entry zone

    Returns:
        List of (entry_price, stop_price, direction) tuples
    """
    active_ifvgs = get_active_ifvgs(ifvgs, current_bar)
    entry_zones = []

    for ifvg in active_ifvgs:
        buffer = atr * buffer_mult

        if ifvg.direction == "bullish":
            # For bullish IFVG, enter at the low edge
            entry_price = ifvg.low + buffer
            stop_price = ifvg.low - buffer
            entry_zones.append((entry_price, stop_price, "long"))
        else:
            # For bearish IFVG, enter at the high edge
            entry_price = ifvg.high - buffer
            stop_price = ifvg.high + buffer
            entry_zones.append((entry_price, stop_price, "short"))

    return entry_zones
