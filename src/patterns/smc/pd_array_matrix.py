"""PD Array Matrix — ICT 2022 mentorship multi-timeframe Premium/Discount tracking.

The PD (Price Delivery) Array Matrix organizes all SMC structural elements
across timeframes into a hierarchical premium/discount framework.

Source: toaz.info ICT glossary, David Woods
"""

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class PDZone(enum.Enum):
    PREMIUM_EXTREME = "premium_extreme"
    PREMIUM = "premium"
    EQUILIBRIUM = "equilibrium"
    DISCOUNT = "discount"
    DISCOUNT_EXTREME = "discount_extreme"


@dataclass
class PDArray:
    """A single Price Delivery array (level/zone) with metadata."""

    level: float
    zone: PDZone
    timeframe: str  # "D", "4H", "1H", "15m"
    array_type: str  # "OB", "FVG", "Breaker", "Mitigation", "Rejection", "LiquidityVoid"
    direction: int  # +1 support, -1 resistance
    bar_index: int
    active: bool = True
    strength: float = 0.5
    ce_level: Optional[float] = None  # Consequent Encroachment (FVG midpoint)


@dataclass
class PDArrayMatrix:
    """Multi-timeframe collection of PD Arrays for a single instrument."""

    daily_range_high: float = 0.0
    daily_range_low: float = 0.0
    equilibrium: float = 0.0
    arrays: List[PDArray] = field(default_factory=list)

    def is_in_premium(self, price: float) -> bool:
        return price > self.equilibrium

    def is_in_discount(self, price: float) -> bool:
        return price < self.equilibrium

    def get_active_arrays(self, zone: Optional[PDZone] = None) -> List[PDArray]:
        active = [a for a in self.arrays if a.active]
        if zone:
            active = [a for a in active if a.zone == zone]
        return active

    def get_nearest_array(self, price: float, direction: int) -> Optional[PDArray]:
        candidates = [a for a in self.arrays if a.active]
        if direction == 1:
            candidates = [a for a in candidates if a.level > price]
            candidates.sort(key=lambda a: a.level)
        else:
            candidates = [a for a in candidates if a.level < price]
            candidates.sort(key=lambda a: -a.level)
        return candidates[0] if candidates else None


def build_pd_array_matrix(
    ohlc: Optional[pd.DataFrame] = None,
    fvg_result: Optional[pd.DataFrame] = None,
    ob_result: Optional[pd.DataFrame] = None,
    breakers: Optional[list] = None,
    mitigations: Optional[list] = None,
    rejections: Optional[list] = None,
    timeframe: str = "1H",
    tf_data: Optional[Dict[str, dict]] = None,
) -> PDArrayMatrix:
    """Build a PD Array Matrix from SMC detected structures.

    Supports both single-timeframe (backward-compatible) and multi-timeframe
    construction via the optional `tf_data` parameter.

    Args:
        ohlc: Single-TF OHLC dataframe (ignored if tf_data provided).
        fvg_result: Single-TF FVG detection output (ignored if tf_data provided).
        ob_result: Single-TF OB detection output (ignored if tf_data provided).
        breakers: Single-TF breaker blocks (ignored if tf_data provided).
        mitigations: Single-TF mitigation blocks (ignored if tf_data provided).
        rejections: Single-TF rejection blocks (ignored if tf_data provided).
        timeframe: Default timeframe string for single-TF mode.
        tf_data: Optional multi-TF dict keyed by timeframe ("D", "4H", "1H",
            "15m"). Each value is a dict with keys matching the single-TF
            params: {"ohlc", "fvg_result", "ob_result", "breakers",
            "mitigations", "rejections"}. When provided, single-TF params
            are ignored and arrays are built per timeframe.

    Returns:
        PDArrayMatrix with arrays populated from all timeframes.
    """
    breakers = breakers or []
    mitigations = mitigations or []
    rejections = rejections or []

    if tf_data:
        return _build_multi_tf_matrix(tf_data)

    if ohlc is None or fvg_result is None or ob_result is None:
        raise ValueError(
            "ohlc, fvg_result, and ob_result are required when tf_data is not provided"
        )

    return _build_single_tf_matrix(
        ohlc, fvg_result, ob_result, breakers, mitigations, rejections, timeframe
    )


def _build_single_tf_matrix(
    ohlc: pd.DataFrame,
    fvg_result: pd.DataFrame,
    ob_result: pd.DataFrame,
    breakers: list,
    mitigations: list,
    rejections: list,
    timeframe: str,
) -> PDArrayMatrix:
    """Build PDArrayMatrix for a single timeframe."""
    high = ohlc["High"]
    low = ohlc["Low"]

    daily_high = float(high.max())
    daily_low = float(low.min())
    equilibrium = (daily_high + daily_low) / 2.0

    matrix = PDArrayMatrix(
        daily_range_high=daily_high,
        daily_range_low=daily_low,
        equilibrium=equilibrium,
    )

    _populate_arrays(
        matrix, ohlc, fvg_result, ob_result, breakers, mitigations, rejections, timeframe
    )
    return matrix


def _build_multi_tf_matrix(tf_data: Dict[str, dict]) -> PDArrayMatrix:
    """Build PDArrayMatrix from multiple timeframe data dicts.

    Computes daily range and equilibrium from the "D" (daily) OHLC if available,
    otherwise falls back to the highest timeframe.
    """
    # Determine dealing range from daily TF if present, else highest TF
    if "D" in tf_data and "ohlc" in tf_data["D"]:
        daily_ohlc = tf_data["D"]["ohlc"]
    elif "1D" in tf_data and "ohlc" in tf_data["1D"]:
        daily_ohlc = tf_data["1D"]["ohlc"]
    else:
        max_tf = max(tf_data.keys(), key=lambda k: _tf_priority(k))
        daily_ohlc = tf_data[max_tf]["ohlc"]

    daily_high = float(daily_ohlc["High"].max())
    daily_low = float(daily_ohlc["Low"].min())
    equilibrium = (daily_high + daily_low) / 2.0

    matrix = PDArrayMatrix(
        daily_range_high=daily_high,
        daily_range_low=daily_low,
        equilibrium=equilibrium,
    )

    for tf, data in tf_data.items():
        ohlc = data.get("ohlc")
        fvg_result = data.get("fvg_result")
        ob_result = data.get("ob_result")
        breakers = data.get("breakers", [])
        mitigations = data.get("mitigations", [])
        rejections = data.get("rejections", [])

        if ohlc is None or fvg_result is None or ob_result is None:
            continue

        _populate_arrays(matrix, ohlc, fvg_result, ob_result, breakers, mitigations, rejections, tf)

    return matrix


def _populate_arrays(
    matrix: PDArrayMatrix,
    ohlc: pd.DataFrame,
    fvg_result: pd.DataFrame,
    ob_result: pd.DataFrame,
    breakers: list,
    mitigations: list,
    rejections: list,
    timeframe: str,
) -> None:
    """Populate arrays into matrix for a given timeframe."""
    daily_high = matrix.daily_range_high
    daily_low = matrix.daily_range_low

    # Classify FVGs
    for i in range(len(ohlc)):
        val = fvg_result["FVG"].iloc[i]
        if pd.isna(val) or val == 0:
            continue
        direction = int(val)
        top = fvg_result["Top"].iloc[i]
        bottom = fvg_result["Bottom"].iloc[i]
        level = top if direction == 1 else bottom
        ce = (top + bottom) / 2.0
        matrix.arrays.append(
            PDArray(
                level=level,
                zone=_classify_zone(level, daily_high, daily_low),
                timeframe=timeframe,
                array_type="FVG",
                direction=direction,
                bar_index=i,
                ce_level=ce,
            )
        )

    # Classify OBs
    for i in range(len(ohlc)):
        val = ob_result["OB"].iloc[i]
        if pd.isna(val) or val == 0:
            continue
        direction = int(val)
        top = ob_result["Top"].iloc[i]
        bottom = ob_result["Bottom"].iloc[i]
        level = bottom if direction == 1 else top
        strength = (
            ob_result["Percentage"].iloc[i] / 100.0
            if not pd.isna(ob_result["Percentage"].iloc[i])
            else 0.5
        )
        matrix.arrays.append(
            PDArray(
                level=level,
                zone=_classify_zone(level, daily_high, daily_low),
                timeframe=timeframe,
                array_type="OB",
                direction=direction,
                bar_index=i,
                strength=float(np.clip(strength, 0.0, 1.0)),
            )
        )

    # Classify Breaker Blocks
    for breaker in breakers:
        level = breaker.bottom if breaker.direction == 1 else breaker.top
        matrix.arrays.append(
            PDArray(
                level=level,
                zone=_classify_zone(level, daily_high, daily_low),
                timeframe=timeframe,
                array_type="Breaker",
                direction=breaker.direction,
                bar_index=breaker.bar_index,
                strength=breaker.strength,
            )
        )

    for mit in mitigations:
        matrix.arrays.append(
            PDArray(
                level=mit.level,
                zone=_classify_zone(mit.level, daily_high, daily_low),
                timeframe=timeframe,
                array_type="Mitigation",
                direction=mit.direction,
                bar_index=mit.bar_index,
                strength=mit.strength,
            )
        )

    for rej in rejections:
        matrix.arrays.append(
            PDArray(
                level=rej.level,
                zone=_classify_zone(rej.level, daily_high, daily_low),
                timeframe=timeframe,
                array_type="Rejection",
                direction=rej.direction,
                bar_index=rej.bar_index,
                strength=rej.confidence,
            )
        )


def _classify_zone(level: float, range_high: float, range_low: float) -> PDZone:
    """Classify a price level into PD Zone based on dealing range."""
    range_size = range_high - range_low
    if range_size <= 0:
        return PDZone.EQUILIBRIUM
    position = (level - range_low) / range_size

    if position > 0.85:
        return PDZone.PREMIUM_EXTREME
    elif position > 0.60:
        return PDZone.PREMIUM
    elif position > 0.35:
        return PDZone.EQUILIBRIUM
    elif position > 0.15:
        return PDZone.DISCOUNT
    else:
        return PDZone.DISCOUNT_EXTREME


_TF_PRIORITY = {"D": 4, "1D": 4, "4H": 3, "2H": 2, "1H": 2, "15m": 1}


def _tf_priority(tf: str) -> int:
    return _TF_PRIORITY.get(tf, 0)


_PREMIUM_ZONES = frozenset({PDZone.PREMIUM, PDZone.PREMIUM_EXTREME})
_DISCOUNT_ZONES = frozenset({PDZone.DISCOUNT, PDZone.DISCOUNT_EXTREME})


def select_entry_array(
    matrix: PDArrayMatrix,
    current_price: float,
    bias: str,
    max_distance_pct: float = 0.10,
) -> Optional[Tuple[PDArray, float]]:
    """Apply ICT entry selection rule to find the nearest PD array in the correct zone.

    ICT rule: in a bullish bias, price should retrace into the discount zone
    and the trader looks for the first PD array below price. In a bearish bias,
    price retraces into the premium zone and the trader looks for the first
    PD array above price.

    Args:
        matrix: The PD Array Matrix to search.
        current_price: Current price level.
        bias: "bullish" or "bearish".
        max_distance_pct: Max price distance as fraction (default 10%).

    Returns:
        Tuple of (PDArray, absolute_distance) or None if no match found.
    """
    bias_lower = bias.lower()
    if bias_lower not in ("bullish", "bearish"):
        return None

    max_dist = current_price * max_distance_pct

    if bias_lower == "bullish":
        # Scan discount zone for nearest PD array below price
        candidates = [
            a
            for a in matrix.arrays
            if a.active and a.zone in _DISCOUNT_ZONES and a.level < current_price
        ]
        if not candidates:
            return None
        nearest = max(candidates, key=lambda a: a.level)
    else:
        # Scan premium zone for nearest PD array above price
        candidates = [
            a
            for a in matrix.arrays
            if a.active and a.zone in _PREMIUM_ZONES and a.level > current_price
        ]
        if not candidates:
            return None
        nearest = min(candidates, key=lambda a: a.level)

    distance = abs(nearest.level - current_price)
    if distance > max_dist:
        return None

    return nearest, distance


def get_confluent_arrays(
    matrix: PDArrayMatrix,
    price_tolerance_pct: float = 0.005,
    min_timeframes: int = 2,
    timeframe_order: Optional[List[str]] = None,
) -> List[Tuple[Tuple[float, float], List[PDArray]]]:
    """Find arrays at the same price level across multiple timeframes.

    Clusters arrays whose levels are within `price_tolerance_pct` of each other,
    then returns groups that span at least `min_timeframes` distinct timeframes.

    Args:
        matrix: The PD Array Matrix to search.
        price_tolerance_pct: Max % deviation to consider levels "the same" (default 0.5%).
        min_timeframes: Minimum distinct timeframes for a cluster to qualify.
        timeframe_order: Optional priority ordering of TFs for sorting clusters.

    Returns:
        List of ((avg_level, max_deviation), [PDArray, ...]) tuples, sorted by
        decreasing number of distinct timeframes then decreasing array count.
    """
    active = [a for a in matrix.arrays if a.active]
    if not active:
        return []

    # Cluster by proximity
    sorted_arrays = sorted(active, key=lambda a: a.level)
    clusters: List[List[PDArray]] = []
    current_cluster = [sorted_arrays[0]]

    for arr in sorted_arrays[1:]:
        ref_level = current_cluster[0].level
        if ref_level == 0:
            current_cluster.append(arr)
            continue
        if abs(arr.level - ref_level) / abs(ref_level) <= price_tolerance_pct:
            current_cluster.append(arr)
        else:
            clusters.append(current_cluster)
            current_cluster = [arr]
    clusters.append(current_cluster)

    # Filter by min_timeframes
    results: List[Tuple[Tuple[float, float], List[PDArray]]] = []
    for cluster in clusters:
        tfs = {a.timeframe for a in cluster}
        if len(tfs) < min_timeframes:
            continue
        levels = [a.level for a in cluster]
        avg_level = sum(levels) / len(levels)
        max_dev = max(abs(lev - avg_level) for lev in levels)
        results.append(((avg_level, max_dev), cluster))

    # Sort by TF diversity desc, then array count desc
    results.sort(
        key=lambda r: (
            -len({a.timeframe for a in r[1]}),
            -len(r[1]),
        )
    )
    return results
