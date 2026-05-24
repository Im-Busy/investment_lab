"""POI (Point of Interest) Grading System.

Grades potential POIs against the 4 SMC quality criteria from tutorial Day 14:
1. BOS-triggered: zone must have caused BOS or market structure shift
2. Liquidity-protected: resting orders (stops) just beyond the zone
3. Unmitigated: zone never fully retested (fresh)
4. Closest-to-price: of all qualified zones, trade nearest first
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class POIGrade:
    """POI quality grading result."""

    criteria_pass: Tuple[bool, bool, bool, bool]  # 4 pass/fail
    score: int  # 0-4
    is_tradeable: bool  # all 4 must pass or trade is rejected
    details: Dict[str, str]  # per-criterion explanation

    def to_dict(self) -> dict:
        return {
            "criteria_pass": list(self.criteria_pass),
            "score": self.score,
            "is_tradeable": self.is_tradeable,
            "details": self.details,
        }


def grade_poi(
    zone_info: dict,
    bos_history: np.ndarray,
    liquidity_levels: np.ndarray,
    retest_history: np.ndarray,
    distance_to_price: float,
) -> POIGrade:
    """Grade a potential POI against 4 SMC quality criteria.

    Args:
        zone_info: Dict with zone details (level, direction, bar_index, zone_type).
        bos_history: BOS signal array (1/-1 at bars where BOS occurred).
        liquidity_levels: Array of nearby liquidity levels (swing highs/lows, session extremes).
        retest_history: Boolean array indicating bars where zone was retested.
        distance_to_price: Current distance from price to the zone.

    Returns:
        POIGrade with score 0-4 and pass/fail on each criterion.
    """
    details: Dict[str, str] = {}
    passes = [False, False, False, False]

    zone_level = zone_info.get("level", 0.0)
    zone_direction = zone_info.get("direction", 0)
    zone_bar = zone_info.get("bar_index", 0)

    # Criterion 1: BOS-triggered (0.25)
    if zone_bar < len(bos_history) and bos_history[zone_bar] != 0:
        if zone_direction > 0 and bos_history[zone_bar] > 0:
            passes[0] = True
            details["bos_triggered"] = "pass"
        elif zone_direction < 0 and bos_history[zone_bar] < 0:
            passes[0] = True
            details["bos_triggered"] = "pass"
        else:
            details["bos_triggered"] = "fail: BOS direction mismatch"
    else:
        details["bos_triggered"] = "fail: no BOS at zone formation"

    # Criterion 2: Liquidity-protected (0.25)
    if len(liquidity_levels) > 0:
        if zone_direction > 0:
            liquidity_below = liquidity_levels[liquidity_levels < zone_level]
            if len(liquidity_below) > 0:
                closest_liq = float(np.max(liquidity_below))
                dist = zone_level - closest_liq
                if dist > 0 and dist < zone_level * 0.05:
                    passes[1] = True
                    details["liquidity_protected"] = f"pass: liquidity at {closest_liq:.2f}"
                else:
                    details["liquidity_protected"] = "fail: no nearby liquidity below"
            else:
                details["liquidity_protected"] = "fail: no liquidity below zone"
        else:
            liquidity_above = liquidity_levels[liquidity_levels > zone_level]
            if len(liquidity_above) > 0:
                closest_liq = float(np.min(liquidity_above))
                dist = closest_liq - zone_level
                if dist > 0 and dist < zone_level * 0.05:
                    passes[1] = True
                    details["liquidity_protected"] = f"pass: liquidity at {closest_liq:.2f}"
                else:
                    details["liquidity_protected"] = "fail: no nearby liquidity above"
            else:
                details["liquidity_protected"] = "fail: no liquidity above zone"
    else:
        details["liquidity_protected"] = "fail: no liquidity levels available"

    # Criterion 3: Unmitigated (never fully retested)
    if zone_bar < len(retest_history):
        future_retests = retest_history[
            max(0, zone_bar - 5) : min(len(retest_history), zone_bar + 50)
        ]
        if not np.any(future_retests):
            passes[2] = True
            details["unmitigated"] = "pass: zone never retested"
        else:
            retest_bars = np.where(future_retests)[0]
            if len(retest_bars) > 0:
                last_retest = zone_bar + retest_bars[-1]
                bars_since = zone_bar - last_retest if last_retest < zone_bar else 0
                if bars_since > 20:
                    passes[2] = True
                    details["unmitigated"] = "pass: zone fresh (not retested recently)"
                else:
                    details["unmitigated"] = "fail: zone retested recently"
            else:
                passes[2] = True
                details["unmitigated"] = "pass"
    else:
        passes[2] = True
        details["unmitigated"] = "insufficient data, pass by default"

    # Criterion 4: Closest-to-price (0.25)
    if distance_to_price < 5.0:  # within reasonable proximity (ATR normalized)
        passes[3] = True
        details["closest_to_price"] = f"pass: distance={distance_to_price:.2f} ATR"
    else:
        details["closest_to_price"] = f"fail: distance={distance_to_price:.2f} ATR too far"

    score = sum(1 for p in passes if p)
    is_tradeable = all(passes)

    return POIGrade(
        criteria_pass=tuple(passes),
        score=score,
        is_tradeable=is_tradeable,
        details=details,
    )


def grade_poi_batch(
    zones: List[dict],
    bos_history: np.ndarray,
    liquidity_levels: np.ndarray,
    retest_history: np.ndarray,
    current_price: float,
    atr: float,
) -> List[POIGrade]:
    """Grade multiple POIs and rank by quality."""
    grades = []
    for zone in zones:
        dist = abs(current_price - zone.get("level", 0.0)) / max(atr, 1e-10)
        grade = grade_poi(zone, bos_history, liquidity_levels, retest_history, dist)
        grades.append(grade)
    grades.sort(key=lambda g: g.score, reverse=True)
    return grades
