"""Supply & Demand Zone Patterns — RBD/DBR/RBR/DBD detection.

Detects the 4 S&D zone patterns from ICT tutorial Day 11:
- RBD (Rally-Base-Drop): price rallies → consolidates (base) → drops sharply → bearish supply
- DBR (Drop-Base-Rally): price drops → consolidates → rallies sharply → bullish demand
- RBR (Rally-Base-Rally): price rallies → pulls back (base) → continues up → bullish continuation demand
- DBD (Drop-Base-Drop): price drops → consolidates → continues down → bearish continuation supply
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class SDZone:
    """Supply & Demand zone detection result."""

    zone_type: str  # 'RBD', 'DBR', 'RBR', 'DBD'
    direction: int  # +1 bullish (demand), -1 bearish (supply)
    zone_high: float
    zone_low: float
    base_start: int  # bar index where base/consolidation starts
    base_end: int  # bar index where base/consolidation ends
    detection_bar: int  # bar where zone was confirmed
    strength: float  # 0.0-1.0
    active: bool = True

    def to_dict(self) -> dict:
        return {
            "zone_type": self.zone_type,
            "direction": self.direction,
            "zone_high": self.zone_high,
            "zone_low": self.zone_low,
            "base_start": self.base_start,
            "base_end": self.base_end,
            "detection_bar": self.detection_bar,
            "strength": self.strength,
            "active": self.active,
        }


def detect_sd_zones(
    df: pd.DataFrame,
    atr: "np.ndarray | pd.Series",
    min_base_bars: int = 2,
    max_base_bars: int = 8,
    min_move_atr: float = 0.5,
) -> List[SDZone]:
    """Detect Supply & Demand zone patterns.

    Detection:
    - RBD (Rally-Base-Drop): price rallies → consolidates (base) → drops sharply → bearish supply zone
    - DBR (Drop-Base-Rally): price drops → consolidates → rallies sharply → bullish demand zone
    - RBR (Rally-Base-Rally): price rallies → pulls back (base) → continues up → bullish continuation demand
    - DBD (Drop-Base-Drop): price drops → consolidates → continues down → bearish continuation supply

    Args:
        df: OHLCV DataFrame.
        atr: ATR values.
        min_base_bars: Minimum bars for base/consolidation.
        max_base_bars: Maximum bars for base.
        min_move_atr: Minimum move size as ATR multiple.

    Returns:
        List of SDZone objects.
    """
    zones: List[SDZone] = []
    n = len(df)

    if n < min_base_bars + 6:
        return zones

    close = df["Close"].to_numpy(dtype=np.float64)
    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)

    if isinstance(atr, pd.Series):
        atr_vals = atr.to_numpy(dtype=np.float64)
    else:
        atr_vals = atr

    i = min_base_bars + 2
    while i < n - min_base_bars - 2:
        atr_val = (
            float(atr_vals[i])
            if i < len(atr_vals) and not np.isnan(atr_vals[i])
            else float(close[i]) * 0.01
        )
        if atr_val <= 0:
            atr_val = float(close[i]) * 0.01

        prev_move = close[i - 1] - close[i - min_base_bars - 2]
        prev_move_abs = abs(prev_move)

        if prev_move_abs < min_move_atr * atr_val:
            i += 1
            continue

        prev_direction = 1 if prev_move > 0 else -1

        base_start = i
        base_end = i
        base_high = float(high[i])
        base_low = float(low[i])

        for j in range(i, min(i + max_base_bars, n - 2)):
            base_high = max(base_high, float(high[j]))
            base_low = min(base_low, float(low[j]))
            range_size = base_high - base_low
            if range_size < 0.3 * atr_val:
                base_end = j
            else:
                break

        base_bars = base_end - base_start + 1
        if base_bars < min_base_bars:
            i = base_end + 1
            continue

        if base_end + 2 >= n:
            break

        next_move = close[base_end + 2] - close[base_end + 1]
        next_move_abs = abs(next_move)
        next_direction = 1 if next_move > 0 else -1

        zone_type = None
        if prev_direction == 1 and next_direction == -1 and next_move_abs > min_move_atr * atr_val:
            zone_type = "RBD"
            direction = -1
        elif (
            prev_direction == -1 and next_direction == 1 and next_move_abs > min_move_atr * atr_val
        ):
            zone_type = "DBR"
            direction = 1
        elif prev_direction == 1 and next_direction == 1 and next_move_abs > min_move_atr * atr_val:
            zone_type = "RBR"
            direction = 1
        elif (
            prev_direction == -1 and next_direction == -1 and next_move_abs > min_move_atr * atr_val
        ):
            zone_type = "DBD"
            direction = -1

        if zone_type is not None:
            reversal_strength = 0.7 if zone_type in ("RBD", "DBR") else 0.55
            size_score = min(1.0, next_move_abs / (atr_val * 2.0)) * 0.3
            strength = float(np.clip(reversal_strength + size_score, 0.0, 1.0))

            zones.append(
                SDZone(
                    zone_type=zone_type,
                    direction=direction,
                    zone_high=base_high,
                    zone_low=base_low,
                    base_start=base_start,
                    base_end=base_end,
                    detection_bar=base_end + 2,
                    strength=strength,
                )
            )
            i = base_end + 3
        else:
            i = base_end + 1

    return zones


def get_sd_zone_at_bar(zones: List[SDZone], bar_idx: int) -> Optional[SDZone]:
    """Get the active SD zone at a specific bar."""
    for zone in reversed(zones):
        if zone.detection_bar <= bar_idx and zone.active:
            return zone
    return None
