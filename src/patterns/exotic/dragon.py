"""Dragon Pattern detector — Duddella 2007, Ch.12 Exotic Patterns.

Structure:
    A (Head/start high) → B (First Leg low) → C (Hump, 38-50% retrace of AB)
    → D (Second Leg low, mirrors AB range)
Entry: Close above trendline connecting A→C (Head to Hump)
Stop: Below lowest low of the two legs (B or D)
TP1: 1.27 × CD range; TP2: A level (Head)
"""

import numpy as np


def detect_dragon_pattern(
    swing_lows: np.ndarray,
    swing_low_levels: np.ndarray,
    swing_highs: np.ndarray,
    swing_high_levels: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
) -> np.ndarray:
    """Detect Dragon (bullish) patterns.

    Returns +1 at Dragon breakout bars.
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)

    sw_low_indices = np.where(swing_lows == 1)[0]
    sw_high_indices = np.where(swing_highs == 1)[0]

    for idx in range(len(sw_low_indices) - 2):
        a_index = sw_low_indices[idx]
        b_index = sw_low_indices[idx + 1]
        d_index = sw_low_indices[idx + 2]

        if any(i >= n for i in (a_index, b_index, d_index)):
            continue

        a_level = swing_low_levels[a_index]
        b_level = swing_low_levels[b_index]
        d_level = swing_low_levels[d_index]

        # First leg must be downward (B < A)
        if b_level >= a_level:
            continue

        ab_range = a_level - b_level
        if ab_range <= 0:
            continue

        # Second leg approximately equal to first (within 15% ATR)
        atr_val = atr[d_index]
        if abs(b_level - d_level) > 0.15 * atr_val:
            continue

        # Find hump (swing high) between B and D
        hump_candidates = [h for h in sw_high_indices if b_index < h < d_index]
        if not hump_candidates:
            continue
        c_index = hump_candidates[-1]
        c_level = swing_high_levels[c_index]

        # Hump must retrace 38-50% of AB leg
        c_retrace = (c_level - b_level) / ab_range
        if c_retrace < 0.38 or c_retrace > 0.50:
            continue

        # Entry: close above trendline from A (head start) through C (hump)
        for j in range(d_index + 1, min(d_index + 20, n)):
            if c_index <= a_index:
                continue
            slope = (c_level - a_level) / (c_index - a_index)
            trendline_at_j = a_level + slope * (j - a_index)
            if close[j] > trendline_at_j:
                signals[j] = 1
                break

    return signals
