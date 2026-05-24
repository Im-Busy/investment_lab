"""Three Valleys and A River — inverse of Three Hills (Duddella 2007 Ch.10).

Structure: 3 progressively higher valleys (swing lows) within a rising trendline channel.
Entry: Break above trendline connecting the 3 valley peaks.
"""

import numpy as np


def detect_three_valleys(
    swing_lows: np.ndarray,
    swing_low_levels: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
) -> np.ndarray:
    """Detect Three Valleys patterns. +1 on bullish breakout.

    Three progressively higher swing lows forming a rising support trendline.
    Entry on close above the trendline projection.
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)

    sw_low_idx = np.where(swing_lows == 1)[0]

    for i in range(len(sw_low_idx) - 2):
        v1_idx = sw_low_idx[i]
        v2_idx = sw_low_idx[i + 1]
        v3_idx = sw_low_idx[i + 2]

        if any(idx >= n for idx in (v1_idx, v2_idx, v3_idx)):
            continue

        v1_level = swing_low_levels[v1_idx]
        v2_level = swing_low_levels[v2_idx]
        v3_level = swing_low_levels[v3_idx]

        # Three progressively higher valleys
        if not (v1_level < v2_level < v3_level):
            continue

        # Each valley must be separated by at least 3 bars
        if v3_idx - v1_idx < 8:
            continue

        atr_val = atr[v3_idx]
        if atr_val <= 0:
            continue

        # Trendline slope from v1 to v3
        slope = (v3_level - v1_level) / (v3_idx - v1_idx)

        # Entry: close above trendline projection from v3 forward
        for j in range(v3_idx + 1, min(v3_idx + 15, n)):
            trendline = v3_level + slope * (j - v3_idx)
            if close[j] > trendline + 0.2 * atr_val:
                signals[j] = 1
                break

    return signals
