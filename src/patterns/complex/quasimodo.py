"""Quasimodo Pattern detector — high-probability reversal (CPF resource).

Structure:
- Left shoulder at level L1
- Head at a higher high (bull trap)
- Right shoulder at level R1 (< L1)
- Neckline break below R1
- Pullback above neckline (false breakout)
- Then breakdown for entry
"""

import numpy as np


def detect_quasimodo(
    swing_highs: np.ndarray,
    swing_high_levels: np.ndarray,
    swing_lows: np.ndarray,
    swing_low_levels: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
) -> np.ndarray:
    """Detect Quasimodo reversal patterns. +1 bullish, -1 bearish.

    Bearish Quasimodo:
        Left Shoulder → Higher Head → Lower Right Shoulder → neckline break → entry

    Bullish Quasimodo:
        Left Shoulder → Lower Head → Higher Right Shoulder → neckline break → entry
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)

    sw_high_idx = np.where(swing_highs == 1)[0]
    sw_low_idx = np.where(swing_lows == 1)[0]

    # Bearish Quasimodo: LShoulder(high) → Head(higher high) → RShoulder(lower high)
    for i in range(len(sw_high_idx) - 2):
        ls_idx = sw_high_idx[i]
        head_idx = sw_high_idx[i + 1]
        rs_idx = sw_high_idx[i + 2]

        if any(idx >= n for idx in (ls_idx, head_idx, rs_idx)):
            continue

        ls_level = swing_high_levels[ls_idx]
        head_level = swing_high_levels[head_idx]
        rs_level = swing_high_levels[rs_idx]

        # Head must be higher than left shoulder
        if head_level <= ls_level:
            continue
        # Right shoulder must be lower than left shoulder
        if rs_level >= ls_level:
            continue

        atr_val = atr[rs_idx]
        # Neckline: lowest low between left shoulder and right shoulder
        between_low = ls_level
        for l_idx in sw_low_idx:
            if ls_idx < l_idx < rs_idx:
                between_low = min(between_low, swing_low_levels[l_idx])

        # Entry: close below neckline after right shoulder
        for j in range(rs_idx + 1, min(rs_idx + 15, n)):
            if close[j] < between_low - 0.3 * atr_val:
                signals[j] = -1
                break

    # Bullish Quasimodo: LShoulder(low) → Head(lower low) → RShoulder(higher low)
    for i in range(len(sw_low_idx) - 2):
        ls_idx = sw_low_idx[i]
        head_idx = sw_low_idx[i + 1]
        rs_idx = sw_low_idx[i + 2]

        if any(idx >= n for idx in (ls_idx, head_idx, rs_idx)):
            continue

        ls_level = swing_low_levels[ls_idx]
        head_level = swing_low_levels[head_idx]
        rs_level = swing_low_levels[rs_idx]

        if head_level >= ls_level:
            continue
        if rs_level <= ls_level:
            continue

        atr_val = atr[rs_idx]
        between_high = ls_level
        for h_idx in sw_high_idx:
            if ls_idx < h_idx < rs_idx:
                between_high = max(between_high, swing_high_levels[h_idx])

        for j in range(rs_idx + 1, min(rs_idx + 15, n)):
            if close[j] > between_high + 0.3 * atr_val:
                signals[j] = 1
                break

    return signals
