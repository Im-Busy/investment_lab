"""Adam-Eve Pattern — sharp (Adam) vs rounded (Eve) double tops/bottoms.

Adam = Sharp V-spike (1-2 bars)
Eve = Rounded, drawn-out formation (3+ bars)
Entry: Breakout of the middle between the two formations.

Source: Duddella 2007 Ch.11
"""

import numpy as np
import pandas as pd


def detect_adam_eve(
    ohlc: pd.DataFrame,
    swing_highs: np.ndarray,
    swing_high_levels: np.ndarray,
    swing_lows: np.ndarray,
    swing_low_levels: np.ndarray,
    atr: np.ndarray,
) -> np.ndarray:
    """Detect Adam-Eve patterns. +1 bullish, -1 bearish.

    Classification heuristic:
    - Adam: 1-2 bars, sharp reversal, wick-to-body > 2
    - Eve: 3+ bars, rounded, gradual reversal
    """
    close = ohlc["Close"].to_numpy(dtype=np.float64)
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)

    sw_high_idx = np.where(swing_highs == 1)[0]
    sw_low_idx = np.where(swing_lows == 1)[0]

    def _classify_formation(swing_indices: np.ndarray, idx_in_list: int) -> str:
        """Classify a formation as Adam or Eve based on bar count and sharpness."""
        if idx_in_list >= len(swing_indices):
            return "adam"
        current = swing_indices[idx_in_list]
        prev = swing_indices[idx_in_list - 1] if idx_in_list > 0 else current - 1
        bars_apart = current - prev
        if bars_apart <= 2:
            return "adam"
        return "eve"

    # Bearish Adam-Eve (double top): two swing highs close in price
    for i in range(len(sw_high_idx) - 1):
        first_idx = sw_high_idx[i]
        second_idx = sw_high_idx[i + 1]

        if second_idx >= n:
            continue

        first_level = swing_high_levels[first_idx]
        second_level = swing_high_levels[second_idx]

        # Peaks must be close in price (within 1 ATR)
        atr_val = atr[second_idx]
        if abs(first_level - second_level) > 0.5 * atr_val:
            continue
        if second_idx - first_idx < 5:
            continue

        first_type = _classify_formation(sw_high_idx, i)
        second_type = _classify_formation(sw_high_idx, i + 1)

        # Find the middle valley
        valley_level = float("inf")
        for l_idx in sw_low_idx:
            if first_idx < l_idx < second_idx:
                valley_level = min(valley_level, swing_low_levels[l_idx])

        if valley_level == float("inf"):
            continue

        # Entry: close below valley
        for j in range(second_idx + 1, min(second_idx + 15, n)):
            if close[j] < valley_level:
                signals[j] = -1
                break

    # Bullish Adam-Eve (double bottom)
    for i in range(len(sw_low_idx) - 1):
        first_idx = sw_low_idx[i]
        second_idx = sw_low_idx[i + 1]

        if second_idx >= n:
            continue

        first_level = swing_low_levels[first_idx]
        second_level = swing_low_levels[second_idx]

        atr_val = atr[second_idx]
        if abs(first_level - second_level) > 0.5 * atr_val:
            continue
        if second_idx - first_idx < 5:
            continue

        # Find the middle peak
        peak_level = 0.0
        for h in sw_high_idx:
            if first_idx < h < second_idx:
                peak_level = max(peak_level, swing_high_levels[h])

        if peak_level <= 0:
            continue

        for j in range(second_idx + 1, min(second_idx + 15, n)):
            if close[j] > peak_level:
                signals[j] = 1
                break

    return signals
