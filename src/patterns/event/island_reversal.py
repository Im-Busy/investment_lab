"""Island Reversal pattern detector.

Island Top: gap up → isolated trading (1+ sessions) → gap down → powerful bearish reversal
Island Bottom: gap down → isolated trading → gap up → powerful bullish reversal

Source: Kirkpatrick (CMT textbook), NCFE, Duddella 2007
Reliability: Among Best Multi-Bar Patterns for downward signals
"""

import numpy as np
import pandas as pd


def detect_island_reversal(
    ohlc: pd.DataFrame,
    gap_threshold_pct: float = 0.005,
    min_isolation_bars: int = 1,
    max_isolation_bars: int = 10,
) -> np.ndarray:
    """Detect Island Reversals.

    Args:
        ohlc: OHLCV DataFrame
        gap_threshold_pct: Minimum gap as fraction of price
        min_isolation_bars: Minimum bars in the island
        max_isolation_bars: Maximum bars in the island

    Returns:
        signals array: +1 bullish island bottom, -1 bearish island top
    """
    open_ = ohlc["Open"].to_numpy(dtype=np.float64)
    high = ohlc["High"].to_numpy(dtype=np.float64)
    low = ohlc["Low"].to_numpy(dtype=np.float64)
    close = ohlc["Close"].to_numpy(dtype=np.float64)
    n = len(open_)

    signals = np.zeros(n, dtype=np.int8)
    price_level = (close + open_) / 2.0
    threshold = gap_threshold_pct * price_level

    for i in range(1, n - max_isolation_bars - 1):
        # Bearish Island Top: gap UP into island, then gap DOWN out
        if i >= n - max_isolation_bars - 1:
            break
        gap_up_size = open_[i] - close[i - 1]
        if gap_up_size > threshold[i]:
            for iso_len in range(min_isolation_bars, max_isolation_bars + 1):
                if i + iso_len >= n - 1:
                    break
                island_high = float(np.max(high[i : i + iso_len]))
                island_low = float(np.min(low[i : i + iso_len]))
                if close[i - 1] < island_low and open_[i + iso_len] < island_low:
                    gap_down = island_low - open_[i + iso_len]
                    if gap_down > threshold[i + iso_len]:
                        signals[i + iso_len] = -1
                        break

        # Bullish Island Bottom: gap DOWN into island, then gap UP out
        if i >= n - max_isolation_bars - 1:
            break
        gap_down_size = close[i - 1] - open_[i]
        if gap_down_size > threshold[i]:
            for iso_len in range(min_isolation_bars, max_isolation_bars + 1):
                if i + iso_len >= n - 1:
                    break
                island_high = float(np.max(high[i : i + iso_len]))
                island_low = float(np.min(low[i : i + iso_len]))
                if close[i - 1] > island_high and open_[i + iso_len] > island_high:
                    gap_up = open_[i + iso_len] - island_high
                    if gap_up > threshold[i + iso_len]:
                        signals[i + iso_len] = 1
                        break

    return signals
