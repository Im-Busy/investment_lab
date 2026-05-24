"""NR4 (Narrow Range 4) and Inside Bar detectors.

NR4: 4-bar pattern where the 4th bar has narrower range than preceding 3.
     Break above NR4 high = buy; break below NR4 low = sell.
     Source: Toby Crabel, Fidelity, Duddella 2007

Inside Bar: Current bar's range entirely within previous bar's range.
            Signals low volatility → imminent expansion.
            Source: Fidelity, NCFE, Duddella 2007
"""

import numpy as np
import pandas as pd


def detect_nr4(ohlc: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect NR4 patterns.

    Returns:
        nr4_bar: Boolean array marking NR4 bars
        buy_breakout: +1 on bar closing above NR4 high
        sell_breakout: -1 on bar closing below NR4 low
    """
    high = ohlc["High"].to_numpy(dtype=np.float64)
    low = ohlc["Low"].to_numpy(dtype=np.float64)
    close = ohlc["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    ranges = high - low
    nr4_bar = np.zeros(n, dtype=bool)
    buy_breakout = np.zeros(n, dtype=np.int8)
    sell_breakout = np.zeros(n, dtype=np.int8)

    for i in range(3, n):
        current_range = ranges[i]
        prev_ranges = ranges[i - 3 : i]
        if current_range < float(np.min(prev_ranges)):
            nr4_bar[i] = True

    for i in range(4, n):
        if nr4_bar[i - 1]:
            nr4_high = high[i - 1]
            nr4_low = low[i - 1]
            if close[i] > nr4_high:
                buy_breakout[i] = 1
            elif close[i] < nr4_low:
                sell_breakout[i] = -1

    return nr4_bar, buy_breakout, sell_breakout


def detect_inside_bar(ohlc: pd.DataFrame) -> np.ndarray:
    """Detect Inside Bars. Returns +1/-1 at break of inside bar range."""
    high = ohlc["High"].to_numpy(dtype=np.float64)
    low = ohlc["Low"].to_numpy(dtype=np.float64)
    close = ohlc["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    signals = np.zeros(n, dtype=np.int8)

    for i in range(2, n):
        if high[i - 1] <= high[i - 2] and low[i - 1] >= low[i - 2]:
            if close[i] > high[i - 1]:
                signals[i] = 1
            elif close[i] < low[i - 1]:
                signals[i] = -1

    return signals
