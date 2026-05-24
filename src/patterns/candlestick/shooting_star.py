"""Shooting Star / Inverted Hammer single-candle pattern detectors.

Shooting Star (after uptrend): Upper wick >= 2x body, no lower wick → bearish reversal
Inverted Hammer (after downtrend): Same shape → bullish reversal

Source: Fidelity/Kirkpatrick
"""

import numpy as np
import pandas as pd


def detect_shooting_star(ohlc: pd.DataFrame, lookback: int = 5) -> np.ndarray:
    """Detect Shooting Star (bearish reversal) patterns. Returns -1 on signal bars.

    Requirements:
    - Prior uptrend (close > close 5 bars ago)
    - Upper wick >= 2x body
    - Lower wick minimal (< 0.2x body)
    """
    open_ = ohlc["Open"].to_numpy(dtype=np.float64)
    high = ohlc["High"].to_numpy(dtype=np.float64)
    low = ohlc["Low"].to_numpy(dtype=np.float64)
    close = ohlc["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    signals = np.zeros(n, dtype=np.int8)

    for i in range(lookback + 1, n):
        body = abs(close[i] - open_[i])
        if body <= 0:
            continue

        upper_wick = high[i] - max(close[i], open_[i])
        lower_wick = min(close[i], open_[i]) - low[i]

        # Shooting star criteria
        if upper_wick >= 2.0 * body and lower_wick < 0.2 * body:
            # Prior uptrend
            if close[i] > close[i - lookback]:
                signals[i] = -1

    return signals


def detect_inverted_hammer(ohlc: pd.DataFrame, lookback: int = 5) -> np.ndarray:
    """Detect Inverted Hammer (bullish reversal) patterns. Returns +1 on signal bars.

    Requirements:
    - Prior downtrend (close < close 5 bars ago)
    - Upper wick >= 2x body
    - Lower wick minimal (< 0.2x body)
    """
    open_ = ohlc["Open"].to_numpy(dtype=np.float64)
    high = ohlc["High"].to_numpy(dtype=np.float64)
    low = ohlc["Low"].to_numpy(dtype=np.float64)
    close = ohlc["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    signals = np.zeros(n, dtype=np.int8)

    for i in range(lookback + 1, n):
        body = abs(close[i] - open_[i])
        if body <= 0:
            continue

        upper_wick = high[i] - max(close[i], open_[i])
        lower_wick = min(close[i], open_[i]) - low[i]

        if upper_wick >= 2.0 * body and lower_wick < 0.2 * body:
            if close[i] < close[i - lookback]:
                signals[i] = 1

    return signals
