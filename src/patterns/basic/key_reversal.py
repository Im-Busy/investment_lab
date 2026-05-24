"""Key Reversal bar — end-of-trend indicator (NCFE).

Bearish Key Reversal: New high but closes lower than previous close, on high volume
Bullish Key Reversal: New low but closes higher than previous close, on high volume
"""

import numpy as np
import pandas as pd


def detect_key_reversal(ohlc: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    """Detect Key Reversal bars. +1 bullish, -1 bearish.

    Bearish: Makes new high (above lookback max) but closes below previous bar's close
    Bullish: Makes new low (below lookback min) but closes above previous bar's close
    """
    high = ohlc["High"].to_numpy(dtype=np.float64)
    low = ohlc["Low"].to_numpy(dtype=np.float64)
    close = ohlc["Close"].to_numpy(dtype=np.float64)
    volume = ohlc["Volume"].to_numpy(dtype=np.float64)
    n = len(close)

    signals = np.zeros(n, dtype=np.int8)

    for i in range(lookback + 1, n):
        recent_highs = high[i - lookback : i]
        recent_lows = low[i - lookback : i]
        lookback_max = float(np.max(recent_highs))
        lookback_min = float(np.min(recent_lows))

        # Bearish Key Reversal: new high → close down
        if high[i] > lookback_max and close[i] < close[i - 1]:
            avg_vol = float(np.mean(volume[i - 20 : i]))
            if avg_vol > 0 and volume[i] > 1.2 * avg_vol:
                signals[i] = -1

        # Bullish Key Reversal: new low → close up
        if low[i] < lookback_min and close[i] > close[i - 1]:
            avg_vol = float(np.mean(volume[i - 20 : i]))
            if avg_vol > 0 and volume[i] > 1.2 * avg_vol:
                signals[i] = 1

    return signals
