# -*- coding: utf-8 -*-
"""
Change In State of Delivery (CISD) Detection

CISD signals when price delivery shifts from buy-side to sell-side (or vice versa).
Core ICT concept: price closes beyond the opening of a previous delivery sequence,
confirming a momentum shift.

Bullish CISD: price was delivering lower (bearish), then closes ABOVE the opening
             of the bearish delivery candle → shift from bearish to bullish.

Bearish CISD: price was delivering higher (bullish), then closes BELOW the opening
             of the bullish delivery candle → shift from bullish to bearish.

Reference: ICT Institutional SMC Trading methodology.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class CISDInfo:
    """CISD detection result.

    Attributes:
        detected: Whether CISD was detected
        direction: 'bullish' or 'bearish'
        bar_index: Bar index where CISD occurred
        delivery_open: Opening price that was broken
        trigger_close: Close price that triggered the signal
        consecutive_bars: Number of consecutive bars in delivery direction before flip
        strength: Signal strength 0.0-1.0 based on range vs delivery size
    """

    detected: bool
    direction: Optional[str]
    bar_index: Optional[int]
    delivery_open: Optional[float]
    trigger_close: Optional[float]
    consecutive_bars: int
    strength: float

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "direction": self.direction,
            "bar_index": self.bar_index,
            "delivery_open": self.delivery_open,
            "trigger_close": self.trigger_close,
            "consecutive_bars": self.consecutive_bars,
            "strength": self.strength,
        }


def detect_cisd(
    df: pd.DataFrame,
    swing_highs_lows: Optional[pd.DataFrame] = None,
    min_delivery_bars: int = 3,
    use_close: bool = True,
) -> List[CISDInfo]:
    """
    Detect Change in State of Delivery (CISD) signals.

    Scans for sequences of same-direction candles followed by a closing break
    beyond the opening of the first delivery candle.

    Args:
        df: OHLCV DataFrame with columns [Open, High, Low, Close] and DatetimeIndex.
        swing_highs_lows: Optional swing high/low DataFrame from smartmoneyconcepts
            or mss module for enhanced detection. If None, uses raw candle sequences.
        min_delivery_bars: Minimum consecutive same-direction bars for valid delivery.
        use_close: If True, CISD triggers on close beyond delivery open.
            If False, triggers on high/low beyond delivery open.

    Returns:
        List of CISDInfo objects, one per detected CISD.

    Example:
        >>> cisd_signals = detect_cisd(df, min_delivery_bars=3)
        >>> for c in cisd_signals:
        ...     if c.detected:
        ...         print(f"CISD {c.direction} at bar {c.bar_index}, strength={c.strength:.2f}")
    """
    results: List[CISDInfo] = []
    n = len(df)

    if n < min_delivery_bars + 2:
        return results

    close = df["Close"].values
    open_ = df["Open"].values
    high = df["High"].values
    low = df["Low"].values

    i = 0
    while i < n:
        # Find the start of a delivery sequence
        # A delivery sequence is consecutive bars all closing in the same direction
        direction = None
        seq_start = i
        seq_open = float(open_[i])
        seq_high = float(high[i])
        seq_low = float(low[i])

        while i < n:
            if close[i] > open_[i]:
                if direction is None:
                    direction = "bullish"
                    seq_start = i
                    seq_open = float(open_[i])
                    seq_high = float(high[i])
                    seq_low = float(low[i])
                elif direction == "bullish":
                    pass  # continue bullish sequence
                else:
                    # Change from bearish to bullish - check for bearish CISD
                    break

                seq_high = max(seq_high, float(high[i]))
                seq_low = min(seq_low, float(low[i]))
                i += 1
            elif close[i] < open_[i]:
                if direction is None:
                    direction = "bearish"
                    seq_start = i
                    seq_open = float(open_[i])
                    seq_high = float(high[i])
                    seq_low = float(low[i])
                elif direction == "bearish":
                    pass  # continue bearish sequence
                else:
                    # Change from bullish to bearish - check for bullish CISD
                    break

                seq_high = max(seq_high, float(high[i]))
                seq_low = min(seq_low, float(low[i]))
                i += 1
            else:
                # Doji - neutral, continue current direction or skip
                i += 1
                if i - seq_start > 1:
                    break

            # Check if we've accumulated enough bars for CISD detection
            seq_length = i - seq_start
            if seq_length >= min_delivery_bars and i < n:
                # Check for CISD: price closes beyond the delivery open in opposite direction
                # Only check at sequence boundaries
                pass

        seq_length = i - seq_start
        if seq_length >= min_delivery_bars and direction is not None:
            # Now scan forward for CISD confirmation
            # The CISD occurs when price closes beyond seq_open in the OPPOSITE direction
            for j in range(i, n):
                if direction == "bullish":
                    # Was bullish delivery → bearish CISD: close below seq_open
                    if use_close:
                        triggered = close[j] < seq_open
                    else:
                        triggered = low[j] < seq_open

                    if triggered:
                        delivery_range = seq_high - seq_low
                        break_distance = seq_open - close[j] if use_close else seq_open - low[j]
                        strength = min(1.0, break_distance / (delivery_range + 1e-10))
                        results.append(
                            CISDInfo(
                                detected=True,
                                direction="bearish",
                                bar_index=j,
                                delivery_open=seq_open,
                                trigger_close=float(close[j]),
                                consecutive_bars=seq_length,
                                strength=strength,
                            )
                        )
                        break
                else:
                    # Was bearish delivery → bullish CISD: close above seq_open
                    if use_close:
                        triggered = close[j] > seq_open
                    else:
                        triggered = high[j] > seq_open

                    if triggered:
                        delivery_range = seq_high - seq_low
                        break_distance = close[j] - seq_open if use_close else high[j] - seq_open
                        strength = min(1.0, break_distance / (delivery_range + 1e-10))
                        results.append(
                            CISDInfo(
                                detected=True,
                                direction="bullish",
                                bar_index=j,
                                delivery_open=seq_open,
                                trigger_close=float(close[j]),
                                consecutive_bars=seq_length,
                                strength=strength,
                            )
                        )
                        break

    return results


def detect_cisd_vectorized(
    df: pd.DataFrame,
    min_delivery_bars: int = 3,
) -> pd.DataFrame:
    """
    Vectorized CISD detection returning boolean columns.

    Faster than detect_cisd() for large datasets. Returns columns
    that can be used directly in strategy signal arrays.

    Args:
        df: OHLCV DataFrame.
        min_delivery_bars: Minimum delivery bars.

    Returns:
        DataFrame with columns: bullish_cisd, bearish_cisd, cisd_strength.
    """
    n = len(df)
    close = df["Close"].values
    open_ = df["Open"].values
    high = df["High"].values
    low = df["Low"].values

    bullish_cisd = np.zeros(n, dtype=bool)
    bearish_cisd = np.zeros(n, dtype=bool)
    cisd_strength = np.zeros(n, dtype=float)

    if n < min_delivery_bars + 2:
        return pd.DataFrame(
            {
                "bullish_cisd": bullish_cisd,
                "bearish_cisd": bearish_cisd,
                "cisd_strength": cisd_strength,
            },
            index=df.index,
        )

    bullish_bars = close > open_
    bearish_bars = close < open_

    # Find delivery sequences using run-length encoding
    current_run = 1
    run_direction = 0  # 1=bullish, -1=bearish, 0=neutral

    if bullish_bars[0]:
        run_direction = 1
    elif bearish_bars[0]:
        run_direction = -1

    run_starts = [0]
    run_dirs = [run_direction]

    for i in range(1, n):
        bar_dir = 1 if bullish_bars[i] else (-1 if bearish_bars[i] else 0)
        if bar_dir == run_direction and bar_dir != 0:
            current_run += 1
        else:
            run_starts.append(i)
            run_direction = bar_dir
            current_run = 1
        run_dirs.append(run_direction)

    run_starts.append(n)

    # For each run, check if it's >= min_delivery_bars and look for CISD after
    for run_idx in range(len(run_starts) - 1):
        start = run_starts[run_idx]
        end = run_starts[run_idx + 1]
        run_len = end - start
        run_dir = run_dirs[start]

        if run_len < min_delivery_bars or run_dir == 0:
            continue

        delivery_open = float(open_[start])
        delivery_high = float(np.max(high[start:end]))
        delivery_low = float(np.min(low[start:end]))
        delivery_range = delivery_high - delivery_low

        # Look for CISD after the sequence
        for j in range(end, n):
            if run_dir == 1:
                # Was bullish delivery → bearish CISD
                if close[j] < delivery_open:
                    bearish_cisd[j] = True
                    cisd_strength[j] = min(
                        1.0, (delivery_open - close[j]) / (delivery_range + 1e-10)
                    )
                    break
            elif run_dir == -1:
                # Was bearish delivery → bullish CISD
                if close[j] > delivery_open:
                    bullish_cisd[j] = True
                    cisd_strength[j] = min(
                        1.0, (close[j] - delivery_open) / (delivery_range + 1e-10)
                    )
                    break

    return pd.DataFrame(
        {
            "bullish_cisd": bullish_cisd,
            "bearish_cisd": bearish_cisd,
            "cisd_strength": cisd_strength,
        },
        index=df.index,
    )


def find_recent_cisd(
    cisd_df: pd.DataFrame,
    current_bar: int,
    lookback: int = 30,
) -> Optional[Tuple[str, float, int]]:
    """
    Find the most recent CISD signal within lookback bars.

    Args:
        cisd_df: DataFrame from detect_cisd_vectorized.
        current_bar: Current bar index.
        lookback: Bars to look back.

    Returns:
        Tuple of (direction, strength, bar_index) or None.
    """
    start = max(0, current_bar - lookback)
    window = cisd_df.iloc[start : current_bar + 1]

    bullish_rows = window[window["bullish_cisd"]]
    bearish_rows = window[window["bearish_cisd"]]

    best_idx = -1
    best_dir = None
    best_strength = 0.0

    if not bullish_rows.empty:
        last_idx = bullish_rows.index[-1]
        idx_pos = window.index.get_loc(last_idx)
        strength = window["cisd_strength"].iloc[idx_pos]
        if idx_pos > best_idx:
            best_idx = idx_pos
            best_dir = "bullish"
            best_strength = float(strength)

    if not bearish_rows.empty:
        last_idx = bearish_rows.index[-1]
        idx_pos = window.index.get_loc(last_idx)
        strength = window["cisd_strength"].iloc[idx_pos]
        if idx_pos > best_idx:
            best_idx = idx_pos
            best_dir = "bearish"
            best_strength = float(strength)

    if best_dir is None:
        return None

    abs_bar = start + best_idx
    return (best_dir, best_strength, abs_bar)
