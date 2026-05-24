# -*- coding: utf-8 -*-
"""
Market Structure Shift (MSS) Detection

Identifies when market structure changes direction through
3-bar pivot breaks. Core SMC/ICT concept for entry confirmation.

Reference: SMC-ICT-ML-Hybrid-Backtester Brief
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class MSSInfo:
    """
    Market Structure Shift information.

    Attributes:
        detected: Whether MSS was detected
        direction: MSS direction - 'bullish' or 'bearish'
        pivot_high: The pivot high that was broken (for bearish MSS)
        pivot_low: The pivot low that was broken (for bullish MSS)
        pivot_high_bar: Bar index of the pivot high
        pivot_low_bar: Bar index of the pivot low
        break_bar: Bar index where the break occurred
        break_price: Price at which structure broke
        confirmation_bar: Bar index where MSS was confirmed
        is_valid: Whether MSS meets all validation criteria
        swing_high: Most recent swing high price
        swing_low: Most recent swing low price
    """

    detected: bool
    direction: Optional[Literal["bullish", "bearish"]]
    pivot_high: Optional[float]
    pivot_low: Optional[float]
    pivot_high_bar: Optional[int]
    pivot_low_bar: Optional[int]
    break_bar: Optional[int]
    break_price: Optional[float]
    confirmation_bar: Optional[int]
    is_valid: bool
    swing_high: Optional[float]
    swing_low: Optional[float]
    displacement_confirmed: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "detected": self.detected,
            "direction": self.direction,
            "pivot_high": self.pivot_high,
            "pivot_low": self.pivot_low,
            "pivot_high_bar": self.pivot_high_bar,
            "pivot_low_bar": self.pivot_low_bar,
            "break_bar": self.break_bar,
            "break_price": self.break_price,
            "confirmation_bar": self.confirmation_bar,
            "is_valid": self.is_valid,
            "swing_high": self.swing_high,
            "swing_low": self.swing_low,
            "displacement_confirmed": self.displacement_confirmed,
        }


@dataclass
class PivotPoint:
    """
    Pivot point information.

    Attributes:
        bar_index: Bar index of the pivot
        price: Price at pivot
        pivot_type: 'high' or 'low'
        confirmed: Whether the pivot is confirmed (lookback bars passed)
    """

    bar_index: int
    price: float
    pivot_type: Literal["high", "low"]
    confirmed: bool = True


def find_pivot_high(df: pd.DataFrame, bar_index: int, lookback: int = 3) -> Optional[PivotPoint]:
    """
    Find the most recent confirmed pivot high.

    A pivot high is a bar whose high is higher than the highs
    of the lookback bars on each side.

    Args:
        df: OHLCV DataFrame
        bar_index: Current bar index (search backward from here)
        lookback: Bars on each side for pivot confirmation

    Returns:
        PivotPoint if found, None otherwise
    """
    # Need enough bars for confirmation
    if bar_index < lookback * 2 + 1:
        return None

    # Search backward for a pivot high
    for i in range(bar_index - lookback, lookback, -1):
        is_pivot = True
        pivot_high = df.iloc[i]["High"]

        # Check bars before
        for j in range(1, lookback + 1):
            if df.iloc[i - j]["High"] >= pivot_high:
                is_pivot = False
                break

        if not is_pivot:
            continue

        # Check bars after
        for j in range(1, lookback + 1):
            if df.iloc[i + j]["High"] >= pivot_high:
                is_pivot = False
                break

        if is_pivot:
            return PivotPoint(bar_index=i, price=pivot_high, pivot_type="high", confirmed=True)

    return None


def find_pivot_low(df: pd.DataFrame, bar_index: int, lookback: int = 3) -> Optional[PivotPoint]:
    """
    Find the most recent confirmed pivot low.

    A pivot low is a bar whose low is lower than the lows
    of the lookback bars on each side.

    Args:
        df: OHLCV DataFrame
        bar_index: Current bar index (search backward from here)
        lookback: Bars on each side for pivot confirmation

    Returns:
        PivotPoint if found, None otherwise
    """
    # Need enough bars for confirmation
    if bar_index < lookback * 2 + 1:
        return None

    # Search backward for a pivot low
    for i in range(bar_index - lookback, lookback, -1):
        is_pivot = True
        pivot_low = df.iloc[i]["Low"]

        # Check bars before
        for j in range(1, lookback + 1):
            if df.iloc[i - j]["Low"] <= pivot_low:
                is_pivot = False
                break

        if not is_pivot:
            continue

        # Check bars after
        for j in range(1, lookback + 1):
            if df.iloc[i + j]["Low"] <= pivot_low:
                is_pivot = False
                break

        if is_pivot:
            return PivotPoint(bar_index=i, price=pivot_low, pivot_type="low", confirmed=True)

    return None


def detect_mss(
    df: pd.DataFrame, lookback: int = 3, require_close: bool = True, end_bar: Optional[int] = None
) -> MSSInfo:
    """
    Detect Market Structure Shift using 3-bar pivot breaks.

    A bullish MSS occurs when price breaks above a pivot high.
    A bearish MSS occurs when price breaks below a pivot low.

    MSS Detection Logic:
    1. Find the most recent pivot high and pivot low
    2. Check if price has broken either pivot
    3. Confirm the break with a close beyond the pivot (if require_close=True)

    Args:
        df: OHLCV DataFrame
        lookback: Number of bars for pivot detection (default: 3)
        require_close: If True, require close beyond pivot; if False, wick is OK
        end_bar: Bar index to check from (default: last bar)

    Returns:
        MSSInfo with MSS details

    Example:
        >>> mss = detect_mss(df, lookback=3, require_close=True)
        >>> if mss.detected:
        ...     print(f"MSS direction: {mss.direction}, break at {mss.break_price}")
    """
    # Default no-MSS result
    no_mss = MSSInfo(
        detected=False,
        direction=None,
        pivot_high=None,
        pivot_low=None,
        pivot_high_bar=None,
        pivot_low_bar=None,
        break_bar=None,
        break_price=None,
        confirmation_bar=None,
        is_valid=False,
        swing_high=None,
        swing_low=None,
        displacement_confirmed=False,
    )

    if df is None or len(df) < lookback * 2 + 3:
        return no_mss

    # Set end bar
    if end_bar is None:
        end_bar = len(df) - 1

    # Find recent pivot points
    pivot_high = find_pivot_high(df, end_bar, lookback)
    pivot_low = find_pivot_low(df, end_bar, lookback)

    if pivot_high is None and pivot_low is None:
        return no_mss

    # Get current swing levels
    recent_highs = df.iloc[end_bar - lookback : end_bar + 1]["High"]
    recent_lows = df.iloc[end_bar - lookback : end_bar + 1]["Low"]
    swing_high = recent_highs.max()
    swing_low = recent_lows.min()

    # Check for bullish MSS (break above pivot high)
    if pivot_high is not None:
        for i in range(pivot_high.bar_index + 1, end_bar + 1):
            bar = df.iloc[i]

            if require_close:
                # Check if close breaks above pivot high
                if bar["Close"] > pivot_high.price:
                    logger.debug(
                        f"Bullish MSS detected: close {bar['Close']:.4f} > "
                        f"pivot high {pivot_high.price:.4f} at bar {i}"
                    )
                    return MSSInfo(
                        detected=True,
                        direction="bullish",
                        pivot_high=pivot_high.price,
                        pivot_low=pivot_low.price if pivot_low else None,
                        pivot_high_bar=pivot_high.bar_index,
                        pivot_low_bar=pivot_low.bar_index if pivot_low else None,
                        break_bar=i,
                        break_price=bar["Close"],
                        confirmation_bar=i,
                        is_valid=True,
                        swing_high=swing_high,
                        swing_low=swing_low,
                        displacement_confirmed=False,
                    )
            else:
                # Check if high breaks above pivot high
                if bar["High"] > pivot_high.price:
                    logger.debug(
                        f"Bullish MSS detected: high {bar['High']:.4f} > "
                        f"pivot high {pivot_high.price:.4f} at bar {i}"
                    )
                    return MSSInfo(
                        detected=True,
                        direction="bullish",
                        pivot_high=pivot_high.price,
                        pivot_low=pivot_low.price if pivot_low else None,
                        pivot_high_bar=pivot_high.bar_index,
                        pivot_low_bar=pivot_low.bar_index if pivot_low else None,
                        break_bar=i,
                        break_price=bar["High"],
                        confirmation_bar=i,
                        is_valid=True,
                        swing_high=swing_high,
                        swing_low=swing_low,
                        displacement_confirmed=False,
                    )

    # Check for bearish MSS (break below pivot low)
    if pivot_low is not None:
        for i in range(pivot_low.bar_index + 1, end_bar + 1):
            bar = df.iloc[i]

            if require_close:
                # Check if close breaks below pivot low
                if bar["Close"] < pivot_low.price:
                    logger.debug(
                        f"Bearish MSS detected: close {bar['Close']:.4f} < "
                        f"pivot low {pivot_low.price:.4f} at bar {i}"
                    )
                    return MSSInfo(
                        detected=True,
                        direction="bearish",
                        pivot_high=pivot_high.price if pivot_high else None,
                        pivot_low=pivot_low.price,
                        pivot_high_bar=pivot_high.bar_index if pivot_high else None,
                        pivot_low_bar=pivot_low.bar_index,
                        break_bar=i,
                        break_price=bar["Close"],
                        confirmation_bar=i,
                        is_valid=True,
                        swing_high=swing_high,
                        swing_low=swing_low,
                        displacement_confirmed=False,
                    )
            else:
                # Check if low breaks below pivot low
                if bar["Low"] < pivot_low.price:
                    logger.debug(
                        f"Bearish MSS detected: low {bar['Low']:.4f} < "
                        f"pivot low {pivot_low.price:.4f} at bar {i}"
                    )
                    return MSSInfo(
                        detected=True,
                        direction="bearish",
                        pivot_high=pivot_high.price if pivot_high else None,
                        pivot_low=pivot_low.price,
                        pivot_high_bar=pivot_high.bar_index if pivot_high else None,
                        pivot_low_bar=pivot_low.bar_index,
                        break_bar=i,
                        break_price=bar["Low"],
                        confirmation_bar=i,
                        is_valid=True,
                        swing_high=swing_high,
                        swing_low=swing_low,
                        displacement_confirmed=False,
                    )

    # No MSS detected, return current state
    return MSSInfo(
        detected=False,
        direction=None,
        pivot_high=pivot_high.price if pivot_high else None,
        pivot_low=pivot_low.price if pivot_low else None,
        pivot_high_bar=pivot_high.bar_index if pivot_high else None,
        pivot_low_bar=pivot_low.bar_index if pivot_low else None,
        break_bar=None,
        break_price=None,
        confirmation_bar=None,
        is_valid=False,
        swing_high=swing_high,
        swing_low=swing_low,
        displacement_confirmed=False,
    )


def validate_mss_with_htf(
    df: pd.DataFrame, mss_info: MSSInfo, htf_bias: Literal["bullish", "bearish", "neutral"]
) -> bool:
    """
    Validate MSS alignment with Higher Timeframe bias.

    Args:
        df: OHLCV DataFrame
        mss_info: MSS information
        htf_bias: Higher timeframe bias direction

    Returns:
        True if MSS aligns with HTF bias
    """
    if not mss_info.detected:
        return False

    if htf_bias == "neutral":
        return True  # Accept any MSS if HTF bias is neutral

    return mss_info.direction == htf_bias


def get_structure_levels(
    df: pd.DataFrame, lookback: int = 3, num_pivots: int = 3
) -> Tuple[List[float], List[float]]:
    """
    Get recent structure levels (pivot highs and lows).

    Args:
        df: OHLCV DataFrame
        lookback: Bars on each side for pivot detection
        num_pivots: Number of pivots to return

    Returns:
        Tuple of (pivot_highs, pivot_lows) lists
    """
    pivot_highs: List[float] = []
    pivot_lows: List[float] = []

    # Start from the end and work backward
    end_bar = len(df) - 1
    search_bar = end_bar

    while len(pivot_highs) < num_pivots or len(pivot_lows) < num_pivots:
        if search_bar < lookback * 2 + 1:
            break

        # Find pivot high
        if len(pivot_highs) < num_pivots:
            ph = find_pivot_high(df, search_bar, lookback)
            if ph is not None and ph.price not in pivot_highs:
                pivot_highs.append(ph.price)
                search_bar = min(search_bar, ph.bar_index - 1)

        # Find pivot low
        if len(pivot_lows) < num_pivots:
            pl = find_pivot_low(df, search_bar, lookback)
            if pl is not None and pl.price not in pivot_lows:
                pivot_lows.append(pl.price)
                search_bar = min(search_bar, pl.bar_index - 1)

        search_bar -= 1

    return pivot_highs[:num_pivots], pivot_lows[:num_pivots]


def detect_break_of_structure(
    df: pd.DataFrame, lookback: int = 3, end_bar: Optional[int] = None
) -> MSSInfo:
    """
    Detect Break of Structure (BOS) - similar to MSS but for continuation.

    BOS occurs when price breaks a recent pivot in the direction of the trend.

    Args:
        df: OHLCV DataFrame
        lookback: Bars for pivot detection
        end_bar: Bar index to check from

    Returns:
        MSSInfo with BOS details
    """
    # This is an alias for detect_mss with different naming
    # In SMC terminology, BOS and MSS are similar concepts
    return detect_mss(df, lookback, require_close=True, end_bar=end_bar)


def calculate_structure_quality(df: pd.DataFrame, mss_info: MSSInfo, lookback: int = 20) -> float:
    """
    Calculate the quality score of an MSS signal.

    Higher scores indicate cleaner structure breaks.

    Factors:
    - Distance from pivot to break
    - Number of bars between pivot and break
    - Volume at break (if available)

    Args:
        df: OHLCV DataFrame
        mss_info: MSS information
        lookback: Bars for context calculation

    Returns:
        Quality score between 0.0 and 1.0
    """
    if not mss_info.detected:
        return 0.0

    score = 0.5  # Base score

    # Factor 1: Number of bars between pivot and break (cleaner = fewer bars)
    if mss_info.pivot_high_bar is not None and mss_info.break_bar is not None:
        bars_between = abs(mss_info.break_bar - mss_info.pivot_high_bar)
        if bars_between <= 5:
            score += 0.2  # Quick break
        elif bars_between <= 10:
            score += 0.1
        elif bars_between > 20:
            score -= 0.1  # Stale pivot

    # Factor 2: Break magnitude
    if mss_info.pivot_high is not None and mss_info.break_price is not None:
        if mss_info.direction == "bullish":
            break_magnitude = mss_info.break_price - mss_info.pivot_high
        elif mss_info.pivot_low is not None:
            break_magnitude = mss_info.pivot_low - mss_info.break_price
        else:
            break_magnitude = 0

        # Normalize by recent range
        if mss_info.swing_high is not None and mss_info.swing_low is not None:
            recent_range = mss_info.swing_high - mss_info.swing_low
            if recent_range > 0:
                normalized = break_magnitude / recent_range
                if normalized > 0.1:  # Break is at least 10% of range
                    score += 0.2
                elif normalized > 0.05:
                    score += 0.1

    # Factor 3: Volume confirmation (if available)
    if "Volume" in df.columns and mss_info.break_bar is not None:
        try:
            avg_volume = df.iloc[mss_info.break_bar - lookback : mss_info.break_bar][
                "Volume"
            ].mean()
            break_volume = df.iloc[mss_info.break_bar]["Volume"]
            if break_volume > avg_volume * 1.5:
                score += 0.1
        except Exception:
            pass

    return min(1.0, max(0.0, score))


@dataclass
class BOSInfo:
    """Break of Structure info (trend continuation, same direction as existing trend)."""

    detected: bool
    direction: Optional[str]  # 'bullish' or 'bearish'
    swing_price: Optional[float]  # level broken
    break_bar: Optional[int]
    displacement_ratio: float  # move size / ATR
    is_valid: bool

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "direction": self.direction,
            "swing_price": self.swing_price,
            "break_bar": self.break_bar,
            "displacement_ratio": self.displacement_ratio,
            "is_valid": self.is_valid,
        }


def detect_bos(
    df: pd.DataFrame,
    atr: "np.ndarray | pd.Series",
    lookback: int = 10,
    min_displacement_mult: float = 0.5,
    end_bar: Optional[int] = None,
) -> BOSInfo:
    """Detect BOS (trend CONTINUATION, same direction as existing trend).

    Different from MSS which detects reversal.

    - Bullish BOS: close breaks above most recent confirmed swing high with full candle body (not wick).
      Previous swing high must have been a HH relative to prior HH.
    - Bearish BOS: close breaks below most recent confirmed swing low with full candle body.
      Previous swing low must have been a LL relative to prior LL.
    - Displacement check: move size must exceed min_displacement_mult * ATR.
    """
    import numpy as np

    no_bos = BOSInfo(
        detected=False,
        direction=None,
        swing_price=None,
        break_bar=None,
        displacement_ratio=0.0,
        is_valid=False,
    )

    n = len(df)
    if end_bar is None:
        end_bar = n - 1
    if end_bar < lookback * 2:
        return no_bos

    close = (
        df["Close"].to_numpy(dtype=np.float64)
        if hasattr(df["Close"], "to_numpy")
        else np.array(df["Close"])
    )
    high = (
        df["High"].to_numpy(dtype=np.float64)
        if hasattr(df["High"], "to_numpy")
        else np.array(df["High"])
    )
    low = (
        df["Low"].to_numpy(dtype=np.float64)
        if hasattr(df["Low"], "to_numpy")
        else np.array(df["Low"])
    )

    if isinstance(atr, pd.Series):
        atr_vals = atr.to_numpy(dtype=np.float64)
    else:
        atr_vals = atr
    if len(atr_vals) != n:
        atr_vals = np.full(n, atr_vals[-1] if len(atr_vals) > 0 else 0.01)

    window = high[end_bar - lookback : end_bar + 1]
    window_low = low[end_bar - lookback : end_bar + 1]

    swing_high_idx = end_bar - lookback + int(np.argmax(window[-lookback // 2 :]))
    swing_low_idx = end_bar - lookback + int(np.argmin(window[-lookback // 2 :]))
    swing_high = float(window[-lookback // 2 :][swing_high_idx - (end_bar - lookback)])
    swing_low = float(window[-lookback // 2 :][swing_low_idx - (end_bar - lookback)])

    atr_val = float(atr_vals[end_bar])
    if atr_val <= 0:
        atr_val = float(close[end_bar]) * 0.01

    for i in range(max(end_bar - 5, 0), end_bar + 1):
        if close[i] > swing_high and (close[i] - swing_high) > min_displacement_mult * atr_val:
            hh_before = np.max(high[max(0, i - lookback * 2) : i - lookback])
            prev_hh = max(high[max(0, i - lookback) : i - 1]) if i - 1 >= 0 else swing_high
            if swing_high > hh_before and prev_hh > hh_before:
                ratio = float((close[i] - swing_high) / atr_val)
                return BOSInfo(
                    detected=True,
                    direction="bullish",
                    swing_price=swing_high,
                    break_bar=i,
                    displacement_ratio=ratio,
                    is_valid=True,
                )

        if close[i] < swing_low and (swing_low - close[i]) > min_displacement_mult * atr_val:
            ll_before = np.min(low[max(0, i - lookback * 2) : i - lookback])
            prev_ll = min(low[max(0, i - lookback) : i - 1]) if i - 1 >= 0 else swing_low
            if swing_low < ll_before and prev_ll < ll_before:
                ratio = float((swing_low - close[i]) / atr_val)
                return BOSInfo(
                    detected=True,
                    direction="bearish",
                    swing_price=swing_low,
                    break_bar=i,
                    displacement_ratio=ratio,
                    is_valid=True,
                )

    return no_bos


@dataclass
class FakeCHOCHInfo:
    """Fake CHOCH detection result."""

    detected: bool
    direction: Optional[str]
    break_level: Optional[float]
    break_bar: Optional[int]
    sweep_only: bool  # True if wick break but no body close
    htf_contradicts: bool  # True if HTF bias contradicts
    no_displacement: bool  # True if no displacement follows

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "direction": self.direction,
            "break_level": self.break_level,
            "break_bar": self.break_bar,
            "sweep_only": self.sweep_only,
            "htf_contradicts": self.htf_contradicts,
            "no_displacement": self.no_displacement,
        }


def detect_fake_choch(
    df: pd.DataFrame,
    atr: "np.ndarray | pd.Series",
    htf_bias: "np.ndarray | None" = None,
    lookback: int = 3,
    end_bar: Optional[int] = None,
) -> FakeCHOCHInfo:
    """Detect fake CHOCH (false reversal signals - liquidity sweeps that look like CHOCH but aren't).

    Rules:
    - Price breaks a swing level with a WICK (not body close) -> sweep, not structure break
    - HTF context contradicts: if daily bullish and 15m prints bearish CHOCH -> likely fake
    - No displacement follows -> likely fake
    """
    import numpy as np

    no_fake = FakeCHOCHInfo(
        detected=False,
        direction=None,
        break_level=None,
        break_bar=None,
        sweep_only=False,
        htf_contradicts=False,
        no_displacement=False,
    )

    n = len(df)
    if end_bar is None:
        end_bar = n - 1
    if end_bar < lookback * 2 + 1:
        return no_fake

    close = (
        df["Close"].to_numpy(dtype=np.float64)
        if hasattr(df["Close"], "to_numpy")
        else np.array(df["Close"])
    )
    high = (
        df["High"].to_numpy(dtype=np.float64)
        if hasattr(df["High"], "to_numpy")
        else np.array(df["High"])
    )
    low = (
        df["Low"].to_numpy(dtype=np.float64)
        if hasattr(df["Low"], "to_numpy")
        else np.array(df["Low"])
    )

    if isinstance(atr, pd.Series):
        atr_vals = atr.to_numpy(dtype=np.float64)
    else:
        atr_vals = atr

    atr_val = float(atr_vals[end_bar]) if end_bar < len(atr_vals) else 0.01
    if atr_val <= 0:
        atr_val = float(close[end_bar]) * 0.01

    recent_pivot = find_pivot_high(df, end_bar, lookback)
    if recent_pivot is not None:
        for i in range(recent_pivot.bar_index + 1, end_bar + 1):
            if high[i] > recent_pivot.price and close[i] <= recent_pivot.price:
                sweep_only = True
                displacement = abs(close[end_bar] - close[end_bar - 1])
                no_disp = displacement < 0.3 * atr_val
                htf_contradicts = False
                if htf_bias is not None and end_bar < len(htf_bias) and htf_bias[end_bar] > 0:
                    htf_contradicts = True
                return FakeCHOCHInfo(
                    detected=True,
                    direction="bearish",
                    break_level=recent_pivot.price,
                    break_bar=i,
                    sweep_only=sweep_only,
                    htf_contradicts=htf_contradicts,
                    no_displacement=no_disp,
                )

    recent_pivot_low = find_pivot_low(df, end_bar, lookback)
    if recent_pivot_low is not None:
        for i in range(recent_pivot_low.bar_index + 1, end_bar + 1):
            if low[i] < recent_pivot_low.price and close[i] >= recent_pivot_low.price:
                sweep_only = True
                displacement = abs(close[end_bar] - close[end_bar - 1])
                no_disp = displacement < 0.3 * atr_val
                htf_contradicts = False
                if htf_bias is not None and end_bar < len(htf_bias) and htf_bias[end_bar] < 0:
                    htf_contradicts = True
                return FakeCHOCHInfo(
                    detected=True,
                    direction="bullish",
                    break_level=recent_pivot_low.price,
                    break_bar=i,
                    sweep_only=sweep_only,
                    htf_contradicts=htf_contradicts,
                    no_displacement=no_disp,
                )

    return no_fake
