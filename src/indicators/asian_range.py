# -*- coding: utf-8 -*-
"""
Asian Session Range Detection

Detects the Asian session range (00:00-08:00 UTC) for liquidity mapping.
Used by SMC/ICT strategies to identify sweep targets.

Reference: SMC-ICT-ML-Hybrid-Backtester Brief
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class AsianRange:
    """
    Asian session range data container.

    Attributes:
        high: Session high price
        low: Session low price
        range_size: High - Low
        bar_count: Number of bars in session
        is_low_vol: True if range < 0.3 * ATR
        session_start: Session start timestamp
        session_end: Session end timestamp
        atr_value: ATR at session end
        date: Trading date
    """

    high: float
    low: float
    range_size: float
    bar_count: int
    is_low_vol: bool
    session_start: pd.Timestamp
    session_end: pd.Timestamp
    atr_value: float
    date: pd.Timestamp

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "high": self.high,
            "low": self.low,
            "range_size": self.range_size,
            "bar_count": self.bar_count,
            "is_low_vol": self.is_low_vol,
            "session_start": str(self.session_start),
            "session_end": str(self.session_end),
            "atr_value": self.atr_value,
            "date": str(self.date),
        }


def _detect_timeframe(df: pd.DataFrame) -> str:
    """
    Detect the approximate timeframe of the DataFrame based on bar frequency.

    Returns:
        'daily', 'hourly', '5min', 'unknown'
    """
    if len(df) < 2:
        return "unknown"

    # Get time difference between first two bars
    time_diff = df.index[1] - df.index[0]
    total_seconds = time_diff.total_seconds()

    if total_seconds >= 86400:  # 1 day or more
        return "daily"
    elif total_seconds >= 3600:  # 1 hour or more
        return "hourly"
    elif total_seconds >= 300:  # 5 minutes or more
        return "5min"
    else:
        return "unknown"


def detect_asian_range(
    df: pd.DataFrame,
    session_start: str = "00:00",
    session_end: str = "08:00",
    atr_period: int = 14,
    low_vol_threshold: float = 0.3,
) -> Optional[AsianRange]:
    """
    Detect Asian session range for liquidity mapping.

    Tracks Asian session high/low for liquidity range detection.
    The Asian session is typically 00:00-08:00 UTC.

    FIXED: Added timeframe compatibility checks.
    - Daily data: Not compatible (returns None with warning)
    - Hourly data: Compatible (expects 8 bars per session)
    - 5-min data: Compatible (expects 96 bars per session)

    Args:
        df: OHLCV DataFrame with UTC datetime index (5-minute bars)
        session_start: UTC time string for session start (default: "00:00")
        session_end: UTC time string for session end (default: "08:00")
        atr_period: ATR period for volatility comparison
        low_vol_threshold: Range/ATR ratio threshold for low volatility flag

    Returns:
        AsianRange object if session data exists, None otherwise

    Raises:
        ValueError: If DataFrame lacks required columns or datetime index

    Example:
        >>> df = pd.read_csv('data.csv', parse_dates=True, index_col=0)
        >>> range_info = detect_asian_range(df, atr_period=14)
        >>> print(f"Asian High: {range_info.high}, Low: {range_info.low}")

    Note:
        - NOT compatible with daily data (returns None)
        - Expects 96 bars for 5-min session, 8 bars for hourly session
        - Uses df.between_time() for session filtering
    """
    # Validate DataFrame
    if df is None or len(df) == 0:
        logger.warning("Empty DataFrame provided")
        return None

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    required_cols = ["High", "Low", "Close"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # FIXED: Check timeframe compatibility
    timeframe = _detect_timeframe(df)
    if timeframe == "daily":
        logger.warning(
            "Asian Range detection is not compatible with daily data. "
            "This indicator requires intraday data (5-min or hourly). "
            "For daily data, consider using previous day high/low instead."
        )
        return None

    # Ensure UTC timezone
    if df.index.tz is None:
        logger.warning("DataFrame index has no timezone, assuming UTC")
        df = df.tz_localize("UTC")
    elif str(df.index.tz) != "UTC":
        df = df.tz_convert("UTC")

    try:
        # Parse session times
        start_time = pd.to_datetime(session_start).time()
        end_time = pd.to_datetime(session_end).time()
    except Exception as e:
        raise ValueError(f"Invalid session time format: {e}")

    # Filter to session hours
    session_df = df.between_time(start_time, end_time)

    if session_df.empty:
        logger.debug("No data in Asian session range")
        return None

    # Get the date (use the first bar's date)
    first_bar = session_df.iloc[0]
    session_date = session_df.index[0].date()

    # Calculate session high/low
    asia_high = session_df["High"].max()
    asia_low = session_df["Low"].min()
    range_size = asia_high - asia_low

    # Count bars
    bar_count = len(session_df)

    # FIXED: Calculate expected bars based on detected timeframe
    if timeframe == "hourly":
        expected_bars = 8  # 8 hours * 1 bar/hour
    elif timeframe == "5min":
        expected_bars = 96  # 8 hours * 60 minutes / 5 minutes
    else:
        expected_bars = None  # Unknown timeframe

    # Log warning if bar count doesn't match expected
    if expected_bars and bar_count != expected_bars:
        logger.warning(
            f"Asian session has {bar_count} bars, expected {expected_bars} "
            f"({timeframe} data). Date: {session_date}"
        )

    # Calculate ATR for low volatility detection
    try:
        from .technical import atr

        atr_series = atr(df, period=atr_period)

        # Get ATR value at the end of session
        session_end_idx = session_df.index[-1]
        atr_value = (
            atr_series.loc[session_end_idx]
            if session_end_idx in atr_series.index
            else atr_series.iloc[-1]
        )

        if pd.isna(atr_value):
            atr_value = range_size  # Fallback to range if ATR is NaN
            logger.warning("ATR value is NaN, using range size as fallback")
    except Exception as e:
        logger.warning(f"Could not calculate ATR: {e}, using range size as fallback")
        atr_value = range_size

    # Check for low volatility session
    is_low_vol = range_size < (low_vol_threshold * atr_value)

    if is_low_vol:
        logger.debug(
            f"Low volatility Asian session: range={range_size:.4f}, "
            f"ATR={atr_value:.4f}, ratio={range_size / atr_value:.2f}"
        )

    return AsianRange(
        high=asia_high,
        low=asia_low,
        range_size=range_size,
        bar_count=bar_count,
        is_low_vol=is_low_vol,
        session_start=session_df.index[0],
        session_end=session_df.index[-1],
        atr_value=atr_value,
        date=pd.Timestamp(session_date),
    )


def get_asian_range_for_day(
    df: pd.DataFrame, date: pd.Timestamp, session_start: str = "00:00", session_end: str = "08:00"
) -> Optional[pd.DataFrame]:
    """
    Extract Asian session data for a specific day.

    Args:
        df: OHLCV DataFrame with UTC datetime index
        date: Target date
        session_start: UTC time string for session start
        session_end: UTC time string for session end

    Returns:
        DataFrame filtered to session bars, or None if no data
    """
    # Ensure UTC timezone
    if df.index.tz is None:  # type: ignore[union-attr]
        df = df.tz_localize("UTC")
    elif str(df.index.tz) != "UTC":  # type: ignore[union-attr]
        df = df.tz_convert("UTC")

    # Convert date to pandas Timestamp if needed
    if not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(date)

    # Filter to the specific date
    day_start = date.normalize()
    day_end = day_start + pd.Timedelta(days=1)

    day_df = df.loc[day_start:day_end]

    if day_df.empty:
        return None

    # Filter to session hours
    start_time = pd.to_datetime(session_start).time()
    end_time = pd.to_datetime(session_end).time()

    session_df = day_df.between_time(start_time, end_time)

    return session_df if not session_df.empty else None


def calculate_session_statistics(
    df: pd.DataFrame, session_start: str = "00:00", session_end: str = "08:00", atr_period: int = 14
) -> pd.DataFrame:
    """
    Calculate daily session statistics over historical data.

    Args:
        df: OHLCV DataFrame with UTC datetime index
        session_start: UTC time string for session start
        session_end: UTC time string for session end
        atr_period: ATR period for calculations

    Returns:
        DataFrame with daily session statistics:
        - date: Trading date
        - session_high: Session high price
        - session_low: Session low price
        - session_range: High - Low
        - bar_count: Number of bars in session
        - atr: ATR value at session end
    """
    # Ensure UTC timezone
    if df.index.tz is None:  # type: ignore[union-attr]
        df = df.tz_localize("UTC")
    elif str(df.index.tz) != "UTC":  # type: ignore[union-attr]
        df = df.tz_convert("UTC")

    # Calculate ATR
    from .technical import atr

    atr_series = atr(df, period=atr_period)

    # Get unique dates
    dates = df.index.normalize().unique()  # type: ignore[union-attr]

    results = []

    for date in dates:
        session_df = get_asian_range_for_day(df, date, session_start, session_end)

        if session_df is None or session_df.empty:
            continue

        session_end_time = session_df.index[-1]
        atr_value = (
            atr_series.loc[session_end_time] if session_end_time in atr_series.index else np.nan
        )

        results.append(
            {
                "date": date,
                "session_high": session_df["High"].max(),
                "session_low": session_df["Low"].min(),
                "session_range": session_df["High"].max() - session_df["Low"].min(),
                "bar_count": len(session_df),
                "atr": atr_value,
            }
        )

    return pd.DataFrame(results).set_index("date")


def get_session_bounds(
    df: pd.DataFrame, session_start: str = "00:00", session_end: str = "08:00"
) -> Tuple[float, float]:
    """
    Get the high and low of the most recent completed session.

    Args:
        df: OHLCV DataFrame with UTC datetime index
        session_start: UTC time string for session start
        session_end: UTC time string for session end

    Returns:
        Tuple of (session_high, session_low)

    Raises:
        ValueError: If no session data available
    """
    range_info = detect_asian_range(df, session_start, session_end)

    if range_info is None:
        raise ValueError("No Asian session data available")

    return range_info.high, range_info.low
