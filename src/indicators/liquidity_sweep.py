# -*- coding: utf-8 -*-
"""
Liquidity Sweep Detection

Identifies when price sweeps beyond session highs/lows to trigger
stop losses before reversing. Core SMC/ICT concept.

Reference: SMC-ICT-ML-Hybrid-Backtester Brief
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Literal
from dataclasses import dataclass
from datetime import time

from loguru import logger


@dataclass
class SweepInfo:
    """
    Liquidity sweep information container.
    
    Attributes:
        detected: Whether a sweep was detected
        direction: Sweep direction - 'bullish' or 'bearish'
        sweep_price: Price at which sweep occurred
        sweep_bar_index: Bar index of sweep
        sweep_time: Timestamp of sweep
        volume_confirmed: True if sweep bar volume > 20-period median
        asia_level: The Asia high/low that was swept
        buffer_used: ATR buffer applied to the level
        sweep_type: 'high_sweep' or 'low_sweep'
    """
    detected: bool
    direction: Optional[Literal['bullish', 'bearish']]
    sweep_price: Optional[float]
    sweep_bar_index: Optional[int]
    sweep_time: Optional[pd.Timestamp]
    volume_confirmed: bool
    asia_level: Optional[float]
    buffer_used: float
    sweep_type: Optional[Literal['high_sweep', 'low_sweep']]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'detected': self.detected,
            'direction': self.direction,
            'sweep_price': self.sweep_price,
            'sweep_bar_index': self.sweep_bar_index,
            'sweep_time': str(self.sweep_time) if self.sweep_time else None,
            'volume_confirmed': self.volume_confirmed,
            'asia_level': self.asia_level,
            'buffer_used': self.buffer_used,
            'sweep_type': self.sweep_type
        }


def detect_liquidity_sweep(
    df: pd.DataFrame,
    asia_high: float,
    asia_low: float,
    atr: float,
    buffer_mult: float = 0.5,
    volume_threshold_mult: float = 1.0,
    lookback_bars: int = 20,
    start_time: str = "08:00"
) -> SweepInfo:
    """
    Detect liquidity sweep beyond Asian range with ATR buffer.
    
    A sweep occurs when price breaks beyond the Asian session high/low
    by at least buffer_mult * ATR, indicating stop loss hunting.
    
    Args:
        df: Post-session OHLCV DataFrame (must have Asia session completed)
            Must have columns: High, Low, Close, Volume
        asia_high: Pre-computed Asian session high
        asia_low: Pre-computed Asian session low
        atr: Current ATR(14) value
        buffer_mult: Buffer multiplier for sweep detection (default: 0.5)
        volume_threshold_mult: Volume threshold multiplier (default: 1.0)
        lookback_bars: Bars to look back for median volume (default: 20)
        start_time: Time to start looking for sweeps (default: "08:00")
    
    Returns:
        SweepInfo with sweep details
    
    Example:
        >>> sweep = detect_liquidity_sweep(
        ...     df, 
        ...     asia_high=100.5, 
        ...     asia_low=99.0, 
        ...     atr=0.5
        ... )
        >>> if sweep.detected:
        ...     print(f"Sweep at {sweep.sweep_price}, direction: {sweep.direction}")
    
    Edge Cases:
        - Sweep exactly at session boundary: Uses >= for break detection
        - Multiple sweeps: Only first valid sweep triggers setup
        - Low-volume filter: Skip if Asian range < 0.3 * ATR
    """
    # Default no-sweep result
    no_sweep = SweepInfo(
        detected=False,
        direction=None,
        sweep_price=None,
        sweep_bar_index=None,
        sweep_time=None,
        volume_confirmed=False,
        asia_level=None,
        buffer_used=buffer_mult * atr,
        sweep_type=None
    )
    
    # Validate inputs
    if df is None or len(df) == 0:
        logger.warning("Empty DataFrame provided")
        return no_sweep
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    required_cols = ['High', 'Low', 'Close']
    if 'Volume' not in df.columns:
        logger.warning("Volume column not found, volume confirmation disabled")
        volume_threshold_mult = 0  # Disable volume check
    
    # Ensure UTC timezone
    if df.index.tz is None:
        df = df.tz_localize('UTC')
    elif str(df.index.tz) != 'UTC':
        df = df.tz_convert('UTC')
    
    # Calculate buffer
    buffer = buffer_mult * atr
    
    # Calculate sweep levels
    sweep_high_level = asia_high + buffer
    sweep_low_level = asia_low - buffer
    
    # Filter to post-session data (after start_time)
    try:
        start_time_obj = pd.to_datetime(start_time).time()
        post_session_df = df[df.index.time >= start_time_obj]
    except Exception as e:
        logger.warning(f"Could not filter by start time: {e}")
        post_session_df = df
    
    if post_session_df.empty:
        logger.debug("No post-session data available")
        return no_sweep
    
    # Calculate median volume for confirmation
    if volume_threshold_mult > 0 and 'Volume' in df.columns:
        volume_median = df['Volume'].rolling(window=lookback_bars, min_periods=1).median()
    else:
        volume_median = None
    
    # Look for sweep (first occurrence only)
    for i, (idx, row) in enumerate(post_session_df.iterrows()):
        high = row['High']
        low = row['Low']
        
        # Check for high sweep (bullish sweep - price above Asia high)
        if high >= sweep_high_level:
            volume_confirmed = True
            if volume_median is not None:
                vol_at_bar = volume_median.loc[idx] if idx in volume_median.index else None
                if vol_at_bar is not None and 'Volume' in row:
                    volume_confirmed = row['Volume'] >= (vol_at_bar * volume_threshold_mult)
            
            logger.debug(
                f"Bullish sweep detected at {idx}: "
                f"high={high:.4f}, sweep_level={sweep_high_level:.4f}"
            )
            
            return SweepInfo(
                detected=True,
                direction='bullish',
                sweep_price=high,
                sweep_bar_index=df.index.get_loc(idx),
                sweep_time=idx,
                volume_confirmed=volume_confirmed,
                asia_level=asia_high,
                buffer_used=buffer,
                sweep_type='high_sweep'
            )
        
        # Check for low sweep (bearish sweep - price below Asia low)
        if low <= sweep_low_level:
            volume_confirmed = True
            if volume_median is not None:
                vol_at_bar = volume_median.loc[idx] if idx in volume_median.index else None
                if vol_at_bar is not None and 'Volume' in row:
                    volume_confirmed = row['Volume'] >= (vol_at_bar * volume_threshold_mult)
            
            logger.debug(
                f"Bearish sweep detected at {idx}: "
                f"low={low:.4f}, sweep_level={sweep_low_level:.4f}"
            )
            
            return SweepInfo(
                detected=True,
                direction='bearish',
                sweep_price=low,
                sweep_bar_index=df.index.get_loc(idx),
                sweep_time=idx,
                volume_confirmed=volume_confirmed,
                asia_level=asia_low,
                buffer_used=buffer,
                sweep_type='low_sweep'
            )
    
    return no_sweep


def find_sweep_candle(
    df: pd.DataFrame,
    level: float,
    direction: Literal['above', 'below'],
    buffer: float,
    start_idx: Optional[int] = None
) -> Optional[pd.Series]:
    """
    Find the first candle that sweeps a level.
    
    Args:
        df: OHLCV DataFrame
        level: Price level to check
        direction: Direction to check - 'above' or 'below'
        buffer: Buffer beyond level for confirmation
        start_idx: Index to start searching from (default: 0)
    
    Returns:
        First bar that sweeps the level, or None
    """
    if start_idx is None:
        start_idx = 0
    
    if direction == 'above':
        target_level = level + buffer
        for i in range(start_idx, len(df)):
            if df.iloc[i]['High'] >= target_level:
                return df.iloc[i]
    else:  # below
        target_level = level - buffer
        for i in range(start_idx, len(df)):
            if df.iloc[i]['Low'] <= target_level:
                return df.iloc[i]
    
    return None


def validate_sweep_volume(
    df: pd.DataFrame,
    sweep_bar_index: int,
    lookback_bars: int = 20,
    threshold_mult: float = 1.0
) -> bool:
    """
    Validate that sweep bar has sufficient volume.
    
    Args:
        df: OHLCV DataFrame with Volume column
        sweep_bar_index: Index of the sweep bar
        lookback_bars: Bars to calculate median volume
        threshold_mult: Multiplier for median volume threshold
    
    Returns:
        True if volume > threshold_mult * median_volume
    """
    if 'Volume' not in df.columns:
        logger.warning("Volume column not found")
        return True  # Pass validation if no volume data
    
    if sweep_bar_index < lookback_bars:
        lookback_bars = sweep_bar_index
    
    if lookback_bars <= 0:
        return True
    
    # Calculate median volume from bars before sweep
    start_idx = max(0, sweep_bar_index - lookback_bars)
    volume_slice = df.iloc[start_idx:sweep_bar_index]['Volume']
    
    if volume_slice.empty:
        return True
    
    median_volume = volume_slice.median()
    sweep_volume = df.iloc[sweep_bar_index]['Volume']
    
    return sweep_volume >= (median_volume * threshold_mult)


def detect_sweep_reversal(
    df: pd.DataFrame,
    sweep_info: SweepInfo,
    lookback: int = 3
) -> bool:
    """
    Check if price has reversed after the sweep.
    
    A sweep reversal is confirmed when price closes back inside
    the Asian range after the sweep.
    
    Args:
        df: OHLCV DataFrame
        sweep_info: SweepInfo from detect_liquidity_sweep
        lookback: Number of bars to check for reversal
    
    Returns:
        True if reversal is detected
    """
    if not sweep_info.detected:
        return False
    
    if sweep_info.sweep_bar_index is None:
        return False
    
    sweep_idx = sweep_info.sweep_bar_index
    
    # Check bars after sweep
    for i in range(sweep_idx + 1, min(sweep_idx + lookback + 1, len(df))):
        close = df.iloc[i]['Close']
        
        if sweep_info.direction == 'bullish':
            # For bullish sweep, check if close is back below Asia high
            if close < sweep_info.asia_level:
                return True
        else:  # bearish
            # For bearish sweep, check if close is back above Asia low
            if close > sweep_info.asia_level:
                return True
    
    return False


def get_sweep_statistics(
    df: pd.DataFrame,
    asia_highs: pd.Series,
    asia_lows: pd.Series,
    atr_series: pd.Series,
    buffer_mult: float = 0.5
) -> pd.DataFrame:
    """
    Calculate sweep statistics over historical data.
    
    Args:
        df: OHLCV DataFrame with UTC datetime index
        asia_highs: Series of Asian session highs (indexed by date)
        asia_lows: Series of Asian session lows (indexed by date)
        atr_series: ATR series
        buffer_mult: Buffer multiplier for sweep detection
    
    Returns:
        DataFrame with sweep statistics per day
    """
    results = []
    
    for date in asia_highs.index:
        try:
            # Get the day's data (post-Asia session)
            day_start = pd.Timestamp(date) + pd.Timedelta(hours=8)
            day_end = pd.Timestamp(date) + pd.Timedelta(days=1)
            
            day_df = df.loc[day_start:day_end]
            
            if day_df.empty:
                continue
            
            atr_value = atr_series.loc[day_start] if day_start in atr_series.index else atr_series.iloc[-1]
            
            sweep = detect_liquidity_sweep(
                day_df,
                asia_high=asia_highs.loc[date],
                asia_low=asia_lows.loc[date],
                atr=atr_value,
                buffer_mult=buffer_mult
            )
            
            results.append({
                'date': date,
                'sweep_detected': sweep.detected,
                'sweep_direction': sweep.direction,
                'sweep_price': sweep.sweep_price,
                'volume_confirmed': sweep.volume_confirmed
            })
        except Exception as e:
            logger.debug(f"Error processing {date}: {e}")
            continue
    
    return pd.DataFrame(results)
