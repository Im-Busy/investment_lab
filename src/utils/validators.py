"""
Validators

Validation functions for data, signals, and pattern parameters.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import fields


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str] = None,
    min_rows: int = 50,
    require_datetime_index: bool = True
) -> tuple:
    """
    Validate a DataFrame for pattern detection.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        require_datetime_index: Whether to require DatetimeIndex
        
    Returns:
        Tuple of (is_valid: bool, errors: list)
    """
    errors = []
    
    # Check if DataFrame is empty
    if df is None or len(df) == 0:
        errors.append("DataFrame is empty or None")
        return False, errors
    
    # Check minimum rows
    if len(df) < min_rows:
        errors.append(f"DataFrame has only {len(df)} rows, minimum required is {min_rows}")
    
    # Check required columns
    if required_columns is None:
        required_columns = ['Open', 'High', 'Low', 'Close']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Check DatetimeIndex
    if require_datetime_index:
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("DataFrame index is not a DatetimeIndex")
        else:
            # Check for duplicate indices
            if df.index.duplicated().any():
                errors.append(f"DataFrame has {df.index.duplicated().sum()} duplicate timestamps")
    
    # Check for NaN values in required columns
    for col in required_columns:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                errors.append(f"Column '{col}' has {nan_count} NaN values")
    
    # Check for valid OHLC relationships
    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        # High should be >= Open, Close, Low
        invalid_high = df['High'] < df[['Open', 'Close', 'Low']].max(axis=1)
        if invalid_high.any():
            errors.append(f"{invalid_high.sum()} rows where High < max(Open, Close, Low)")
        
        # Low should be <= Open, Close, High
        invalid_low = df['Low'] > df[['Open', 'Close', 'High']].min(axis=1)
        if invalid_low.any():
            errors.append(f"{invalid_low.sum()} rows where Low > min(Open, Close, High)")
    
    # Check for non-positive prices
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            non_positive = (df[col] <= 0).sum()
            if non_positive > 0:
                errors.append(f"Column '{col}' has {non_positive} non-positive values")
    
    is_valid = len(errors) == 0
    
    return is_valid, errors


def validate_signal(signal: Any) -> tuple:
    """
    Validate a trading signal.
    
    Args:
        signal: TradeSignal or AggregatedSignal object
        
    Returns:
        Tuple of (is_valid: bool, errors: list)
    """
    errors = []
    
    # Check required attributes
    required_attrs = ['pattern_name', 'direction', 'entry_price', 'stop_loss', 'take_profit_1']
    
    for attr in required_attrs:
        if not hasattr(signal, attr):
            errors.append(f"Missing required attribute: {attr}")
    
    if errors:
        return False, errors
    
    # Validate entry price
    if signal.entry_price <= 0:
        errors.append(f"Entry price must be positive: {signal.entry_price}")
    
    # Validate stop loss
    if signal.stop_loss <= 0:
        errors.append(f"Stop loss must be positive: {signal.stop_loss}")
    
    # Validate stop loss relative to entry
    from ..patterns.base import SignalDirection
    
    if hasattr(signal, 'direction'):
        if signal.direction == SignalDirection.LONG:
            if signal.stop_loss >= signal.entry_price:
                errors.append(f"Long position: Stop loss ({signal.stop_loss}) must be below entry ({signal.entry_price})")
        elif signal.direction == SignalDirection.SHORT:
            if signal.stop_loss <= signal.entry_price:
                errors.append(f"Short position: Stop loss ({signal.stop_loss}) must be above entry ({signal.entry_price})")
    
    # Validate take profit levels
    if signal.take_profit_1:
        if signal.take_profit_1 <= 0:
            errors.append(f"Take profit 1 must be positive: {signal.take_profit_1}")
    
    if signal.take_profit_2 and signal.take_profit_2 <= 0:
        errors.append(f"Take profit 2 must be positive: {signal.take_profit_2}")
    
    if signal.take_profit_3 and signal.take_profit_3 <= 0:
        errors.append(f"Take profit 3 must be positive: {signal.take_profit_3}")
    
    # Validate confidence
    if hasattr(signal, 'confidence'):
        if not (0 <= signal.confidence <= 1):
            errors.append(f"Confidence must be between 0 and 1: {signal.confidence}")
    
    is_valid = len(errors) == 0
    
    return is_valid, errors


def validate_pattern_params(
    pattern_name: str,
    params: Dict[str, Any]
) -> tuple:
    """
    Validate pattern parameters.
    
    Args:
        pattern_name: Name of the pattern
        params: Dictionary of parameters
        
    Returns:
        Tuple of (is_valid: bool, errors: list)
    """
    errors = []
    
    # Common parameter validations
    if 'lookback' in params:
        if not isinstance(params['lookback'], int) or params['lookback'] < 1:
            errors.append(f"lookback must be a positive integer: {params['lookback']}")
    
    if 'min_bars' in params:
        if not isinstance(params['min_bars'], int) or params['min_bars'] < 1:
            errors.append(f"min_bars must be a positive integer: {params['min_bars']}")
    
    if 'entry_offset' in params:
        if not isinstance(params['entry_offset'], (int, float)) or params['entry_offset'] < 0:
            errors.append(f"entry_offset must be a non-negative number: {params['entry_offset']}")
    
    if 'stop_offset' in params:
        if not isinstance(params['stop_offset'], (int, float)) or params['stop_offset'] < 0:
            errors.append(f"stop_offset must be a non-negative number: {params['stop_offset']}")
    
    if 'tolerance' in params:
        if not isinstance(params['tolerance'], (int, float)) or not (0 <= params['tolerance'] <= 1):
            errors.append(f"tolerance must be between 0 and 1: {params['tolerance']}")
    
    if 'volume_filter' in params:
        if not isinstance(params['volume_filter'], bool):
            errors.append(f"volume_filter must be a boolean: {params['volume_filter']}")
    
    # Pattern-specific validations
    if pattern_name == 'DoubleTop' or pattern_name == 'DoubleBottom':
        if 'peak_tolerance' in params:
            if not isinstance(params['peak_tolerance'], (int, float)) or not (0 <= params['peak_tolerance'] <= 1):
                errors.append(f"peak_tolerance must be between 0 and 1: {params['peak_tolerance']}")
    
    if pattern_name == 'TripleTop':
        if 'peak_tolerance' in params:
            if not isinstance(params['peak_tolerance'], (int, float)) or not (0 <= params['peak_tolerance'] <= 0.1):
                errors.append(f"peak_tolerance for TripleTop should be <= 0.1 (10%): {params['peak_tolerance']}")
    
    if pattern_name == 'DeadCatBounce':
        if 'event_decline_pct' in params:
            if not isinstance(params['event_decline_pct'], (int, float)) or params['event_decline_pct'] < 0:
                errors.append(f"event_decline_pct must be non-negative: {params['event_decline_pct']}")
    
    if pattern_name == 'Gartley':
        if 'xab_tolerance' in params:
            if not isinstance(params['xab_tolerance'], (int, float)) or not (0 <= params['xab_tolerance'] <= 0.1):
                errors.append(f"xab_tolerance must be between 0 and 0.1: {params['xab_tolerance']}")
    
    is_valid = len(errors) == 0
    
    return is_valid, errors


def validate_backtest_config(config: Dict[str, Any]) -> tuple:
    """
    Validate backtest configuration.
    
    Args:
        config: Dictionary with backtest configuration
        
    Returns:
        Tuple of (is_valid: bool, errors: list)
    """
    errors = []
    
    # Equity validation
    if 'initial_equity' in config:
        if not isinstance(config['initial_equity'], (int, float)) or config['initial_equity'] <= 0:
            errors.append(f"initial_equity must be positive: {config['initial_equity']}")
    
    # Commission validation
    if 'commission_per_trade' in config:
        if not isinstance(config['commission_per_trade'], (int, float)) or config['commission_per_trade'] < 0:
            errors.append(f"commission_per_trade must be non-negative: {config['commission_per_trade']}")
    
    if 'commission_pct' in config:
        if not isinstance(config['commission_pct'], (int, float)) or not (0 <= config['commission_pct'] <= 1):
            errors.append(f"commission_pct must be between 0 and 1: {config['commission_pct']}")
    
    # Slippage validation
    if 'slippage_pct' in config:
        if not isinstance(config['slippage_pct'], (int, float)) or not (0 <= config['slippage_pct'] <= 0.1):
            errors.append(f"slippage_pct must be between 0 and 0.1: {config['slippage_pct']}")
    
    # Risk validation
    if 'risk_per_trade' in config:
        if not isinstance(config['risk_per_trade'], (int, float)) or not (0 < config['risk_per_trade'] <= 0.1):
            errors.append(f"risk_per_trade must be between 0 and 0.1: {config['risk_per_trade']}")
    
    # Position limits
    if 'max_open_positions' in config:
        if not isinstance(config['max_open_positions'], int) or config['max_open_positions'] < 1:
            errors.append(f"max_open_positions must be a positive integer: {config['max_open_positions']}")
    
    # Confidence threshold
    if 'min_confidence' in config:
        if not isinstance(config['min_confidence'], (int, float)) or not (0 <= config['min_confidence'] <= 1):
            errors.append(f"min_confidence must be between 0 and 1: {config['min_confidence']}")
    
    is_valid = len(errors) == 0
    
    return is_valid, errors


def validate_date_range(
    start_date: str,
    end_date: str,
    df: Optional[pd.DataFrame] = None
) -> tuple:
    """
    Validate date range for backtest.
    
    Args:
        start_date: Start date string
        end_date: End date string
        df: Optional DataFrame to check against
        
    Returns:
        Tuple of (is_valid: bool, errors: list)
    """
    errors = []
    
    try:
        start = pd.to_datetime(start_date)
    except:
        errors.append(f"Invalid start_date format: {start_date}")
        start = None
    
    try:
        end = pd.to_datetime(end_date)
    except:
        errors.append(f"Invalid end_date format: {end_date}")
        end = None
    
    if start and end:
        if start > end:
            errors.append(f"start_date ({start_date}) must be before end_date ({end_date})")
    
    if df is not None and isinstance(df.index, pd.DatetimeIndex):
        data_start = df.index.min()
        data_end = df.index.max()
        
        if start and start < data_start:
            errors.append(f"start_date ({start_date}) is before data start ({data_start})")
        
        if end and end > data_end:
            errors.append(f"end_date ({end_date}) is after data end ({data_end})")
    
    is_valid = len(errors) == 0
    
    return is_valid, errors
