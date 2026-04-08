"""
Helper Functions

Utility functions for data handling, formatting, and calculations.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(
    filepath: Union[str, Path],
    date_column: Optional[str] = None,
    parse_dates: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Load OHLCV data from file.
    
    Args:
        filepath: Path to data file (CSV, Parquet, or Excel)
        date_column: Name of date/datetime column
        parse_dates: Whether to parse dates
        **kwargs: Additional arguments for pd.read_* functions
        
    Returns:
        DataFrame with OHLCV data
    """
    path: Path = Path(filepath) if isinstance(filepath, str) else filepath
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    # Determine file type and load
    suffix = path.suffix.lower()
    
    if suffix == '.csv':
        df = pd.read_csv(filepath, **kwargs)
    elif suffix == '.parquet':
        df = pd.read_parquet(filepath, **kwargs)
    elif suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    # Parse dates
    if parse_dates and date_column:
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
    elif parse_dates:
        # Try to find date column automatically
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            df.set_index(date_cols[0], inplace=True)
    
    # Standardize column names
    df.columns = [col.capitalize() for col in df.columns]
    
    # Validate OHLCV columns
    required = ['Open', 'High', 'Low', 'Close']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Add Volume if not present
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    
    logger.info(f"Loaded {len(df)} rows from {path}")
    
    return df


def save_results(
    results: Dict[str, Any],
    filepath: Union[str, Path],
    format: str = 'json'
) -> None:
    """
    Save backtest results to file.
    
    Args:
        results: Dictionary of results to save
        filepath: Output file path
        format: Output format ('json', 'csv', 'parquet')
    """
    path: Path = Path(filepath) if isinstance(filepath, str) else filepath
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        # Convert any non-serializable types
        serializable = _make_serializable(results)
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
    elif format == 'csv':
        if 'trades' in results:
            pd.DataFrame(results['trades']).to_csv(path, index=False)
        else:
            pd.DataFrame(results).to_csv(path)
    elif format == 'parquet':
        if 'trades' in results:
            pd.DataFrame(results['trades']).to_parquet(path, index=False)
        else:
            pd.DataFrame(results).to_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Results saved to {path}")


def _make_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    else:
        return obj


def format_signal(signal: Any) -> str:
    """
    Format a trading signal for display.
    
    Args:
        signal: TradeSignal or AggregatedSignal object
        
    Returns:
        Formatted string representation
    """
    lines = []
    lines.append(f"Pattern: {signal.pattern_name}")
    lines.append(f"Direction: {signal.direction.value}")
    lines.append(f"Entry: ${signal.entry_price:.2f}")
    lines.append(f"Stop Loss: ${signal.stop_loss:.2f}")
    lines.append(f"Take Profit 1: ${signal.take_profit_1:.2f}")
    
    if signal.take_profit_2:
        lines.append(f"Take Profit 2: ${signal.take_profit_2:.2f}")
    if signal.take_profit_3:
        lines.append(f"Take Profit 3: ${signal.take_profit_3:.2f}")
    
    lines.append(f"Confidence: {signal.confidence:.2%}")
    
    if hasattr(signal, 'timestamp') and signal.timestamp:
        lines.append(f"Timestamp: {signal.timestamp}")
    
    return "\n".join(lines)


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_loss: float,
    risk_pct: float = 0.02,
    method: str = 'fixed_fractional'
) -> float:
    """
    Calculate position size based on risk parameters.
    
    Args:
        equity: Current equity
        entry_price: Entry price
        stop_loss: Stop loss price
        risk_pct: Risk percentage (default 2%)
        method: Position sizing method
        
    Returns:
        Position size in shares/contracts
    """
    risk_distance = abs(entry_price - stop_loss)
    
    if risk_distance == 0:
        return 0
    
    if method == 'fixed_fractional':
        risk_amount = equity * risk_pct
        size = risk_amount / risk_distance
    elif method == 'fixed_amount':
        size = equity / entry_price
    else:
        risk_amount = equity * risk_pct
        size = risk_amount / risk_distance
    
    return size


def validate_ohlcv_data(df: pd.DataFrame) -> tuple:
    """
    Validate OHLCV data integrity.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid: bool, issues: list)
    """
    issues = []
    
    # Check required columns
    required = ['Open', 'High', 'Low', 'Close']
    missing = [col for col in required if col not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")
    
    # Check for NaN values
    for col in required:
        if col in df.columns and df[col].isna().any():
            na_count = df[col].isna().sum()
            issues.append(f"Column {col} has {na_count} NaN values")
    
    # Check OHLC relationships
    if all(col in df.columns for col in required):
        # High should be >= Open, Close, Low
        invalid_high = df[df['High'] < df[['Open', 'Close', 'Low']].max(axis=1)]
        if len(invalid_high) > 0:
            issues.append(f"{len(invalid_high)} rows where High is not the highest price")
        
        # Low should be <= Open, Close, High
        invalid_low = df[df['Low'] > df[['Open', 'Close', 'High']].min(axis=1)]
        if len(invalid_low) > 0:
            issues.append(f"{len(invalid_low)} rows where Low is not the lowest price")
    
    # Check for negative prices
    for col in required:
        if col in df.columns and (df[col] <= 0).any():
            issues.append(f"Column {col} has non-positive values")
    
    # Check index
    if not isinstance(df.index, pd.DatetimeIndex):
        issues.append("Index is not a DatetimeIndex")
    elif df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        issues.append(f"Index has {dup_count} duplicate timestamps")
    
    is_valid = len(issues) == 0
    
    return is_valid, issues


def resample_data(
    df: pd.DataFrame,
    timeframe: str = '1D'
) -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe.
    
    Args:
        df: DataFrame with OHLCV data
        timeframe: Target timeframe ('1H', '4H', '1D', '1W', etc.)
        
    Returns:
        Resampled DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex for resampling")
    
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    resampled = df.resample(timeframe).agg(agg_dict)  # type: ignore[arg-type]
    resampled.dropna(inplace=True)
    
    return resampled


def add_indicators(
    df: pd.DataFrame,
    indicators: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Add technical indicators to DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        indicators: List of indicator names to add
            Options: 'sma', 'ema', 'atr', 'rsi', 'adx'
        
    Returns:
        DataFrame with added indicator columns
    """
    from ..indicators.technical import sma, ema, atr, rsi, adx
    
    df = df.copy()
    
    if indicators is None:
        indicators = ['sma', 'atr', 'rsi']
    
    for indicator in indicators:  # type: ignore[union-attr]
        if indicator.lower() == 'sma':
            df['SMA_20'] = sma(df['Close'], 20)
            df['SMA_50'] = sma(df['Close'], 50)
        elif indicator.lower() == 'ema':
            df['EMA_20'] = ema(df['Close'], 20)
        elif indicator.lower() == 'atr':
            df['ATR_14'] = atr(df, 14)
        elif indicator.lower() == 'rsi':
            df['RSI_14'] = rsi(df['Close'], 14)
        elif indicator.lower() == 'adx':
            df['ADX_14'] = adx(df, 14)
    
    return df


def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary information about the data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Dictionary with data summary
    """
    info = {
        'rows': len(df),
        'columns': list(df.columns),
        'start_date': str(df.index.min()) if isinstance(df.index, pd.DatetimeIndex) else None,
        'end_date': str(df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else None,
        'days': None,
        'missing_values': df.isna().sum().to_dict(),
        'statistics': {
            'close_mean': df['Close'].mean(),
            'close_std': df['Close'].std(),
            'close_min': df['Close'].min(),
            'close_max': df['Close'].max(),
        }
    }
    
    if isinstance(df.index, pd.DatetimeIndex):
        info['days'] = (df.index.max() - df.index.min()).days
    
    return info
