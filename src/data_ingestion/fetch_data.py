# -*- coding: utf-8 -*-
"""
Data Ingestion Module

Fetches historical OHLCV data from Yahoo Finance for multiple instruments
and timeframes. Supports both daily and intraday data for strategy backtesting.

Note: Yahoo Finance API limitations as of 2024:
- Daily data: Up to 10+ years available
- Intraday 5m data: Only last 60 days available
- Intraday 1h data: Up to 730 days (2 years) available
"""

import yfinance as yf
import pandas as pd
import os
import time
from loguru import logger
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Literal

# Configure Logger
logger.add("logs/data_ingestion.log", rotation="10 MB", retention="30 days", level="INFO")


# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Multi-Pattern Strategy: Daily bars (10 years for most, 5 years for crypto)
MULTI_PATTERN_DAILY = {
    # US Equity ETFs
    "SPY": {"start": "2015-01-01", "end": "2025-01-01", "description": "S&P 500 ETF"},
    "QQQ": {"start": "2015-01-01", "end": "2025-01-01", "description": "Nasdaq 100 ETF"},
    
    # Commodities & Bonds
    "GLD": {"start": "2015-01-01", "end": "2025-01-01", "description": "SPDR Gold Shares ETF"},
    "IAU": {"start": "2015-01-01", "end": "2025-01-01", "description": "iShares Gold Trust ETF"},
    "TLT": {"start": "2015-01-01", "end": "2025-01-01", "description": "20+ Year Treasury ETF"},
    
    # Forex (Yahoo Finance format)
    "EURUSD=X": {"start": "2015-01-01", "end": "2025-01-01", "description": "EUR/USD"},
    # Note: XAU/USD spot price not available on Yahoo Finance
    # Use GLD/IAU for gold exposure, or GC=F futures for intraday
    
    # Crypto (shorter history due to market age)
    "BTC-USD": {"start": "2019-01-01", "end": "2025-01-01", "description": "Bitcoin"},
}

# SMC/ICT Strategy: Hourly bars (2 years - better for SMC than 5m limited data)
# Using 1h bars which have 2 years of history vs 5m which only has 60 days
SMC_HOURLY = {
    # Crypto - 24/7 market, ideal for Asian session logic
    "BTC-USD": {"period": "730d", "interval": "1h", "description": "Bitcoin Hourly"},
    
    # Futures - Extended/globex hours (23 hours)
    "GC=F": {"period": "730d", "interval": "1h", "description": "Gold Futures Hourly"},
    "NQ=F": {"period": "730d", "interval": "1h", "description": "Nasdaq Futures Hourly"},
    
    # Forex - 24-hour market (weekdays)
    "EURUSD=X": {"period": "730d", "interval": "1h", "description": "EUR/USD Hourly"},
    "GBPJPY=X": {"period": "730d", "interval": "1h", "description": "GBP/JPY Hourly"},
    # Note: XAU/USD spot not available; use GC=F gold futures for intraday gold
}

# SMC/ICT Strategy: 5-minute bars (last 60 days only - for recent testing)
SMC_INTRADAY_5M = {
    "BTC-USD": {"period": "60d", "interval": "5m", "description": "Bitcoin 5-min"},
    "GC=F": {"period": "60d", "interval": "5m", "description": "Gold Futures 5-min"},
    "EURUSD=X": {"period": "60d", "interval": "5m", "description": "EUR/USD 5-min"},
    # Note: XAU/USD spot not available; use GC=F gold futures for intraday gold
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names from yfinance output.
    
    yfinance can return MultiIndex columns or different capitalizations.
    This function normalizes to: Date/Datetime, Open, High, Low, Close, Volume
    """
    # Handle MultiIndex columns (yfinance sometimes returns tuples)
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten MultiIndex by taking the first level
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # Rename columns to standardized names
    column_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if 'date' in col_lower or 'datetime' in col_lower:
            column_map[col] = 'Date'
        elif col_lower == 'open':
            column_map[col] = 'Open'
        elif col_lower == 'high':
            column_map[col] = 'High'
        elif col_lower == 'low':
            column_map[col] = 'Low'
        elif col_lower == 'close':
            column_map[col] = 'Close'
        elif col_lower == 'volume':
            column_map[col] = 'Volume'
        else:
            column_map[col] = col  # Keep as is
    
    df = df.rename(columns=column_map)
    return df


# =============================================================================
# DATA FETCHING FUNCTIONS
# =============================================================================

def fetch_daily_data(
    ticker: str,
    start: str,
    end: str,
    output_dir: str = "data/raw"
) -> bool:
    """
    Fetches daily historical data and saves to CSV.
    
    Args:
        ticker: Yahoo Finance ticker symbol
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        output_dir: Output directory for CSV files
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Fetching daily data for {ticker} ({start} to {end})...")
        
        # Download data
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        if df.empty:
            logger.warning(f"No data retrieved for {ticker}")
            return False
        
        # Standardize column names
        df = _standardize_columns(df)
        
        # Reset index to make 'Date' a column
        df = df.reset_index()
        
        # Ensure we have the right date column name
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV with standardized naming
        safe_ticker = ticker.replace("=", "_").replace("-", "_")
        filename = f"{safe_ticker}_daily.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        # Log summary
        start_date = df['Date'].min()
        end_date = df['Date'].max()
        row_count = len(df)
        logger.info(f"✅ Saved {row_count} daily bars to {filepath}")
        logger.info(f"   Date range: {start_date} to {end_date}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed for {ticker}: {e}")
        return False


def fetch_intraday_data(
    ticker: str,
    period: str = "730d",
    interval: str = "1h",
    output_dir: str = "data/raw"
) -> bool:
    """
    Fetches intraday historical data and saves to CSV.
    
    Note: Yahoo Finance limits:
    - 5m data: Only last 60 days
    - 1h data: Up to 730 days (2 years)
    
    Args:
        ticker: Yahoo Finance ticker symbol
        period: Period to fetch (e.g., "730d" for 2 years, "60d" for 5m data)
        interval: Bar interval (e.g., "5m", "15m", "1h")
        output_dir: Output directory for CSV files
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Fetching {interval} data for {ticker} (last {period})...")
        
        # Download data using period (required for intraday)
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df.empty:
            logger.warning(f"No data retrieved for {ticker}")
            return False
        
        # Standardize column names
        df = _standardize_columns(df)
        
        # Reset index to make 'Datetime' a column
        df = df.reset_index()
        
        # Ensure we have the right datetime column name
        if 'Date' in df.columns and interval in ['5m', '15m', '1h', '1h', '4h']:
            df = df.rename(columns={'Date': 'Datetime'})
        
        # Ensure UTC timezone for SMC strategy
        if 'Datetime' in df.columns:
            if hasattr(df['Datetime'].iloc[0], 'tzinfo'):
                if df['Datetime'].iloc[0].tzinfo is None:
                    df['Datetime'] = df['Datetime'].dt.tz_localize('UTC')
                else:
                    df['Datetime'] = df['Datetime'].dt.tz_convert('UTC')
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV with standardized naming
        safe_ticker = ticker.replace("=", "_").replace("-", "_")
        filename = f"{safe_ticker}_{interval}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        # Log date range
        start_date = df['Datetime'].min() if 'Datetime' in df.columns else df['Date'].min()
        end_date = df['Datetime'].max() if 'Datetime' in df.columns else df['Date'].max()
        row_count = len(df)
        
        logger.info(f"✅ Saved {row_count} {interval} bars to {filepath}")
        logger.info(f"   Date range: {start_date} to {end_date}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed for {ticker}: {e}")
        return False


def fetch_all_daily(
    instruments: Optional[Dict] = None,
    output_dir: str = "data/raw"
) -> Dict[str, bool]:
    """
    Fetch daily data for all instruments in the configuration.
    
    Args:
        instruments: Dictionary of instruments to fetch (default: MULTI_PATTERN_DAILY)
        output_dir: Output directory for CSV files
    
    Returns:
        Dictionary with ticker as key and success status as value
    """
    instruments = instruments or MULTI_PATTERN_DAILY
    results = {}
    success_count = 0
    
    logger.info("=" * 60)
    logger.info("Fetching Daily Data for Multi-Pattern Strategy")
    logger.info("=" * 60)
    
    for ticker, config in instruments.items():
        success = fetch_daily_data(
            ticker=ticker,
            start=config["start"],
            end=config["end"],
            output_dir=output_dir
        )
        results[ticker] = success
        if success:
            success_count += 1
        
        # Rate limiting
        time.sleep(1)
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Daily Data Complete: {success_count}/{len(instruments)} successful")
    logger.info(f"{'=' * 60}\n")
    
    return results


def fetch_all_hourly(
    instruments: Optional[Dict] = None,
    output_dir: str = "data/raw"
) -> Dict[str, bool]:
    """
    Fetch hourly data for all instruments in the configuration.
    
    Args:
        instruments: Dictionary of instruments to fetch (default: SMC_HOURLY)
        output_dir: Output directory for CSV files
    
    Returns:
        Dictionary with ticker as key and success status as value
    """
    instruments = instruments or SMC_HOURLY
    results = {}
    success_count = 0
    
    logger.info("=" * 60)
    logger.info("Fetching Hourly Data for SMC/ICT Strategy (2 years)")
    logger.info("=" * 60)
    
    for ticker, config in instruments.items():
        success = fetch_intraday_data(
            ticker=ticker,
            period=config["period"],
            interval=config["interval"],
            output_dir=output_dir
        )
        results[ticker] = success
        if success:
            success_count += 1
        
        # Rate limiting (longer for intraday)
        time.sleep(2)
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Hourly Data Complete: {success_count}/{len(instruments)} successful")
    logger.info(f"{'=' * 60}\n")
    
    return results


def fetch_all_5min(
    instruments: Optional[Dict] = None,
    output_dir: str = "data/raw"
) -> Dict[str, bool]:
    """
    Fetch 5-minute data for all instruments in the configuration.
    
    Note: Yahoo Finance only provides last 60 days of 5m data.
    
    Args:
        instruments: Dictionary of instruments to fetch (default: SMC_INTRADAY_5M)
        output_dir: Output directory for CSV files
    
    Returns:
        Dictionary with ticker as key and success status as value
    """
    instruments = instruments or SMC_INTRADAY_5M
    results = {}
    success_count = 0
    
    logger.info("=" * 60)
    logger.info("Fetching 5-Minute Data for SMC/ICT Strategy (60 days only)")
    logger.info("=" * 60)
    
    for ticker, config in instruments.items():
        success = fetch_intraday_data(
            ticker=ticker,
            period=config["period"],
            interval=config["interval"],
            output_dir=output_dir
        )
        results[ticker] = success
        if success:
            success_count += 1
        
        # Rate limiting
        time.sleep(2)
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"5-Minute Data Complete: {success_count}/{len(instruments)} successful")
    logger.info(f"{'=' * 60}\n")
    
    return results


def fetch_all_data(output_dir: str = "data/raw") -> Dict[str, Dict[str, bool]]:
    """
    Fetch all data for both strategies.
    
    Args:
        output_dir: Output directory for CSV files
    
    Returns:
        Dictionary with 'daily', 'hourly', and '5min' results
    """
    logger.info("\n" + "=" * 60)
    logger.info("DATA INGESTION PIPELINE - ALL DATA")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60 + "\n")
    
    results = {
        'daily': fetch_all_daily(output_dir=output_dir),
        'hourly': fetch_all_hourly(output_dir=output_dir),
        '5min': fetch_all_5min(output_dir=output_dir)
    }
    
    # Summary
    daily_success = sum(1 for v in results['daily'].values() if v)
    hourly_success = sum(1 for v in results['hourly'].values() if v)
    five_min_success = sum(1 for v in results['5min'].values() if v)
    total_success = daily_success + hourly_success + five_min_success
    total_count = len(results['daily']) + len(results['hourly']) + len(results['5min'])
    
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Daily data:     {daily_success}/{len(results['daily'])} successful")
    logger.info(f"Hourly data:    {hourly_success}/{len(results['hourly'])} successful")
    logger.info(f"5-min data:     {five_min_success}/{len(results['5min'])} successful")
    logger.info(f"Total:          {total_success}/{total_count} successful")
    logger.info("=" * 60 + "\n")
    
    return results


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def fetch_and_save(ticker: str, start: str, end: str, output_dir: str = "data/raw") -> bool:
    """
    Legacy function for backward compatibility.
    
    Args:
        ticker: Yahoo Finance ticker symbol
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        output_dir: Output directory for CSV files
    
    Returns:
        True if successful, False otherwise
    """
    return fetch_daily_data(ticker, start, end, output_dir)


def main():
    """Main entry point for data ingestion."""
    fetch_all_data()


if __name__ == "__main__":
    main()
