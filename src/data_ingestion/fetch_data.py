import yfinance as yf
import pandas as pd
import os
import time
from loguru import logger
from datetime import datetime

# Configure Logger
logger.add("logs/data_ingestion.log", rotation="10 MB", retention="30 days", level="INFO")

def fetch_and_save(ticker: str, start: str, end: str, output_dir: str = "data/raw"):
    """
    Fetches historical data and saves to CSV.
    Includes basic error handling and rate limiting.
    """
    try:
        logger.info(f"Fetching data for {ticker}...")
        
        # Download data
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        if df.empty:
            logger.warning(f"No data retrieved for {ticker}")
            return False
        
        # Reset index to make 'Date' a column (optional but common for storage)
        df.reset_index(inplace=True)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        filename = f"{ticker.replace('.', '_')}_historical.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        logger.info(f"✅ Saved {len(df)} rows to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed for {ticker}: {e}")
        return False

def main():
    logger.info("=== Data Ingestion Pipeline Started ===")
    
    # Sample Basket: HKEX + US Market
    tickers = ["0005.HK", "0700.HK", "SPY", "QQQ"] 
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    
    success_count = 0
    
    for ticker in tickers:
        if fetch_and_save(ticker, start_date, end_date):
            success_count += 1
        # Rate Limiting: Wait 1 second between requests to avoid IP ban
        time.sleep(1) 
        
    logger.info(f"=== Pipeline Complete: {success_count}/{len(tickers)} successful ===")

if __name__ == "__main__":
    main()
