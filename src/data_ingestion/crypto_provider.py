"""Crypto data provider via CCXT for free OHLCV from Binance.

Usage:
    from src.data_ingestion.crypto_provider import CCXTCryptoProvider

    provider = CCXTCryptoProvider()
    df = provider.fetch_ohlcv("BTC/USDT", "1d", limit=365)
    # df is a pd.DataFrame with Open, High, Low, Close, Volume columns
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CryptoDataProvider(Protocol):
    """Protocol for crypto data providers."""

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1d", limit: int = 365, since: int | None = None
    ) -> pd.DataFrame: ...

    def save_cache(self, symbol: str, df: pd.DataFrame, timeframe: str = "1d") -> None: ...

    def load_cache(self, symbol: str, timeframe: str = "1d") -> pd.DataFrame | None: ...


_CCXT_AVAILABLE = False
try:
    import ccxt  # noqa: F401

    _CCXT_AVAILABLE = True
except ImportError:
    pass


class CCXTCryptoProvider:
    """Free crypto OHLCV data from Binance via CCXT.

    Parameters:
        exchange_id: CCXT exchange ID (default 'binance').
        cache_dir: Directory for CSV cache files (default 'data/raw/').
        atr_multiplier: Multiplier for ATR stops on crypto (default 3.0).
                        Higher than equities (1.5-3.0) due to crypto volatility.
    """

    CRYPTO_ATR_MULTIPLIER: float = 3.0

    def __init__(
        self,
        exchange_id: str = "binance",
        cache_dir: str = "data/raw/",
    ) -> None:
        if not _CCXT_AVAILABLE:
            raise ImportError("ccxt is required for crypto data. Install with: uv add ccxt")

        self.exchange_id = exchange_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._exchange = None

    @property
    def exchange(self):
        if self._exchange is None:
            self._exchange = getattr(ccxt, self.exchange_id)()
        return self._exchange

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        limit: int = 365,
        since: int | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from exchange.

        Args:
            symbol: CCXT symbol format (e.g. 'BTC/USDT', 'ETH/USDT').
            timeframe: Bar interval ('1d', '4h', '1h', etc.).
            limit: Maximum number of bars to fetch.
            since: Timestamp in milliseconds for start time.

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume.
        """
        if since is None:
            since_dt = datetime(2015, 1, 1, tzinfo=timezone.utc)
            since = int(since_dt.timestamp() * 1000)

        bars = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

        df = pd.DataFrame(
            bars,
            columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        df = df.astype(float)

        return df

    def cache_key(self, symbol: str, timeframe: str = "1d") -> str:
        safe = symbol.replace("/", "_")
        return f"{safe}_{timeframe}"

    def cache_path(self, symbol: str, timeframe: str = "1d") -> Path:
        return self.cache_dir / f"{self.cache_key(symbol, timeframe)}.csv"

    def save_cache(self, symbol: str, df: pd.DataFrame, timeframe: str = "1d") -> None:
        path = self.cache_path(symbol, timeframe)
        df.to_csv(path)
        logger.info("Cached %s bars for %s to %s", len(df), symbol, path)

    def load_cache(self, symbol: str, timeframe: str = "1d") -> pd.DataFrame | None:
        path = self.cache_path(symbol, timeframe)
        if not path.exists():
            return None
        df = pd.read_csv(path, parse_dates=True, index_col=0)
        logger.info("Loaded %s bars for %s from cache", len(df), symbol)
        return df

    def get_or_fetch(
        self,
        symbol: str,
        timeframe: str = "1d",
        limit: int = 365,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Get data from cache or fetch from exchange."""
        if not refresh:
            cached = self.load_cache(symbol, timeframe)
            if cached is not None and len(cached) > 0:
                return cached

        df = self.fetch_ohlcv(symbol, timeframe, limit=limit)
        self.save_cache(symbol, df, timeframe)
        return df

    @staticmethod
    def prepare_for_backtest(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for backtesting.py (standardized columns)."""
        if "Date" in df.columns:
            df = df.set_index("Date")
        elif df.index.name != "Date" and not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col not in df.columns:
                if col == "Volume":
                    df[col] = 0
                else:
                    df[col] = df.iloc[:, 0]
        return df


CRYPTO_SYMBOLS: list[str] = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "LINK/USDT",
    "AVAX/USDT",
    "DOT/USDT",
]


def fetch_all_crypto(
    timeframe: str = "1d",
    limit: int = 2000,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch all supported crypto symbols and cache to data/raw/."""
    provider = CCXTCryptoProvider()
    results: dict[str, pd.DataFrame] = {}

    for symbol in CRYPTO_SYMBOLS:
        try:
            df = provider.get_or_fetch(symbol, timeframe, limit=limit, refresh=refresh)
            key = symbol.replace("/", "_")
            results[key] = df
            logger.info("Fetched %s: %d bars", symbol, len(df))
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", symbol, e)

    return results
