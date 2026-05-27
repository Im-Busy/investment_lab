"""P28-14: Futures inventory alternative data signals.

Provides warehouse stock levels and inventory data for commodities as
supply/demand signals. Monitors stock changes at major exchanges (LME,
CME, ICE) as leading indicators for commodity price moves.

Source: stock-sdk API — futures inventory data feeds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_COMMODITY_EXCHANGES: dict[str, str] = {
    "GC=F": "COMEX",
    "SI=F": "COMEX",
    "HG=F": "COMEX",
    "CL=F": "NYMEX",
    "NG=F": "NYMEX",
    "RB=F": "NYMEX",
    "HO=F": "NYMEX",
    "ZC=F": "CBOT",
    "ZW=F": "CBOT",
    "ZS=F": "CBOT",
    "ZM=F": "CBOT",
    "KC=F": "ICE",
    "CC=F": "ICE",
    "SB=F": "ICE",
    "CT=F": "ICE",
    "OJ=F": "ICE",
    "PL=F": "NYMEX",
    "PA=F": "NYMEX",
}

_STOCK_CHANGE_THRESHOLDS: dict[str, float] = {
    "GC=F": 0.03,
    "SI=F": 0.05,
    "CL=F": 0.04,
    "NG=F": 0.05,
    "ZC=F": 0.04,
    "ZS=F": 0.04,
    "ZW=F": 0.04,
    "KC=F": 0.03,
    "CC=F": 0.05,
}


@dataclass
class InventoryRecord:
    """Single inventory observation."""

    date: datetime
    symbol: str
    exchange: str
    warehouse_stocks: float
    """Total exchange-monitored stock in standard units."""
    registered_stocks: float
    """Registered (deliverable) vs eligible stock."""
    eligible_stocks: float
    """Eligible but not registered stock."""
    cancelled_warrants: float
    """Warrants cancelled (earmarked for delivery)."""


@dataclass
class InventorySignal:
    """Derived trading signal from inventory data."""

    symbol: str
    date: datetime
    stock_level: float
    stock_change_1w: float
    stock_change_4w: float
    stock_zscore: float
    signal: str
    """'tight' (low stock), 'ample' (high stock), or 'neutral'."""
    signal_strength: float
    """0-1 signal confidence."""


class FuturesInventory:
    """Track futures inventory data and generate supply-side signals.

    Uses publicly available exchange data (weekly CFTC COT reports are
    the primary source for position data; inventory data requires exchange
    APIs or data providers).

    This module provides the framework for when data becomes available.
    For now, the signal computation logic is functional and can be used
    with synthetic or manually-collected inventory data.
    """

    def __init__(self) -> None:
        self._records: dict[str, list[InventoryRecord]] = {}

    def load_records(
        self,
        symbol: str,
        records: list[InventoryRecord],
    ) -> None:
        """Load inventory records for a symbol.

        Args:
            symbol: Yahoo Finance futures symbol (e.g., 'GC=F').
            records: Time-sorted list of InventoryRecord.
        """
        self._records[symbol] = sorted(records, key=lambda r: r.date)

    def get_signals(
        self,
        symbol: str,
        as_of: Optional[datetime] = None,
        lookback_weeks: int = 52,
    ) -> Optional[InventorySignal]:
        """Compute the current inventory-based signal.

        Args:
            symbol: Futures symbol.
            as_of: Reference date (default: latest available).
            lookback_weeks: Weeks of history for z-score calculation.

        Returns:
            InventorySignal if sufficient data, None otherwise.
        """
        recs = self._records.get(symbol, [])
        if len(recs) < 4:
            return None

        as_of = as_of or recs[-1].date
        recent = [r for r in recs if r.date <= as_of]
        if not recent:
            return None

        latest = recent[-1]
        stock_series = np.array(
            [r.warehouse_stocks for r in recent[-lookback_weeks:]], dtype=np.float64
        )

        change_1w = 0.0
        if len(recent) >= 2:
            prev = recent[-2]
            if prev.warehouse_stocks > 0:
                change_1w = (
                    latest.warehouse_stocks - prev.warehouse_stocks
                ) / prev.warehouse_stocks

        change_4w = 0.0
        if len(recent) >= 5:
            prev4 = recent[-5]
            if prev4.warehouse_stocks > 0:
                change_4w = (
                    latest.warehouse_stocks - prev4.warehouse_stocks
                ) / prev4.warehouse_stocks

        zscore = 0.0
        if len(stock_series) >= 8 and stock_series.std() > 0:
            zscore = (float(latest.warehouse_stocks) - float(stock_series.mean())) / float(
                stock_series.std()
            )

        threshold = _STOCK_CHANGE_THRESHOLDS.get(symbol, 0.04)
        signal = "neutral"
        strength = 0.5

        if zscore < -1.5:
            signal = "tight"
            strength = min(1.0, abs(zscore) / 3.0)
        elif zscore > 1.5:
            signal = "ample"
            strength = min(1.0, zscore / 3.0)

        return InventorySignal(
            symbol=symbol,
            date=latest.date,
            stock_level=float(latest.warehouse_stocks),
            stock_change_1w=change_1w,
            stock_change_4w=change_4w,
            stock_zscore=zscore,
            signal=signal,
            signal_strength=strength,
        )

    def get_all_signals(
        self,
        as_of: Optional[datetime] = None,
    ) -> dict[str, Optional[InventorySignal]]:
        """Get signals for all loaded symbols."""
        return {sym: self.get_signals(sym, as_of) for sym in self._records}

    def to_dataframe(self, as_of: Optional[datetime] = None) -> pd.DataFrame:
        """Return all current signals as a DataFrame."""
        signals = self.get_all_signals(as_of)
        rows = []
        for sym, sig in signals.items():
            if sig is None:
                continue
            rows.append(
                {
                    "symbol": sig.symbol,
                    "date": sig.date,
                    "stock_level": sig.stock_level,
                    "change_1w_pct": round(sig.stock_change_1w * 100, 2),
                    "change_4w_pct": round(sig.stock_change_4w * 100, 2),
                    "zscore": round(sig.stock_zscore, 2),
                    "signal": sig.signal,
                    "strength": round(sig.signal_strength, 2),
                    "exchange": _COMMODITY_EXCHANGES.get(sym, "unknown"),
                }
            )
        return pd.DataFrame(rows)

    def supply_demand_bias(self, symbol: str) -> float:
        """Map inventory signal to a supply-demand bias for trading.

        Returns -1 to +1 where:
          +1 = tight supply (bullish for price)
          -1 = ample supply (bearish for price)
           0 = neutral / insufficient data
        """
        sig = self.get_signals(symbol)
        if sig is None:
            return 0.0
        if sig.signal == "tight":
            return sig.signal_strength
        if sig.signal == "ample":
            return -sig.signal_strength
        return 0.0


def estimate_bias_from_known_data(
    symbol: str,
    stock_change_4w: float,
) -> float:
    """Estimate supply-demand bias from known stock change without full history.

    Fallback when historical inventory data is unavailable.
    For commodities, declining stock = tightening = bullish.
    """
    if stock_change_4w < -0.05:
        return -stock_change_4w * 5.0
    if stock_change_4w > 0.05:
        return -stock_change_4w * 3.0
    return 0.0
