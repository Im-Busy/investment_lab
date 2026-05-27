"""P28-1: Survivorship-bias-free historical universe.

Retains delisted tickers per date so backtests use the actual investable
universe at each point in time. Without this, backtests are biased by only
testing tickers that survived to the present day.

Source: FinanceDatabase FAQ — survivorship bias in backtesting.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE_PATH = Path("data/universe")
META_FILE = "delisted_meta.json"


@dataclass
class DelistedRecord:
    """Metadata for a delisted security."""

    ticker: str
    name: str
    exchange: str
    sector: str | None = None
    last_date: dt.date | None = None
    delist_reason: str | None = None


@dataclass
class HistoricalUniverse:
    """Queryable historical universe with delisted ticker retention.

    On any date D, universe(D) returns all tickers that were listed on that
    date — including those that later delisted.
    """

    listings: pd.DataFrame
    delisted: dict[str, DelistedRecord] = field(default_factory=dict)
    _active_cache: dict[str, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def from_delisted_file(
        cls,
        path: Path,
        active_tickers: Optional[list[str]] = None,
    ) -> HistoricalUniverse:
        """Load delisted metadata from a JSON file and combine with active list.

        Expected JSON format:
        [
            {
                "ticker": "LEHMQ",
                "name": "Lehman Brothers",
                "exchange": "OTC",
                "last_date": "2008-09-15",
                "delist_reason": "bankruptcy"
            },
            ...
        ]

        Args:
            path: Path to delisted metadata JSON.
            active_tickers: Currently listed tickers (if None, no active filter).

        Returns:
            HistoricalUniverse with delisted records loaded.
        """
        records: dict[str, DelistedRecord] = {}
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            for item in raw:
                ticker = item["ticker"].upper()
                last_date = None
                if item.get("last_date"):
                    last_date = dt.date.fromisoformat(item["last_date"])
                records[ticker] = DelistedRecord(
                    ticker=ticker,
                    name=item.get("name", ticker),
                    exchange=item.get("exchange", ""),
                    sector=item.get("sector"),
                    last_date=last_date,
                    delist_reason=item.get("delist_reason"),
                )

        df_data: list[dict] = []
        for ticker, rec in records.items():
            if rec.last_date:
                df_data.append(
                    {
                        "ticker": ticker,
                        "first_date": "1900-01-01",
                        "last_date": rec.last_date.isoformat(),
                        "status": "delisted",
                    }
                )

        if active_tickers:
            today = dt.date.today().isoformat()
            for t in active_tickers:
                df_data.append(
                    {
                        "ticker": t.upper(),
                        "first_date": "1900-01-01",
                        "last_date": today,
                        "status": "active",
                    }
                )

        listings = (
            pd.DataFrame(df_data)
            if df_data
            else pd.DataFrame(columns=["ticker", "first_date", "last_date", "status"])
        )
        return cls(listings=listings, delisted=records)

    def universe_on(self, date: dt.date | str) -> list[str]:
        """Return all tickers listed on the given date.

        Args:
            date: Query date (ISO string or date object).

        Returns:
            Sorted list of ticker symbols.
        """
        if isinstance(date, str):
            date = dt.date.fromisoformat(date)
        date_str = date.isoformat()
        cache_key = date_str
        if cache_key in self._active_cache:
            return sorted(self._active_cache[cache_key])

        mask = (self.listings["first_date"] <= date_str) & (self.listings["last_date"] >= date_str)
        tickers = sorted(self.listings.loc[mask, "ticker"].unique().tolist())
        self._active_cache[cache_key] = frozenset(tickers)
        return tickers

    def universe_slice(
        self,
        start: dt.date | str,
        end: dt.date | str,
    ) -> list[str]:
        """Return tickers listed for the entire date range.

        Args:
            start: Start date (inclusive).
            end: End date (inclusive).

        Returns:
            Sorted list of tickers continuously listed through the range.
        """
        if isinstance(start, str):
            start = dt.date.fromisoformat(start)
        if isinstance(end, str):
            end = dt.date.fromisoformat(end)
        start_str = start.isoformat()
        end_str = end.isoformat()
        mask = (self.listings["first_date"] <= start_str) & (self.listings["last_date"] >= end_str)
        return sorted(self.listings.loc[mask, "ticker"].unique().tolist())

    def get_delisted(self, before_date: dt.date | str | None = None) -> dict[str, DelistedRecord]:
        """Return delisted records, optionally filtered by last_date.

        Args:
            before_date: Only return tickers delisted before this date.

        Returns:
            Dict of ticker → DelistedRecord.
        """
        if before_date is None:
            return dict(self.delisted)
        if isinstance(before_date, str):
            before_date = dt.date.fromisoformat(before_date)
        return {k: v for k, v in self.delisted.items() if v.last_date and v.last_date < before_date}

    @property
    def total_tracked(self) -> int:
        """Total number of tracked securities (active + delisted)."""
        return len(self.listings["ticker"].unique())

    @property
    def total_delisted(self) -> int:
        """Number of delisted securities."""
        return len(self.delisted)

    @property
    def total_active(self) -> int:
        """Number of currently active securities."""
        return self.total_tracked - self.total_delisted

    def summary(self) -> str:
        return (
            f"HistoricalUniverse: {self.total_tracked} tickers "
            f"({self.total_active} active, {self.total_delisted} delisted)"
        )

    def date_range(self) -> tuple[str, str]:
        """Overall date range covered by the universe."""
        return (
            str(self.listings["first_date"].min()),
            str(self.listings["last_date"].max()),
        )

    def tickers_by_sector(self) -> dict[str, list[str]]:
        """Group delisted tickers by GICS sector."""
        sectors: dict[str, list[str]] = {}
        for ticker, rec in self.delisted.items():
            sector = rec.sector or "Unknown"
            sectors.setdefault(sector, []).append(ticker)
        return sectors
