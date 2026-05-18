"""
B8: Treasury Auction Cycle Effects — Predictable Liquidity/Vol Patterns.

U.S. Treasury auctions create predictable calendar-driven effects on
equity market liquidity and volatility. This module models:
- Auction calendar identification (3m, 10y, 30y auctions)
- Pre-auction liquidity withdrawal patterns
- Post-auction volatility compression effects

Key finding: Treasury auctions are the single most predictable calendar
event for equity market microstructure. The auction cycle affects repo
markets, dealer balance sheets, and ultimately equity volumes.

Reference: U.S. Treasury auction calendar (TreasuryDirect).

Usage:
    >>> cal = TreasuryAuctionCalendar()
    >>> cal.is_auction_day("2024-01-10")
    >>> df = cal.auction_cycle_features(price_df, date_col="Date")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Auction schedule patterns (typical monthly) ──

BILL_AUCTIONS = {
    (0, 0): "4-week",
    (0, 1): "8-week",
    (0, 2): "13-week",
    (0, 3): "26-week",
    (0, 4): "52-week",
}
NOTE_AUCTIONS = {(1, 0): "2-year", (1, 1): "3-year", (1, 2): "5-year", (1, 3): "7-year"}
BOND_AUCTIONS = {(2, 0): "10-year", (2, 1): "30-year"}
TIPS_AUCTIONS = {(3, 0): "5-year TIPS", (3, 1): "10-year TIPS", (3, 2): "30-year TIPS"}
FRN_AUCTIONS = {(4, 0): "2-year FRN"}

ALL_AUCTIONS = {**BILL_AUCTIONS, **NOTE_AUCTIONS, **BOND_AUCTIONS, **TIPS_AUCTIONS, **FRN_AUCTIONS}

# Typical auction week patterns within each month
# Format: (week_of_month, day_of_week, auction_type)
# Week 1 = first full week, etc.
AUCTION_SLOTS: List[Tuple[int, int, str]] = [
    (1, 1, "13-week"),  # Monday
    (1, 1, "26-week"),
    (1, 2, "52-week"),  # Tuesday
    (1, 2, "3-year"),
    (1, 3, "10-year"),  # Wednesday
    (1, 4, "30-year"),  # Thursday
    (2, 1, "4-week"),  # Week 2
    (2, 2, "8-week"),
    (2, 2, "2-year"),
    (2, 3, "5-year"),
    (2, 4, "7-year"),
    (3, 1, "4-week"),
    (3, 2, "2-year FRN"),
    (3, 3, "10-year TIPS"),
    (3, 3, "5-year TIPS"),
    (3, 4, "30-year TIPS"),
    (4, 1, "4-week"),
    (4, 2, "8-week"),
]

MID_MONTH_REFUNDING = {2, 5, 8, 11}  # Feb, May, Aug, Nov
QUARTERLY_REFUNDING = {2, 5, 8, 11}  # Same months but with 10y/30y emphasis


@dataclass
class AuctionDay:
    date: date
    auction_type: str
    maturity_category: str
    expected_size_bn: float


class TreasuryAuctionCalendar:
    """U.S. Treasury auction calendar and cycle feature generator.

    Generates auction-related equity market features without requiring
    live auction data. Uses the standard Treasury auction schedule pattern.

    Args:
        start_year: First year to generate.
        end_year: Last year to generate.
    """

    def __init__(self, start_year: int = 2015, end_year: int = 2030):
        self.start_year = start_year
        self.end_year = end_year
        self._auction_dates: Set[date] = set()
        self._auction_records: List[AuctionDay] = []
        self._generate()

    def _generate(self) -> None:
        """Generate auction dates from standard schedule pattern."""
        for year in range(self.start_year, self.end_year + 1):
            for month in range(1, 13):
                # Find first weekday of the month
                first_day = date(year, month, 1)
                first_weekday = first_day.weekday()

                for week_num in range(1, 5):
                    for slot_idx, (slot_week, slot_dow, auction_type) in enumerate(AUCTION_SLOTS):
                        if slot_week != week_num:
                            continue

                        day_offset = (week_num - 1) * 7 + slot_dow - first_weekday
                        auction_date = first_day + timedelta(days=day_offset)

                        if auction_date.month != month:
                            continue
                        if auction_date.weekday() >= 5:
                            continue

                        cat = self._classify(auction_type)
                        self._auction_dates.add(auction_date)
                        self._auction_records.append(
                            AuctionDay(
                                date=auction_date,
                                auction_type=auction_type,
                                maturity_category=cat,
                                expected_size_bn=self._size_estimate(auction_type),
                            )
                        )

    @staticmethod
    def _classify(auction_type: str) -> str:
        if "TIPS" in auction_type:
            return "tips"
        if any(x in auction_type for x in ("10-year", "30-year")) and "TIPS" not in auction_type:
            return "long_bond"
        if any(x in auction_type for x in ("2-year", "3-year", "5-year", "7-year")):
            return "note"
        if "FRN" in auction_type:
            return "frn"
        return "bill"

    @staticmethod
    def _size_estimate(auction_type: str) -> float:
        estimates = {
            "4-week": 75.0,
            "8-week": 60.0,
            "13-week": 70.0,
            "26-week": 60.0,
            "52-week": 45.0,
            "2-year": 60.0,
            "3-year": 55.0,
            "5-year": 55.0,
            "7-year": 40.0,
            "10-year": 35.0,
            "30-year": 20.0,
            "5-year TIPS": 22.0,
            "10-year TIPS": 17.0,
            "30-year TIPS": 9.0,
            "2-year FRN": 28.0,
        }
        return estimates.get(auction_type, 30.0)

    def is_auction_day(self, d: date) -> bool:
        """Check if a given date is a Treasury auction day."""
        return d in self._auction_dates

    def get_auctions(self, d: date) -> List[AuctionDay]:
        """Get all auctions scheduled for a given date."""
        return [a for a in self._auction_records if a.date == d]

    def auction_cycle_features(
        self,
        df: pd.DataFrame,
        date_col: str = "Date",
    ) -> pd.DataFrame:
        """Generate auction cycle features for a price DataFrame.

        Adds columns:
          is_auction_day: 1 if Treasury auction day.
          days_to_auction: Days to next scheduled auction (-ve after).
          days_since_auction: Days since last auction.
          auction_stress: Composite stress indicator (pre-auction buildup).
          is_refunding_week: 1 during quarterly refunding weeks.

        Args:
            df: DataFrame with at minimum a date column.
            date_col: Name of the date column.

        Returns:
            DataFrame with added auction cycle feature columns.
        """
        df = df.copy()
        df["_date"] = pd.to_datetime(df[date_col]).dt.date
        dates = df["_date"].values
        n = len(df)

        is_auction = np.zeros(n, dtype=np.int8)
        days_to_next = np.zeros(n, dtype=np.float64)
        days_since_prev = np.zeros(n, dtype=np.float64)
        auction_stress = np.zeros(n, dtype=np.float64)
        is_refunding = np.zeros(n, dtype=np.int8)

        auction_dates_sorted = sorted(self._auction_dates)

        for i, d in enumerate(dates):
            if self.is_auction_day(d):
                is_auction[i] = 1

            # Days to next auction
            for ad in auction_dates_sorted:
                if ad >= d:
                    days_to_next[i] = (ad - d).days
                    break
            else:
                days_to_next[i] = 7.0

            # Days since last auction
            for ad in reversed(auction_dates_sorted):
                if ad <= d:
                    days_since_prev[i] = (d - ad).days
                    break

            # Auction stress: peaks 1-2 days before auction, decays after
            if days_to_next[i] <= 2:
                auction_stress[i] = 1.0
            elif days_to_next[i] <= 5:
                auction_stress[i] = 0.5
            elif days_to_next[i] <= 10:
                auction_stress[i] = 0.25

            # Refunding weeks
            if d.month in MID_MONTH_REFUNDING and 7 <= d.day <= 14:
                is_refunding[i] = 1

        df["is_auction_day"] = is_auction
        df["days_to_auction"] = days_to_next
        df["days_since_auction"] = days_since_prev
        df["auction_stress"] = auction_stress
        df["is_refunding_week"] = is_refunding

        df.drop(columns=["_date"], inplace=True)
        return df

    def cycle_signal(self, d: date) -> float:
        """Generate a single auction cycle signal for a given date.

        Returns a value in [-1, 1]:
          > 0: Post-auction period (higher liquidity, bullish drift).
          < 0: Pre-auction buildup (lower liquidity, cautious/driftless).
          ~0: Normal inter-auction period.

        Args:
            d: Trading date.

        Returns:
            Signal strength in [-1, 1].
        """
        auction_dates_sorted = sorted(self._auction_dates)

        days_to = 7.0
        days_since = 7.0

        for ad in auction_dates_sorted:
            if ad >= d:
                days_to = (ad - d).days
                break

        for ad in reversed(auction_dates_sorted):
            if ad <= d:
                days_since = (d - ad).days
                break

        if days_to <= 2:
            return -0.5
        if days_to <= 5:
            return -0.25
        if days_since <= 2:
            return 0.3
        if days_since <= 5:
            return 0.15
        return 0.0
