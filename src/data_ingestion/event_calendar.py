"""Financial event calendar backed by DuckDB.

Provides hardcoded macro event schedules (FOMC, OPEX, CPI, NFP, Triple Witching)
plus earnings dates, with methods to query events by date range and check event days.

Usage:
    db = EventCalendarDB("data/event_calendar.duckdb")
    db.populate_defaults()
    db.get_events_in_range("2024-01-01", "2024-12-31")
    db.is_event_day("2024-03-20")  # FOMC day → True
"""

from __future__ import annotations

import calendar
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

_IMPACT_ORDER = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL,
    event_date DATE NOT NULL,
    symbol TEXT,
    expected_impact TEXT NOT NULL DEFAULT 'HIGH',
    description TEXT,
    UNIQUE(event_type, event_date, symbol)
);
"""


def _third_friday(year: int, month: int) -> date:
    """Return the third Friday of a given year/month."""
    c = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in c if week[calendar.FRIDAY] != 0]
    return date(year, month, fridays[2])


def _first_friday(year: int, month: int) -> date:
    """Return the first Friday of a given year/month."""
    c = calendar.monthcalendar(year, month)
    for week in c:
        if week[calendar.FRIDAY] != 0:
            return date(year, month, week[calendar.FRIDAY])
    return date(year, month, 7)


def _second_week_wednesday(year: int, month: int) -> date:
    """Return the Wednesday of the second full week (approximate CPI release)."""
    first = date(year, month, 1)
    days_ahead = (2 - first.weekday()) % 7  # next Wednesday
    first_wed = first + timedelta(days=days_ahead)
    return first_wed + timedelta(days=7)


_FOMC_DATES: dict[int, list[date]] = {
    2020: [
        date(2020, 1, 29),
        date(2020, 3, 18),
        date(2020, 4, 29),
        date(2020, 6, 10),
        date(2020, 7, 29),
        date(2020, 9, 16),
        date(2020, 11, 5),
        date(2020, 12, 16),
    ],
    2021: [
        date(2021, 1, 27),
        date(2021, 3, 17),
        date(2021, 4, 28),
        date(2021, 6, 16),
        date(2021, 7, 28),
        date(2021, 9, 22),
        date(2021, 11, 3),
        date(2021, 12, 15),
    ],
    2022: [
        date(2022, 1, 26),
        date(2022, 3, 16),
        date(2022, 5, 4),
        date(2022, 6, 15),
        date(2022, 7, 27),
        date(2022, 9, 21),
        date(2022, 11, 2),
        date(2022, 12, 14),
    ],
    2023: [
        date(2023, 2, 1),
        date(2023, 3, 22),
        date(2023, 5, 3),
        date(2023, 6, 14),
        date(2023, 7, 26),
        date(2023, 9, 20),
        date(2023, 11, 1),
        date(2023, 12, 13),
    ],
    2024: [
        date(2024, 1, 31),
        date(2024, 3, 20),
        date(2024, 5, 1),
        date(2024, 6, 12),
        date(2024, 7, 31),
        date(2024, 9, 18),
        date(2024, 11, 7),
        date(2024, 12, 18),
    ],
    2025: [
        date(2025, 1, 29),
        date(2025, 3, 19),
        date(2025, 5, 7),
        date(2025, 6, 18),
        date(2025, 7, 30),
        date(2025, 9, 17),
        date(2025, 11, 5),
        date(2025, 12, 17),
    ],
    2026: [
        date(2026, 1, 28),
        date(2026, 3, 18),
        date(2026, 4, 29),
        date(2026, 6, 17),
        date(2026, 7, 29),
        date(2026, 9, 16),
        date(2026, 11, 4),
        date(2026, 12, 16),
    ],
}


class EventCalendarDB:
    """DuckDB-backed financial event calendar.

    Stores macro events (FOMC, OPEX, CPI, NFP, Triple Witching) and
    per-symbol events (earnings, dividends) with impact classification.
    """

    def __init__(self, db_path: str | Path = "data/event_calendar.duckdb") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self._db_path))
        self._con.execute(SCHEMA_SQL)

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._con

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "EventCalendarDB":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def populate_defaults(self, years: tuple[int, int] = (2020, 2026)) -> int:
        """Populate with hardcoded FOMC, OPEX, CPI, NFP, Triple Witching dates.

        Returns:
            Number of rows inserted.
        """
        records: list[dict] = []
        event_id = 0

        for year in range(years[0], years[1] + 1):
            # FOMC
            for d in _FOMC_DATES.get(year, []):
                records.append(
                    {
                        "event_id": event_id,
                        "event_type": "FOMC",
                        "event_date": d,
                        "symbol": None,
                        "expected_impact": "HIGH",
                        "description": f"FOMC meeting {d.strftime('%b %d %Y')}",
                    }
                )
                event_id += 1

            # OPEX (third Friday)
            for month in range(1, 13):
                tf = _third_friday(year, month)
                records.append(
                    {
                        "event_id": event_id,
                        "event_type": "OPEX",
                        "event_date": tf,
                        "symbol": None,
                        "expected_impact": "MEDIUM",
                        "description": f"Options expiration {tf.strftime('%b %d %Y')}",
                    }
                )
                event_id += 1

            # CPI (second-week Wednesday, monthly)
            for month in range(1, 13):
                cpi_date = _second_week_wednesday(year, month)
                records.append(
                    {
                        "event_id": event_id,
                        "event_type": "CPI",
                        "event_date": cpi_date,
                        "symbol": None,
                        "expected_impact": "HIGH",
                        "description": f"CPI release {cpi_date.strftime('%b %d %Y')}",
                    }
                )
                event_id += 1

            # NFP (first Friday, monthly)
            for month in range(1, 13):
                nfp_date = _first_friday(year, month)
                records.append(
                    {
                        "event_id": event_id,
                        "event_type": "NFP",
                        "event_date": nfp_date,
                        "symbol": None,
                        "expected_impact": "HIGH",
                        "description": f"Nonfarm payrolls {nfp_date.strftime('%b %d %Y')}",
                    }
                )
                event_id += 1

            # Triple Witching (third Friday of Mar/Jun/Sep/Dec)
            for month in [3, 6, 9, 12]:
                tw_date = _third_friday(year, month)
                records.append(
                    {
                        "event_id": event_id,
                        "event_type": "TRIPLE_WITCHING",
                        "event_date": tw_date,
                        "symbol": None,
                        "expected_impact": "HIGH",
                        "description": f"Triple witching {tw_date.strftime('%b %d %Y')}",
                    }
                )
                event_id += 1

        if records:
            df = pd.DataFrame(records)
            self._con.execute("DELETE FROM events")
            self._con.execute("INSERT INTO events SELECT * FROM df")
            logger.info("Inserted %d event records for %d-%d", len(records), years[0], years[1])

        return len(records)

    def add_earnings(
        self,
        symbol: str,
        earnings_dates: list[date],
        impact: str = "HIGH",
    ) -> int:
        """Add per-symbol earnings dates."""
        max_id = self._con.execute("SELECT COALESCE(MAX(event_id), -1) FROM events").fetchone()[0]
        records = []
        for i, d in enumerate(earnings_dates):
            records.append(
                {
                    "event_id": max_id + 1 + i,
                    "event_type": "EARNINGS",
                    "event_date": d,
                    "symbol": symbol,
                    "expected_impact": impact,
                    "description": f"{symbol} earnings {d.strftime('%b %d %Y')}",
                }
            )
        if records:
            df = pd.DataFrame(records)
            self._con.execute("INSERT OR REPLACE INTO events SELECT * FROM df")
            logger.info("Added %d earnings dates for %s", len(records), symbol)
        return len(records)

    def get_events_in_range(
        self,
        start: str | date,
        end: str | date,
        event_types: list[str] | None = None,
        min_impact: str | None = None,
    ) -> pd.DataFrame:
        """Query events within a date range.

        Args:
            start: Start date (inclusive).
            end: End date (inclusive).
            event_types: Filter by event type (e.g. ['FOMC', 'CPI']).
            min_impact: Minimum impact level ('LOW', 'MEDIUM', 'HIGH').

        Returns:
            DataFrame with event columns.
        """
        conditions = ["event_date >= ?", "event_date <= ?"]
        params: list = [str(start), str(end)]

        if event_types:
            placeholders = ",".join(["?"] * len(event_types))
            conditions.append(f"event_type IN ({placeholders})")
            params.extend(event_types)

        if min_impact:
            impact_val = _IMPACT_ORDER.get(min_impact.upper(), 1)
            allowed = [k for k, v in _IMPACT_ORDER.items() if v >= impact_val]
            placeholders = ",".join(["?"] * len(allowed))
            conditions.append(f"expected_impact IN ({placeholders})")
            params.extend(allowed)

        where_clause = " AND ".join(conditions)
        return self._con.execute(
            f"SELECT * FROM events WHERE {where_clause} ORDER BY event_date",
            params,
        ).df()

    def is_event_day(
        self,
        check_date: str | date,
        symbol: str | None = None,
        min_impact: str = "LOW",
    ) -> bool:
        """Check if a date is an event day.

        Args:
            check_date: Date to check.
            symbol: If provided, also match symbol-specific events.
            min_impact: Minimum impact to count as an event.

        Returns:
            True if any event exists on this date.
        """
        conditions = ["event_date = ?"]
        params: list = [str(check_date)]

        if symbol:
            conditions.append("(symbol IS NULL OR symbol = ?)")
            params.append(symbol)

        impact_val = _IMPACT_ORDER.get(min_impact.upper(), 1)
        allowed = [k for k, v in _IMPACT_ORDER.items() if v >= impact_val]
        placeholders = ",".join(["?"] * len(allowed))
        conditions.append(f"expected_impact IN ({placeholders})")
        params.extend(allowed)

        where_clause = " AND ".join(conditions)
        result = self._con.execute(
            f"SELECT COUNT(*) FROM events WHERE {where_clause}", params
        ).fetchone()
        return result[0] > 0

    def get_event_impact(
        self,
        check_date: str | date,
        symbol: str | None = None,
    ) -> str | None:
        """Get the highest impact level for events on a given date.

        Returns:
            'HIGH', 'MEDIUM', 'LOW', or None if no events.
        """
        conditions = ["event_date = ?"]
        params: list = [str(check_date)]
        if symbol:
            conditions.append("(symbol IS NULL OR symbol = ?)")
            params.append(symbol)

        where_clause = " AND ".join(conditions)
        result = self._con.execute(
            f"SELECT expected_impact FROM events WHERE {where_clause}", params
        ).fetchall()

        if not result:
            return None

        impacts = [row[0] for row in result]
        for impact in ["HIGH", "MEDIUM", "LOW"]:
            if impact in impacts:
                return impact
        return None

    def count_events(self) -> int:
        """Return total number of rows in the events table."""
        return self._con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
