"""Build the financial event calendar database with hardcoded schedules.

Populates DuckDB with FOMC, OPEX, CPI, NFP, and Triple Witching dates for 2020-2026.

Usage:
    uv run scripts/build_event_calendar.py
    uv run scripts/build_event_calendar.py --db data/event_calendar.duckdb
    uv run scripts/build_event_calendar.py --years 2015 2030
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_ingestion.event_calendar import EventCalendarDB


def main() -> None:
    parser = argparse.ArgumentParser(description="Build financial event calendar")
    parser.add_argument(
        "--db",
        type=str,
        default="data/event_calendar.duckdb",
        help="Path to DuckDB database file.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs=2,
        default=[2020, 2026],
        metavar=("START", "END"),
        help="Year range to populate (default: 2020 2026).",
    )
    args = parser.parse_args()

    db = EventCalendarDB(args.db)
    count = db.populate_defaults(years=(args.years[0], args.years[1]))

    print(f"Populated {count} events in {args.db}")
    print(f"Year range: {args.years[0]}-{args.years[1]}")

    # Print by type
    types = db.connection.execute(
        "SELECT event_type, COUNT(*) FROM events GROUP BY event_type ORDER BY event_type"
    ).fetchall()
    print("\nEvents by type:")
    for event_type, n in types:
        print(f"  {event_type}: {n}")

    # Sample events
    print("\nSample FOMC events:")
    fomc = db.connection.execute(
        "SELECT event_date, description FROM events WHERE event_type='FOMC' ORDER BY event_date LIMIT 5"
    ).fetchall()
    for d, desc in fomc:
        print(f"  {d}: {desc}")

    db.close()


if __name__ == "__main__":
    main()
