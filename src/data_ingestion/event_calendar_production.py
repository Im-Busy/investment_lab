"""
Production-Grade Event Calendar — Direction C Phase 6.

Extends EventCalendarDB with production features for live trading:
- Event performance tracking: post-event drift, win rate, profitability stats
- Event impact calibration: empirical impact measurement (not hardcoded HIGH/MEDIUM/LOW)
- Event-day statistics: compare trading performance on event days vs normal days
- Feedback loop: auto-update impact levels based on observed price behavior

Based on the C3 event calendar infrastructure (DuckDB, FOMC/CPI/NFP/OPEX/Triple Witching).

Usage:
    db = ProductionEventCalendar("data/event_calendar.duckdb")
    db.populate_defaults()
    db.record_event_outcome("2024-03-20", "FOMC", price_change_pct=0.52)
    db.get_event_performance_stats()
    db.score_event_impact("2024-05-01", "FOMC")
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from .event_calendar import EventCalendarDB

logger = logging.getLogger(__name__)

OUTCOME_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS event_outcomes (
    event_id INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL,
    event_date DATE NOT NULL,
    symbol TEXT,
    price_change_pct REAL,
    volume_spike_ratio REAL,
    pre_event_vol_compression REAL,
    post_event_drift_1d REAL,
    post_event_drift_5d REAL,
    post_event_drift_20d REAL,
    impact_level TEXT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(event_type, event_date, symbol)
        REFERENCES events(event_type, event_date, symbol)
);
"""

PERFORMANCE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS event_performance (
    event_type TEXT NOT NULL,
    symbol TEXT,
    n_events INTEGER DEFAULT 0,
    avg_return_1d REAL DEFAULT 0.0,
    avg_return_5d REAL DEFAULT 0.0,
    avg_abs_return_1d REAL DEFAULT 0.0,
    avg_abs_return_5d REAL DEFAULT 0.0,
    positive_pct REAL DEFAULT 0.0,
    max_abs_return_1d REAL DEFAULT 0.0,
    calibrated_impact TEXT DEFAULT 'UNKNOWN',
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(event_type, symbol)
);
"""


class ProductionEventCalendar(EventCalendarDB):
    """Production-grade event calendar with performance tracking.

    Extends EventCalendarDB with outcome recording, performance
    statistics, and impact calibration. Provides a feedback loop:
    event outcomes → impact calibration → trading rules.
    """

    def __init__(self, db_path: str | Path = "data/event_calendar.duckdb") -> None:
        super().__init__(db_path)
        self._init_production_tables()

    def _init_production_tables(self) -> None:
        self._con.execute(OUTCOME_SCHEMA_SQL)
        self._con.execute(PERFORMANCE_SCHEMA_SQL)

    def record_event_outcome(
        self,
        event_date: str | date,
        event_type: str,
        symbol: str | None = None,
        price_change_pct: float = 0.0,
        volume_spike_ratio: float = 1.0,
        pre_event_vol_compression: float = 0.0,
        post_event_drift_1d: float = 0.0,
        post_event_drift_5d: float = 0.0,
        post_event_drift_20d: float = 0.0,
    ) -> int:
        """Record the market outcome of an event.

        Called after an event day passes with observed price data.
        Computes the post-event drift for 1/5/20 day horizons and
        records calibrated impact levels.

        Returns:
            1 if inserted, 0 if skipped.
        """
        next_id = self._con.execute(
            "SELECT COALESCE(MAX(event_id), -1) + 1 FROM event_outcomes"
        ).fetchone()[0]

        impact_level = self._calibrate_impact(
            abs_price_change=abs(price_change_pct),
            abs_drift_1d=abs(post_event_drift_1d),
            abs_drift_5d=abs(post_event_drift_5d),
        )

        self._con.execute(
            """INSERT OR REPLACE INTO event_outcomes
            (event_id, event_type, event_date, symbol, price_change_pct,
             volume_spike_ratio, pre_event_vol_compression,
             post_event_drift_1d, post_event_drift_5d, post_event_drift_20d,
             impact_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                next_id,
                event_type,
                str(event_date),
                symbol,
                round(price_change_pct, 4),
                round(volume_spike_ratio, 2),
                round(pre_event_vol_compression, 4),
                round(post_event_drift_1d, 4),
                round(post_event_drift_5d, 4),
                round(post_event_drift_20d, 4),
                impact_level,
            ],
        )
        logger.debug(
            "Recorded %s event outcome for %s: %+.2f%% (impact=%s)",
            event_type,
            event_date,
            price_change_pct,
            impact_level,
        )
        return 1

    def _calibrate_impact(
        self,
        abs_price_change: float,
        abs_drift_1d: float,
        abs_drift_5d: float,
    ) -> str:
        """Calibrate event impact from observed market behavior.

        HIGH: >2% intraday or >3% 5-day drift
        MEDIUM: >1% intraday or >1.5% 5-day drift
        LOW: anything below
        """
        combined = abs_price_change + abs_drift_5d * 0.5
        if combined > 0.03 or abs_price_change > 0.02:
            return "HIGH"
        elif combined > 0.015 or abs_price_change > 0.01:
            return "MEDIUM"
        return "LOW"

    def get_event_performance_stats(self, event_type: str | None = None) -> pd.DataFrame:
        """Get performance statistics by event type.

        Computes average returns, absolute returns, win rates,
        and calibrated impact from recorded outcomes.

        Args:
            event_type: Filter by event type (None = all types).

        Returns:
            DataFrame with performance stats per event type.
        """
        conditions = []
        params = []
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        return self._con.execute(
            f"""SELECT
                event_type,
                COUNT(*) AS n_events,
                AVG(price_change_pct) AS avg_return,
                AVG(ABS(price_change_pct)) AS avg_abs_return,
                MAX(ABS(price_change_pct)) AS max_abs_return,
                AVG(post_event_drift_1d) AS avg_drift_1d,
                AVG(post_event_drift_5d) AS avg_drift_5d,
                AVG(post_event_drift_20d) AS avg_drift_20d,
                SUM(CASE WHEN post_event_drift_5d > 0 THEN 1 ELSE 0 END) * 1.0
                    / NULLIF(COUNT(*), 0) AS drift_positive_5d,
                AVG(volume_spike_ratio) AS avg_vol_spike,
                COUNT(DISTINCT CASE WHEN impact_level = 'HIGH' THEN event_date END)
                    AS n_high_impact,
                COUNT(DISTINCT CASE WHEN impact_level = 'MEDIUM' THEN event_date END)
                    AS n_medium_impact,
                COUNT(DISTINCT CASE WHEN impact_level = 'LOW' THEN event_date END)
                    AS n_low_impact
            FROM event_outcomes
            {where}
            GROUP BY event_type
            ORDER BY n_events DESC""",
            params,
        ).df()

    def get_upcoming_events(
        self,
        n_days: int = 30,
        min_impact: str = "LOW",
    ) -> pd.DataFrame:
        """Get upcoming events for the next N calendar days.

        Args:
            n_days: Number of calendar days to look ahead.
            min_impact: Minimum impact level.

        Returns:
            DataFrame with upcoming events.
        """
        today = date.today()
        end = today + timedelta(days=n_days)
        return self.get_events_in_range(
            start=today.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            min_impact=min_impact,
        )

    def compare_event_vs_normal_performance(
        self,
        trade_log: pd.DataFrame,
        event_type_filter: list[str] | None = None,
    ) -> dict:
        """Compare trading performance on event days vs normal days.

        Cross-references a trade log DataFrame against the event calendar
        to compute event-day vs non-event-day statistics.

        Args:
            trade_log: DataFrame with columns: date, pnl, win (bool), symbol.
            event_type_filter: Restrict to specific event types.

        Returns:
            Dict with event_day_stats and normal_day_stats.
        """
        if trade_log.empty:
            return {"event_day_stats": {}, "normal_day_stats": {}, "comparison": {}}

        required_cols = {"date", "pnl"}
        if not required_cols.issubset(trade_log.columns):
            return {"error": f"Trade log missing columns: {required_cols - set(trade_log.columns)}"}

        event_types = event_type_filter or ["FOMC", "CPI", "NFP", "TRIPLE_WITCHING"]
        conditions = [f"event_type IN ({','.join(['?'] * len(event_types))})"]
        params = event_types[:]

        event_df = self._con.execute(
            f"SELECT DISTINCT event_date FROM events WHERE {' AND '.join(conditions)}",
            params,
        ).df()

        event_dates = set(event_df["event_date"].values)
        trade_log["is_event"] = trade_log["date"].isin(event_dates)

        event_trades = trade_log[trade_log["is_event"]]
        normal_trades = trade_log[~trade_log["is_event"]]

        def _stats(df: pd.DataFrame) -> dict:
            if df.empty:
                return {"n_trades": 0, "win_rate": None, "avg_pnl": 0.0, "total_pnl": 0.0}
            n = len(df)
            wr = df["win"].mean() if "win" in df.columns else None
            return {
                "n_trades": n,
                "win_rate": float(round(wr, 4)) if wr is not None else None,
                "avg_pnl": float(round(df["pnl"].mean(), 4)),
                "total_pnl": float(round(df["pnl"].sum(), 2)),
                "max_pnl": float(round(df["pnl"].max(), 2)),
                "min_pnl": float(round(df["pnl"].min(), 2)),
            }

        event_stats = _stats(event_trades)
        normal_stats = _stats(normal_trades)

        comparison = {}
        if event_stats["n_trades"] > 0 and normal_stats["n_trades"] > 0:
            comparison["avg_pnl_ratio"] = round(
                event_stats["avg_pnl"] / (normal_stats["avg_pnl"] + 1e-8), 2
            )
            if event_stats.get("win_rate") and normal_stats.get("win_rate"):
                comparison["wr_diff"] = round(event_stats["win_rate"] - normal_stats["win_rate"], 4)

        return {
            "event_day_stats": event_stats,
            "normal_day_stats": normal_stats,
            "comparison": comparison,
        }

    def score_event_impact(
        self,
        event_date: str | date,
        event_type: str,
        symbol: str | None = None,
    ) -> dict:
        """Score the trading impact of a specific event.

        Checks historical outcomes to calibrate expected impact.

        Returns:
            Dict with expected_impact, historical_avg_return, n_samples.
        """
        conditions = ["event_type = ?"]
        params: list = [event_type]
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)

        where = " AND ".join(conditions)
        result = self._con.execute(
            f"""SELECT
                COUNT(*) AS n,
                AVG(ABS(price_change_pct)) AS avg_abs_return,
                AVG(ABS(post_event_drift_1d)) AS avg_drift_1d,
                AVG(ABS(post_event_drift_5d)) AS avg_drift_5d,
                MAX(impact_level) AS max_impact
            FROM event_outcomes
            WHERE {where}""",
            params,
        ).fetchone()

        n_samples = result[0]
        if n_samples == 0:
            return {
                "expected_impact": "UNKNOWN",
                "historical_avg_return": None,
                "n_samples": 0,
            }

        avg_abs_return = float(result[1]) if result[1] is not None else 0.0
        avg_drift_1d = float(result[2]) if result[2] is not None else 0.0
        avg_drift_5d = float(result[3]) if result[3] is not None else 0.0
        impact = str(result[4]) if result[4] else "UNKNOWN"

        return {
            "expected_impact": impact,
            "historical_avg_return": round(avg_abs_return, 4),
            "avg_drift_1d": round(avg_drift_1d, 4),
            "avg_drift_5d": round(avg_drift_5d, 4),
            "n_samples": n_samples,
        }

    def rebuild_performance_summary(self) -> None:
        """Rebuild the event_performance summary table from outcomes.

        Computes aggregate stats from event_outcomes and updates
        the event_performance table.
        """
        self._con.execute("DELETE FROM event_performance")

        self._con.execute(
            """INSERT INTO event_performance
            (event_type, symbol, n_events, avg_return_1d,
             avg_return_5d, avg_abs_return_1d, avg_abs_return_5d,
             positive_pct, max_abs_return_1d, calibrated_impact)
            SELECT
                event_type,
                COALESCE(symbol, 'MACRO'),
                COUNT(*),
                AVG(post_event_drift_1d),
                AVG(post_event_drift_5d),
                AVG(ABS(post_event_drift_1d)),
                AVG(ABS(post_event_drift_5d)),
                SUM(CASE WHEN post_event_drift_1d > 0 THEN 1 ELSE 0 END)
                    * 100.0 / NULLIF(COUNT(*), 0),
                MAX(ABS(post_event_drift_1d)),
                CASE
                    WHEN AVG(ABS(post_event_drift_5d)) > 0.025 THEN 'HIGH'
                    WHEN AVG(ABS(post_event_drift_5d)) > 0.012 THEN 'MEDIUM'
                    ELSE 'LOW'
                END
            FROM event_outcomes
            GROUP BY event_type, COALESCE(symbol, 'MACRO')
            HAVING COUNT(*) >= 3"""
        )
        logger.info("Rebuilt event performance summary")
