"""Append-only JSONL performance ledger for per-instrument tracking.

Phase 25 — Every backtest run appends a record to
logs/per_instrument_performance.jsonl, creating a cumulative history of
per-instrument results over time. No data is ever overwritten.

Schema:
    timestamp: ISO datetime of the backtest run
    instrument: ticker symbol
    period: "IS" | "OOS" | "FULL"
    date_range: "start:end" date string
    strategy: "rules_first" | "smc" | "combined"
    config: dict of params used
    metrics: dict of performance metrics
    benchmark: dict of buy-and-hold comparison
    verdict: "PASS" | "FAIL" | "MARGINAL"
    notes: free-text field
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

LOG_PATH = Path("logs/per_instrument_performance.jsonl")


@dataclass
class PerformanceRecord:
    """Single per-instrument performance record."""

    timestamp: str = ""
    instrument: str = ""
    period: str = ""
    date_range: str = ""
    strategy: str = "rules_first"
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    benchmark: dict[str, Any] = field(default_factory=dict)
    verdict: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PerformanceRecord":
        return cls(**{k: d.get(k, "") for k in cls.__dataclass_fields__})


def log_performance(
    instrument: str,
    period: str,
    date_range: str,
    strategy: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    benchmark: dict[str, Any],
    verdict: str,
    notes: str = "",
) -> None:
    """Append a performance record to the cumulative log.

    Args:
        instrument: Ticker symbol.
        period: "IS", "OOS", or "FULL".
        date_range: "YYYY-MM-DD:YYYY-MM-DD" string.
        strategy: "rules_first", "smc", or "combined".
        config: Dict of strategy params used.
        metrics: Dict with sharpe, return_pct, trades, win_rate, etc.
        benchmark: Dict with buyhold_return_pct, buyhold_sharpe.
        verdict: "PASS" (OOS Sharpe > 0.1), "FAIL" (OOS Sharpe < -0.05),
                 or "MARGINAL".
        notes: Free-text field for context.
    """
    record = PerformanceRecord(
        timestamp=datetime.now().isoformat(),
        instrument=instrument,
        period=period,
        date_range=date_range,
        strategy=strategy,
        config=config,
        metrics=metrics,
        benchmark=benchmark,
        verdict=verdict,
        notes=notes,
    )
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def query_log(
    instrument: str | None = None,
    period: str | None = None,
    verdict: str | None = None,
    strategy: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Query the cumulative performance log with filters.

    Returns most recent records first.

    Args:
        instrument: Filter by ticker symbol (None = all).
        period: Filter by period (None = all).
        verdict: Filter by verdict (None = all).
        strategy: Filter by strategy type (None = all).
        limit: Max records to return.

    Returns:
        List of performance record dicts, sorted by timestamp descending.
    """
    records: list[dict[str, Any]] = []
    if not LOG_PATH.exists():
        return records
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            if instrument and r.get("instrument") != instrument:
                continue
            if period and r.get("period") != period:
                continue
            if verdict and r.get("verdict") != verdict:
                continue
            if strategy and r.get("strategy") != strategy:
                continue
            records.append(r)
    records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return records[:limit]


def get_latest_for(instrument: str) -> dict[str, Any] | None:
    """Get the most recent performance record for an instrument."""
    results = query_log(instrument=instrument, limit=1)
    return results[0] if results else None


def compare_across_runs(
    instrument: str,
    metric: str = "sharpe",
) -> list[dict[str, Any]]:
    """Show performance evolution for an instrument across runs.

    Args:
        instrument: Ticker symbol.
        metric: Key within the 'metrics' dict to track (e.g. "sharpe").

    Returns:
        List of {"date": "...", "value": ...} sorted ascending by date.
    """
    records = query_log(instrument=instrument, limit=200)
    results: list[dict[str, Any]] = []
    for r in records:
        ts = r.get("timestamp", "")[:10]
        metrics = r.get("metrics", {})
        value = metrics.get(metric)
        results.append({"date": ts, "value": value, "verdict": r.get("verdict", "")})
    results.reverse()  # oldest first
    return results


def get_summary_stats() -> dict[str, Any]:
    """Return aggregate statistics across all logged instruments."""
    records = query_log(limit=10_000)
    if not records:
        return {"total_records": 0}

    instruments: set[str] = set()
    periods: dict[str, int] = {}
    verdicts: dict[str, int] = {}
    sharpe_values: list[float] = []

    for r in records:
        instruments.add(r.get("instrument", ""))
        p = r.get("period", "unknown")
        periods[p] = periods.get(p, 0) + 1
        v = r.get("verdict", "unknown")
        verdicts[v] = verdicts.get(v, 0) + 1
        s = r.get("metrics", {}).get("sharpe")
        if s is not None:
            try:
                sharpe_values.append(float(s))
            except (ValueError, TypeError):
                pass

    avg_sharpe = sum(sharpe_values) / len(sharpe_values) if sharpe_values else 0.0
    pass_count = verdicts.get("PASS", 0)
    total_with_verdict = sum(verdicts.values())

    return {
        "total_records": len(records),
        "unique_instruments": len(instruments),
        "instruments": sorted(instruments),
        "by_period": periods,
        "by_verdict": verdicts,
        "pass_rate": round(pass_count / total_with_verdict, 3) if total_with_verdict else 0,
        "mean_sharpe": round(avg_sharpe, 3),
    }
