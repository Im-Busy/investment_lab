#!/usr/bin/env python3
"""Query the per-instrument performance log.

Phase 25 — CLI for querying logs/per_instrument_performance.jsonl.
Supports filtering by instrument, period, verdict, and strategy.

Usage:
    # Show summary stats across all logged instruments
    uv run scripts/log_query.py --summary

    # Show latest result for SPY
    uv run scripts/log_query.py --instrument SPY

    # Show performance evolution for XLK across all runs
    uv run scripts/log_query.py --instrument XLK --history

    # Show all PASS verdicts
    uv run scripts/log_query.py --verdict PASS

    # Show all OOS records
    uv run scripts/log_query.py --period OOS

    # Show the full log
    uv run scripts/log_query.py --all --limit 50

    # Show by strategy type
    uv run scripts/log_query.py --strategy rules_first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.per_instrument.performance_tracker import (
    query_log,
    get_latest_for,
    compare_across_runs,
    get_summary_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query per-instrument performance log",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--summary", action="store_true", help="Show aggregate statistics")
    parser.add_argument("--instrument", type=str, default=None, help="Filter by ticker symbol")
    parser.add_argument(
        "--period", type=str, default=None, choices=["IS", "OOS", "FULL"], help="Filter by period"
    )
    parser.add_argument(
        "--verdict",
        type=str,
        default=None,
        choices=["PASS", "FAIL", "MARGINAL"],
        help="Filter by verdict",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["rules_first", "smc", "combined"],
        help="Filter by strategy",
    )
    parser.add_argument(
        "--history", action="store_true", help="Show performance evolution for an instrument"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="sharpe",
        help="Metric to track with --history (default: sharpe)",
    )
    parser.add_argument("--all", action="store_true", help="Show all records (no filter)")
    parser.add_argument("--limit", type=int, default=20, help="Max records to show (default: 20)")

    args = parser.parse_args()

    if args.summary:
        stats = get_summary_stats()
        print("\nPerformance Log Summary")
        print("=" * 60)
        if stats["total_records"] == 0:
            print("  No records found. Run backtests with log_performance() enabled.")
            return
        print(f"  Total records:          {stats['total_records']}")
        print(f"  Unique instruments:     {stats['unique_instruments']}")
        print(f"  Instruments:            {', '.join(stats.get('instruments', []))}")
        print(f"  Mean Sharpe (all):      {stats['mean_sharpe']}")
        print(
            f"  Pass rate:              {stats['pass_rate']:.1%} ({stats['by_verdict'].get('PASS', 0)}/{stats['total_records']})"
        )
        print(f"  By period:              {stats['by_period']}")
        print(f"  By verdict:             {stats['by_verdict']}")
        return

    if args.history:
        if not args.instrument:
            print("Error: --history requires --instrument")
            sys.exit(1)
        results = compare_across_runs(args.instrument, metric=args.metric)
        if not results:
            print(f"No records found for {args.instrument}")
            return
        print(f"\nPerformance Evolution — {args.instrument} ({args.metric})")
        print("=" * 60)
        print(f"{'Date':<12} {'Value':>10} {'Verdict':>10}")
        print("-" * 35)
        for r in results:
            val = r["value"]
            val_str = f"{val:.3f}" if isinstance(val, (int, float)) else str(val)
            print(f"{r['date']:<12} {val_str:>10} {r['verdict']:>10}")
        return

    records = query_log(
        instrument=args.instrument if not args.all else None,
        period=args.period,
        verdict=args.verdict,
        strategy=args.strategy,
        limit=args.limit,
    )

    if not records:
        print("No matching records found.")
        return

    print(f"\nPerformance Log — {len(records)} records")
    print("=" * 120)
    header = f"{'Date':<12} {'Symbol':<10} {'Period':<6} {'Strat':<12} {'Sharpe':>8} {'Ret%':>8} {'Trades':>7} {'WR%':>6} {'PF':>7} {'MaxDD%':>7} {'Verdict':>10}"
    print(header)
    print("-" * 120)
    for r in records:
        ts = r.get("timestamp", "")[:10] if r.get("timestamp") else ""
        sym = r.get("instrument", "")
        period = r.get("period", "")
        strat = r.get("strategy", "")
        m = r.get("metrics", {})
        sharpe = m.get("sharpe", "")
        ret = m.get("return_pct", "")
        trades = m.get("trades", "")
        wr = m.get("win_rate", "")
        pf = m.get("profit_factor", "")
        mdd = m.get("max_dd_pct", "")
        verdict = r.get("verdict", "")

        def _fmt(v: object) -> str:
            if v is None or v == "":
                return "-"
            try:
                return f"{float(str(v)):.2f}"
            except (ValueError, TypeError):
                return str(v)[:6]

        print(
            f"{ts:<12} {sym:<10} {period:<6} {strat:<12} "
            f"{_fmt(sharpe):>8} {_fmt(ret):>8} {_fmt(trades):>7} "
            f"{_fmt(wr):>6} {_fmt(pf):>7} {_fmt(mdd):>7} {verdict:>10}"
        )


if __name__ == "__main__":
    main()
