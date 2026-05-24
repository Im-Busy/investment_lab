#!/usr/bin/env python3
"""RF2.2: Short-side sweep across instruments.

Runs short-side backtest (use_short=True) on all instruments to identify
which tickers benefit from short selling.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.backtest_rules_first import run_single

DEFAULT_TICKERS = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLP",
    "XLU",
    "XLB",
    "XLY",
    "GLD",
    "TLT",
    "BTC_USD",
]


def sweep_short_side(
    tickers: list[str],
    start: str,
    end: str,
    min_reliability: float = 0.70,
    entry_threshold: float = 0.55,
) -> list[dict]:
    """Run long-only and long+short on each ticker, compare."""
    results = []

    for ticker in tickers:
        try:
            long_only = run_single(
                ticker,
                cash=100_000,
                start=start,
                end=end,
                entry_threshold=entry_threshold,
                min_reliability=min_reliability,
                use_short=False,
                use_quality_registry=True,
                use_multi_tp=True,
            )
            long_short = run_single(
                ticker,
                cash=100_000,
                start=start,
                end=end,
                entry_threshold=entry_threshold,
                min_reliability=min_reliability,
                use_short=True,
                use_quality_registry=True,
                use_multi_tp=True,
            )

            results.append(
                {
                    "symbol": ticker,
                    "long_sharpe": long_only.get("sharpe"),
                    "long_short_sharpe": long_short.get("sharpe"),
                    "long_return_pct": long_only.get("return_pct"),
                    "long_short_return_pct": long_short.get("return_pct"),
                    "long_trades": long_only.get("trades", 0),
                    "long_short_trades": long_short.get("trades", 0),
                    "delta_sharpe": long_short.get("sharpe", -999) - long_only.get("sharpe", -999),
                    "edge_from_short": (
                        (long_short.get("sharpe", -999) - long_only.get("sharpe", -999)) > 0.05
                    ),
                }
            )
        except Exception as e:
            results.append({"symbol": ticker, "error": str(e)})

    return results


def main():
    parser = argparse.ArgumentParser(description="RF2.2: Short-side sweep across instruments")
    parser.add_argument(
        "--tickers", default=",".join(DEFAULT_TICKERS[:8]), help="Comma-separated tickers"
    )
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-05-20")
    parser.add_argument("--mr", type=float, default=0.70, help="Min reliability")
    parser.add_argument("--et", type=float, default=0.55, help="Entry threshold")
    parser.add_argument("--json-output", help="Save results to JSON")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")]

    print(f"Sweeping short-side on {len(tickers)} tickers")
    print(
        f"{'Symbol':8s} {'Long Sharpe':>12s} {'L+S Sharpe':>12s} {'Delta':>8s} {'Edge?':>6s} {'L Trades':>9s} {'L+S Trades':>10s}"
    )
    print("-" * 80)

    results = sweep_short_side(tickers, args.start, args.end, args.mr, args.et)

    for r in results:
        if "error" in r:
            print(f"{r['symbol']:8s} ERROR: {r['error']}")
            continue
        edge = "YES" if r["edge_from_short"] else "no"
        print(
            f"{r['symbol']:8s} {r['long_sharpe']:>12.3f} {r['long_short_sharpe']:>12.3f} "
            f"{r['delta_sharpe']:>+8.3f} {edge:>6s} {r['long_trades']:>9} {r['long_short_trades']:>10}"
        )

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump({"swept_at": datetime.now().isoformat(), "results": results}, f, indent=2)
        print(f"\nSaved to {args.json_output}")


if __name__ == "__main__":
    main()
