#!/usr/bin/env python3
"""RF1.1: Per-instrument optimal config auto-discovery.

Sweeps entry_threshold × min_reliability × trail_stop_atr × confluence_bonus
for each instrument and reports the OOS-best configuration.

Usage:
    uv run scripts/auto_tune_per_instrument.py --symbols SPY,QQQ,IWM --fast
    uv run scripts/auto_tune_per_instrument.py --symbols SPY --full
    uv run scripts/auto_tune_per_instrument.py --symbols SPY,QQQ,XLK,GLD,TLT --json-output outputs/auto_tune.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.backtest_rules_first import run_single, sweep_entry

DEFAULT_START = "2016-01-01"
DEFAULT_OOS_START = "2025-01-01"
DEFAULT_END = "2026-05-20"

GRID_FAST = {
    "entry_thresholds": [0.35, 0.55, 0.75],
    "min_reliability": [0.40, 0.70],
    "trail_stop_atr": [2.0, 3.0],
    "confluence_bonus": [0.05, 0.10],
}

GRID_FULL = {
    "entry_thresholds": [0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    "min_reliability": [0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    "trail_stop_atr": [1.5, 2.0, 2.5, 3.0, 4.0],
    "confluence_bonus": [0.0, 0.05, 0.10, 0.15],
}


def auto_tune_instrument(
    symbol: str,
    grid: dict,
    start: str = DEFAULT_START,
    oos_start: str = DEFAULT_OOS_START,
    end: str = DEFAULT_END,
) -> dict:
    """Find optimal config for one instrument via IS+OOS grid search."""
    best_oos_sharpe = -np.inf
    best_config: dict = {}
    best_is_result: dict = {}
    best_oos_result: dict = {}

    for et in grid["entry_thresholds"]:
        for mr in grid["min_reliability"]:
            for tsa in grid["trail_stop_atr"]:
                for cb in grid["confluence_bonus"]:
                    try:
                        is_result = run_single(
                            symbol,
                            cash=100_000,
                            start=start,
                            end=oos_start,
                            entry_threshold=et,
                            min_reliability=mr,
                            trail_stop_atr=tsa,
                            confluence_bonus=cb,
                            use_quality_registry=True,
                            use_multi_tp=True,
                            use_vix_gate=False,
                            use_yield_curve_gate=False,
                        )
                        oos_result = run_single(
                            symbol,
                            cash=100_000,
                            start=oos_start,
                            end=end,
                            entry_threshold=et,
                            min_reliability=mr,
                            trail_stop_atr=tsa,
                            confluence_bonus=cb,
                            use_quality_registry=True,
                            use_multi_tp=True,
                            use_vix_gate=False,
                            use_yield_curve_gate=False,
                        )
                        oos_sharpe = oos_result.get("sharpe", -np.inf)
                        if oos_sharpe > best_oos_sharpe:
                            best_oos_sharpe = oos_sharpe
                            best_config = {
                                "entry_threshold": et,
                                "min_reliability": mr,
                                "trail_stop_atr": tsa,
                                "confluence_bonus": cb,
                            }
                            best_is_result = is_result
                            best_oos_result = oos_result
                    except Exception:
                        continue

    return {
        "symbol": symbol,
        "config": best_config,
        "is_sharpe": best_is_result.get("sharpe", np.nan),
        "oos_sharpe": best_oos_result.get("sharpe", np.nan),
        "is_return_pct": best_is_result.get("return_pct", np.nan),
        "oos_return_pct": best_oos_result.get("return_pct", np.nan),
        "is_trades": best_is_result.get("trades", 0),
        "oos_trades": best_oos_result.get("trades", 0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="RF1.1: Auto-discover per-instrument optimal config"
    )
    parser.add_argument(
        "--symbols", required=True, help="Comma-separated tickers (SPY,QQQ,IWM,...)"
    )
    parser.add_argument("--fast", action="store_true", help="Use fast grid (3×2×2×2 = 24 combos)")
    parser.add_argument("--full", action="store_true", help="Use full grid (6×6×5×4 = 720 combos)")
    parser.add_argument("--start", default=DEFAULT_START, help="IS start date")
    parser.add_argument("--oos-start", default=DEFAULT_OOS_START, help="OOS start date")
    parser.add_argument("--end", default=DEFAULT_END, help="End date")
    parser.add_argument("--json-output", help="Save results to JSON file")
    args = parser.parse_args()

    grid = GRID_FAST if args.fast or not args.full else GRID_FULL
    symbols = [s.strip() for s in args.symbols.split(",")]

    print(
        f"Auto-tuning {len(symbols)} instruments with {len(grid['entry_thresholds'])}×"
        f"{len(grid['min_reliability'])}×{len(grid['trail_stop_atr'])}×{len(grid['confluence_bonus'])} "
        f"= {np.prod([len(v) for v in grid.values()])} combos"
    )
    print(f"IS: {args.start}→{args.oos_start} | OOS: {args.oos_start}→{args.end}\n")

    results: List[dict] = []
    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Tuning {sym} ...")
        r = auto_tune_instrument(sym, grid, args.start, args.oos_start, args.end)
        results.append(r)
        print(
            f"  → Best: et={r['config']['entry_threshold']} mr={r['config']['min_reliability']} "
            f"tsa={r['config']['trail_stop_atr']} cb={r['config']['confluence_bonus']} | "
            f"IS Sharpe={r['is_sharpe']:.3f} OOS Sharpe={r['oos_sharpe']:.3f}"
        )

    print("\n" + "=" * 80)
    print("PER-INSTRUMENT OPTIMAL CONFIGS")
    print("=" * 80)
    for r in results:
        print(
            f"{r['symbol']:8s} et={r['config']['entry_threshold']:.2f} "
            f"mr={r['config']['min_reliability']:.2f} "
            f"tsa={r['config']['trail_stop_atr']:.1f} "
            f"cb={r['config']['confluence_bonus']:.2f} | "
            f"IS Sharpe={r['is_sharpe']:+.3f} OOS Sharpe={r['oos_sharpe']:+.3f} "
            f"OOS Trades={r['oos_trades']}"
        )

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump({"tuned_at": datetime.now().isoformat(), "results": results}, f, indent=2)
        print(f"\nSaved to {args.json_output}")


if __name__ == "__main__":
    main()
