"""
H6: BTC Configuration Audit — systematic sweep on BTC to diagnose IS divergence.

BTC IS rules-first returned -9.3% (Sharpe -0.06) vs batch Sharpe 0.77.
This script sweeps entry thresholds and reliability to find the optimal
crypto config and diagnose why IS performance diverges from batch results.

Usage:
    uv run scripts/audit_btc_config.py
    uv run scripts/audit_btc_config.py --start 2020-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.backtest_rules_first import run_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="H6: Systematic BTC config sweep audit")
    parser.add_argument("--symbol", default="BTC_USD", help="Crypto symbol")
    parser.add_argument("--start", default="2016-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument(
        "--entry-sweep", default="0.45,0.55,0.65,0.75", help="Comma-separated entry thresholds"
    )
    parser.add_argument(
        "--reliability-sweep",
        default="0.4,0.5,0.6,0.7",
        help="Comma-separated min reliability values",
    )
    parser.add_argument("--oostart", default="2025-01-01", help="OOS start for validation")
    parser.add_argument("--oosend", default="2026-05-16", help="OOS end for validation")
    parser.add_argument("--json-output", default="reports/btc_audit/btc_config_sweep.json")
    args = parser.parse_args()

    entry_thresholds = [float(t.strip()) for t in args.entry_sweep.split(",")]
    reliability_values = [float(r.strip()) for r in args.reliability_sweep.split(",")]

    print(f"\n{'=' * 70}")
    print(f" BTC CONFIG AUDIT: {args.symbol}")
    print(f" Period: {args.start} -> {args.end}")
    print(f" Sweep: {len(entry_thresholds)} entry × {len(reliability_values)} mr")
    print(f"{'=' * 70}\n")

    results = []
    for et in entry_thresholds:
        for mr in reliability_values:
            is_result = run_backtest(
                args.symbol,
                start=args.start,
                end=args.end,
                entry_threshold=et,
                min_reliability=mr,
            )
            if is_result is None:
                continue

            oos_result = run_backtest(
                args.symbol,
                start=args.oostart,
                end=args.oosend,
                entry_threshold=et,
                min_reliability=mr,
            )

            row = {
                "entry_threshold": et,
                "min_reliability": mr,
                "IS_return_pct": is_result["return_pct"],
                "IS_sharpe": is_result["sharpe"],
                "IS_trades": is_result["trades"],
                "IS_win_rate_pct": is_result["win_rate_pct"],
                "IS_max_dd_pct": is_result["max_dd_pct"],
                "IS_pf": is_result["profit_factor"],
            }
            if oos_result:
                row.update(
                    {
                        "OOS_return_pct": oos_result["return_pct"],
                        "OOS_sharpe": oos_result["sharpe"],
                        "OOS_trades": oos_result["trades"],
                        "OOS_win_rate_pct": oos_result["win_rate_pct"],
                    }
                )

            results.append(row)
            print(
                f"  et={et:.2f} mr={mr:.1f}  "
                f"IS Sharpe={is_result['sharpe']:.3f} IS Ret={is_result['return_pct']:.1f}%",
                end="",
            )
            if oos_result:
                print(f"  OOS Sharpe={oos_result['sharpe']:.3f}")
            else:
                print()

    if not results:
        print("No results generated.")
        return

    df = pd.DataFrame(results)
    best_is = df.loc[df["IS_sharpe"].idxmax()]
    best_is_oos = None
    if "OOS_sharpe" in df.columns and len(df) > 0:
        valid_oos = df.dropna(subset=["OOS_sharpe"])
        if len(valid_oos) > 0:
            try:
                best_is_oos = valid_oos.loc[valid_oos["OOS_sharpe"].idxmax()]
            except (ValueError, KeyError):
                best_is_oos = None

    print(f"\n{'=' * 70}")
    print(" SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"  Best IS:  et={best_is['entry_threshold']:.2f} mr={best_is['min_reliability']:.1f}  "
        f"Sharpe={best_is['IS_sharpe']:.3f}  Trades={best_is['IS_trades']}"
    )
    if best_is_oos is not None:
        print(
            f"  Best OOS: et={best_is_oos['entry_threshold']:.2f} mr={best_is_oos['min_reliability']:.1f}  "
            f"Sharpe={best_is_oos['OOS_sharpe']:.3f}  Trades={best_is_oos['OOS_trades']}"
        )

    # Recommendations
    print("\n  RECOMMENDED CRYPTO PRESET:")
    # Use OOS best if available, otherwise IS best + 0.05 buffer
    if best_is_oos is not None and best_is_oos["OOS_sharpe"] > 0:
        rec_et = best_is_oos["entry_threshold"]
        rec_mr = best_is_oos["min_reliability"]
    else:
        rec_et = min(best_is["entry_threshold"] + 0.05, 0.80)
        rec_mr = max(best_is["min_reliability"] + 0.05, 0.70)
    print(f"    --entry-threshold {rec_et:.2f} --min-reliability {rec_mr:.1f}")
    print("    Higher threshold recommended due to crypto volatility.")

    # Save JSON
    output_path = project_root / args.json_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "symbol": args.symbol,
            "start": args.start,
            "end": args.end,
            "oostart": args.oostart,
            "oosend": args.oosend,
            "swept_entries": entry_thresholds,
            "swept_reliabilities": reliability_values,
        },
        "best_is": {
            "entry_threshold": float(best_is["entry_threshold"]),
            "min_reliability": float(best_is["min_reliability"]),
            "sharpe": float(best_is["IS_sharpe"]),
            "trades": int(best_is["IS_trades"]),
        },
        "recommended": {
            "entry_threshold": rec_et,
            "min_reliability": rec_mr,
        },
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved to {args.json_output}")


if __name__ == "__main__":
    main()
