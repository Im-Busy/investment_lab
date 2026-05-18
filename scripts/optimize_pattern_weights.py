"""
B4: Optimize pattern reliability weights from ablation solo results.

Loads ablation results, maps empirical Sharpe contributions to
adjusted reliability weights, and outputs updated PATTERN_RELIABILITY dict.

Usage:
    uv run scripts/optimize_pattern_weights.py --ablation-file reports/ablation/solo_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.rules_first_strategy import PATTERN_RELIABILITY as DEFAULT_WEIGHTS


def load_ablation_results(path: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df


def map_pattern_name(ablation_name: str) -> str:
    """Map ablation pattern names to PATTERN_RELIABILITY keys."""
    mapping = {
        "Matching Lows": "Matching Lows",
        "Market Structure Low": "Market Structure Low",
        "Triple Bottom": "Triple Bottom",
        "Double Bottom": "Double Bottom",
        "NR7ID": "NR7ID",
        "Piercing Line": "Piercing Line",
        "Bollinger Bands": "Bollinger Bands",
        "Symmetric Triangle": "Symmetric Triangle",
        "ABC Pattern": "ABC Pattern",
        "Hammer": "Hammer",
        "Donchian Channel Breakout": "Donchian Channel Breakout",
        "Cup and Handle": "Cup and Handle",
        "n-Bar Decline": "N-Bar Decline",
        "Double Top": "Double Top",
        "Triple Top": "Triple Top",
        "Dark Cloud Cover": "Dark Cloud Cover",
        "Floor Pivot Breakout": "Floor Pivot Breakout",
        "Engulfing": "Engulfing",
        "Head and Shoulders": "Head and Shoulders",
        "Two-Bar Reversal": "Two Bar Reversal",
        "Gartley Pattern": "Gartley Pattern",
        "Spike and Ledge": "Spike and Ledge",
        "Three Hills and Mountain": "Three Hills and a Mountain",
        "Parabolic Arc": "Parabolic Arc",
        "Trader Vic's 2B": "Trader Vic 2B",
        "Ascending Triangle": "Ascending Triangle",
        "Descending Triangle": "Descending Triangle",
        "Rectangle": "Rectangle",
        "Wedge": "Wedge",
        "Dead Cat Bounce": "Dead Cat Bounce",
        "Flag": "Flag",
        "Pennant": "Pennant",
        "Doji": "Doji",
        "Harami": "Harami",
    }
    return mapping.get(ablation_name, ablation_name)


def compute_empirical_weight(sharpe: float | None, trades: int) -> float:
    """Convert empirical Sharpe to reliability weight [0, 1].

    Only compute if patterns have sufficient trades to be meaningful.
    """
    if trades < 3 or sharpe is None:
        return 0.0
    # Map Sharpe to weight: 0.0 Sharpe -> 0.30, 1.0 Sharpe -> 0.80
    weight = 0.30 + 0.50 * min(max(sharpe, 0.0), 1.0)
    return round(weight, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize pattern weights from ablation data")
    parser.add_argument(
        "--ablation-file",
        default="reports/ablation/solo_results.json",
        help="Path to solo_results.json",
    )
    args = parser.parse_args()

    df = load_ablation_results(args.ablation_file)

    print(f"Loaded {len(df)} patterns from ablation\n")
    print(
        f"{'Pattern':30s} {'Trades':>6s} {'Sharpe':>8s} {'Return%':>8s} "
        f"{'DefaultW':>8s} {'EmpirW':>8s} {'Change':>8s}"
    )
    print("-" * 80)

    adjusted: dict[str, float] = {}
    updates: list[dict] = []

    for _, row in df.iterrows():
        name = map_pattern_name(row["pattern_name"])
        trades = int(row["total_trades"])
        sharpe = row["sharpe_ratio"]
        ret = row["total_return_pct"]
        default_w = DEFAULT_WEIGHTS.get(name, 0.0)
        empir_w = compute_empirical_weight(sharpe, trades)

        # Only use empirical if the pattern actually traded enough
        if trades >= 3 and empir_w > 0:
            # Blend: 70% default + 30% empirical (default has more stability)
            blended = 0.7 * default_w + 0.3 * empir_w
        else:
            blended = default_w

        adjusted[name] = round(blended, 2)
        change = blended - default_w

        flag = ""
        if trades >= 3 and abs(change) > 0.02:
            flag = " *"

        print(
            f"{name:30s} {trades:>6d} {sharpe or 0:>8.3f} {ret or 0:>8.1f} "
            f"{default_w:>8.2f} {blended:>8.2f} {change:>+8.2f}{flag}"
        )

        updates.append(
            {
                "name": name,
                "trades": trades,
                "sharpe": sharpe,
                "default_weight": default_w,
                "adjusted_weight": blended,
            }
        )

    print("\nAdjusted PATTERN_RELIABILITY dict:")
    print("```python")
    print("PATTERN_RELIABILITY: dict[str, float] = {")
    for name, w in sorted(adjusted.items(), key=lambda x: x[1], reverse=True):
        print(f'    "{name}": {w:.2f},')
    print("}")
    print("```")

    # Print summary
    changed = [u for u in updates if abs(u["adjusted_weight"] - u["default_weight"]) > 0.01]
    if changed:
        print(f"\n{len(changed)} weights changed by >0.01:")
        for u in sorted(
            changed, key=lambda x: abs(x["adjusted_weight"] - x["default_weight"]), reverse=True
        ):
            print(
                f"  {u['name']}: {u['default_weight']:.2f} -> {u['adjusted_weight']:.2f} "
                f"({u['adjusted_weight'] - u['default_weight']:+.2f}) [trades={u['trades']}, Sharpe={u['sharpe']:.3f}]"
            )


if __name__ == "__main__":
    main()
