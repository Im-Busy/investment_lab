#!/usr/bin/env python3
"""RF4.1: Auto-update BESTS.md after every backtest.

Reads the latest backtest JSON output and updates BESTS.md leaderboard.
Enforces the protocol that every backtest must update the leaderboard.

Usage:
    uv run scripts/update_bests.py --json outputs/backtest_result.json
    uv run scripts/update_bests.py --json outputs/backtest_result.json --strategy rules-first
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def load_bests(bests_path: Path) -> str:
    """Load BESTS.md content."""
    if bests_path.exists():
        return bests_path.read_text(encoding="utf-8")
    return ""


def format_leaderboard_entry(
    strategy: str,
    config: str,
    return_pct: float,
    sharpe: float,
    trades: int,
    win_rate: float,
    profit_factor: float,
    max_dd: float,
    exposure: float = 100.0,
    ann_return: float | None = None,
) -> str:
    """Format a BESTS.md leaderboard table entry."""
    cols = [
        strategy,
        config,
        f"{return_pct:.1f}%",
        f"{sharpe:.2f}",
        str(trades),
        f"{win_rate:.0f}%",
        f"{profit_factor:.2f}" if profit_factor != float("inf") else "inf",
        f"{max_dd:.1f}%",
        f"{exposure:.0f}%",
        f"{ann_return:.1f}%" if ann_return is not None else "-",
    ]
    return "| " + " | ".join(cols) + " |"


def update_bests(
    bests_path: Path,
    result: dict,
    strategy: str = "rules-first",
    condition: str = "default",
) -> bool:
    """Update BESTS.md with a new backtest result.

    Returns True if the result sets a new best.
    """
    content = load_bests(bests_path)

    entry = format_leaderboard_entry(
        strategy=strategy,
        config=result.get("config", result.get("description", "default")),
        return_pct=result.get("return_pct", result.get("Return [%]", 0.0)),
        sharpe=result.get("sharpe", result.get("Sharpe Ratio", 0.0)),
        trades=result.get("trades", result.get("# Trades", 0)),
        win_rate=result.get("win_rate", result.get("Win Rate [%]", 0.0)),
        profit_factor=result.get("profit_factor", result.get("Profit Factor", 0.0)),
        max_dd=result.get("max_dd", result.get("Max. Drawdown [%]", 0.0)),
        exposure=result.get("exposure", result.get("Exposure Time [%]", 100.0)),
        ann_return=result.get("ann_return", result.get("Return (Ann.) [%]")),
    )

    new_section = (
        f"\n### {condition} (Updated {datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n"
        f"| Strategy | Config | Return | Sharpe | Trades | Win% | PF | MaxDD | Exposure | AnnRet |\n"
        f"|----------|--------|--------|--------|--------|------|----|-------|----------|--------|\n"
        f"{entry}\n"
    )

    if condition in content:
        # Insert after existing section header
        section_start = content.index(f"### {condition}")
        section_end = (
            content.find("\n##", section_start + 1)
            if content.find("\n##", section_start + 1) > 0
            else len(content)
        )
        content = content[:section_end] + f"\n{entry}\n" + content[section_end:]
    else:
        content += new_section

    timestamp_line = f"\n\n*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    if "*Last updated:" in content:
        content = content[: content.index("*Last updated:")] + timestamp_line
    else:
        content += timestamp_line

    bests_path.write_text(content, encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser(description="RF4.1: Auto-update BESTS.md")
    parser.add_argument("--json", required=True, help="JSON backtest result file")
    parser.add_argument("--strategy", default="rules-first", help="Strategy label")
    parser.add_argument("--condition", default="default", help="Condition category")
    parser.add_argument("--bests-path", default="BESTS.md", help="Path to BESTS.md")
    args = parser.parse_args()

    bests_path = Path(args.bests_path)
    with open(args.json) as f:
        result = json.load(f)

    # Handle single result vs results array
    if isinstance(result, dict) and "results" in result:
        for r in result["results"]:
            update_bests(bests_path, r, args.strategy, args.condition)
            print(f"Updated BESTS.md with {r.get('symbol', 'unknown')} result")
    else:
        updated = update_bests(bests_path, result, args.strategy, args.condition)
        print(f"{'Updated' if updated else 'Appended'} BESTS.md")

    print(f"BESTS.md last modified: {datetime.fromtimestamp(bests_path.stat().st_mtime)}")


if __name__ == "__main__":
    main()
