#!/usr/bin/env python3
"""RF4.2: Generate per-instrument config cards.

Reads auto_tune results (JSON) and outputs markdown config cards
with optimal et/mr/tsa/cb per ticker.

Usage:
    uv run scripts/generate_config_cards.py --json outputs/auto_tune.json
    uv run scripts/generate_config_cards.py --json outputs/auto_tune.json --output reports/config_cards.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def generate_config_markdown(results: list[dict]) -> str:
    """Generate per-instrument config cards in markdown."""
    lines = [
        "# Per-Instrument Configuration Cards",
        f"\n*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
        "| Ticker | et | mr | tsa | cb | IS Sharpe | OOS Sharpe | OOS Return | OOS Trades | OOS Win% |",
        "|--------|----|-----|-----|----|-----------|------------|------------|------------|----------|",
    ]

    for r in results:
        cfg = r.get("config", {})
        lines.append(
            f"| {r['symbol']:6s} | {cfg.get('entry_threshold', '-'):.2f} if isinstance(cfg.get('entry_threshold'), float) else str(cfg.get('entry_threshold', '-')) | "
            f"{cfg.get('min_reliability', '-'):.2f} if isinstance(cfg.get('min_reliability'), float) else str(cfg.get('min_reliability', '-')) | "
            f"{cfg.get('trail_stop_atr', '-'):.1f} if isinstance(cfg.get('trail_stop_atr'), float) else str(cfg.get('trail_stop_atr', '-')) | "
            f"{cfg.get('confluence_bonus', '-'):.2f} if isinstance(cfg.get('confluence_bonus'), float) else str(cfg.get('confluence_bonus', '-')) | "
            f"{r.get('is_sharpe', '-'):.3f} | "
            f"{r.get('oos_sharpe', '-'):.3f} | "
            f"{r.get('oos_return_pct', '-'):.1f}% | "
            f"{r.get('oos_trades', '-')} | "
            f"{r.get('oos_win_rate', '-')} | "
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="RF4.2: Generate per-instrument config cards")
    parser.add_argument("--json", required=True, help="auto_tune JSON results file")
    parser.add_argument(
        "--output", default="reports/per_instrument_configs.md", help="Output markdown file"
    )
    args = parser.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    results = data.get("results", data)
    md = generate_config_markdown(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")

    print(f"Generated config cards for {len(results)} instruments")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
