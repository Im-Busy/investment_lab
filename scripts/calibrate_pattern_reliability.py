"""
C17: Empirical Pattern Reliability Calibration.

Backtests each pattern detector solo on SPY 20-year data and replaces
literature-default PATTERN_RELIABILITY weights with empirically calibrated
values based on actual win rates and profit factors.

Usage:
    # Full calibration on SPY 2016-2024
    uv run scripts/calibrate_pattern_reliability.py --ticker SPY --start 2016-01-01 --end 2024-12-31

    # IS only (2016-2024 training period) — recommended for weight updates
    uv run scripts/calibrate_pattern_reliability.py --ticker SPY --start 2016-01-01 --end 2024-12-31 --apply

    # Quick test (top 10 patterns only)
    uv run scripts/calibrate_pattern_reliability.py --ticker SPY --quick --output reports/calibration/quick.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CalibratedWeight:
    pattern_name: str
    literature_weight: float
    empirical_win_rate: float
    empirical_profit_factor: float
    empirical_sharpe: float
    empirical_trades: int
    calibrated_weight: float
    calibration_source: str  # "empirical" or "literature_fallback"


@dataclass
class CalibrationResult:
    weights: dict[str, float]
    details: list[CalibratedWeight]
    n_empirical: int
    n_fallback: int
    timestamp: str


PATTERN_CATEGORIES: dict[str, list[str]] = {
    "harmonic": [
        "Gartley Pattern",
        "ABC Pattern",
        "Symmetric Triangle",
        "Bollinger Bands",
        "Butterfly Pattern",
        "Bat Pattern",
        "Crab Pattern",
        "Cypher Pattern",
        "Shark Pattern",
    ],
    "classic_reversal": [
        "Head and Shoulders",
        "Inverse Head and Shoulders",
        "Double Top",
        "Double Bottom",
        "Triple Top",
        "Triple Bottom",
        "Trader Vic 2B",
    ],
    "classic_continuation": [
        "Ascending Triangle",
        "Descending Triangle",
        "Rectangle",
        "Wedge",
    ],
    "complex": [
        "Cup and Handle",
        "Spike and Ledge",
        "Three Hills and a Mountain",
        "Pipe Pattern",
        "Parabolic Arc",
        "Dead Cat Bounce",
    ],
    "breakout": [
        "Donchian Channel Breakout",
        "Gap Pattern",
    ],
    "continuation": [
        "Bull Flag",
        "Bear Flag",
        "Pennant",
        "Flag",
    ],
    "basic": [
        "Market Structure Low",
        "Market Structure High",
        "Matching Lows",
        "NR7ID",
        "N-Bar Decline",
        "Floor Pivot Breakout",
        "Two Bar Reversal",
    ],
    "candlestick": [
        "Doji",
        "Harami",
        "Hammer",
        "Engulfing",
        "Dark Cloud Cover",
        "Piercing Line",
    ],
    "fmz": [
        "Alpha Beast",
        "Multi-Factor Trend",
        "Momentum ZigZag",
        "EMA-MACD HF",
        "Adaptive Bollinger",
        "AI Volatility Breakout",
    ],
}


def _load_data(ticker: str) -> pd.DataFrame:
    path = Path(f"data/raw/{ticker}_daily.csv")
    if not path.exists():
        raise FileNotFoundError(f"No data for {ticker} at {path}")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0.0 if col == "Volume" else df.iloc[:, 0]
    return df.dropna()


def _run_solo_backtests(
    df: pd.DataFrame,
    cash: float = 100_000,
    commission: float = 0.001,
    max_workers: int = 4,
    quick: bool = False,
) -> pd.DataFrame:
    from src.analysis.ablation_engine import AblationEngine
    from src.strategies.backtest_py.multi_pattern_strategy_optimized import (
        MultiPatternStrategyOptimized,
    )

    engine = AblationEngine(
        data=df,
        strategy_class=MultiPatternStrategyOptimized,
        cash=cash,
        commission=commission,
        output_dir="reports/ablation",
        max_workers=max_workers,
        verbose=True,
        quick_test=quick,
        quick_test_patterns=10 if quick else 100,
    )

    solo_df = engine.run_all_solo_backtests()
    return solo_df


def _calibrate_weight(
    win_rate: float,
    profit_factor: float,
    n_trades: int,
    literature_weight: float,
    min_trades: int = 3,
) -> tuple[float, str]:
    if n_trades < min_trades:
        return literature_weight, "literature_fallback"

    composite_score = 0.6 * win_rate + 0.4 * min(profit_factor / 3.0, 1.0)
    calibrated = max(composite_score, 0.30)
    calibrated = min(calibrated, 0.90)
    calibrated = round(calibrated, 2)

    return calibrated, "empirical"


def calibrate(
    df: pd.DataFrame,
    ticker: str,
    max_workers: int = 4,
    quick: bool = False,
) -> CalibrationResult:
    from src.strategies.rules_first_strategy import PATTERN_RELIABILITY

    logger.info(f"Running solo backtests for {ticker} ({'quick' if quick else 'full'})...")
    solo_df = _run_solo_backtests(df, max_workers=max_workers, quick=quick)
    logger.info(f"Solo backtests complete: {len(solo_df)} patterns tested")

    details: list[CalibratedWeight] = []
    weights: dict[str, float] = {}
    n_empirical = 0
    n_fallback = 0

    for _, row in solo_df.iterrows():
        name = str(row.get("pattern_name", ""))
        wr = float(row.get("win_rate", 0)) / 100.0
        pf = float(row.get("profit_factor", 0))
        sharpe = float(row.get("sharpe_ratio", 0))
        trades = int(row.get("total_trades", 0))
        lit_weight = PATTERN_RELIABILITY.get(name, 0.55)

        cal_weight, source = _calibrate_weight(wr, pf, trades, lit_weight)
        weights[name] = cal_weight

        if source == "empirical":
            n_empirical += 1
        else:
            n_fallback += 1

        details.append(
            CalibratedWeight(
                pattern_name=name,
                literature_weight=lit_weight,
                empirical_win_rate=round(wr, 3),
                empirical_profit_factor=round(pf, 2),
                empirical_sharpe=round(sharpe, 3),
                empirical_trades=trades,
                calibrated_weight=cal_weight,
                calibration_source=source,
            )
        )

    for lit_name, lit_w in PATTERN_RELIABILITY.items():
        if lit_name not in weights:
            weights[lit_name] = lit_w
            details.append(
                CalibratedWeight(
                    pattern_name=lit_name,
                    literature_weight=lit_w,
                    empirical_win_rate=0.0,
                    empirical_profit_factor=0.0,
                    empirical_sharpe=0.0,
                    empirical_trades=0,
                    calibrated_weight=lit_w,
                    calibration_source="literature_fallback",
                )
            )
            n_fallback += 1

    return CalibrationResult(
        weights=weights,
        details=details,
        n_empirical=n_empirical,
        n_fallback=n_fallback,
        timestamp=datetime.now().isoformat(),
    )


def _print_report(result: CalibrationResult) -> None:
    details = sorted(result.details, key=lambda d: d.calibrated_weight, reverse=True)

    print(f"\n{'=' * 80}")
    print("  Empirical Pattern Reliability Calibration")
    print(f"  Patterns calibrated empirically: {result.n_empirical}")
    print(f"  Patterns kept at literature default: {result.n_fallback}")
    print(f"{'=' * 80}")

    print(
        f"\n{'Pattern':<32} {'Lit':>5} {'WR':>6} {'Trades':>6} {'PF':>6} {'Sharpe':>7} {'Cal':>5} {'Src':<10}"
    )
    print("-" * 85)

    by_cat = _group_by_category(details)
    for cat, items in by_cat.items():
        print(f"\n  [{cat}]")
        for d in items:
            src = "empirical" if d.calibration_source == "empirical" else "lit"
            print(
                f"  {d.pattern_name:<30} {d.literature_weight:>5.2f} "
                f"{d.empirical_win_rate:>6.1%} {d.empirical_trades:>5d} "
                f"{d.empirical_profit_factor:>6.2f} {d.empirical_sharpe:>7.3f} "
                f"{d.calibrated_weight:>5.2f} {src:>10}"
            )

    print(f"\n{'=' * 80}")
    print("  Biggest changes from literature -> empirical:")
    changes = sorted(
        details,
        key=lambda d: abs(d.calibrated_weight - d.literature_weight),
        reverse=True,
    )[:10]
    for d in changes:
        delta = d.calibrated_weight - d.literature_weight
        arrow = "^" if delta > 0 else "v" if delta < 0 else "="
        print(
            f"  {d.pattern_name:<30} {d.literature_weight:.2f} -> {d.calibrated_weight:.2f} ({arrow}{abs(delta):.2f})"
        )

    print("\n  Calibrated PATTERN_RELIABILITY dict:")
    print("  PATTERN_RELIABILITY: dict[str, float] = {")
    grouped = _group_by_category(details)
    for cat, items in grouped.items():
        print(f"      # {cat}")
        for d in items:
            print(f'      "{d.pattern_name}": {d.calibrated_weight},')
    print("  }")
    print()


def _group_by_category(details: list[CalibratedWeight]) -> dict[str, list[CalibratedWeight]]:
    detail_map = {d.pattern_name: d for d in details}
    result: dict[str, list[CalibratedWeight]] = {}
    for cat, names in PATTERN_CATEGORIES.items():
        items = [detail_map[n] for n in names if n in detail_map]
        if items:
            result[cat] = items
    return result


def _update_source_files(calibrated_weights: dict[str, float]) -> list[str]:
    updated: list[str] = []

    targets = [
        Path("src/strategies/rules_first_strategy.py"),
        Path("src/signals/pattern_boost.py"),
    ]

    for target in targets:
        if not target.exists():
            logger.warning(f"Target not found: {target}")
            continue

        content = target.read_text()
        original = content

        lines = content.split("\n")
        in_dict = False
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if "PATTERN_RELIABILITY" in line and ("dict" in line or "{" in line):
                new_lines.append(line)
                if "{" in line:
                    in_dict = True
                    indent = " " * (len(line) - len(line.lstrip()) + 4)
                else:
                    i += 1
                    while i < len(lines) and "{" not in lines[i]:
                        new_lines.append(lines[i])
                        i += 1
                    if i < len(lines):
                        new_lines.append(lines[i])
                        indent = " " * (len(lines[i]) - len(lines[i].lstrip()) + 4)
                        in_dict = True
                i += 1

                old_weights: dict[str, float] = {}
                while i < len(lines) and "}" not in lines[i]:
                    stripped = lines[i].strip().strip(",")
                    if ":" in stripped:
                        key = stripped.split(":", 1)[0].strip().strip('"').strip("'")
                        try:
                            val_str = stripped.split(":", 1)[1].strip().rstrip(",")
                            old_weights[key] = float(val_str)
                        except ValueError:
                            pass
                    i += 1

                for k, v in calibrated_weights.items():
                    old_weights[k] = v

                grouped = _group_by_category(
                    [
                        d
                        for d in [
                            CalibratedWeight(
                                pattern_name=k,
                                literature_weight=0.0,
                                empirical_win_rate=0.0,
                                empirical_profit_factor=0.0,
                                empirical_sharpe=0.0,
                                empirical_trades=0,
                                calibrated_weight=old_weights.get(k, 0.55),
                                calibration_source="empirical",
                            )
                            for k in old_weights
                        ]
                    ]
                )

                for cat, items in grouped.items():
                    new_lines.append(f"{indent}# {cat}")
                    for d in items:
                        new_lines.append(f'{indent}"{d.pattern_name}": {d.calibrated_weight},')

                new_lines.append(indent[:-4] + "}")

                while i < len(lines) and lines[i].strip() == "}":
                    i += 1
            else:
                new_lines.append(line)
                i += 1

        new_content = "\n".join(new_lines)
        if new_content != original:
            target.write_text(new_content)
            updated.append(str(target))

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="C17: Empirical pattern reliability calibration")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker for calibration")
    parser.add_argument("--start", type=str, default="2016-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date")
    parser.add_argument("--quick", action="store_true", help="Quick test (10 patterns only)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument(
        "--apply", action="store_true", help="Update PATTERN_RELIABILITY in source files"
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    df = _load_data(args.ticker)
    if args.start:
        df = df[df.index >= pd.Timestamp(args.start)]
    if args.end:
        df = df[df.index <= pd.Timestamp(args.end)]
    logger.info(f"Data: {len(df)} bars ({df.index[0]} -> {df.index[-1]})")

    result = calibrate(df, args.ticker, max_workers=args.workers, quick=args.quick)

    _print_report(result)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "timestamp": result.timestamp,
                    "n_empirical": result.n_empirical,
                    "n_fallback": result.n_fallback,
                    "weights": result.weights,
                    "details": [
                        {
                            "pattern_name": d.pattern_name,
                            "literature_weight": d.literature_weight,
                            "calibrated_weight": d.calibrated_weight,
                            "empirical_win_rate": d.empirical_win_rate,
                            "empirical_trades": d.empirical_trades,
                            "empirical_sharpe": d.empirical_sharpe,
                            "calibration_source": d.calibration_source,
                        }
                        for d in result.details
                    ],
                },
                f,
                indent=2,
            )
        logger.info(f"Saved calibration to {path}")

    if args.apply:
        updated = _update_source_files(result.weights)
        if updated:
            logger.info(f"Updated {len(updated)} source files: {', '.join(updated)}")
        else:
            logger.warning("No files updated")

        pf_path = Path("scripts/paper_trade_wf_honest.py")
        if pf_path.exists():
            content = pf_path.read_text()
            old_content = content
            for k, v in result.weights.items():
                import re

                pattern = re.compile(rf'("{re.escape(k)}":\s*)\d+\.\d+')
                content = pattern.sub(rf"\g<1>{v}", content)
            if content != old_content:
                pf_path.write_text(content)
                updated.append(str(pf_path))
                logger.info(f"Updated {pf_path}")

        comb_path = Path("src/strategies/combined_strategy.py")
        if comb_path.exists():
            content = comb_path.read_text()
            old_content = content
            for k, v in result.weights.items():
                import re

                pattern = re.compile(rf'("{re.escape(k)}":\s*)\d+\.\d+')
                content = pattern.sub(rf"\g<1>{v}", content)
            if content != old_content:
                comb_path.write_text(content)
                updated.append(str(comb_path))
                logger.info(f"Updated {comb_path}")


if __name__ == "__main__":
    main()
