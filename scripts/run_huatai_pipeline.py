#!/usr/bin/env python3
"""
CLI for the Huatai 4-phase optimization pipeline (R13).

Usage:
    uv run scripts/run_huatai_pipeline.py SPY QQQ IWM --start 2020-01-01 --end 2024-12-31
    uv run scripts/run_huatai_pipeline.py SPY QQQ XLK GLD --risk-cap 0.12 --max-weight 0.30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_ingestion.data_fetcher import fetch_ohlcv
from src.optimization.huatai_pipeline import (
    HuataiPipeline,
    HuataiResult,
    run_huatai_pipeline,
    DEFAULT_RISK_CAP,
    DEFAULT_MAX_WEIGHT,
)
from src.ml.expected_returns import HP_LAMBDA_DAILY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("huatai_cli")


def generate_synthetic_signals(prices: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic pattern signals from price momentum for testing."""
    rng = np.random.default_rng(seed)
    signals = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for col in prices.columns:
        ret = prices[col].pct_change().fillna(0)
        sma_signal = ret.rolling(20).mean() * 10
        noise = rng.normal(0, 0.1, len(prices))
        signals[col] = sma_signal + noise

    return signals.fillna(0)


def format_weights_table(result: HuataiResult) -> str:
    """Format weights as an aligned text table."""
    lines = ["  Asset          Weight   Exp Return   HF Trend"]
    lines.append("  ─────────────  ──────   ──────────   ────────")
    sorted_items = sorted(result.weights.items(), key=lambda x: x[1], reverse=True)
    for name, w in sorted_items:
        er = result.expected_returns.get(name, 0.0)
        # HP trend: last value or slope
        hp_val = "—"
        if result.hp_trend and name in result.hp_trend:
            trend_arr = result.hp_trend[name]
            if len(trend_arr) > 0:
                hp_val = f"{trend_arr[-1]:.4f}"
        lines.append(f"  {name:<13}  {w:6.4f}   {er:+9.4f}   {hp_val:>8}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Huatai 4-Phase Optimization Pipeline (华泰多因子 §2.5-4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run scripts/run_huatai_pipeline.py SPY QQQ IWM --start 2020-01-01
  uv run scripts/run_huatai_pipeline.py SPY QQQ XLK GLD TLT --risk-cap 0.10
  uv run scripts/run_huatai_pipeline.py --signals-file signals.csv --returns-file returns.csv --prices-file prices.csv
        """,
    )

    parser.add_argument("symbols", nargs="*", help="Asset symbols to optimize (fetches OHLCV)")
    parser.add_argument("--signals-file", type=Path, help="CSV with signal scores (T x N)")
    parser.add_argument("--returns-file", type=Path, help="CSV with returns (T x N)")
    parser.add_argument("--prices-file", type=Path, help="CSV with prices (T x N)")
    parser.add_argument("--start", default="2020-01-01", help="Start date (default: 2020-01-01)")
    parser.add_argument("--end", default="2024-12-31", help="End date (default: 2024-12-31)")
    parser.add_argument(
        "--risk-cap",
        type=float,
        default=DEFAULT_RISK_CAP,
        help=f"Annualized volatility cap (default: {DEFAULT_RISK_CAP})",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=DEFAULT_MAX_WEIGHT,
        help=f"Max weight per asset (default: {DEFAULT_MAX_WEIGHT})",
    )
    parser.add_argument(
        "--min-weight", type=float, default=0.0, help="Min weight per asset (default: 0.0)"
    )
    parser.add_argument(
        "--ir-window", type=int, default=252, help="Rolling IR window (default: 252)"
    )
    parser.add_argument(
        "--hp-lambda",
        type=float,
        default=HP_LAMBDA_DAILY,
        help=f"HP filter lambda (default: {HP_LAMBDA_DAILY})",
    )
    parser.add_argument("--json-output", type=Path, help="Save results to JSON file")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for synthetic signals (default: 42)"
    )

    args = parser.parse_args()

    # ── Load data ──
    if args.prices_file:
        logger.info("Loading prices from %s", args.prices_file)
        prices = pd.read_csv(args.prices_file, index_col=0, parse_dates=True)
    elif args.symbols:
        logger.info("Fetching OHLCV for %d symbols: %s", len(args.symbols), args.symbols)
        prices_dict: dict[str, pd.Series] = {}
        for sym in args.symbols:
            try:
                df = fetch_ohlcv(sym, args.start, args.end, interval="1d")
                prices_dict[sym] = df["Close"]
            except Exception as exc:
                logger.warning("Failed to fetch %s: %s", sym, exc)

        if not prices_dict:
            logger.error("No price data fetched. Exiting.")
            sys.exit(1)

        prices = pd.DataFrame(prices_dict)
    else:
        # Demo mode with synthetic data
        logger.info("No symbols or files provided. Running demo with synthetic data.")
        dates = pd.date_range(args.start, args.end, freq="B")
        rng = np.random.default_rng(args.seed)
        prices = pd.DataFrame(
            {
                "Asset_A": 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, len(dates)))),
                "Asset_B": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(dates)))),
                "Asset_C": 100 * np.exp(np.cumsum(rng.normal(0.0007, 0.018, len(dates)))),
                "Asset_D": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.010, len(dates)))),
                "Asset_E": 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.020, len(dates)))),
            },
            index=dates,
        )

    logger.info("Price data: %d bars × %d assets", len(prices), len(prices.columns))

    # ── Compute returns ──
    if args.returns_file:
        returns = pd.read_csv(args.returns_file, index_col=0, parse_dates=True)
    else:
        returns = prices.pct_change().fillna(0.0)

    # ── Generate/load signals ──
    if args.signals_file:
        signals = pd.read_csv(args.signals_file, index_col=0, parse_dates=True)
    else:
        signals = generate_synthetic_signals(prices, seed=args.seed)

    # ── Run pipeline ──
    logger.info(
        "Running Huatai pipeline with risk_cap=%.4f, max_weight=%.4f",
        args.risk_cap,
        args.max_weight,
    )

    pipeline = HuataiPipeline(
        risk_cap=args.risk_cap,
        max_weight=args.max_weight,
        min_weight=args.min_weight,
        hp_lambda=args.hp_lambda,
    )

    result = pipeline.optimize(signals, returns, prices, ir_window=args.ir_window)

    # ── Print results ──
    print(f"\n{'=' * 60}")
    print("  Huatai 4-Phase Optimization Pipeline (华泰多因子 §2.5-4)")
    print(f"{'=' * 60}")
    print(f"\n  Status: {result.status.upper()}")
    print(f"  Iterations: {result.n_iterations}")
    if result.message:
        print(f"  Message: {result.message}")

    print("\n  Portfolio Metrics:")
    print(
        f"    Expected Return:  {result.portfolio_return:+.4f}  ({result.portfolio_return * 100:+.2f}%)"
    )
    print(
        f"    Portfolio Risk:   {result.portfolio_risk:.4f}  ({result.portfolio_risk * 100:.2f}%)"
    )
    print(f"    Sharpe Ratio:     {result.portfolio_sharpe:+.4f}")

    print("\n  Optimized Weights:")
    print(format_weights_table(result))

    # IR signal summary
    if result.ir_signal:
        print("\n  IR Signals (Phase 1):")
        for name, ir in sorted(result.ir_signal.items(), key=lambda x: x[1], reverse=True):
            print(f"    {name:<13}  {ir:+.6f}")

    # Constraint violations
    if result.constraint_violations:
        print("\n  ⚠ Constraint Violations:")
        for v in result.constraint_violations:
            print(f"    - {v}")

    print(f"\n{'=' * 60}")

    # ── Save JSON ──
    output_path = args.json_output
    if output_path:
        output_dict = {
            "timestamp": datetime.now().isoformat(),
            "status": result.status,
            "weights": result.weights,
            "expected_returns": result.expected_returns,
            "portfolio_return": result.portfolio_return,
            "portfolio_risk": result.portfolio_risk,
            "portfolio_sharpe": result.portfolio_sharpe,
            "ir_signals": result.ir_signal,
            "n_iterations": result.n_iterations,
            "constraint_violations": result.constraint_violations,
            "config": {
                "risk_cap": args.risk_cap,
                "max_weight": args.max_weight,
                "min_weight": args.min_weight,
                "ir_window": args.ir_window,
                "hp_lambda": args.hp_lambda,
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_dict, f, indent=2)

        logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
