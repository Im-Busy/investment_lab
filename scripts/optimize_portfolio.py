#!/usr/bin/env python
"""Portfolio optimization CLI using eiten strategies.

Optimizes a multi-asset portfolio using Eigen, MVP, MSR, or GA strategies
with optional Random Matrix Theory covariance denoising.

Usage:
    uv run scripts/optimize_portfolio.py --symbols SPY QQQ TLT GLD XLK --strategy msr
    uv run scripts/optimize_portfolio.py --compare --symbols SPY QQQ TLT GLD XLK
    uv run scripts/optimize_portfolio.py --from-signals reports/batch/signals.json --strategy msr
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.portfolio.eiten_builder import EitenPortfolioBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("optimize_portfolio")


def fetch_data(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data and compute daily returns."""
    import yfinance as yf

    logger.info("Downloading data for %d symbols: %s", len(symbols), ", ".join(symbols))
    data = yf.download(symbols, start=start, end=end, auto_adjust=False, progress=False)

    if "Adj Close" in data.columns:
        close = data["Adj Close"]
    else:
        close = data["Close"]

    if isinstance(close, pd.Series):
        close = close.to_frame()

    returns = close.pct_change().dropna()
    returns.columns = [str(c) for c in returns.columns]
    return returns


def load_signals(path: Path) -> dict[str, float]:
    """Load signals from JSON file."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "signals" in data:
        return data["signals"]
    if isinstance(data, dict):
        return data
    raise ValueError(f"Cannot parse signals from {path}")


def format_weights_table(weights: dict[str, float]) -> str:
    """Format weights as a table with ASCII bar chart."""
    sorted_items = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
    lines = []
    for sym, w in sorted_items:
        bar = "#" * int(abs(w) * 50)
        sign = "+" if w >= 0 else "-"
        lines.append(f"  {sym:6s}  {sign}{abs(w):.4f}  {bar}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portfolio optimization using eiten strategies",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY", "QQQ", "TLT", "GLD", "XLK"],
        help="Asset symbols to include (default: SPY QQQ TLT GLD XLK)",
    )
    parser.add_argument(
        "--strategy",
        choices=("eigen", "mvp", "msr", "ga"),
        default="msr",
        help="Optimization strategy (default: msr)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run all 4 strategies and compare",
    )
    parser.add_argument(
        "--no-rmt",
        action="store_true",
        help="Disable RMT covariance denoising",
    )
    parser.add_argument(
        "--eigen-number",
        type=int,
        default=2,
        help="Eigen portfolio number (default: 2, orthogonal to market)",
    )
    parser.add_argument(
        "--ga-generations",
        type=int,
        default=50,
        help="GA generations (default: 50)",
    )
    parser.add_argument(
        "--ga-population",
        type=int,
        default=200,
        help="GA population size (default: 200)",
    )
    parser.add_argument(
        "--ga-seed",
        type=int,
        default=None,
        help="GA random seed for reproducibility",
    )
    parser.add_argument(
        "--start",
        default="2020-01-01",
        help="Start date for data (default: 2020-01-01)",
    )
    parser.add_argument(
        "--end",
        default="2025-12-31",
        help="End date for data (default: 2025-12-31)",
    )
    parser.add_argument(
        "--returns-csv",
        type=Path,
        default=None,
        help="Load returns from CSV instead of yfinance",
    )
    parser.add_argument(
        "--from-signals",
        type=Path,
        default=None,
        help="Load signals from JSON and use as expected return proxies",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    # Load data
    if args.returns_csv:
        returns_df = pd.read_csv(args.returns_csv, index_col=0, parse_dates=True)
    else:
        returns_df = fetch_data(args.symbols, args.start, args.end)

    symbols = [s for s in args.symbols if s in returns_df.columns]
    if not symbols:
        logger.error("No symbols found in data. Available: %s", list(returns_df.columns))
        sys.exit(1)

    builder = EitenPortfolioBuilder(
        use_rmt=not args.no_rmt,
        eigen_number=args.eigen_number,
        ga_population_size=args.ga_population,
        ga_generations=args.ga_generations,
        ga_seed=args.ga_seed,
    )

    if args.from_signals:
        signals = load_signals(args.from_signals)
        logger.info("Loaded %d signals from %s", len(signals), args.from_signals)
        if args.compare:
            results = {"base": signals}
        else:
            result = builder.optimize_from_signals(signals, returns_df, strategy=args.strategy)
            print(f"\nOptimized weights ({result.strategy}, RMT={result.rmt_applied}):")
            print(format_weights_table(result.weights))
            print(
                f"\nExpected return: {result.expected_return:.2%}" if result.expected_return else ""
            )
            print(
                f"Expected volatility: {result.expected_volatility:.2%}"
                if result.expected_volatility
                else ""
            )
            print(
                f"Expected Sharpe: {result.expected_sharpe:.2f}" if result.expected_sharpe else ""
            )
            if args.json_output:
                output = {
                    "strategy": result.strategy,
                    "weights": result.weights,
                    "expected_return": result.expected_return,
                    "expected_volatility": result.expected_volatility,
                    "expected_sharpe": result.expected_sharpe,
                    "rmt_applied": result.rmt_applied,
                }
                args.json_output.parent.mkdir(parents=True, exist_ok=True)
                args.json_output.write_text(json.dumps(output, indent=2, default=float))
                logger.info("Results saved to %s", args.json_output)
            return

    if args.compare:
        results = builder.compare(symbols, returns_df)
        print(f"\n{'=' * 70}")
        print("Portfolio Optimization Comparison")
        print(f"Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
        print(f"RMT denoising: {'ON' if not args.no_rmt else 'OFF'}")
        print(f"{'=' * 70}\n")

        for strategy, result in results.items():
            print(f"--- {strategy.upper()} ---")
            print(format_weights_table(result.weights))
            print(f"  Expected Sharpe: {result.expected_sharpe:.3f}")
            print(f"  Expected Return: {result.expected_return:.2%}")
            print(f"  Expected Vol:    {result.expected_volatility:.2%}")
            if result.rmt_n_kept is not None:
                print(
                    f"  RMT kept:        {result.rmt_n_kept}/{result.rmt_n_kept + (len(symbols) - result.rmt_n_kept)}"
                )
            print()
    else:
        result = builder.optimize(symbols, returns_df, strategy=args.strategy)
        print(f"\n{'=' * 50}")
        print(f"Portfolio: {args.strategy.upper()}")
        print(f"Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
        print(f"RMT denoising: {'ON' if not args.no_rmt else 'OFF'}")
        print(f"{'=' * 50}\n")
        print(format_weights_table(result.weights))
        print(f"\nExpected Sharpe: {result.expected_sharpe:.3f}")
        print(f"Expected Return: {result.expected_return:.2%}")
        print(f"Expected Volatility: {result.expected_volatility:.2%}")
        if result.rmt_n_kept is not None:
            n_noise = len(symbols) - result.rmt_n_kept
            print(
                f"RMT filter: {result.rmt_n_kept} signal eigenvalues kept, {n_noise} noise filtered"
            )
        if result.ga_history:
            final = result.ga_history[-1]
            print(f"GA best Sharpe: {final:.3f} (over {len(result.ga_history)} generations)")

    if args.json_output:
        output = {
            "dates": f"{returns_df.index[0].date()} to {returns_df.index[-1].date()}",
            "rmt_applied": not args.no_rmt,
        }
        if args.compare:
            output["results"] = {
                s: {
                    "weights": r.weights,
                    "expected_sharpe": r.expected_sharpe,
                    "expected_return": r.expected_return,
                    "expected_volatility": r.expected_volatility,
                }
                for s, r in results.items()
            }
        else:
            output["strategy"] = result.strategy
            output["weights"] = result.weights
            output["expected_sharpe"] = result.expected_sharpe
            output["expected_return"] = result.expected_return
            output["expected_volatility"] = result.expected_volatility

        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(output, indent=2, default=float))
        logger.info("Results saved to %s", args.json_output)


if __name__ == "__main__":
    main()
