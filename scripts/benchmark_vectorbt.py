"""
VectorBT Performance Benchmark

Compares VectorBT performance vs sequential backtesting.py library.
Measures speedup for single-strategy and portfolio-level backtests.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
import sys

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backtesting import Backtest, Strategy

from src.backtest.vectorbt_adapter import VectorBTAdapter, VectorBTConfig, warmup_vectorbt


def load_spy_data(start: str = "2015-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Load SPY daily data."""
    data_path = project_root / "data" / "raw" / "SPY_daily.csv"
    df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={c: c.replace(" ", "") for c in df.columns})

    df = df.rename(
        columns={
            "Close": "Close",
            "High": "High",
            "Low": "Low",
            "Open": "Open",
            "Volume": "Volume",
        }
    )

    if "Volume" not in df.columns:
        df["Volume"] = 0

    df = df.sort_index()
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]

    return df


def load_strategy_classes() -> Dict[str, Any]:
    """Load strategy classes for benchmarking."""
    from importlib import import_module

    strategies = {
        "SMA Crossover 50/200": ("src.strategies.sma_crossover", "SMACrossoverStrategy"),
        "EMA Ribbon 9/21/55": ("src.strategies.ema_ribbon", "EMARibbonStrategy"),
        "VWAP Bounce": ("src.strategies.vwap_bounce", "VWAPBounceStrategy"),
        "Ultimate Oscillator": ("src.strategies.ultimate_oscillator", "UltimateOscillatorStrategy"),
    }

    classes = {}
    for name, (module_path, cls_name) in strategies.items():
        try:
            module = import_module(module_path)
            cls = getattr(module, cls_name)
            classes[name] = cls
        except Exception as e:
            print(f"[{name}] Failed to load: {e}")

    return classes


def run_backtesting_benchmark(
    df: pd.DataFrame, strategy_cls: Strategy, iterations: int = 5
) -> Dict[str, float]:
    """Benchmark sequential backtesting.py library."""
    times = []
    returns = []

    for _ in range(iterations):
        start = time.time()

        bt = Backtest(df, strategy_cls, cash=1_000_000, commission=0.001, exclusive_orders=True)
        stats = bt.run()

        elapsed = time.time() - start
        times.append(elapsed)
        returns.append(float(stats.get("Return [%]", 0)))

    return {
        "avg_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "avg_return": np.mean(returns),
        "iterations": iterations,
    }


def run_vectorbt_benchmark(
    df: pd.DataFrame, adapter: VectorBTAdapter, strategy_cls: Any, iterations: int = 5
) -> Dict[str, float]:
    """Benchmark VectorBT single-strategy backtesting."""
    times = []
    returns = []

    for _ in range(iterations):
        start = time.time()

        signals = adapter.generate_strategy_signals(df, strategy_cls)
        result = adapter.run_backtest(df, signals, name=strategy_cls.__name__)

        elapsed = time.time() - start
        times.append(elapsed)
        returns.append(result.return_pct)

    return {
        "avg_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "avg_return": np.mean(returns),
        "iterations": iterations,
    }


def run_portfolio_benchmark(
    df: pd.DataFrame,
    adapter: VectorBTAdapter,
    strategy_classes: Dict[str, Any],
    iterations: int = 5,
) -> Dict[str, float]:
    """Benchmark VectorBT portfolio-level backtesting."""
    times = []
    returns = []

    for _ in range(iterations):
        start = time.time()

        signal_dict = {}
        for name, strategy_cls in strategy_classes.items():
            signals = adapter.generate_strategy_signals(df, strategy_cls)
            signal_dict[name] = signals

        result = adapter.run_portfolio_backtest(df, signal_dict, weight_type="equal_weight")

        elapsed = time.time() - start
        times.append(elapsed)
        returns.append(result.return_pct)

    return {
        "avg_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "avg_return": np.mean(returns),
        "iterations": iterations,
    }


def print_comparison_table(strategy_name: str, backtesting_results: Dict, vectorbt_results: Dict):
    """Print formatted comparison table."""
    backtesting_time = backtesting_results["avg_time"]
    vectorbt_time = vectorbt_results["avg_time"]
    speedup = backtesting_time / vectorbt_time

    print(f"\n{'Strategy':<30} {strategy_name}")
    print("=" * 80)
    print(f"{'Metric':<20} {'backtesting.py':<20} {'VectorBT':<20} {'Speedup':<15}")
    print("-" * 80)

    print(
        f"{'Avg Time (s)':<20} {backtesting_time:>18.4f}  {vectorbt_time:>18.4f}  {speedup:>12.1f}x"
    )
    print(
        f"{'Std Time (s)':<20} {backtesting_results['std_time']:>18.4f}  {vectorbt_results['std_time']:>18.4f}"
    )
    print(
        f"{'Min Time (s)':<20} {backtesting_results['min_time']:>18.4f}  {vectorbt_results['min_time']:>18.4f}"
    )
    print(
        f"{'Max Time (s)':<20} {backtesting_results['max_time']:>18.4f}  {vectorbt_results['max_time']:>18.4f}"
    )

    print("\nReturns:")
    print(
        f"{'Avg Return (%)':<20} {backtesting_results['avg_return']:>18.2f}%  {vectorbt_results['avg_return']:>18.2f}%"
    )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="VectorBT Performance Benchmark")
    parser.add_argument("--start", default="2015-01-01", help="Start date")
    parser.add_argument(
        "--end", default="2024-1212", help="End date (shorter for faster benchmark)"
    )
    parser.add_argument("--iterations", type=int, default=5, help="Benchmark iterations")
    parser.add_argument("--warmup", action="store_true", default=True, help="Warm up VectorBT")
    args = parser.parse_args()

    print("=" * 80)
    print("VectorBT Performance Benchmark")
    print("=" * 80)

    print(f"\nLoading SPY data from {args.start} to {args.end}...")
    df = load_spy_data(args.start, args.end)
    print(f"Loaded {len(df)} bars")

    print("\nLoading strategy classes...")
    strategy_classes = load_strategy_classes()
    print(f"Loaded {len(strategy_classes)} strategies")

    if args.warmup:
        print("\nWarming up VectorBT...")
        warmup_vectorbt()

    adapter = VectorBTAdapter(VectorBTConfig())

    results = {"single_strategy": {}, "portfolio": {}}

    # Benchmark single strategies
    print("\n" + "=" * 80)
    print("Single-Strategy Backtest Benchmark")
    print("=" * 80)

    for name, strategy_cls in list(strategy_classes.items())[:2]:
        print(f"\nBenchmarking: {name}")

        backtesting_result = run_backtesting_benchmark(df, strategy_cls, args.iterations)
        vectorbt_result = run_vectorbt_benchmark(df, adapter, strategy_cls, args.iterations)

        print_comparison_table(name, backtesting_result, vectorbt_result)

        results["single_strategy"][name] = {
            "backtesting": backtesting_result,
            "vectorbt": vectorbt_result,
            "speedup": backtesting_result["avg_time"] / vectorbt_result["avg_time"],
        }

    # Benchmark portfolio
    print("\n" + "=" * 80)
    print("Portfolio-Level Backtest Benchmark (All Strategies)")
    print("=" * 80)

    portfolio_result = run_portfolio_benchmark(df, adapter, strategy_classes, args.iterations)

    print(f"\nPortfolio Results ({len(strategy_classes)} strategies)")
    print("=" * 80)
    print(f"{'Metric':<25} {'Value'}")
    print("-" * 80)
    print(f"{'Avg Time (s)':<25} {portfolio_result['avg_time']:.4f}")
    print(f"{'Std Time (s)':<25} {portfolio_result['std_time']:.4f}")
    print(f"{'Min Time (s)':<25} {portfolio_result['min_time']:.4f}")
    print(f"{'Max Time (s)':<25} {portfolio_result['max_time']:.4f}")
    print(f"{'Avg Return (%)':<25} {portfolio_result['avg_return']:.2f}%")
    print("=" * 80)

    results["portfolio"] = portfolio_result

    # Save results
    output_dir = project_root / "reports" / "vectorbt_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "vectorbt_benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
