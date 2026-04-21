"""
Vectorized Backtest Engine Benchmark

Demonstrates high-performance vectorized backtesting without VectorBT dependency.
Compares different backtest approaches and measures performance.

Usage:
    uv run scripts/benchmark_vectorized_engine.py --start 2020-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(project_root))


def load_spy_data(start: str = "2015-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Load SPY daily data."""
    data_path = project_root / "data" / "raw" / "SPY_daily.csv"
    df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
    df.columns = [c.strip() for c in df.columns]
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


def generate_sma_signals(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    """Generate SMA crossover signals."""
    close = df["Close"]
    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean()
    signals = pd.Series(0, index=df.index, dtype=np.int8)
    signals[sma_fast > sma_slow] = 1
    signals[sma_fast <= sma_slow] = -1
    return signals


def generate_ema_signals(df: pd.DataFrame, periods: List[int] = None) -> pd.Series:
    """Generate EMA ribbon signals."""
    if periods is None:
        periods = [9, 21, 55]
    close = df["Close"]
    emas = [close.ewm(span=p, adjust=False).mean() for p in periods]
    signals = pd.Series(0, index=df.index, dtype=np.int8)
    bullish = (emas[0] > emas[1]) & (emas[1] > emas[2])
    bearish = (emas[0] < emas[1]) & (emas[1] < emas[2])
    signals[bullish] = 1
    signals[bearish] = -1
    return signals


def generate_rsi_signals(
    df: pd.DataFrame, period: int = 14, lower: float = 30, upper: float = 70
) -> pd.Series:
    """Generate RSI signals."""
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    signals = pd.Series(0, index=df.index, dtype=np.int8)
    signals[rsi < lower] = 1
    signals[rsi > upper] = -1
    return signals


def run_loop_backtest(df: pd.DataFrame, signals: pd.Series) -> Dict[str, Any]:
    """Simple loop-based backtest (baseline)."""
    equity = 1_000_000.0
    position = 0
    entry_price = 0
    trades = []
    equity_curve = []

    for i in range(1, len(df)):
        signal = signals.iloc[i]
        price = df["Close"].iloc[i]

        if signal > 0 and position == 0:
            position = equity / price
            entry_price = price
        elif signal < 0 and position > 0:
            pnl = (price - entry_price) * position
            equity += pnl
            trades.append(pnl / equity * 100)
            position = 0
            entry_price = 0

        equity_curve.append(equity)

    equity_curve = [1_000_000] + equity_curve
    total_return = (equity_curve[-1] / 1_000_000 - 1) * 100
    return {
        "equity_curve": equity_curve,
        "total_return": total_return,
        "trades": len(trades),
        "trade_returns": trades,
    }


def run_vectorized_backtest(df: pd.DataFrame, signals: pd.Series) -> Dict[str, Any]:
    """Vectorized backtest using NumPy."""
    import sys

    sys.path.insert(0, str(project_root))

    from src.backtest.vectorbt_alternative import VectorizedPortfolioEngine

    engine = VectorizedPortfolioEngine()
    result = engine.run_backtest(df, signals, name="Vectorized")

    return {
        "equity_curve": result.equity_curve.values.tolist(),
        "total_return": result.total_return_pct,
        "trades": result.trades,
        "trade_returns": [result.avg_trade_pct] * result.trades if result.trades > 0 else [],
        "sharpe": result.sharpe_ratio,
        "max_dd": result.max_drawdown_pct,
    }


def benchmark_engines(
    df: pd.DataFrame,
    signal_generators: Dict[str, callable],
    iterations: int = 5,
) -> Dict[str, Any]:
    """Benchmark loop vs vectorized approaches."""
    results = {
        "loop": {"times": [], "returns": []},
        "vectorized": {"times": [], "returns": [], "metrics": []},
    }

    # Warmup: run once to load modules
    print("Warming up vectorized engine...")
    test_signals = generate_sma_signals(df, 50, 200)
    _ = run_vectorized_backtest(df, test_signals)
    print("Warmup complete.\n")

    for sig_name, sig_func in signal_generators.items():
        print(f"Benchmarking: {sig_name}")

        signals = sig_func(df)

        # Run loop-based backtest
        loop_times = []
        loop_returns = []
        for _ in range(iterations):
            start = time.time()
            loop_result = run_loop_backtest(df, signals)
            elapsed = time.time() - start
            loop_times.append(elapsed)
            loop_returns.append(loop_result["total_return"])

        # Run vectorized backtest
        vec_times = []
        vec_returns = []
        vec_metrics = []
        for _ in range(iterations):
            start = time.time()
            vec_result = run_vectorized_backtest(df, signals)
            elapsed = time.time() - start
            vec_times.append(elapsed)
            vec_returns.append(vec_result["total_return"])
            vec_metrics.append(
                {
                    "sharpe": vec_result.get("sharpe", 0),
                    "max_dd": vec_result.get("max_dd", 0),
                    "trades": vec_result["trades"],
                }
            )

        loop_time = np.mean(loop_times)
        vec_time = np.mean(vec_times)
        speedup = loop_time / vec_time if vec_time > 0 else float("inf")

        print(f"  Loop:       {loop_time:>10.4f}s")
        print(f"  Vectorized: {vec_time:>10.4f}s")
        print(f"  Speedup:    {speedup:>10.1f}x")

        results["loop"]["times"].extend(loop_times)
        results["loop"]["returns"].extend(loop_returns)
        results["vectorized"]["times"].extend(vec_times)
        results["vectorized"]["returns"].extend(vec_returns)
        results["vectorized"]["metrics"].extend(vec_metrics)

    return results


def print_summary_table(results: Dict[str, Any], signal_generators: Dict[str, callable]):
    """Print benchmark summary table."""
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY")
    print("=" * 90)

    strategies = list(signal_generators.keys())

    print(
        f"\n{'Strategy':<20} {'Loop Time (s)':<15} {'Vectorized (s)':<15} {'Speedup':<12} {'Return (%)':<12}"
    )
    print("-" * 90)

    for i, strat in enumerate(strategies):
        loop_t = np.mean(results["loop"]["times"]) if i < len(results["loop"]["times"]) else 0
        vec_t = (
            np.mean(results["vectorized"]["times"])
            if i < len(results["vectorized"]["times"])
            else 0
        )
        speedup = loop_t / vec_t if vec_t > 0 else float("inf")
        ret = (
            results["vectorized"]["returns"][i] if i < len(results["vectorized"]["returns"]) else 0
        )
        print(f"{strat:<20} {loop_t:>14.4f} {vec_t:>14.4f} {speedup:>10.1f}x {ret:>10.2f}%")

    print("=" * 90)

    avg_loop = np.mean(results["loop"]["times"])
    avg_vec = np.mean(results["vectorized"]["times"])
    avg_speedup = avg_loop / avg_vec if avg_vec > 0 else float("inf")

    print(f"\nAverage Loop Time:      {avg_loop:.4f}s")
    print(f"Average Vectorized Time: {avg_vec:.4f}s")
    print(f"Average Speedup:         {avg_speedup:.1f}x")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Vectorized Engine Benchmark")
    parser.add_argument("--start", default="2020-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument("--iterations", type=int, default=3, help="Benchmark iterations")
    args = parser.parse_args()

    print("=" * 90)
    print("Vectorized Portfolio Backtest Engine Benchmark")
    print("=" * 90)
    print("\nThis benchmark demonstrates NumPy vectorization performance vs loop-based approach.")
    print("No VectorBT dependency required - pure NumPy implementation.\n")

    print(f"Loading SPY data from {args.start} to {args.end}...")
    df = load_spy_data(args.start, args.end)
    print(f"Loaded {len(df)} bars\n")

    signal_generators = {
        "SMA Crossover": lambda x: generate_sma_signals(x, 50, 200),
        "RSI": lambda x: generate_rsi_signals(x, 14, 30, 70),
    }

    print("=" * 90)
    print("Running Benchmarks...")
    print("=" * 90)

    results = benchmark_engines(df, signal_generators, args.iterations)

    print_summary_table(results, signal_generators)

    output_dir = project_root / "reports" / "vectorized_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "benchmark_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Vectorized Engine Benchmark Results\n")
        f.write(f"Date Range: {args.start} to {args.end}\n")
        f.write(f"Data: SPY ({len(df)} bars)\n\n")
        f.write(
            f"Speedup: {np.mean([r for r in results['vectorized']['times']]) / np.mean([r for r in results['loop']['times']]) * 100:.1f}x\n"
        )

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
