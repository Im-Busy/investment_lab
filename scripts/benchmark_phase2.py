"""
Phase 2 Performance Benchmark Script

Measures performance improvements from Numba JIT compilation and vectorized detection.
Compares:
- Pivot detection: Pure Python vs Numba JIT
- Technical indicators: Pandas vs Numba JIT
- Pattern detection: Bar-by-bar vs Vectorized
- Full backtest: Original vs Optimized strategy
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def generate_test_data(n: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for benchmarking.

    Args:
        n: Number of bars
        seed: Random seed for reproducibility

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(seed)

    # Generate realistic-looking price data
    base_price = 100.0
    returns = np.random.randn(n) * 0.02
    closes = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    intraday_range = np.abs(np.random.randn(n) * 0.01 + 0.005)
    highs = closes * (1 + intraday_range)
    lows = closes * (1 - intraday_range)
    opens = closes + (np.random.randn(n) * intraday_range * 0.3)

    # Ensure High >= Open, Close and Low <= Open, Close
    highs = np.maximum.reduce([highs, opens, closes])
    lows = np.minimum.reduce([lows, opens, closes])

    # Volume with some randomness
    volumes = np.random.randint(100000, 1000000, n)

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        }
    )

    return df


def benchmark_pivot_detection(df: pd.DataFrame, iterations: int = 100) -> Dict[str, Any]:
    """
    Benchmark swing high/low detection.

    Compares:
    - Pure Python implementation
    - Numba JIT implementation
    """
    from src.indicators.pivots import find_swing_highs, find_swing_lows

    highs = df["High"].values
    lows = df["Low"].values

    results = {}

    # Benchmark pure Python (pandas-based)
    print("\n  Pivot Detection (Pure Python):")
    start = time.perf_counter()
    for _ in range(iterations):
        swing_highs = find_swing_highs(df, lookback=5)
        swing_lows = find_swing_lows(df, lookback=5)
    python_time = time.perf_counter() - start
    results["python_time"] = python_time
    results["python_per_iter"] = python_time / iterations
    print(f"    Total: {python_time:.4f}s ({iterations} iterations)")
    print(f"    Per iteration: {python_time / iterations:.4f}s")

    # Benchmark Numba
    try:
        from src.indicators.pivots_numba import (
            find_swing_highs_numba,
            find_swing_lows_numba,
            NUMBA_AVAILABLE,
        )

        if NUMBA_AVAILABLE:
            # Warm up JIT (first call compiles)
            _ = find_swing_highs_numba(highs[:100], 5)
            _ = find_swing_lows_numba(lows[:100], 5)

            print("\n  Pivot Detection (Numba JIT):")
            start = time.perf_counter()
            for _ in range(iterations):
                swing_highs = find_swing_highs_numba(highs, 5)
                swing_lows = find_swing_lows_numba(lows, 5)
            numba_time = time.perf_counter() - start
            results["numba_time"] = numba_time
            results["numba_per_iter"] = numba_time / iterations
            results["numba_speedup"] = python_time / numba_time if numba_time > 0 else 0
            print(f"    Total: {numba_time:.4f}s ({iterations} iterations)")
            print(f"    Per iteration: {numba_time / iterations:.4f}s")
            print(f"    Speedup: {results['numba_speedup']:.1f}x")
        else:
            print("\n  Numba not available - skipping")
            results["numba_time"] = None
            results["numba_speedup"] = None
    except ImportError as e:
        print(f"\n  Numba import failed: {e}")
        results["numba_time"] = None
        results["numba_speedup"] = None

    return results


def benchmark_indicators(df: pd.DataFrame, iterations: int = 100) -> Dict[str, Any]:
    """
    Benchmark technical indicators.

    Compares:
    - Pandas implementation
    - Numba JIT implementation
    """
    from src.indicators.technical import sma, ema, atr, rsi

    results = {}

    # Benchmark SMA
    print("\n  SMA (20-period):")
    start = time.perf_counter()
    for _ in range(iterations):
        _ = sma(df["Close"], 20)
    pandas_sma_time = time.perf_counter() - start
    results["sma_pandas"] = pandas_sma_time
    print(f"    Pandas: {pandas_sma_time:.4f}s")

    try:
        from src.indicators.technical_numba import sma_numba, NUMBA_AVAILABLE

        if NUMBA_AVAILABLE:
            values = df["Close"].values.astype(np.float64)
            _ = sma_numba(values[:100], 20)  # Warm up

            start = time.perf_counter()
            for _ in range(iterations):
                _ = sma_numba(values, 20)
            numba_sma_time = time.perf_counter() - start
            results["sma_numba"] = numba_sma_time
            results["sma_speedup"] = pandas_sma_time / numba_sma_time if numba_sma_time > 0 else 0
            print(f"    Numba:  {numba_sma_time:.4f}s")
            print(f"    Speedup: {results['sma_speedup']:.1f}x")
    except ImportError:
        results["sma_numba"] = None
        results["sma_speedup"] = None

    # Benchmark EMA
    print("\n  EMA (20-period):")
    start = time.perf_counter()
    for _ in range(iterations):
        _ = ema(df["Close"], 20)
    pandas_ema_time = time.perf_counter() - start
    results["ema_pandas"] = pandas_ema_time
    print(f"    Pandas: {pandas_ema_time:.4f}s")

    try:
        from src.indicators.technical_numba import ema_numba, NUMBA_AVAILABLE

        if NUMBA_AVAILABLE:
            values = df["Close"].values.astype(np.float64)
            _ = ema_numba(values[:100], 20)  # Warm up

            start = time.perf_counter()
            for _ in range(iterations):
                _ = ema_numba(values, 20)
            numba_ema_time = time.perf_counter() - start
            results["ema_numba"] = numba_ema_time
            results["ema_speedup"] = pandas_ema_time / numba_ema_time if numba_ema_time > 0 else 0
            print(f"    Numba:  {numba_ema_time:.4f}s")
            print(f"    Speedup: {results['ema_speedup']:.1f}x")
    except ImportError:
        results["ema_numba"] = None
        results["ema_speedup"] = None

    # Benchmark ATR
    print("\n  ATR (14-period):")
    start = time.perf_counter()
    for _ in range(iterations):
        _ = atr(df, 14)
    pandas_atr_time = time.perf_counter() - start
    results["atr_pandas"] = pandas_atr_time
    print(f"    Pandas: {pandas_atr_time:.4f}s")

    try:
        from src.indicators.technical_numba import atr_numba, NUMBA_AVAILABLE

        if NUMBA_AVAILABLE:
            highs = df["High"].values.astype(np.float64)
            lows = df["Low"].values.astype(np.float64)
            closes = df["Close"].values.astype(np.float64)
            _ = atr_numba(highs[:100], lows[:100], closes[:100], 14)  # Warm up

            start = time.perf_counter()
            for _ in range(iterations):
                _ = atr_numba(highs, lows, closes, 14)
            numba_atr_time = time.perf_counter() - start
            results["atr_numba"] = numba_atr_time
            results["atr_speedup"] = pandas_atr_time / numba_atr_time if numba_atr_time > 0 else 0
            print(f"    Numba:  {numba_atr_time:.4f}s")
            print(f"    Speedup: {results['atr_speedup']:.1f}x")
    except ImportError:
        results["atr_numba"] = None
        results["atr_speedup"] = None

    # Benchmark RSI
    print("\n  RSI (14-period):")
    start = time.perf_counter()
    for _ in range(iterations):
        _ = rsi(df["Close"], 14)
    pandas_rsi_time = time.perf_counter() - start
    results["rsi_pandas"] = pandas_rsi_time
    print(f"    Pandas: {pandas_rsi_time:.4f}s")

    try:
        from src.indicators.technical_numba import rsi_numba, NUMBA_AVAILABLE

        if NUMBA_AVAILABLE:
            values = df["Close"].values.astype(np.float64)
            _ = rsi_numba(values[:100], 14)  # Warm up

            start = time.perf_counter()
            for _ in range(iterations):
                _ = rsi_numba(values, 14)
            numba_rsi_time = time.perf_counter() - start
            results["rsi_numba"] = numba_rsi_time
            results["rsi_speedup"] = pandas_rsi_time / numba_rsi_time if numba_rsi_time > 0 else 0
            print(f"    Numba:  {numba_rsi_time:.4f}s")
            print(f"    Speedup: {results['rsi_speedup']:.1f}x")
    except ImportError:
        results["rsi_numba"] = None
        results["rsi_speedup"] = None

    return results


def benchmark_pattern_detection(df: pd.DataFrame, iterations: int = 10) -> Dict[str, Any]:
    """
    Benchmark pattern detection.

    Compares:
    - Bar-by-bar detection
    - Vectorized detection (Numba)
    """
    from src.patterns.basic.msl import MarketStructureLow
    from src.patterns.basic.nr7id import NR7ID
    from src.patterns.basic.floor_pivot import FloorPivotBreakout

    results = {}

    patterns = [
        ("MSL", MarketStructureLow()),
        ("NR7ID", NR7ID()),
        ("FloorPivot", FloorPivotBreakout()),
    ]

    for name, pattern in patterns:
        print(f"\n  {name} Pattern Detection:")

        # Bar-by-bar detection
        start = time.perf_counter()
        for _ in range(iterations):
            for i in range(pattern.min_bars_required + 1, len(df)):
                _ = pattern.detect(df, i)
        bar_by_bar_time = time.perf_counter() - start
        results[f"{name}_bar_by_bar"] = bar_by_bar_time
        print(f"    Bar-by-bar: {bar_by_bar_time:.4f}s")

        # Vectorized detection
        try:
            start = time.perf_counter()
            for _ in range(iterations):
                _ = pattern.detect_vectorized(df)
            vectorized_time = time.perf_counter() - start
            results[f"{name}_vectorized"] = vectorized_time
            results[f"{name}_speedup"] = bar_by_bar_time / vectorized_time if vectorized_time > 0 else 0
            print(f"    Vectorized: {vectorized_time:.4f}s")
            print(f"    Speedup: {results[f'{name}_speedup']:.1f}x")
        except Exception as e:
            print(f"    Vectorized failed: {e}")
            results[f"{name}_vectorized"] = None
            results[f"{name}_speedup"] = None

    return results


def benchmark_backtest(df: pd.DataFrame, n_bars: int = 5000) -> Dict[str, Any]:
    """
    Benchmark full backtest with optimized strategy.

    Measures:
    - Time to run backtest
    - Number of signals generated
    - Performance metrics
    """
    from src.strategies.backtest_py.multi_pattern_strategy_optimized import (
        MultiPatternStrategyOptimized,
    )

    results = {}

    # Trim data if needed
    if len(df) > n_bars:
        df_test = df.iloc[-n_bars:].copy()
    else:
        df_test = df.copy()

    print(f"\n  Backtest on {len(df_test)} bars:")

    try:
        from backtesting import Backtest

        # Run backtest
        bt = Backtest(
            df_test,
            MultiPatternStrategyOptimized,
            cash=100000,
            commission=0.001,
            exclusive_orders=True,
        )

        start = time.perf_counter()
        stats = bt.run()
        backtest_time = time.perf_counter() - start

        results["backtest_time"] = backtest_time
        results["bars_per_second"] = len(df_test) / backtest_time
        results["trades"] = stats.get("# Trades", 0)
        results["return_pct"] = stats.get("Return [%]", 0)

        print(f"    Time: {backtest_time:.4f}s")
        print(f"    Speed: {results['bars_per_second']:.0f} bars/second")
        print(f"    Trades: {results['trades']}")
        print(f"    Return: {results['return_pct']:.2f}%")

    except Exception as e:
        print(f"    Backtest failed: {e}")
        results["backtest_time"] = None
        results["error"] = str(e)

    return results


def benchmark_warmup() -> float:
    """
    Benchmark Numba JIT warm-up time.

    Returns:
        Time in seconds for warm-up
    """
    from src.indicators.pivots_numba import warmup as pivot_warmup
    from src.indicators.technical_numba import warmup as tech_warmup

    print("\n  Numba JIT Warm-up:")

    start = time.perf_counter()
    try:
        pivot_warmup()
        tech_warmup()
        warmup_time = time.perf_counter() - start
        print(f"    Warm-up time: {warmup_time:.4f}s")
        return warmup_time
    except Exception as e:
        print(f"    Warm-up failed: {e}")
        return 0.0


def run_benchmarks():
    """Run all Phase 2 benchmarks."""
    print("=" * 70)
    print("PHASE 2 PERFORMANCE BENCHMARKS")
    print("=" * 70)

    # Data sizes to test
    data_sizes = [1000, 5000, 10000]

    all_results = {}

    for n in data_sizes:
        print(f"\n{'=' * 70}")
        print(f"DATA SIZE: {n} bars")
        print(f"{'=' * 70}")

        df = generate_test_data(n)

        # Warm up Numba
        benchmark_warmup()

        # Run benchmarks
        print("\n" + "-" * 40)
        print("1. PIVOT DETECTION BENCHMARK")
        print("-" * 40)
        pivot_results = benchmark_pivot_detection(df, iterations=max(10, 1000 // n * 10))
        all_results[f"pivot_{n}"] = pivot_results

        print("\n" + "-" * 40)
        print("2. TECHNICAL INDICATORS BENCHMARK")
        print("-" * 40)
        indicator_results = benchmark_indicators(df, iterations=max(10, 1000 // n * 10))
        all_results[f"indicators_{n}"] = indicator_results

        print("\n" + "-" * 40)
        print("3. PATTERN DETECTION BENCHMARK")
        print("-" * 40)
        pattern_results = benchmark_pattern_detection(df, iterations=max(5, 100 // n * 5))
        all_results[f"patterns_{n}"] = pattern_results

        print("\n" + "-" * 40)
        print("4. FULL BACKTEST BENCHMARK")
        print("-" * 40)
        backtest_results = benchmark_backtest(df, n_bars=n)
        all_results[f"backtest_{n}"] = backtest_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nPivot Detection Speedup:")
    for n in data_sizes:
        r = all_results.get(f"pivot_{n}", {})
        if r.get("numba_speedup"):
            print(f"  {n} bars: {r['numba_speedup']:.1f}x")

    print("\nTechnical Indicators Speedup (average):")
    speedups = {"sma": [], "ema": [], "atr": [], "rsi": []}
    for n in data_sizes:
        r = all_results.get(f"indicators_{n}", {})
        for ind in speedups:
            if r.get(f"{ind}_speedup"):
                speedups[ind].append(r[f"{ind}_speedup"])

    for ind, su_list in speedups.items():
        if su_list:
            avg = sum(su_list) / len(su_list)
            print(f"  {ind.upper()}: {avg:.1f}x (avg)")

    print("\nPattern Detection Speedup:")
    for n in data_sizes:
        r = all_results.get(f"patterns_{n}", {})
        for pattern in ["MSL", "NR7ID", "FloorPivot"]:
            if r.get(f"{pattern}_speedup"):
                print(f"  {pattern} @ {n} bars: {r[f'{pattern}_speedup']:.1f}x")

    print("\nBacktest Performance:")
    for n in data_sizes:
        r = all_results.get(f"backtest_{n}", {})
        if r.get("backtest_time"):
            print(f"  {n} bars: {r['backtest_time']:.2f}s ({r['bars_per_second']:.0f} bars/sec)")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    results = run_benchmarks()
