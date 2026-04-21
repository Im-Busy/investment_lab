"""
Multi-Strategy Portfolio Backtest Runner

This script demonstrates running multiple trading strategies
in parallel and combining them into a portfolio.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict

sys.path.append(str(Path(__file__).parent.parent))

from src.portfolio.multi_strategy_engine import (
    MultiStrategyEngine,
    EqualWeightScheme,
    SharpeWeightScheme,
    InverseVolatilityWeightScheme,
    KellyWeightScheme,
)
from src.portfolio.signal_aggregator import AggregationMethod, NormalizationMethod
from src.indicators.technical_numba import sma_numba, rsi_numba
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def load_sample_data(ticker: str = "AAPL", start_date: str = "2020-01-01") -> pd.DataFrame:
    """Load sample price data for backtesting."""
    try:
        import yfinance as yf

        data = yf.download(ticker, start=start_date, progress=False)
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0].lower() for col in data.columns]
        else:
            data.columns = data.columns.str.lower()

        return data
    except ImportError:
        # Generate synthetic data if yfinance is not available
        np.random.seed(42)
        n_days = 1000

        returns = np.random.normal(0.0005, 0.015, n_days)
        prices = 100 * np.cumprod(1 + returns)

        data = pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.001, n_days)),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
                "close": prices,
                "volume": np.random.randint(1000000, 10000000, n_days),
            }
        )

        return data


def sma_crossover_strategy(data: pd.DataFrame, **kwargs) -> np.ndarray:
    """Simple Moving Average Crossover strategy."""
    close = data["close"].values
    short_period = kwargs.get("short_period", 50)
    long_period = kwargs.get("long_period", 200)

    sma_short = sma_numba(close, short_period)
    sma_long = sma_numba(close, long_period)

    # Generate signals
    signals = np.zeros_like(close)

    for i in range(long_period, len(close)):
        if sma_short[i] > sma_long[i]:
            signals[i] = 1.0
        elif sma_short[i] < sma_long[i]:
            signals[i] = -1.0

    return signals


def rsi_mean_reversion_strategy(data: pd.DataFrame, **kwargs) -> np.ndarray:
    """RSI Mean Reversion strategy."""
    close = data["close"].values
    period = kwargs.get("period", 14)
    oversold = kwargs.get("oversold", 30)
    overbought = kwargs.get("overbought", 70)

    rsi = rsi_numba(close, period)

    signals = np.zeros_like(close)

    for i in range(period, len(close)):
        if rsi[i] < oversold:
            signals[i] = 1.0
        elif rsi[i] > overbought:
            signals[i] = -1.0

    return signals


def momentum_strategy(data: pd.DataFrame, **kwargs) -> np.ndarray:
    """Simple momentum strategy."""
    close = data["close"].values
    period = kwargs.get("period", 20)
    threshold = kwargs.get("threshold", 0.02)

    signals = np.zeros_like(close)

    for i in range(period, len(close)):
        momentum = (close[i] - close[i - period]) / close[i - period]

        if momentum > threshold:
            signals[i] = 1.0
        elif momentum < -threshold:
            signals[i] = -1.0

    return signals


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Print formatted metrics."""
    print(f"\n{title}")
    print("-" * 50)
    print(f"Total Return:  {metrics.get('total_return', 0):.4f}")
    print(f"Sharpe Ratio:  {metrics.get('sharpe', 0):.2f}")
    print(f"Max Drawdown:  {metrics.get('max_drawdown', 0):.4f}")
    print(f"Win Rate:      {metrics.get('win_rate', 0):.2%}")
    print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"Num Trades:    {metrics.get('num_trades', 0)}")


def compare_weight_schemes(engine: MultiStrategyEngine, strategy_funcs: List, data: pd.DataFrame):
    """Compare different weighting schemes."""
    print("\n" + "=" * 70)
    print("Comparing Weighting Schemes")
    print("=" * 70)

    schemes = {
        "Equal Weight": EqualWeightScheme(),
        "Sharpe Weight": SharpeWeightScheme(),
        "Inverse Volatility": InverseVolatilityWeightScheme(),
        "Kelly Criterion": KellyWeightScheme(),
    }

    results = {}

    for scheme_name, scheme in schemes.items():
        print(f"\n--- {scheme_name} ---")

        result = engine.run_portfolio_backtest(
            strategy_funcs=strategy_funcs, data=data, use_parallel=True
        )

        print_metrics(result.metrics, scheme_name)
        print(f"Execution Time: {result.execution_time:.2f}s")
        print(f"Number of Strategies: {result.num_strategies}")

        results[scheme_name] = result

    # Find best performing scheme
    best_scheme = max(results.items(), key=lambda x: x[1].metrics["sharpe"])

    print("\n" + "=" * 70)
    print(f"Best Scheme: {best_scheme[0]}")
    print("=" * 70)
    print_metrics(best_scheme[1].metrics, "Best Portfolio Metrics")

    return results


def main():
    """Main function to run portfolio backtest."""
    print("=" * 70)
    print("Multi-Strategy Portfolio Backtest")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    data = load_sample_data(ticker="AAPL")
    print(f"Data loaded: {len(data)} days")

    # Define strategies
    strategy_funcs = [sma_crossover_strategy, rsi_mean_reversion_strategy, momentum_strategy]

    print(f"\nRunning {len(strategy_funcs)} strategies:")
    for func in strategy_funcs:
        print(f"  - {func.__name__}")

    # Create engine with Numba-optimized aggregation
    engine = MultiStrategyEngine(
        n_workers=4,
        agg_method=AggregationMethod.WEIGHTED_MEAN,
        norm_method=NormalizationMethod.MIN_MAX,
    )

    # Compare different weighting schemes
    results = compare_weight_schemes(engine, strategy_funcs, data)

    # Test with custom aggregation methods
    print("\n" + "=" * 70)
    print("Testing Different Aggregation Methods")
    print("=" * 70)

    agg_methods = [
        ("Mean", AggregationMethod.MEAN),
        ("Median", AggregationMethod.MEDIAN),
        ("Vote", AggregationMethod.VOTE),
        ("Confluence", AggregationMethod.CONFLUENCE),
    ]

    for agg_name, agg_method in agg_methods:
        print(f"\n--- {agg_name} Aggregation ---")

        engine.aggregator.agg_method = agg_method
        result = engine.run_portfolio_backtest(
            strategy_funcs=strategy_funcs, data=data, use_parallel=True
        )

        print_metrics(result.metrics, agg_name)
        print(f"Execution Time: {result.execution_time:.2f}s")

    print("\n" + "=" * 70)
    print("Portfolio Backtest Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
