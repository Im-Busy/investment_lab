"""
Optimize MACD - Parameter Sweep

Sweep MACD fast/slow/signal period combinations using vectorbt.

Usage:
    python scripts/optimize_macd.py --symbol BTC --save
"""

import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

sys.path.append(str(project_root))

from scripts.optimize_rsi import load_data


def optimize_macd(
    symbol: str = "BTC",
    fast_periods: range = range(6, 24, 2),
    slow_periods: range = range(20, 60, 4),
    signal_periods: range = range(4, 14, 2),
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Sweep MACD fast/slow/signal period parameters.

    Args:
        symbol: Data symbol to optimize on
        fast_periods: Fast EMA period values to sweep
        slow_periods: Slow EMA period values to sweep
        signal_periods: Signal line period values to sweep
        start_date: Start date for backtest
        end_date: End date for backtest

    Returns:
        DataFrame with all parameter combinations and their Sharpe ratios
    """
    try:
        import vectorbt as vbt
    except ImportError:
        print("vectorbt not installed. Run: uv add vectorbt")
        return pd.DataFrame()

    df = load_data(symbol)
    df = df[start_date:end_date]
    close = df["Close"]

    total_combos = len(fast_periods) * len(slow_periods) * len(signal_periods)
    print(
        f"Running MACD parameter sweep: {len(fast_periods)} fast × {len(slow_periods)} slow × {len(signal_periods)} signal = {total_combos} combinations"
    )

    # Filter valid combinations (fast < slow)
    valid_combos = [
        (f, s, sg) for f, s, sg in product(fast_periods, slow_periods, signal_periods) if f < s
    ]
    print(f"Valid combinations (fast < slow): {len(valid_combos)}")

    # Use vectorbt's built-in MACD indicator with parameter broadcasting
    fast_vals = [c[0] for c in valid_combos]
    slow_vals = [c[1] for c in valid_combos]
    signal_vals = [c[2] for c in valid_combos]

    macd = vbt.MACD.run(
        close,
        fast_window=fast_vals,
        slow_window=slow_vals,
        signal_window=signal_vals,
    )

    # Generate signals: entry when MACD crosses above signal, exit when below
    entries = macd.macd_crossed_above_signal()
    exits = macd.macd_crossed_below_signal()

    # Run portfolio
    pf = vbt.Portfolio.from_signals(
        close.reshape(-1, 1).repeat(len(valid_combos), axis=1),
        entries.fillna(False).astype(bool) if hasattr(entries, "fillna") else entries.astype(bool),
        exits.fillna(False).astype(bool) if hasattr(exits, "fillna") else exits.astype(bool),
        init_cash=100000,
        fees=0.001,
        freq="1h" if symbol == "BTC" else "1d",
    )

    # Extract results
    results = []
    for idx, (fast, slow, signal) in enumerate(valid_combos):
        if idx < len(pf.total_return()):
            total_ret = (
                pf.total_return().iloc[idx]
                if hasattr(pf.total_return(), "iloc")
                else pf.total_return()[idx]
            )
            sharpe = (
                pf.sharpe_ratio().iloc[idx]
                if hasattr(pf.sharpe_ratio(), "iloc")
                else pf.sharpe_ratio()[idx]
            )
            max_dd = (
                pf.max_drawdown().iloc[idx]
                if hasattr(pf.max_drawdown(), "iloc")
                else pf.max_drawdown()[idx]
            )
            trades = (
                pf.total_trades().iloc[idx]
                if hasattr(pf.total_trades(), "iloc")
                else pf.total_trades()[idx]
            )
            win_rt = (
                pf.win_rate().iloc[idx] if hasattr(pf.win_rate(), "iloc") else pf.win_rate()[idx]
            )

            results.append(
                {
                    "fast_period": fast,
                    "slow_period": slow,
                    "signal_period": signal,
                    "total_return": total_ret,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": max_dd,
                    "total_trades": trades,
                    "win_rate": win_rt,
                }
            )

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values("sharpe_ratio", ascending=False)

    # Save results
    output_path = project_root / "reports" / "parameter_optimization" / "macd_parameters.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(results_df) > 0:
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MACD Parameter Optimization")
    parser.add_argument("--symbol", default="BTC", help="Symbol to optimize on")
    parser.add_argument("--save", action="store_true", help="Save results to CSV")

    args = parser.parse_args()

    results = optimize_macd(symbol=args.symbol)
    if len(results) > 0:
        print("\nTop 10 MACD configurations:")
        print(results.head(10).to_string(index=False))
