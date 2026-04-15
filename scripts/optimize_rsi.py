"""
Parameter Optimization Scripts - Phase 3

Uses vectorbt for research-only parameter sweeps across strategy thresholds.
These scripts do NOT replace the custom backtest engine — they are for
research and finding stable parameter ranges per asset class.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_data(symbol: str = "BTC") -> pd.DataFrame:
    """Load OHLCV data for the given symbol."""
    if symbol == "BTC":
        data_path = project_root / "data" / "raw" / "BTC_USD_1h.csv"
    elif symbol == "SPY":
        data_path = project_root / "data" / "raw" / "SPY_daily.csv"
    else:
        data_path = project_root / "data" / "raw" / f"{symbol}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, parse_dates=True, index_col=0)
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df


def optimize_rsi(
    symbol: str = "BTC",
    windows: range = range(8, 30, 2),
    oversold: range = range(20, 40, 5),
    overbought: range = range(60, 80, 5),
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Sweep RSI window and threshold parameters.

    Args:
        symbol: Data symbol to optimize on
        windows: RSI window values to sweep
        oversold: Oversold threshold values to sweep
        overbought: Overbought threshold values to sweep
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

    print(
        f"Running RSI parameter sweep: {len(windows)} windows × {len(oversold)} OS × {len(overbought)} OB = {len(windows) * len(oversold) * len(overbought)} combinations"
    )

    # Use vectorbt's built-in RSI indicator with parameter broadcasting
    rsi = vbt.RSI.run(close, window=windows)

    # Create entry/exit signals for each parameter combo
    # Entry: RSI crosses above oversold threshold (oversold bounce)
    # Exit: RSI crosses below overbought threshold (overbought exit)
    entries = pd.DataFrame(index=rsi.rsi.index, columns=rsi.rsi.columns)
    exits = pd.DataFrame(index=rsi.rsi.index, columns=rsi.rsi.columns)

    for i, window in enumerate(windows):
        for j, os in enumerate(oversold):
            for k, ob in enumerate(overbought):
                col_idx = i * len(oversold) * len(overbought) + j * len(overbought) + k
                if col_idx < rsi.rsi.shape[1]:
                    rsi_series = rsi.rsi.iloc[:, col_idx]
                    entries.iloc[:, col_idx] = (rsi_series.shift(1) < os) & (rsi_series >= os)
                    exits.iloc[:, col_idx] = rsi_series > ob

    # Run backtest
    pf = vbt.Portfolio.from_signals(
        close,
        entries.fillna(False).astype(bool),
        exits.fillna(False).astype(bool),
        init_cash=100000,
        fees=0.001,
        freq="1h" if symbol == "BTC" else "1d",
    )

    # Extract results
    results = []
    for i, window in enumerate(windows):
        for j, os in enumerate(oversold):
            for k, ob in enumerate(overbought):
                col_idx = i * len(oversold) * len(overbought) + j * len(overbought) + k
                if col_idx < len(pf.total_return()):
                    results.append(
                        {
                            "window": window,
                            "oversold_threshold": os,
                            "overbought_threshold": ob,
                            "total_return": pf.total_return().iloc[col_idx]
                            if hasattr(pf.total_return(), "iloc")
                            else pf.total_return()[col_idx],
                            "sharpe_ratio": pf.sharpe_ratio().iloc[col_idx]
                            if hasattr(pf.sharpe_ratio(), "iloc")
                            else pf.sharpe_ratio()[col_idx],
                            "max_drawdown": pf.max_drawdown().iloc[col_idx]
                            if hasattr(pf.max_drawdown(), "iloc")
                            else pf.max_drawdown()[col_idx],
                            "total_trades": pf.total_trades().iloc[col_idx]
                            if hasattr(pf.total_trades(), "iloc")
                            else pf.total_trades()[col_idx],
                            "win_rate": pf.win_rate().iloc[col_idx]
                            if hasattr(pf.win_rate(), "iloc")
                            else pf.win_rate()[col_idx],
                        }
                    )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("sharpe_ratio", ascending=False)

    # Save results
    output_path = project_root / "reports" / "parameter_optimization" / "rsi_parameters.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    return results_df


def find_stable_parameters(
    results: pd.DataFrame,
    metric: str = "sharpe_ratio",
    pct_threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Find stable parameter ranges where performance varies < pct_threshold.

    Args:
        results: Results DataFrame from optimization
        metric: Metric to check stability on
        pct_threshold: Maximum performance variation (e.g., 0.10 = 10%)

    Returns:
        DataFrame with stable parameter ranges
    """
    if len(results) == 0:
        return pd.DataFrame()

    # Sort by the metric
    best = results.nlargest(1, metric).iloc[0]
    best_value = best[metric]

    # Find all parameters within pct_threshold of best
    stable = results[results[metric] >= best_value * (1 - pct_threshold)]

    return stable


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parameter Optimization Scripts")
    parser.add_argument("--strategy", choices=["rsi"], default="rsi", help="Strategy to optimize")
    parser.add_argument("--symbol", default="BTC", help="Symbol to optimize on")
    parser.add_argument("--output", help="Output directory for results")

    args = parser.parse_args()

    if args.strategy == "rsi":
        results = optimize_rsi(symbol=args.symbol)
        if len(results) > 0:
            stable = find_stable_parameters(results)
            print(f"\nTop 10 RSI configurations:")
            print(results.head(10).to_string(index=False))
            print(f"\nStable parameters (within 10% of best):")
            print(stable.to_string(index=False))
