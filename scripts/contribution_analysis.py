"""
Pattern Contribution Analysis Script

Runs a backtest with signal logging enabled and generates a pattern contribution report
showing which patterns made money and which lost money.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def generate_test_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(seed)

    base_price = 100.0
    returns = np.random.randn(n) * 0.02
    closes = base_price * np.exp(np.cumsum(returns))

    intraday_range = np.abs(np.random.randn(n) * 0.01 + 0.005)
    highs = closes * (1 + intraday_range)
    lows = closes * (1 - intraday_range)
    opens = closes + (np.random.randn(n) * intraday_range * 0.3)

    highs = np.maximum.reduce([highs, opens, closes])
    lows = np.minimum.reduce([lows, opens, closes])

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


def run_contribution_analysis():
    """Run backtest with signal logging and generate contribution report."""
    print("=" * 70)
    print("PATTERN CONTRIBUTION ANALYSIS")
    print("=" * 70)

    # Generate test data
    n = 2000
    print(f"\nGenerating {n} bars of test data...")
    df = generate_test_data(n)

    # Import strategy with signal logging enabled
    print("\nImporting strategy...")
    from src.strategies.backtest_py.multi_pattern_strategy_optimized import (
        MultiPatternStrategyOptimized,
    )

    # Create strategy with signal logging enabled
    class ContributionStrategy(MultiPatternStrategyOptimized):
        enable_signal_log = True

    # Run backtest
    print("Running backtest with signal logging...")
    try:
        from backtesting import Backtest

        bt = Backtest(
            df,
            ContributionStrategy,
            cash=100000,
            commission=0.001,
            exclusive_orders=True,
        )

        stats = bt.run()
        print(f"Backtest complete: {stats.get('# Trades', 0)} trades")

        # Debug: check what _trades is
        raw_trades = stats._trades
        print(f"Trades type: {type(raw_trades)}")

        # Convert trades to DataFrame
        trades_data = []

        if isinstance(raw_trades, pd.DataFrame):
            # It's a DataFrame
            for idx, row in raw_trades.iterrows():
                trades_data.append(
                    {
                        "entry_time": row.get("EntryTime", row.get("entry_time", None)),
                        "exit_time": row.get("ExitTime", row.get("exit_time", None)),
                        "entry_price": row.get("EntryPrice", row.get("entry_price", None)),
                        "exit_price": row.get("ExitPrice", row.get("exit_price", None)),
                        "size": row.get("Size", row.get("size", 0)),
                        "pnl": row.get("PL", row.get("pl", row.get("PnL", 0))),
                        "pnl_pct": row.get("PLPct", row.get("pl_pct", row.get("PnL%", 0))),
                    }
                )
        elif isinstance(raw_trades, list):
            for i, t in enumerate(raw_trades):
                if isinstance(t, str):
                    continue
                trades_data.append(
                    {
                        "entry_time": getattr(t, "entry_time", None),
                        "exit_time": getattr(t, "exit_time", None),
                        "entry_price": getattr(t, "entry_price", None),
                        "exit_price": getattr(t, "exit_price", None),
                        "size": getattr(t, "size", 0),
                        "pnl": getattr(t, "pl", 0),
                        "pnl_pct": getattr(t, "pl_pct", 0),
                    }
                )

        trades_df = pd.DataFrame(trades_data)
        print(f"Trades collected: {len(trades_df)}")

        if len(trades_df) > 0:
            print(f"Sample trade columns: {list(trades_df.columns)}")

        # Access the strategy instance to get signal log
        strategy = stats._strategy

        if hasattr(strategy, "_signal_event_log") and strategy._signal_event_log is not None:
            signal_log = strategy._signal_event_log
            print(f"\nSignal events logged: {len(signal_log.events)}")

            # Generate contribution report
            from src.analysis.trade_attributor import TradeAttributor

            if not trades_df.empty:
                # Attribute trades to patterns
                attributor = TradeAttributor(signal_log, trades_df)
                attributor.attribute_trades()

                # Get pattern trade stats
                pattern_stats = attributor.get_pattern_trade_stats()

                if not pattern_stats.empty:
                    print("\n" + "=" * 70)
                    print("PATTERN CONTRIBUTION LEADERBOARD")
                    print("=" * 70)
                    print(
                        f"\n{'Pattern':<25} {'Trades':>8} {'Wins':>8} {'Win%':>8} {'Pnl':>12} {'Pnl%':>10}"
                    )
                    print("-" * 70)

                    # Sort by PnL descending
                    pattern_stats_sorted = pattern_stats.sort_values("total_pnl", ascending=False)

                    for _, row in pattern_stats_sorted.iterrows():
                        pattern = row.get("pattern_name", row.name)
                        trades_count = row.get("trade_count", 0)
                        wins = row.get("wins", 0)
                        win_rate = row.get("win_rate", 0) * 100 if row.get("win_rate", 0) else 0
                        pnl = row.get("total_pnl", 0)
                        pnl_pct = (
                            row.get("avg_pnl_pct", 0) * 100 if row.get("avg_pnl_pct", 0) else 0
                        )

                        marker = "+" if pnl > 0 else "-"
                        print(
                            f"{marker} {pattern:<24} {trades_count:>8} {wins:>8} {win_rate:>7.1f}% {pnl:>12.2f} {pnl_pct:>9.2f}%"
                        )

                    print("\n" + "=" * 70)
                    print("WINNERS (Positive PnL)")
                    print("=" * 70)
                    winners = pattern_stats_sorted[pattern_stats_sorted["total_pnl"] > 0]
                    for _, row in winners.head(10).iterrows():
                        pattern = row.get("pattern_name", row.name)
                        pnl = row.get("total_pnl", 0)
                        trades = row.get("trade_count", 0)
                        print(f"  ✅ {pattern}: ${pnl:.2f} ({trades} trades)")

                    print("\n" + "=" * 70)
                    print("LOSERS (Negative PnL)")
                    print("=" * 70)
                    losers = pattern_stats_sorted[pattern_stats_sorted["total_pnl"] < 0]
                    for _, row in losers.head(10).iterrows():
                        pattern = row.get("pattern_name", row.name)
                        pnl = row.get("total_pnl", 0)
                        trades = row.get("trade_count", 0)
                        print(f"  ❌ {pattern}: ${pnl:.2f} ({trades} trades)")

                else:
                    print("\nNo pattern trade stats available")
            else:
                print("\nNo trades to analyze")
        else:
            print("\nSignal logging not enabled or no events recorded")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_contribution_analysis()
