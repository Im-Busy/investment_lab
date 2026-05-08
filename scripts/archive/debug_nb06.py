"""Debug notebook 06 - check trades extraction."""

import pandas as pd
from src.strategies.backtest_py.multi_pattern_strategy_optimized import (
    MultiPatternStrategyOptimized,
)
from src.strategies.backtest_py.runner import BacktestPyRunner


def main() -> None:
    df = pd.read_csv("data/raw/SPY_daily.csv", index_col=0, parse_dates=True)

    MultiPatternStrategyOptimized.enable_signal_log = True
    runner = BacktestPyRunner(
        data=df,
        cash=100_000,
        commission=0.001,
        exclusive_orders=True,
    )
    results = runner.run(
        strategy_class=MultiPatternStrategyOptimized,
        min_confidence=0.60,
        min_confluence_count=2,
        risk_per_trade=0.02,
        max_open_positions=5,
    )

    print(f"Has _trades: {hasattr(runner.results, '_trades')}")
    r = runner.results
    print(f"_trades type: {type(r._trades)}")
    print(f"_trades shape: {r._trades.shape}")
    print(f"_trades columns: {r._trades.keys()}")
    print("_trades head:")
    print(r._trades.head(2))

    t = runner.get_trades()
    print(f"\nget_trades shape: {t.shape}")
    print(f"get_trades cols: {list(t.columns)}")
    print(t.head(3))


if __name__ == "__main__":
    main()
