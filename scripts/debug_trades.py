"""Debug trades extraction from backtesting.py Results."""

import pandas as pd

df = pd.read_csv("data/raw/SPY_daily.csv", index_col=0, parse_dates=True)

from backtesting import Backtest


class SimpleStrategy(Backtest.Strategy):  # noqa: F821
    def init(self):
        pass

    def next(self):
        if not self.position:
            self.buy()


bt = Backtest(df, SimpleStrategy, cash=100_000, commission=0.001)
stats = bt.run()

print(f"_trades type: {type(stats._trades)}")
print(f"_trades keys: {stats._trades.keys()}")
print(f"_trades shape: {stats._trades.shape}")
print(stats._trades.head(2))
