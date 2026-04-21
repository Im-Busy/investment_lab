"""Test script to understand backtesting.py API."""

import pandas as pd
import numpy as np
from src.strategies.donchian_breakout import DonchianChannelStrategy
from backtesting import Backtest

rng = np.random.default_rng(42)
prices = np.cumsum(rng.normal(0, 0.001, 200)) + 100

data = pd.DataFrame(
    {"Open": prices, "High": prices + 0.1, "Low": prices - 0.1, "Close": prices},
    index=pd.date_range("2023-01-01", periods=200, freq="D"),
)

bt = Backtest(data, DonchianChannelStrategy, cash=10000)
result = bt.run()

print(f"Result type: {type(result)}")
print(f"Result: {result}")
print(f"\nBacktest attributes: {dir(bt)}")

if hasattr(bt, "trades"):
    print(f"\nTrades: {bt.trades}")

if hasattr(bt, "equity_curve"):
    print(f"\nEquity curve available")
