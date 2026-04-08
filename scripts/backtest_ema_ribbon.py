"""Solo backtest for EMA Ribbon 9/21/55 strategy on BTC 1H data."""

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backtesting import Backtest

from src.strategies.ema_ribbon import EMARibbonStrategy


def main() -> None:
    """Run EMA Ribbon solo backtest."""
    data_path = project_root / "data" / "raw" / "BTC_USD_1h.csv"
    df = pd.read_csv(data_path, parse_dates=["Datetime"], index_col="Datetime")

    if "Volume" not in df.columns:
        df["Volume"] = 0

    print(f"Data: {len(df)} bars, {df.index[0]} to {df.index[-1]}")
    print(f"Price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")

    bt = Backtest(
        df,
        EMARibbonStrategy,
        cash=1_000_000,
        commission=0.001,
        exclusive_orders=True,
    )

    stats = bt.run()

    print("\n" + "=" * 60)
    print("EMA Ribbon 9/21/55 - Solo Backtest Results")
    print("=" * 60)
    print(f"Total Trades:             {stats.get('# Trades', 'N/A')}")
    print(f"Win Rate [%]:             {stats.get('Win Rate [%]', 'N/A'):.2f}")
    print(f"Sharpe Ratio:             {stats.get('Sharpe Ratio', 'N/A'):.4f}")
    print(f"Profit Factor:            {stats.get('Profit Factor', 'N/A'):.4f}")
    print(f"Max Drawdown [%]:         {stats.get('Max. Drawdown [%]', 'N/A'):.2f}")
    print(f"Return [%]:               {stats.get('Return [%]', 'N/A'):.2f}")
    print(f"Buy & Hold Return [%]:    {stats.get('Buy & Hold Return [%]', 'N/A'):.2f}")

    # Pass/fail assessment
    trades = int(stats.get("# Trades", 0))
    sharpe = float(stats.get("Sharpe Ratio", -999))
    win_rate = float(stats.get("Win Rate [%]", 0))
    pf = float(stats.get("Profit Factor", 0))

    print("\n--- Assessment ---")
    if trades < 10:
        print(f"[FAIL] INSUFFICIENT TRADES ({trades} < 10)")
    elif sharpe < 0.3:
        print(f"[FAIL] LOW SHARPE ({sharpe:.4f} < 0.3)")
    elif pf < 1.0:
        print(f"[FAIL] LOW PROFIT FACTOR ({pf:.4f} < 1.0)")
    else:
        print(f"[PASS] Trades: {trades}, Sharpe: {sharpe:.4f}, WR: {win_rate:.2f}%, PF: {pf:.4f}")


if __name__ == "__main__":
    main()
