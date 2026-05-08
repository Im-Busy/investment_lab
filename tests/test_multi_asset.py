"""Test top strategies across all available tickers."""

import sys
from pathlib import Path

import pandas as pd
from backtesting import Backtest

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.strategies.awesome_oscillator import AwesomeOscillatorStrategy
from src.strategies.cci_strategy import CCIStrategy
from src.strategies.chaikin_oscillator import ChaikinOscillatorStrategy
from src.strategies.ichimoku_cloud import IchimokuCloudStrategy
from src.strategies.linear_regression_channel import LinearRegressionChannelStrategy
from src.strategies.sma_crossover import SMACrossoverStrategy
from src.strategies.stoch_rsi_crossover import StochRSICrossoverStrategy
from src.strategies.tsi_strategy import TSIStrategy
from src.strategies.ultimate_oscillator import UltimateOscillatorStrategy
from src.strategies.williams_r_reversal import WilliamsRReversalStrategy

DATA_DIR = project_root / "data" / "raw"

STRATEGIES = {
    "CCI": CCIStrategy,
    "SMA_Cross": SMACrossoverStrategy,
    "Ichimoku": IchimokuCloudStrategy,
    "StochRSI": StochRSICrossoverStrategy,
    "TSI": TSIStrategy,
    "WilliamsR": WilliamsRReversalStrategy,
    "UltOsc": UltimateOscillatorStrategy,
    "AO": AwesomeOscillatorStrategy,
    "Chaikin": ChaikinOscillatorStrategy,
    "LinReg": LinearRegressionChannelStrategy,
}


def load_data(filepath: Path) -> pd.DataFrame:
    """Load OHLCV data from CSV."""
    df = pd.read_csv(filepath)
    for col in ["Datetime", "date", "Date", "timestamp"]:
        if col in df.columns:
            df = df.set_index(col)
            df.index = pd.to_datetime(df.index)
            break
    col_map = {c.lower(): c for c in df.columns}
    for std, variants in [
        ("Open", ["open"]),
        ("High", ["high"]),
        ("Low", ["low"]),
        ("Close", ["close"]),
        ("Volume", ["volume"]),
    ]:
        for v in variants:
            if v in col_map and col_map[v] != std:
                df[std] = df[col_map[v]]
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df[["Open", "High", "Low", "Close", "Volume"]]


def main() -> None:
    """Run multi-asset backtest."""
    files = sorted(DATA_DIR.glob("*.csv"))
    print(f"Found {len(files)} data files\n")

    results = []
    for f in files:
        try:
            df = load_data(f)
        except Exception as e:
            print(f"[{f.name}] SKIP: {e}")
            continue

        if len(df) < 200:
            print(f"[{f.name}] SKIP: only {len(df)} bars")
            continue

        print(f"[{f.name}] {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")
        for name, cls in STRATEGIES.items():
            try:
                bt = Backtest(df, cls, cash=1_000_000, commission=0.001, exclusive_orders=True)
                stats = bt.run()
                trades = int(stats.get("# Trades", 0))
                wr = float(stats.get("Win Rate [%]", 0))
                sharpe = float(stats.get("Sharpe Ratio", -999))
                pf = float(stats.get("Profit Factor", 0))
                ret = float(stats.get("Return [%]", 0))
                results.append(
                    {
                        "File": f.name,
                        "Strategy": name,
                        "Trades": trades,
                        "WR": wr,
                        "Sharpe": sharpe,
                        "PF": pf,
                        "Return": ret,
                    }
                )
                status = "PASS" if sharpe >= 0.3 and pf >= 1.0 else "FAIL"
                print(
                    f"  {name:12s}: {trades:5d} trades, {wr:5.1f}% WR, Sharpe {sharpe:6.2f}, PF {pf:.2f}, Ret {ret:6.1f}% [{status}]"
                )
            except Exception as e:
                print(f"  {name:12s}: ERROR - {e}")

    if results:
        print("\n" + "=" * 120)
        print(
            f"{'File':<25} {'Strategy':<12} {'Trades':>7} {'WR%':>6} {'Sharpe':>8} {'PF':>6} {'Return%':>8}"
        )
        print("=" * 120)
        for r in sorted(results, key=lambda x: x["Sharpe"], reverse=True):
            print(
                f"{r['File']:<25} {r['Strategy']:<12} {r['Trades']:>7} {r['WR']:>5.1f}% {r['Sharpe']:>8.2f} {r['PF']:>6.2f} {r['Return']:>7.1f}%"
            )


if __name__ == "__main__":
    main()
