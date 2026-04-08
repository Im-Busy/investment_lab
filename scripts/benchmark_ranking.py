"""
Benchmark Ranking Script

Tests every implemented strategy on every available ticker,
ranks them by return %, and outputs a clean leaderboard.
"""

import sys
from pathlib import Path

import pandas as pd
from backtesting import Backtest

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import all strategies
from src.strategies.adx_trend_strength import ADXTrendStrengthStrategy
from src.strategies.awesome_oscillator import AwesomeOscillatorStrategy
from src.strategies.cci_strategy import CCIStrategy
from src.strategies.chaikin_oscillator import ChaikinOscillatorStrategy
from src.strategies.chandelier_exit import ChandelierExitStrategy
from src.strategies.ema_ribbon import EMARibbonStrategy
from src.strategies.ichimoku_cloud import IchimokuCloudStrategy
from src.strategies.keltner_channel import KeltnerChannelStrategy
from src.strategies.linear_regression_channel import LinearRegressionChannelStrategy
from src.strategies.parabolic_sar import ParabolicSARStrategy
from src.strategies.rsi_divergence import RSIDivergenceStrategy
from src.strategies.sma_crossover import SMACrossoverStrategy
from src.strategies.stoch_rsi_crossover import StochRSICrossoverStrategy
from src.strategies.tsi_strategy import TSIStrategy
from src.strategies.ultimate_oscillator import UltimateOscillatorStrategy
from src.strategies.vwap_bounce import VWAPBounceStrategy
from src.strategies.williams_r_reversal import WilliamsRReversalStrategy

DATA_DIR = project_root / "data" / "raw"

STRATEGIES = {
    "SMA_Cross": SMACrossoverStrategy,
    "EMA_Ribbon": EMARibbonStrategy,
    "VWAP_Bounce": VWAPBounceStrategy,
    "Keltner": KeltnerChannelStrategy,
    "Chandelier": ChandelierExitStrategy,
    "ADX": ADXTrendStrengthStrategy,
    "ParabolicSAR": ParabolicSARStrategy,
    "Ichimoku": IchimokuCloudStrategy,
    "LinReg": LinearRegressionChannelStrategy,
    "RSI_Div": RSIDivergenceStrategy,
    "StochRSI": StochRSICrossoverStrategy,
    "CCI": CCIStrategy,
    "WilliamsR": WilliamsRReversalStrategy,
    "TSI": TSIStrategy,
    "UltOsc": UltimateOscillatorStrategy,
    "AO": AwesomeOscillatorStrategy,
    "Chaikin": ChaikinOscillatorStrategy,
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
    """Run benchmark and produce ranked leaderboard."""
    files = sorted(DATA_DIR.glob("*.csv"))
    print(
        f"Testing {len(STRATEGIES)} strategies on {len(files)} assets = {len(STRATEGIES) * len(files)} backtests\n"
    )

    results = []
    total = len(STRATEGIES) * len(files)
    count = 0

    for f in files:
        try:
            df = load_data(f)
        except Exception as e:
            print(f"SKIP {f.name}: {e}")
            continue
        if len(df) < 200:
            print(f"SKIP {f.name}: only {len(df)} bars")
            continue

        for name, cls in STRATEGIES.items():
            count += 1
            print(f"[{count}/{total}] {f.name} x {name}...", end=" ")
            try:
                bt = Backtest(df, cls, cash=1_000_000, commission=0.001, exclusive_orders=True)
                stats = bt.run()
                trades = int(stats.get("# Trades", 0))
                wr = float(stats.get("Win Rate [%]", 0))
                sharpe = float(stats.get("Sharpe Ratio", -999))
                pf = float(stats.get("Profit Factor", 0))
                ret = float(stats.get("Return [%]", 0))
                bnh = float(stats.get("Buy & Hold Return [%]", 0))
                max_dd = float(stats.get("Max. Drawdown [%]", 0))
                results.append(
                    {
                        "Asset": f.name,
                        "Strategy": name,
                        "Trades": trades,
                        "WR": wr,
                        "Sharpe": sharpe,
                        "PF": pf,
                        "Return": ret,
                        "BnH": bnh,
                        "MaxDD": max_dd,
                    }
                )
                status = "PASS" if sharpe >= 0.3 and pf >= 1.0 else "fail"
                print(f"Ret={ret:6.1f}% Sharpe={sharpe:6.2f} [{status}]")
            except Exception as e:
                print(f"ERROR - {e}")

    # Produce ranked leaderboard
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("Return", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 130)
    print(
        f"{'Rank':>4} {'Asset':<25} {'Strategy':<14} {'Trades':>6} {'WR%':>6} {'Sharpe':>7} {'PF':>5} {'Return%':>8} {'BnH%':>7} {'MaxDD%':>7}"
    )
    print("=" * 130)
    for rank, (_, r) in enumerate(df_results.iterrows(), 1):
        marker = " [PASS]" if r["Sharpe"] >= 0.3 and r["PF"] >= 1.0 else ""
        print(
            f"{rank:>4} {r['Asset']:<25} {r['Strategy']:<14} {r['Trades']:>6} {r['WR']:>5.1f}% {r['Sharpe']:>7.2f} {r['PF']:>5.2f} {r['Return']:>7.1f}% {r['BnH']:>6.1f}% {r['MaxDD']:>6.1f}%{marker}"
        )

    # Summary: top 10
    print("\n=== TOP 10 BY RETURN ===")
    for rank, (_, r) in enumerate(df_results.head(10).iterrows(), 1):
        print(
            f"  {rank}. {r['Asset']:25s} {r['Strategy']:14s}  Return={r['Return']:7.1f}%  Sharpe={r['Sharpe']:6.2f}  PF={r['PF']:.2f}  Trades={r['Trades']}"
        )

    # Summary: passing strategies
    passing = df_results[(df_results["Sharpe"] >= 0.3) & (df_results["PF"] >= 1.0)]
    print(f"\n=== PASSING (Sharpe>=0.3, PF>=1.0): {len(passing)} combos ===")
    for i, r in passing.iterrows():
        print(
            f"  {r['Asset']:25s} {r['Strategy']:14s}  Return={r['Return']:7.1f}%  Sharpe={r['Sharpe']:6.2f}  PF={r['PF']:.2f}  Trades={r['Trades']}"
        )

    # Save to CSV
    out_path = project_root / "reports" / "benchmark_ranking.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
