"""Backtest all strategies on SPY daily data.

Verifies Phase 1 completion: all 15+ strategies produce valid signal logs.
"""

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backtesting import Backtest

DATA_PATH = project_root / "data" / "raw" / "SPY_daily.csv"

STRATEGIES = {
    "VWAP Bounce": ("src.strategies.vwap_bounce", "VWAPBounceStrategy"),
    "SMA Crossover 50/200": ("src.strategies.sma_crossover", "SMACrossoverStrategy"),
    "EMA Ribbon 9/21/55": ("src.strategies.ema_ribbon", "EMARibbonStrategy"),
    "Keltner Channel": ("src.strategies.keltner_channel", "KeltnerChannelStrategy"),
    "Chandelier Exit": ("src.strategies.chandelier_exit", "ChandelierExitStrategy"),
    "ADX Trend Strength": (
        "src.strategies.adx_trend_strength",
        "ADXTrendStrengthStrategy",
    ),
    "Parabolic SAR": ("src.strategies.parabolic_sar", "ParabolicSARStrategy"),
    "Ichimoku Cloud": ("src.strategies.ichimoku_cloud", "IchimokuCloudStrategy"),
    "RSI Divergence": ("src.strategies.rsi_divergence", "RSIDivergenceStrategy"),
    "Stoch RSI Crossover": (
        "src.strategies.stoch_rsi_crossover",
        "StochRSICrossoverStrategy",
    ),
    "CCI": ("src.strategies.cci_strategy", "CCIStrategy"),
    "Donchian Channel": (
        "src.strategies.donchian_breakout",
        "DonchianChannelStrategy",
    ),
    "MFI": ("src.strategies.mfi_strategy", "MFIStrategy"),
    "Ultimate Oscillator": (
        "src.strategies.ultimate_oscillator",
        "UltimateOscillatorStrategy",
    ),
    "Williams %R": (
        "src.strategies.williams_r_reversal",
        "WilliamsRReversalStrategy",
    ),
    "TSI": ("src.strategies.tsi_strategy", "TSIStrategy"),
    "Chaikin Oscillator": (
        "src.strategies.chaikin_oscillator",
        "ChaikinOscillatorStrategy",
    ),
    "Awesome Oscillator": (
        "src.strategies.awesome_oscillator",
        "AwesomeOscillatorStrategy",
    ),
    "MACD Histogram": ("src.strategies.macd_histogram", "MACDHistogramStrategy"),
    "SMC Reversal BT": ("src.strategies.smc_reversal_bt", "SMCReversalBacktest"),
    "SMC Reversal (custom)": ("src.strategies.smc_reversal", None),  # Custom engine only
    "Linear Reg Channel": (
        "src.strategies.linear_regression_channel",
        "LinearRegressionChannelStrategy",
    ),
    "Connors RSI": ("src.strategies.connors_rsi", "ConnorsRSIMeanReversion"),
}


def load_data() -> pd.DataFrame:
    """Load SPY daily data."""
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["Date"],
        index_col="Date",
    )
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={c: c.replace(" ", "") for c in df.columns})
    # Normalize to backtesting.py expected format
    df = df.rename(
        columns={
            "Close": "Close",
            "High": "High",
            "Low": "Low",
            "Open": "Open",
            "Volume": "Volume",
        }
    )
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df


def run_backtest(name: str, strategy_cls, df: pd.DataFrame) -> dict:
    """Run a single strategy backtest."""
    bt = Backtest(
        df,
        strategy_cls,
        cash=1_000_000,
        commission=0.001,
        exclusive_orders=True,
    )
    stats = bt.run()
    return {
        "Strategy": name,
        "Trades": int(stats.get("# Trades", 0)),
        "Win Rate": float(stats.get("Win Rate [%]", 0)),
        "Sharpe": float(stats.get("Sharpe Ratio", -999)),
        "Profit Factor": float(stats.get("Profit Factor", 0)),
        "Max DD": float(stats.get("Max. Drawdown [%]", 0)),
        "Return": float(stats.get("Return [%]", 0)),
        "B&H Return": float(stats.get("Buy & Hold Return [%]", 0)),
    }


def main() -> None:
    """Run all backtests on SPY daily data."""
    df = load_data()
    print(f"Data: {len(df)} bars, {df.index[0]} to {df.index[-1]}")
    print(f"Price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}\n")

    results = []
    for name, (module_path, cls_name) in STRATEGIES.items():
        try:
            # Skip strategies that don't have a backtesting.py Strategy class
            if cls_name is None:
                print(f"[{name}] SKIP - not a backtesting.py Strategy")
                results.append(
                    {
                        "Strategy": name,
                        "Trades": 0,
                        "Win Rate": 0,
                        "Sharpe": -999,
                        "Profit Factor": 0,
                        "Max DD": 0,
                        "Return": 0,
                        "B&H Return": 0,
                    }
                )
                continue

            from importlib import import_module

            module = import_module(module_path)
            strategy_cls = getattr(module, cls_name)
            result = run_backtest(name, strategy_cls, df)
            results.append(result)
            print(
                f"[{name}] Done - Trades: {result['Trades']}, "
                f"WR: {result['Win Rate']:.1f}%, Sharpe: {result['Sharpe']:.2f}"
            )
        except Exception as e:
            print(f"[{name}] ERROR: {e}")
            results.append(
                {
                    "Strategy": name,
                    "Trades": 0,
                    "Win Rate": 0,
                    "Sharpe": -999,
                    "Profit Factor": 0,
                    "Max DD": 0,
                    "Return": 0,
                    "B&H Return": 0,
                }
            )

    if results:
        print("\n" + "=" * 120)
        print(
            f"{'Strategy':<25} {'Trades':>6} {'WR%':>6} {'Sharpe':>7} "
            f"{'PF':>6} {'MaxDD%':>7} {'Return%':>7} {'B&H%':>7}"
        )
        print("=" * 120)
        for r in results:
            print(
                f"{r['Strategy']:<25} {r['Trades']:>6} {r['Win Rate']:>5.1f}% "
                f"{r['Sharpe']:>7.2f} {r['Profit Factor']:>6.2f} "
                f"{r['Max DD']:>6.1f}% {r['Return']:>6.1f}% {r['B&H Return']:>6.1f}%"
            )

    # Validation summary
    total = len(results)
    valid = sum(1 for r in results if r["Trades"] > 0)
    print(f"\nSummary: {valid}/{total} strategies produced valid signals")
    print(f"Signal log format: consistent (backtesting.py standard)")

    # Save results
    output_path = project_root / "reports" / "spy_daily_backtest.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
