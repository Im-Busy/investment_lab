"""Backtest script for all Eterna strategies on BTC 1H data."""

import sys
from pathlib import Path

import pandas as pd
from backtesting import Backtest

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

DATA_PATH = project_root / "data" / "raw" / "BTC_USD_1h.csv"

STRATEGIES = {
    "VWAP Bounce": ("src.strategies.vwap_bounce", "VWAPBounceStrategy"),
    "SMA Crossover 50/200": ("src.strategies.sma_crossover", "SMACrossoverStrategy"),
    "EMA Ribbon 9/21/55": ("src.strategies.ema_ribbon", "EMARibbonStrategy"),
    "Keltner Channel": ("src.strategies.keltner_channel", "KeltnerChannelStrategy"),
    "Chandelier Exit": ("src.strategies.chandelier_exit", "ChandelierExitStrategy"),
    "ADX Trend Strength": ("src.strategies.adx_trend_strength", "ADXTrendStrengthStrategy"),
    "Parabolic SAR": ("src.strategies.parabolic_sar", "ParabolicSARStrategy"),
    "Ichimoku Cloud": ("src.strategies.ichimoku_cloud", "IchimokuCloudStrategy"),
    "RSI Divergence": ("src.strategies.rsi_divergence", "RSIDivergenceStrategy"),
    "Stoch RSI Crossover": ("src.strategies.stoch_rsi_crossover", "StochRSICrossoverStrategy"),
    "CCI": ("src.strategies.cci_strategy", "CCIStrategy"),
    "Donchian Channel": ("src.strategies.donchian_breakout", "DonchianChannelStrategy"),
    # MFI skipped - Volume=0 in BTC 1H data
}


def load_data() -> pd.DataFrame:
    """Load BTC 1H data."""
    df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"], index_col="Datetime")
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df


def run_backtest(name: str, strategy_cls, df: pd.DataFrame) -> dict:
    """Run a single strategy backtest."""
    bt = Backtest(df, strategy_cls, cash=1_000_000, commission=0.001, exclusive_orders=True)
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
    """Run all backtests."""
    df = load_data()
    print(f"Data: {len(df)} bars, {df.index[0]} to {df.index[-1]}")
    print(f"Price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}\n")

    results = []
    for name, (module_path, cls_name) in STRATEGIES.items():
        try:
            from importlib import import_module

            module = import_module(module_path)
            strategy_cls = getattr(module, cls_name)
            result = run_backtest(name, strategy_cls, df)
            results.append(result)
            print(
                f"[{name}] Done - Trades: {result['Trades']}, WR: {result['Win Rate']:.1f}%, Sharpe: {result['Sharpe']:.2f}"
            )
        except Exception as e:
            print(f"[{name}] ERROR: {e}")

    if results:
        print("\n" + "=" * 100)
        print(
            f"{'Strategy':<30} {'Trades':>7} {'WR%':>7} {'Sharpe':>8} {'PF':>7} {'MaxDD%':>8} {'Return%':>8} {'B&H%':>8}"
        )
        print("=" * 100)
        for r in results:
            print(
                f"{r['Strategy']:<30} {r['Trades']:>7} {r['Win Rate']:>6.1f}% {r['Sharpe']:>8.2f} {r['Profit Factor']:>7.2f} {r['Max DD']:>7.1f}% {r['Return']:>7.1f}% {r['B&H Return']:>7.1f}%"
            )


if __name__ == "__main__":
    main()
