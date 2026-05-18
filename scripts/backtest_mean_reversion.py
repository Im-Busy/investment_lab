"""Backtest MeanReversionStrategy and RegimeRouterStrategy.

Usage:
    # Mean-reversion only (default: ADX < 20 gated)
    uv run scripts/backtest_mean_reversion.py SPY --start 2016-01-01

    # Mean-reversion without regime gate (test all markets)
    uv run scripts/backtest_mean_reversion.py SPY --no-regime-gate

    # Regime-routed (trending + ranging)
    uv run scripts/backtest_mean_reversion.py SPY --router

    # Compare all three: trend-only, reversion-only, routed
    uv run scripts/backtest_mean_reversion.py SPY --compare
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backtesting import Backtest
from src.strategies.mean_reversion_strategy import MeanReversionStrategy
from src.strategies.regime_router_strategy import RegimeRouterStrategy


def load_data(symbol: str) -> pd.DataFrame:
    path = Path(f"data/raw/{symbol}_daily.csv")
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol} at {path}")
    df = pd.read_csv(path, parse_dates=True, index_col=0).dropna()
    df.columns = [c.capitalize() for c in df.columns]
    return df


def format_stats(stats, label: str) -> str:
    return (
        f"{label:20s}  "
        f"Return={stats['Return [%]']:7.2f}%  "
        f"Sharpe={stats['Sharpe Ratio']:6.2f}  "
        f"Trades={stats['# Trades']:4d}  "
        f"Win={stats['Win Rate [%]']:6.1f}%  "
        f"PF={stats['Profit Factor']:5.2f}  "
        f"MaxDD={stats['Max. Drawdown [%]']:7.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest mean-reversion and regime-routed strategies"
    )
    parser.add_argument("symbol", type=str, help="Ticker symbol (e.g. SPY)")
    parser.add_argument("--start", type=str, default="2016-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--cash", type=float, default=100_000)
    parser.add_argument(
        "--no-regime-gate", action="store_true", help="Disable ADX regime gate (test all markets)"
    )
    parser.add_argument(
        "--router", action="store_true", help="Use RegimeRouterStrategy (trend+reversion)"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare trend-only vs reversion-only vs routed"
    )
    parser.add_argument(
        "--tp-atr", type=float, default=2.0, help="Take-profit ATR multiplier (default 2.0)"
    )
    parser.add_argument(
        "--sl-atr", type=float, default=3.0, help="Stop-loss ATR multiplier (default 3.0)"
    )
    parser.add_argument(
        "--max-hold", type=int, default=10, help="Max hold bars for mean-reversion (default 10)"
    )
    args = parser.parse_args()

    df = load_data(args.symbol)
    if args.start:
        df = df[df.index >= args.start]
    if args.end:
        df = df[df.index <= args.end]

    results = []

    if args.compare:
        # Trend-only (RulesFirstStrategy if available, or simplified)
        from src.strategies.rules_first_strategy import RulesFirstStrategy

        bt_trend = Backtest(df, RulesFirstStrategy, cash=args.cash, commission=0.001)
        trend_stats = bt_trend.run()
        results.append(("RulesFirst (trend)", trend_stats))

    # Mean-reversion
    bt_mr = Backtest(df, MeanReversionStrategy, cash=args.cash, commission=0.001)
    mr_stats = bt_mr.run(
        require_ranging=not args.no_regime_gate,
        tp_atr_mult=args.tp_atr,
        sl_atr_mult=args.sl_atr,
        max_hold_bars=args.max_hold,
    )
    results.append(("MeanReversion", mr_stats))

    # Regime-routed
    if args.router or args.compare:
        bt_router = Backtest(df, RegimeRouterStrategy, cash=args.cash, commission=0.001)
        router_stats = bt_router.run(
            mr_tp_atr=args.tp_atr,
            mr_sl_atr=args.sl_atr,
            mr_max_hold=args.max_hold,
        )
        results.append(("RegimeRouter", router_stats))

    # B&H
    bh_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    sep = "=" * 80
    dash = "-" * 80
    print(f"\n{sep}")
    print(
        f"Mean-Reversion Backtest: {args.symbol}  ({args.start} to {args.end or df.index[-1].strftime('%Y-%m-%d')})"
    )
    print(f"{sep}")
    print(f"B&H Return: {bh_return:.2f}%")
    print(f"{dash}")
    print(
        f"{'Strategy':20s}  {'Return':>8s}  {'Sharpe':>6s}  {'Trades':>5s}  {'Win%':>6s}  {'PF':>5s}  {'MaxDD':>8s}"
    )
    print(f"{dash}")
    for label, stats in results:
        print(format_stats(stats, label))
    print(f"{dash}")


if __name__ == "__main__":
    main()
