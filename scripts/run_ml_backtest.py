"""
Run ML strategy backtest on one or more tickers using backtesting.py.

Usage:
    uv run scripts/run_ml_backtest.py SPY
    uv run scripts/run_ml_backtest.py SPY --model models/pattern_classifier_v3_SPY_20260511_164601.pkl
    uv run scripts/run_ml_backtest.py SPY,QQQ,XLK,D,SO --compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.ml_strategy import MLStrategy


def load_data(symbol: str) -> pd.DataFrame:
    path = Path(f"data/raw/{symbol}_daily.csv")
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol}")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df.dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0 if col == "Volume" else df.iloc[:, 0]
    # backtesting.py expects specific column name capitalization
    df.columns = [c.capitalize() for c in df.columns]
    return df


def run_single(
    symbol: str,
    model_path: str,
    cash: float,
    start: str | None,
    entry_threshold: float = 0.50,
    vol_gate: float | None = None,
    confirm: int = 1,
    trail_stop: bool = False,
    trail_atr: float = 3.0,
    conviction: bool = False,
    multi_tp: bool = False,
) -> dict:
    from backtesting import Backtest

    df = load_data(symbol)
    if start:
        df = df[df.index >= start]

    bt = Backtest(
        df,
        MLStrategy,
        cash=cash,
        commission=0.001,
        exclusive_orders=True,
    )

    stats = bt.run(
        model_path=model_path,
        ticker=symbol,
        entry_threshold=entry_threshold,
        exit_threshold=entry_threshold * 0.7,
        tp_atr_mult=3.0,
        sl_atr_mult=1.5,
        risk_pct=0.02,
        vol_gate_threshold=vol_gate,
        confirm_bars=confirm,
        use_trail_stop=trail_stop,
        trail_atr_mult=trail_atr,
        conviction_scale=conviction,
        use_multi_tp=multi_tp,
    )

    return {
        "symbol": symbol,
        "start": str(stats["Start"]),
        "end": str(stats["End"]),
        "duration": stats["Duration"],
        "exposure": stats["Exposure Time [%]"],
        "equity_final": round(stats["Equity Final [$]"], 2),
        "equity_peak": round(stats["Equity Peak [$]"], 2),
        "return_pct": round(stats["Return [%]"], 2),
        "buy_hold_return": round(stats["Buy & Hold Return [%]"], 2),
        "return_annual": round(stats["Return (Ann.) [%]"], 2),
        "volatility_annual": round(stats["Volatility (Ann.) [%]"], 2),
        "sharpe": round(stats["Sharpe Ratio"], 2),
        "sortino": round(stats["Sortino Ratio"], 2),
        "calmar": round(stats["Calmar Ratio"], 2),
        "max_drawdown": round(stats["Max. Drawdown [%]"], 2),
        "avg_drawdown": round(stats["Avg. Drawdown [%]"], 2),
        "win_rate": round(stats["Win Rate [%]"], 2),
        "best_trade": round(stats["Best Trade [%]"], 2),
        "worst_trade": round(stats["Worst Trade [%]"], 2),
        "avg_trade": round(stats["Avg. Trade [%]"], 2),
        "num_trades": stats["# Trades"],
        "profit_factor": round(stats["Profit Factor"], 2),
        "expectancy": round(stats["Expectancy [%]"], 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ML strategy backtest")
    parser.add_argument(
        "symbols",
        type=str,
        help="Comma-separated ticker symbols (e.g. SPY,QQQ,XLK)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/pattern_classifier_v3_SPY_20260511_164601.pkl",
        help="Path to trained model pickle",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=100_000,
        help="Initial capital",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Default: use all available data.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run all tickers and print comparison table",
    )
    # ── C7 Refinement parameters ──
    parser.add_argument(
        "--vol-gate",
        type=float,
        default=None,
        help="Volatility gate threshold (e.g. 1.5). Max vol_regime for entry.",
    )
    parser.add_argument(
        "--confirm",
        type=int,
        default=1,
        help="Consecutive bars above entry_threshold required (default 1=off).",
    )
    parser.add_argument(
        "--trail-stop",
        action="store_true",
        help="Use ATR trailing stop instead of fixed take-profit.",
    )
    parser.add_argument(
        "--trail-atr",
        type=float,
        default=3.0,
        help="ATR multiplier for trailing stop distance (default 3.0).",
    )
    parser.add_argument(
        "--conviction",
        action="store_true",
        help="Scale position size by conviction (prob above entry_threshold).",
    )
    parser.add_argument(
        "--multi-tp",
        action="store_true",
        help="Use multi-TP exit: partial TP at 50% of target with breakeven SL.",
    )
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=0.50,
        help="Minimum ML probability to enter (default 0.50). Lower = more trades.",
    )
    args = parser.parse_args()

    tickers = [t.strip() for t in args.symbols.split(",")]

    if args.compare:
        results = []
        for ticker in tickers:
            print(f"Running {ticker}...")
            try:
                r = run_single(
                    ticker,
                    args.model,
                    args.cash,
                    args.start,
                    args.entry_threshold,
                    args.vol_gate,
                    args.confirm,
                    args.trail_stop,
                    args.trail_atr,
                    args.conviction,
                    args.multi_tp,
                )
                results.append(r)
            except Exception as e:
                print(f"  FAILED: {e}")
                continue

        if not results:
            print("No results.")
            return

        df = pd.DataFrame(results)
        # Print comparison table
        print("\n" + "=" * 120)
        print("ML Strategy Backtest Comparison")
        print("=" * 120)
        cols = [
            "symbol",
            "return_pct",
            "buy_hold_return",
            "sharpe",
            "max_drawdown",
            "win_rate",
            "num_trades",
            "profit_factor",
            "avg_trade",
            "return_annual",
        ]
        print(df[cols].to_string(index=False))

        # Highlight winners
        winners = df[df["return_pct"] > df["buy_hold_return"]]
        if len(winners) > 0:
            print(f"\nBeat buy-and-hold: {', '.join(winners['symbol'].tolist())}")
        print(f"\nMean Sharpe: {df['sharpe'].mean():.2f}")
        print(f"Mean Win Rate: {df['win_rate'].mean():.1f}%")
        print(f"Total Trades: {df['num_trades'].sum()}")

        # Save
        out_path = Path("reports/ml_backtest/comparison.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")
    else:
        ticker = tickers[0]
        r = run_single(
            ticker,
            args.model,
            args.cash,
            args.start,
            args.entry_threshold,
            args.vol_gate,
            args.confirm,
            args.trail_stop,
            args.trail_atr,
            args.conviction,
            args.multi_tp,
        )
        print("\n" + "=" * 80)
        print(f"ML Strategy Backtest: {ticker}")
        print(
            f"C7 features: vol_gate={args.vol_gate} confirm={args.confirm}"
            f" trail={args.trail_stop} conviction={args.conviction}"
        )
        print("=" * 80)
        for k, v in r.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
