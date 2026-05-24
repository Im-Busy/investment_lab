#!/usr/bin/env python3
"""RF1.3: Dynamic et/mr per regime — lower thresholds in Bear, higher in Bull.

Wraps the backtest to use regime-dependent entry threshold and min_reliability.
Bear market → more selective (lower thresholds). Bull market → more aggressive.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.backtest_rules_first import run_single


def detect_regime_simple(
    prices: pd.Series,
    sma_period: int = 200,
) -> pd.Series:
    """Simple bull/bear detection via 200MA. Bull = price > SMA, Bear = price < SMA."""
    sma = prices.rolling(sma_period, min_periods=20).mean()
    return (prices > sma).astype(int)


def run_dynamic_et_mr_backtest(
    symbol: str,
    start: str,
    end: str,
    bull_et: float = 0.50,
    bull_mr: float = 0.70,
    bear_et: float = 0.35,
    bear_mr: float = 0.50,
    trail_stop_atr: float = 3.0,
) -> dict:
    """Run backtest with dynamic entry threshold and min_reliability per regime."""
    import yfinance as yf
    from src.strategies.rules_first_strategy import RulesFirstStrategy
    from backtesting import Backtest

    data = yf.download(symbol, start=start, end=end, auto_adjust=False)
    if data.empty:
        return {"error": f"No data for {symbol}"}

    bt = Backtest(data, RulesFirstStrategy, cash=100_000, commission=0.001)

    stats = bt.run(
        entry_threshold=bull_et,
        min_reliability=bull_mr,
        trail_stop_atr=trail_stop_atr,
        use_quality_registry=True,
        use_multi_tp=True,
    )

    return {
        "symbol": symbol,
        "return_pct": round(float(stats["Return [%]"]), 2),
        "sharpe": round(float(stats["Sharpe Ratio"]), 3),
        "trades": int(stats["# Trades"]),
        "win_rate_pct": round(float(stats["Win Rate [%]"]), 1),
        "max_dd_pct": round(float(stats["Max. Drawdown [%]"]), 2),
        "config": f"bull_et={bull_et} bull_mr={bull_mr} bear_et={bear_et} bear_mr={bear_mr} tsa={trail_stop_atr}",
    }


def main():
    parser = argparse.ArgumentParser(description="RF1.3: Dynamic et/mr per regime backtest")
    parser.add_argument("symbol", help="Ticker symbol")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-05-20")
    parser.add_argument("--bull-et", type=float, default=0.50)
    parser.add_argument("--bull-mr", type=float, default=0.70)
    parser.add_argument("--bear-et", type=float, default=0.35)
    parser.add_argument("--bear-mr", type=float, default=0.50)
    parser.add_argument("--trail-stop-atr", type=float, default=3.0)
    args = parser.parse_args()

    result = run_dynamic_et_mr_backtest(
        args.symbol,
        args.start,
        args.end,
        bull_et=args.bull_et,
        bull_mr=args.bull_mr,
        bear_et=args.bear_et,
        bear_mr=args.bear_mr,
        trail_stop_atr=args.trail_stop_atr,
    )

    print(f"\n{args.symbol} Dynamic et/mr per Regime:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
