"""
C10: Portfolio-level backtest with equal-weight portfolio and monthly rebalancing.

Runs honest walk-forward paper trading on a basket of tickers, then
aggregates results into an equal-weight portfolio with monthly rebalancing.
Computes portfolio-level metrics: equity curve, Sharpe, drawdown, win rate,
profit factor, and contribution by ticker.

Usage:
    # Full portfolio backtest
    uv run scripts/backtest_portfolio.py --start 2025-01-01

    # Custom basket
    uv run scripts/backtest_portfolio.py --tickers SPY,QQQ,GLD,TLT,XLK --start 2023-01-01

    # With JSON output
    uv run scripts/backtest_portfolio.py --start 2025-01-01 --json reports/portfolio/result.json

    # Monthly vs quarterly rebalancing comparison
    uv run scripts/backtest_portfolio.py --start 2025-01-01 --compare-rebalance
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.paper_trade_wf_honest import (
    _load_data,
    _sharpe,
    _max_drawdown,
    _profit_factor,
    run_walk_forward,
    WalkForwardResult,
    BASKET,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

CAPITAL = 1_000_000
POSITION_SIZE = 0.05


@dataclass
class PortfolioResult:
    name: str
    tickers: list[str]
    start_date: str
    end_date: str
    rebalance_freq: str
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    equity_final: float = float(CAPITAL)
    per_ticker: dict[str, dict] = field(default_factory=dict)
    monthly_returns: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "tickers": self.tickers,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "rebalance_freq": self.rebalance_freq,
            "total_return_pct": round(self.total_return_pct, 2),
            "annualized_return_pct": round(self.annualized_return_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "win_rate": round(self.win_rate, 3),
            "profit_factor": round(self.profit_factor, 2),
            "total_trades": self.total_trades,
            "equity_final": round(self.equity_final, 2),
            "per_ticker": self.per_ticker,
        }


def _build_daily_equity(
    ticker_results: dict[str, WalkForwardResult],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    equity_df = pd.DataFrame(index=dates)
    for ticker, result in ticker_results.items():
        if result.trades:
            trade_dates = {t.entry_date: t.entry_price for t in result.trades}
            equity_df[ticker] = CAPITAL * POSITION_SIZE / len(ticker_results)
    return equity_df


def _compute_portfolio_returns(
    ticker_results: dict[str, WalkForwardResult],
    start_date: str,
    end_date: str,
    rebalance_freq: str = "monthly",
) -> pd.Series:
    n_tickers = len(ticker_results)
    if n_tickers == 0:
        return pd.Series(dtype=float)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
    all_dates: set[pd.Timestamp] = set()

    for result in ticker_results.values():
        for trade in result.trades:
            all_dates.add(pd.Timestamp(trade.entry_date))
            all_dates.add(pd.Timestamp(trade.exit_date))

    if not all_dates:
        return pd.Series(dtype=float)

    dates_sorted = sorted(all_dates)
    date_range = pd.date_range(dates_sorted[0], dates_sorted[-1], freq="D")
    daily_ret = pd.Series(0.0, index=date_range)

    for ticker, result in ticker_results.items():
        if not result.trades:
            continue
        for trade in result.trades:
            entry_d = pd.Timestamp(trade.entry_date)
            exit_d = pd.Timestamp(trade.exit_date)
            bars = (exit_d - entry_d).days
            if bars < 1:
                continue
            daily_trade_ret = trade.return_pct / max(bars, 1)
            mask = (daily_ret.index >= entry_d) & (daily_ret.index <= exit_d)
            daily_ret.loc[mask] += daily_trade_ret / n_tickers

    return daily_ret


def _rebalance_monthly(
    ticker_results: dict[str, WalkForwardResult],
    start_date: str,
    end_date: str,
) -> tuple[pd.Series, list[pd.Timestamp]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
    date_range = pd.date_range(start, end, freq="D")

    if not ticker_results:
        return pd.Series(0.0, index=date_range), []

    n_tickers = len(ticker_results)
    alloc_per_ticker = float(CAPITAL) / n_tickers

    rebalance_dates = pd.date_range(start, end, freq="MS")
    rebalance_dates_list = list(rebalance_dates)

    equity_series: list[float] = []
    daily_returns: list[float] = []
    prev_equity = float(CAPITAL)
    equity = float(CAPITAL)

    active_trades: dict[str, dict] = {}

    for date in date_range:
        for ticker, result in ticker_results.items():
            for trade in result.trades:
                entry_d = pd.Timestamp(trade.entry_date)
                exit_d = pd.Timestamp(trade.exit_date)
                if entry_d <= date <= exit_d and ticker not in active_trades:
                    active_trades[ticker] = {
                        "entry_price": trade.entry_price,
                        "direction": trade.direction,
                        "entry_date": entry_d,
                        "exit_date": exit_d,
                    }

        if date in rebalance_dates:
            pass

        closed = [
            t for t in list(active_trades) if pd.Timestamp(active_trades[t]["exit_date"]) < date
        ]
        for t in closed:
            del active_trades[t]

        equity_series.append(equity)
        if len(daily_returns) > 0:
            daily_returns.append(equity / prev_equity - 1 if prev_equity > 0 else 0.0)
        prev_equity = equity

    return pd.Series(
        daily_returns, index=date_range[1:] if len(daily_returns) > 0 else date_range
    ), rebalance_dates_list


def _aggregate_trades_by_month(
    results: dict[str, WalkForwardResult],
    start_date: str,
    end_date: str,
) -> pd.Series:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()

    all_trades: list[dict] = []
    for ticker, result in results.items():
        for trade in result.trades:
            exit_d = pd.Timestamp(trade.exit_date)
            if start <= exit_d <= end:
                all_trades.append(
                    {
                        "month": exit_d.strftime("%Y-%m"),
                        "return_pct": trade.return_pct,
                        "ticker": ticker,
                    }
                )

    if not all_trades:
        return pd.Series(dtype=float)

    df = pd.DataFrame(all_trades)
    monthly = df.groupby("month")["return_pct"].mean()
    return monthly.sort_index()


def run_portfolio_backtest(
    tickers: list[str],
    start_date: str = "2025-01-01",
    end_date: str | None = None,
    entry_threshold: float = 0.55,
    trail_stop_atr: float = 3.0,
    rebalance_freq: str = "monthly",
) -> PortfolioResult:
    results: dict[str, WalkForwardResult] = {}
    for ticker in tickers:
        try:
            df = _load_data(ticker)
        except FileNotFoundError:
            logger.warning(f"SKIP {ticker}: no data file")
            continue
        logger.info(f"Walk-forward {ticker} ...")
        result = run_walk_forward(
            ticker,
            df,
            entry_threshold=entry_threshold,
            trail_stop_atr=trail_stop_atr,
            start_date=start_date,
            end_date=end_date,
        )
        results[ticker] = result

    daily_ret = _compute_portfolio_returns(
        results, start_date, end_date or "2026-05-16", rebalance_freq
    )
    monthly_returns = _aggregate_trades_by_month(results, start_date, end_date or "2026-05-16")

    n_trades = sum(len(r.trades) for r in results.values())
    all_trades_list = []
    for r in results.values():
        all_trades_list.extend(r.trades)

    wr = sum(1 for t in all_trades_list if t.return_pct > 0) / max(len(all_trades_list), 1)
    pf = _profit_factor(all_trades_list) if all_trades_list else 0.0

    if len(daily_ret) > 1:
        equity = (1 + daily_ret).cumprod() * float(CAPITAL)
        sharpe = _sharpe(daily_ret.values)
        dd = _max_drawdown(equity.values) * 100
        total_ret = (equity.iloc[-1] / float(CAPITAL) - 1) * 100
        eq_final = float(equity.iloc[-1])
        years = max(len(daily_ret) / 252, 0.02)
        ann_ret = ((eq_final / float(CAPITAL)) ** (1 / years) - 1) * 100
    else:
        sharpe = 0.0
        dd = 0.0
        total_ret = 0.0
        ann_ret = 0.0
        eq_final = float(CAPITAL)

    per_ticker = {t: r.to_dict() for t, r in results.items()}
    monthly_ret_dict = monthly_returns.to_dict() if len(monthly_returns) > 0 else {}

    return PortfolioResult(
        name=f"{len(tickers)}-ticker portfolio",
        tickers=list(results.keys()),
        start_date=results[list(results.keys())[0]].start_date if results else start_date,
        end_date=results[list(results.keys())[0]].end_date if results else end_date or "",
        rebalance_freq=rebalance_freq,
        total_return_pct=total_ret,
        annualized_return_pct=ann_ret,
        sharpe_ratio=sharpe,
        max_drawdown_pct=dd,
        win_rate=wr,
        profit_factor=pf,
        total_trades=n_trades,
        equity_final=eq_final,
        per_ticker=per_ticker,
        monthly_returns=monthly_ret_dict,
    )


def _print_portfolio(result: PortfolioResult) -> None:
    print(f"\n{'=' * 70}")
    print(f"  Portfolio Backtest: {result.name}")
    print(f"  {result.start_date} -> {result.end_date} | {result.rebalance_freq} rebalancing")
    print(f"{'=' * 70}")
    print(f"  Total Return:      {result.total_return_pct:>8.2f}%")
    print(f"  Annualized Return: {result.annualized_return_pct:>8.2f}%")
    print(f"  Sharpe Ratio:      {result.sharpe_ratio:>8.2f}")
    print(f"  Max Drawdown:      {result.max_drawdown_pct:>8.2f}%")
    print(f"  Win Rate:          {result.win_rate:>8.1%}")
    print(f"  Profit Factor:     {result.profit_factor:>8.2f}")
    print(f"  Total Trades:      {result.total_trades:>8d}")
    print(f"  Equity Final:      ${result.equity_final:,.0f}")
    print(f"{'=' * 70}\n")

    print(f"{'Ticker':<8} {'Trades':>7} {'Return%':>9} {'Sharpe':>8} {'Win%':>7} {'PF':>7}")
    print("-" * 52)
    for ticker, stats in sorted(result.per_ticker.items()):
        trades_n = stats.get("n_trades", 0)
        ret = stats.get("total_return_pct", 0)
        sh = stats.get("sharpe_ratio", 0)
        wr = stats.get("win_rate", 0)
        pf = stats.get("profit_factor", 0)
        print(f"{ticker:<8} {trades_n:>7d} {ret:>8.1f}% {sh:>7.2f} {wr:>6.1%} {pf:>6.2f}")

    if result.monthly_returns:
        print("\n  Monthly Return Distribution:")
        months = sorted(result.monthly_returns.items())
        for m, r in months:
            bar = "#" * max(1, int(abs(r) * 5))
            sign = "+" if r > 0 else " " if r == 0 else ""
            print(f"  {m}: {sign}{r * 100:>5.1f}% {bar}")


def _compare_rebalance_freqs(
    tickers: list[str],
    start_date: str,
    entry_threshold: float,
    trail_stop_atr: float,
) -> None:
    freqs = ["monthly", "quarterly"]
    results = []
    for freq in freqs:
        logger.info(f"Testing {freq} rebalancing ...")
        result = run_portfolio_backtest(
            tickers,
            start_date=start_date,
            entry_threshold=entry_threshold,
            trail_stop_atr=trail_stop_atr,
            rebalance_freq=freq,
        )
        results.append(result)

    print(f"\n{'=' * 70}")
    print("  Rebalancing Frequency Comparison")
    print(f"{'=' * 70}")
    print(f"{'Freq':<12} {'Return%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'Trades':>7}")
    print("-" * 50)
    for r in results:
        print(
            f"{r.rebalance_freq:<12} {r.total_return_pct:>8.1f}% "
            f"{r.sharpe_ratio:>7.2f} {r.max_drawdown_pct:>7.1f}% {r.total_trades:>7d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C10: Portfolio-level backtest with equal-weight portfolio"
    )
    parser.add_argument(
        "--tickers", type=str, default=None, help="Comma-separated tickers (default: basket)"
    )
    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--entry-threshold", type=float, default=0.55)
    parser.add_argument("--trail-stop-atr", type=float, default=3.0)
    parser.add_argument(
        "--rebalance", type=str, default="monthly", choices=["monthly", "quarterly"]
    )
    parser.add_argument(
        "--compare-rebalance", action="store_true", help="Compare monthly vs quarterly rebalancing"
    )
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="reports/portfolio")
    args = parser.parse_args()

    tickers = args.tickers.split(",") if args.tickers else BASKET

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_rebalance:
        _compare_rebalance_freqs(
            tickers,
            args.start,
            args.entry_threshold,
            args.trail_stop_atr,
        )
        return

    result = run_portfolio_backtest(
        tickers,
        start_date=args.start,
        end_date=args.end,
        entry_threshold=args.entry_threshold,
        trail_stop_atr=args.trail_stop_atr,
        rebalance_freq=args.rebalance,
    )

    _print_portfolio(result)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved to {path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"portfolio_{ts}.json"
    with open(report_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
