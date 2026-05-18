"""Pairs trading backtest runner.

Usage:
    # Single pair backtest
    uv run scripts/backtest_pairs.py KO-PEP --start 2016-01-01 --end 2024-12-31

    # All pre-built pairs side-by-side
    uv run scripts/backtest_pairs.py --compare

    # Auto-discover pairs from data/raw/ by correlation
    uv run scripts/backtest_pairs.py --auto-pairs --min-correlation 0.7 --top-n 5

    # JSON output for machine reading
    uv run scripts/backtest_pairs.py KO-PEP --json

    # Sweep entry z-score
    uv run scripts/backtest_pairs.py KO-PEP --sweep-entry 1.5,2.0,2.5,3.0

    # With ATR trailing exit
    uv run scripts/backtest_pairs.py KO-PEP --atr-exit 2.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.pairs_trading_strategy import PairsTradingStrategy

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

PRE_BUILT_PAIRS: list[tuple[str, str]] = [
    ("KO", "PEP"),
    ("CVX", "XOM"),
    ("UNP", "CSX"),
    ("XLK", "QQQ"),
    ("SPY", "IWM"),
    ("JNJ", "MRK"),
    ("NEM", "GOLD"),
    ("DUK", "SO"),
    ("OXY", "COP"),
]


@dataclass
class PairResult:
    pair: str
    return_pct: float
    sharpe: float
    trades: int
    win_rate: float
    profit_factor: float
    max_dd: float
    avg_bars_held: float
    exposure_pct: float
    coint_pvalue: float
    avg_correlation: float
    annual_return: float


def load_instrument(symbol: str) -> pd.DataFrame:
    path = Path(f"data/raw/{symbol}_daily.csv")
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol} at {path}")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df.dropna()
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            df[col] = 0 if col == "Volume" else df.iloc[:, 0]
    return df


def build_pair_df(
    sym_a: str,
    sym_b: str,
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.DataFrame, float, float]:
    """Build DataFrame with instrument A as primary OHLCV + Close_B column.

    Returns (df, coint_pvalue, avg_correlation).
    """
    df_a = load_instrument(sym_a)
    df_b = load_instrument(sym_b)

    common_idx = df_a.index.intersection(df_b.index)
    df_a = df_a.loc[common_idx].sort_index()
    df_b = df_b.loc[common_idx].sort_index()

    if start:
        df_a = df_a[df_a.index >= start]
        df_b = df_b[df_b.index >= start]
    if end:
        df_a = df_a[df_a.index <= end]
        df_b = df_b[df_b.index <= end]

    common_idx = df_a.index.intersection(df_b.index)
    df_a = df_a.loc[common_idx]
    df_b = df_b.loc[common_idx]

    close_a = df_a["Close"].values.astype(float)
    close_b = df_b["Close"].values.astype(float)

    log_a = np.log(close_a)
    log_b = np.log(np.maximum(close_b, 1e-10))

    _, coint_pvalue, _ = coint(log_a, log_b, trend="c", maxlag=int(np.sqrt(len(log_a))))

    lookback = 252
    corr_roll = np.full(len(log_a), np.nan)
    for i in range(lookback, len(log_a)):
        x = log_a[i - lookback : i]
        y = log_b[i - lookback : i]
        sx = np.std(x)
        sy = np.std(y)
        if sx > 0 and sy > 0:
            corr_roll[i] = np.corrcoef(x, y)[0, 1]
    avg_correlation = float(np.nanmean(corr_roll))

    result = pd.DataFrame(
        {
            "Open": df_a["Open"].values,
            "High": df_a["High"].values,
            "Low": df_a["Low"].values,
            "Close": df_a["Close"].values,
            "Volume": df_a["Volume"].values,
            "Close_B": df_b["Close"].values,
        },
        index=df_a.index,
    )

    return result, coint_pvalue, avg_correlation


def run_backtest(
    sym_a: str,
    sym_b: str,
    cash: float = 10_000,
    start: str | None = None,
    end: str | None = None,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    stop_z: float = 3.0,
    lookback: int = 252,
    min_correlation: float = 0.7,
    atr_exit: float = 0.0,
) -> PairResult | None:
    from backtesting import Backtest

    try:
        pair_df, coint_pvalue, avg_corr = build_pair_df(sym_a, sym_b, start, end)
    except FileNotFoundError as e:
        _log.warning("Skipping %s-%s: %s", sym_a, sym_b, e)
        return None

    if len(pair_df) < lookback + 60:
        _log.warning("Skipping %s-%s: %d bars (need %d)", sym_a, sym_b, len(pair_df), lookback + 60)
        return None

    bt = Backtest(
        pair_df,
        PairsTradingStrategy,
        cash=cash,
        commission=0.001,
        exclusive_orders=True,
        finalize_trades=True,
    )

    stats = bt.run(
        lookback=lookback,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
        min_correlation=min_correlation,
        atr_exit_mult=atr_exit,
    )

    trades_df = stats._trades if hasattr(stats, "_trades") else pd.DataFrame()
    num_trades = len(trades_df) if not trades_df.empty else 0

    win_rate = 0.0
    profit_factor = 0.0
    avg_bars = 0.0

    if num_trades > 0 and "PnL" in trades_df.columns:
        winners = trades_df[trades_df["PnL"] > 0]
        losers = trades_df[trades_df["PnL"] < 0]
        win_rate = len(winners) / num_trades * 100 if num_trades else 0
        total_win = winners["PnL"].sum() if len(winners) else 0
        total_loss = abs(losers["PnL"].sum()) if len(losers) else 0
        profit_factor = total_win / total_loss if total_loss > 0 else 0.0
    if num_trades > 0 and "EntryTime" in trades_df.columns and "ExitTime" in trades_df.columns:
        bars_held = (
            pd.to_datetime(trades_df["ExitTime"]) - pd.to_datetime(trades_df["EntryTime"])
        ).dt.days
        avg_bars = bars_held.mean() if len(bars_held) else 0.0

    total_days = len(pair_df)
    years = total_days / 252 if total_days > 0 else 1
    annual_return = ((1 + stats["Return [%]"] / 100) ** (1 / years) - 1) * 100 if years > 0 else 0

    return PairResult(
        pair=f"{sym_a}-{sym_b}",
        return_pct=float(stats["Return [%]"]),
        sharpe=float(stats.get("Sharpe Ratio", 0) or 0),
        trades=num_trades,
        win_rate=round(win_rate, 1),
        profit_factor=round(profit_factor, 2),
        max_dd=float(stats.get("Max. Drawdown [%]", 0) or 0),
        avg_bars_held=round(avg_bars, 1),
        exposure_pct=float(stats.get("Exposure Time [%]", 0) or 0),
        coint_pvalue=round(coint_pvalue, 4),
        avg_correlation=round(avg_corr, 4),
        annual_return=round(annual_return, 2),
    )


def find_auto_pairs(
    min_corr: float = 0.7, top_n: int = 5, start_years: int = 3
) -> list[tuple[str, str]]:
    data_dir = Path("data/raw")
    tickers = sorted(
        [
            f.stem.replace("_daily", "")
            for f in data_dir.glob("*_daily.csv")
            if "_daily" in f.name and not f.name.startswith("CN_") and not f.name.startswith("HK_")
        ]
    )

    scored: list[tuple[float, str, str]] = []
    for i, t1 in enumerate(tickers):
        for t2 in tickers[i + 1 :]:
            try:
                df1 = load_instrument(t1)
                df2 = load_instrument(t2)
                common = df1.index.intersection(df2.index)
                if len(common) < 252 * start_years:
                    continue
                c1 = df1.loc[common, "Close"].values.astype(float)
                c2 = df2.loc[common, "Close"].values.astype(float)
                l1 = np.log(c1)
                l2 = np.log(np.maximum(c2, 1e-10))
                corr = np.corrcoef(l1[-252:], l2[-252:])[0, 1]
                if abs(corr) < min_corr:
                    continue
                _, pvalue, _ = coint(l1, l2, trend="c", maxlag=int(np.sqrt(len(l1))))
                scored.append((pvalue, t1, t2))
            except Exception:
                continue

    scored.sort()
    return [(t1, t2) for _, t1, t2 in scored[:top_n]]


def compare_pairs(
    pairs: list[tuple[str, str]],
    start: str | None = None,
    end: str | None = None,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    stop_z: float = 3.0,
    lookback: int = 252,
    min_correlation: float = 0.7,
    atr_exit: float = 0.0,
) -> list[PairResult]:
    results = []
    for sym_a, sym_b in pairs:
        print(f"  Backtesting {sym_a}-{sym_b}...")
        result = run_backtest(
            sym_a,
            sym_b,
            start=start,
            end=end,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            lookback=lookback,
            min_correlation=min_correlation,
            atr_exit=atr_exit,
        )
        if result:
            results.append(result)
    return results


def print_results_table(results: list[PairResult]) -> None:
    if not results:
        print("No results to display.")
        return
    print(
        f"\n{'Pair':<12} {'Return%':>8} {'Sharpe':>7} {'Trades':>6} "
        f"{'Win%':>6} {'PF':>6} {'MaxDD%':>7} {'Expo%':>6} "
        f"{'CoPval':>8} {'AvgCorr':>8} {'AnnRet%':>8}"
    )
    print("-" * 95)
    for r in results:
        print(
            f"{r.pair:<12} {r.return_pct:>8.2f} {r.sharpe:>7.2f} {r.trades:>6} "
            f"{r.win_rate:>5.1f} {r.profit_factor:>6.2f} {r.max_dd:>7.2f} "
            f"{r.exposure_pct:>5.1f} {r.coint_pvalue:>8.4f} {r.avg_correlation:>8.4f} "
            f"{r.annual_return:>8.2f}"
        )
    avg_sharpe = np.mean([r.sharpe for r in results])
    avg_ret = np.mean([r.return_pct for r in results])
    total_trades = sum(r.trades for r in results)
    print("-" * 95)
    print(f"{'AVERAGE':<12} {avg_ret:>8.2f} {avg_sharpe:>7.2f} {total_trades:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairs trading backtest runner")
    parser.add_argument("pair", nargs="?", help="Pair as TICKER-TICKER (e.g. KO-PEP)")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--entry-z", type=float, default=2.0, help="Entry z-score threshold")
    parser.add_argument("--exit-z", type=float, default=0.0, help="Exit z-score threshold")
    parser.add_argument("--stop-z", type=float, default=3.0, help="Stop-loss z-score threshold")
    parser.add_argument("--lookback", type=int, default=252, help="Rolling window size")
    parser.add_argument(
        "--min-correlation", type=float, default=0.7, help="Min rolling correlation"
    )
    parser.add_argument(
        "--atr-exit", type=float, default=0.0, help="ATR multiplier for trailing stop"
    )
    parser.add_argument("--compare", action="store_true", help="Compare all pre-built pairs")
    parser.add_argument(
        "--auto-pairs", action="store_true", help="Auto-discover pairs from data/raw/"
    )
    parser.add_argument("--top-n", type=int, default=5, help="Top N pairs for auto-pairs")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--sweep-entry", help="Comma-separated entry z thresholds to sweep")
    args = parser.parse_args()

    if args.sweep_entry:
        if not args.pair or "-" not in args.pair:
            print("ERROR: --sweep-entry requires a pair like KO-PEP")
            sys.exit(1)
        sym_a, sym_b = args.pair.split("-")
        entry_values = [float(x.strip()) for x in args.sweep_entry.split(",")]
        print(f"\nSweeping entry z-score for {sym_a}-{sym_b}: {entry_values}")
        print(
            f"{'EntryZ':>8} {'Return%':>8} {'Sharpe':>7} {'Trades':>6} {'Win%':>6} {'PF':>6} {'MaxDD%':>7}"
        )
        print("-" * 60)
        for ez in entry_values:
            r = run_backtest(
                sym_a,
                sym_b,
                start=args.start,
                end=args.end,
                entry_z=ez,
                exit_z=args.exit_z,
                stop_z=args.stop_z,
                lookback=args.lookback,
                min_correlation=args.min_correlation,
                atr_exit=args.atr_exit,
            )
            if r:
                print(
                    f"{ez:>8.1f} {r.return_pct:>8.2f} {r.sharpe:>7.2f} {r.trades:>6} "
                    f"{r.win_rate:>5.1f} {r.profit_factor:>6.2f} {r.max_dd:>7.2f}"
                )
        return

    if args.compare:
        print("\nPairs Trading Comparison (all pre-built pairs)")
        print(f"Period: {args.start or 'all'} -> {args.end or 'all'}")
        results = compare_pairs(
            PRE_BUILT_PAIRS,
            start=args.start,
            end=args.end,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            stop_z=args.stop_z,
            lookback=args.lookback,
            min_correlation=args.min_correlation,
            atr_exit=args.atr_exit,
        )
        print_results_table(results)
        if args.json:
            print(json.dumps([r.__dict__ for r in results], indent=2))
        return

    if args.auto_pairs:
        print(f"\nAuto-discovering pairs (min corr={args.min_correlation}, top {args.top_n})...")
        pairs = find_auto_pairs(min_corr=args.min_correlation, top_n=args.top_n)
        if not pairs:
            print("No pairs found meeting criteria.")
            return
        print(f"Found {len(pairs)} pairs: {pairs}")
        results = compare_pairs(
            pairs,
            start=args.start,
            end=args.end,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            stop_z=args.stop_z,
            lookback=args.lookback,
            min_correlation=args.min_correlation,
            atr_exit=args.atr_exit,
        )
        print_results_table(results)
        if args.json:
            print(json.dumps([r.__dict__ for r in results], indent=2))
        return

    if not args.pair or "-" not in args.pair:
        parser.print_help()
        sys.exit(1)

    sym_a, sym_b = args.pair.split("-")
    print(f"\nBacktesting pair: {sym_a}-{sym_b}")
    result = run_backtest(
        sym_a,
        sym_b,
        start=args.start,
        end=args.end,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stop_z=args.stop_z,
        lookback=args.lookback,
        min_correlation=args.min_correlation,
        atr_exit=args.atr_exit,
    )
    if result:
        print_results_table([result])
        if args.json:
            print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
