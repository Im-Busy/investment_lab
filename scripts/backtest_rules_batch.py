"""
AB3: Batch backtest Rules-First strategy across all instruments.

Runs the production Rules-First config (mr=0.70, et=0.55) across a curated
basket of instruments covering multiple asset classes and sectors.

Usage:
    # Run on default basket (IS + OOS)
    uv run scripts/backtest_rules_batch.py

    # Custom basket
    uv run scripts/backtest_rules_batch.py --symbols SPY,QQQ,GLD,TLT,BTC_USD

    # OOS only
    uv run scripts/backtest_rules_batch.py --start 2025-01-01 --end 2026-05-14

    # IS sweep on basket
    uv run scripts/backtest_rules_batch.py --sweep-entry 0.55,0.60,0.65,0.70
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.rules_first_strategy import RulesFirstStrategy

logger = logging.getLogger(__name__)

# Default basket: indices, sector ETFs, commodities, single stocks, crypto, forex
BASKET_DEFAULT = [
    # Major indices
    "SPY",
    "QQQ",
    "IWM",
    # Sector ETFs
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    # Commodities / Bonds (non-equity regimes)
    "GLD",
    "TLT",
    # Stocks (one per sector from SECTOR_MAP)
    "KO",
    "JPM",
    "XOM",
    "JNJ",
    "SO",
    # Crypto
    "BTC_USD",
    # Forex
    "EURUSD_X",
]

BASKET_LABELS: dict[str, str] = {
    "SPY": "Index - Large Cap",
    "QQQ": "Index - NASDAQ 100",
    "IWM": "Index - Small Cap",
    "XLK": "Sector - Technology",
    "XLF": "Sector - Financials",
    "XLE": "Sector - Energy",
    "XLV": "Sector - Healthcare",
    "GLD": "Commodity - Gold",
    "TLT": "Bond - 20Y Treasury",
    "KO": "Stock - Consumer (KO)",
    "JPM": "Stock - Financials (JPM)",
    "XOM": "Stock - Energy (XOM)",
    "JNJ": "Stock - Healthcare (JNJ)",
    "SO": "Stock - Utilities (SO)",
    "BTC_USD": "Crypto - Bitcoin",
    "EURUSD_X": "Forex - EUR/USD",
}


def load_data(symbol: str) -> pd.DataFrame:
    path = Path(f"data/raw/{symbol}_daily.csv")
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol} at {path}")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df.dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0 if col == "Volume" else df.iloc[:, 0]
    df.columns = [c.capitalize() for c in df.columns]
    return df


def run_single(
    symbol: str,
    cash: float = 10_000,
    start: str | None = None,
    end: str | None = None,
    entry_threshold: float = 0.55,
    exit_threshold: float = 0.30,
    trail_stop_atr: float = 3.0,
    min_reliability: float = 0.70,
    confluence_bonus: float = 0.10,
    volume_confirm: bool = True,
) -> dict | None:
    from backtesting import Backtest

    try:
        df = load_data(symbol)
    except FileNotFoundError:
        print(f"  SKIP {symbol}: no data file")
        return None

    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]

    if len(df) < 60:
        print(f"  SKIP {symbol}: only {len(df)} bars")
        return None

    bt = Backtest(
        df,
        RulesFirstStrategy,
        cash=cash,
        commission=0.001,
        exclusive_orders=True,
        finalize_trades=True,
    )

    try:
        stats = bt.run(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            trail_stop_atr=trail_stop_atr,
            min_reliability=min_reliability,
            confluence_bonus=confluence_bonus,
            volume_confirm=volume_confirm,
        )
    except Exception as e:
        print(f"  FAIL {symbol}: {e}")
        return None

    n_bars = len(df)
    result = {
        "symbol": symbol,
        "label": BASKET_LABELS.get(symbol, ""),
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
        "bars": n_bars,
        "return_pct": round(stats["Return [%]"], 2),
        "sharpe": round(stats["Sharpe Ratio"], 3),
        "sortino": round(stats.get("Sortino Ratio", 0), 3),
        "calmar": round(stats.get("Calmar Ratio", 0), 3),
        "max_dd_pct": round(stats["Max. Drawdown [%]"], 2),
        "trades": stats["# Trades"],
        "win_rate_pct": round(stats["Win Rate [%]"], 1),
        "profit_factor": round(stats["Profit Factor"], 2),
        "exposure_pct": round(stats["Exposure Time [%]"], 1),
        "annual_return_pct": round(stats.get("Return (Ann.) [%]", 0), 2),
    }
    return result


def print_table(results: list[dict]) -> None:
    if not results:
        print("No results to display.")
        return

    cols = [
        "symbol",
        "label",
        "return_pct",
        "sharpe",
        "sortino",
        "calmar",
        "max_dd_pct",
        "trades",
        "win_rate_pct",
        "profit_factor",
        "exposure_pct",
    ]
    header = [
        "Symbol",
        "Category",
        "Return%",
        "Sharpe",
        "Sortino",
        "Calmar",
        "MaxDD%",
        "Trades",
        "Win%",
        "PF",
        "Exp%",
    ]

    print()
    widths = [max(len(h), 8) for h in header]
    for r in results:
        for i, c in enumerate(cols):
            v = r.get(c, "")
            widths[i] = max(widths[i], len(str(v)))

    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for r in results:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if c in ("sharpe", "sortino", "calmar", "profit_factor") and isinstance(v, float):
                vals.append(f"{v:.3f}")
            elif isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        print(fmt.format(*vals))
    print()


def print_insights(results: list[dict]) -> None:
    """Print human-readable insights from batch results."""
    if not results:
        return

    print("=" * 70)
    print(" INSIGHTS")
    print("=" * 70)

    # Sort by Sharpe
    sorted_by_sharpe = sorted(results, key=lambda r: r["sharpe"], reverse=True)
    positive = [r for r in results if r["sharpe"] > 0]
    negative = [r for r in results if r["sharpe"] <= 0]
    profitable = [r for r in results if r["return_pct"] > 0]

    print(f"\nTotal instruments tested: {len(results)}")
    print(
        f"Positive Sharpe: {len(positive)}/{len(results)} ({100 * len(positive) / len(results):.0f}%)"
    )
    print(
        f"Profitable (return > 0): {len(profitable)}/{len(results)} ({100 * len(profitable) / len(results):.0f}%)"
    )

    # Top 5 performers
    print("\n--- Top 5 by Sharpe ---")
    for r in sorted_by_sharpe[:5]:
        print(
            f"  {r['symbol']:8s} ({r['label']:30s}) Sharpe={r['sharpe']:.3f}  "
            f"Return={r['return_pct']:.1f}%  Trades={r['trades']}"
        )

    # Bottom 5
    if len(sorted_by_sharpe) > 5:
        print("\n--- Bottom 5 by Sharpe ---")
        for r in sorted_by_sharpe[-5:]:
            print(
                f"  {r['symbol']:8s} ({r['label']:30s}) Sharpe={r['sharpe']:.3f}  "
                f"Return={r['return_pct']:.1f}%  Trades={r['trades']}"
            )

    # Trade count distribution
    trades = [r["trades"] for r in results]
    if trades:
        print("\n--- Trade Count ---")
        print(f"  Min: {min(trades)}  Max: {max(trades)}  Mean: {sum(trades) / len(trades):.1f}")
        zero_trades = [r for r in results if r["trades"] == 0]
        few_trades = [r for r in results if 1 <= r["trades"] <= 5]
        if zero_trades:
            print(
                f"  Zero trades ({len(zero_trades)}): {', '.join(r['symbol'] for r in zero_trades)}"
            )
        if few_trades:
            print(f"  1-5 trades ({len(few_trades)}): {', '.join(r['symbol'] for r in few_trades)}")

    # Win rate analysis
    win_rates = [r["win_rate_pct"] for r in results if r["trades"] > 0]
    if win_rates:
        mean_wr = sum(win_rates) / len(win_rates)
        print("\n--- Win Rate (profitable instruments only) ---")
        print(f"  Mean: {mean_wr:.1f}%")
        high_wr = [r for r in results if r["win_rate_pct"] >= 55 and r["trades"] > 0]
        if high_wr:
            print(f"  >=55% win rate: {', '.join(r['symbol'] for r in high_wr)}")

    # Category analysis
    categories: dict[str, list[dict]] = {}
    for r in results:
        label = r.get("label", "Other")
        cat = label.split(" - ")[0] if " - " in label else "Other"
        categories.setdefault(cat, []).append(r)

    print("\n--- By Category ---")
    for cat, items in sorted(categories.items()):
        avg_sharpe = sum(r["sharpe"] for r in items) / len(items)
        avg_return = sum(r["return_pct"] for r in items) / len(items)
        avg_trades = sum(r["trades"] for r in items) / len(items)
        print(
            f"  {cat:20s}: n={len(items)}  avg Sharpe={avg_sharpe:.3f}  "
            f"avg Return={avg_return:.1f}%  avg Trades={avg_trades:.1f}"
        )

    # Correlation with SPY? Only if SPY is in results
    spy_result = next((r for r in results if r["symbol"] == "SPY"), None)
    if spy_result:
        print("\n--- SPY Reference ---")
        print(f"  SPY Sharpe: {spy_result['sharpe']:.3f}")
        beat_spy = [
            r for r in results if r["sharpe"] > spy_result["sharpe"] and r["symbol"] != "SPY"
        ]
        if beat_spy:
            print(f"  Beat SPY Sharpe: {', '.join(r['symbol'] for r in beat_spy)}")
        else:
            print("  No instrument beat SPY Sharpe in this period.")

    # Data sufficiency
    low_bars = [r for r in results if r["bars"] < 252]
    if low_bars:
        print("\n--- Warning: Low data (<252 bars) ---")
        for r in low_bars:
            print(f"  {r['symbol']}: {r['bars']} bars")

    # Summary conclusion
    print("\n--- Summary ---")
    if len(positive) >= 0.5 * len(results):
        print("  Rules-First generalizes across asset classes. Patterns are universal.")
    elif len(positive) >= 0.3 * len(results):
        print("  Mixed results. Rules-First works best on liquid/trending instruments.")
    else:
        print("  Rules-First struggles outside SPY. Consider instrument-specific tuning.")

    # Check if crypto/forex need higher mr
    crypto_forex = [r for r in results if r["symbol"] in ("BTC_USD", "EURUSD_X")]
    if crypto_forex:
        cf_positive = [r for r in crypto_forex if r["sharpe"] > 0]
        if cf_positive:
            print("  Crypto/Forex show positive results — rules-first handles 24/7 assets.")
        else:
            print(
                "  Crypto/Forex negative — may need higher min_reliability or different ATR calibration."
            )

    print("=" * 70)


def sweep_entries(
    symbols: list[str],
    thresholds: list[float],
    start: str | None,
    end: str | None,
    **kwargs,
) -> dict[str, list[dict]]:
    """Run entry threshold sweep across multiple symbols."""
    all_results: dict[str, list[dict]] = {}
    for symbol in symbols:
        symbol_results = []
        for et in thresholds:
            r = run_single(symbol, start=start, end=end, entry_threshold=et, **kwargs)
            if r:
                r["entry_threshold"] = et
                symbol_results.append(r)
                print(
                    f"  {symbol} et={et:.2f}: Sharpe {r['sharpe']:.3f}  "
                    f"Return {r['return_pct']:.1f}%  Trades {r['trades']}"
                )
            else:
                print(f"  {symbol} et={et:.2f}: FAILED")
        if symbol_results:
            all_results[symbol] = symbol_results
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch backtest Rules-First across instruments")
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated ticker list (default: 15-instrument basket)",
    )
    parser.add_argument("--start", default="2016-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument("--cash", type=float, default=10_000, help="Initial cash")
    parser.add_argument(
        "--entry-threshold", type=float, default=0.55, help="Entry signal threshold"
    )
    parser.add_argument("--exit-threshold", type=float, default=0.30, help="Exit signal threshold")
    parser.add_argument(
        "--trail-stop-atr", type=float, default=3.0, help="ATR multiplier for trailing stop"
    )
    parser.add_argument(
        "--min-reliability", type=float, default=0.70, help="Minimum pattern reliability"
    )
    parser.add_argument("--confluence-bonus", type=float, default=0.10, help="Confluence bonus")
    parser.add_argument(
        "--no-volume-confirm", action="store_true", help="Disable volume confirmation"
    )
    parser.add_argument("--sweep-entry", help="Comma-separated entry thresholds")
    parser.add_argument("--json-output", help="Path to save JSON results")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary insights, skip per-symbol output",
    )
    parser.add_argument(
        "--optimize-portfolio",
        action="store_true",
        help="Run GA portfolio optimization on batch results",
    )
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else BASKET_DEFAULT

    kwargs = dict(
        exit_threshold=args.exit_threshold,
        trail_stop_atr=args.trail_stop_atr,
        min_reliability=args.min_reliability,
        confluence_bonus=args.confluence_bonus,
        volume_confirm=not args.no_volume_confirm,
    )

    print(f"\n{'=' * 70}")
    print(" RULES-FIRST BATCH BACKTEST")
    print(f" Period: {args.start} -> {args.end}")
    print(
        f" Config: mr={args.min_reliability:.1f}  et={args.entry_threshold:.2f}  "
        f"trail={args.trail_stop_atr}  confl={args.confluence_bonus}"
    )
    print(f" Instruments: {len(symbols)}")
    print(f"{'=' * 70}\n")

    if args.sweep_entry:
        thresholds = [float(t.strip()) for t in args.sweep_entry.split(",")]
        print(f"Sweeping entry thresholds: {thresholds}\n")
        sweep_results = sweep_entries(symbols, thresholds, args.start, args.end, **kwargs)
        for symbol, res in sweep_results.items():
            print(f"\n  --- {symbol} sweep ---")
            for r in res:
                print(
                    f"  et={r['entry_threshold']:.2f}: Sharpe {r['sharpe']:.3f}  "
                    f"Return {r['return_pct']:.1f}%  Trades {r['trades']}"
                )
        if args.json_output:
            Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_output).write_text(json.dumps(sweep_results, indent=2))
        return

    all_results = []
    for i, symbol in enumerate(symbols):
        print(f"[{i + 1}/{len(symbols)}] {symbol} ...", end=" ", flush=True)
        r = run_single(
            symbol,
            cash=args.cash,
            start=args.start,
            end=args.end,
            entry_threshold=args.entry_threshold,
            **kwargs,
        )
        if r:
            all_results.append(r)
            print(f"Sharpe={r['sharpe']:.3f}  Return={r['return_pct']:.1f}%  Trades={r['trades']}")
        else:
            print("SKIPPED")

    if not args.summary_only:
        print_table(all_results)

    print_insights(all_results)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "config": {
                "start": args.start,
                "end": args.end,
                "entry_threshold": args.entry_threshold,
                "min_reliability": args.min_reliability,
                "trail_stop_atr": args.trail_stop_atr,
                "confluence_bonus": args.confluence_bonus,
            },
            "results": all_results,
            "summary": {
                "n_total": len(results := all_results),
                "n_positive_sharpe": sum(1 for r in results if r["sharpe"] > 0),
                "n_profitable": sum(1 for r in results if r["return_pct"] > 0),
                "n_traded": sum(1 for r in results if r["trades"] > 0),
            },
        }
        output_path.write_text(json.dumps(output, indent=2))
        print(f"Results saved to {args.json_output}")

    if args.optimize_portfolio and all_results:
        _run_portfolio_optimization(
            all_results, args.start or "2016-01-01", args.end or "2026-05-16"
        )


def _run_portfolio_optimization(results: list[dict], start: str, end: str) -> None:
    """H5: Run GA portfolio optimization on batch backtest results."""
    print(f"\n{'=' * 70}")
    print(" PORTFOLIO OPTIMIZATION (GA via Eiten)")
    print(f"{'=' * 70}\n")

    # Build signals dict from batch results
    # Filter to instruments with positive Sharpe
    passed = [r for r in results if r.get("sharpe", 0) > 0 and r.get("trades", 0) > 0]
    if len(passed) < 3:
        print("  Insufficient instruments with positive Sharpe (< 3). Skipping.")
        return

    symbols = [r["symbol"] for r in passed]
    signals = {r["symbol"]: r.get("sharpe", 0) for r in passed}
    print(f"  Selected {len(symbols)}/{len(results)} instruments with positive Sharpe:")
    for sym in symbols:
        print(f"    {sym}: Sharpe={signals[sym]:.3f}")

    try:
        from src.portfolio.eiten_builder import PortfolioBuilder
        from src.portfolio.eiten_adapters.ga import optimize_ga

        builder = PortfolioBuilder()
        result = builder.optimize_from_signals(signals, strategy="ga")
        print(f"\n  GA Portfolio Allocation ({len(result.weights)} assets):")
        for sym, wt in sorted(result.weights.items(), key=lambda x: -x[1]):
            bar = "#" * int(wt * 40)
            print(f"    {sym:<8} {wt:>6.1%}  {bar}")

        print(f"\n  Expected Sharpe: {result.sharpe:.3f}")
        if result.max_dd is not None:
            print(f"  Expected MaxDD:  {result.max_dd:.1%}")

    except Exception as e:
        print(f"  Portfolio optimization failed: {e}")
        logger.debug("GA optimization error", exc_info=True)


if __name__ == "__main__":
    main()
