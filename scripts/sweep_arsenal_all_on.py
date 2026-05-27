"""
Sweep parameter grid across production basket tickers with ALL ON flags.

Tests variations of: max_loss_pct × min_confluence × entry_threshold
All signal enhancers ON: voting + catalog + divergence + wm_bollinger + garch_atr + signal_strength + kelly

Usage:
    uv run scripts/sweep_arsenal_all_on.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np

os.environ["TQDM_DISABLE"] = "1"

logging.getLogger("backtesting").setLevel(logging.ERROR)
logging.getLogger("backtesting.backtesting").setLevel(logging.ERROR)
logging.getLogger("backtesting.lib").setLevel(logging.ERROR)
logging.getLogger("backtesting._plotting").setLevel(logging.ERROR)
logging.getLogger("src.strategies.rules_first_strategy").setLevel(logging.ERROR)
logging.getLogger("src.ml.garch_forecaster").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.backtest_rules_first import run_single

PRODUCTION_TICKERS = [
    "XLK",
    "XLE",
    "GLD",
    "SPY",
    "SLV",
    "QQQ",
    "NUE",
    "STLD",
    "HAL",
    "MPC",
    "EOG",
    "INTC",
    "AMD",
    "LMT",
    "JNJ",
    "MRK",
    "NEM",
]

ALL_ON_KWARGS = dict(
    use_voting_signal=True,
    use_rules_catalog=True,
    use_divergence=True,
    use_wm_bollinger=True,
    use_garch_atr=True,
    use_signal_strength_sizing=True,
    use_kelly_sizing=True,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep arsenal with ALL ON + varied stats")
    parser.add_argument("--start", default="2025-01-01", help="Start date (default OOS 2025)")
    parser.add_argument("--end", default=None, help="End date")
    parser.add_argument("--output", default="reports/sweep_arsenal_all_on.json", help="JSON output")
    parser.add_argument(
        "--md-output", default="reports/sweep_arsenal_all_on.md", help="Markdown output"
    )
    parser.add_argument("--max-tickers", type=int, default=0, help="Max tickers (0=all)")
    parser.add_argument("--single-ticker", default="", help="Run only this ticker")
    args = parser.parse_args()

    tickers = [args.single_ticker] if args.single_ticker else PRODUCTION_TICKERS
    if args.max_tickers > 0:
        tickers = tickers[: args.max_tickers]

    max_loss_pcts = [0.06, 0.10, 0.15]
    min_confluences = [0, 1, 2]
    entry_thresholds = [0.45, 0.55, 0.65]

    total_combos = len(max_loss_pcts) * len(min_confluences) * len(entry_thresholds)
    print(
        f"Grid: {len(max_loss_pcts)} max_loss × {len(min_confluences)} min_confl × {len(entry_thresholds)} et = {total_combos} combos/ticker"
    )
    print(f"Tickers: {len(tickers)}")
    print(f"Total backtests: {total_combos * len(tickers)}")
    print(f"Period: {args.start} -> {args.end or 'today'}")
    print()

    all_ticker_bests: list[dict] = []
    all_results: list[dict] = []

    for ti, ticker in enumerate(tickers):
        print(f"[{ti + 1}/{len(tickers)}] {ticker} ... ", end="", flush=True)

        ticker_results: list[dict] = []
        for max_loss in max_loss_pcts:
            for min_confl in min_confluences:
                for et in entry_thresholds:
                    try:
                        result = run_single(
                            ticker,
                            start=args.start,
                            end=args.end,
                            entry_threshold=et,
                            max_loss_pct=max_loss,
                            max_concurrent_orders=3,
                            min_confluence=min_confl,
                            **ALL_ON_KWARGS,
                        )
                        result.update(
                            {
                                "max_loss_pct": max_loss,
                                "min_confluence": min_confl,
                                "entry_threshold": et,
                            }
                        )
                        ticker_results.append(result)
                    except Exception as e:
                        print(
                            f"\n  WARNING: {ticker} et={et} max_loss={max_loss} min_confl={min_confl} FAILED: {e}"
                        )

        if not ticker_results:
            print("NO RESULTS")
            continue

        best = max(ticker_results, key=lambda r: r.get("sharpe", -999) or -999)
        best_sharpe = best.get("sharpe") or -999
        best_return = best.get("return_pct") or 0
        best_trades = best.get("trades") or 0
        best_win = best.get("win_rate_pct") or 0
        best_pf = best.get("profit_factor") or 0
        best_dd = best.get("max_dd_pct") or 0

        all_ticker_bests.append(
            {
                "symbol": ticker,
                "best_max_loss_pct": best["max_loss_pct"],
                "best_min_confluence": best["min_confluence"],
                "best_entry_threshold": best["entry_threshold"],
                "sharpe": best_sharpe,
                "return_pct": best_return,
                "trades": best_trades,
                "win_rate_pct": best_win,
                "profit_factor": best_pf,
                "max_dd_pct": best_dd,
            }
        )
        all_results.extend(ticker_results)

        config = f"et={best['entry_threshold']} max_loss={best['max_loss_pct']} min_confl={best['min_confluence']}"
        print(
            f"BEST: Sharpe={best_sharpe:.3f} Ret={best_return:.1f}% Tr={best_trades} WR={best_win:.0f}% DD={best_dd:.1f}%  [{config}]"
        )

    print(f"\n{'=' * 80}")
    print("FULL LEADERBOARD (ranked by Sharpe)")
    print(f"{'=' * 80}")

    sorted_bests = sorted(all_ticker_bests, key=lambda r: r["sharpe"] or -999, reverse=True)
    pos = sum(1 for r in sorted_bests if (r["sharpe"] or 0) > 0)
    mean_sh = float(np.mean([r["sharpe"] or 0 for r in sorted_bests]))

    for i, b in enumerate(sorted_bests):
        sh = b["sharpe"]
        flag = " ^" if i < pos else ""
        print(
            f"  {i + 1:2d}. {b['symbol']:8s}  Sharpe={sh:+.3f}  Ret={b['return_pct']:+.1f}%  "
            f"Tr={b['trades']:3d}  WR={b['win_rate_pct']:.0f}%  "
            f"PF={b['profit_factor']:.2f}  DD={b['max_dd_pct']:.1f}%  "
            f"[et={b['best_entry_threshold']} max_loss={b['best_max_loss_pct']} min_confl={b['best_min_confluence']}]{flag}"
        )

    print(
        f"\n  Mean Sharpe: {mean_sh:+.3f}  |  Positive: {pos}/{len(sorted_bests)} ({100 * pos // max(len(sorted_bests), 1)}%)"
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    clean_results = []
    for r in all_results:
        clean = {k: v for k, v in r.items() if k != "_equity_curve"}
        clean_results.append(clean)
    Path(args.output).write_text(
        json.dumps(
            {
                "ticker_bests": all_ticker_bests,
                "all_results": clean_results,
                "grid": {
                    "max_loss_pcts": max_loss_pcts,
                    "min_confluences": min_confluences,
                    "entry_thresholds": entry_thresholds,
                },
            },
            indent=2,
        )
    )

    _write_md(
        all_ticker_bests,
        sorted_bests,
        pos,
        mean_sh,
        max_loss_pcts,
        min_confluences,
        entry_thresholds,
        args,
    )

    print(f"\nJSON → {args.output}")
    print(f"MD  → {args.md_output}")


def _write_md(bests, sorted_bests, pos, mean_sh, mlp, mc, ets, args):
    lines = [
        "---",
        "generated: 2026-05-27",
        f"period: {args.start} → {args.end or 'today'}",
        "config: ALL ON (voting + catalog + divergence + wm_bollinger + garch_atr + signal_strength + kelly)",
        f"grid: max_loss_pct={mlp}, min_confluence={mc}, entry_threshold={ets}",
        "---",
        "",
        "# Arsenal Sweep — ALL ON with Varied Stats",
        "",
        f"> **Period:** {args.start} → {args.end or 'today'} (OOS)",
        "> **Config:** ALL signal enhancers ON — Voting + RulesCatalog + Divergence + WM_Bollinger + GARCH_ATR + SignalStrength + Kelly",
        "> **Defaults:** max_concurrent_orders=3, trail_stop_atr=3.0, min_reliability=0.40",
        f"> **Grid:** {len(mlp)} max_loss × {len(mc)} min_confl × {len(ets)} et = {len(mlp) * len(mc) * len(ets)} combos per ticker",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Tickers | {len(bests)} |",
        f"| Positive OOS Sharpe | {pos}/{len(bests)} ({100 * pos // max(len(bests), 1)}%) |",
        f"| Mean OOS Sharpe | {mean_sh:+.3f} |",
        "",
        "## Leaderboard — Best Per Ticker (ranked by Sharpe)",
        "",
        "| # | Symbol | Sharpe | Return% | Trades | Win% | PF | MaxDD% | et | max_loss | min_confl |",
        "|---|--------|--------|---------|--------|------|-----|--------|-----|----------|----------|",
    ]

    for i, b in enumerate(sorted_bests):
        lines.append(
            f"| {i + 1} | {b['symbol']} | {b['sharpe']:+.3f} | {b['return_pct']:+.1f} | "
            f"{b['trades']} | {b['win_rate_pct']:.0f} | {b['profit_factor']:.2f} | "
            f"{b['max_dd_pct']:.1f} | {b['best_entry_threshold']} | {b['best_max_loss_pct']} | {b['best_min_confluence']} |"
        )

    lines += [
        "",
        "## Best Parameter Distributions",
        "",
        "### Best entry_threshold distribution",
    ]

    et_counts = {}
    for b in bests:
        et_counts[b["best_entry_threshold"]] = et_counts.get(b["best_entry_threshold"], 0) + 1
    for et_val in sorted(et_counts):
        bar = "█" * et_counts[et_val]
        pct = 100 * et_counts[et_val] // len(bests) if bests else 0
        lines.append(f"- et={et_val}: {et_counts[et_val]} tickers ({pct}%) {bar}")

    lines += ["", "### Best max_loss_pct distribution"]
    ml_counts = {}
    for b in bests:
        ml_counts[b["best_max_loss_pct"]] = ml_counts.get(b["best_max_loss_pct"], 0) + 1
    for ml_val in sorted(ml_counts):
        pct = 100 * ml_counts[ml_val] // len(bests) if bests else 0
        bar = "█" * ml_counts[ml_val]
        lines.append(f"- max_loss={ml_val}: {ml_counts[ml_val]} tickers ({pct}%) {bar}")

    lines += ["", "### Best min_confluence distribution"]
    mc_counts = {}
    for b in bests:
        mc_counts[b["best_min_confluence"]] = mc_counts.get(b["best_min_confluence"], 0) + 1
    for mc_val in sorted(mc_counts):
        pct = 100 * mc_counts[mc_val] // len(bests) if bests else 0
        bar = "█" * mc_counts[mc_val]
        lines.append(f"- min_confl={mc_val}: {mc_counts[mc_val]} tickers ({pct}%) {bar}")

    lines += [
        "",
        "## Tier Summary",
    ]

    tiers = {
        "S": ["XLK", "XLE", "GLD", "SPY", "SLV", "QQQ"],
        "A": ["NUE", "STLD", "HAL", "MPC", "EOG"],
        "B": ["INTC", "AMD", "LMT", "JNJ", "MRK", "NEM"],
    }
    lines.append("| Tier | N | Sharpe | Positive |")
    lines.append("|------|---|--------|----------|")
    for tier, syms in tiers.items():
        tier_bests = [b for b in bests if b["symbol"] in syms]
        if tier_bests:
            tier_sh = float(np.mean([b["sharpe"] or 0 for b in tier_bests]))
            tier_pos = sum(1 for b in tier_bests if (b["sharpe"] or 0) > 0)
            lines.append(
                f"| {tier} | {len(tier_bests)} | {tier_sh:+.3f} | {tier_pos}/{len(tier_bests)} |"
            )

    md_path = Path(args.md_output)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
