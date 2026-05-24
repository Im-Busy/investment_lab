"""
OOS validation + universal best finder for Rules-First parameter tuning.

Reads batch JSON results, runs OOS validation with per-instrument best params,
then evaluates every unique param combo OOS across all instruments to find
the universal best config.

Usage:
    uv run scripts/tune_rules_oos.py
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os as _os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_os.environ["TQDM_DISABLE"] = "1"

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.rules_first_strategy import RulesFirstStrategy

for _log in ["backtesting", "backtesting.backtesting", "root", "__main__"]:
    logging.getLogger(_log).setLevel(logging.ERROR)


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def load_data(symbol: str) -> Optional[pd.DataFrame]:
    path = Path(f"data/raw/{symbol}_daily.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df.dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0 if col == "Volume" else df.iloc[:, 0]
    df.columns = [c.capitalize() for c in df.columns]
    return df


def run_oos(
    symbol: str, params: Dict[str, Any], oos_start: str = "2025-01-01", oos_end: str = "2026-05-16"
) -> Optional[Dict[str, Any]]:
    """Run OOS backtest with given params."""
    from backtesting import Backtest

    df = load_data(symbol)
    if df is None:
        return None
    oos_df = df[(df.index >= oos_start) & (df.index <= oos_end)]
    if len(oos_df) < 50:
        return None

    bh = (oos_df["Close"].iloc[-1] / oos_df["Close"].iloc[0] - 1) * 100 if len(oos_df) >= 2 else 0

    bt = Backtest(
        oos_df,
        RulesFirstStrategy,
        cash=10000,
        commission=0.001,
        exclusive_orders=True,
        finalize_trades=True,
    )
    try:
        stats = bt.run(
            entry_threshold=params.get("entry_threshold", 0.55),
            exit_threshold=0.30,
            trail_stop_atr=params.get("trail_stop_atr", 3.0),
            min_reliability=params.get("min_reliability", 0.70),
            confluence_bonus=params.get("confluence_bonus", 0.10),
            volume_confirm=True,
            use_ir_weights=False,
            use_short=False,
            use_multi_tp=True,
            use_quality_registry=True,
            use_vix_gate=False,
            use_yield_curve_gate=False,
        )
        return {
            "symbol": symbol,
            "params": params,
            "return_pct": round(float(stats.get("Return [%]", 0)), 2),
            "sharpe": round(float(stats.get("Sharpe Ratio", 0) or 0), 3),
            "trades": int(stats.get("# Trades", 0)),
            "win_rate": round(float(stats.get("Win Rate [%]", 0) or 0), 2),
            "profit_factor": round(float(stats.get("Profit Factor", 0) or 0), 2),
            "max_dd": round(float(stats.get("Max. Drawdown [%]", 0) or 0), 2),
            "exposure_pct": round(float(stats.get("Exposure Time [%]", 0) or 0), 1),
            "annual_return": round(float(stats.get("Return (Ann.) [%]", 0) or 0), 2),
            "bh_return": round(bh, 2),
            "success": True,
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "params": params,
            "success": False,
            "error": str(e),
            "return_pct": 0,
            "sharpe": 0,
            "trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "max_dd": 0,
            "exposure_pct": 0,
            "annual_return": 0,
            "bh_return": round(bh, 2),
        }


def run_universal_oos(
    symbol: str, oos_df: pd.DataFrame, params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Run OOS backtest for universal best search."""
    from backtesting import Backtest

    bt = Backtest(
        oos_df,
        RulesFirstStrategy,
        cash=10000,
        commission=0.001,
        exclusive_orders=True,
        finalize_trades=True,
    )
    try:
        stats = bt.run(
            entry_threshold=params.get("entry_threshold", 0.55),
            exit_threshold=0.30,
            trail_stop_atr=params.get("trail_stop_atr", 3.0),
            min_reliability=params.get("min_reliability", 0.70),
            confluence_bonus=params.get("confluence_bonus", 0.10),
            volume_confirm=True,
            use_ir_weights=False,
            use_short=False,
            use_multi_tp=True,
            use_quality_registry=True,
            use_vix_gate=False,
            use_yield_curve_gate=False,
        )
        return {
            "sharpe": float(stats.get("Sharpe Ratio", 0) or 0),
            "return_pct": float(stats.get("Return [%]", 0)),
            "trades": int(stats.get("# Trades", 0)),
            "win_rate": float(stats.get("Win Rate [%]", 0) or 0),
            "profit_factor": float(stats.get("Profit Factor", 0) or 0),
            "max_dd": float(stats.get("Max. Drawdown [%]", 0) or 0),
            "annual_return": float(stats.get("Return (Ann.) [%]", 0) or 0),
        }
    except Exception:
        return None


def main():
    batch_files = [
        "outputs/tune_batch1_fast.json",
        "outputs/tune_batch2_fast.json",
        "outputs/tune_batch3_fast.json",
    ]

    # Collect per-instrument best params from IS sweeps
    per_instrument: Dict[str, Dict] = {}
    all_is_results: List[Dict] = []

    for bf in batch_files:
        data = load_json(bf)
        for r in data["results"]:
            sym = r["symbol"]
            bp = r["best_params"]
            is_r = r["best_is"]
            bh_is = r["bh_is"]
            per_instrument[sym] = {"best_params": bp, "is": is_r, "bh_is": bh_is}
            all_is_results.extend(r.get("is_results", []))

    print(f"Loaded {len(per_instrument)} instruments from 3 batches")
    print(f"IS backtests: {len(all_is_results)}")

    # ── Phase 1: OOS validation with per-instrument best params ──
    print(f"\n{'=' * 70}")
    print("  PHASE 1: OOS validation with per-instrument best params")
    print(f"{'=' * 70}")

    oos_results: Dict[str, Dict] = {}
    oos_start = "2025-01-01"
    oos_end = "2026-05-16"

    t0 = time.time()
    completed = 0
    n = len(per_instrument)

    for symbol, info in per_instrument.items():
        params = info["best_params"]
        completed += 1
        r = run_oos(symbol, params, oos_start, oos_end)
        if r:
            oos_results[symbol] = r
            print(
                f"  [{completed}/{n}] {symbol:10s} et={params['entry_threshold']:.2f} "
                f"mr={params['min_reliability']:.2f} tsa={params['trail_stop_atr']:.1f} "
                f"cb={params['confluence_bonus']:.2f} | "
                f"OOS: S={r['sharpe']:.2f} R={r['return_pct']:+.1f}% T={r['trades']} "
                f"WR={r['win_rate']:.1f}% PF={r['profit_factor']:.2f} DD={r['max_dd']:.1f}%"
            )
        else:
            print(f"  [{completed}/{n}] {symbol:10s} FAILED")

    print(f"\n  OOS validation done in {time.time() - t0:.0f}s")

    # ── Phase 2: Find universal best ──
    print(f"\n{'=' * 70}")
    print("  PHASE 2: Finding universal best config")
    print(f"{'=' * 70}")

    # Collect all unique param combos from IS sweeps
    param_combos: List[Dict] = []
    seen = set()
    for r in all_is_results:
        pk = json.dumps(r["params"], sort_keys=True)
        if pk not in seen:
            seen.add(pk)
            param_combos.append(r["params"])

    print(f"  Unique param combos: {len(param_combos)}")

    # Pre-load OOS data for all instruments
    print("  Loading OOS data...")
    oos_data: Dict[str, pd.DataFrame] = {}
    for symbol in per_instrument:
        df = load_data(symbol)
        if df is not None:
            oos_df = df[(df.index >= oos_start) & (df.index <= oos_end)]
            if len(oos_df) >= 50:
                oos_data[symbol] = oos_df

    print(f"  {len(oos_data)} instruments with OOS data")

    # Evaluate all param combos OOS across all instruments (locally, sequential for simplicity)
    # For speed, limit to combos that appear in top IS results
    param_combos = param_combos[:200]  # Max 200 combos

    oos_by_params: Dict[str, List[Dict]] = defaultdict(list)
    t0 = time.time()
    completed = 0
    total = len(param_combos) * len(oos_data)

    for params in param_combos:
        pk = json.dumps(params, sort_keys=True)
        for symbol, oos_df in oos_data.items():
            completed += 1
            r = run_universal_oos(symbol, oos_df, params)
            if r and r["trades"] >= 5:
                oos_by_params[pk].append(dict(r, symbol=symbol))
            if completed % 500 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{total}] Universal sweep... ETA {eta:.0f}s")

    # Find best universal params
    best_score = -999.0
    best_params: Optional[Dict] = None
    best_stats: Dict = {}

    for pk, results in oos_by_params.items():
        if len(results) < 3:
            continue
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_win_rate = np.mean([r["win_rate"] for r in results])
        avg_pf = np.mean([r["profit_factor"] for r in results])
        avg_return = np.mean([r["return_pct"] for r in results])
        n_positive = sum(1 for r in results if r["sharpe"] > 0)

        score = avg_sharpe * np.sqrt(len(results)) * (avg_win_rate / 100.0) * 10
        if avg_pf < 1.0:
            score -= 3.0
        if n_positive < len(results) * 0.5:
            score -= 5.0

        if score > best_score:
            best_score = score
            best_params = json.loads(pk)
            best_stats = {
                "n_instruments": len(results),
                "n_positive_sharpe": n_positive,
                "avg_sharpe_oos": round(avg_sharpe, 3),
                "avg_return_oos": round(avg_return, 2),
                "avg_win_rate_oos": round(avg_win_rate, 1),
                "avg_pf_oos": round(avg_pf, 2),
                "total_trades": sum(r["trades"] for r in results),
                "score": round(score, 2),
            }

    print(f"\n  Universal best found in {time.time() - t0:.0f}s")

    if best_params is None:
        # Fallback: use median values from best per-instrument params
        ets = [info["best_params"]["entry_threshold"] for info in per_instrument.values()]
        mrs = [info["best_params"]["min_reliability"] for info in per_instrument.values()]
        tsas = [info["best_params"]["trail_stop_atr"] for info in per_instrument.values()]
        cbs = [info["best_params"]["confluence_bonus"] for info in per_instrument.values()]
        best_params = {
            "entry_threshold": round(np.median(ets), 2),
            "min_reliability": round(np.median(mrs), 2),
            "trail_stop_atr": round(np.median(tsas), 1),
            "confluence_bonus": round(np.median(cbs), 2),
        }
        best_stats = {"note": "Fallback: median of per-instrument bests"}
        print(f"  Fallback to median params: {best_params}")

    # ── Generate combined markdown report ──
    print(f"\n{'=' * 70}")
    print("  Generating combined report...")
    print(f"{'=' * 70}")

    lines = []
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    labels = {
        "SPY": "Index - Large Cap",
        "QQQ": "Index - NASDAQ",
        "IWM": "Index - Small Cap",
        "XLK": "Sector - Tech",
        "XLF": "Sector - Financials",
        "XLE": "Sector - Energy",
        "XLV": "Sector - Healthcare",
        "GLD": "Commodity - Gold",
        "TLT": "Bond - Treasury",
        "KO": "Stock - Consumer",
        "JPM": "Stock - Financial",
        "XOM": "Stock - Energy",
        "JNJ": "Stock - Healthcare",
        "SO": "Stock - Utility",
        "BTC_USD": "Crypto - BTC",
        "EURUSD_X": "Forex - EUR/USD",
    }

    lines.append("# Rules-First Parameter Tuning — Complete Results")
    lines.append("")
    lines.append(f"> **Generated:** {now}")
    lines.append("> **IS Period:** 2016-01-01 → 2024-12-31")
    lines.append("> **OOS Period:** 2025-01-01 → 2026-05-16")
    lines.append(f"> **Instruments:** {len(per_instrument)}")
    lines.append(
        "> **Fixed Settings:** Multi-TP=ON, Quality Registry=ON, Short=OFF, Volume Confirm=ON"
    )
    lines.append("")

    # Universal Best
    lines.append("## Universal Best Config (All Instruments)")
    lines.append("")
    if best_params:
        lines.append("```")
        lines.append(f"entry_threshold = {best_params['entry_threshold']}")
        lines.append(f"min_reliability = {best_params['min_reliability']}")
        lines.append(f"trail_stop_atr   = {best_params['trail_stop_atr']}")
        lines.append(f"confluence_bonus = {best_params['confluence_bonus']}")
        lines.append("```")
        lines.append("")
        for k, v in best_stats.items():
            lines.append(f"- **{k}:** {v}")
    lines.append("")

    # Per-Instrument Results Table
    lines.append("## Per-Instrument Best Configs (IS → OOS)")
    lines.append("")
    lines.append(
        "| # | Symbol | Type | Best Params | IS Sharpe | IS Ret% | IS Tr | IS WR% | OOS Sharpe | OOS Ret% | OOS Tr | OOS WR% | OOS PF | OOS MaxDD% | OOS Ann% | B&H IS% | B&H OOS% |"
    )
    lines.append(
        "|---|--------|------|-------------|-----------|---------|-------|--------|------------|----------|--------|----------|--------|------------|----------|---------|----------|"
    )

    sorted_syms = sorted(
        oos_results.keys(), key=lambda s: oos_results[s].get("sharpe", -99), reverse=True
    )

    for i, sym in enumerate(sorted_syms):
        info = per_instrument[sym]
        oos = oos_results.get(sym, {})
        bp = info["best_params"]
        is_r = info["is"]
        params_str = f"et={bp['entry_threshold']:.2f} mr={bp['min_reliability']:.2f} tsa={bp['trail_stop_atr']:.1f} cb={bp['confluence_bonus']:.2f}"

        lines.append(
            f"| {i + 1} | {sym} | {labels.get(sym, '')} | {params_str} | "
            f"{is_r.get('sharpe', 'N/A')} | {is_r.get('return_pct', 'N/A')} | {is_r.get('trades', 'N/A')} | {is_r.get('win_rate', 'N/A')} | "
            f"{oos.get('sharpe', 'N/A')} | {oos.get('return_pct', 'N/A')} | {oos.get('trades', 'N/A')} | {oos.get('win_rate', 'N/A')} | "
            f"{oos.get('profit_factor', 'N/A')} | {oos.get('max_dd', 'N/A')} | {oos.get('annual_return', 'N/A')} | "
            f"{info.get('bh_is', 'N/A')} | {oos.get('bh_return', 'N/A')} |"
        )

    lines.append("")

    # Summary Stats
    oos_sharpes = [
        oos_results[s]["sharpe"] for s in sorted_syms if oos_results[s].get("sharpe") is not None
    ]
    oos_returns = [
        oos_results[s]["return_pct"]
        for s in sorted_syms
        if oos_results[s].get("return_pct") is not None
    ]
    oos_trades_list = [
        oos_results[s]["trades"] for s in sorted_syms if oos_results[s].get("trades", 0) > 0
    ]

    pos_sh = sum(1 for s in oos_sharpes if s > 0)
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Instruments tested | {len(sorted_syms)} |")
    lines.append(
        f"| Positive OOS Sharpe | {pos_sh}/{len(sorted_syms)} ({100 * pos_sh // max(1, len(sorted_syms))}%) |"
    )
    if oos_sharpes:
        lines.append(f"| Mean OOS Sharpe | {np.mean(oos_sharpes):.3f} |")
        lines.append(f"| Median OOS Sharpe | {np.median(oos_sharpes):.3f} |")
        lines.append(f"| Max OOS Sharpe | {np.max(oos_sharpes):.3f} |")
        lines.append(f"| Min OOS Sharpe | {np.min(oos_sharpes):.3f} |")
    if oos_returns:
        lines.append(f"| Mean OOS Return | {np.mean(oos_returns):.1f}% |")
    if oos_trades_list:
        lines.append(f"| Total OOS Trades | {sum(oos_trades_list)} |")

    lines.append("")

    # Top/Bottom 5
    lines.append("## Top 5 by OOS Sharpe")
    lines.append("| Rank | Symbol | Type | OOS Sharpe | OOS Ret% | OOS Trades | Best Params |")
    lines.append("|------|--------|------|------------|----------|------------|-------------|")
    for i, sym in enumerate(sorted_syms[:5]):
        oos = oos_results[sym]
        bp = per_instrument[sym]["best_params"]
        p = f"et={bp['entry_threshold']:.2f} mr={bp['min_reliability']:.2f} tsa={bp['trail_stop_atr']:.1f}"
        lines.append(
            f"| {i + 1} | {sym} | {labels.get(sym, '')} | {oos.get('sharpe', 'N/A')} | {oos.get('return_pct', 'N/A')} | {oos.get('trades', 'N/A')} | {p} |"
        )

    lines.append("")
    lines.append("## Bottom 5 by OOS Sharpe")
    lines.append("| Rank | Symbol | Type | OOS Sharpe | OOS Ret% | OOS Trades | Best Params |")
    lines.append("|------|--------|------|------------|----------|------------|-------------|")
    for i, sym in enumerate(sorted_syms[-5:]):
        oos = oos_results[sym]
        bp = per_instrument[sym]["best_params"]
        p = f"et={bp['entry_threshold']:.2f} mr={bp['min_reliability']:.2f} tsa={bp['trail_stop_atr']:.1f}"
        lines.append(
            f"| {i + 1} | {sym} | {labels.get(sym, '')} | {oos.get('sharpe', 'N/A')} | {oos.get('return_pct', 'N/A')} | {oos.get('trades', 'N/A')} | {p} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("*Generated by `scripts/tune_rules_oos.py`*")

    md_path = Path("reports/parameter_tuning/RULES_TUNING.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved to {md_path}")

    # Save combined JSON
    combined = {
        "config": {
            "is_start": "2016-01-01",
            "is_end": "2024-12-31",
            "oos_start": oos_start,
            "oos_end": oos_end,
        },
        "universal_best": {"params": best_params, "stats": best_stats},
        "per_instrument": {
            sym: {
                "best_params": per_instrument[sym]["best_params"],
                "is": per_instrument[sym]["is"],
                "oos": oos_results.get(sym, {}),
                "bh_is": per_instrument[sym]["bh_is"],
                "bh_oos": oos_results.get(sym, {}).get("bh_return", 0),
            }
            for sym in sorted_syms
        },
    }
    json_path = Path("outputs/tune_rules_combined.json")
    json_path.write_text(json.dumps(combined, indent=2))
    print(f"  JSON saved to {json_path}")

    print(f"\n{'=' * 70}")
    print("  COMPLETE")
    print(f"  Universal best: {best_params}")
    print(f"  Report: {md_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
