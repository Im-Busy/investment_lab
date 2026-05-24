"""
High-Speed Parameter Optimizer for Rules-First Strategy.
... etc
"""

from __future__ import annotations

# MUST disable tqdm BEFORE any imports that use it
import os as _os

_os.environ["TQDM_DISABLE"] = "1"

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.rules_first_strategy import RulesFirstStrategy

# Suppress all logging for speed
for _log in ["backtesting", "backtesting.backtesting", "root", "__main__", "tqdm"]:
    logging.getLogger(_log).setLevel(logging.ERROR)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG - dates set by main() and passed through function chain
# ──────────────────────────────────────────────────────────────────────────────

_DATES = {
    "is_start": "2016-01-01",
    "is_end": "2024-12-31",
    "oos_start": "2025-01-01",
    "oos_end": "2026-05-16",
}

CASH = 10_000
COMMISSION = 0.001

# Full parameter grid
PARAM_GRID = [
    {
        "entry_threshold": et,
        "min_reliability": mr,
        "trail_stop_atr": tsa,
        "confluence_bonus": cb,
    }
    for et in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    for mr in [0.40, 0.55, 0.70]
    for tsa in [2.0, 3.0, 4.0]
    for cb in [0.05, 0.10, 0.15]
]

# Reduced "fast" grid for quick iterations (30 combos)
PARAM_GRID_FAST = [
    {"entry_threshold": et, "min_reliability": mr, "trail_stop_atr": tsa, "confluence_bonus": cb}
    for et in [0.40, 0.50, 0.55, 0.60, 0.70]
    for mr in [0.40, 0.55, 0.70]
    for tsa in [2.0, 4.0]
    for cb in [0.05, 0.10]
]

# Ultra-fast grid for initial screening (15 combos)
PARAM_GRID_MINI = [
    {"entry_threshold": et, "min_reliability": mr, "trail_stop_atr": tsa, "confluence_bonus": cb}
    for et in [0.40, 0.55, 0.70]
    for mr in [0.40, 0.70]
    for tsa in [2.0, 4.0]
    for cb in [0.10]
]

# Default instrument basket
DEFAULT_BASKET = [
    "SPY",
    "QQQ",
    "IWM",  # Major indices
    "XLK",
    "XLF",
    "XLE",
    "XLV",  # Sector ETFs
    "GLD",
    "TLT",  # Commodities / Bonds
    "KO",
    "JPM",
    "XOM",
    "JNJ",
    "SO",  # Single stocks
    "BTC_USD",
    "EURUSD_X",  # Crypto / Forex
]

BASKET_LABELS: Dict[str, str] = {
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


# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TuneResult:
    symbol: str
    params: Dict[str, Any]
    period: str
    return_pct: float = 0.0
    sharpe: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_dd: float = 0.0
    exposure_pct: float = 0.0
    annual_return: float = 0.0
    success: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "params": self.params,
            "period": self.period,
            "return_pct": round(self.return_pct, 2),
            "sharpe": round(self.sharpe, 3),
            "trades": self.trades,
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(self.profit_factor, 2),
            "max_dd": round(self.max_dd, 2),
            "exposure_pct": round(self.exposure_pct, 1),
            "annual_return": round(self.annual_return, 2),
            "success": self.success,
            "error": self.error,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading (pre-load all before any backtest)
# ──────────────────────────────────────────────────────────────────────────────


def load_data(symbol: str) -> Optional[pd.DataFrame]:
    """Load OHLCV data from data/raw/."""
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


def split_is_oos(
    df: pd.DataFrame, is_start: str, is_end: str, oos_start: str, oos_end: str
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Split into IS and OOS periods."""
    is_df = df[(df.index >= is_start) & (df.index <= is_end)]
    oos_df = df[(df.index >= oos_start) & (df.index <= oos_end)]
    if len(is_df) < 100:
        is_df = None  # type: ignore[assignment]
    if len(oos_df) < 50:
        oos_df = None  # type: ignore[assignment]
    return is_df, oos_df


def load_all_data(symbols: List[str]) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Pre-load all data: full, IS, OOS for each symbol."""
    d = _DATES
    result = {}
    for symbol in symbols:
        df = load_data(symbol)
        if df is None:
            continue
        is_df, oos_df = split_is_oos(df, d["is_start"], d["is_end"], d["oos_start"], d["oos_end"])
        if is_df is None or len(is_df) < 100:
            continue
        result[symbol] = (df, is_df, oos_df)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Backtest Runner
# ──────────────────────────────────────────────────────────────────────────────


def run_single_backtest(df: pd.DataFrame, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Run a single Backtest and return stats dict."""
    from backtesting import Backtest

    bt = Backtest(
        df,
        RulesFirstStrategy,
        cash=CASH,
        commission=COMMISSION,
        exclusive_orders=True,
        finalize_trades=True,
    )
    bt_run_kwargs = {
        "entry_threshold": params["entry_threshold"],
        "exit_threshold": 0.30,
        "trail_stop_atr": params["trail_stop_atr"],
        "min_reliability": params["min_reliability"],
        "confluence_bonus": params["confluence_bonus"],
        "volume_confirm": True,
        "use_ir_weights": False,
        "use_short": False,
        "use_multi_tp": True,
        "use_quality_registry": True,
        "use_vix_gate": False,
        "use_yield_curve_gate": False,
    }
    stats = bt.run(**bt_run_kwargs)  # type: ignore[arg-type]
    return {
        "Return [%]": float(stats.get("Return [%]", 0)),
        "Sharpe Ratio": float(stats.get("Sharpe Ratio", 0) or 0),
        "# Trades": int(stats.get("# Trades", 0)),
        "Win Rate [%]": float(stats.get("Win Rate [%]", 0) or 0),
        "Profit Factor": float(stats.get("Profit Factor", 0) or 0),
        "Max. Drawdown [%]": float(stats.get("Max. Drawdown [%]", 0) or 0),
        "Exposure Time [%]": float(stats.get("Exposure Time [%]", 0) or 0),
        "Return (Ann.) [%]": float(stats.get("Return (Ann.) [%]", 0) or 0),
    }


def score_result(r: TuneResult) -> float:
    """Composite score for ranking: Sharpe * sqrt(trades) * win_rate, penalize low trades."""
    if not r.success or r.trades < 5:
        return -999.0
    score = r.sharpe * np.sqrt(min(r.trades, 100)) * (r.win_rate / 100.0) * 10
    if r.profit_factor < 1.0:
        score -= 5.0
    return float(score)


# ──────────────────────────────────────────────────────────────────────────────
# Single Instrument Sweep (run in subprocess)
# ──────────────────────────────────────────────────────────────────────────────


def sweep_instrument(
    symbol: str,
    is_df: pd.DataFrame,
    oos_df: Optional[pd.DataFrame],
    param_grid: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Sweep all param combos on one instrument. Returns best IS + OOS validation."""
    is_results: List[TuneResult] = []
    oos_results: List[TuneResult] = []

    # Buy & Hold baselines
    bh_is = _buy_hold_return(is_df)
    bh_oos = _buy_hold_return(oos_df) if oos_df is not None and len(oos_df) >= 2 else 0.0

    best_is_score = -999.0
    best_is_params: Optional[Dict[str, Any]] = None
    best_is_result: Optional[TuneResult] = None

    for params in param_grid:
        try:
            stats = run_single_backtest(is_df, params)
            r = TuneResult(
                symbol=symbol,
                params=dict(params),
                period="IS",
                return_pct=stats["Return [%]"],
                sharpe=stats["Sharpe Ratio"],
                trades=stats["# Trades"],
                win_rate=stats["Win Rate [%]"],
                profit_factor=stats["Profit Factor"],
                max_dd=stats["Max. Drawdown [%]"],
                exposure_pct=stats["Exposure Time [%]"],
                annual_return=stats["Return (Ann.) [%]"],
            )
            is_results.append(r)
            score = score_result(r)
            if score > best_is_score:
                best_is_score = score
                best_is_params = dict(params)
                best_is_result = r
        except Exception as e:
            r = TuneResult(
                symbol=symbol, params=dict(params), period="IS", success=False, error=str(e)
            )
            is_results.append(r)

    # Validate best IS params on OOS
    best_oos_result: Optional[TuneResult] = None
    if best_is_params and oos_df is not None and len(oos_df) >= 50:
        try:
            oos_stats = run_single_backtest(oos_df, best_is_params)
            best_oos_result = TuneResult(
                symbol=symbol,
                params=best_is_params,
                period="OOS",
                return_pct=oos_stats["Return [%]"],
                sharpe=oos_stats["Sharpe Ratio"],
                trades=oos_stats["# Trades"],
                win_rate=oos_stats["Win Rate [%]"],
                profit_factor=oos_stats["Profit Factor"],
                max_dd=oos_stats["Max. Drawdown [%]"],
                exposure_pct=oos_stats["Exposure Time [%]"],
                annual_return=oos_stats["Return (Ann.) [%]"],
            )
            oos_results.append(best_oos_result)
        except Exception as e:
            best_oos_result = TuneResult(
                symbol=symbol, params=best_is_params, period="OOS", success=False, error=str(e)
            )
            oos_results.append(best_oos_result)

    return {
        "symbol": symbol,
        "label": BASKET_LABELS.get(symbol, ""),
        "best_params": best_is_params or {},
        "best_is": best_is_result.to_dict() if best_is_result else {},
        "best_oos": best_oos_result.to_dict() if best_oos_result else {},
        "bh_is": round(bh_is, 2),
        "bh_oos": round(bh_oos, 2),
        "n_is_combos": len(is_results),
        "is_results": [r.to_dict() for r in is_results],
        "oos_results": [r.to_dict() for r in oos_results],
    }


def _buy_hold_return(df: Optional[pd.DataFrame]) -> float:
    if df is None or len(df) < 2:
        return 0.0
    return float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100)


# ──────────────────────────────────────────────────────────────────────────────
# Parallel Sweep Across Instruments
# ──────────────────────────────────────────────────────────────────────────────


def sweep_all(
    instruments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    param_grid: List[Dict[str, Any]],
    is_only: bool = False,
    max_workers: int = 0,
) -> List[Dict[str, Any]]:
    """Run parameter sweeps across all instruments in parallel."""
    results: List[Dict[str, Any]] = []
    n_instruments = len(instruments)
    n_combos = len(param_grid)
    total_backtests = n_instruments * n_combos

    print(f"\n{'=' * 70}")
    print(f"  Swapping {n_combos} param combos x {n_instruments} instruments")
    print(f"  Total: ~{total_backtests} backtests | Workers: {max_workers or mp.cpu_count()}")
    print("  Multi-TP: ON | Quality Registry: ON | Short: OFF")
    print(f"{'=' * 70}\n")

    t0 = time.time()
    completed = 0

    if max_workers <= 0:
        max_workers = min(mp.cpu_count(), n_instruments)

    # Sequential (for debugging/small runs)
    if max_workers <= 1 or n_instruments <= 1:
        for symbol, (_, is_df, oos_df) in instruments.items():
            oos = oos_df if not is_only else None
            result = sweep_instrument(symbol, is_df, oos, param_grid)
            results.append(result)
            completed += 1
            _print_progress(result, completed, n_instruments, time.time() - t0)
    else:
        # Parallel via multiprocessing
        # On Windows, we must use a context manager and not pass DataFrames through
        # if they cause pickling issues. Instead, pass symbol key and let subprocess
        # re-load data from the pre-built cache.
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for symbol, (full_df, is_df, oos_df) in instruments.items():
                oos = oos_df if not is_only else None
                future = executor.submit(sweep_instrument, symbol, is_df, oos, param_grid)
                futures[future] = symbol

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"  {symbol}: FAILED - {e}")
                completed += 1
                _print_progress(
                    result if "result" in dir() else None,
                    completed,
                    n_instruments,
                    time.time() - t0,
                )

    # Sort results by OOS Sharpe
    results.sort(key=lambda r: r.get("best_oos", {}).get("sharpe", -999), reverse=True)

    total_elapsed = time.time() - t0
    print(
        f"\n  Sweep complete in {total_elapsed:.0f}s "
        f"({total_backtests / total_elapsed:.1f} backtests/sec)"
    )

    return results


def _print_progress(
    result: Optional[Dict[str, Any]], completed: int, total: int, elapsed: float
) -> None:
    """Print per-instrument progress line."""
    if result is None:
        print(f"  [{completed}/{total}] FAILED")
        return
    sym = result.get("symbol", "?")
    is_r = result.get("best_is", {})
    oos_r = result.get("best_oos", {})
    bp = result.get("best_params", {})
    et = bp.get("entry_threshold", "?")
    mr = bp.get("min_reliability", "?")
    tsa = bp.get("trail_stop_atr", "?")
    cb = bp.get("confluence_bonus", "?")
    params_str = (
        f"et={float(et) if et != '?' else '?':.2f} "
        f"mr={float(mr) if mr != '?' else '?':.2f} "
        f"tsa={float(tsa) if tsa != '?' else '?':.1f} "
        f"cb={float(cb) if cb != '?' else '?':.2f}"
    )
    is_sharpe = is_r.get("sharpe", 0)
    is_ret = is_r.get("return_pct", 0)
    oos_sharpe = oos_r.get("sharpe", 0)
    oos_ret = oos_r.get("return_pct", 0)
    is_tr = is_r.get("trades", 0)
    oos_tr = oos_r.get("trades", 0)
    rate = completed / elapsed if elapsed > 0 else 0
    eta = (total - completed) / rate if rate > 0 else 0
    print(
        f"  [{completed:>2d}/{total}] {sym:10s} | {params_str} | "
        f"IS: S={is_sharpe:.2f} R={is_ret:+.1f}% T={is_tr} | "
        f"OOS: S={oos_sharpe:.2f} R={oos_ret:+.1f}% T={oos_tr} | "
        f"ETA {eta:.0f}s"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Universal Best Finder
# ──────────────────────────────────────────────────────────────────────────────


def find_universal_best(
    results: List[Dict[str, Any]],
    param_grid: List[Dict[str, Any]],
    instruments: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> Dict[str, Any]:
    """Find the single param set that works best across ALL instruments.

    Evaluates every param combo's OOS performance on every instrument.
    Returns params with highest average OOS Sharpe across >=3 instruments.
    """
    print(f"\n{'=' * 70}")
    print("  Finding Universal Best Config...")
    print(f"  Testing all {len(param_grid)} param combos x {len(instruments)} instruments OOS")
    print(f"{'=' * 70}")

    def param_key(p):
        return json.dumps(p, sort_keys=True)

    # Pre-compute params -> OOS results across instruments
    oos_by_params: Dict[str, List[Dict]] = {}
    t0 = time.time()
    completed = 0
    total = len(param_grid) * len(instruments)

    # Run OOS for every param combo on every instrument (the slow but correct way)
    for params in param_grid:
        pk = param_key(params)
        oos_by_params[pk] = []

        for symbol, (full_df, is_df, oos_df) in instruments.items():
            if oos_df is None or len(oos_df) < 50:
                continue
            try:
                stats = run_single_backtest(oos_df, params)
                oos_by_params[pk].append(
                    {
                        "symbol": symbol,
                        "sharpe": stats["Sharpe Ratio"],
                        "return_pct": stats["Return [%]"],
                        "trades": stats["# Trades"],
                        "win_rate": stats["Win Rate [%]"],
                        "profit_factor": stats["Profit Factor"],
                        "max_dd": stats["Max. Drawdown [%]"],
                        "annual_return": stats["Return (Ann.) [%]"],
                    }
                )
            except Exception:
                pass
            completed += 1
            if completed % 200 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{total}] Universal sweep... ETA {eta:.0f}s")

    # Score each param set
    best_score = -999.0
    best_params = None
    best_stats = {}

    for pk, instrument_results in oos_by_params.items():
        if len(instrument_results) < 3:
            continue

        valid = [r for r in instrument_results if r["trades"] >= 5]
        if len(valid) < 3:
            continue

        avg_sharpe = np.mean([r["sharpe"] for r in valid])
        avg_win_rate = np.mean([r["win_rate"] for r in valid])
        avg_pf = np.mean([r["profit_factor"] for r in valid])
        avg_return = np.mean([r["return_pct"] for r in valid])
        n_instruments = len(valid)
        total_trades = sum(r["trades"] for r in valid)

        # Count positive Sharpe instruments
        n_positive = sum(1 for r in valid if r["sharpe"] > 0)

        score = avg_sharpe * np.sqrt(n_instruments) * (avg_win_rate / 100.0) * 10
        if avg_pf < 1.0:
            score -= 3.0
        if n_positive < n_instruments * 0.5:
            score -= 5.0

        if score > best_score:
            best_score = score
            best_params = json.loads(pk)
            best_stats = {
                "n_instruments": n_instruments,
                "n_positive_sharpe": n_positive,
                "avg_sharpe_oos": round(avg_sharpe, 3),
                "avg_return_oos": round(avg_return, 2),
                "avg_win_rate_oos": round(avg_win_rate, 1),
                "avg_pf_oos": round(avg_pf, 2),
                "total_trades": int(total_trades),
                "score": round(score, 2),
            }

    total_elapsed = time.time() - t0
    print(f"  Universal best found in {total_elapsed:.0f}s")
    return {"params": best_params or {}, "stats": best_stats}


# ──────────────────────────────────────────────────────────────────────────────
# Markdown Report Generator
# ──────────────────────────────────────────────────────────────────────────────


def generate_markdown(
    results: List[Dict[str, Any]],
    universal_best: Dict[str, Any],
    param_grid_size: int,
) -> str:
    """Generate a comprehensive markdown report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Rules-First Parameter Tuning Report",
        "",
        f"> **Generated:** {now}",
        f"> **IS Period:** {_DATES['is_start']} → {_DATES['is_end']}",
        f"> **OOS Period:** {_DATES['oos_start']} → {_DATES['oos_end']}",
        f"> **Param Combos:** {param_grid_size} per instrument",
        f"> **Instruments:** {len(results)}",
        "> **Fixed Settings:** Multi-TP=ON, Quality Registry=ON, Short=OFF, Volume Confirm=ON",
        "",
    ]

    # ── Universal Best ──
    ub_params = universal_best.get("params", {})
    ub_stats = universal_best.get("stats", {})
    lines.append("## Universal Best Config (All Instruments)")
    lines.append("")
    if ub_params:
        lines.append("```")
        lines.append(f"entry_threshold = {ub_params.get('entry_threshold', 'N/A')}")
        lines.append(f"min_reliability = {ub_params.get('min_reliability', 'N/A')}")
        lines.append(f"trail_stop_atr   = {ub_params.get('trail_stop_atr', 'N/A')}")
        lines.append(f"confluence_bonus = {ub_params.get('confluence_bonus', 'N/A')}")
        lines.append("```")
        lines.append("")
        for k, v in ub_stats.items():
            lines.append(f"- **{k}:** {v}")
    else:
        lines.append("*No universal best found (insufficient OOS data)*")
    lines.append("")

    # ── Per-Instrument Best Configs ──
    lines.append("## Per-Instrument Best Configs (IS → OOS)")
    lines.append("")
    lines.append(
        "| # | Symbol | Type | Best Params | IS Sharpe | IS Ret% | IS Tr | "
        "OOS Sharpe | OOS Ret% | OOS Tr | OOS Win% | OOS PF | OOS MaxDD% | "
        "OOS Ann% | B&H IS% | B&H OOS% |"
    )
    lines.append(
        "|---|--------|------|-------------|-----------|---------|-------|"
        "-----------|----------|--------|----------|--------|-----------|"
        "---------|---------|----------|"
    )

    for i, r in enumerate(results):
        sym = r["symbol"]
        label = r.get("label", "")
        bp = r.get("best_params", {})
        params_str = (
            f"et={bp.get('entry_threshold', ''):.2f} "
            f"mr={bp.get('min_reliability', ''):.2f} "
            f"tsa={bp.get('trail_stop_atr', ''):.1f} "
            f"cb={bp.get('confluence_bonus', ''):.2f}"
        )
        is_r = r.get("best_is", {})
        oos_r = r.get("best_oos", {})

        lines.append(
            f"| {i + 1} | {sym} | {label} | {params_str} | "
            f"{is_r.get('sharpe', 'N/A')} | {is_r.get('return_pct', 'N/A')} | {is_r.get('trades', 'N/A')} | "
            f"{oos_r.get('sharpe', 'N/A')} | {oos_r.get('return_pct', 'N/A')} | {oos_r.get('trades', 'N/A')} | "
            f"{oos_r.get('win_rate', 'N/A')} | {oos_r.get('profit_factor', 'N/A')} | {oos_r.get('max_dd', 'N/A')} | "
            f"{oos_r.get('annual_return', 'N/A')} | {r.get('bh_is', 'N/A')} | {r.get('bh_oos', 'N/A')} |"
        )

    lines.append("")

    # ── Summary Stats ──
    oos_sharpes = [
        r.get("best_oos", {}).get("sharpe", -99)
        for r in results
        if r.get("best_oos", {}).get("sharpe") is not None
    ]
    oos_returns = [
        r.get("best_oos", {}).get("return_pct", -99)
        for r in results
        if r.get("best_oos", {}).get("return_pct") is not None
    ]
    oos_trades = [
        r.get("best_oos", {}).get("trades", 0)
        for r in results
        if r.get("best_oos", {}).get("trades", 0) != "N/A"
    ]

    pos_sharpe = sum(1 for s in oos_sharpes if s > 0)
    pos_return = sum(1 for s in oos_returns if s > 0)

    lines.append("## Summary Statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Instruments tested | {len(results)} |")
    lines.append(
        f"| Positive OOS Sharpe | {pos_sharpe}/{len(results)} ({100 * pos_sharpe // max(1, len(results))}%) |"
    )
    lines.append(
        f"| Positive OOS Return | {pos_return}/{len(results)} ({100 * pos_return // max(1, len(results))}%) |"
    )
    if oos_sharpes:
        lines.append(f"| Mean OOS Sharpe | {np.mean(oos_sharpes):.3f} |")
        lines.append(f"| Median OOS Sharpe | {np.median(oos_sharpes):.3f} |")
        lines.append(f"| Max OOS Sharpe | {np.max(oos_sharpes):.3f} |")
        lines.append(f"| Min OOS Sharpe | {np.min(oos_sharpes):.3f} |")
    if oos_returns:
        lines.append(f"| Mean OOS Return | {np.mean(oos_returns):.1f}% |")
        lines.append(f"| Median OOS Return | {np.median(oos_returns):.1f}% |")
    if oos_trades:
        lines.append(f"| Total OOS Trades | {sum(oos_trades)} |")
        lines.append(f"| Mean OOS Trades | {np.mean(oos_trades):.1f} |")

    lines.append("")

    # ── Top/Bottom 5 ──
    sorted_by_oos = sorted(
        results, key=lambda r: r.get("best_oos", {}).get("sharpe", -999), reverse=True
    )
    lines.append("## Top 5 by OOS Sharpe")
    lines.append("")
    lines.append("| Rank | Symbol | Type | OOS Sharpe | OOS Ret% | OOS Trades | Best Params |")
    lines.append("|------|--------|------|------------|----------|------------|-------------|")
    for i, r in enumerate(sorted_by_oos[:5]):
        sym = r["symbol"]
        oos_r = r.get("best_oos", {})
        bp = r.get("best_params", {})
        params_str = f"et={bp.get('entry_threshold', ''):.2f} mr={bp.get('min_reliability', ''):.2f} tsa={bp.get('trail_stop_atr', ''):.1f}"
        lines.append(
            f"| {i + 1} | {sym} | {r.get('label', '')} | "
            f"{oos_r.get('sharpe', 'N/A')} | {oos_r.get('return_pct', 'N/A')} | {oos_r.get('trades', 'N/A')} | "
            f"{params_str} |"
        )

    lines.append("")
    lines.append("## Bottom 5 by OOS Sharpe")
    lines.append("")
    lines.append("| Rank | Symbol | Type | OOS Sharpe | OOS Ret% | OOS Trades | Best Params |")
    lines.append("|------|--------|------|------------|----------|------------|-------------|")
    for i, r in enumerate(sorted_by_oos[-5:]):
        sym = r["symbol"]
        oos_r = r.get("best_oos", {})
        bp = r.get("best_params", {})
        params_str = f"et={bp.get('entry_threshold', ''):.2f} mr={bp.get('min_reliability', ''):.2f} tsa={bp.get('trail_stop_atr', ''):.1f}"
        lines.append(
            f"| {i + 1} | {sym} | {r.get('label', '')} | "
            f"{oos_r.get('sharpe', 'N/A')} | {oos_r.get('return_pct', 'N/A')} | {oos_r.get('trades', 'N/A')} | "
            f"{params_str} |"
        )

    lines.append("")

    # ── Universal Config Validation: per-instrument OOS with universal params ──
    lines.append("## Universal Config Cross-Validation")
    lines.append("")
    lines.append("Using universal params on ALL instruments OOS:")
    lines.append("```")
    lines.append(
        f"uv run scripts/backtest_rules_first.py SYMBOL --start {_DATES['oos_start']} --end {_DATES['oos_end']} \\"
    )
    lines.append(f"    --entry-threshold {ub_params.get('entry_threshold', 'N/A')} \\")
    lines.append(f"    --min-reliability {ub_params.get('min_reliability', 'N/A')} \\")
    lines.append(f"    --trail-stop-atr {ub_params.get('trail_stop_atr', 'N/A')} \\")
    lines.append(f"    --confluence-bonus {ub_params.get('confluence_bonus', 'N/A')}")
    lines.append("```")
    lines.append("")
    lines.append("*(Run after tuning to see universal config performance per symbol)*")

    lines.append("")
    lines.append("---")
    lines.append("*Report generated by `scripts/tune_rules_params.py`*")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="High-speed Rules-First parameter optimizer")
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated ticker list (default: 16-instrument basket)",
    )
    parser.add_argument("--is-start", default="2016-01-01", help="IS start date")
    parser.add_argument("--is-end", default="2024-12-31", help="IS end date")
    parser.add_argument("--oos-start", default="2025-01-01", help="OOS start date")
    parser.add_argument("--oos-end", default="2026-05-16", help="OOS end date")
    parser.add_argument(
        "--fast", action="store_true", help="Use reduced param grid (30 combos vs 243)"
    )
    parser.add_argument(
        "--mini", action="store_true", help="Use ultra-minimal param grid (12 combos)"
    )
    parser.add_argument(
        "--is-only", action="store_true", help="IS optimization only (skip OOS + universal best)"
    )
    parser.add_argument(
        "--skip-universal", action="store_true", help="Skip universal best search (saves time)"
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Number of parallel workers (0=auto = CPU count)"
    )
    parser.add_argument("--json-output", default=None, help="JSON output path")
    parser.add_argument(
        "--md-output",
        default="reports/parameter_tuning/RULES_TUNING.md",
        help="Markdown report output path",
    )
    args = parser.parse_args()

    # Set date ranges from args
    _DATES["is_start"] = args.is_start
    _DATES["is_end"] = args.is_end
    _DATES["oos_start"] = args.oos_start
    _DATES["oos_end"] = args.oos_end

    if args.mini:
        param_grid = PARAM_GRID_MINI
    elif args.fast:
        param_grid = PARAM_GRID_FAST
    else:
        param_grid = PARAM_GRID

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else DEFAULT_BASKET

    print("=" * 70)
    print("  RULES-FIRST PARAMETER OPTIMIZER")
    print(
        f"  IS: {_DATES['is_start']} -> {_DATES['is_end']}  |  OOS: {_DATES['oos_start']} -> {_DATES['oos_end']}"
    )
    print(f"  Param grid: {len(param_grid)} combos  |  Instruments: {len(symbols)}")
    print("  Multi-TP: ON | Quality Registry: ON | Short: OFF")
    print("=" * 70)

    # [Step 1] Pre-load all data
    print("\n[1/4] Loading data...")
    t0 = time.time()
    instruments = load_all_data(symbols)
    print(f"  Loaded {len(instruments)}/{len(symbols)} instruments in {time.time() - t0:.1f}s")

    not_found = [s for s in symbols if s not in instruments]
    if not_found:
        print(f"  Missing: {', '.join(not_found)}")

    if len(instruments) < 1:
        print("ERROR: No instruments loaded. Aborting.")
        return

    # [Step 2] Run per-instrument sweeps
    print("\n[2/4] Running per-instrument parameter sweeps...")
    results = sweep_all(instruments, param_grid, is_only=args.is_only, max_workers=args.workers)

    # [Step 3] Find universal best
    universal_best: Dict[str, Any] = {"params": {}, "stats": {}}
    if not args.is_only and not args.skip_universal and len(results) >= 3:
        print("\n[3/4] Finding universal best config...")
        universal_best = find_universal_best(results, param_grid, instruments)

    # [Step 4] Generate report
    print("\n[4/4] Generating report...")
    md_content = generate_markdown(results, universal_best, len(param_grid))
    md_path = Path(args.md_output)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_content, encoding="utf-8")
    print(f"  Report saved to {md_path}")

    # Save JSON
    json_path = (
        args.json_output
        or f"outputs/tune_rules_params_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )
    json_output = {
        "config": {
            "is_start": _DATES["is_start"],
            "is_end": _DATES["is_end"],
            "oos_start": _DATES["oos_start"],
            "oos_end": _DATES["oos_end"],
            "param_grid_size": len(param_grid),
            "n_instruments": len(results),
        },
        "universal_best": universal_best,
        "results": results,
    }
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(json_output, indent=2, default=str))
    print(f"  JSON saved to {json_path}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("  TUNING COMPLETE")
    ub = universal_best.get("params", {})
    if ub:
        print(
            f"  Universal best: et={ub['entry_threshold']} mr={ub['min_reliability']} "
            f"tsa={ub['trail_stop_atr']} cb={ub['confluence_bonus']}"
        )
        for k, v in universal_best.get("stats", {}).items():
            print(f"    {k}: {v}")
    print(f"  Report: {md_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    # Windows multiprocessing: must use spawn-compatible approach
    # freeze_support() helps with pyinstaller but not strictly needed here
    main()
