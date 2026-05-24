"""
ICT Strategy Parameter Sweep & Cross-Instrument Optimization

Sweeps all 3 new ICT strategies (Silver Bullet, Turtle Soup, Cameron's Model)
across 20 instruments from 3 asset classes. Uses IS/OOS split to prevent
overfitting: IS=2024-01-01..2024-09-30, OOS=2024-10-01..now.

Output: per-instrument best configs + universal best config + BESTS.md entries.
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.strategies.silver_bullet import SilverBulletStrategy
from src.strategies.turtle_soup import TurtleSoupStrategy
from src.strategies.cameron_model import CameronModelStrategy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(message)s")

# Suppress backtesting.py progress bar output
logging.getLogger("backtesting").setLevel(logging.ERROR)
logging.getLogger("backtesting.backtesting").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

IS_START = "2024-01-01"
IS_END = "2024-09-30"
OOS_START = "2024-10-01"
OOS_END = None  # auto = latest

CASH = 100_000
COMMISSION = 0.001

# Instrument universe: 10 equities + 5 forex + 5 crypto
INSTRUMENTS = {
    # Top 10 daily traded equities
    "SPY": ("equity", "S&P 500 ETF"),
    "QQQ": ("equity", "Nasdaq 100 ETF"),
    "AAPL": ("equity", "Apple"),
    "MSFT": ("equity", "Microsoft"),
    "NVDA": ("equity", "NVIDIA"),
    "AMZN": ("equity", "Amazon"),
    "GOOGL": ("equity", "Alphabet"),
    "META": ("equity", "Meta"),
    "TSLA": ("equity", "Tesla"),
    "BRK-B": ("equity", "Berkshire Hathaway"),
    # Top 5 most traded forex pairs
    "EURUSD=X": ("forex", "EUR/USD"),
    "USDJPY=X": ("forex", "USD/JPY"),
    "GBPUSD=X": ("forex", "GBP/USD"),
    "AUDUSD=X": ("forex", "AUD/USD"),
    "USDCAD=X": ("forex", "USD/CAD"),
    # Top 5 most traded crypto pairs
    "BTC-USD": ("crypto", "Bitcoin"),
    "ETH-USD": ("crypto", "Ethereum"),
    "SOL-USD": ("crypto", "Solana"),
    "XRP-USD": ("crypto", "XRP"),
    "DOGE-USD": ("crypto", "Dogecoin"),
}


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    params: Dict[str, Any]
    period: str  # "IS" or "OOS"
    return_pct: float
    sharpe: float
    trades: int
    win_rate: float
    profit_factor: float
    max_dd: float
    exposure_pct: float
    annual_return: float
    success: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "params": self.params,
            "period": self.period,
            "return_pct": round(self.return_pct, 2),
            "sharpe": round(self.sharpe, 2),
            "trades": self.trades,
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(self.profit_factor, 2),
            "max_dd": round(self.max_dd, 2),
            "exposure_pct": round(self.exposure_pct, 2),
            "annual_return": round(self.annual_return, 2),
        }


@dataclass
class SweepConfig:
    strategy_class: type
    strategy_name: str
    param_grid: List[Dict[str, Any]]


# Parameter grids for each strategy
SWEEP_CONFIGS = {
    "SilverBullet": SweepConfig(
        strategy_class=SilverBulletStrategy,
        strategy_name="SilverBullet",
        param_grid=[
            {"kill_zone": kz, "trail_stop_atr": tsa, "sweep_lookback": sl, "fvg_min_gap": fvg}
            for kz in ["london_open", "new_york_am", "london_close"]
            for tsa in [1.5, 2.5]
            for sl in [8, 12]
            for fvg in [0.2, 0.3]
        ],
    ),
    "TurtleSoup": SweepConfig(
        strategy_class=TurtleSoupStrategy,
        strategy_name="TurtleSoup",
        param_grid=[
            {
                "session_bars": sb,
                "breakout_buffer_atr": bba,
                "reversal_confirm_bars": rcb,
                "trail_stop_atr": tsa,
            }
            for sb in [12, 24]
            for bba in [0.2, 0.3]
            for rcb in [2, 3]
            for tsa in [1.5, 2.5]
        ],
    ),
    "CameronModel": SweepConfig(
        strategy_class=CameronModelStrategy,
        strategy_name="CameronModel",
        param_grid=[
            {"swing_lookback": sw, "sweep_buffer_atr": sba, "trail_stop_atr": tsa}
            for sw in [50, 70]
            for sba in [0.2, 0.3]
            for tsa in [1.5, 2.0, 2.5]
        ],
    ),
}


def fetch_data(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch hourly data from Yahoo Finance."""
    try:
        df = yf.download(ticker, period="730d", interval="1h", progress=False)
        if df is None or df.empty:
            logger.warning(f"No data for {ticker}")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(set(df.columns)):
            return None
        return df
    except Exception as e:
        logger.warning(f"Fetch failed {ticker}: {e}")
        return None


def split_is_oos(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Split DataFrame into IS and OOS periods."""
    # Handle timezone: if df index is tz-aware, make comparison timestamps tz-aware
    is_end_ts = pd.Timestamp(IS_END)
    oos_start_ts = pd.Timestamp(OOS_START)

    df_sorted = df.sort_index()
    if df_sorted.index.tz is not None:
        is_end_ts = is_end_ts.tz_localize("UTC")
        oos_start_ts = oos_start_ts.tz_localize("UTC")

    is_df = df_sorted[df_sorted.index <= is_end_ts]
    oos_df = df_sorted[df_sorted.index >= oos_start_ts]

    if OOS_END:
        oos_end_ts = pd.Timestamp(OOS_END)
        if df_sorted.index.tz is not None:
            oos_end_ts = oos_end_ts.tz_localize("UTC")
        oos_df = oos_df[oos_df.index <= oos_end_ts]

    if len(is_df) < 100:
        is_df = None
    if len(oos_df) < 50:
        oos_df = None

    return is_df, oos_df


def run_backtest(
    strategy_class, df: pd.DataFrame, params: Dict[str, Any]
) -> Optional[BacktestResult]:
    """Run a single backtest."""
    try:
        bt = Backtest(df, strategy_class, cash=CASH, commission=COMMISSION)
        stats = bt.run(**params)

        return BacktestResult(
            symbol="",
            strategy="",
            params=params,
            period="",
            return_pct=float(stats.get("Return [%]", 0)),
            sharpe=float(stats.get("Sharpe Ratio", 0) or 0),
            trades=int(stats.get("# Trades", 0)),
            win_rate=float(stats.get("Win Rate [%]", 0) or 0),
            profit_factor=float(stats.get("Profit Factor", 0) or 0),
            max_dd=float(stats.get("Max. Drawdown [%]", 0) or 0),
            exposure_pct=float(stats.get("Exposure Time [%]", 0) or 0),
            annual_return=float(stats.get("Return (Ann.) [%]", 0) or 0),
        )
    except Exception as e:
        return BacktestResult(
            symbol="",
            strategy="",
            params=params,
            period="",
            return_pct=0,
            sharpe=0,
            trades=0,
            win_rate=0,
            profit_factor=0,
            max_dd=0,
            exposure_pct=0,
            annual_return=0,
            success=False,
            error=str(e),
        )


def buy_and_hold_return(df: pd.DataFrame) -> float:
    """Calculate buy-and-hold return percentage."""
    if df is None or len(df) < 2:
        return 0.0
    return float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100)


def score_result(r: BacktestResult) -> float:
    """Composite score: Sharpe * sqrt(trades) * win_rate, penalize low trades."""
    if not r.success or r.trades < 5:
        return -999.0
    score = r.sharpe * np.sqrt(min(r.trades, 100)) * (r.win_rate / 100.0) * 10
    if r.profit_factor < 1.0:
        score -= 5.0
    return float(score)


def sweep_strategy(
    sweep_cfg: SweepConfig,
    instrument_data: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    Sweep a strategy across all instruments.
    Returns per-instrument best and universal best configs.
    """
    strategy_name = sweep_cfg.strategy_name
    param_grid = sweep_cfg.param_grid
    strategy_class = sweep_cfg.strategy_class

    print(f"\n{'=' * 70}")
    print(
        f"  {strategy_name} — {len(param_grid)} param combos × {len(instrument_data)} instruments"
    )
    print(f"{'=' * 70}")

    all_is_results: List[BacktestResult] = []
    all_oos_results: List[BacktestResult] = []
    per_instrument_best: Dict[str, Dict] = {}

    total_combos = len(param_grid) * len(instrument_data)
    completed = 0

    t0 = time.time()

    for symbol, df in instrument_data.items():
        is_df, oos_df = split_is_oos(df)
        if is_df is None or oos_df is None:
            logger.warning(f"Skipping {symbol}: insufficient IS/OOS data")
            continue

        bh_is = buy_and_hold_return(is_df)
        bh_oos = buy_and_hold_return(oos_df)

        best_is_score = -999.0
        best_params = None
        best_oos = None
        symbol_is_results = []

        for params in param_grid:
            completed += 1
            if completed % 50 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_combos - completed) / rate if rate > 0 else 0
                print(
                    f"  [{completed}/{total_combos}] {strategy_name} — {elapsed:.0f}s elapsed, ETA {eta:.0f}s"
                )

            r_is = run_backtest(strategy_class, is_df, params)
            r_is.symbol = symbol
            r_is.strategy = strategy_name
            r_is.period = "IS"
            all_is_results.append(r_is)
            symbol_is_results.append(r_is)

            if not r_is.success:
                continue

            score = score_result(r_is)
            if score > best_is_score:
                best_is_score = score
                best_params = dict(params)

        # Evaluate best IS params on OOS
        if best_params:
            r_oos = run_backtest(strategy_class, oos_df, best_params)
            r_oos.symbol = symbol
            r_oos.strategy = strategy_name
            r_oos.period = "OOS"
            all_oos_results.append(r_oos)

            if r_oos.success:
                print(
                    f"  {symbol:12s} | best_params={best_params} | "
                    f"IS: {r_is.return_pct:+.1f}% S={r_is.sharpe:.2f} T={r_is.trades} | "
                    f"OOS: {r_oos.return_pct:+.1f}% S={r_oos.sharpe:.2f} T={r_oos.trades} | "
                    f"B&H IS={bh_is:+.1f}% OOS={bh_oos:+.1f}%"
                )
            else:
                print(
                    f"  {symbol:12s} | best_params={best_params} | OOS FAILED: {r_oos.error[:80]}"
                )

            per_instrument_best[symbol] = {
                "params": best_params,
                "is": r_is.to_dict() if r_is.success else {},
                "oos": r_oos.to_dict() if r_oos.success else {},
                "bh_is": round(bh_is, 2),
                "bh_oos": round(bh_oos, 2),
            }

    # Find universal best params (maximize average OOS Sharpe across instruments)
    # Strategy: find params that have valid OOS results on most instruments and best avg Sharpe
    universal_params = find_universal_best(all_is_results, all_oos_results, param_grid)

    total_elapsed = time.time() - t0
    print(f"\n  ✓ {strategy_name} complete in {total_elapsed:.0f}s")

    return {
        "strategy": strategy_name,
        "per_instrument": per_instrument_best,
        "universal_params": universal_params,
        "n_instruments_tested": len(per_instrument_best),
        "n_param_combos": len(param_grid),
    }


def find_universal_best(
    is_results: List[BacktestResult],
    oos_results: List[BacktestResult],
    param_grid: List[Dict],
) -> Dict[str, Any]:
    """Find the single param set that works best across all instruments."""

    # For each param combo, aggregate OOS performance across instruments
    def param_key(p):
        return json.dumps(p, sort_keys=True)

    # Map params -> list of OOS results across instruments
    oos_by_params: Dict[str, List[BacktestResult]] = {}
    for r in oos_results:
        if r.success and r.trades >= 5:
            k = param_key(r.params)
            if k not in oos_by_params:
                oos_by_params[k] = []
            oos_by_params[k].append(r)

    # Also check IS results for params not yet in OOS (shortcut: use same map)
    is_by_params: Dict[str, List[BacktestResult]] = {}
    for r in is_results:
        if r.success and r.trades >= 5:
            k = param_key(r.params)
            if k not in is_by_params:
                is_by_params[k] = []
            is_by_params[k].append(r)

    best_score = -999.0
    best_params = None
    best_stats = {}

    for pk, results in oos_by_params.items():
        if len(results) < 3:  # Need at least 3 instruments
            continue
        avg_sharpe = np.mean([r.sharpe for r in results])
        avg_win_rate = np.mean([r.win_rate for r in results])
        avg_pf = np.mean([r.profit_factor for r in results])
        total_trades = sum(r.trades for r in results)
        n_instruments = len(results)

        # Score: prefer high avg sharpe, high coverage, consistent win rate
        score = avg_sharpe * np.sqrt(n_instruments) * (avg_win_rate / 100.0) * 10
        if avg_pf < 1.0:
            score -= 3.0

        if score > best_score:
            best_score = score
            best_params = json.loads(pk)
            best_stats = {
                "n_instruments": n_instruments,
                "avg_sharpe": round(avg_sharpe, 3),
                "avg_win_rate": round(avg_win_rate, 1),
                "avg_pf": round(avg_pf, 2),
                "total_trades": int(total_trades),
                "score": round(score, 2),
            }

    # Fallback: use IS results if no OOS coverage
    if best_params is None:
        for pk, results in is_by_params.items():
            if len(results) < 3:
                continue
            avg_sharpe = np.mean([r.sharpe for r in results])
            avg_pf = np.mean([r.profit_factor for r in results])
            if avg_sharpe > best_score:
                best_score = avg_sharpe
                best_params = json.loads(pk)
                best_stats = {
                    "n_instruments": len(results),
                    "avg_sharpe_is": round(avg_sharpe, 3),
                    "avg_pf_is": round(avg_pf, 2),
                    "note": "IS-only (no valid OOS universal found)",
                }

    return {"params": best_params or {}, "stats": best_stats}


# ---------------------------------------------------------------------------
# BESTS.md Generator
# ---------------------------------------------------------------------------

BESTS_PATH = project_root / "BESTS.md"


def generate_bests_section(all_sweep_results: List[Dict]) -> str:
    """Generate markdown for BESTS.md."""
    lines = []
    lines.append("\n## ICT Strategy Sweep Results (2026-05-20)")
    lines.append("\n> **Instruments:** 10 equities + 5 forex + 5 crypto = 20 total")
    lines.append(f"> **Data:** 1h bars, IS={IS_START}..{IS_END}, OOS={OOS_START}..present")
    lines.append(
        "> **Method:** Per-instrument IS optimization → OOS validation. Universal = max avg OOS Sharpe across >=3 instruments."
    )
    lines.append("")

    for sweep_result in all_sweep_results:
        strategy = sweep_result["strategy"]
        universal = sweep_result["universal_params"]
        per_instrument = sweep_result["per_instrument"]

        lines.append(f"### {strategy}")
        lines.append("")

        # Universal best
        if universal["params"]:
            lines.append(f"**Universal Best:** `{universal['params']}`")
            for k, v in universal["stats"].items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        # Per-instrument table
        lines.append(
            "| Symbol | Class | Best Params | IS Ret% | IS Sharpe | IS Tr | OOS Ret% | OOS Sharpe | OOS Tr | OOS Win% | OOS PF | B&H OOS% |"
        )
        lines.append(
            "|--------|-------|-------------|---------|-----------|-------|----------|------------|--------|----------|--------|----------|"
        )

        for symbol, info in sorted(per_instrument.items()):
            cat, desc = INSTRUMENTS.get(symbol, ("?", symbol))
            p = info.get("params", {})
            param_str = ", ".join(f"{k}={v}" for k, v in p.items())
            is_r = info.get("is", {})
            oos_r = info.get("oos", {})

            is_ret = is_r.get("return_pct", "N/A")
            is_sh = is_r.get("sharpe", "N/A")
            is_tr = is_r.get("trades", "N/A")
            oos_ret = oos_r.get("return_pct", "N/A")
            oos_sh = oos_r.get("sharpe", "N/A")
            oos_tr = oos_r.get("trades", "N/A")
            oos_wr = oos_r.get("win_rate", "N/A")
            oos_pf = oos_r.get("profit_factor", "N/A")
            bh = info.get("bh_oos", "N/A")

            lines.append(
                f"| {symbol} | {cat} | {param_str} | {is_ret} | {is_sh} | {is_tr} | "
                f"{oos_ret} | {oos_sh} | {oos_tr} | {oos_wr} | {oos_pf} | {bh} |"
            )

        lines.append("")

    # Summary comparison
    lines.append("### Cross-Strategy OOS Summary")
    lines.append("")
    lines.append(
        "| Strategy | Univ Avg Sharpe OOS | Best Single Sharpe | Best Single Ret% | Instruments Passing |"
    )
    lines.append(
        "|----------|---------------------|--------------------|------------------|---------------------|"
    )

    for sweep_result in all_sweep_results:
        strategy = sweep_result["strategy"]
        universal = sweep_result["universal_params"]
        per_instrument = sweep_result["per_instrument"]
        avg_s = universal.get("stats", {}).get("avg_sharpe", 0)

        best_sharpe = -99.0
        best_ret = -99.0
        passing = 0
        for sym, info in per_instrument.items():
            oos = info.get("oos", {})
            sh = oos.get("sharpe", -99)
            ret = oos.get("return_pct", -99)
            if sh > best_sharpe:
                best_sharpe = sh
            if ret > best_ret:
                best_ret = ret
            tr = oos.get("trades", 0)
            if isinstance(tr, (int, float)) and tr >= 5:
                passing += 1

        lines.append(
            f"| {strategy} | {avg_s} | {best_sharpe} | {best_ret} | {passing}/{sweep_result['n_instruments_tested']} |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("  ICT Strategy Sweep — 20 instruments × 3 strategies")
    print(f"  IS: {IS_START} -> {IS_END}  |  OOS: {OOS_START} -> present")
    print("=" * 70)

    # Fetch all data
    print("\n[1/4] Fetching data...")
    instrument_data = {}
    for symbol, (cat, desc) in INSTRUMENTS.items():
        print(f"  Fetching {symbol} ({cat}: {desc})...", end=" ")
        df = fetch_data(symbol)
        if df is not None:
            instrument_data[symbol] = df
            print(f"OK ({len(df)} bars)")
        else:
            print("FAILED")
        time.sleep(0.3)  # Rate limit

    print(f"\n  Fetched {len(instrument_data)}/{len(INSTRUMENTS)} instruments")

    if len(instrument_data) < 5:
        print("ERROR: Not enough data. Aborting.")
        return

    # Sweep each strategy
    print("\n[2/4] Running parameter sweeps...")
    all_results = []

    for name, sweep_cfg in SWEEP_CONFIGS.items():
        result = sweep_strategy(sweep_cfg, instrument_data)
        all_results.append(result)

    # Generate BESTS section
    print("\n[3/4] Generating BESTS.md section...")
    bests_section = generate_bests_section(all_results)

    # Append to BESTS.md
    existing = BESTS_PATH.read_text(encoding="utf-8") if BESTS_PATH.exists() else ""
    # Update last_updated
    import re

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    existing = re.sub(r"last_updated:.*", f"last_updated: {now_str}", existing)

    # Remove previous ICT Strategy Sweep Results section if exists
    existing = re.sub(
        r"\n## ICT Strategy Sweep Results.*?(?=\n## |\Z)", "", existing, flags=re.DOTALL
    )

    new_content = existing.rstrip() + "\n" + bests_section + "\n"
    BESTS_PATH.write_text(new_content, encoding="utf-8")
    print(f"  ✓ Updated {BESTS_PATH}")

    # Save full results JSON
    output_path = (
        project_root / "outputs" / f"ict_sweep_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  ✓ Full results saved to {output_path}")

    # Summary
    print("\n[4/4] Summary")
    print("=" * 70)
    for r in all_results:
        universal = r["universal_params"]
        print(f"\n{r['strategy']}:")
        print(f"  Instruments tested: {r['n_instruments_tested']}")
        print(f"  Universal params: {universal['params']}")
        for k, v in universal.get("stats", {}).items():
            print(f"    {k}: {v}")

        # Count successful instruments
        per_instrument = r["per_instrument"]
        oos_success = sum(
            1 for info in per_instrument.values() if info.get("oos", {}).get("trades", 0) >= 5
        )
        print(f"  OOS valid (>=5 trades): {oos_success}/{len(per_instrument)}")

    print("\n✓ Sweep complete.")


if __name__ == "__main__":
    main()
