"""
Pattern Detector Ablation Study (P1.3).

Systematically measure which of the 34+ pattern detectors produce edge by
running each in isolation through a standard backtest. Produces per-pattern
metrics (Sharpe, win rate, profit factor, trade count, max DD) and flags
underperforming patterns for removal or retuning.

Usage:
    uv run scripts/ablate_patterns.py --symbol SPY --start 2019-01-01 --end 2024-12-31
    uv run scripts/ablate_patterns.py --symbol BTC-USD --start 2020-01-01 --end 2024-12-31
    uv run scripts/ablate_patterns.py --all  # Runs SPY and BTC together
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PatternAblationResult:
    """Results for a single pattern detector.

    Attributes:
        pattern_name: Name of the pattern detector.
        category: Category (Basic, Classic, Candlestick, etc.).
        n_trades: Number of trades generated.
        win_rate: Fraction of winning trades.
        profit_factor: Gross profit / gross loss.
        total_pnl: Total P&L in dollar terms.
        avg_win: Average winning trade P&L.
        avg_loss: Average losing trade P&L.
        max_drawdown_pct: Maximum drawdown during backtest.
        sharpe: Annualized Sharpe ratio.
        sortino: Sortino ratio.
        pct_return: Total percentage return.
        avg_bars_held: Average bars per trade.
        recommendation: 'keep', 'cut', 'retune'.
        flag_reason: Reason for flagging (if recommendation is not 'keep').
    """

    pattern_name: str
    category: str
    n_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    pct_return: float = 0.0
    avg_bars_held: float = 0.0
    recommendation: str = "keep"
    flag_reason: str = ""


def get_all_patterns() -> List[Any]:
    """Instantiate and return all pattern detectors."""
    patterns = []

    # Basic (6)
    from src.patterns.basic.msl import MarketStructureLow
    from src.patterns.basic.matching_lows import MatchingLows
    from src.patterns.basic.nr7id import NR7ID
    from src.patterns.basic.n_bar_decline import NBarDecline
    from src.patterns.basic.floor_pivot import FloorPivotBreakout
    from src.patterns.basic.two_bar_reversal import TwoBarReversal

    patterns.extend(
        [
            MarketStructureLow(),
            MatchingLows(),
            NR7ID(),
            NBarDecline(),
            FloorPivotBreakout(),
            TwoBarReversal(),
        ]
    )

    # Candlestick (6)
    from src.patterns.candlestick.doji import Doji
    from src.patterns.candlestick.harami import Harami
    from src.patterns.candlestick.hammer import Hammer
    from src.patterns.candlestick.engulfing import Engulfing
    from src.patterns.candlestick.dark_cloud import DarkCloudCover, PiercingLine

    patterns.extend(
        [
            Doji(),
            Harami(),
            Hammer(),
            Engulfing(),
            DarkCloudCover(),
            PiercingLine(),
        ]
    )

    # Harmonic (4)
    from src.patterns.harmonic.gartley import GartleyPattern
    from src.patterns.harmonic.abc import ABCPattern
    from src.patterns.harmonic.symmetric_triangle import SymmetricTriangle
    from src.patterns.harmonic.bollinger import BollingerBands

    patterns.extend(
        [
            GartleyPattern(),
            ABCPattern(),
            SymmetricTriangle(),
            BollingerBands(),
        ]
    )

    # Complex (5)
    from src.patterns.complex.cup_handle import CupAndHandle
    from src.patterns.complex.head_shoulders import HeadAndShoulders
    from src.patterns.complex.spike_ledge import SpikeAndLedge
    from src.patterns.complex.three_hills import ThreeHillsMountain
    from src.patterns.complex.parabolic_arc import ParabolicArc

    patterns.extend(
        [
            CupAndHandle(),
            HeadAndShoulders(),
            SpikeAndLedge(),
            ThreeHillsMountain(),
            ParabolicArc(),
        ]
    )

    # Classic (10)
    from src.patterns.classic.double_top import DoubleTop
    from src.patterns.classic.double_bottom import DoubleBottom
    from src.patterns.classic.trader_vic_2b import TraderVic2B
    from src.patterns.classic.triple_top import TripleTop
    from src.patterns.classic.triple_bottom import TripleBottom
    from src.patterns.classic.ascending_triangle import AscendingTriangle
    from src.patterns.classic.descending_triangle import DescendingTriangle
    from src.patterns.classic.rectangle import Rectangle
    from src.patterns.classic.wedge import Wedge
    from src.patterns.classic.dead_cat_bounce import DeadCatBounce

    patterns.extend(
        [
            DoubleTop(),
            DoubleBottom(),
            TraderVic2B(),
            TripleTop(),
            TripleBottom(),
            AscendingTriangle(),
            DescendingTriangle(),
            Rectangle(),
            Wedge(),
            DeadCatBounce(),
        ]
    )

    # Breakout (2)
    from src.patterns.breakout.donchian import DonchianChannelBreakout
    from src.patterns.breakout.gap import GapPattern

    patterns.extend([DonchianChannelBreakout(), GapPattern()])

    # Continuation (2)
    from src.patterns.continuation.flag import Flag
    from src.patterns.continuation.pennant import Pennant

    patterns.extend([Flag(), Pennant()])

    return patterns


PATTERN_CATEGORIES = {
    "Market Structure Low": "Basic",
    "Matching Lows": "Basic",
    "NR7ID": "Basic",
    "n-Bar Decline": "Basic",
    "Floor Pivot Breakout": "Basic",
    "Two-Bar Reversal": "Basic",
    "Doji": "Candlestick",
    "Harami": "Candlestick",
    "Hammer": "Candlestick",
    "Engulfing": "Candlestick",
    "Dark Cloud Cover": "Candlestick",
    "Piercing Line": "Candlestick",
    "Gartley Pattern": "Harmonic",
    "ABC Pattern": "Harmonic",
    "Symmetric Triangle": "Harmonic",
    "Bollinger Bands": "Harmonic",
    "Cup and Handle": "Complex",
    "Head and Shoulders": "Complex",
    "Spike and Ledge": "Complex",
    "Three Hills and Mountain": "Complex",
    "Parabolic Arc": "Complex",
    "Double Top": "Classic",
    "Double Bottom": "Classic",
    "Trader Vic's 2B": "Classic",
    "Triple Top": "Classic",
    "Triple Bottom": "Classic",
    "Ascending Triangle": "Classic",
    "Descending Triangle": "Classic",
    "Rectangle": "Classic",
    "Wedge": "Classic",
    "Dead Cat Bounce": "Classic",
    "Donchian Channel Breakout": "Breakout",
    "Gap Pattern": "Breakout",
    "Flag": "Continuation",
    "Pennant": "Continuation",
}


def fetch_data(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data for a symbol."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed")
        return None

    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        logger.warning("No data for %s (%s to %s)", symbol, start, end)
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            col_lower = col.lower()
            if col_lower in df.columns:
                df.rename(columns={col_lower: col}, inplace=True)

    df = df.rename(
        columns={
            "Adj Close": "Close",
            "adj close": "Close",
        }
    )
    return df


def compute_sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Compute annualized Sharpe ratio."""
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    ann_factor = np.sqrt(252)
    excess = returns - risk_free / 252
    return float(excess.mean() / max(excess.std(), 1e-10) * ann_factor)


def compute_sortino(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Compute Sortino ratio."""
    if len(returns) < 2:
        return 0.0
    ann_factor = np.sqrt(252)
    excess = returns - risk_free / 252
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / max(downside.std(), 1e-10) * ann_factor)


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown percentage."""
    if equity_curve.empty:
        return 0.0
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak.replace(0, 1.0)
    return float(drawdown.min())


def backtest_pattern(
    df: pd.DataFrame,
    pattern: Any,
    starting_capital: float = 10000.0,
    position_size_pct: float = 0.1,
) -> PatternAblationResult:
    """Run a single pattern through a simple backtest.

    Args:
        df: OHLCV DataFrame.
        pattern: Instantiated pattern detector.
        starting_capital: Initial capital.
        position_size_pct: Fraction of capital per trade.

    Returns:
        PatternAblationResult with metrics.
    """
    pattern_name = pattern.name
    category = PATTERN_CATEGORIES.get(pattern_name, "Unknown")

    min_bars = max(pattern.min_bars_required, 20)
    n = len(df)

    if n < min_bars:
        return PatternAblationResult(
            pattern_name=pattern_name, category=category, flag_reason="insufficient data"
        )

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values

    equity = float(starting_capital)
    equity_curve: List[float] = [equity]
    trades: List[Dict[str, Any]] = []
    position: Optional[Dict[str, Any]] = None

    for i in range(min_bars, n):
        # Check exit first
        if position is not None:
            pos = position
            exit_idx = i
            if pos["direction"] == "long":
                if low[i] <= pos["stop_loss"]:
                    pnl = (pos["stop_loss"] - pos["entry_price"]) * pos["shares"]
                    trades.append(
                        {
                            "pnl": pnl,
                            "entry_idx": pos["entry_idx"],
                            "exit_idx": exit_idx,
                            "direction": "long",
                            "exit_reason": "sl",
                        }
                    )
                    equity += pnl
                    position = None
                elif high[i] >= pos["take_profit"]:
                    pnl = (pos["take_profit"] - pos["entry_price"]) * pos["shares"]
                    trades.append(
                        {
                            "pnl": pnl,
                            "entry_idx": pos["entry_idx"],
                            "exit_idx": exit_idx,
                            "direction": "long",
                            "exit_reason": "tp",
                        }
                    )
                    equity += pnl
                    position = None
            else:
                if high[i] >= pos["stop_loss"]:
                    pnl = (pos["entry_price"] - pos["stop_loss"]) * pos["shares"]
                    trades.append(
                        {
                            "pnl": pnl,
                            "entry_idx": pos["entry_idx"],
                            "exit_idx": exit_idx,
                            "direction": "short",
                            "exit_reason": "sl",
                        }
                    )
                    equity += pnl
                    position = None
                elif low[i] <= pos["take_profit"]:
                    pnl = (pos["entry_price"] - pos["take_profit"]) * pos["shares"]
                    trades.append(
                        {
                            "pnl": pnl,
                            "entry_idx": pos["entry_idx"],
                            "exit_idx": exit_idx,
                            "direction": "short",
                            "exit_reason": "tp",
                        }
                    )
                    equity += pnl
                    position = None

        equity_curve.append(equity)

        # Skip if already in position
        if position is not None:
            continue

        # Generate signals
        try:
            result = pattern.detect(df, i)
        except Exception:
            continue

        if not result.detected or result.signal is None:
            continue

        signal = result.signal
        entry_price = signal.entry_price
        stop_loss = signal.stop_loss
        take_profit = signal.take_profit_1

        if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
            continue
        if signal.direction.value == "Long" and (
            stop_loss >= entry_price or take_profit <= entry_price
        ):
            continue
        if signal.direction.value == "Short" and (
            stop_loss <= entry_price or take_profit >= entry_price
        ):
            continue

        position_size = equity * position_size_pct
        shares = position_size / entry_price if entry_price > 0 else 0
        if shares <= 0:
            continue

        position = {
            "direction": signal.direction.value.lower(),
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "shares": shares,
            "entry_idx": i,
        }

    # Close any open position at end
    if position is not None:
        last_idx = n - 1
        exit_price = close[last_idx]
        pos = position
        if pos["direction"] == "long":
            pnl = (exit_price - pos["entry_price"]) * pos["shares"]
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["shares"]
        trades.append(
            {
                "pnl": pnl,
                "entry_idx": pos["entry_idx"],
                "exit_idx": last_idx,
                "direction": pos["direction"],
                "exit_reason": "eod",
            }
        )
        equity += pnl
        equity_curve.append(equity)

    n_trades = len(trades)
    if n_trades == 0:
        return PatternAblationResult(
            pattern_name=pattern_name,
            category=category,
            n_trades=0,
            recommendation="cut",
            flag_reason="no trades",
        )

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]

    n_wins = len(wins)
    n_losses = len(losses)

    win_rate = n_wins / n_trades if n_trades > 0 else 0.0
    total_wins = sum(t["pnl"] for t in wins) if wins else 0.0
    total_losses = abs(sum(t["pnl"] for t in losses)) if losses else 0.0
    profit_factor = (
        total_wins / total_losses if total_losses > 0 else (float("inf") if total_wins > 0 else 0.0)
    )

    avg_win = total_wins / n_wins if n_wins > 0 else 0.0
    avg_loss = total_losses / n_losses if n_losses > 0 else 0.0

    total_pnl = sum(t["pnl"] for t in trades)
    pct_return = (equity / starting_capital - 1) * 100

    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()

    sharpe = compute_sharpe(returns)
    sortino = compute_sortino(returns)
    max_dd = compute_max_drawdown(equity_series)

    avg_bars = np.mean([t["exit_idx"] - t["entry_idx"] for t in trades]) if trades else 0.0

    flag_reason = ""
    recommendation = "keep"
    if n_trades < 30:
        recommendation = "cut"
        flag_reason = f"too few trades ({n_trades})"
    elif sharpe <= 0:
        recommendation = "cut"
        flag_reason = f"negative/zero Sharpe ({sharpe:.3f})"
    elif profit_factor < 1.0:
        recommendation = "cut"
        flag_reason = f"profit factor < 1.0 ({profit_factor:.3f})"
    elif win_rate < 0.35:
        recommendation = "retune"
        flag_reason = f"low win rate ({win_rate:.2%})"
    elif max_dd < -0.25:
        recommendation = "retune"
        flag_reason = f"large max drawdown ({max_dd:.1%})"

    return PatternAblationResult(
        pattern_name=pattern_name,
        category=category,
        n_trades=n_trades,
        win_rate=win_rate,
        profit_factor=profit_factor if profit_factor != float("inf") else 999.0,
        total_pnl=total_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_drawdown_pct=max_dd,
        sharpe=sharpe,
        sortino=sortino,
        pct_return=pct_return,
        avg_bars_held=avg_bars,
        recommendation=recommendation,
        flag_reason=flag_reason,
    )


def run_ablation(
    symbol: str,
    start: str,
    end: str,
    output_dir: str = "reports",
) -> List[PatternAblationResult]:
    """Run ablation study for a single symbol.

    Args:
        symbol: Ticker symbol (e.g., 'SPY', 'BTC-USD').
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        output_dir: Directory for output files.

    Returns:
        List of PatternAblationResult sorted by Sharpe (descending).
    """
    logger.info("Fetching %s data (%s to %s)", symbol, start, end)
    df = fetch_data(symbol, start, end)
    if df is None or df.empty:
        logger.error("No data for %s", symbol)
        return []

    logger.info("Loaded %d bars for %s", len(df), symbol)
    patterns = get_all_patterns()
    logger.info("Running ablation on %d pattern detectors...", len(patterns))
    results = []

    for i, pattern in enumerate(patterns):
        name = pattern.name
        logger.info("[%d/%d] %s...", i + 1, len(patterns), name)
        result = backtest_pattern(df, pattern)
        results.append(result)

    results.sort(key=lambda r: r.sharpe, reverse=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_symbol = symbol.replace("-", "_").replace("=", "_")
    csv_path = output_path / f"pattern_ablation_{safe_symbol}.csv"
    _save_results_csv(results, csv_path)
    logger.info("Saved CSV: %s", csv_path)

    _print_summary(results, symbol)

    return results


def _save_results_csv(results: List[PatternAblationResult], path: Path) -> None:
    """Save results to CSV."""
    rows = []
    for r in results:
        rows.append(
            {
                "pattern_name": r.pattern_name,
                "category": r.category,
                "n_trades": r.n_trades,
                "win_rate": round(r.win_rate, 4),
                "profit_factor": round(r.profit_factor, 2),
                "total_pnl": round(r.total_pnl, 2),
                "avg_win": round(r.avg_win, 2),
                "avg_loss": round(r.avg_loss, 2),
                "sharpe": round(r.sharpe, 3),
                "sortino": round(r.sortino, 3),
                "pct_return": round(r.pct_return, 2),
                "max_drawdown_pct": round(r.max_drawdown_pct, 4),
                "avg_bars_held": round(r.avg_bars_held, 1),
                "recommendation": r.recommendation,
                "flag_reason": r.flag_reason,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _print_summary(results: List[PatternAblationResult], symbol: str) -> None:
    """Print a formatted summary table."""
    keep = [r for r in results if r.recommendation == "keep"]
    retune = [r for r in results if r.recommendation == "retune"]
    cut = [r for r in results if r.recommendation == "cut"]

    print(f"\n{'=' * 80}")
    print(f"  Pattern Ablation Results — {symbol}")
    print(f"{'=' * 80}")
    print(f"  Total patterns: {len(results)}")
    print(f"  Keep:  {len(keep)} (Sharpe > 0, PF > 1.0, >= 30 trades)")
    print(f"  Retune: {len(retune)} (low win rate or high drawdown)")
    print(f"  Cut:   {len(cut)} (negative Sharpe, PF < 1.0, or too few trades)")
    print(f"{'=' * 80}")

    print(
        f"\n  {'Pattern':<30} {'Cat':<14} {'Trades':>7} {'Win%':>7} {'PF':>7} {'Sharpe':>8} {'DD%':>7} {'Rec':>8}"
    )
    print(f"  {'-' * 30} {'-' * 14} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 8}")
    for r in results[:20]:
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor != 999.0 else "inf"
        print(
            f"  {r.pattern_name:<30} {r.category:<14} {r.n_trades:>7} "
            f"{r.win_rate:>6.1%} {pf_str:>7} {r.sharpe:>8.3f} "
            f"{r.max_drawdown_pct:>6.1%} {r.recommendation:>8}"
        )

    if len(results) > 20:
        print(f"  ... ({len(results) - 20} more patterns — see CSV for full results)")

    if cut:
        print(f"\n  Flagged for removal ({len(cut)}):")
        for r in cut:
            print(f"    ✗ {r.pattern_name}: {r.flag_reason}")

    if retune:
        print(f"\n  Flagged for retuning ({len(retune)}):")
        for r in retune:
            print(f"    ~ {r.pattern_name}: {r.flag_reason}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pattern Detector Ablation Study — measure which patterns produce edge"
    )
    parser.add_argument("--symbol", type=str, default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument(
        "--start", type=str, default="2019-01-01", help="Start date (default: 2019-01-01)"
    )
    parser.add_argument(
        "--end", type=str, default="2024-12-31", help="End date (default: 2024-12-31)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="reports", help="Output directory (default: reports)"
    )
    parser.add_argument("--all", action="store_true", help="Run on both SPY and BTC-USD")
    args = parser.parse_args()

    if args.all:
        for sym in ["SPY", "BTC-USD"]:
            run_ablation(sym, args.start, args.end, args.output_dir)
    else:
        run_ablation(args.symbol, args.start, args.end, args.output_dir)


if __name__ == "__main__":
    main()
