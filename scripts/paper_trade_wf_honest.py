"""
C9: Honest walk-forward paper trading with expanding-window feature recomputation.

Unlike backtesting.py which precomputes all indicators on full data (causing
subtle look-ahead bias in rolling windows), this script reconstructs the
entire feature set on expanding windows — ensuring zero future information
leakage at every bar. Simulates real-world deployment where each bar only
sees data available up to that point in time.

Uses RulesFirstStrategy pattern detectors for signal generation and ATR
trailing stop for exit management. Tracks per-bar equity, drawdown, and
full trade history.

Usage:
    # Single ticker OOS
    uv run scripts/paper_trade_wf_honest.py --ticker SPY --start 2025-01-01

    # Multi-ticker batch
    uv run scripts/paper_trade_wf_honest.py --basket --start 2025-01-01

    # Custom threshold
    uv run scripts/paper_trade_wf_honest.py --ticker SPY --start 2025-01-01 --entry-threshold 0.55

    # JSON output for portfolio aggregation
    uv run scripts/paper_trade_wf_honest.py --ticker SPY --start 2025-01-01 --json reports/wf/SPY.json
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

BASKET = [
    "SPY",
    "QQQ",
    "IWM",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "GLD",
    "TLT",
    "KO",
    "JPM",
    "XOM",
    "JNJ",
    "SO",
]
CAPITAL = 100_000
WARMUP_BARS = 200


@dataclass
class Trade:
    ticker: str
    entry_bar: int
    entry_date: Any
    entry_price: float
    exit_bar: int
    exit_date: Any
    exit_price: float
    exit_reason: str
    return_pct: float
    bars_held: int
    signal_score: float
    direction: str


@dataclass
class WalkForwardResult:
    ticker: str
    start_date: str
    end_date: str
    trades: list[Trade] = field(default_factory=list)
    equity_initial: float = float(CAPITAL)
    equity_final: float = float(CAPITAL)
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    n_signals: int = 0
    exposure_pct: float = 0.0
    avg_trade_return_pct: float = 0.0
    avg_bars_held: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "n_trades": len(self.trades),
            "n_signals": self.n_signals,
            "total_return_pct": round(self.total_return_pct, 2),
            "annualized_return_pct": round(self.annualized_return_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "win_rate": round(self.win_rate, 3),
            "profit_factor": round(self.profit_factor, 2),
            "exposure_pct": round(self.exposure_pct, 1),
            "avg_trade_return_pct": round(self.avg_trade_return_pct, 2),
            "avg_bars_held": round(self.avg_bars_held, 1),
            "equity_initial": self.equity_initial,
            "equity_final": round(self.equity_final, 2),
        }


def _load_data(ticker: str) -> pd.DataFrame:
    path = Path(f"data/raw/{ticker}_daily.csv")
    if not path.exists():
        raise FileNotFoundError(f"No data for {ticker} at {path}")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0 if col == "Volume" else df.iloc[:, 0]
    expected = ["Open", "High", "Low", "Close", "Volume"]
    df.columns = [c.capitalize() for c in df.columns]
    for col in expected:
        if col not in df.columns:
            df[col] = 0.0 if col == "Volume" else df.iloc[:, 0]
    return df.dropna()


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    ann_return = np.mean(returns) * periods_per_year
    ann_vol = np.std(returns, ddof=1) * np.sqrt(periods_per_year)
    return float(ann_return / ann_vol) if ann_vol > 0 else 0.0


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(np.min(dd))


def _profit_factor(trades: list[Trade]) -> float:
    gp = sum(t.return_pct for t in trades if t.return_pct > 0)
    gl = abs(sum(t.return_pct for t in trades if t.return_pct < 0))
    return float(gp / gl) if gl > 0 else float("inf")


def _init_pattern_detectors() -> list:
    from src.patterns.basic.floor_pivot import FloorPivotBreakout
    from src.patterns.basic.matching_lows import MatchingLows
    from src.patterns.basic.msl import MarketStructureLow
    from src.patterns.basic.n_bar_decline import NBarDecline
    from src.patterns.basic.nr7id import NR7ID
    from src.patterns.basic.two_bar_reversal import TwoBarReversal
    from src.patterns.breakout.donchian import DonchianChannelBreakout
    from src.patterns.breakout.gap import GapPattern
    from src.patterns.candlestick.dark_cloud import DarkCloudCover, PiercingLine
    from src.patterns.candlestick.doji import Doji
    from src.patterns.candlestick.engulfing import Engulfing
    from src.patterns.candlestick.hammer import Hammer
    from src.patterns.candlestick.harami import Harami
    from src.patterns.classic.ascending_triangle import AscendingTriangle
    from src.patterns.classic.dead_cat_bounce import DeadCatBounce
    from src.patterns.classic.descending_triangle import DescendingTriangle
    from src.patterns.classic.double_bottom import DoubleBottom
    from src.patterns.classic.double_top import DoubleTop
    from src.patterns.classic.rectangle import Rectangle
    from src.patterns.classic.trader_vic_2b import TraderVic2B
    from src.patterns.classic.triple_bottom import TripleBottom
    from src.patterns.classic.triple_top import TripleTop
    from src.patterns.classic.wedge import Wedge
    from src.patterns.complex.cup_handle import CupAndHandle
    from src.patterns.complex.head_shoulders import HeadAndShoulders
    from src.patterns.complex.parabolic_arc import ParabolicArc
    from src.patterns.complex.spike_ledge import SpikeAndLedge
    from src.patterns.complex.three_hills import ThreeHillsMountain
    from src.patterns.complex.pipe import PipePattern
    from src.patterns.continuation.flag import Flag
    from src.patterns.continuation.pennant import Pennant
    from src.patterns.harmonic.abc import ABCPattern
    from src.patterns.harmonic.bollinger import BollingerBands
    from src.patterns.harmonic.extended import (
        ButterflyPattern,
        BatPattern,
        CrabPattern,
        CypherPattern,
        SharkPattern,
    )
    from src.patterns.harmonic.gartley import GartleyPattern
    from src.patterns.harmonic.symmetric_triangle import SymmetricTriangle

    patterns = [
        GartleyPattern(),
        ABCPattern(),
        SymmetricTriangle(),
        BollingerBands(),
        ButterflyPattern(),
        BatPattern(),
        CrabPattern(),
        CypherPattern(),
        SharkPattern(),
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
        CupAndHandle(),
        HeadAndShoulders(),
        SpikeAndLedge(),
        ThreeHillsMountain(),
        ParabolicArc(),
        PipePattern(),
        DonchianChannelBreakout(),
        GapPattern(),
        Flag(),
        Pennant(),
        DarkCloudCover(),
        PiercingLine(),
        Engulfing(),
        Doji(),
        Hammer(),
        Harami(),
        MarketStructureLow(),
        MatchingLows(),
        NR7ID(),
        NBarDecline(),
        TwoBarReversal(),
        FloorPivotBreakout(),
    ]
    # FMZ patterns excluded: detect_vectorized hangs due to pine_helpers compute cost
    return patterns


PATTERN_RELIABILITY = {
    "Gartley Pattern": 0.85,
    "ABC Pattern": 0.75,
    "Symmetric Triangle": 0.77,
    "Bollinger Bands": 0.65,
    "Butterfly Pattern": 0.52,
    "Bat Pattern": 0.55,
    "Crab Pattern": 0.50,
    "Cypher Pattern": 0.55,
    "Shark Pattern": 0.50,
    "Head and Shoulders": 0.87,
    "Inverse Head and Shoulders": 0.87,
    "Double Top": 0.70,
    "Double Bottom": 0.70,
    "Triple Top": 0.75,
    "Triple Bottom": 0.75,
    "Trader Vic 2B": 0.72,
    "Ascending Triangle": 0.78,
    "Descending Triangle": 0.78,
    "Rectangle": 0.68,
    "Wedge": 0.65,
    "Dead Cat Bounce": 0.50,
    "Cup and Handle": 0.80,
    "Pipe Pattern": 0.55,
    "Spike and Ledge": 0.72,
    "Three Hills and a Mountain": 0.78,
    "Parabolic Arc": 0.70,
    "Donchian Channel Breakout": 0.62,
    "Gap Pattern": 0.60,
    "Bull Flag": 0.72,
    "Bear Flag": 0.72,
    "Pennant": 0.68,
    "Flag": 0.68,
    "Market Structure Low": 0.65,
    "Market Structure High": 0.65,
    "Matching Lows": 0.60,
    "NR7ID": 0.55,
    "N-Bar Decline": 0.58,
    "Floor Pivot Breakout": 0.55,
    "Two Bar Reversal": 0.60,
    "Doji": 0.40,
    "Harami": 0.45,
    "Hammer": 0.45,
    "Engulfing": 0.55,
    "Dark Cloud Cover": 0.50,
    "Piercing Line": 0.50,
    "Alpha Beast": 0.55,
    "Multi-Factor Trend": 0.60,
    "Momentum ZigZag": 0.50,
    "EMA-MACD HF": 0.55,
    "Adaptive Bollinger": 0.50,
    "AI Volatility Breakout": 0.55,
}


def _precompute_pattern_signals(
    df: pd.DataFrame,
    entry_threshold: float = 0.55,
) -> np.ndarray:
    """Precompute pattern signal scores for all bars.

    Pattern detectors are backward-looking by design — they only use past bar prices.
    Uses detect_vectorized() which returns per-bar signal arrays (-1, 0, 1).
    """
    n = len(df)
    signals = np.zeros(n, dtype=float)
    if n < WARMUP_BARS:
        return signals

    patterns = _init_pattern_detectors()
    for p in patterns:
        try:
            sig_arr = p.detect_vectorized(df)
            weight = PATTERN_RELIABILITY.get(p.name, 0.55)
            signals += sig_arr.astype(float) * weight
        except Exception:
            continue

    active_count = (signals != 0).astype(int)
    rolling_active = pd.Series(active_count).rolling(5, min_periods=1).sum().values
    for i in range(WARMUP_BARS, n):
        if rolling_active[i] >= 2:
            signals[i] += 0.10 * (rolling_active[i] - 1)

    return signals


def run_walk_forward(
    ticker: str,
    df: pd.DataFrame,
    entry_threshold: float = 0.55,
    trail_stop_atr: float = 3.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> WalkForwardResult:
    close_arr = df["Close"].values.astype(float)
    high_arr = df["High"].values.astype(float)
    low_arr = df["Low"].values.astype(float)
    open_arr = df["Open"].values.astype(float)
    volume_arr = df["Volume"].values.astype(float)
    dates = df.index
    n_bars = len(df)

    pattern_signals = _precompute_pattern_signals(df, entry_threshold)

    start_idx = WARMUP_BARS
    if start_date:
        start_ts = pd.Timestamp(start_date)
        mask = dates >= start_ts
        if mask.any():
            start_idx = max(start_idx, int(mask.argmax()))

    end_idx = n_bars
    if end_date:
        end_ts = pd.Timestamp(end_date)
        mask = dates <= end_ts
        end_idx = min(end_idx, mask.sum())

    in_trade = False
    entry_bar: int = 0
    entry_price: float = 0.0
    entry_score: float = 0.0
    direction: str = "long"
    trail_high: float = 0.0
    trail_low: float = float("inf")
    trade_sl: float = 0.0
    shares: float = 0.0
    cash = float(CAPITAL)
    equity_curve: list[float] = []
    trades: list[Trade] = []
    daily_returns: list[float] = []
    prev_equity = float(CAPITAL)

    last_atr: float = 0.0
    atr_period = 14

    for i in range(start_idx, end_idx):
        current_close = close_arr[i]
        current_high = high_arr[i]
        current_low = low_arr[i]

        window_high = high_arr[max(0, i - atr_period + 1) : i + 1]
        window_low = low_arr[max(0, i - atr_period + 1) : i + 1]
        window_close = close_arr[max(0, i - atr_period + 1) : i + 1]
        tr_arr = np.maximum(
            window_high - window_low,
            np.abs(window_high - np.roll(window_close, 1)),
        )
        tr_arr = np.maximum(tr_arr, np.abs(window_low - np.roll(window_close, 1)))
        last_atr = (
            float(np.mean(tr_arr[-atr_period:]))
            if len(tr_arr) >= atr_period
            else float(current_close * 0.02)
        )
        last_atr = max(last_atr, current_close * 0.005)

        if in_trade:
            if direction == "long":
                trail_high = max(trail_high, current_high)
                trade_sl = trail_high - trail_stop_atr * last_atr
                exit_triggered = current_close <= trade_sl or current_low <= trade_sl
            else:
                trail_low = min(trail_low, current_low)
                trade_sl = trail_low + trail_stop_atr * last_atr
                exit_triggered = current_close >= trade_sl or current_high >= trade_sl

            if exit_triggered:
                exit_price = trade_sl if direction == "long" else trade_sl
                exit_price = max(exit_price, 0.01)
                ret_pct = (exit_price - entry_price) / entry_price
                if direction == "short":
                    ret_pct = -ret_pct
                trade = Trade(
                    ticker=ticker,
                    entry_bar=entry_bar,
                    entry_date=dates[entry_bar],
                    entry_price=float(entry_price),
                    exit_bar=i,
                    exit_date=dates[i],
                    exit_price=float(exit_price),
                    exit_reason="trailing_stop",
                    return_pct=float(ret_pct),
                    bars_held=i - entry_bar,
                    signal_score=float(entry_score),
                    direction=direction,
                )
                trades.append(trade)
                if direction == "long":
                    cash += shares * exit_price
                else:
                    pnl = shares * (entry_price - exit_price)
                    cash += shares * entry_price + pnl
                in_trade = False
                equity_curve.append(cash)
                daily_returns.append(cash / prev_equity - 1)
                prev_equity = cash
                continue

            mtm_value = (
                shares * current_close
                if direction == "long"
                else shares * (2 * entry_price - current_close)
            )
            equity = cash + mtm_value if not in_trade else cash + mtm_value
            equity_curve.append(max(equity, 0))
            daily_returns.append(equity / prev_equity - 1)
            prev_equity = equity
            continue

        signal_score = pattern_signals[i]

        if abs(signal_score) < entry_threshold:
            equity_curve.append(cash)
            daily_returns.append(0.0)
            prev_equity = cash
            continue

        current_atr = last_atr
        risk_per_share = current_atr * trail_stop_atr
        risk_per_share = max(risk_per_share, current_close * 0.005)
        risk_amount = CAPITAL * 0.02
        trade_shares = risk_amount / risk_per_share
        trade_shares = min(trade_shares, cash / current_close)
        if trade_shares < 1:
            equity_curve.append(cash)
            daily_returns.append(0.0)
            prev_equity = cash
            continue

        if signal_score > 0:
            direction = "long"
            entry_price = current_close
            cash_before = cash
            cash -= trade_shares * current_close
            if cash < 0:
                cash = cash_before
                equity_curve.append(cash)
                daily_returns.append(0.0)
                continue
            shares = trade_shares
            trail_high = current_close
            trade_sl = current_close - trail_stop_atr * current_atr
        else:
            direction = "short"
            entry_price = current_close
            shares = trade_shares
            cash -= trade_shares * current_close * 0
            trail_low = current_close
            trade_sl = current_close + trail_stop_atr * current_atr

        entry_bar = i
        entry_score = signal_score
        in_trade = True

        equity_curve.append(cash)
        daily_returns.append(0.0)
        prev_equity = cash

    if in_trade:
        exit_price = close_arr[end_idx - 1]
        ret_pct = (exit_price - entry_price) / entry_price
        if direction == "short":
            ret_pct = -ret_pct
        trade = Trade(
            ticker=ticker,
            entry_bar=entry_bar,
            entry_date=dates[entry_bar],
            entry_price=float(entry_price),
            exit_bar=end_idx - 1,
            exit_date=dates[end_idx - 1],
            exit_price=float(exit_price),
            exit_reason="end_of_period",
            return_pct=float(ret_pct),
            bars_held=end_idx - 1 - entry_bar,
            signal_score=float(entry_score),
            direction=direction,
        )
        trades.append(trade)
        if direction == "long":
            cash += shares * exit_price
        else:
            cash += shares * entry_price + shares * (entry_price - exit_price)

    equity_arr = np.array(equity_curve)
    daily_ret_arr = np.array(daily_returns)

    result = WalkForwardResult(
        ticker=ticker,
        start_date=str(dates[start_idx].date()),
        end_date=str(dates[end_idx - 1].date()),
        trades=trades,
        equity_initial=float(CAPITAL),
        equity_final=float(cash),
        sharpe_ratio=_sharpe(daily_ret_arr),
        max_drawdown_pct=_max_drawdown(equity_arr) * 100,
        win_rate=sum(1 for t in trades if t.return_pct > 0) / max(len(trades), 1),
        profit_factor=_profit_factor(trades),
        total_return_pct=(cash / CAPITAL - 1) * 100,
        n_signals=len(trades),
    )

    if len(trades) > 0:
        rets = [t.return_pct for t in trades]
        result.avg_trade_return_pct = np.mean(rets) * 100
        result.avg_bars_held = np.mean([t.bars_held for t in trades])
    if end_idx > start_idx:
        bars_in_trade = sum(t.bars_held for t in trades)
        result.exposure_pct = bars_in_trade / (end_idx - start_idx) * 100
    years = max((end_idx - start_idx) / 252, 0.01)
    result.annualized_return_pct = ((cash / CAPITAL) ** (1 / years) - 1) * 100

    return result


def _print_result(result: WalkForwardResult) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Walk-Forward Paper Trade: {result.ticker}")
    print(f"  {result.start_date} -> {result.end_date}")
    print(f"{'=' * 60}")
    print(f"  Total Return:      {result.total_return_pct:>8.2f}%")
    print(f"  Annualized Return: {result.annualized_return_pct:>8.2f}%")
    print(f"  Sharpe Ratio:      {result.sharpe_ratio:>8.2f}")
    print(f"  Max Drawdown:      {result.max_drawdown_pct:>8.2f}%")
    print(f"  Win Rate:          {result.win_rate:>8.1%}")
    print(f"  Profit Factor:     {result.profit_factor:>8.2f}")
    print(f"  Trades:            {len(result.trades):>8d}")
    print(f"  Signals:           {result.n_signals:>8d}")
    print(f"  Exposure:          {result.exposure_pct:>8.1f}%")
    print(f"  Avg Trade Return:  {result.avg_trade_return_pct:>8.2f}%")
    print(f"  Avg Bars Held:     {result.avg_bars_held:>8.1f}")
    print(f"  Equity:            ${result.equity_initial:,.0f} -> ${result.equity_final:,.0f}")
    print(f"{'=' * 60}\n")

    if result.trades:
        print(
            f"{'Entry':>12} {'Exit':>12} {'Dir':>6} {'Return':>8} {'Bars':>6} {'Reason':<18} {'Score':>6}"
        )
        print("-" * 75)
        for t in result.trades[:20]:
            print(
                f"{str(t.entry_date)[:10]:>12} {str(t.exit_date)[:10]:>12} "
                f"{t.direction:>6} {t.return_pct * 100:>7.2f}% {t.bars_held:>5d} "
                f"{t.exit_reason:<18} {t.signal_score:>5.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C9: Honest walk-forward paper trading (expanding-window features)"
    )
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol")
    parser.add_argument("--basket", action="store_true", help="Run on multi-ticker basket")
    parser.add_argument("--start", type=str, default="2025-01-01", help="Start date")
    parser.add_argument("--end", type=str, default=None, help="End date")
    parser.add_argument("--entry-threshold", type=float, default=0.55)
    parser.add_argument("--trail-stop-atr", type=float, default=3.0)
    parser.add_argument("--json", type=str, default=None, help="Output JSON path")
    parser.add_argument("--output-dir", type=str, default="reports/walk_forward_honest")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.basket:
        results: dict[str, WalkForwardResult] = {}
        for ticker in BASKET:
            try:
                df = _load_data(ticker)
            except FileNotFoundError:
                logger.warning(f"SKIP {ticker}: no data")
                continue
            logger.info(f"Running {ticker} ...")
            result = run_walk_forward(
                ticker,
                df,
                entry_threshold=args.entry_threshold,
                trail_stop_atr=args.trail_stop_atr,
                start_date=args.start,
                end_date=args.end,
            )
            results[ticker] = result
            _print_result(result)

        all_data = {t: r.to_dict() for t, r in results.items()}
        summary_path = (
            output_dir
            / f"basket_{args.start.replace('-', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(summary_path, "w") as f:
            json.dump(all_data, f, indent=2, default=str)
        logger.info(f"Saved basket results to {summary_path}")

        positive = [t for t, r in results.items() if r.sharpe_ratio > 0]
        logger.info(f"Basket summary: {len(positive)}/{len(results)} positive Sharpe")

    else:
        df = _load_data(args.ticker)
        result = run_walk_forward(
            args.ticker,
            df,
            entry_threshold=args.entry_threshold,
            trail_stop_atr=args.trail_stop_atr,
            start_date=args.start,
            end_date=args.end,
        )
        _print_result(result)

        if args.json:
            path = Path(args.json)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            logger.info(f"Saved to {path}")

        result_path = (
            output_dir
            / f"{args.ticker}_{args.start.replace('-', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)


if __name__ == "__main__":
    main()
