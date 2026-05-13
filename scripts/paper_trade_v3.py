"""
Paper Trading Simulation — point-in-time OOS evaluation.

Loads trained V3 model, walks forward bar-by-bar on each ticker's OOS period
using only past data for features. Enters long when probability > threshold,
exits via triple-barrier TP/SL/Time. Tracks P&L, Sharpe, drawdown, win rate.

Usage:
    uv run scripts/paper_trade_v3.py --model models/pattern_classifier_v3_JOE_20260511_031835.pkl --tickers JOE,SPY,QQQ

    uv run scripts/paper_trade_v3.py --model models/pattern_classifier_v3_JOE_20260511_031835.pkl --tickers SPY,KODK,QQQ --start 2021-01-01
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

from src.ml.cross_asset_features import CrossAssetFeatureExtractor, load_market_data
from src.ml.feature_engineering import FeatureExtractor
from src.ml.pattern_classifier import PatternClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5
DEFAULT_TP_MULT = 1.5
DEFAULT_SL_MULT = 1.0
DEFAULT_HORIZON = 5
CAPITAL = 100_000
RISK_PER_TRADE = 0.02


@dataclass
class Trade:
    ticker: str
    entry_date: Any
    entry_price: float
    exit_date: Any
    exit_price: float
    exit_reason: str
    return_pct: float
    bars_held: int
    model_prob: float


@dataclass
class PaperTradingResult:
    ticker: str
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series | None = None
    n_signals: int = 0
    total_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_return: float = 0.0
    avg_bars: float = 0.0

    def summary(self) -> dict:
        return {
            "ticker": self.ticker,
            "n_signals": self.n_signals,
            "n_trades": len(self.trades),
            "total_return_pct": round(self.total_return * 100, 2),
            "sharpe": round(self.sharpe, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "win_rate": round(self.win_rate, 3),
            "profit_factor": round(self.profit_factor, 2),
            "avg_return_pct": round(self.avg_return * 100, 2),
            "avg_bars_held": round(self.avg_bars, 1),
        }


def load_model(model_path: str | Path) -> PatternClassifier:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    clf = PatternClassifier(model_type="catboost", n_estimators=100)
    clf.load(path)
    logger.info(f"Loaded model: {path} ({len(clf.feature_names_)} features)")
    return clf


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
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
    gross_profit = sum(t.return_pct for t in trades if t.return_pct > 0)
    gross_loss = abs(sum(t.return_pct for t in trades if t.return_pct < 0))
    return float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")


def paper_trade_ticker(
    model: PatternClassifier,
    ticker: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    start_date: str = "2021-01-01",
    end_date: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    tp_mult: float = DEFAULT_TP_MULT,
    sl_mult: float = DEFAULT_SL_MULT,
    horizon: int = DEFAULT_HORIZON,
) -> PaperTradingResult:
    """Simulate bar-by-bar paper trading on a single ticker.

    Only uses data available at each bar for features (backward-looking).
    Enters long when model probability > threshold and no active trade.
    Exits via triple-barrier (TP/SL/Time).
    Tracks equity mark-to-market at every bar for accurate drawdown.
    """

    extractor = FeatureExtractor()

    # Pre-compute all features once (backward-looking only)
    full_features = extractor.extract_all_features(df, include_forward_returns=False)
    cols = [c for c in feature_cols if c in full_features.columns]
    full_features = full_features[cols]

    # Add cross-asset features
    try:
        market_data = load_market_data(df)
        ca = CrossAssetFeatureExtractor(market_data=market_data)
        ca_fx = ca.extract(df)
        ca_cols = [c for c in feature_cols if c in ca_fx.columns]
        if ca_cols:
            full_features = full_features.join(ca_fx[ca_cols], how="left")
            full_features = full_features.ffill().bfill()
    except Exception:
        pass

    cols = [c for c in feature_cols if c in full_features.columns]
    full_features = full_features[cols].ffill().bfill()

    atr = _compute_atr(df, period=14)

    close_arr = df["Close"].values
    high_arr = df["High"].values
    low_arr = df["Low"].values
    dates = df.index

    results = PaperTradingResult(ticker=ticker)
    results.n_signals = 0

    in_trade = False
    trade_entry_bar: int = 0
    entry_price: float = 0.0
    entry_prob: float = 0.0
    trade_tp: float = 0.0
    trade_sl: float = 0.0
    trade_position: float = 0.0  # number of shares
    bars_in_trade: int = 0
    cash = float(CAPITAL)
    equity_curve: list[float] = []

    start_idx = 0
    if start_date:
        start_idx = max(
            0,
            (dates >= pd.Timestamp(start_date)).argmax()
            if any(dates >= pd.Timestamp(start_date))
            else 0,
        )

    end_idx = len(df)
    if end_date:
        end_idx = min(end_idx, (dates <= pd.Timestamp(end_date)).sum())

    warmup = 200  # Need enough bars for feature windows
    start_idx = max(start_idx, warmup)

    for i in range(start_idx, end_idx):
        current_close = close_arr[i]
        current_high = high_arr[i]
        current_low = low_arr[i]

        # --- Mark-to-market during active trade ---
        if in_trade:
            bars_in_trade += 1
            position_value = trade_position * current_close

            # Check intra-bar exit (TP/SL hit on this bar)
            if current_high >= trade_tp:
                exit_close = trade_tp
                exit_reason = "TP"
            elif current_low <= trade_sl:
                exit_close = trade_sl
                exit_reason = "SL"
            elif bars_in_trade >= horizon:
                exit_close = current_close
                exit_reason = "TIME"
            else:
                # Still in trade — mark-to-market equity
                equity = cash + position_value
                equity_curve.append(equity)
                continue

            # Trade exit
            ret = (exit_close - entry_price) / entry_price
            pnl = trade_position * (exit_close - entry_price)
            cash += trade_position * exit_close

            trade = Trade(
                ticker=ticker,
                entry_date=dates[trade_entry_bar],
                entry_price=float(entry_price),
                exit_date=dates[i],
                exit_price=float(exit_close),
                exit_reason=exit_reason,
                return_pct=float(ret),
                bars_held=bars_in_trade,
                model_prob=float(entry_prob),
            )
            results.trades.append(trade)

            in_trade = False
            equity_curve.append(cash)
            continue

        # --- Not in trade: generate signal ---
        try:
            feats = full_features.iloc[i : i + 1]
            if feats.isnull().any(axis=1).iloc[0]:
                equity_curve.append(cash)
                continue
            pred = model.predict(feats)
            prob = (
                pred.iloc[0, pred.columns.get_loc("probability_profitable")]
                if isinstance(pred, pd.DataFrame)
                else float(pred)
            )
        except Exception:
            equity_curve.append(cash)
            continue

        results.n_signals += 1

        if prob < threshold:
            equity_curve.append(cash)
            continue

        # Enter long at next available price (current close, for paper trading)
        current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else current_close * 0.02
        trade_tp = current_close + tp_mult * current_atr
        trade_sl = current_close - sl_mult * current_atr
        risk_per_share = max(abs(current_close - trade_sl), current_close * 0.005)

        # Position sizing: risk 2% of cash per trade
        risk_amount = cash * RISK_PER_TRADE
        trade_position = risk_amount / risk_per_share
        # Cap position at 50% of cash
        max_position = cash * 0.5 / current_close
        trade_position = min(trade_position, max_position)

        entry_price = current_close
        trade_entry_bar = i
        entry_prob = prob
        bars_in_trade = 0

        # Deduct purchase cost from cash
        cost = trade_position * current_close
        cash -= cost

        in_trade = True
        # Record equity after entry (cash + position mark-to-market)
        equity_curve.append(cash + trade_position * current_close)

    if not equity_curve:
        return results

    results.equity_curve = pd.Series(
        equity_curve, index=dates[start_idx : start_idx + len(equity_curve)]
    )

    # Compute daily returns from equity curve for proper Sharpe
    if len(results.equity_curve) > 1:
        daily_returns = results.equity_curve.pct_change().dropna().values
    else:
        daily_returns = np.array([])

    results.total_return = equity_curve[-1] / CAPITAL - 1
    results.sharpe = _sharpe(daily_returns) if len(daily_returns) > 5 else 0.0
    results.max_drawdown = _max_drawdown(np.array(equity_curve))
    results.win_rate = (
        float(np.mean([1 if t.return_pct > 0 else 0 for t in results.trades]))
        if results.trades
        else 0.0
    )
    results.profit_factor = _profit_factor(results.trades) if results.trades else 0.0
    results.avg_return = (
        float(np.mean([t.return_pct for t in results.trades])) if results.trades else 0.0
    )
    results.avg_bars = (
        float(np.mean([t.bars_held for t in results.trades])) if results.trades else 0.0
    )

    return results


def plot_equity_curves(
    results: dict[str, PaperTradingResult],
    save_path: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, (ticker, r) in enumerate(sorted(results.items())):
        ax = axes[idx // cols][idx % cols]
        if r.equity_curve is not None and len(r.equity_curve) > 0:
            ax.plot(r.equity_curve.index, r.equity_curve.values, color="steelblue", linewidth=1)
            ax.axhline(CAPITAL, color="gray", linestyle="--", alpha=0.5)
            ax.set_title(
                f"{ticker} | Ret={r.total_return * 100:.1f}% "
                f"S={r.sharpe:.2f} DD={r.max_drawdown * 100:.1f}% "
                f"W={r.win_rate:.2f}"
            )
        ax.grid(alpha=0.3)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Equity curves saved to {save_path}")
    plt.close()


def _compute_spy_benchmark(spy_df: pd.DataFrame, start_date: str, end_date: str | None) -> dict:
    """Compute SPY buy-and-hold returns over the paper trading period."""
    if spy_df is None:
        return {}
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) if end_date else spy_df.index[-1]
    mask = (spy_df.index >= start_ts) & (spy_df.index <= end_ts)
    window = spy_df.loc[mask]
    if len(window) < 2:
        return {}
    entry = window["Close"].iloc[0]
    exit_p = window["Close"].iloc[-1]
    bh_return = (exit_p - entry) / entry
    daily_ret = window["Close"].pct_change().dropna().values
    bh_sharpe = _sharpe(daily_ret) if len(daily_ret) > 5 else 0.0
    bh_max_dd = _max_drawdown(window["Close"].values)
    return {
        "spy_bh_return_pct": round(bh_return * 100, 2),
        "spy_bh_sharpe": round(bh_sharpe, 2),
        "spy_bh_max_dd_pct": round(bh_max_dd * 100, 2),
    }


def run_paper_trading(
    model_path: str,
    tickers: list[str],
    start_date: str = "2021-01-01",
    end_date: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    tp_mult: float = DEFAULT_TP_MULT,
    sl_mult: float = DEFAULT_SL_MULT,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, PaperTradingResult]:
    model = load_model(model_path)
    feature_cols = model.feature_names_

    results: dict[str, PaperTradingResult] = {}
    spy_df: pd.DataFrame | None = None

    for ticker in tickers:
        path = Path(f"data/raw/{ticker}_daily.csv")
        if not path.exists():
            logger.warning(f"No data for {ticker}, skipping")
            continue

        df = pd.read_csv(path, parse_dates=True, index_col=0).dropna()
        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                df[col] = 0 if col == "Volume" else df["Close"]

        if ticker == "SPY":
            spy_df = df.copy()

        logger.info(
            f"Paper trading {ticker}: {len(df)} bars, "
            f"start={start_date}, {len(feature_cols)} features"
        )

        r = paper_trade_ticker(
            model=model,
            ticker=ticker,
            df=df,
            feature_cols=feature_cols,
            start_date=start_date,
            end_date=end_date,
            threshold=threshold,
            tp_mult=tp_mult,
            sl_mult=sl_mult,
            horizon=horizon,
        )

        s = r.summary()
        logger.info(
            f"  {ticker}: trades={s['n_trades']}/{s['n_signals']} signals, "
            f"ret={s['total_return_pct']}%, sharpe={s['sharpe']:.2f}, "
            f"dd={s['max_drawdown_pct']}%, win={s['win_rate']:.2f}, "
            f"PF={s['profit_factor']:.2f}"
        )
        results[ticker] = r

    # Compute SPY benchmark (load if SPY not in ticker list)
    if spy_df is None:
        spy_path = Path("data/raw/SPY_daily.csv")
        if spy_path.exists():
            spy_df = pd.read_csv(spy_path, parse_dates=True, index_col=0).dropna()

    benchmark = _compute_spy_benchmark(spy_df, start_date, end_date or "")
    if benchmark:
        logger.info(
            f"  SPY B&H: ret={benchmark['spy_bh_return_pct']}%, "
            f"sharpe={benchmark['spy_bh_sharpe']:.2f}, "
            f"dd={benchmark['spy_bh_max_dd_pct']}%"
        )
        results["__benchmark__"] = benchmark  # type: ignore[assignment]

    return results


def main():
    parser = argparse.ArgumentParser(description="Paper trade V3 ML model")
    parser.add_argument("--model", required=True, help="Path to trained .pkl model")
    parser.add_argument("--tickers", default="SPY,QQQ,IWM,JOE,KODK", help="Comma-separated tickers")
    parser.add_argument("--start", default="2021-01-01", help="OOS start date")
    parser.add_argument("--end", default=None, help="OOS end date")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD, help="Entry probability threshold"
    )
    parser.add_argument(
        "--tp-mult", type=float, default=DEFAULT_TP_MULT, help="ATR multiplier for take-profit"
    )
    parser.add_argument(
        "--sl-mult", type=float, default=DEFAULT_SL_MULT, help="ATR multiplier for stop-loss"
    )
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Max bars to hold")
    parser.add_argument("--output", default=None, help="Output directory for reports")

    args = parser.parse_args()
    tickers = [t.strip() for t in args.tickers.split(",")]

    results = run_paper_trading(
        model_path=args.model,
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        threshold=args.threshold,
        tp_mult=args.tp_mult,
        sl_mult=args.sl_mult,
        horizon=args.horizon,
    )

    # Save summary
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path("reports/paper_trading") / datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)

    summary = {t: r.summary() for t, r in results.items() if not t.startswith("__")}
    benchmark_data = results.get("__benchmark__", {})
    if benchmark_data:
        summary["__benchmark__"] = {
            "ticker": "SPY B&H",
            "n_signals": 0,
            "n_trades": 0,
            "total_return_pct": benchmark_data["spy_bh_return_pct"],
            "sharpe": benchmark_data["spy_bh_sharpe"],
            "max_drawdown_pct": benchmark_data["spy_bh_max_dd_pct"],
            "win_rate": 0,
            "profit_factor": 0,
            "avg_return_pct": 0,
            "avg_bars_held": 0,
        }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    plot_equity_curves(
        {t: r for t, r in results.items() if not t.startswith("__")},
        save_path=str(out_dir / "equity_curves.png"),
    )

    # Console table
    logger.info("=" * 100)
    logger.info(
        f"{'Ticker':<8} {'Trades':>6} {'Ret%':>8} {'Sharpe':>7} "
        f"{'MaxDD%':>8} {'WinRate':>8} {'PF':>6} {'AvgRet%':>8} {'vsB&H':>8}"
    )
    logger.info("-" * 100)
    for ticker in sorted(results.keys()):
        if ticker.startswith("__"):
            continue
        s = results[ticker].summary()
        bh_diff = ""
        if benchmark_data and ticker == "SPY":
            delta = s["total_return_pct"] - benchmark_data["spy_bh_return_pct"]
            bh_diff = f"{delta:+.1f}%"
        logger.info(
            f"{s['ticker']:<8} {s['n_trades']:>6} {s['total_return_pct']:>8.1f} "
            f"{s['sharpe']:>7.2f} {s['max_drawdown_pct']:>8.1f} "
            f"{s['win_rate']:>8.2f} {s['profit_factor']:>6.2f} "
            f"{s['avg_return_pct']:>8.1f} {bh_diff:>8}"
        )
    if benchmark_data:
        logger.info("-" * 100)
        logger.info(
            f"{'SPY B&H':<8} {'':>6} {benchmark_data['spy_bh_return_pct']:>8.1f} "
            f"{benchmark_data['spy_bh_sharpe']:>7.2f} {benchmark_data['spy_bh_max_dd_pct']:>8.1f} "
            f"{'':>8} {'':>6} {'':>8} {'':>8}"
        )

    logger.info(f"\nReport saved to {out_dir}")


if __name__ == "__main__":
    main()
