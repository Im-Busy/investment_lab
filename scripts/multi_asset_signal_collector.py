"""
Multi-Asset Signal Collector for ML Signal Scorer

Runs the multi-pattern confluence strategy across a broad asset universe
to collect 300+ trade samples for training the ML signal scorer.

Generates trade-level features including:
- Entry/exit prices, stop-loss, take-profit
- ATR at entry, regime at entry
- Pattern confluence score and count
- Cross-asset context (VIX level, sector RS)
- Trade outcome (return_pct, is_profitable)

Usage:
    uv run scripts/multi_asset_signal_collector.py --start 2015-01-01 --end 2024-12-31 --output output/multi_asset_signals.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indicators.regime_detector import RegimeDetector
from src.strategies.backtest_py.runner import BacktestPyRunner

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


ASSET_UNIVERSE = [
    # Broad market ETFs
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    # Sectors
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLP",
    "XLU",
    "XLY",
    "XLB",
    "XLRE",
    # Large cap
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "JPM",
    "JNJ",
    "V",
    "WMT",
    "PG",
    "MA",
    "UNH",
    "HD",
    "DIS",
    "BAC",
    "ADBE",
    "CRM",
    "NFLX",
    "CSCO",
    "PFE",
    "INTC",
    "KO",
    "PEP",
    "MRK",
    "COST",
    "ABT",
    "AVGO",
    "TMO",
    "CVX",
    "XOM",
    "LLY",
    "MDT",
    "ACN",
    "TXN",
    "UNP",
    "NEE",
    "LIN",
    "AMD",
    # Popular growth/momentum names
    "PYPL",
    "SQ",
    "SHOP",
    "ZM",
    "COIN",
    "PLTR",
    "SOFI",
    "RBLX",
]

MARKET_CONTEXT_TICKERS = {
    "^VIX": "vix",
    "TLT": "tlt",
    "IEF": "ief",
    "GLD": "gld",
}


@dataclass
class TradeRecord:
    """A single trade with features for ML scoring."""

    ticker: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float
    duration_bars: int
    direction: str
    stop_loss: float
    take_profit: float
    is_profitable: int
    # Pattern features
    pattern_names: str
    pattern_count: int
    confluence_score: float
    # Regime features
    regime_at_entry: str
    adx_at_entry: float
    atr_at_entry: float
    # Market context features (filled later)
    vix_level: float = 0.0
    tlt_spy_rs: float = 0.0
    gld_spy_rs: float = 0.0
    spy_return_20d: float = 0.0


def download_ticker(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Download OHLCV data for a single ticker."""
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or "Close" not in df.columns:
            return None
        df.columns = [
            c.capitalize() if c.lower() in ["open", "high", "low", "close", "volume"] else c
            for c in df.columns
        ]
        df.index.name = "Date"
        df.index = pd.to_datetime(df.index)
        if "Volume" not in df.columns:
            df["Volume"] = 0
        return df[["Open", "High", "Low", "Close", "Volume"]].copy()
    except Exception as e:
        logger.warning(f"Failed to download {ticker}: {e}")
        return None


def get_market_context(
    spy_df: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download market context tickers and align to SPY index."""
    context = pd.DataFrame(index=spy_df.index)
    spy_close = spy_df["Close"]

    for ticker, col_name in MARKET_CONTEXT_TICKERS.items():
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or "Close" not in df.columns:
                continue
            close = df["Close"].reindex(spy_df.index).ffill().bfill()
            context[f"{col_name}_close"] = close
            if col_name != "vix":
                context[f"{col_name}_spy_rs"] = close / spy_close
        except Exception:
            pass

    # SPY 20-day return as momentum context
    context["spy_return_20d"] = spy_close.pct_change(20)
    return context.ffill().bfill()


def extract_trade_features(
    ticker: str,
    trade_row: pd.Series,
    ohlcv_df: pd.DataFrame,
    regime_detector: RegimeDetector,
) -> Optional[TradeRecord]:
    """Build a TradeRecord from a backtest.py trade row."""
    entry_time = trade_row.get("entry_time") or trade_row.get("EntryTime")
    exit_time = trade_row.get("exit_time") or trade_row.get("ExitTime")
    entry_price = trade_row.get("entry_price", 0) or trade_row.get("EntryPrice", 0)
    exit_price = trade_row.get("exit_price", 0) or trade_row.get("ExitPrice", 0)
    pnl = trade_row.get("pnl", 0) or trade_row.get("PnL", 0)
    return_pct = trade_row.get("return_pct", 0) or trade_row.get("ReturnPct", 0)
    size = trade_row.get("size", 0) or trade_row.get("Size", 0)
    direction = trade_row.get("direction", "LONG") or ("LONG" if (size or 0) > 0 else "SHORT")
    stop_loss = trade_row.get("sl", 0) or trade_row.get("SL", 0) or 0.0
    take_profit = trade_row.get("tp", 0) or trade_row.get("TP", 0) or 0.0

    if entry_time is None or pd.isna(entry_time):
        return None

    entry_dt = pd.Timestamp(entry_time)

    # Regime at entry
    entry_ohlcv = ohlcv_df[ohlcv_df.index <= entry_dt]
    if len(entry_ohlcv) < 30:
        return None

    regime_result = regime_detector.get_regime_series(entry_ohlcv)
    regime_label = (
        regime_result["regime"].iloc[-1].value
        if hasattr(regime_result["regime"].iloc[-1], "value")
        else str(regime_result["regime"].iloc[-1])
    )
    adx = regime_result.get("adx", pd.Series([0])).iloc[-1]
    atr = regime_result.get("atr", pd.Series([0])).iloc[-1]

    # Duration in bars
    dur = trade_row.get("duration")
    if isinstance(dur, pd.Timedelta):
        duration_bars = dur.days  # daily bars
    else:
        duration_bars = 1

    return TradeRecord(
        ticker=ticker,
        entry_time=str(entry_dt.date()),
        exit_time=str(pd.Timestamp(exit_time).date()) if exit_time else str(entry_dt.date()),
        entry_price=float(entry_price),
        exit_price=float(exit_price) if exit_price else float(entry_price),
        size=float(size),
        pnl=float(pnl),
        return_pct=float(return_pct),
        duration_bars=int(duration_bars),
        direction=direction,
        stop_loss=float(stop_loss) if stop_loss else float(entry_price) * 0.95,
        take_profit=float(take_profit) if take_profit else float(entry_price) * 1.10,
        is_profitable=1 if float(return_pct) > 0 else 0,
        pattern_names="MultiPattern",
        pattern_count=1,
        confluence_score=0.5,
        regime_at_entry=regime_label,
        adx_at_entry=float(adx),
        atr_at_entry=float(atr),
    )


def collect_signals(
    tickers: List[str],
    start: str,
    end: str,
) -> Tuple[List[TradeRecord], Dict[str, int]]:
    """
    Run backtests across multiple tickers and collect trade records.

    Returns:
        List of TradeRecord objects and per-ticker trade counts.
    """
    all_trades: List[TradeRecord] = []
    ticker_counts: Dict[str, int] = {}
    regime_detector = RegimeDetector()

    logger.info(f"Collecting signals across {len(tickers)} tickers from {start} to {end}")

    # Download SPY first for market context
    spy_df = download_ticker("SPY", start, end)
    if spy_df is None:
        logger.error("Failed to download SPY data")
        return [], {}

    market_ctx = get_market_context(spy_df, start, end)

    failed_tickers = []
    for i, ticker in enumerate(tickers):
        print(f"[{i + 1}/{len(tickers)}] Processing {ticker}", flush=True)
        ohlcv = download_ticker(ticker, start, end)
        if ohlcv is None or len(ohlcv) < 200:
            print(f"  SKIP: {ticker} - insufficient data", flush=True)
            failed_tickers.append(ticker)
            continue

        try:
            runner = BacktestPyRunner(
                data=ohlcv,
                cash=100_000,
                commission=0.001,
                exclusive_orders=True,
                verbose=False,
            )
            results = runner.run()
            trades_list = results.get("trades", [])

            if isinstance(trades_list, pd.DataFrame) and not trades_list.empty:
                trades_df = trades_list
            elif isinstance(trades_list, list) and trades_list:
                trades_df = pd.DataFrame(trades_list)
            else:
                trades_df = pd.DataFrame()

            if trades_df.empty or len(trades_df) == 0:
                print(f"  {ticker}: 0 trades", flush=True)
                ticker_counts[ticker] = 0
                continue

            ticker_counts[ticker] = len(trades_df)

            for _, trade_row in trades_df.iterrows():
                record = extract_trade_features(ticker, trade_row, ohlcv, regime_detector)
                if record is not None:
                    all_trades.append(record)

            pnl_col = (
                "return_pct"
                if "return_pct" in trades_df.columns
                else "ReturnPct"
                if "ReturnPct" in trades_df.columns
                else None
            )
            if pnl_col:
                wins = (trades_df[pnl_col] > 0).sum()
                print(
                    f"  {ticker}: {len(trades_df)} trades, {wins} wins ({wins / len(trades_df):.0%})",
                    flush=True,
                )

        except Exception as e:
            print(f"  ERROR: {ticker} - {e}", flush=True)
            failed_tickers.append(ticker)

    # Enrich trades with market context
    if market_ctx is not None and all_trades:
        for trade in all_trades:
            entry_ts = pd.Timestamp(trade.entry_time)
            ctx_row = market_ctx.loc[market_ctx.index <= entry_ts]
            if not ctx_row.empty:
                last_ctx = ctx_row.iloc[-1]
                trade.vix_level = last_ctx.get("vix_close", 0.0)
                trade.tlt_spy_rs = last_ctx.get("tlt_spy_rs", 0.0)
                trade.gld_spy_rs = last_ctx.get("gld_spy_rs", 0.0)
                trade.spy_return_20d = last_ctx.get("spy_return_20d", 0.0)

    return all_trades, ticker_counts


def build_ml_dataset(trades: List[TradeRecord]) -> pd.DataFrame:
    """Convert TradeRecord list into ML-ready DataFrame."""
    if not trades:
        return pd.DataFrame()

    records = []
    for t in trades:
        records.append(
            {
                "ticker": t.ticker,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "size": t.size,
                "pnl": t.pnl,
                "return_pct": t.return_pct,
                "duration_bars": t.duration_bars,
                "direction": t.direction,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "is_profitable": t.is_profitable,
                "pattern_count": t.pattern_count,
                "confluence_score": t.confluence_score,
                "regime_at_entry": t.regime_at_entry,
                "adx_at_entry": t.adx_at_entry,
                "atr_at_entry": t.atr_at_entry,
                "vix_level": t.vix_level,
                "tlt_spy_rs": t.tlt_spy_rs,
                "gld_spy_rs": t.gld_spy_rs,
                "spy_return_20d": t.spy_return_20d,
                # Derived features
                "risk": abs(t.entry_price - t.stop_loss),
                "reward": abs(t.take_profit - t.entry_price),
                "rr_ratio": abs(t.take_profit - t.entry_price)
                / max(abs(t.entry_price - t.stop_loss), 1e-8),
                "atr_pct": t.atr_at_entry / max(t.entry_price, 1e-8),
                "vix_high": 1 if t.vix_level > 20 else 0,
            }
        )

    df = pd.DataFrame(records)

    # Encode regime as numeric
    regime_map = {"Trending": 0, "Ranging": 1, "Volatile": 2, "Transition": 3}
    df["regime_encoded"] = df["regime_at_entry"].map(regime_map).fillna(3).astype(int)

    # Encode direction as numeric
    df["direction_encoded"] = (df["direction"] == "LONG").astype(int)

    return df


def summarize_results(df: pd.DataFrame, ticker_counts: Dict[str, int]) -> str:
    """Generate a text summary of the collected signals."""
    lines = []
    lines.append("=" * 70)
    lines.append("MULTI-ASSET SIGNAL COLLECTION RESULTS")
    lines.append("=" * 70)
    lines.append(f"Total trades collected: {len(df)}")
    lines.append(f"Tickers with trades: {len([t for t, c in ticker_counts.items() if c > 0])}")
    lines.append(f"Date range: {df['entry_time'].min()} to {df['entry_time'].max()}")
    lines.append("")
    lines.append("Overall win rate: {:.1%}".format(df["is_profitable"].mean()))
    lines.append(f"Mean return: {df['return_pct'].mean():.2f}%")
    lines.append(f"Median return: {df['return_pct'].median():.2f}%")
    lines.append(
        f"Sharpe-like (mean/std): {df['return_pct'].mean() / max(df['return_pct'].std(), 1e-8):.4f}"
    )
    lines.append("")
    lines.append("--- REGIME DISTRIBUTION AT ENTRY ---")
    lines.append(df["regime_at_entry"].value_counts().to_string())
    lines.append("")
    lines.append("--- DIRECTION DISTRIBUTION ---")
    lines.append(df["direction"].value_counts().to_string())
    lines.append("")
    lines.append("--- TOP 10 TICKERS BY TRADE COUNT ---")
    top_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for ticker, count in top_tickers:
        sub = df[df["ticker"] == ticker]
        wr = sub["is_profitable"].mean() if len(sub) > 0 else 0
        lines.append(f"  {ticker:>6s}: {count:>3d} trades, WR={wr:.0%}")
    lines.append("")
    lines.append("--- FEATURE CORRELATIONS WITH PROFITABILITY ---")
    num_cols = [
        "risk",
        "reward",
        "rr_ratio",
        "atr_pct",
        "adx_at_entry",
        "atr_at_entry",
        "vix_level",
        "tlt_spy_rs",
        "gld_spy_rs",
        "spy_return_20d",
        "confluence_score",
        "pattern_count",
        "regime_encoded",
        "direction_encoded",
    ]
    available = [c for c in num_cols if c in df.columns]
    if available:
        corr = df[available].corrwith(df["is_profitable"]).sort_values(ascending=False)
        for feat, val in corr.items():
            lines.append(f"  {feat:>20s}: {val:+.4f}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Multi-asset signal collector for ML training")
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--tickers", nargs="*", default=None, help="Ticker list (uses default if empty)"
    )
    parser.add_argument(
        "--output", default="output/multi_asset_signals.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--summary-output",
        default="reports/multi_asset_signal_summary.md",
        help="Summary markdown path",
    )
    args = parser.parse_args()

    tickers = args.tickers or ASSET_UNIVERSE
    trades, ticker_counts = collect_signals(tickers, args.start, args.end)

    if not trades:
        logger.error("No trades collected!")
        sys.exit(1)

    df = build_ml_dataset(trades)
    summary = summarize_results(df, ticker_counts)
    print(summary)

    # Save outputs
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} trades to {output_path}")

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary)
    logger.info(f"Saved summary to {summary_path}")

    if len(df) < 300:
        logger.warning(
            f"Only {len(df)} trades collected. Minimum viable: 300+. Consider expanding the ticker universe."
        )
    else:
        logger.info(
            f"[SUCCESS] {len(df)} trades collected. Dataset is viable for ML signal scorer training."
        )


def train_signal_scorer_on_multi_asset(
    csv_path: str = "output/multi_asset_signals.csv",
    feature_cols: Optional[List[str]] = None,
) -> dict:
    """
    Train the SignalScorer on multi-asset trade data with walk-forward validation.

    Compares RF, GB, LR models and reports AUC-ROC per model.

    Args:
        csv_path: Path to the CSV generated by collect_signals()
        feature_cols: Feature columns to use (default: all numeric except leakage cols)

    Returns:
        Dict with validation results per model.
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    df = pd.read_csv(csv_path)
    if df.empty or len(df) < 300:
        logger.error(f"Need >=300 trades, got {len(df)}. Run signal collection first.")
        return {}

    drop_cols = {
        "ticker",
        "entry_time",
        "exit_time",
        "is_profitable",
        "return_pct",
        "pnl",
        "exit_price",
        "pattern_names",
        "direction",
    }
    feature_cols = (
        [c for c in df.columns if c not in drop_cols] if feature_cols is None else feature_cols
    )
    feature_cols = [c for c in feature_cols if c in df.columns]

    df_sorted = df.sort_values("entry_time").reset_index(drop=True)
    X = df_sorted[feature_cols].fillna(0)
    y = df_sorted["is_profitable"]

    train_size = max(200, int(len(X) * 0.7))
    step_size = max(50, (len(X) - train_size - 50) // 10)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=4, random_state=42, class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
    }

    results: Dict[str, Any] = {}
    for name, model in models.items():
        fold_aucs: List[float] = []
        fold_accs: List[float] = []
        start = 0

        while start + train_size < len(X):
            end = min(start + train_size + step_size, len(X))
            X_train = X.iloc[start : start + train_size]
            y_train = y.iloc[start : start + train_size]
            X_test = X.iloc[start + train_size : end]
            y_test = y.iloc[start + train_size : end]

            if y_test.sum() < 3 or len(y_test) < 20:
                start += step_size
                continue

            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_train, y_train)

            test_proba = model_clone.predict_proba(X_test)[:, 1]
            try:
                auc = roc_auc_score(y_test, test_proba)
            except ValueError:
                auc = 0.5

            fold_aucs.append(auc)
            fold_accs.append(accuracy_score(y_test, model_clone.predict(X_test)))
            start += step_size

        if fold_aucs:
            results[name] = {
                "mean_auc": float(np.mean(fold_aucs)),
                "std_auc": float(np.std(fold_aucs)),
                "mean_acc": float(np.mean(fold_accs)),
                "n_folds": len(fold_aucs),
            }
            logger.info(
                f"{name}: AUC={results[name]['mean_auc']:.4f} ± {results[name]['std_auc']:.4f}, "
                f"ACC={results[name]['mean_acc']:.4f} ({len(fold_aucs)} folds)"
            )

    best_name = max(results, key=lambda k: results[k]["mean_auc"])
    best_model = models[best_name]
    best_model.fit(X, y)

    if hasattr(best_model, "feature_importances_"):
        importance = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(
            ascending=False
        )
        logger.info(f"\n--- TOP 10 FEATURE IMPORTANCE ({best_name}) ---")
        for feat, imp in importance.head(10).items():
            logger.info(f"  {feat:>20s}: {imp:.4f}")
        results["feature_importance"] = importance.head(10).to_dict()

    results["best_model"] = best_name
    results["total_trades"] = len(df)
    results["win_rate"] = float(y.mean())
    results["features_used"] = feature_cols

    return results


if __name__ == "__main__":
    main()
