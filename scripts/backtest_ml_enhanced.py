"""
ML-Enhanced Backtest Comparison

Compares ML-enhanced pattern detection vs baseline strategy.
Reports uplift in Sharpe, win rate, drawdown, and analyzes which patterns benefit most.

Usage:
    uv run scripts/backtest_ml_enhanced.py --symbol SPY --start 2015-01-01 --end 2024-12-31
    uv run scripts/backtest_ml_enhanced.py --pattern "VWAP Bounce"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.indicators.regime_detector import RegimeDetector
from src.ml.feature_engineering import FeatureExtractor
from src.ml.pattern_classifier import PatternClassifier
from src.ml.signal_scorer import SignalScorer
from src.strategies.vwap_bounce import VWAPBounceStrategy
from src.strategies.ema_ribbon import EMARibbonStrategy
from src.strategies.sma_crossover import SMACrossoverStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("reports/ml_backtest")


@dataclass
class BacktestMetrics:
    """Backtest performance metrics."""

    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    avg_holding_period: float
    exposure: float


@dataclass
class PatternAnalysis:
    """Analysis of pattern performance with ML enhancement."""

    pattern_name: str
    baseline_trades: int
    ml_filtered_trades: int
    baseline_win_rate: float
    ml_win_rate: float
    baseline_avg_return: float
    ml_avg_return: float
    win_rate_improvement: float
    return_improvement: float


@dataclass
class MLBacktestComparison:
    """Results from ML-enhanced vs baseline backtest comparison."""

    baseline_metrics: BacktestMetrics
    ml_metrics: BacktestMetrics
    pattern_analyses: List[PatternAnalysis]
    uplift_metrics: Dict[str, float]
    config: Dict[str, Any]


def load_data(
    symbol: str = "SPY",
    start: str = "2015-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """Load OHLCV data."""
    import yfinance as yf

    logger.info(f"Loading {symbol} data from {start} to {end}")
    df = yf.download(symbol, start=start, end=end, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    logger.info(f"Loaded {len(df)} bars")

    return df


def generate_signal_features(
    df: pd.DataFrame,
    strategy_class: type,
    feature_extractor: FeatureExtractor,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate signal features for a strategy.

    Returns:
        Tuple of (signal timestamps, feature matrix)
    """
    signals = generate_strategy_signals(df, strategy_class)

    features = feature_extractor.extract_all_features(df)

    signal_features = features.reindex(signals.index)

    return signals, signal_features


def generate_strategy_signals(
    df: pd.DataFrame,
    strategy_class: type,
) -> pd.DataFrame:
    """
    Generate signal timestamps from a strategy.

    This is a simplified signal generator that uses basic logic
    from the strategy without full backtesting.
    """
    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["direction"] = "none"

    if strategy_class == VWAPBounceStrategy:
        vwap = (df["Close"] * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum()
        price_below_vwap = df["Close"] < vwap
        price_crosses_above = (df["Close"] > vwap) & (df["Close"].shift(1) <= vwap.shift(1))

        signals.loc[price_crosses_above, "signal"] = 1
        signals.loc[price_crosses_above, "direction"] = "long"

    elif strategy_class == EMARibbonStrategy:
        ema_9 = df["Close"].ewm(span=9).mean()
        ema_21 = df["Close"].ewm(span=21).mean()
        ema_55 = df["Close"].ewm(span=55).mean()

        bullish = (ema_9 > ema_21) & (ema_21 > ema_55)
        bearish = (ema_9 < ema_21) & (ema_21 < ema_55)

        signals.loc[bullish, "signal"] = 1
        signals.loc[bullish, "direction"] = "long"
        signals.loc[bearish, "signal"] = -1
        signals.loc[bearish, "direction"] = "short"

    elif strategy_class == SMACrossoverStrategy:
        sma_50 = df["Close"].rolling(50).mean()
        sma_200 = df["Close"].rolling(200).mean()

        golden_cross = (sma_50 > sma_200) & (sma_50.shift(1) <= sma_200.shift(1))
        death_cross = (sma_50 < sma_200) & (sma_50.shift(1) >= sma_200.shift(1))

        signals.loc[golden_cross, "signal"] = 1
        signals.loc[golden_cross, "direction"] = "long"
        signals.loc[death_cross, "signal"] = -1
        signals.loc[death_cross, "direction"] = "short"

    return signals[signals["signal"] != 0]


def train_classifier_on_signals(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    feature_extractor: FeatureExtractor,
    horizon: int = 5,
) -> Tuple[PatternClassifier, pd.DataFrame]:
    """
    Train pattern classifier on signal outcomes.

    Returns:
        Tuple of (trained classifier, signal features)
    """
    features = feature_extractor.extract_all_features(df)

    signal_features = features.reindex(signals.index).dropna()

    future_returns = df["Close"].shift(-horizon) / df["Close"] - 1
    signal_outcomes = future_returns.reindex(signal_features.index)
    y = (signal_outcomes > 0).astype(int)

    X = signal_features.dropna()
    y = y.reindex(X.index).dropna()

    if len(X) < 100:
        raise ValueError(f"Insufficient signal samples: {len(X)}")

    classifier = PatternClassifier(
        model_type="lightgbm",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
    )

    result = classifier.train(X, y)

    logger.info(
        f"Classifier trained: Test AUC={result.test_auc:.4f}, Accuracy={result.test_accuracy:.4f}"
    )

    return classifier, signal_features


def run_baseline_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    capital: float = 1_000_000,
    commission: float = 0.001,
) -> BacktestMetrics:
    """
    Run simplified backtest on baseline signals.

    This is a vectorized backtest that doesn't use backtesting.py
    for speed.
    """
    trades = []
    position = None

    capital_alloc = capital
    trades_log = []

    for idx in signals.index:
        if idx not in df.index:
            continue

        row = df.loc[idx]
        signal_row = signals.loc[idx]

        if signal_row["direction"] == "long":
            entry_price = row["Close"]
            stop_loss = entry_price * 0.95
            take_profit = entry_price * 1.10

            future_prices = df.loc[idx:, "Close"]
            exit_prices = future_prices.shift(-1)
            exit_idx = None

            for i, (date, price) in enumerate(future_prices.items()):
                if i == 0:
                    continue
                if i > 10:
                    exit_price = price
                    exit_idx = date
                    break
                if price <= stop_loss:
                    exit_price = stop_loss
                    exit_idx = date
                    break
                if price >= take_profit:
                    exit_price = take_profit
                    exit_idx = date
                    break

            if exit_idx is None:
                exit_idx = future_prices.index[-1]
                exit_price = future_prices.iloc[-1]

            if exit_idx is not None:
                trade_return = (exit_price - entry_price) / entry_price
                trade_pnl = trade_return * capital_alloc * (1 - commission * 2)

                trades_log.append(
                    {
                        "entry_date": idx,
                        "exit_date": exit_idx,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return": trade_return,
                        "pnl": trade_pnl,
                        "direction": "long",
                    }
                )

    if not trades_log:
        return BacktestMetrics(
            total_return=0,
            annualized_return=0,
            sharpe_ratio=0,
            max_drawdown=0,
            win_rate=0,
            profit_factor=0,
            total_trades=0,
            avg_trade_return=0,
            best_trade=0,
            worst_trade=0,
            avg_holding_period=0,
            exposure=0,
        )

    trades_df = pd.DataFrame(trades_log)
    total_trades = len(trades_df)
    win_trades = trades_df[trades_df["return"] > 0]
    loss_trades = trades_df[trades_df["return"] <= 0]

    win_rate = len(win_trades) / total_trades if total_trades > 0 else 0
    avg_return = trades_df["return"].mean()

    gross_profit = win_trades["pnl"].sum() if len(win_trades) > 0 else 0
    gross_loss = abs(loss_trades["pnl"].sum()) if len(loss_trades) > 0 else 1e-10
    profit_factor = gross_profit / (gross_loss + 1e-10)

    cumulative_pnl = trades_df["pnl"].cumsum()
    peak = cumulative_pnl.cummax()
    drawdown = (cumulative_pnl - peak) / (peak + capital + 1e-10) * 100
    max_drawdown = abs(drawdown.min())

    total_return = (
        (cumulative_pnl.iloc[-1] + capital - capital) / capital * 100
        if len(cumulative_pnl) > 0
        else 0
    )

    daily_returns = trades_df.groupby(trades_df["entry_date"].dt.date)["return"].sum()
    sharpe_ratio = (
        (daily_returns.mean() / (daily_returns.std() + 1e-10)) * np.sqrt(252)
        if len(daily_returns) > 10
        else 0
    )

    holding_periods = (trades_df["exit_date"] - trades_df["entry_date"]).dt.days
    avg_holding_period = holding_periods.mean()

    exposure = len(trades_df) / len(df) * 100

    return BacktestMetrics(
        total_return=float(total_return),
        annualized_return=float(total_return / (len(df) / 252)),
        sharpe_ratio=float(sharpe_ratio),
        max_drawdown=float(max_drawdown),
        win_rate=float(win_rate * 100),
        profit_factor=float(profit_factor),
        total_trades=int(total_trades),
        avg_trade_return=float(avg_return * 100),
        best_trade=float(trades_df["return"].max() * 100),
        worst_trade=float(trades_df["return"].min() * 100),
        avg_holding_period=float(avg_holding_period),
        exposure=float(exposure),
    )


def run_ml_filtered_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    signal_features: pd.DataFrame,
    classifier: PatternClassifier,
    probability_threshold: float = 0.55,
    capital: float = 1_000_000,
    commission: float = 0.001,
) -> BacktestMetrics:
    """
    Run backtest with ML-filtered signals.

    Only trades with ML probability above threshold are executed.
    """
    predictions = classifier.predict(signal_features, threshold=probability_threshold)

    filtered_signals = signals[predictions["is_recommended"]].copy()

    logger.info(f"ML filtered: {len(filtered_signals)} trades (from {len(signals)} original)")

    if len(filtered_signals) == 0:
        return BacktestMetrics(
            total_return=0,
            annualized_return=0,
            sharpe_ratio=0,
            max_drawdown=0,
            win_rate=0,
            profit_factor=0,
            total_trades=0,
            avg_trade_return=0,
            best_trade=0,
            worst_trade=0,
            avg_holding_period=0,
            exposure=0,
        )

    return run_baseline_backtest(df, filtered_signals, capital, commission)


def analyze_pattern_performance(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    signal_features: pd.DataFrame,
    classifier: PatternClassifier,
    probability_threshold: float = 0.55,
) -> List[PatternAnalysis]:
    """
    Analyze which patterns benefit most from ML filtering.
    """
    pattern_groups = []

    if "pattern_name" in signals.columns:
        pattern_groups = signals["pattern_name"].unique()
    else:
        pattern_groups = ["all"]

    analyses = []

    for pattern_name in pattern_groups:
        if pattern_name != "all":
            pattern_signals = signals[signals["pattern_name"] == pattern_name]
            pattern_features = signal_features.loc[pattern_signals.index]
        else:
            pattern_signals = signals
            pattern_features = signal_features

        if len(pattern_signals) < 20:
            continue

        baseline_analysis = run_baseline_backtest(df, pattern_signals)

        predictions = classifier.predict(pattern_features, threshold=probability_threshold)
        filtered_signals = pattern_signals[predictions["is_recommended"]].copy()

        if len(filtered_signals) < 5:
            continue

        ml_analysis = run_baseline_backtest(df, filtered_signals)

        analysis = PatternAnalysis(
            pattern_name=pattern_name,
            baseline_trades=baseline_analysis.total_trades,
            ml_filtered_trades=ml_analysis.total_trades,
            baseline_win_rate=baseline_analysis.win_rate,
            ml_win_rate=ml_analysis.win_rate,
            baseline_avg_return=baseline_analysis.avg_trade_return,
            ml_avg_return=ml_analysis.avg_trade_return,
            win_rate_improvement=ml_analysis.win_rate - baseline_analysis.win_rate,
            return_improvement=ml_analysis.avg_trade_return - baseline_analysis.avg_trade_return,
        )
        analyses.append(analysis)

    return analyses


def compute_uplift_metrics(
    baseline: BacktestMetrics,
    ml_enhanced: BacktestMetrics,
) -> Dict[str, float]:
    """Compute uplift metrics from ML enhancement."""
    return {
        "return_uplift": ml_enhanced.total_return - baseline.total_return,
        "sharpe_uplift": ml_enhanced.sharpe_ratio - baseline.sharpe_ratio,
        "win_rate_uplift": ml_enhanced.win_rate - baseline.win_rate,
        "drawdown_reduction": baseline.max_drawdown - ml_enhanced.max_drawdown,
        "profit_factor_uplift": ml_enhanced.profit_factor - baseline.profit_factor,
        "trade_count_change": ml_enhanced.total_trades - baseline.total_trades,
        "avg_return_improvement": ml_enhanced.avg_trade_return - baseline.avg_trade_return,
    }


def print_comparison_table(
    baseline: BacktestMetrics,
    ml_enhanced: BacktestMetrics,
    uplift: Dict[str, float],
) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print(f"{'Metric':<25} {'Baseline':<18} {'ML-Enhanced':<18} {'Uplift':<18}")
    print("=" * 100)

    metrics = [
        ("Total Return [%]", "total_return", "%"),
        ("Annualized Return [%]", "annualized_return", "%"),
        ("Sharpe Ratio", "sharpe_ratio", ""),
        ("Max Drawdown [%]", "max_drawdown", "%"),
        ("Win Rate [%]", "win_rate", "%"),
        ("Profit Factor", "profit_factor", ""),
        ("Total Trades", "total_trades", ""),
        ("Avg Trade Return [%]", "avg_trade_return", "%"),
        ("Best Trade [%]", "best_trade", "%"),
        ("Worst Trade [%]", "worst_trade", "%"),
        ("Avg Holding Period [days]", "avg_holding_period", ""),
        ("Exposure [%]", "exposure", "%"),
    ]

    for label, key, suffix in metrics:
        baseline_val = getattr(baseline, key)
        ml_val = getattr(ml_enhanced, key)
        uplift_val = uplift.get(
            f"{key.replace('_return', '').replace('_ratio', '')}_uplift", ml_val - baseline_val
        )

        if key in ["total_trades", "avg_holding_period"]:
            print(f"{label:<25} {baseline_val:>17.0f}   {ml_val:>17.0f}   ", end="")
        elif key == "max_drawdown":
            print(f"{label:<25} {baseline_val:>17.1f}%   {ml_val:>17.1f}%   ", end="")
        else:
            print(f"{label:<25} {baseline_val:>17.2f}   {ml_val:>17.2f}   ", end="")

        if key == "max_drawdown":
            if uplift_val < 0:
                print(f"v{abs(uplift_val):.1f}% (better)", end="")
            else:
                print(f"^{uplift_val:.1f}% (worse)", end="")
        elif uplift_val > 0:
            print(f"^{uplift_val:+.2f}{suffix} (better)", end="")
        elif uplift_val < 0:
            print(f"v{uplift_val:.2f}{suffix} (worse)", end="")
        else:
            print(f"= (same)", end="")
        print()

    print("=" * 100)


def run_ml_backtest_comparison(
    df: pd.DataFrame,
    strategy_class: type = None,
    symbol: str = "SPY",
) -> MLBacktestComparison:
    """
    Run full ML-enhanced vs baseline backtest comparison.
    """
    if strategy_class is None:
        strategy_class = VWAPBounceStrategy

    logger.info(f"Running ML backtest comparison for {strategy_class.__name__}")

    feature_extractor = FeatureExtractor()

    signals, signal_features = generate_signal_features(df, strategy_class, feature_extractor)

    if len(signals) < 100:
        raise ValueError(f"Insufficient signals for ML training: {len(signals)}")

    classifier, _ = train_classifier_on_signals(df, signals, feature_extractor, horizon=5)

    baseline_metrics = run_baseline_backtest(df, signals)

    ml_metrics = run_ml_filtered_backtest(
        df, signals, signal_features, classifier, probability_threshold=0.55
    )

    pattern_analyses = analyze_pattern_performance(
        df, signals, signal_features, classifier, probability_threshold=0.55
    )

    uplift_metrics = compute_uplift_metrics(baseline_metrics, ml_metrics)

    config = {
        "symbol": symbol,
        "strategy": strategy_class.__name__,
        "start_date": str(df.index[0].date()),
        "end_date": str(df.index[-1].date()),
        "n_bars": len(df),
        "n_signals": len(signals),
        "model_type": classifier.model_type,
        "probability_threshold": 0.55,
    }

    return MLBacktestComparison(
        baseline_metrics=baseline_metrics,
        ml_metrics=ml_metrics,
        pattern_analyses=pattern_analyses,
        uplift_metrics=uplift_metrics,
        config=config,
    )


def save_comparison_results(
    comparison: MLBacktestComparison,
    run_id: str,
) -> None:
    """Save comparison results to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "baseline_metrics": {
            "total_return": comparison.baseline_metrics.total_return,
            "annualized_return": comparison.baseline_metrics.annualized_return,
            "sharpe_ratio": comparison.baseline_metrics.sharpe_ratio,
            "max_drawdown": comparison.baseline_metrics.max_drawdown,
            "win_rate": comparison.baseline_metrics.win_rate,
            "profit_factor": comparison.baseline_metrics.profit_factor,
            "total_trades": comparison.baseline_metrics.total_trades,
            "avg_trade_return": comparison.baseline_metrics.avg_trade_return,
        },
        "ml_metrics": {
            "total_return": comparison.ml_metrics.total_return,
            "annualized_return": comparison.ml_metrics.annualized_return,
            "sharpe_ratio": comparison.ml_metrics.sharpe_ratio,
            "max_drawdown": comparison.ml_metrics.max_drawdown,
            "win_rate": comparison.ml_metrics.win_rate,
            "profit_factor": comparison.ml_metrics.profit_factor,
            "total_trades": comparison.ml_metrics.total_trades,
            "avg_trade_return": comparison.ml_metrics.avg_trade_return,
        },
        "uplift_metrics": comparison.uplift_metrics,
        "pattern_analyses": [
            {
                "pattern": p.pattern_name,
                "baseline_trades": p.baseline_trades,
                "ml_filtered_trades": p.ml_filtered_trades,
                "win_rate_improvement": p.win_rate_improvement,
                "return_improvement": p.return_improvement,
            }
            for p in comparison.pattern_analyses
        ],
        "config": comparison.config,
    }

    results_path = OUTPUT_DIR / f"ml_backtest_comparison_{run_id}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {results_path}")


def main():
    """Main backtest comparison pipeline."""
    parser = argparse.ArgumentParser(description="ML-Enhanced Backtest Comparison")
    parser.add_argument("--symbol", default="SPY", help="Symbol to backtest")
    parser.add_argument("--start", default="2015-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument(
        "--strategy",
        default="vwap",
        choices=["vwap", "ema", "sma"],
        help="Strategy to test",
    )

    args = parser.parse_args()

    run_id = f"{datetime.now():%Y%m%d_%H%M%S}"
    logger.info(f"Run ID: {run_id}")

    df = load_data(args.symbol, args.start, args.end)

    strategy_map = {
        "vwap": VWAPBounceStrategy,
        "ema": EMARibbonStrategy,
        "sma": SMACrossoverStrategy,
    }
    strategy_class = strategy_map[args.strategy]

    logger.info("=" * 60)
    logger.info("ML-Enhanced Backtest Comparison")
    logger.info("=" * 60)

    comparison = run_ml_backtest_comparison(df, strategy_class, args.symbol)

    logger.info("\n" + "=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)

    print_comparison_table(
        comparison.baseline_metrics,
        comparison.ml_metrics,
        comparison.uplift_metrics,
    )

    if comparison.pattern_analyses:
        logger.info("\nPattern Performance Analysis:")
        for analysis in sorted(
            comparison.pattern_analyses,
            key=lambda x: x.win_rate_improvement,
            reverse=True,
        )[:5]:
            logger.info(
                f"  {analysis.pattern_name}: "
                f"Win rate {analysis.baseline_win_rate:.1f}% -> {analysis.ml_win_rate:.1f}% "
                f"(+{analysis.win_rate_improvement:.1f}%)"
            )

    save_comparison_results(comparison, run_id)

    logger.info("\n" + "=" * 60)
    logger.info("Backtest Comparison Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
