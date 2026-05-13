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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Suppress FutureWarning for fillna downcasting
pd.set_option("future.no_silent_downcasting", True)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  # noqa: E402

from src.ml.feature_engineering import FeatureExtractor  # noqa: E402
from src.ml.pattern_classifier import PatternClassifier  # noqa: E402
from src.strategies.vwap_bounce import VWAPBounceStrategy  # noqa: E402
from src.strategies.ema_ribbon import EMARibbonStrategy  # noqa: E402
from src.strategies.sma_crossover import SMACrossoverStrategy  # noqa: E402

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


def generate_strategy_signals(
    df: pd.DataFrame,
    strategy_class: type,
) -> pd.DataFrame:
    """Generate entry signal timestamps from a strategy.

    Uses state-based detection (not strict cross-point) to produce sufficient
    signals for ML filtering. The pre-trained PatternClassifier handles the
    quality filtering — not the crossover logic.
    """
    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["direction"] = "none"

    if strategy_class == VWAPBounceStrategy:
        vwap = (df["Close"] * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum()
        above = df["Close"] > vwap
        entry = above & ~above.shift(1).fillna(False)
        signals.loc[entry, "signal"] = 1
        signals.loc[entry, "direction"] = "long"

    elif strategy_class == EMARibbonStrategy:
        ema_9 = df["Close"].ewm(span=9).mean()
        ema_21 = df["Close"].ewm(span=21).mean()
        ema_55 = df["Close"].ewm(span=55).mean()

        bullish = (ema_9 > ema_21) & (ema_21 > ema_55)
        bearish = (ema_9 < ema_21) & (ema_21 < ema_55)
        entry_long = bullish & ~bullish.shift(1).fillna(False)
        entry_short = bearish & ~bearish.shift(1).fillna(False)

        signals.loc[entry_long, "signal"] = 1
        signals.loc[entry_long, "direction"] = "long"
        signals.loc[entry_short, "signal"] = -1
        signals.loc[entry_short, "direction"] = "short"

    elif strategy_class == SMACrossoverStrategy:
        sma_50 = df["Close"].rolling(50).mean()
        sma_200 = df["Close"].rolling(200).mean()

        golden = (sma_50 > sma_200) & (sma_50.shift(1) <= sma_200.shift(1))
        death = (sma_50 < sma_200) & (sma_50.shift(1) >= sma_200.shift(1))

        signals.loc[golden, "signal"] = 1
        signals.loc[golden, "direction"] = "long"
        signals.loc[death, "signal"] = -1
        signals.loc[death, "direction"] = "short"

    return signals[signals["signal"] != 0]


def load_pretrained_model(
    model_path: str = "models/pattern_classifier_v3_SPY_20260511_224704.pkl",
) -> PatternClassifier | None:
    """Load pre-trained PatternClassifier V3 model (no local training)."""
    path = Path(model_path)
    if not path.exists():
        logger.warning(f"Model not found: {path}. Searching for alternative...")
        alternatives = sorted(Path("models").glob("pattern_classifier_v3_SPY_*.pkl"))
        if alternatives:
            path = alternatives[-1]
            logger.info(f"Using: {path}")
        else:
            return None
    model = PatternClassifier(model_type="catboost")
    model.load(str(path))
    return model


def score_signals_with_model(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    model: PatternClassifier,
    feature_extractor: FeatureExtractor,
) -> pd.DataFrame:
    """Score strategy signals using pre-trained model features."""
    features = feature_extractor.extract_all_features(df)
    signal_features = features.reindex(signals.index).dropna()
    if len(signal_features) == 0:
        return pd.DataFrame()
    return model.predict(signal_features, threshold=0.0)


def _empty_metrics() -> BacktestMetrics:
    """Return zero-value metrics when no trades exist."""
    return BacktestMetrics(
        total_return=0.0,
        annualized_return=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        win_rate=0.0,
        profit_factor=0.0,
        total_trades=0,
        avg_trade_return=0.0,
        best_trade=0.0,
        worst_trade=0.0,
        avg_holding_period=0.0,
        exposure=0.0,
    )


def _prepare_df_for_backtesting(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure OHLCV columns are capitalized for backtesting.py compatibility."""
    df_bt = df.copy()
    df_bt.columns = [c.capitalize() for c in df_bt.columns]
    return df_bt


def _make_filtered_strategy(base_class: type, allowed_dates: list):
    """Create a strategy class that only enters on ML-approved dates."""
    allowed = set(pd.to_datetime(allowed_dates))

    class FilteredStrategy(base_class):
        _allowed = allowed

        def next(self):
            if self.data.index[-1] not in self._allowed:
                return
            super().next()

    FilteredStrategy.__name__ = f"Filtered{base_class.__name__}"
    return FilteredStrategy


def run_baseline_backtest(
    df: pd.DataFrame,
    strategy_class: type,
    capital: float = 1_000_000,
    commission: float = 0.001,
) -> BacktestMetrics:
    """Run backtest using backtesting.py with proper position sizing."""
    from backtesting import Backtest

    df_bt = _prepare_df_for_backtesting(df)

    bt = Backtest(df_bt, strategy_class, cash=capital, commission=commission, exclusive_orders=True)
    stats = bt.run()

    return BacktestMetrics(
        total_return=round(stats["Return [%]"], 2),
        annualized_return=round(stats["Return (Ann.) [%]"], 2),
        sharpe_ratio=round(stats["Sharpe Ratio"], 2),
        max_drawdown=round(stats["Max. Drawdown [%]"], 2),
        win_rate=round(stats["Win Rate [%]"], 2),
        profit_factor=round(stats["Profit Factor"], 2),
        total_trades=int(stats["# Trades"]),
        avg_trade_return=round(stats["Avg. Trade [%]"], 2) if stats["# Trades"] > 0 else 0.0,
        best_trade=round(stats["Best Trade [%]"], 2) if stats["# Trades"] > 0 else 0.0,
        worst_trade=round(stats["Worst Trade [%]"], 2) if stats["# Trades"] > 0 else 0.0,
        avg_holding_period=0.0,
        exposure=round(stats["Exposure Time [%]"], 2),
    )


def run_ml_filtered_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    signal_features: pd.DataFrame,
    classifier: PatternClassifier,
    probability_threshold: float = 0.45,
    capital: float = 1_000_000,
    commission: float = 0.001,
    strategy_class: type = None,
) -> BacktestMetrics:
    """Run backtest with ML-filtered signals using backtesting.py and pre-trained model."""
    from backtesting import Backtest

    if strategy_class is None:
        strategy_class = VWAPBounceStrategy

    if len(signals) == 0:
        return _empty_metrics()

    feature_extractor = FeatureExtractor()
    test_features = feature_extractor.extract_all_features(df)
    test_signal_features = test_features.reindex(signals.index).dropna()

    if len(test_signal_features) == 0:
        return _empty_metrics()

    predictions = classifier.predict(test_signal_features, threshold=probability_threshold)
    approved = predictions[predictions["is_recommended"]]
    approved_dates = approved.index.tolist()

    logger.info(f"ML filtered: {len(approved_dates)} trades (from {len(signals)} original)")

    if len(approved_dates) == 0:
        return _empty_metrics()

    FilteredStrategy = _make_filtered_strategy(strategy_class, approved_dates)

    df_bt = _prepare_df_for_backtesting(df)
    bt = Backtest(
        df_bt, FilteredStrategy, cash=capital, commission=commission, exclusive_orders=True
    )
    stats = bt.run()

    return BacktestMetrics(
        total_return=round(stats["Return [%]"], 2),
        annualized_return=round(stats["Return (Ann.) [%]"], 2),
        sharpe_ratio=round(stats["Sharpe Ratio"], 2),
        max_drawdown=round(stats["Max. Drawdown [%]"], 2),
        win_rate=round(stats["Win Rate [%]"], 2),
        profit_factor=round(stats["Profit Factor"], 2),
        total_trades=int(stats["# Trades"]),
        avg_trade_return=round(stats["Avg. Trade [%]"], 2) if stats["# Trades"] > 0 else 0.0,
        best_trade=round(stats["Best Trade [%]"], 2) if stats["# Trades"] > 0 else 0.0,
        worst_trade=round(stats["Worst Trade [%]"], 2) if stats["# Trades"] > 0 else 0.0,
        avg_holding_period=0.0,
        exposure=round(stats["Exposure Time [%]"], 2),
    )


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
            print("= (same)", end="")
        print()

    print("=" * 100)


def run_ml_backtest_comparison(
    df: pd.DataFrame,
    strategy_class: type = None,
    symbol: str = "SPY",
) -> MLBacktestComparison:
    """Run ML-enhanced vs baseline backtest using pre-trained PatternClassifier V3.

    Chronological split: train patterns on first 70% (generates features for model),
    test on last 30% with ML-filtered entries. No local model training — uses
    the existing 33-ticker PatternClassifier V3.
    """
    if strategy_class is None:
        strategy_class = VWAPBounceStrategy

    logger.info(f"Running ML backtest comparison for {strategy_class.__name__}")

    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    logger.info(
        f"Walk-forward split: train {train_df.index[0].date()}..{train_df.index[-1].date()} "
        f"({len(train_df)} bars), test {test_df.index[0].date()}..{test_df.index[-1].date()} "
        f"({len(test_df)} bars)"
    )

    model = load_pretrained_model()
    if model is None:
        logger.error("No pre-trained model found. Run training pipeline first.")
        baseline_metrics = run_baseline_backtest(test_df, strategy_class)
        return MLBacktestComparison(
            baseline_metrics=baseline_metrics,
            ml_metrics=_empty_metrics(),
            pattern_analyses=[],
            uplift_metrics={},
            config={},
        )

    test_signals = generate_strategy_signals(test_df, strategy_class)
    logger.info(f"Strategy signals (test period): {len(test_signals)}")

    baseline_metrics = run_baseline_backtest(test_df, strategy_class)

    ml_metrics = run_ml_filtered_backtest(
        test_df,
        test_signals,
        pd.DataFrame(),
        model,
        probability_threshold=0.45,
        strategy_class=strategy_class,
    )

    uplift_metrics = compute_uplift_metrics(baseline_metrics, ml_metrics)

    config = {
        "symbol": symbol,
        "strategy": strategy_class.__name__,
        "start_date": str(df.index[0].date()),
        "end_date": str(df.index[-1].date()),
        "test_start": str(test_df.index[0].date()),
        "n_bars": len(df),
        "n_train_bars": len(train_df),
        "n_test_bars": len(test_df),
        "n_test_signals": len(test_signals),
        "model": model.model_type if model else "none",
        "probability_threshold": 0.45,
    }

    return MLBacktestComparison(
        baseline_metrics=baseline_metrics,
        ml_metrics=ml_metrics,
        pattern_analyses=[],
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
