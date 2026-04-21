"""
ML Validation Script - Execute ML regime classification and signal scoring on SPY data.

This script replicates the ML validation notebook logic as a standalone script
that can be executed via `uv run scripts/ml_validation_exec.py`.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Use Agg backend for headless saving figures
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indicators.regime_detector import RegimeDetector
from src.ml.features import FeatureEngineer
from src.ml.pipeline import MLPipeline
from src.ml.regime_model import RegimeClassifier
from src.ml.signal_scorer import SignalScorer
from src.strategies.backtest_py.runner import BacktestPyRunner

warnings.filterwarnings("ignore")


def setup_dirs() -> None:
    """Create output directories."""
    Path("reports/ml_validation").mkdir(parents=True, exist_ok=True)


def save_fig(name: str) -> None:
    """Save the current matplotlib figure."""
    plt.tight_layout()
    plt.savefig(f"reports/ml_validation/{name}", dpi=140)
    plt.close()


def load_spy_data() -> pd.DataFrame:
    """Download SPY data."""
    print("=" * 60)
    print("1. LOADING SPY DATA")
    print("=" * 60)

    df = yf.download("SPY", start="2015-01-01", end="2024-12-31", auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "date"
    df = df.sort_index()

    print(f"Data loaded: {len(df)} bars, columns: {df.columns.tolist()}")
    print(df.head(3).to_string())
    return df


def generate_regime_labels(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Generate regime labels using rule-based detector."""
    print("\n" + "=" * 60)
    print("2. GENERATING REGIME LABELS (Rule-Based)")
    print("=" * 60)

    detector = RegimeDetector()
    regime_result = detector.get_regime_series(df)
    regime_labels = regime_result["regime"].apply(lambda r: r.value)

    print("Regime distribution:")
    print(regime_labels.value_counts().sort_index())
    print(f"\nRegime classes: {sorted(regime_labels.unique())}")

    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    color_map = {"Trending": "blue", "Ranging": "orange", "Volatile": "red", "Transition": "gray"}
    colors = regime_labels.map(color_map)

    axes[0].scatter(df.index, df["Close"], c=colors, s=1)
    axes[0].set_title("SPY Close Price with Regime Labels")
    axes[0].set_ylabel("Price ($)")

    axes[1].plot(df.index, regime_result["adx"], color="purple", linewidth=0.8)
    axes[1].axhline(y=25, color="red", linestyle="--", alpha=0.5)
    axes[1].axhline(y=20, color="orange", linestyle="--", alpha=0.5)
    axes[1].set_ylabel("ADX")

    axes[2].plot(df.index, regime_result["atr"], color="green", linewidth=0.8)
    axes[2].set_ylabel("ATR")
    axes[2].set_xlabel("Date")

    # ADX and ATR in df for ML
    df["adx"] = regime_result["adx"]
    df["atr"] = regime_result["atr"]

    return regime_labels, df


def generate_ml_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Generate ML features."""
    print("\n" + "=" * 60)
    print("3. GENERATING ML FEATURES")
    print("=" * 60)

    engineer = FeatureEngineer()
    features_df = engineer.generate_features(df)

    numeric_features = features_df.select_dtypes(include=[np.number])
    numeric_features = numeric_features.ffill().bfill().dropna()

    y = pd.Series(numeric_features.index.map(lambda idx: idx), index=numeric_features.index)
    y_regime = pd.Series(numeric_features.index).reindex(numeric_features.index)

    print(f"Generated {len(numeric_features.columns)} features, {len(numeric_features)} samples")
    print(f"Features (first 20): {list(numeric_features.columns[:20])}")

    return numeric_features


def train_regime_classifier(
    numeric_features: pd.DataFrame, regime_labels: pd.Series
) -> tuple[RegimeClassifier, pd.DataFrame, pd.Series, pd.Series, float]:
    """Train ML regime classifier."""
    print("\n" + "=" * 60)
    print("4. TRAINING ML REGIME CLASSIFIER (Random Forest)")
    print("=" * 60)

    # Align y with numeric_features
    y = regime_labels.reindex(numeric_features.index).dropna()
    X = numeric_features.reindex(y.index)

    print(f"Final dataset: X={X.shape}, y={y.shape}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")

    # Split
    test_size = 0.3
    split_idx = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print(f"Train: {len(X_test)}")

    regime_classifier = RegimeClassifier(
        model_type="random_forest",
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )

    train_results = regime_classifier.train(X_train, y_train, test_size=0.2)
    print("\nTraining Results:")
    for k, v in train_results.items():
        if isinstance(v, dict):
            continue
        print(f"  {k}: {v}")

    # Evaluate on X_test
    y_pred_test = regime_classifier.predict(X_test)
    ml_accuracy = accuracy_score(y_test, y_pred_test)

    print(f"\nML Regime Classifier Test Accuracy: {ml_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

    # Feature importance
    importance_df = regime_classifier.get_feature_importance(top_n=15)
    print("\nTop 15 Features for Regime Classification:")
    print(importance_df)

    # Save feature importance figure
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df,
        x="importance",
        y="feature",
        palette="viridis",
        hue="feature",
        legend=False,
    )
    plt.title("Top 15 Features for Regime Classification")
    save_fig("regime_feature_importance.png")

    return regime_classifier, X, y, y_test, ml_accuracy


def compare_ml_vs_rule_based(
    regime_classifier: RegimeClassifier, X: pd.DataFrame, y: pd.Series
) -> float:
    """Compare ML vs rule-based."""
    print("\n" + "=" * 60)
    print("5. ML vs Rule-Based Comparison")
    print("=" * 60)

    X_filled = X.ffill().bfill()
    ml_regime_all = regime_classifier.predict(X_filled)

    comparison_df = pd.DataFrame({"rule_based": y, "ml": ml_regime_all})
    overall_agreement = (comparison_df["rule_based"] == comparison_df["ml"]).mean()

    print(f"Overall agreement: {overall_agreement:.2%}")
    print("\nML regime distribution:")
    print(comparison_df["ml"].value_counts().sort_index())
    print("\nRule-based regime distribution:")
    print(comparison_df["rule_based"].value_counts().sort_index())

    return overall_agreement


def generate_trade_signals(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, float]:
    """Generate trade signals for signal scorer."""
    print("\n" + "=" * 60)
    print("6. GENERATING TRADE SIGNALS (ConnorsRSI)")
    print("=" * 60)

    # Remove the ConnorsRSI import since it requires backtesting.py internal setup
    # Instead, use MultiPatternStrategy which works natively with BacktestPyRunner
    runner = BacktestPyRunner(data=df, cash=100000, exclusive_orders=True)
    runner.run(
        min_confidence=0.60,
        min_confluence_count=2,
        risk_per_trade=0.02,
        max_open_positions=5,
        use_regime_filter=False,
    )

    trades_df = runner.get_trades()
    num_trades = len(trades_df)
    print(f"Generated {num_trades} trades")

    if num_trades == 0 or trades_df.empty:
        print("No trades generated - skipping signal scorer")
        return pd.DataFrame(), pd.Series(dtype=int), 0

    trades_df["is_profitable"] = (trades_df["pnl"] > 0).astype(int)
    print(
        f"Profitable trades: {trades_df['is_profitable'].sum()}/{num_trades} ({trades_df['is_profitable'].mean():.1%})"
    )
    print(f"\nTrade P&L summary:")
    print(trades_df["pnl"].describe())

    signal_feat_cols = ["entry_price", "exit_price", "size", "return_pct", "duration"]
    available_cols = [c for c in signal_feat_cols if c in trades_df.columns]

    # Exclude pnl/label columns from features
    available_cols = [c for c in available_cols if c not in ("pnl", "is_profitable")]

    if len(available_cols) < 2:
        print(f"Insufficient columns for signal features. Available: {trades_df.columns.tolist()}")
        return pd.DataFrame(), pd.Series(dtype=int), num_trades

    signal_X = trades_df[available_cols].copy()
    signal_y = (trades_df["pnl"] > 0).astype(int)

    # Convert duration to numeric if present
    if "duration" in signal_X.columns:
        signal_X["duration"] = pd.to_timedelta(signal_X["duration"]).dt.total_seconds()

    signal_X = signal_X.ffill().bfill().dropna()
    signal_y = signal_y.loc[signal_X.index]

    return signal_X, signal_y, num_trades


def train_signal_scorer(
    signal_X: pd.DataFrame, signal_y: pd.Series, num_trades: int
) -> tuple[SignalScorer, float, float]:
    """Train signal scorer."""
    print("\n" + "=" * 60)
    print("7. TRAINING SIGNAL SCORER (Gradient Boosting)")
    print("=" * 60)

    if len(signal_X) == 0:
        print("No signal features - skipping signal scorer")
        return SignalScorer(), 0.0, 0.0

    scorer = SignalScorer(
        model_type="gradient_boosting",
        n_estimators=100,
        max_depth=4,
        random_state=42,
    )

    signal_results = scorer.train(signal_X, signal_y)
    print("\nSignal Scorer Training Results:")
    for k, v in signal_results.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")

    if hasattr(scorer.model, "feature_importances_"):
        importance_df = scorer.get_top_features_by_importance(top_n=10)
        print("\nTop 10 Features for Signal Scoring:")
        print(importance_df)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=importance_df,
            x="importance",
            y="feature",
            palette="viridis",
            hue="feature",
            legend=False,
        )
        plt.title("Top 10 Features for Signal Quality")
        save_fig("signal_feature_importance.png")

    test_accuracy = signal_results.get("test_accuracy", 0.0)
    test_auc = signal_results.get("test_auc_roc", 0.0)

    return scorer, test_accuracy, test_auc


def walk_forward_validation(df: pd.DataFrame, regime_labels: pd.Series) -> dict:
    """Run walk-forward validation."""
    print("\n" + "=" * 60)
    print("8. WALK-FORWARD VALIDATION")
    print("=" * 60)

    pipeline = MLPipeline(
        regime_model_type="random_forest",
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )

    wf_results = pipeline.walk_forward_validation(
        df=df,
        regime_labels=regime_labels,
    )

    if "regime" in wf_results:
        print("\nWalk-Forward Regime Results:")
        regime_wf = wf_results["regime"]
        for k, v in regime_wf.items():
            if isinstance(v, (int, float)):
                continue
            elif isinstance(v, list):
                print(f"  {k} ({len(v)} folds): {np.mean(v):.4f} +/- {np.std(v):.4f}")

    return wf_results


def run_pipeline_end_to_end(
    df: pd.DataFrame, regime_labels: pd.Series, signal_X: pd.DataFrame, signal_y: pd.Series
) -> None:
    """Run complete pipeline."""
    print("\n" + "=" * 60)
    print("9. RUNNING COMPLETE ML PIPELINE")
    print("=" * 60)

    pipeline = MLPipeline(
        regime_model_type="random_forest",
        signal_model_type="gradient_boosting",
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )

    pipeline_result = pipeline.run_pipeline(
        df=df,
        regime_labels=regime_labels,
        signal_features=signal_X if len(signal_X) > 0 else None,
        signal_labels=signal_y if len(signal_y) > 0 else None,
    )

    print(f"\nPipeline regime accuracy: {pipeline_result.pipeline_metrics['regime_accuracy']:.4f}")
    print(f"Features used: {pipeline_result.pipeline_metrics['n_features']}")


def print_summary(
    ml_accuracy: float, overall_agreement: float, signal_test_acc: float, signal_auc: float
) -> None:
    """Print validation summary."""
    print("\n" + "=" * 60)
    print("10. ML VALIDATION SUMMARY")
    print("=" * 60)

    summary = {
        "ML Regime Classifier (RF)": f"accuracy={ml_accuracy:.2%}",
        "Rule-Based vs ML Agreement": f"{overall_agreement:.2%}",
        "Signal Scorer (GB)": f"accuracy={signal_test_acc:.2%}, AUC-ROC={signal_auc:.4f}",
    }

    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    if ml_accuracy > 0.7:
        print("  ML regime detection shows promise (>70% accuracy)")
        print("  -> Integrate with ConfluenceScorer for ML-enhanced signal scoring")
        print("  -> Run full backtest with ML enhancement")
    else:
        print("  ML regime accuracy is not significantly better than random")
        print("  -> Try feature engineering improvements")
        print("  -> Consider different model types (XGBoost, neural nets)")
        print("  -> Add more input features (volume, volatility, cross-asset)")


def main() -> None:
    """Execute ML validation pipeline."""
    setup_dirs()

    # 1. Load data
    df = load_spy_data()

    # 2. Generate regime labels
    regime_labels, df = generate_regime_labels(df)

    # 3. Generate features
    numeric_features = generate_ml_features(df)

    # 4. Train regime classifier
    regime_classifier, X, y, y_test, ml_accuracy = train_regime_classifier(
        numeric_features, regime_labels
    )

    # 5. ML vs rule-based
    overall_agreement = compare_ml_vs_rule_based(regime_classifier, X, y)

    # 6. Generate trade signals
    signal_X, signal_y, num_trades = generate_trade_signals(df)

    # 7. Train signal scorer
    scorer, signal_test_acc, signal_auc = train_signal_scorer(signal_X, signal_y, num_trades)

    # 8. Walk-forward validation
    wf_results = walk_forward_validation(df, regime_labels)

    # 9. Run pipeline end-to-end
    run_pipeline_end_to_end(df, regime_labels, signal_X, signal_y)

    # 10. Summary
    print_summary(ml_accuracy, overall_agreement, signal_test_acc, signal_auc)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
