"""
ML Model Training Pipeline

Trains pattern classifier with walk-forward validation.
Generates feature importance visualizations and model persistence.

Usage:
    uv run scripts/train_ml_model.py --symbol data/raw/SPY_daily.csv
    uv run scripts/train_ml_model.py --symbol data/raw/BTC_USD_daily.csv --iterations 5
    uv run scripts/train_ml_model.py --symbol SPY --start 2015-01-01 --end 2024-12-31 --iterations -1 --sleep 60
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.feature_engineering import FeatureExtractor
from src.ml.feature_store import FeatureStore
from src.ml.features import FeatureEngineer
from src.ml.metrics import compute_ic, compute_rank_ic, ic_summary
from src.ml.pattern_classifier import PatternClassifier
from src.ml.purged_cv import PurgedKFold
from src.ml.signal_scorer import SignalScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("reports/ml_training")
MODEL_DIR = Path("models")
PLOT_DIR = Path("reports/ml_plots")


def load_pattern_data(
    symbol_or_path: str = "SPY",
    start: str = "2015-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """Load OHLCV data from file or download from Yahoo Finance.

    Args:
        symbol_or_path: Either a symbol name (e.g., "SPY") or file path (e.g., "SPY_daily.csv")
        start: Start date for Yahoo Finance download
        end: End date for Yahoo Finance download

    Returns:
        OHLCV DataFrame
    """
    import yfinance as yf

    path = Path(symbol_or_path)

    if path.exists():
        logger.info(f"Loading data from {path}")
        if path.suffix == ".csv":
            df = pd.read_csv(path, parse_dates=True, index_col=0)
        elif path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    else:
        logger.info(f"Downloading {symbol_or_path} data from Yahoo Finance")
        df = yf.download(symbol_or_path, start=start, end=end, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    logger.info(f"Loaded {len(df)} bars")

    if len(df) == 0:
        raise ValueError("No data loaded. Check file path or symbol.")

    if "Close" not in df.columns:
        raise ValueError("Data must contain 'Close' column")

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            if col == "Volume":
                df[col] = 0
            else:
                df[col] = df["Close"]

    return df


def generate_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.0,
) -> pd.Series:
    """
    Generate binary labels for pattern outcomes.

    Args:
        df: OHLCV DataFrame
        horizon: Number of bars forward to evaluate
        threshold: Return threshold for binary classification

    Returns:
        Binary labels (1 = profitable, 0 = not profitable)
    """
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1

    if threshold == 0.0:
        labels = (future_return > 0).astype(int)
    else:
        labels = (future_return > threshold).astype(int)

    return labels


def extract_features(
    df: pd.DataFrame,
    pattern_signals: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract features for ML model.

    Args:
        df: OHLCV DataFrame
        pattern_signals: Optional DataFrame with pattern detection timestamps

    Returns:
        Tuple of (feature DataFrame, feature names)
    """
    logger.info("Extracting features...")

    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract_all_features(df)

    if pattern_signals is not None:
        features = features.reindex(pattern_signals.index)

    features = features.dropna(axis=1, how="all")

    logger.info(f"Extracted {features.shape[1]} features for {features.shape[0]} samples")

    return features, list(features.columns)


def prepare_training_data(
    df: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.Series,
    min_samples: int = 1000,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare final training data.

    Args:
        df: OHLCV DataFrame
        features: Feature matrix
        labels: Binary labels

    Returns:
        Tuple of (cleaned features, cleaned labels)
    """
    valid_mask = features.notna().all(axis=1) & labels.notna()

    X = features[valid_mask]
    y = labels[valid_mask]

    if len(X) < min_samples:
        raise ValueError(f"Insufficient samples after cleaning: {len(X)} < {min_samples}")

    logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Positive samples: {y.sum()} ({y.sum() / len(y) * 100:.1f}%)")

    return X, y


def train_with_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "lightgbm",
    n_splits: int = 5,
    train_size: int = 500,
    step_size: int = 100,
) -> Dict[str, Any]:
    """
    Train model with walk-forward validation.

    Args:
        X: Feature matrix
        y: Labels
        model_type: Type of model
        n_splits: Number of splits for PurgedKFold
        train_size: Initial training window
        step_size: Step for expanding window

    Returns:
        Dict with training metrics and model
    """
    logger.info("=" * 60)
    logger.info("Walk-Forward Validation")
    logger.info("=" * 60)

    classifier = PatternClassifier(
        model_type=model_type,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
    )

    wf_results = classifier.walk_forward_validation(
        X=X,
        y=y,
        train_size=train_size,
        step_size=step_size,
    )

    if wf_results["test_auc"]:
        logger.info(f"Average test AUC: {np.mean(wf_results['test_auc']):.4f}")
        logger.info(f"Average test accuracy: {np.mean(wf_results['test_accuracy']):.4f}")
        logger.info(f"AUC std: {np.std(wf_results['test_auc']):.4f}")

    return {
        "classifier": classifier,
        "walk_forward_results": wf_results,
        "n_folds": len(wf_results.get("test_auc", [])),
    }


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "lightgbm",
    test_size: float = 0.3,
) -> Tuple[PatternClassifier, Dict[str, Any]]:
    """
    Train final model on full dataset with train/test split.

    Args:
        X: Feature matrix
        y: Labels
        model_type: Type of model
        test_size: Test set proportion

    Returns:
        Tuple of (trained classifier, training results)
    """
    logger.info("=" * 60)
    logger.info("Training Final Model")
    logger.info("=" * 60)

    split_idx = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    classifier = PatternClassifier(
        model_type=model_type,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    calibration_data = (X_test, y_test)
    result = classifier.train(
        X=X_train,
        y=y_train,
        calibration_data=calibration_data,
    )

    logger.info(f"Train AUC: {result.train_auc:.4f}")
    logger.info(f"Test AUC: {result.test_auc:.4f}")
    logger.info(f"Test Accuracy: {result.test_accuracy:.4f}")
    logger.info(f"Calibration Error: {result.calibration_error:.4f}")

    top_features = list(result.feature_importance.items())[:10]
    logger.info("\nTop 10 Features:")
    for feat, imp in top_features:
        logger.info(f"  {feat}: {imp:.4f}")

    return classifier, {
        "training_result": result,
        "X_train_size": len(X_train),
        "X_test_size": len(X_test),
    }


def plot_feature_importance(
    classifier: PatternClassifier,
    top_n: int = 20,
    save_path: Optional[Path] = None,
) -> None:
    """Plot and save feature importance."""
    import matplotlib.pyplot as plt

    importance_df = classifier.get_feature_importance(top_n=top_n)

    if importance_df.empty:
        logger.warning("No feature importance available")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(range(len(importance_df)), importance_df["importance"])
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importance")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Feature importance plot saved to {save_path}")

    plt.close()


def plot_walk_forward_results(
    wf_results: Dict[str, List[float]],
    save_path: Optional[Path] = None,
) -> None:
    """Plot walk-forward validation results."""
    import matplotlib.pyplot as plt

    if not wf_results.get("test_auc"):
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    folds = range(len(wf_results["test_auc"]))

    axes[0].plot(folds, wf_results["train_auc"], "b-o", label="Train AUC")
    axes[0].plot(folds, wf_results["test_auc"], "r-o", label="Test AUC")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("AUC")
    axes[0].set_title("Walk-Forward AUC")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(folds, wf_results["train_accuracy"], "b-o", label="Train Accuracy")
    axes[1].plot(folds, wf_results["test_accuracy"], "r-o", label="Test Accuracy")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Walk-Forward Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Walk-forward plot saved to {save_path}")

    plt.close()


def save_training_artifacts(
    classifier: PatternClassifier,
    training_results: Dict[str, Any],
    run_id: str,
) -> None:
    """Save trained model and metadata."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"pattern_classifier_{run_id}.pkl"
    classifier.save(model_path)
    logger.info(f"Model saved to {model_path}")

    metadata = {
        "run_id": run_id,
        "model_type": classifier.model_type,
        "n_estimators": classifier.n_estimators,
        "max_depth": classifier.max_depth,
        "learning_rate": classifier.learning_rate,
        "feature_names": classifier.feature_names_,
        "training_metrics": {
            "train_auc": training_results["training_result"].train_auc,
            "test_auc": training_results["training_result"].test_auc,
            "test_accuracy": training_results["training_result"].test_accuracy,
            "calibration_error": training_results["training_result"].calibration_error,
        },
        "feature_importance_top_20": dict(
            list(training_results["training_result"].feature_importance.items())[:20]
        ),
    }

    metadata_path = MODEL_DIR / f"pattern_classifier_{run_id}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Metadata saved to {metadata_path}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train ML pattern classifier")
    parser.add_argument(
        "--symbol",
        default="SPY",
        help="Symbol name (e.g., 'SPY') or file path (e.g., 'data/raw/SPY_daily.csv')",
    )
    parser.add_argument(
        "--start", default="2015-01-01", help="Start date for Yahoo Finance download"
    )
    parser.add_argument("--end", default="2024-12-31", help="End date for Yahoo Finance download")
    parser.add_argument(
        "--model-type",
        default="lightgbm",
        choices=["lightgbm", "xgboost", "random_forest", "gradient_boosting"],
        help="Model type",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix for model filename (e.g., '_v1', '_experimental')",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Train and compare all model types",
    )
    parser.add_argument(
        "--horizon",
        default=5,
        type=int,
        help="Forward horizon for labels",
    )
    parser.add_argument(
        "--no-walk-forward",
        action="store_true",
        help="Skip walk-forward validation",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of training iterations (default: 1, use -1 for indefinite loop)",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=0,
        help="Sleep seconds between iterations (default: 0)",
    )

    args = parser.parse_args()
    model_types = (
        ["lightgbm", "xgboost", "random_forest", "gradient_boosting"]
        if args.compare_models
        else [args.model_type]
    )

    iteration = 0
    while True:
        if args.iterations != -1 and iteration >= args.iterations:
            logger.info(f"Completed {iteration} iterations")
            break

        iteration += 1
        logger.info("=" * 80)
        logger.info(f"ITERATION {iteration}")
        logger.info("=" * 80)

        for model_type in model_types:
            args.model_type = model_type
            _run_training(args, model_type)

        if args.sleep > 0 and (args.iterations == -1 or iteration < args.iterations):
            logger.info(f"Sleeping for {args.sleep} seconds...")
            time.sleep(args.sleep)


def _run_training(args, model_type):
    run_id = f"{datetime.now():%Y%m%d_%H%M%S}"
    suffix = f"_{args.suffix}" if args.suffix else ""
    run_id = f"{run_id}{suffix}_{model_type}"
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Model type: {model_type}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Loading Data")
    logger.info("=" * 60)
    df = load_pattern_data(args.symbol, args.start, args.end)

    logger.info("=" * 60)
    logger.info("Feature Extraction")
    logger.info("=" * 60)
    features, feature_names = extract_features(df)

    logger.info("=" * 60)
    logger.info("Label Generation")
    logger.info("=" * 60)
    labels = generate_labels(df, horizon=args.horizon)

    X, y = prepare_training_data(df, features, labels)

    if not args.no_walk_forward:
        wf_results = train_with_walk_forward(
            X=X,
            y=y,
            model_type=model_type,
            train_size=500,
            step_size=100,
        )

        plot_walk_forward_results(
            wf_results["walk_forward_results"],
            save_path=PLOT_DIR / f"walk_forward_{run_id}.png",
        )

    logger.info("=" * 60)
    logger.info("Final Model Training")
    logger.info("=" * 60)
    classifier, training_results = train_final_model(
        X=X,
        y=y,
        model_type=model_type,
        test_size=0.3,
    )

    plot_feature_importance(
        classifier,
        top_n=20,
        save_path=PLOT_DIR / f"feature_importance_{run_id}.png",
    )

    save_training_artifacts(classifier, training_results, run_id)

    summary = {
        "run_id": run_id,
        "symbol": args.symbol,
        "model_type": model_type,
        "horizon": args.horizon,
        "n_samples": len(X),
        "n_features": len(feature_names),
        "train_auc": training_results["training_result"].train_auc,
        "test_auc": training_results["training_result"].test_auc,
        "test_accuracy": training_results["training_result"].test_accuracy,
        "calibration_error": training_results["training_result"].calibration_error,
        "top_features": dict(
            list(training_results["training_result"].feature_importance.items())[:10]
        ),
    }

    summary_path = OUTPUT_DIR / f"training_summary_{run_id}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info("=" * 60)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
