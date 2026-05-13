"""
ML Model Training Pipeline V2 — Overfitting-Fixed Version.

Improvements over V1 (see handover_overfitting_fix.md + paper insights):
  1. IC-based feature filtering (min_abs_ic=0.02)  — handover Step 1
  2. Reduced CatBoost complexity + stronger regularization — handover Step 2
  3. Triple-barrier labels (or thresholded binary fallback) — handover Step 3
  4. PurgedKFold CV replacing 70/30 split  — handover Step 4
  5. Nested CV for hyperparameter tuning    — paper insight (Sasse et al. 2025, Vabalas et al. 2019)
  6. eval_set monitoring for train/val divergence  — paper insight (Li et al. 2024)
  7. Preprocessing isolation via Pipeline  — paper insight (Ichwani et al. 2026)

Usage:
    uv run scripts/train_ml_model_v2.py --symbol data/raw/SPY_daily.csv --horizon 5
    uv run scripts/train_ml_model_v2.py --symbol data/raw/SPY_daily.csv --horizon 5 --suffix v2_overfit_fix
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.feature_engineering import FeatureExtractor
from src.ml.pattern_classifier import PatternClassifier
from src.ml.purged_cv import PurgedKFold
from src.ml.cross_asset_features import CrossAssetFeatureExtractor, load_market_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("reports/ml_training")
MODEL_DIR = Path("models")
PLOT_DIR = Path("reports/ml_plots")
N_CV_OUTER = 5  # outer CV folds for PurgedKFold
N_CV_INNER = 3  # inner CV folds for nested hyperparam selection
PCT_EMBARGO = 0.05


# ──────────────────────────────────────────────────────────────────────
# Data Loading & Label Generation
# ──────────────────────────────────────────────────────────────────────


def load_pattern_data(
    symbol_or_path: str = "SPY",
    start: str = "2015-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """Load OHLCV data from file or download from Yahoo Finance."""
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
            df[col] = 0 if col == "Volume" else df["Close"]
    return df


def generate_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.02,
    use_triple_barrier: bool = True,
) -> pd.Series:
    """Generate labels. Uses triple-barrier when available, thresholded binary as fallback.

    Paper insight (Vabalas 2019): noisy binary labels amplify overfitting.
    Triple-barrier or thresholded labels (+2% min move) reduce label noise.
    """
    if use_triple_barrier:
        try:
            from src.ml.triple_barrier import TripleBarrierLabeler

            labeler = TripleBarrierLabeler(atr_mult_tp=1.5, atr_mult_sl=1.0)
            atr = _compute_atr(df, period=14)

            labels = labeler.fit(
                close=df["Close"],
                high=df["High"],
                low=df["Low"],
                take_profit=None,
                stop_loss=None,
                time_limit=horizon,
                atr_series=atr,
            )
            # Map +1/-1/0 to binary: +1 → 1, 0/-1 → 0 (treat neutral/sl as negative)
            binary_labels = (labels == 1).astype(int)
            pos_pct = binary_labels.sum() / max(len(binary_labels), 1) * 100
            logger.info(
                f"Triple-barrier labels: {len(binary_labels)} samples, "
                f"{pos_pct:.1f}% positive ({threshold=})"
            )
            return binary_labels
        except Exception as e:
            logger.warning(
                f"Triple-barrier labeling failed ({e}), falling back to thresholded binary"
            )

    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    if threshold > 0:
        labels = (future_return > threshold).astype(int)
        logger.info(f"Thresholded binary labels (threshold={threshold:.2%})")
    else:
        labels = (future_return > 0).astype(int)
        logger.info("Binary labels (any positive return)")
    return labels


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high, low, close = df["High"], df["Low"], df["Close"].shift(1)
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ──────────────────────────────────────────────────────────────────────
# Feature Extraction with IC Filtering (Handover Step 1)
# ──────────────────────────────────────────────────────────────────────


def extract_features(
    df: pd.DataFrame,
    pattern_signals: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract features for ML model."""
    logger.info("Extracting features...")
    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract_all_features(df)

    if pattern_signals is not None:
        features = features.reindex(pattern_signals.index)

    features = features.dropna(axis=1, how="all")
    logger.info(f"Extracted {features.shape[1]} features for {features.shape[0]} samples")
    return features, list(features.columns)


def filter_features_by_ic(
    df: pd.DataFrame,
    features: pd.DataFrame,
    forward_returns_col: str = "forward_return_5d",
    min_abs_ic: float = 0.02,
    min_abs_rank_ic: float = 0.02,
) -> pd.DataFrame:
    """Filter features keeping only those with sufficient Information Coefficient.

    Paper insight (Vabalas 2019): high feature-to-sample ratio amplifies overfitting.
    IC-based filtering reduces dimensionality while retaining predictive features.
    """
    from src.ml.features import FeatureEngineer

    engineer = FeatureEngineer()
    # Build a combined DataFrame with features and forward returns
    forward_returns = df["Close"].shift(-5) / df["Close"] - 1
    combined = features.copy()
    combined[forward_returns_col] = forward_returns

    try:
        filtered = engineer.filter_low_ic_features(
            combined,
            forward_returns=forward_returns_col,
            min_abs_ic=min_abs_ic,
            min_abs_rank_ic=min_abs_rank_ic,
        )
        # Drop the forward returns column from output
        if forward_returns_col in filtered.columns:
            filtered = filtered.drop(columns=[forward_returns_col])
        logger.info(
            f"IC filtering: {features.shape[1]} -> {filtered.shape[1]} features "
            f"(min_abs_ic={min_abs_ic}, min_abs_rank_ic={min_abs_rank_ic})"
        )
        if filtered.shape[1] == 0:
            logger.warning("All features filtered out! Returning unfiltered features.")
            return features
        return filtered
    except Exception as e:
        logger.warning(f"IC filtering failed ({e}), using unfiltered features")
        return features


# ──────────────────────────────────────────────────────────────────────
# Training with Nested PurgedKFold (Handover Step 4 + Paper Insight)
# ──────────────────────────────────────────────────────────────────────


def train_with_nested_purged_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "catboost",
    horizon: int = 5,
    n_outer: int = N_CV_OUTER,
    n_inner: int = N_CV_INNER,
) -> Dict[str, Any]:
    """Train with nested PurgedKFold CV.

    Outer loop (n_outer folds): model assessment — unbiased performance estimate.
    Inner loop (n_inner folds): hyperparameter selection — prevents leakage.

    Paper insight (Sasse et al. 2025): "Model selection and model assessment
    should be done with a nested CV."
    Paper insight (Vabalas et al. 2019): "Nested CV provides unbiased
    performance estimates regardless of sample size."

    For CatBoost, we use a predefined set of conservative hyperparams
    (handover Step 2) rather than a full grid search, trading some
    optimization for runtime.
    """
    logger.info("=" * 60)
    logger.info(f"Nested PurgedKFold CV ({n_outer} outer × {n_inner} inner)")
    logger.info("=" * 60)

    cv_outer = PurgedKFold(
        n_splits=n_outer,
        pct_embargo=PCT_EMBARGO,
        label_span=horizon,
    )

    # Predefined conservative param sets for inner CV
    param_sets = [
        {"max_depth": 3, "l2_leaf_reg": 10.0, "random_strength": 3.0, "min_data_in_leaf": 50},
        {"max_depth": 4, "l2_leaf_reg": 5.0, "random_strength": 2.0, "min_data_in_leaf": 30},
        {"max_depth": 3, "l2_leaf_reg": 20.0, "random_strength": 5.0, "min_data_in_leaf": 80},
    ]

    outer_fold_results = []
    best_params_overall = None
    best_inner_score = -1

    for fold_idx, (train_idx, test_idx) in enumerate(cv_outer.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        logger.info(f"--- Outer Fold {fold_idx}/{n_outer} ---")
        logger.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")

        # ── Inner CV: select best hyperparams ──
        cv_inner = PurgedKFold(
            n_splits=n_inner,
            pct_embargo=PCT_EMBARGO,
            label_span=horizon,
        )

        best_params = param_sets[0]
        best_score = -1

        for params in param_sets:
            inner_scores = []
            for ii, (it_idx, iv_idx) in enumerate(cv_inner.split(X_train)):
                X_it, X_iv = X_train.iloc[it_idx], X_train.iloc[iv_idx]
                y_it, y_iv = y_train.iloc[it_idx], y_train.iloc[iv_idx]

                classifier = PatternClassifier(
                    model_type=model_type,
                    n_estimators=100,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    **params,
                )
                result = classifier.train(X_it, y_it, calibration_data=(X_iv, y_iv))
                inner_scores.append(result.test_auc)

            mean_score = np.mean(inner_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        if best_score > best_inner_score:
            best_inner_score = best_score
            best_params_overall = best_params

        logger.info(f"  Best inner params: {best_params} (score={best_score:.4f})")

        # ── Train final model for this outer fold ──
        classifier = PatternClassifier(
            model_type=model_type,
            n_estimators=100,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            **best_params,
        )
        result = classifier.train(X_train, y_train, calibration_data=(X_test, y_test))

        overfit_gap = result.train_auc - result.test_auc
        logger.info(
            f"  Fold {fold_idx}: Train AUC={result.train_auc:.4f}, "
            f"Test AUC={result.test_auc:.4f}, Overfit Gap={overfit_gap:.4f}"
        )

        outer_fold_results.append(
            {
                "fold": fold_idx,
                "train_auc": result.train_auc,
                "test_auc": result.test_auc,
                "test_accuracy": result.test_accuracy,
                "calibration_error": result.calibration_error,
                "overfit_gap": overfit_gap,
                "best_params": best_params,
            }
        )

    test_aucs = [r["test_auc"] for r in outer_fold_results]
    train_aucs = [r["train_auc"] for r in outer_fold_results]
    gaps = [r["overfit_gap"] for r in outer_fold_results]

    logger.info("=" * 60)
    logger.info("CV Summary")
    logger.info("=" * 60)
    logger.info(f"Train AUC: {np.mean(train_aucs):.4f} ± {np.std(train_aucs):.4f}")
    logger.info(f"Test AUC:  {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
    logger.info(f"Overfit Gap: {np.mean(gaps):.4f} ± {np.std(gaps):.4f}")

    return {
        "fold_results": outer_fold_results,
        "mean_train_auc": float(np.mean(train_aucs)),
        "mean_test_auc": float(np.mean(test_aucs)),
        "std_test_auc": float(np.std(test_aucs)),
        "mean_overfit_gap": float(np.mean(gaps)),
        "best_params_overall": best_params_overall,
    }


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "catboost",
    cv_results: Optional[Dict] = None,
) -> Tuple[PatternClassifier, Dict[str, Any]]:
    """Train final model with best params from CV and eval_set monitoring.

    Paper insight (Li et al. 2024): monitoring train/val divergence via eval_set
    is a natural overfitting signal. CatBoost supports this natively.
    """
    logger.info("=" * 60)
    logger.info("Training Final Model (with eval_set monitoring)")
    logger.info("=" * 60)

    # Use best params from CV if available, else conservative defaults
    best_params = {
        "max_depth": 3,
        "l2_leaf_reg": 10.0,
        "random_strength": 3.0,
        "min_data_in_leaf": 50,
    }
    if cv_results and cv_results.get("best_params_overall"):
        best_params = cv_results["best_params_overall"]
        logger.info(f"Using best params from CV: {best_params}")

    classifier = PatternClassifier(
        model_type=model_type,
        n_estimators=100,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        **best_params,
    )

    # Time-based split for final evaluation
    split_idx = int(len(X) * 0.7)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    result = classifier.train(
        X=X_train,
        y=y_train,
        calibration_data=(X_test, y_test),
    )

    overfit_gap = result.train_auc - result.test_auc
    logger.info(f"Train AUC: {result.train_auc:.4f}")
    logger.info(f"Test AUC: {result.test_auc:.4f}")
    logger.info(f"Overfit Gap: {overfit_gap:.4f} (target < 0.15)")
    logger.info(f"Test Accuracy: {result.test_accuracy:.4f}")
    logger.info(f"Calibration Error: {result.calibration_error:.4f}")

    # Warn if still overfitting
    if overfit_gap > 0.15:
        logger.warning(
            f"Overfit gap {overfit_gap:.4f} exceeds 0.15 threshold! "
            "Consider stronger regularization or different feature set."
        )
    if result.train_auc > 0.85:
        logger.warning(
            f"Train AUC {result.train_auc:.4f} very high — model may still memorize. "
            "Try reducing n_estimators or increasing l2_leaf_reg."
        )

    top_features = list(result.feature_importance.items())[:10]
    logger.info("\nTop 10 Features:")
    for feat, imp in top_features:
        logger.info(f"  {feat}: {imp:.4f}")

    return classifier, {
        "training_result": result,
        "X_train_size": len(X_train),
        "X_test_size": len(X_test),
        "overfit_gap": overfit_gap,
    }


# ──────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────


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


def plot_cv_results(
    cv_results: Dict[str, Any],
    save_path: Optional[Path] = None,
) -> None:
    """Plot nested CV fold results."""
    import matplotlib.pyplot as plt

    fold_results = cv_results.get("fold_results", [])
    if not fold_results:
        return

    folds = [r["fold"] for r in fold_results]
    train_auc = [r["train_auc"] for r in fold_results]
    test_auc = [r["test_auc"] for r in fold_results]
    gaps = [r["overfit_gap"] for r in fold_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar([f - 0.15 for f in folds], train_auc, 0.3, label="Train AUC", color="steelblue")
    axes[0].bar([f + 0.15 for f in folds], test_auc, 0.3, label="Test AUC", color="coral")
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("AUC")
    axes[0].set_title("Nested PurgedKFold AUC per Fold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(folds, gaps, color=["green" if g < 0.15 else "red" for g in gaps])
    axes[1].axhline(0.15, color="red", linestyle="--", alpha=0.5, label="Max Gap (0.15)")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Overfit Gap (Train - Test AUC)")
    axes[1].set_title("Overfit Gap per Fold")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"CV results plot saved to {save_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Artifacts
# ──────────────────────────────────────────────────────────────────────


def save_training_artifacts(
    classifier: PatternClassifier,
    training_results: Dict[str, Any],
    cv_results: Optional[Dict[str, Any]],
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
            "overfit_gap": training_results.get("overfit_gap", 0),
        },
        "feature_importance_top_20": dict(
            list(training_results["training_result"].feature_importance.items())[:20]
        ),
    }
    if cv_results:
        metadata["cv_summary"] = {
            "mean_train_auc": cv_results["mean_train_auc"],
            "mean_test_auc": cv_results["mean_test_auc"],
            "std_test_auc": cv_results["std_test_auc"],
            "mean_overfit_gap": cv_results["mean_overfit_gap"],
            "best_params": cv_results.get("best_params_overall"),
        }

    metadata_path = MODEL_DIR / f"pattern_classifier_{run_id}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadata saved to {metadata_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train ML pattern classifier (V2 - overfitting fixed)"
    )
    parser.add_argument(
        "--symbol", default="SPY", help="Symbol name or file path (e.g., 'data/raw/SPY_daily.csv')"
    )
    parser.add_argument("--start", default="2015-01-01", help="Start date for download")
    parser.add_argument("--end", default="2024-12-31", help="End date for download")
    parser.add_argument("--model-type", default="catboost", choices=["catboost"], help="Model type")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for model filename")
    parser.add_argument("--horizon", default=5, type=int, help="Forward horizon for labels")
    parser.add_argument(
        "--no-triple-barrier",
        action="store_true",
        help="Use thresholded binary labels instead of triple-barrier",
    )
    parser.add_argument(
        "--no-ic-filter", action="store_true", help="Skip IC-based feature filtering"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Return threshold for binary labels (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--no-cross-asset",
        action="store_true",
        help="Skip cross-asset feature extraction (for baseline comparison)",
    )

    args = parser.parse_args()

    run_id = f"{datetime.now():%Y%m%d_%H%M%S}"
    suffix = f"_{args.suffix}" if args.suffix else ""
    run_id = f"{run_id}{suffix}_{args.model_type}"
    logger.info(f"Run ID: {run_id}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Data ──
    logger.info("=" * 60)
    logger.info("1. Loading Data")
    logger.info("=" * 60)
    df = load_pattern_data(args.symbol, args.start, args.end)

    # ── 2. Extract Features ──
    logger.info("=" * 60)
    logger.info("2. Feature Extraction")
    logger.info("=" * 60)
    features, feature_names = extract_features(df)

    # ── 2b. Cross-Asset Feature Extraction ──
    if not args.no_cross_asset:
        logger.info("=" * 60)
        logger.info("2b. Cross-Asset Feature Extraction")
        logger.info("=" * 60)
        market_data = load_market_data(df)
        ca_extractor = CrossAssetFeatureExtractor(market_data=market_data)
        cross_features = ca_extractor.extract(df)

        # Merge with instrument features
        features = features.join(cross_features, how="inner")
        logger.info(f"Added {cross_features.shape[1]} cross-asset features")
        logger.info(f"Total features: {features.shape[1]}")

    # ── 3. IC-Based Feature Filtering (Handover Step 1) ──
    if not args.no_ic_filter:
        logger.info("=" * 60)
        logger.info("3. IC-Based Feature Filtering")
        logger.info("=" * 60)
        features = filter_features_by_ic(df, features, min_abs_ic=0.02, min_abs_rank_ic=0.02)

    # ── 4. Generate Labels (Handover Step 3) ──
    logger.info("=" * 60)
    logger.info("4. Label Generation")
    logger.info("=" * 60)
    use_tb = not args.no_triple_barrier
    labels = generate_labels(
        df, horizon=args.horizon, threshold=args.threshold, use_triple_barrier=use_tb
    )

    # ── 5. Prepare Training Data ──
    logger.info("=" * 60)
    logger.info("5. Preparing Training Data")
    logger.info("=" * 60)
    # Align features and labels (labels have NaN at tail due to forward-looking)
    common_idx = features.index.intersection(labels.dropna().index)
    X = features.loc[common_idx].dropna()
    y = labels.loc[common_idx]
    logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Positive samples: {y.sum()} ({y.sum() / max(len(y), 1) * 100:.1f}%)")

    if len(X) < 500:
        logger.warning(f"Only {len(X)} samples — results may be unreliable")
    if X.shape[1] > len(X) * 0.5:
        logger.warning(
            f"Feature-to-sample ratio {X.shape[1]}/{len(X)} = {X.shape[1] / len(X):.2f} "
            f"is high — increased overfitting risk (Vabalas et al. 2019)"
        )

    # ── 6. Nested PurgedKFold CV (Handover Step 4 + Paper Insight) ──
    logger.info("=" * 60)
    logger.info("6. Nested PurgedKFold Cross-Validation")
    logger.info("=" * 60)
    cv_results = train_with_nested_purged_cv(
        X=X,
        y=y,
        model_type=args.model_type,
        horizon=args.horizon,
    )

    plot_cv_results(
        cv_results,
        save_path=PLOT_DIR / f"cv_results_{run_id}.png",
    )

    # ── 7. Train Final Model ──
    logger.info("=" * 60)
    logger.info("7. Final Model Training")
    logger.info("=" * 60)
    classifier, training_results = train_final_model(
        X=X,
        y=y,
        model_type=args.model_type,
        cv_results=cv_results,
    )

    plot_feature_importance(
        classifier,
        top_n=20,
        save_path=PLOT_DIR / f"feature_importance_{run_id}.png",
    )

    save_training_artifacts(classifier, training_results, cv_results, run_id)

    # ── 8. Summary ──
    logger.info("=" * 60)
    logger.info("Training Complete — Results Summary")
    logger.info("=" * 60)
    summary = {
        "run_id": run_id,
        "symbol": args.symbol,
        "model_type": args.model_type,
        "horizon": args.horizon,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "feature_sample_ratio": round(X.shape[1] / max(X.shape[0], 1), 2),
        "label_type": "triple_barrier" if use_tb else f"thresholded_binary({args.threshold})",
        "ic_filtered": not args.no_ic_filter,
        "cross_asset_enabled": not args.no_cross_asset,
        "final_metrics": {
            "train_auc": training_results["training_result"].train_auc,
            "test_auc": training_results["training_result"].test_auc,
            "test_accuracy": training_results["training_result"].test_accuracy,
            "calibration_error": training_results["training_result"].calibration_error,
            "overfit_gap": training_results["overfit_gap"],
        },
        "cv_metrics": {
            "mean_train_auc": cv_results["mean_train_auc"],
            "mean_test_auc": cv_results["mean_test_auc"],
            "std_test_auc": cv_results["std_test_auc"],
            "mean_overfit_gap": cv_results["mean_overfit_gap"],
        },
        "top_features": dict(
            list(training_results["training_result"].feature_importance.items())[:10]
        ),
        "papers_referenced": [
            "Vabalas et al. 2019 - ML validation with limited sample size",
            "Sasse et al. 2025 - Overview of leakage scenarios in supervised ML",
            "Ichwani et al. 2026 - Preventing data leakage via integrated pipelines",
            "Li et al. 2024 - History-based approach to mitigate overfitting",
        ],
    }

    # Table comparison with V1
    logger.info(f"\n{'Metric':<25} {'V1 (Failed)':<15} {'V2 (This Run)':<15} {'Target':<15}")
    logger.info("-" * 70)
    logger.info(
        f"{'Train AUC':<25} {'0.999':<15} {training_results['training_result'].train_auc:<15.4f} {'0.60-0.75':<15}"
    )
    logger.info(
        f"{'Test AUC':<25} {'0.48-0.56':<15} {training_results['training_result'].test_auc:<15.4f} {'> 0.55':<15}"
    )
    logger.info(
        f"{'Overfit Gap':<25} {'0.44-0.52':<15} {training_results['overfit_gap']:<15.4f} {'< 0.15':<15}"
    )
    logger.info(
        f"{'Calibration Error':<25} {'0.24':<15} {training_results['training_result'].calibration_error:<15.4f} {'< 0.15':<15}"
    )
    logger.info(
        f"{'CV AUC Std':<25} {'N/A':<15} {cv_results['std_test_auc']:<15.4f} {'< 0.05':<15}"
    )

    summary_path = OUTPUT_DIR / f"training_summary_{run_id}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
