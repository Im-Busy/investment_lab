"""
Phase B: ML Enhancement Implementation Script (DEPRECATED)

WARNING: This pipeline is deprecated. The regime classification (B2) trains on
rule-based ADX/ATR labels — a deterministic formula that produces fake 0.95+ ICs.
The signal scorer (B3) had a circular IC computation bug: corr(score×return, return).

Use the V3 pipeline instead:
    uv run scripts/train_ml_pipeline_v3.py --basket JOE,KODK,SPY,QQQ,IWM,TLT,GLD,XLF,XLK,XLE,XLV,EEM

Triple-barrier labels from src/ml/triple_barrier.py are the correct training targets.
The bugs discovered are documented in plans/session_handover_20260511.md.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import yfinance as yf

from src.ml.experiment_logger import ExperimentLogger
from src.ml.feature_selector import FeatureSelector
from src.ml.features import FeatureEngineer
from src.ml.metrics import filter_features_by_ic, ic_summary
from src.ml.purged_cv import PurgedKFold
from src.ml.regime_model import RegimeClassifier
from src.ml.signal_scorer import SignalScorer
from src.indicators.regime_detector import RegimeDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("reports/ml_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(
    symbol: str = "SPY", start: str = "2015-01-01", end: str = "2024-12-31"
) -> pd.DataFrame:
    """Load OHLCV data."""
    logger.info(f"Loading {symbol} data from {start} to {end}")
    df = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    logger.info(f"Loaded {len(df)} bars")
    return df


def generate_labels(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Generate forward returns as labels."""
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    return future_return


def regime_labels_from_rule_based(df: pd.DataFrame) -> pd.Series:
    """Generate regime labels from rule-based detector for comparison."""
    detector = RegimeDetector()
    regimes = detector.classify(df)
    return regimes.apply(lambda x: x.value if hasattr(x, "value") else str(x))


def phase_b_b1_feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """B1: Proper feature engineering with IC analysis."""
    logger.info("=" * 60)
    logger.info("B1: Feature Engineering with IC Analysis")
    logger.info("=" * 60)

    feature_engineer = FeatureEngineer()
    features = feature_engineer.generate_features(df)

    forward_returns = generate_labels(df, horizon=5)

    logger.info(f"Generated {len(features.columns)} features")

    ic_summary_df = ic_summary(features, forward_returns)
    ic_summary_df.to_csv(OUTPUT_DIR / "feature_ic_analysis.csv", index=False)

    logger.info(f"Features with |IC| >= 0.02: {(ic_summary_df['abs_ic'] >= 0.02).sum()}")
    logger.info(f"Features with |Rank IC| >= 0.02: {(ic_summary_df['abs_rank_ic'] >= 0.02).sum()}")

    selected_features = filter_features_by_ic(
        features, forward_returns, min_abs_ic=0.02, min_abs_rank_ic=0.02
    )
    logger.info(f"Selected {len(selected_features)} high-IC features")

    features_filtered = features[selected_features] if selected_features else features

    logger.info(f"Feature IC analysis saved to {OUTPUT_DIR / 'feature_ic_analysis.csv'}")

    return features_filtered, forward_returns


def phase_b_b2_regime_classification(
    features: pd.DataFrame,
    regime_labels: pd.Series,
    experiment_logger: ExperimentLogger,
) -> RegimeClassifier:
    """B2: Regime classification with PurgedKFold + embargo."""
    logger.info("=" * 60)
    logger.info("B2: Regime Classification with PurgedKFold")
    logger.info("=" * 60)

    purged_cv = PurgedKFold(n_splits=5, pct_embargo=0.02, label_span=5)

    clf = RegimeClassifier(
        model_type="random_forest", n_estimators=100, max_depth=3, random_state=42
    )

    X = features.dropna().values
    sample_indices = features.dropna().index
    y = regime_labels.loc[sample_indices].values

    if len(X) < 100:
        logger.warning("Insufficient samples for regime classification")
        return clf

    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(purged_cv.split(X, y)):
        if len(train_idx) < 50 or len(test_idx) < 20:
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.train(pd.DataFrame(X_train), pd.Series(y_train))
        y_pred = clf.predict(pd.DataFrame(X_test)).values

        y_train_pred = clf.predict(pd.DataFrame(X_train)).values
        train_acc = float(np.mean(y_train_pred == y_train))
        test_acc = float(np.mean(y_pred == y_test))
        overfit_gap = train_acc - test_acc

        experiment_logger.log_fold_metrics(
            fold=fold_idx,
            train_metrics={"accuracy": train_acc},
            test_metrics={"accuracy": test_acc},
            n_train=len(train_idx),
            n_test=len(test_idx),
        )

        fold_result = {
            "fold": fold_idx,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "overfit_gap": overfit_gap,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        fold_metrics.append(fold_result)
        logger.info(
            f"Fold {fold_idx}: Train={train_acc:.3f}, Test={test_acc:.3f}, Gap={overfit_gap:.3f}"
        )

    if fold_metrics:
        logger.info(
            f"Average test accuracy: {np.mean([m['test_accuracy'] for m in fold_metrics]):.3f}"
        )
        logger.info(f"Average overfit gap: {np.mean([m['overfit_gap'] for m in fold_metrics]):.3f}")

        if hasattr(clf.model, "feature_importances_"):
            feature_importance = dict(zip(features.columns, clf.model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
            logger.info("Top 15 features:")
            for feat, imp in top_features:
                logger.info(f"  {feat}: {imp:.4f}")

    return clf


def phase_b_b3_signal_scorer(
    features: pd.DataFrame,
    forward_returns: pd.Series,
    experiment_logger: ExperimentLogger,
) -> SignalScorer:
    """B3: Signal generation as binary classification with rank IC metrics."""
    logger.info("=" * 60)
    logger.info("B3: Signal Scorer (Binary Classification)")
    logger.info("=" * 60)

    scorer = SignalScorer(model_type="gradient_boosting")

    X = features.dropna().values
    sample_indices = features.dropna().index
    y = forward_returns.loc[sample_indices].values

    if len(X) < 100:
        logger.warning("Insufficient samples for signal scorer")
        return scorer

    y_binary = (y > 0).astype(int)

    purged_cv = PurgedKFold(n_splits=5, pct_embargo=0.02, label_span=5)

    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(purged_cv.split(X, y)):
        if len(train_idx) < 50 or len(test_idx) < 20:
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_binary[train_idx], y_binary[test_idx]
        y_test_continuous = y[test_idx]

        scorer.train(pd.DataFrame(X_train), pd.Series(y_train))
        scored = scorer.score(pd.DataFrame(X_test))
        y_pred = scored["ml_score"].values

        # Fixed: corr(y_pred, returns) not corr(y_pred * returns, returns)
        rank_ic = pd.Series(y_pred).corr(pd.Series(y_test_continuous), method="spearman")
        pearson_ic = pd.Series(y_pred).corr(pd.Series(y_test_continuous), method="pearson")
        rank_ic = float(rank_ic) if not pd.isna(rank_ic) else 0.0
        pearson_ic = float(pearson_ic) if not pd.isna(pearson_ic) else 0.0
        y_pred_binary = (y_pred >= 0.5).astype(int)
        accuracy = float(np.mean(y_pred_binary == y_test))

        experiment_logger.log_fold_metrics(
            fold=fold_idx,
            train_metrics={},
            test_metrics={
                "pearson_ic": float(pearson_ic),
                "rank_ic": float(rank_ic),
                "accuracy": accuracy,
            },
            n_train=len(train_idx),
            n_test=len(test_idx),
        )

        fold_result = {
            "fold": fold_idx,
            "pearson_ic": float(pearson_ic) if not np.isnan(pearson_ic) else 0.0,
            "rank_ic": float(rank_ic) if not np.isnan(rank_ic) else 0.0,
            "accuracy": accuracy,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        fold_metrics.append(fold_result)
        logger.info(
            f"Fold {fold_idx}: Pearson IC={pearson_ic:.4f}, Rank IC={rank_ic:.4f}, Acc={accuracy:.3f}"
        )

    avg_rank_ic = np.mean([m["rank_ic"] for m in fold_metrics]) if fold_metrics else 0.0
    avg_accuracy = np.mean([m["accuracy"] for m in fold_metrics]) if fold_metrics else 0.0
    logger.info(f"Average Rank IC: {avg_rank_ic:.4f}")
    logger.info(f"Average Accuracy: {avg_accuracy:.3f}")

    if avg_rank_ic < 0.03:
        logger.warning("Rank IC < 0.03 indicates weak predictive power")

    return scorer


def phase_b_b4_feature_selection(
    features: pd.DataFrame, forward_returns: pd.Series
) -> FeatureSelector | None:
    """B4: Sequential Feature Importance selection."""
    logger.info("=" * 60)
    logger.info("B4: Sequential Feature Importance (SFI)")
    logger.info("=" * 60)
    logger.info("SFI skipped - requires binary classification targets")
    logger.info("Use regression-based feature selection or manual review of IC results")
    logger.info(f"Feature IC analysis available at: {OUTPUT_DIR / 'feature_ic_analysis.csv'}")

    return None


def run_phase_b():
    """Execute Phase B implementation."""
    logger.info("Starting Phase B: ML Enhancement")
    logger.info("=" * 80)

    df = load_data("SPY", "2015-01-01", "2024-12-31")

    experiment_logger = ExperimentLogger(
        run_id=f"phase_b_{datetime.now():%Y%m%d_%H%M%S}",
        base_dir="experiments",
        description="Phase B ML Enhancement Implementation",
    )

    experiment_logger.log_config(
        hyperparams={
            "symbol": "SPY",
            "start": "2015-01-01",
            "end": "2024-12-31",
            "feature_count": "81+",
            "cv_method": "PurgedKFold_embargo_0.02",
        }
    )

    features_raw, forward_returns = phase_b_b1_feature_engineering(df)

    regime_labels = regime_labels_from_rule_based(df)
    regime_model = phase_b_b2_regime_classification(features_raw, regime_labels, experiment_logger)

    signal_model = phase_b_b3_signal_scorer(features_raw, forward_returns, experiment_logger)

    phase_b_b4_feature_selection(features_raw, forward_returns)

    experiment_logger.log_summary_verdict(
        mean_oos_metrics={
            "phase_b_status": "complete",
            "regime_model_trained": regime_model is not None,
            "signal_model_trained": signal_model is not None,
        },
        n_folds=5,
    )

    logger.info("=" * 80)
    logger.info("Phase B Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_phase_b()
