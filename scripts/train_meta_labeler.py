"""
Train MetaLabelerV2 secondary filter (B13).

Given a trained primary model, this script:
1. Loads OHLCV data for a symbol
2. Loads the primary model and generates probability scores
3. Identifies signal bars (where primary prob >= entry_threshold)
4. Builds meta-labeler features (regime, volatility, signal clustering, recency)
5. Trains CatBoost classifier on "was this signal profitable?" binary target
6. Saves the trained MetaLabelerV2 for use in MLStrategy

Usage:
    uv run scripts/train_meta_labeler.py SPY
    uv run scripts/train_meta_labeler.py SPY --model models/pattern_classifier_v3_SPY_20260511_224704.pkl
    uv run scripts/train_meta_labeler.py SPY --entry-threshold 0.45 --horizon 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.feature_engineering import FeatureExtractor
from src.ml.meta_labeler_v2 import MetaLabelerV2, MetaLabelV2Result
from src.ml.pattern_classifier import PatternClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(symbol: str, start: str = "2015-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Load OHLCV data from CSV."""
    path = Path(f"data/raw/{symbol}_daily.csv")
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol} at {path}")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df.loc[start:end]
    df = df.dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0 if col == "Volume" else df.iloc[:, 0]
    return df


def generate_primary_probs(
    df: pd.DataFrame,
    model: PatternClassifier,
) -> pd.Series:
    """Generate primary model probability scores for all bars."""
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(df, include_forward_returns=False)
    features = features.ffill().bfill().fillna(0)

    model_features = list(model.feature_names_)
    missing = [c for c in model_features if c not in features.columns]
    for c in missing:
        features[c] = 0.0

    predictions = model.predict(features[model_features])
    return predictions["probability_profitable"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MetaLabelerV2 secondary filter (B13)")
    parser.add_argument("symbol", type=str, help="Ticker symbol (e.g. SPY)")
    parser.add_argument(
        "--model",
        type=str,
        default="models/pattern_classifier_v3_SPY_20260514_124612.pkl",
        help="Path to primary PatternClassifier model",
    )
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=0.45,
        help="Minimum primary probability to identify signals for training (default 0.45)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Profit horizon in bars (default 5)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="PurgedKFold splits (default 5)",
    )
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for meta-labeler model (default: models/meta_labeler_v2_<symbol>_<ts>.pkl)",
    )

    args = parser.parse_args()

    # Load data
    logger.info("Loading data for %s (%s → %s)", args.symbol, args.start, args.end)
    df = load_data(args.symbol, args.start, args.end)
    logger.info("Loaded %d bars", len(df))

    # Load primary model
    logger.info("Loading primary model: %s", args.model)
    model = PatternClassifier()
    model.load(args.model)
    logger.info("Primary model: %d features", len(model.feature_names_))

    # Generate primary probabilities
    logger.info("Generating primary model probabilities...")
    primary_probs = generate_primary_probs(df, model)
    logger.info(
        "Primary prob stats: mean=%.6f std=%.6f min=%.6f max=%.6f",
        primary_probs.mean(),
        primary_probs.std(),
        primary_probs.min(),
        primary_probs.max(),
    )

    # Identify signal bars
    signal_mask = primary_probs >= args.entry_threshold
    signal_indices = primary_probs[signal_mask].index
    logger.info(
        "Signals at entry_threshold=%.2f: %d (%.1f%% of bars)",
        args.entry_threshold,
        len(signal_indices),
        100 * len(signal_indices) / max(len(df), 1),
    )

    if len(signal_indices) < 50:
        logger.error(
            "Only %d signals found — need >= 50 for reliable meta-labeler. "
            "Lower --entry-threshold or use more data.",
            len(signal_indices),
        )
        sys.exit(1)

    # Train MetaLabelerV2
    logger.info("=" * 60)
    logger.info("Training MetaLabelerV2 (CatBoost)")
    logger.info("=" * 60)

    meta = MetaLabelerV2(
        profit_horizon=args.horizon,
        n_splits=args.n_splits,
        pct_embargo=0.05,
    )

    result = meta.fit(df, signal_indices, primary_probs)

    logger.info("Training complete:")
    logger.info("  AUC: %.4f ± %.4f", result.auc_mean, result.auc_std)
    logger.info("  Threshold: %.4f", result.threshold)
    logger.info(
        "  Signals: %d (%.1f%% profitable)", result.n_signals, result.baseline_win_rate * 100
    )
    logger.info("  Top features:")
    for feat, imp in sorted(result.feature_importance.items(), key=lambda x: x[1], reverse=True)[
        :8
    ]:
        logger.info("    %-25s %.4f", feat, imp)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"models/meta_labeler_v2_{args.symbol}_{timestamp}.pkl"
    meta.save(output_path)
    logger.info("Saved meta-labeler to %s", output_path)

    # Signal reduction analysis
    meta_filtered = meta.predict_batch(meta._build_features(df, signal_indices, primary_probs))
    n_passed = meta_filtered["meta_take_trade"].sum()
    reduction_pct = (1 - n_passed / max(len(signal_indices), 1)) * 100
    logger.info(
        "Signal reduction: %d/%d signals pass meta-filter (%.1f%% reduction)",
        n_passed,
        len(signal_indices),
        reduction_pct,
    )


if __name__ == "__main__":
    main()
