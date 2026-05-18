"""
Train per-regime CatBoost models for RegimeRouter.

Labels bars with TrendRegime + VolRegime (Bull_Low, Bull_High, Bear_Low, Bear_High),
then trains one CatBoost model per regime with sufficient bars.

Usage:
    uv run scripts/train_per_regime_models.py SPY --start 2015-01-01 --end 2024-12-31
    uv run scripts/train_per_regime_models.py SPY --fast
    uv run scripts/train_per_regime_models.py SPY --exclude vol_regime,volatility_regime,ema_21_55_spread

Config JSON saved to models/regime_router_SPY.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.feature_engineering import FeatureExtractor
from src.ml.pattern_classifier import PatternClassifier
from src.ml.simple_regime import CombinedSimpleRegimeDetector
from src.ml.triple_barrier import TripleBarrierLabeler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("models")
MIN_BARS_PER_REGIME = 250


def load_data(symbol: str) -> pd.DataFrame:
    path = Path(f"data/raw/{symbol}_daily.csv")
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol}")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df.dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0 if col == "Volume" else df.iloc[:, 0]
    return df


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def train_regime_model(
    X: pd.DataFrame,
    y: pd.Series,
    regime_name: str,
    fast: bool = False,
) -> tuple[object, dict, float]:
    """Train a CatBoost model on regime-specific data."""
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    n_estimators = 50 if fast else 100
    clf = PatternClassifier(
        model_type="catboost",
        n_estimators=n_estimators,
        learning_rate=0.03,
        max_depth=3,
        l2_leaf_reg=10.0,
        random_strength=3.0,
        min_data_in_leaf=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    result = clf.train(X_train, y_train, calibration_data=(X_test, y_test))
    gap = result.train_auc - result.test_auc

    logger.info(
        f"  {regime_name}: AUC={result.test_auc:.4f}, Gap={gap:.4f}, "
        f"n_train={len(y_train)}, n_test={len(y_test)}"
    )

    top = list(result.feature_importance.items())[:5]
    logger.info(f"  Top features: {[f'{k}={v:.3f}' for k, v in top]}")

    metrics = {
        "train_auc": result.train_auc,
        "test_auc": result.test_auc,
        "calibration_error": result.calibration_error,
        "overfit_gap": gap,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "top_features": dict(top),
    }
    return clf, metrics, gap


def main() -> None:
    parser = argparse.ArgumentParser(description="Train per-regime CatBoost models")
    parser.add_argument("symbol", default="SPY", nargs="?", help="Ticker symbol")
    parser.add_argument("--start", default="2015-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument("--fast", action="store_true", help="Fast mode (fewer estimators)")
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated features to exclude (e.g. vol_regime,ema_21_55_spread)",
    )
    parser.add_argument(
        "--ma-period", type=int, default=200, help="SMA period for trend regime (default 200)"
    )
    parser.add_argument(
        "--vol-pctile", type=float, default=80.0, help="ATR percentile for high vol (default 80)"
    )
    args = parser.parse_args()

    # ── Load data ──
    df = load_data(args.symbol)
    if args.start:
        df = df.loc[args.start :]
    if args.end:
        df = df.loc[: args.end]
    logger.info(f"Loaded {args.symbol}: {len(df)} bars")

    # ── Generate features ──
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(df, include_forward_returns=False)
    features = features.dropna()
    logger.info(f"Features: {features.shape[1]} cols, {len(features)} rows")

    # ── Exclude flipped features ──
    exclude_cols = [c.strip() for c in args.exclude.split(",") if c.strip()]
    if exclude_cols:
        existing = [c for c in exclude_cols if c in features.columns]
        if existing:
            features = features.drop(columns=existing)
            logger.info(f"Excluded features: {existing}")

    # ── Generate triple-barrier labels ──
    atr = _compute_atr(df)
    labeler = TripleBarrierLabeler(atr_mult_tp=1.5, atr_mult_sl=1.0)
    labels = labeler.fit(
        close=df["Close"],
        high=df["High"],
        low=df["Low"],
        take_profit=None,
        stop_loss=None,
        time_limit=5,
        atr_series=atr,
    )
    labels_binary = (labels == 1).astype(int)
    common = features.index.intersection(labels_binary.dropna().index)
    features = features.loc[common]
    labels_binary = labels_binary.loc[common]
    close = df.loc[common, "Close"]
    logger.info(f"Labels: {len(labels_binary)} samples, {labels_binary.mean() * 100:.1f}% positive")

    # ── Detect regimes (trend-only for balanced splits) ──
    from src.ml.simple_regime import SimpleTrendRegimeDetector

    regime_detector = SimpleTrendRegimeDetector(
        ma_period=args.ma_period,
    )
    regime_detector.fit(df)
    regimes = regime_detector.predict(df)
    regimes = regimes.loc[features.index]

    regime_counts = regimes.value_counts().to_dict()
    logger.info(f"Regime distribution: {regime_counts}")

    # ── Train per-regime models ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_entries: dict[str, str] = {}
    regime_metrics: dict[str, dict] = {}

    for regime_name, count in sorted(regime_counts.items(), key=lambda x: x[1], reverse=True):
        if count < MIN_BARS_PER_REGIME:
            logger.warning(f"Skipping {regime_name}: {count} bars < {MIN_BARS_PER_REGIME} minimum")
            continue

        mask = regimes == regime_name
        X_r = features.loc[mask]
        y_r = labels_binary.loc[mask]

        logger.info(f"\nTraining {regime_name} ({len(y_r)} bars, {y_r.mean() * 100:.1f}% positive)")

        model, metrics, gap = train_regime_model(X_r, y_r, regime_name, fast=args.fast)

        model_path = OUTPUT_DIR / f"regime_{regime_name}_{args.symbol}_{timestamp}.pkl"
        model.save(str(model_path))
        model_entries[regime_name] = str(model_path)
        regime_metrics[regime_name] = metrics

    # ── Train fallback (all-regime) model ──
    logger.info(f"\nTraining FALLBACK model ({len(features)} bars)")
    fallback_model, fallback_metrics, fallback_gap = train_regime_model(
        features, labels_binary, "FALLBACK", fast=args.fast
    )
    fallback_path = OUTPUT_DIR / f"regime_FALLBACK_{args.symbol}_{timestamp}.pkl"
    fallback_model.save(str(fallback_path))

    # ── Save RegimeRouter config ──
    config = {
        "symbol": args.symbol,
        "timestamp": timestamp,
        "detector": "SimpleTrendRegimeDetector",
        "ma_period": args.ma_period,
        "regime_models": model_entries,
        "fallback_model": str(fallback_path),
        "excluded_features": exclude_cols,
        "n_regimes_trained": len(model_entries),
        "regime_counts": regime_counts,
        "regime_metrics": regime_metrics,
        "fallback_metrics": fallback_metrics,
    }
    config_path = OUTPUT_DIR / f"regime_router_{args.symbol}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(f"\n{'=' * 60}")
    logger.info("Per-Regime (Trend) Training Complete")
    logger.info(f"{'=' * 60}")
    logger.info(f"Trained {len(model_entries)} regime models:")
    for regime, path in model_entries.items():
        m = regime_metrics[regime]
        logger.info(
            f"  {regime:20s} | AUC={m['test_auc']:.4f} | Gap={m['overfit_gap']:.4f} "
            f"| n_train={m['n_train']} | n_test={m['n_test']} | {path}"
        )
    logger.info(
        f"  {'FALLBACK':20s} | AUC={fallback_metrics['test_auc']:.4f} "
        f"| Gap={fallback_metrics['overfit_gap']:.4f} | {fallback_path}"
    )
    logger.info(f"\nConfig: {config_path}")


if __name__ == "__main__":
    main()
