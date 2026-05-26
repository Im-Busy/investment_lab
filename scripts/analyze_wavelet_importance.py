"""
Analyze wavelet feature importance with SHAP.

Phase 27B B3.2 — Train a CatBoost model with wavelet features, then use SHAP
to rank feature importance. Reports which wavelet features appear in the top-20,
which decomposition levels matter most, and whether wavelet features outrank
traditional indicators.

Gate: Wavelet features must appear in top-20 SHAP importance for ≥3 of 5 instruments.

Usage:
    uv run scripts/analyze_wavelet_importance.py --symbol SPY
    uv run scripts/analyze_wavelet_importance.py --basket SPY,QQQ,GLD,XLK,SLV
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.pattern_classifier import PatternClassifier
from scripts.train_ml_pipeline_v3 import (
    load_data,
    generate_labels,
    extract_features,
    DEFAULT_HORIZON,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def _get_shap_values(model: Any, X: pd.DataFrame) -> np.ndarray | None:
    try:
        import shap

        booster = model.model
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X)
        return shap_values
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return None


def analyze_ticker(
    ticker: str,
    start: str = "2016-01-01",
    end: str = "2024-12-31",
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, Any]:
    """Train CatBoost with wavelet features on a single ticker and compute SHAP importance."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Analyzing {ticker}")
    logger.info("=" * 60)

    df = load_data(ticker, start=start, end=end)
    features = extract_features({"TICKER": df}, use_cross_asset=False, use_wavelet=True)
    fx = features["TICKER"]

    labels = generate_labels(df, horizon=horizon)
    labels.attrs = {}

    common_idx = fx.dropna().index.intersection(labels.dropna().index)
    X = fx.loc[common_idx].fillna(0)
    y = labels.loc[common_idx]

    logger.info(f"  Samples: {len(X)}, Features: {X.shape[1]}, Positive: {y.mean():.1%}")

    clf = PatternClassifier(
        model_type="catboost",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        l2_leaf_reg=5.0,
        random_state=42,
    )
    result = clf.train(X, y)
    logger.info(f"  Train AUC: {result.train_auc:.4f}, Test AUC: {result.test_auc:.4f}")

    shap_values = _get_shap_values(clf, X)
    if shap_values is None:
        return {
            "ticker": ticker,
            "train_auc": result.train_auc,
            "test_auc": result.test_auc,
            "error": "SHAP computation failed",
        }

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_names = X.columns.tolist()
    ranked = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)

    top_20 = ranked[:20]
    top_50 = ranked[:50]

    wavelet_in_top20 = [
        (name, rank + 1, score)
        for rank, (name, score) in enumerate(ranked)
        if "wavelet" in name.lower() and rank < 20
    ]
    wavelet_in_top50 = [
        (name, rank + 1, score)
        for rank, (name, score) in enumerate(ranked)
        if "wavelet" in name.lower() and rank < 50
    ]
    total_wavelet = sum(1 for name, _ in ranked if "wavelet" in name.lower())

    level_counts: dict[str, dict] = {}
    for name, _rank, score in wavelet_in_top50:
        parts = name.split("_")
        for p in parts:
            if p.startswith("cA") or p.startswith("cD"):
                if p not in level_counts:
                    level_counts[p] = {"count": 0, "total_shap": 0.0, "features": []}
                level_counts[p]["count"] += 1
                level_counts[p]["total_shap"] += float(score)
                level_counts[p]["features"].append(name)
                break

    logger.info("\n  SHAP Feature Importance (Top 20):")
    for rank, (name, score) in enumerate(top_20, 1):
        marker = " [WAVELET]" if "wavelet" in name.lower() else ""
        logger.info(f"    {rank:2d}. {name:<50s} {score:8.4f}{marker}")

    logger.info("\n  Wavelet Features Summary:")
    logger.info(f"    Total wavelet features: {total_wavelet}")
    logger.info(f"    Wavelet in Top-20: {len(wavelet_in_top20)}")
    logger.info(f"    Wavelet in Top-50: {len(wavelet_in_top50)}")

    if level_counts:
        logger.info("\n  Wavelet Level Importance:")
        for level_name in sorted(level_counts.keys()):
            lc = level_counts[level_name]
            logger.info(
                f"    {level_name}: {lc['count']} features, total SHAP={lc['total_shap']:.4f}"
            )

    return {
        "ticker": ticker,
        "train_auc": result.train_auc,
        "test_auc": result.test_auc,
        "n_features": X.shape[1],
        "n_wavelet_features": total_wavelet,
        "wavelet_top20_count": len(wavelet_in_top20),
        "wavelet_top50_count": len(wavelet_in_top50),
        "wavelet_top20": [(name, rank, float(score)) for name, rank, score in wavelet_in_top20],
        "wavelet_top50": [(name, rank, float(score)) for name, rank, score in wavelet_in_top50],
        "level_importance": {
            k: {"count": v["count"], "total_shap": float(v["total_shap"])}
            for k, v in level_counts.items()
        },
        "top_20": [(name, rank, float(score)) for rank, (name, score) in enumerate(top_20, 1)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze wavelet feature importance with SHAP")
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument(
        "--basket",
        type=str,
        default=None,
        help="Comma-separated tickers for multi-instrument analysis",
    )
    parser.add_argument("--start", type=str, default="2016-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    args = parser.parse_args()

    tickers = [t.strip() for t in args.basket.split(",")] if args.basket else [args.symbol]
    results: list[dict] = []

    for ticker in tickers:
        result = analyze_ticker(ticker, start=args.start, end=args.end, horizon=args.horizon)
        results.append(result)

    n_pass = sum(1 for r in results if r.get("wavelet_top20_count", 0) >= 1)
    gate = n_pass >= min(3, len(tickers))

    print("\n" + "=" * 80)
    print("MULTI-INSTRUMENT SUMMARY")
    print("=" * 80)
    print(
        f"{'Ticker':<10} {'Train AUC':>10} {'Test AUC':>10} {'Wavelet Top20':>14} {'Wavelet Top50':>14}"
    )
    print("-" * 60)
    for r in results:
        print(
            f"  {r['ticker']:<8} {r['train_auc']:10.4f} {r['test_auc']:10.4f} "
            f"{r.get('wavelet_top20_count', 0):14d} {r.get('wavelet_top50_count', 0):14d}"
        )

    print(
        f"\n  Gate: {'PASSED' if gate else 'FAILED'} "
        f"(>=1 wavelet feature in top-20 for >= {min(3, len(tickers))} of {len(tickers)} instruments)"
    )

    out_path = (
        Path("outputs") / "wavelet_benchmark" / f"shap_analysis_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {"results": results, "gate_passed": gate, "timestamp": datetime.now().isoformat()},
            f,
            indent=2,
            default=str,
        )
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
