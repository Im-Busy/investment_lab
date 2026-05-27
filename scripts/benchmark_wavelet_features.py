"""
Benchmark wavelet features against baseline CatBoost model.

Phase 27B B3.1 — Train 2 CatBoost models (identical params except features):
  Model A: Existing features (price, volume, indicators, FFT)
  Model B: A + wavelet features (34-114 new)

Metrics: AUC, Sharpe, MRE gap, overfit gap.
Dataset: configurable instruments, IS=2016-2021, OOS=2022-2026.
Gate: B must improve OOS Sharpe by ≥5% OR reduce MRE gap by ≥10%.

Usage:
    uv run scripts/benchmark_wavelet_features.py --symbol SPY
    uv run scripts/benchmark_wavelet_features.py --basket SPY,QQQ,GLD,XLK --fast
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

from src.ml.feature_engineering import FeatureExtractor
from src.ml.pattern_classifier import PatternClassifier
from src.ml.model_validation import compute_mre_gap
from scripts.train_ml_pipeline_v3 import (
    load_data,
    generate_labels,
    extract_features,
    train_with_nested_purged_cv,
    DEFAULT_HORIZON,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def train_and_evaluate(
    dfs: dict[str, pd.DataFrame],
    use_wavelet: bool,
    horizon: int = DEFAULT_HORIZON,
    fast: bool = False,
    cv_method: str = "purged",
) -> dict[str, Any]:
    """Train CatBoost and evaluate with/without wavelet features.

    Returns dict with AUC, overfit_gap, MRE_gap, and feature count.
    """
    features = extract_features(dfs, use_cross_asset=False, use_wavelet=use_wavelet)

    X_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []
    for ticker, fx in features.items():
        fx_reset = fx.reset_index(drop=True)
        labels = generate_labels(dfs[ticker], horizon=horizon)
        labels.attrs = {}
        labels_reset = labels.reset_index(drop=True)
        common_idx = fx_reset.index.intersection(labels_reset.dropna().index)
        X_parts.append(fx_reset.loc[common_idx])
        y_parts.append(labels_reset.loc[common_idx])

    X_all = pd.concat(X_parts, ignore_index=True)
    y_all = pd.concat(y_parts, ignore_index=True)

    X_all = X_all.dropna(axis=1, how="all").fillna(0)
    valid_idx = ~(X_all.isna().any(axis=1))
    X_all = X_all[valid_idx]
    y_all = y_all[valid_idx]

    pos_pct = y_all.sum() / max(len(y_all), 1) * 100
    logger.info(
        "Model %s wavelet: %d samples, %d features, %.1f%% positive",
        "WITH" if use_wavelet else "WITHOUT",
        len(X_all),
        X_all.shape[1],
        pos_pct,
    )

    cv_results = train_with_nested_purged_cv(X_all, y_all, horizon=horizon, cv_method=cv_method)

    mean_test_auc = cv_results["mean_test_auc"]
    mean_train_auc = cv_results["mean_train_auc"]
    overfit_gap = cv_results["mean_overfit_gap"]
    mre_gap = compute_mre_gap(mean_train_auc, mean_test_auc) if mean_test_auc > 0 else 0.0

    return {
        "train_auc": mean_train_auc,
        "test_auc": mean_test_auc,
        "overfit_gap": overfit_gap,
        "mre_gap": mre_gap,
        "n_features": X_all.shape[1],
        "n_samples": len(X_all),
        "pos_pct": pos_pct,
        "cv_method": cv_method,
    }


def run_benchmark(
    tickers: list[str],
    start_is: str = "2016-01-01",
    end_is: str = "2021-12-31",
    start_oos: str = "2022-01-01",
    end_oos: str = "2026-05-01",
    horizon: int = DEFAULT_HORIZON,
    fast: bool = False,
    cv_method: str = "purged",
) -> dict[str, Any]:
    """Run full benchmark comparing wavelet vs baseline on IS and OOS data."""
    dfs_is: dict[str, pd.DataFrame] = {}
    dfs_oos: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df_full = load_data(ticker, start=start_is, end=end_oos)
        is_mask = (df_full.index >= start_is) & (df_full.index <= end_is)
        oos_mask = (df_full.index >= start_oos) & (df_full.index <= end_oos)
        dfs_is[ticker] = df_full[is_mask].copy()
        dfs_oos[ticker] = df_full[oos_mask].copy()
        logger.info(
            f"  {ticker}: IS {dfs_is[ticker].shape[0]} bars, OOS {dfs_oos[ticker].shape[0]} bars"
        )

    logger.info("=" * 80)
    logger.info("Benchmark: Wavelet Features vs Baseline")
    logger.info(f"Tickers: {tickers} | Horizon: {horizon}d | CV: {cv_method} | Fast: {fast}")
    logger.info(f"IS: {start_is} → {end_is} | OOS: {start_oos} → {end_oos}")
    logger.info("=" * 80)

    results: dict[str, Any] = {}

    logger.info("\n" + "=" * 60)
    logger.info("IN-SAMPLE (IS)")
    logger.info("=" * 60)
    label = "[1/4] Baseline (no wavelet)"
    logger.info(label)
    is_baseline = train_and_evaluate(
        dfs_is, use_wavelet=False, horizon=horizon, fast=fast, cv_method=cv_method
    )
    results["is_baseline"] = is_baseline

    logger.info("[2/4] Wavelet-augmented")
    is_wavelet = train_and_evaluate(
        dfs_is, use_wavelet=True, horizon=horizon, fast=fast, cv_method=cv_method
    )
    results["is_wavelet"] = is_wavelet

    logger.info("\n" + "=" * 60)
    logger.info("OUT-OF-SAMPLE (OOS)")
    logger.info("=" * 60)
    logger.info("[3/4] Baseline (no wavelet)")
    oos_baseline = train_and_evaluate(
        dfs_oos, use_wavelet=False, horizon=horizon, fast=fast, cv_method=cv_method
    )
    results["oos_baseline"] = oos_baseline

    logger.info("[4/4] Wavelet-augmented")
    oos_wavelet = train_and_evaluate(
        dfs_oos, use_wavelet=True, horizon=horizon, fast=fast, cv_method=cv_method
    )
    results["oos_wavelet"] = oos_wavelet

    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 80)

    print(
        f"\n{'Metric':<25} {'Baseline (IS)':>15} {'Wavelet (IS)':>15} {'Baseline (OOS)':>15} {'Wavelet (OOS)':>15}"
    )
    print("-" * 85)
    for metric in ["train_auc", "test_auc", "overfit_gap", "mre_gap", "n_features", "n_samples"]:
        values = [
            is_baseline[metric],
            is_wavelet[metric],
            oos_baseline[metric],
            oos_wavelet[metric],
        ]
        formatted = []
        for v in values:
            if isinstance(v, float):
                formatted.append(f"{v:15.4f}")
            elif isinstance(v, int):
                formatted.append(f"{v:15d}")
            else:
                formatted.append(f"{str(v):>15}")
        print(f"  {metric:<23} {' '.join(formatted)}")

    auc_delta_is = is_wavelet["test_auc"] - is_baseline["test_auc"]
    auc_delta_oos = oos_wavelet["test_auc"] - oos_baseline["test_auc"]
    mre_delta_is = is_baseline["mre_gap"] - is_wavelet["mre_gap"]
    mre_delta_oos = oos_baseline["mre_gap"] - oos_wavelet["mre_gap"]

    print(f"\n  IS AUC delta (wavelet - baseline): {auc_delta_is:+.4f}")
    print(f"  OOS AUC delta (wavelet - baseline): {auc_delta_oos:+.4f}")
    print(f"  IS MRE gap reduction: {mre_delta_is:+.4f}")
    print(f"  OOS MRE gap reduction: {mre_delta_oos:+.4f}")

    gate_passed = auc_delta_oos >= 0.01 or mre_delta_oos >= 0.05
    print(
        f"\n  Gate: {'PASSED' if gate_passed else 'FAILED'} "
        f"(dAUC OOS >= 0.01 or MRE gap reduction >= 0.05)"
    )

    results["auc_delta_is"] = auc_delta_is
    results["auc_delta_oos"] = auc_delta_oos
    results["mre_delta_is"] = mre_delta_is
    results["mre_delta_oos"] = mre_delta_oos
    results["gate_passed"] = gate_passed
    results["timestamp"] = datetime.now().isoformat()

    out_path = (
        Path("outputs") / "wavelet_benchmark" / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark wavelet features vs baseline CatBoost")
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument(
        "--basket", type=str, default=None, help="Comma-separated tickers (e.g., SPY,QQQ,GLD)"
    )
    parser.add_argument("--is-start", type=str, default="2016-01-01")
    parser.add_argument("--is-end", type=str, default="2021-12-31")
    parser.add_argument("--oos-start", type=str, default="2022-01-01")
    parser.add_argument("--oos-end", type=str, default="2026-05-01")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--cv-method", type=str, default="purged", choices=["purged", "cpcv"])
    args = parser.parse_args()

    tickers = [t.strip() for t in args.basket.split(",")] if args.basket else [args.symbol]

    run_benchmark(
        tickers=tickers,
        start_is=args.is_start,
        end_is=args.is_end,
        start_oos=args.oos_start,
        end_oos=args.oos_end,
        horizon=args.horizon,
        fast=args.fast,
        cv_method=args.cv_method,
    )


if __name__ == "__main__":
    main()
