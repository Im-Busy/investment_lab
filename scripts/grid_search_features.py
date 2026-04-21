"""
Grid Search: Optimal Feature Count and Correlation Threshold

Tests combinations of:
- target_features: [10, 15, 20, 25, 30, 40]
- corr_threshold: [0.75, 0.80, 0.85, 0.90]

For each combination:
1. Select features using FeatureSelector
2. Train RandomForest with walk-forward validation
3. Record accuracy

Outputs: Accuracy table and plot of accuracy vs feature count
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indicators.regime_detector import RegimeDetector
from src.ml.cross_asset_features import CrossAssetFeatures, prepare_cross_asset_data
from src.ml.feature_selector import FeatureSelector
from src.ml.features import FeatureEngineer

warnings.filterwarnings("ignore")


def download_spy() -> pd.DataFrame:
    print("Downloading SPY data...")
    df = yf.download("SPY", start="2015-01-01", end="2024-12-31", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "date"
    return df.sort_index()


def get_regime_labels(df: pd.DataFrame) -> pd.Series:
    detector = RegimeDetector()
    result = detector.get_regime_series(df)
    return result["regime"].apply(lambda r: r.value)


def eval_walk_forward(X, y, n_folds=10):
    train_size = 300
    step_size = max(50, (len(X) - train_size - 50) // n_folds)

    test_scores = []
    train_scores = []
    start = 0

    while start + train_size + 50 < len(X):
        end = min(start + train_size + step_size, len(X))
        X_train = X.iloc[start : start + train_size]
        X_test = X.iloc[start + train_size : end]
        y_train = y.iloc[start : start + train_size]
        y_test = y.iloc[start + train_size : end]

        if len(X_test) < 20:
            start += step_size
            continue

        valid_train = X_train.notna().all(axis=1) & y_train.notna()
        valid_test = X_test.notna().all(axis=1) & y_test.notna()

        if valid_train.sum() < 50 or valid_test.sum() < 20:
            start += step_size
            continue

        model = clone(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                class_weight="balanced",
            )
        )
        model.fit(X_train[valid_train].fillna(0), y_train[valid_train])

        train_pred = model.predict(X_train[valid_train].fillna(0))
        test_pred = model.predict(X_test[valid_test].fillna(0))

        train_scores.append(accuracy_score(y_train[valid_train], train_pred))
        test_scores.append(accuracy_score(y_test[valid_test], test_pred))

        start += step_size

    return {
        "mean_train": np.mean(train_scores) if train_scores else 0,
        "mean_test": np.mean(test_scores) if test_scores else 0,
        "std_test": np.std(test_scores) if test_scores else 0,
        "n_folds": len(test_scores),
    }


def main():
    print("=" * 70)
    print("GRID SEARCH: Optimal Feature Count + Correlation Threshold")
    print("=" * 70)

    spy_df = download_spy()
    y = get_regime_labels(spy_df)

    engineer = FeatureEngineer()
    baseline_feats = engineer.generate_features(spy_df).select_dtypes(include=[np.number])
    baseline_feats = baseline_feats.ffill().bfill().dropna()
    y_base = y.reindex(baseline_feats.index).dropna()
    X_base = baseline_feats.reindex(y_base.index)
    print(f"\nBaseline: {X_base.shape[1]} features, {X_base.shape[0]} samples")

    print("\nDownloading cross-asset data...")
    cross_asset = prepare_cross_asset_data(spy_df, start="2014-06-01", end="2024-12-31")
    print(f"  Downloaded: {list(cross_asset.keys())}")

    print("\nGenerating cross-asset features...")
    caf = CrossAssetFeatures()
    ca_feats = caf.generate_features(spy_df, cross_asset)
    combined = pd.concat([baseline_feats, ca_feats], axis=1)
    combined = combined.ffill().bfill().dropna()
    y_ca = y.reindex(combined.index).dropna()
    X_ca = combined.reindex(y_ca.index)
    print(f"  Combined: {X_ca.shape[1]} features, {X_ca.shape[0]} samples")

    target_features_list = [10, 15, 20, 25, 30, 40]
    corr_threshold_list = [0.75, 0.80, 0.85, 0.90]

    results = []

    total_combos = len(target_features_list) * len(corr_threshold_list)
    combo = 0

    for corr_t in corr_threshold_list:
        for target_n in target_features_list:
            combo += 1
            print(f"\n[{combo}/{total_combos}] target_features={target_n}, corr_threshold={corr_t}")

            selector = FeatureSelector(
                target_features=target_n,
                corr_threshold=corr_t,
                random_state=42,
            )
            selector.fit(X_ca.fillna(0), y_ca)
            X_sel = selector.transform(X_ca.fillna(0))

            wf = eval_walk_forward(X_sel, y_ca)

            n_features_after_corr = X_sel.shape[1]
            actual_selected = min(target_n, n_features_after_corr)

            results.append(
                {
                    "target_features": target_n,
                    "corr_threshold": corr_t,
                    "actual_features": actual_selected,
                    "wf_train": wf["mean_train"],
                    "wf_test": wf["mean_test"],
                    "wf_std": wf["std_test"],
                    "n_folds": wf["n_folds"],
                }
            )

            print(
                f"  -> WF Test: {wf['mean_test']:.4f} ± {wf['std_test']:.4f} ({wf['n_folds']} folds)"
            )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("GRID SEARCH RESULTS")
    print("=" * 70)

    print("\n--- By Correlation Threshold ---")
    for corr_t in corr_threshold_list:
        subset = results_df[results_df["corr_threshold"] == corr_t]
        print(f"\n  corr_threshold = {corr_t}:")
        print(
            f"    {'Target N':>10s} {'Actual N':>10s} {'WF Train':>10s} {'WF Test':>10s} {'WF Std':>10s}"
        )
        for _, row in subset.iterrows():
            print(
                f"    {int(row['target_features']):>10d} "
                f"{int(row['actual_features']):>10d} "
                f"{row['wf_train']:>10.4f} "
                f"{row['wf_test']:>10.4f} "
                f"{row['wf_std']:>10.4f}"
            )

    print("\n--- BEST RESULT ---")
    best_idx = results_df["wf_test"].idxmax()
    best = results_df.loc[best_idx]
    print(
        f"  target_features={int(best['target_features'])}, "
        f"corr_threshold={best['corr_threshold']}, "
        f"actual_features={int(best['actual_features'])}"
    )
    print(f"  WF Test Accuracy: {best['wf_test']:.4f} ± {best['wf_std']:.4f}")
    print(f"  WF Train Accuracy: {best['wf_train']:.4f}")
    print(f"  Overfit Gap: {best['wf_train'] - best['wf_test']:+.4f}")

    baseline_wf_test = 0.4697
    improvement = best["wf_test"] - baseline_wf_test
    print(f"\n  Improvement vs Baseline (46.97%): {improvement:+.4f} ({improvement * 100:+.2f}pp)")

    return results_df, best


if __name__ == "__main__":
    main()
