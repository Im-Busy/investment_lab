"""
Enhanced ML Validation: Cross-Asset Features + Feature Selection vs Baseline

Compares three approaches:
1. Baseline: Original 81 SPY-only features
2. Cross-Asset: 81 + cross-asset features (VIX, bonds, sectors, oil, gold)
3. Feature-Selected: Cross-asset reduced via correlation + MI selection

Tests:
- Regime classification accuracy (holdout + walk-forward)
- Feature importance comparison
- Overfit gap comparison
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score, classification_report

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


def eval_holdout_rf(X, y, n_estimators=100, max_depth=5):
    from sklearn.ensemble import RandomForestClassifier

    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train.fillna(0), y_train)
    train_pred = model.predict(X_train.fillna(0))
    test_pred = model.predict(X_test.fillna(0))

    return {
        "train_acc": accuracy_score(y_train, train_pred),
        "test_acc": accuracy_score(y_test, test_pred),
        "overfit_gap": accuracy_score(y_train, train_pred) - accuracy_score(y_test, test_pred),
        "n_features": X_train.shape[1],
    }


def eval_walk_forward(X, y, n_folds=10):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.base import clone

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


def compare_baseline_vs_cross_asset(spy_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("COMPARISON: Baseline vs Cross-Asset Features")
    print("=" * 70)

    # Regime labels
    y = get_regime_labels(spy_df)
    print(f"\nRegime distribution: {y.value_counts().sort_index().to_dict()}")

    # Baseline features
    print("\nGenerating baseline (SPY-only) features...")
    engineer = FeatureEngineer()
    baseline_feats = engineer.generate_features(spy_df).select_dtypes(include=[np.number])
    baseline_feats = baseline_feats.ffill().bfill().dropna()
    y_base = y.reindex(baseline_feats.index).dropna()
    X_base = baseline_feats.reindex(y_base.index)
    print(f"  Baseline: {X_base.shape[1]} features, {X_base.shape[0]} samples")

    # Cross-asset features
    print("\nDownloading cross-asset data (VIX, TLT, IEF, GLD, USO, sectors)...")
    cross_asset = prepare_cross_asset_data(spy_df, start="2014-06-01", end="2024-12-31")
    print(f"  Downloaded: {list(cross_asset.keys())}")

    print("\nGenerating cross-asset features...")
    caf = CrossAssetFeatures()
    try:
        ca_feats = caf.generate_features(spy_df, cross_asset)
        combined = pd.concat([baseline_feats, ca_feats], axis=1)
        combined = combined.ffill().bfill().dropna()
        y_ca = y.reindex(combined.index).dropna()
        X_ca = combined.reindex(y_ca.index)
        print(f"  Combined: {X_ca.shape[1]} features, {X_ca.shape[0]} samples")
    except Exception as e:
        print(f"  Cross-asset feature generation failed: {e}")
        X_ca, y_ca = X_base, y_base

    # Feature selection on combined
    print(f"\nApplying feature selection (target=25, corr_threshold=0.85)...")
    selector = FeatureSelector(target_features=25, corr_threshold=0.85, random_state=42)
    result = selector.fit(X_ca.fillna(0), y_ca)
    X_sel = selector.transform(X_ca.fillna(0))
    y_sel = y_ca
    print(
        f"  Selected: {result.selected_features} features (removed {result.removed_variance} variance, {result.removed_correlation} correlation)"
    )

    # Holdout comparison
    print("\n--- HOLDOUT COMPARISON (70/30 time split, RF 100/5) ---")

    baseline = eval_holdout_rf(X_base, y_base)
    cross_asset = eval_holdout_rf(X_ca, y_ca)
    selected = eval_holdout_rf(X_sel, y_sel)

    print(f"  {'Model':<25s} {'Features':>8s} {'Train':>8s} {'Test':>8s} {'Gap':>8s}")
    print(
        f"  {'Baseline (SPY-only)':<25s} {baseline['n_features']:>8d} {baseline['train_acc']:>8.4f} {baseline['test_acc']:>8.4f} {baseline['overfit_gap']:>+8.4f}"
    )
    print(
        f"  {'Cross-Asset (full)':<25s} {cross_asset['n_features']:>8d} {cross_asset['train_acc']:>8.4f} {cross_asset['test_acc']:>8.4f} {cross_asset['overfit_gap']:>+8.4f}"
    )
    print(
        f"  {'Feature-Selected (25)':<25s} {selected['n_features']:>8d} {selected['train_acc']:>8.4f} {selected['test_acc']:>8.4f} {selected['overfit_gap']:>+8.4f}"
    )

    # Walk-forward comparison
    print(f"\n--- WALK-FORWARD COMPARISON ---")

    wf_base = eval_walk_forward(X_base, y_base)
    wf_ca = eval_walk_forward(X_ca, y_ca)
    wf_sel = eval_walk_forward(X_sel, y_sel)

    print(f"  {'Model':<25s} {'WF Folds':>8s} {'WF Train':>8s} {'WF Test':>8s} {'WF Std':>8s}")
    print(
        f"  {'Baseline (SPY-only)':<25s} {wf_base['n_folds']:>8d} {wf_base['mean_train']:>8.4f} {wf_base['mean_test']:>8.4f} {wf_base['std_test']:>8.4f}"
    )
    print(
        f"  {'Cross-Asset (full)':<25s} {wf_ca['n_folds']:>8d} {wf_ca['mean_train']:>8.4f} {wf_ca['mean_test']:>8.4f} {wf_ca['std_test']:>8.4f}"
    )
    print(
        f"  {'Feature-Selected (25)':<25s} {wf_sel['n_folds']:>8d} {wf_sel['mean_train']:>8.4f} {wf_sel['mean_test']:>8.4f} {wf_sel['std_test']:>8.4f}"
    )

    # Top selected features
    print(f"\n--- TOP 25 SELECTED FEATURES (MI Score) ---")
    rankings = selector.get_feature_rankings(top_n=25)
    for i, row in rankings.iterrows():
        print(f"  {i + 1:>2d}. {row['feature']:<25s} {row['mi_score']:.4f}")

    return {
        "baseline": {"holdout": baseline, "walk_forward": wf_base},
        "cross_asset": {"holdout": cross_asset, "walk_forward": wf_ca},
        "selected": {"holdout": selected, "walk_forward": wf_sel, "selector": selector},
    }


def main():
    print("=" * 70)
    print("ENHANCED ML VALIDATION: CROSS-ASSET + FEATURE SELECTION")
    print("=" * 70)

    spy_df = download_spy()
    results = compare_baseline_vs_cross_asset(spy_df)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    wf_base = results["baseline"]["walk_forward"]
    wf_sel = results["selected"]["walk_forward"]

    print(f"  Baseline WF test accuracy:     {wf_base['mean_test']:.4f}")
    print(f"  Feature-Selected WF accuracy:   {wf_sel['mean_test']:.4f}")
    print(f"  Improvement:                    {wf_sel['mean_test'] - wf_base['mean_test']:+.4f}")

    if wf_sel["mean_test"] - wf_base["mean_test"] > 0.02:
        print(
            f"\n  [SUCCESS] Feature selection with cross-asset data shows meaningful improvement! (>2pp)"
        )
    else:
        print(
            f"\n  [INFO] Cross-asset + feature selection shows marginal change. Further work needed."
        )


if __name__ == "__main__":
    main()
