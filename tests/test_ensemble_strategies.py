"""
Test All 4 Ensemble Strategies

Compares:
1. Rule-Based alone (baseline)
2. ML alone (RandomForest with best feature selection)
3. Ensemble: Weighted
4. Ensemble: RuleVeto
5. Ensemble: Consensus
6. Ensemble: TimeDecay

For each, measures accuracy vs rule-based ground truth.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indicators.regime_detector import RegimeDetector
from src.ml.cross_asset_features import CrossAssetFeatures, prepare_cross_asset_data
from src.ml.feature_selector import FeatureSelector
from src.ml.features import FeatureEngineer
from src.ml.ensemble_regime import EnsembleRegimeDetector, EnsembleStrategy

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


def eval_walk_forward_ensemble(X_base, X_sel, y_full, spy_df, n_folds=10):
    """
    Walk-forward test comparing 6 approaches per fold:
    1. Rule-Based (ground truth)
    2. ML only
    3. Ensemble: Weighted
    4. Ensemble: RuleVeto
    5. Ensemble: Consensus
    6. Ensemble: TimeDecay

    ML is trained on the training fold and tested on test fold.
    Ensemble combines ML predictions with rule-based for test fold.
    """
    train_size = 300
    step_size = max(50, (len(X_base) - train_size - 50) // n_folds)

    results_by_method = {
        "rule_based": [],
        "ml_only": [],
        "ensemble_weighted": [],
        "ensemble_rule_veto": [],
        "ensemble_consensus": [],
        "ensemble_time_decay": [],
    }

    start = 0
    fold = 0

    while start + train_size + 50 < len(X_base):
        end = min(start + train_size + step_size, len(X_base))

        X_base_train = X_base.iloc[start : start + train_size]
        X_sel_train = X_sel.iloc[start : start + train_size]
        X_base_test = X_base.iloc[start + train_size : end]
        X_sel_test = X_sel.iloc[start + train_size : end]
        y_train = y_full.iloc[start : start + train_size]
        y_test = y_full.iloc[start + train_size : end]

        if len(X_base_test) < 20:
            start += step_size
            continue

        valid_train = X_base_train.notna().all(axis=1) & y_train.notna()
        valid_test = X_base_test.notna().all(axis=1) & y_test.notna()

        if valid_train.sum() < 50 or valid_test.sum() < 20:
            start += step_size
            continue

        train_idx = X_base_train[valid_train].index
        test_idx = X_base_test[valid_test].index

        spy_train = spy_df.loc[train_idx]
        spy_test_full = spy_df.loc[test_idx]

        ml_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight="balanced",
        )
        ml_model.fit(X_sel_train[valid_train].fillna(0), y_train[valid_train])

        y_test_true = y_test[valid_test]

        ml_pred = pd.Series(
            ml_model.predict(X_sel_test[valid_test].fillna(0)),
            index=test_idx,
        )

        ml_proba = None
        try:
            proba_array = ml_model.predict_proba(X_sel_test[valid_test].fillna(0))
            classes = ml_model.classes_
            ml_proba = pd.DataFrame(proba_array, columns=classes, index=test_idx)
        except Exception:
            pass

        rule_pred = get_regime_labels(spy_test_full).reindex(test_idx)

        rule_acc = (rule_pred == y_test_true).mean()
        ml_acc = (ml_pred == y_test_true).mean()

        ensemble_accs = {}
        for strategy_name, strategy in [
            ("ensemble_weighted", EnsembleStrategy.WEIGHTED),
            ("ensemble_rule_veto", EnsembleStrategy.RULE_VETO),
            ("ensemble_consensus", EnsembleStrategy.CONSENSUS),
            ("ensemble_time_decay", EnsembleStrategy.TIME_DECAY),
        ]:
            detector = RegimeDetector()

            ens = EnsembleRegimeDetector(
                rule_detector=detector,
                ml_classifier=ml_model,
                strategy=strategy,
                ml_weight=0.4,
                consensus_threshold=0.7,
            )

            try:
                ens_pred = ens.predict(spy_test_full)
                common = ens_pred.index.intersection(y_test_true.index)
                ens_acc = (ens_pred[common] == y_test_true[common]).mean()
                ensemble_accs[strategy_name] = ens_acc
            except Exception:
                ensemble_accs[strategy_name] = np.nan

        results_by_method["rule_based"].append(rule_acc)
        results_by_method["ml_only"].append(ml_acc)
        results_by_method["ensemble_weighted"].append(
            ensemble_accs.get("ensemble_weighted", np.nan)
        )
        results_by_method["ensemble_rule_veto"].append(
            ensemble_accs.get("ensemble_rule_veto", np.nan)
        )
        results_by_method["ensemble_consensus"].append(
            ensemble_accs.get("ensemble_consensus", np.nan)
        )
        results_by_method["ensemble_time_decay"].append(
            ensemble_accs.get("ensemble_time_decay", np.nan)
        )

        start += step_size
        fold += 1

    summary = {}
    for method, scores in results_by_method.items():
        scores = [s for s in scores if not np.isnan(s)]
        if scores:
            summary[method] = {
                "mean_test": np.mean(scores),
                "std_test": np.std(scores),
                "n_folds": len(scores),
            }
        else:
            summary[method] = {
                "mean_test": np.nan,
                "std_test": np.nan,
                "n_folds": 0,
            }

    return summary


def main():
    print("=" * 70)
    print("ENSEMBLE STRATEGY COMPARISON")
    print("=" * 70)

    spy_df = download_spy()
    y = get_regime_labels(spy_df)

    engineer = FeatureEngineer()
    baseline_feats = engineer.generate_features(spy_df).select_dtypes(include=[np.number])
    baseline_feats = baseline_feats.ffill().bfill().dropna()
    y_base = y.reindex(baseline_feats.index).dropna()
    X_base = baseline_feats.reindex(y_base.index)

    print("\nDownloading cross-asset data...")
    cross_asset = prepare_cross_asset_data(spy_df, start="2014-06-01", end="2024-12-31")

    print("Generating cross-asset features...")
    caf = CrossAssetFeatures()
    ca_feats = caf.generate_features(spy_df, cross_asset)
    combined = pd.concat([baseline_feats, ca_feats], axis=1)
    combined = combined.ffill().bfill().dropna()
    y_ca = y.reindex(combined.index).dropna()
    X_ca = combined.reindex(y_ca.index)

    print("\nApplying best feature selection (target=25, corr=0.85)...")
    selector = FeatureSelector(target_features=25, corr_threshold=0.85, random_state=42)
    selector.fit(X_ca.fillna(0), y_ca)
    X_sel = selector.transform(X_ca.fillna(0))

    print("\nRunning walk-forward ensemble comparison...")
    summary = eval_walk_forward_ensemble(X_base, X_sel, y_ca, spy_df)

    print("\n" + "=" * 70)
    print("ENSEMBLE RESULTS")
    print("=" * 70)

    print(f"\n  {'Method':<25s} {'WF Mean':>8s} {'WF Std':>8s} {'Folds':>6s}")

    for method, stats in summary.items():
        name_map = {
            "rule_based": "Rule-Based (baseline)",
            "ml_only": "ML Only (RF)",
            "ensemble_weighted": "Ensemble: Weighted",
            "ensemble_rule_veto": "Ensemble: RuleVeto",
            "ensemble_consensus": "Ensemble: Consensus",
            "ensemble_time_decay": "Ensemble: TimeDecay",
        }
        label = name_map.get(method, method)
        print(
            f"  {label:<25s} {stats['mean_test']:>8.4f} {stats['std_test']:>8.4f} {stats['n_folds']:>6d}"
        )

    print("\n--- KEY INSIGHTS ---")
    ml_std = summary["ml_only"]["std_test"]
    for method in [
        "ensemble_weighted",
        "ensemble_rule_veto",
        "ensemble_consensus",
        "ensemble_time_decay",
    ]:
        ens_std = summary[method]["std_test"]
        ens_mean = summary[method]["mean_test"]
        ml_mean = summary["ml_only"]["mean_test"]
        if ens_std < ml_std:
            print(f"  {method}: More stable (std {ens_std:.4f} vs {ml_std:.4f})")
        else:
            print(f"  {method}: Less stable (std {ens_std:.4f} vs {ml_std:.4f})")

        if ens_mean > ml_mean:
            print(f"  {method}: Higher accuracy ({ens_mean:.4f} vs {ml_mean:.4f})")


if __name__ == "__main__":
    main()
