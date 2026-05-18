"""
P2-2: Feature Importance Ranking Comparison.

Compares CatBoost MDI, SHAP, and stability selected features to check
if the model's signal is robust across importance methods.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.stats import spearmanr


def main() -> None:
    MODEL_PATH = "models/pattern_classifier_v3_SPY_20260514_124612.pkl"
    SHAP_PATH = "experiments/v3_SPY_20260514_124612/shap_report.json"

    # Load model data
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    feature_names = model_data["feature_names"]
    catboost_model = model_data["model"]

    # 1. CatBoost MDI importance
    mdi_raw = catboost_model.get_feature_importance()
    mdi = dict(zip(feature_names, mdi_raw))

    # 2. SHAP importance
    with open(SHAP_PATH) as f:
        shap_data = json.load(f)
    shap_importance = {item["feature"]: item["mean_abs_shap"] for item in shap_data["top_features"]}
    # Fill missing SHAP values with 0
    for f in feature_names:
        if f not in shap_importance:
            shap_importance[f] = 0.0

    # 3. Get ranked lists
    mdi_ranked = sorted(mdi.items(), key=lambda x: -x[1])
    shap_ranked = sorted(shap_importance.items(), key=lambda x: -x[1])

    # Rank correlation
    mdi_vals = np.array([mdi[f] for f in feature_names])
    shap_vals = np.array([shap_importance[f] for f in feature_names])

    rho, p_val = spearmanr(mdi_vals, shap_vals)
    print("=" * 80)
    print("Feature Importance Ranking Comparison")
    print("=" * 80)
    print(f"Spearman rank correlation (MDI vs SHAP): rho={rho:.4f}, p={p_val:.4f}")
    print()

    if rho > 0.7:
        print("AGREEMENT: MDI and SHAP rankings are strongly correlated.")
        print("The model's signal is robust across importance methods.")
    elif rho > 0.3:
        print("MODERATE: MDI and SHAP rankings show moderate agreement.")
        print("Some features have divergent importance across methods.")
    else:
        print("DISAGREEMENT: MDI and SHAP rankings diverge significantly.")
        print("The model's signal may be fragile.")

    print()
    print("Top 10 by CatBoost MDI:")
    for i, (f, v) in enumerate(mdi_ranked[:10]):
        print(f"  {i + 1:>2}. {f:<30} {v:>8.4f}")

    print()
    print("Top 10 by SHAP (mean_abs_shap):")
    for i, (f, v) in enumerate(shap_ranked[:15]):
        if f in feature_names:
            print(f"  {i + 1:>2}. {f:<30} {v:>8.4f}")

    # Overlap analysis
    mdi_top10 = set(dict(mdi_ranked[:10]).keys())
    shap_top10 = set(dict(shap_ranked[:10]).keys())
    overlap = mdi_top10 & shap_top10
    print(f"\nOverlap in Top 10: {len(overlap)}/10")
    if len(overlap) < 10:
        only_mdi = mdi_top10 - shap_top10
        only_shap = shap_top10 - mdi_top10
        if only_mdi:
            print(f"  Only in MDI top 10: {only_mdi}")
        if only_shap:
            print(f"  Only in SHAP top 10: {only_shap}")

    # Jaccard at top 5, 10, 15
    for k in [5, 10, 15]:
        mdi_k = set(dict(mdi_ranked[:k]).keys())
        shap_k = set(dict(shap_ranked[:k]).keys())
        jaccard = len(mdi_k & shap_k) / len(mdi_k | shap_k)
        print(f"  Jaccard at top {k}: {jaccard:.2%}")

    # Check: are cross-asset features dominating?
    print()
    print("Cross-asset feature check:")
    cross_asset = [
        f
        for f in feature_names
        if any(prefix in f for prefix in ["rel_ret_", "beta_", "resid_vol_", "SPY_", "breadth_"])
    ]
    mdi_cross = sum(mdi.get(f, 0) for f in cross_asset)
    shap_cross = sum(shap_importance.get(f, 0) for f in cross_asset)
    mdi_total = sum(mdi.values())
    shap_total = sum(shap_importance.values())
    print(f"  {len(cross_asset)} cross-asset features")
    print(f"  MDI cross-asset share: {mdi_cross / mdi_total:.1%} of total importance")
    print(f"  SHAP cross-asset share: {shap_cross / (shap_total + 1e-10):.1%} of total importance")
    if cross_asset:
        print(
            f"  Top cross-asset (MDI): {', '.join(sorted(cross_asset, key=lambda f: -mdi.get(f, 0.0))[:3])}"
        )
        print(
            f"  Top cross-asset (SHAP): {', '.join(sorted(cross_asset, key=lambda f: -shap_importance.get(f, 0.0))[:3])}"
        )


if __name__ == "__main__":
    main()
