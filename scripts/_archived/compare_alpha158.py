"""Compare Alpha158 factors vs V3 features in CatBoost pipeline.

Evaluates three feature sets using identical nested PurgedKFold CV:
  - V3 only (88 instrument features) — baseline
  - Alpha158 only (158 Qlib factors)
  - Combined (88 + 158)

Reports rank IC, AUC, overfit gap for each.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# --- Path setup ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.feature_engineering import FeatureExtractor
from src.ml.triple_barrier import TripleBarrierLabeler
from src.ml.purged_cv import PurgedKFold
from src.ml.pattern_classifier import PatternClassifier


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOL = "SPY"
START = "2015-01-01"
END = "2024-12-31"
HORIZON = 5
N_CV_OUTER = 5
N_CV_INNER = 3
PCT_EMBARGO = 0.05
MIN_ABS_IC = 0.02  # threshold used by filter_features_by_ic

PARAM_SETS = [
    {"max_depth": 3, "l2_leaf_reg": 10.0, "random_strength": 3.0, "min_data_in_leaf": 50},
    {"max_depth": 4, "l2_leaf_reg": 5.0, "random_strength": 2.0, "min_data_in_leaf": 30},
    {"max_depth": 3, "l2_leaf_reg": 20.0, "random_strength": 5.0, "min_data_in_leaf": 80},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def compute_rank_ic(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Spearman rank correlation between predictions and actuals."""
    sr = stats.spearmanr(predictions, actuals)
    return float(sr.statistic)


def filter_by_ic(X: pd.DataFrame, y: pd.Series, min_abs_ic: float = MIN_ABS_IC) -> pd.DataFrame:
    """Drop features whose |rank IC| is below threshold."""
    keep: list[str] = []
    for col in X.columns:
        valid = X[col].notna() & y.notna()
        if valid.sum() < 30:
            continue
        ic, _ = stats.spearmanr(X.loc[valid, col], y.loc[valid])
        if abs(ic) >= min_abs_ic:
            keep.append(col)
    dropped = len(X.columns) - len(keep)
    if dropped > 0:
        print(f"  IC filter: dropped {dropped} features (|IC| < {min_abs_ic}), kept {len(keep)}")
    return X[keep]


def nested_cv_eval(
    X: pd.DataFrame,
    y: pd.Series,
    label: str,
) -> dict[str, Any]:
    """Run nested PurgedKFold CV and return aggregate metrics."""
    cv_outer = PurgedKFold(n_splits=N_CV_OUTER, pct_embargo=PCT_EMBARGO, label_span=HORIZON)

    outer_results: list[dict] = []
    all_preds: list[float] = []
    all_actuals: list[int] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_outer.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Inner CV for param selection
        cv_inner = PurgedKFold(n_splits=N_CV_INNER, pct_embargo=PCT_EMBARGO, label_span=HORIZON)
        best_params = PARAM_SETS[0]
        best_score = -1.0

        for params in PARAM_SETS:
            inner_scores: list[float] = []
            for it_idx, iv_idx in cv_inner.split(X_train):
                X_it, X_iv = X_train.iloc[it_idx], X_train.iloc[iv_idx]
                y_it, y_iv = y_train.iloc[it_idx], y_train.iloc[iv_idx]

                clf = PatternClassifier(
                    model_type="catboost",
                    n_estimators=100,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    **params,
                )
                result = clf.train(X_it, y_it, calibration_data=(X_iv, y_iv))
                inner_scores.append(result.test_auc)

            mean_score = np.mean(inner_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        # Outer fold with best params
        clf = PatternClassifier(
            model_type="catboost",
            n_estimators=100,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            **best_params,
        )
        result = clf.train(X_train, y_train, calibration_data=(X_test, y_test))
        overfit_gap = result.train_auc - result.test_auc

        # Predictions for rank IC
        y_pred_df = clf.predict(X_test)
        y_pred_arr = y_pred_df["probability_profitable"].values
        all_preds.extend(y_pred_arr.tolist())
        all_actuals.extend(y_test.tolist())

        outer_results.append(
            {
                "fold": fold_idx,
                "train_auc": result.train_auc,
                "test_auc": result.test_auc,
                "test_accuracy": result.test_accuracy,
                "calibration_error": result.calibration_error,
                "overfit_gap": overfit_gap,
            }
        )

    test_aucs = [r["test_auc"] for r in outer_results]
    train_aucs = [r["train_auc"] for r in outer_results]
    gaps = [r["overfit_gap"] for r in outer_results]
    rank_ic = compute_rank_ic(np.array(all_preds), np.array(all_actuals))

    return {
        "label": label,
        "n_features": X.shape[1],
        "n_samples": len(y),
        "mean_train_auc": float(np.mean(train_aucs)),
        "mean_test_auc": float(np.mean(test_aucs)),
        "std_test_auc": float(np.std(test_aucs)),
        "mean_overfit_gap": float(np.mean(gaps)),
        "rank_ic": rank_ic,
        "fold_results": outer_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ---- 1. Load SPY data ----
    data_path = Path(f"data/raw/{SYMBOL}_daily.csv")
    if data_path.exists():
        df = pd.read_csv(data_path, parse_dates=True, index_col=0)
    else:
        import yfinance as yf

        df = yf.download(SYMBOL, start=START, end=END, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0 if col == "Volume" else df["Close"]

    # ---- 2. Generate V3 features ----
    extractor = FeatureExtractor()
    features_v3 = extractor.extract_all_features(df, include_forward_returns=False)
    print(f"V3 features: {features_v3.shape[1]} cols x {features_v3.shape[0]} rows")

    # ---- 3. Generate labels (triple barrier, same as V3) ----
    labeler = TripleBarrierLabeler(atr_mult_tp=1.5, atr_mult_sl=1.0)
    atr = _compute_atr(df)
    raw_labels = labeler.fit(
        close=df["Close"],
        high=df["High"],
        low=df["Low"],
        take_profit=None,
        stop_loss=None,
        time_limit=HORIZON,
        atr_series=atr,
    )
    y = (raw_labels == 1).astype(int)
    pos_pct = y.sum() / max(len(y), 1) * 100
    print(f"Labels: {y.sum()} positive / {len(y)} total ({pos_pct:.1f}%)")

    # Align features to labels (labels may be shorter due to horizon truncation)
    common_idx = features_v3.index.intersection(y.index)
    features_v3 = features_v3.loc[common_idx]
    y = y.loc[common_idx]
    print(f"After alignment: {len(common_idx)} samples")

    # ---- 4. Load Alpha158 features ----
    alpha_path = Path("data/qlib_alpha158_raw.parquet")
    if not alpha_path.exists():
        print("ERROR: Alpha158 parquet not found. Run scripts/extract_alpha158.py first.")
        sys.exit(1)

    alpha_raw = pd.read_parquet(alpha_path)
    # Flatten multi-index: keep date, drop instrument
    alpha_raw.index = alpha_raw.index.get_level_values("datetime")
    # Flatten column multi-level: drop 'feature'/'label' prefix, keep column name
    alpha_raw.columns = alpha_raw.columns.get_level_values(1)

    # Drop Qlib's label column (we use V3 triple-barrier labels)
    if "LABEL0" in alpha_raw.columns:
        alpha_raw = alpha_raw.drop(columns=["LABEL0"])

    # Drop VWAP0 (all NaN)
    if "VWAP0" in alpha_raw.columns:
        alpha_raw = alpha_raw.drop(columns=["VWAP0"])

    print(f"Alpha158 raw: {alpha_raw.shape[1]} cols x {alpha_raw.shape[0]} rows")

    # ---- 5. Merge on date ----
    combined = features_v3.join(alpha_raw, how="inner")
    # Drop rows with remaining NaN in features
    combined = combined.dropna()
    y_combined = y.loc[combined.index]

    # Also clip Alpha158-only to same index
    alpha_clipped = alpha_raw.loc[combined.index]
    # Forward-fill any small gaps then drop remaining
    alpha_clipped = alpha_clipped.ffill().dropna()
    # Re-align
    idx = combined.index.intersection(alpha_clipped.index)
    combined = combined.loc[idx]
    alpha_clipped = alpha_clipped.loc[idx]
    y_combined = y_combined.loc[idx]

    # V3-only on same index for fair comparison
    v3_only = features_v3.loc[idx]
    v3_only = v3_only.dropna(axis=1).fillna(0)  # drop cols with any NaN

    print(f"\nFinal aligned samples: {len(idx)}")
    print(f"  V3-only features:    {v3_only.shape[1]}")
    print(f"  Alpha158-only features: {alpha_clipped.shape[1]}")
    print(f"  Combined features:   {combined.shape[1]}")
    print(f"  Positive label rate: {y_combined.sum() / len(y_combined) * 100:.1f}%")

    # ---- 6. IC-filter each feature set ----
    print("\n--- IC Filter ---")
    v3_only = filter_by_ic(v3_only, y_combined)
    alpha_clipped = filter_by_ic(alpha_clipped, y_combined)
    combined = filter_by_ic(combined, y_combined)
    # Re-drop any inf/nan introduced
    v3_only = v3_only.replace([np.inf, -np.inf], np.nan).fillna(0)
    alpha_clipped = alpha_clipped.replace([np.inf, -np.inf], np.nan).fillna(0)
    combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ---- 7. Run nested CV for each feature set ----
    results: list[dict] = []

    for X, label in [
        (v3_only, "V3 (baseline)"),
        (alpha_clipped, "Alpha158 only"),
        (combined, "Combined"),
    ]:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {label} ({X.shape[1]} features)")
        print(f"{'=' * 60}")
        r = nested_cv_eval(X, y_combined, label)
        results.append(r)
        print(
            f"  AUC: train={r['mean_train_auc']:.4f}, test={r['mean_test_auc']:.4f} ± {r['std_test_auc']:.4f}"
        )
        print(f"  Overfit gap: {r['mean_overfit_gap']:.4f}")
        print(f"  Rank IC:     {r['rank_ic']:.4f}")

    # ---- 8. Summary ----
    print(f"\n{'=' * 80}")
    print("FINAL COMPARISON vs V3 Baseline (IC=0.182, OOS Sharpe=-0.44)")
    print(f"{'=' * 80}")
    print(
        f"{'Feature Set':<22} {'N Feat':>7} {'Train AUC':>10} {'Test AUC':>10} {'Overfit Gap':>12} {'Rank IC':>9}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['label']:<22} {r['n_features']:>7} "
            f"{r['mean_train_auc']:>10.4f} {r['mean_test_auc']:>10.4f} "
            f"{r['mean_overfit_gap']:>12.4f} {r['rank_ic']:>9.4f}"
        )
    print("-" * 80)

    baseline_ic = 0.182
    for r in results:
        delta = r["rank_ic"] - baseline_ic
        direction = "BETTER" if delta > 0 else "WORSE"
        print(f"  {r['label']}: IC delta = {delta:+.4f} ({direction} than V3 baseline)")

    # ---- 9. Save results JSON ----
    import json

    out = {
        "config": {
            "symbol": SYMBOL,
            "start": START,
            "end": END,
            "horizon": HORIZON,
            "cv_outer": N_CV_OUTER,
            "cv_inner": N_CV_INNER,
            "pct_embargo": PCT_EMBARGO,
            "baseline_ic": baseline_ic,
            "baseline_oos_sharpe": -0.44,
        },
        "results": results,
    }
    out_path = Path("experiments/alpha158_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
