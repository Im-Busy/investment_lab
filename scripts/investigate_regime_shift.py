"""Regime shift investigation: KS test on features IS vs OOS.

Answers: did 2025-2026 market dynamics break the model?
Checks: feature distribution shifts AND feature->label relationship shifts.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  # noqa: E402

from src.ml.feature_engineering import FeatureExtractor  # noqa: E402
from src.ml.pattern_classifier import PatternClassifier  # noqa: E402
from src.ml.triple_barrier import TripleBarrierLabeler  # noqa: E402

MODEL_PATH = "models/pattern_classifier_v3_SPY_20260511_224704.pkl"
DATA_PATH = "data/raw/SPY_daily.csv"
HORIZON = 5
OUTPUT_DIR = Path("reports/calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SHAP top features from metadata
TOP_FEATURES = [
    "vol_regime",
    "dist_to_low_10",
    "volatility_20",
    "bb_width_20",
    "vol_regime_state",
    "price_to_ma_10",
    "volatility_regime",
    "price_to_ma_100",
    "dist_to_low_50",
    "close_to_vwap_10",
]

IS_END = "2025-01-01"
OOS_START = "2025-01-01"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0.0 if col == "Volume" else df.iloc[:, 0]
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def ks_test_is_vs_oos(is_vals: np.ndarray, oos_vals: np.ndarray) -> tuple[float, float]:
    """Two-sample KS test. Returns (statistic, p-value)."""
    clean_is = is_vals[~np.isnan(is_vals)]
    clean_oos = oos_vals[~np.isnan(oos_vals)]
    if len(clean_is) < 10 or len(clean_oos) < 10:
        return np.nan, np.nan
    return stats.ks_2samp(clean_is, clean_oos)


def feature_target_corr_change(feature_is, y_is, feature_oos, y_oos, method="pearson"):
    """Compute correlation change between feature and target IS vs OOS."""
    mask_is = ~(np.isnan(feature_is) | np.isnan(y_is))
    mask_oos = ~(np.isnan(feature_oos) | np.isnan(y_oos))

    if mask_is.sum() < 10 or mask_oos.sum() < 10:
        return np.nan, np.nan, np.nan

    corr_is = stats.pearsonr(feature_is[mask_is], y_is[mask_is])[0]
    corr_oos = stats.pearsonr(feature_oos[mask_oos], y_oos[mask_oos])[0]
    return corr_is, corr_oos, corr_oos - corr_is


def main() -> None:
    print("Loading model and data...")
    model = PatternClassifier()
    model.load(MODEL_PATH)

    df = load_data(DATA_PATH)

    extractor = FeatureExtractor()
    features = extractor.extract_all_features(df)

    common_cols = [c for c in model.feature_names_ if c in features.columns]
    missing = [c for c in model.feature_names_ if c not in features.columns]

    X = features[common_cols].copy()
    for m in missing:
        X[m] = 0.0
    X = X.dropna()

    # Triple-barrier labels
    aligned = df.loc[X.index]
    atr = compute_atr(aligned)
    labeler = TripleBarrierLabeler(atr_mult_tp=1.5, atr_mult_sl=1.0)
    labels = labeler.fit(
        close=aligned["Close"],
        high=aligned["High"],
        low=aligned["Low"],
        take_profit=None,
        stop_loss=None,
        time_limit=HORIZON,
        atr_series=atr,
    )
    y = (labels == 1).astype(int).values

    # Split IS vs OOS
    is_mask = X.index < IS_END
    oos_mask = X.index >= OOS_START
    X_is = X[is_mask].values.astype(np.float64)
    X_oos = X[oos_mask].values.astype(np.float64)
    y_is = y[is_mask]
    y_oos = y[oos_mask]

    print(f"IS: {len(X_is)} bars, OOS: {len(X_oos)} bars")

    # ── KS tests on all features ──
    print("\n=== Feature Distribution Shifts (KS Test) ===")
    print(
        f"{'Feature':<30} | {'KS stat':>8} | {'p-value':>10} | {'Shift?':>8} | "
        f"{'IS mean':>10} | {'OOS mean':>10}"
    )
    print("-" * 90)

    results = []
    for i, fname in enumerate(model.feature_names_):
        ks_stat, p_val = ks_test_is_vs_oos(X_is[:, i], X_oos[:, i])
        is_mean = np.nanmean(X_is[:, i])
        oos_mean = np.nanmean(X_oos[:, i])
        shifted = "YES" if (p_val is not None and not np.isnan(p_val) and p_val < 0.01) else ""
        results.append(
            {
                "feature": fname,
                "ks_stat": ks_stat,
                "p_value": p_val,
                "shifted": bool(p_val is not None and not np.isnan(p_val) and p_val < 0.01),
                "is_mean": is_mean,
                "oos_mean": oos_mean,
            }
        )
        print(
            f"{fname:<30} | {ks_stat:>8.4f} | {p_val:>10.2e} | {shifted:>8} | "
            f"{is_mean:>10.4f} | {oos_mean:>10.4f}"
        )

    n_shifted = sum(1 for r in results if r["shifted"])
    print(f"\nFeatures with significant distribution shift (p<0.01): {n_shifted}/{len(results)}")

    # ── Correlation changes ──
    print("\n=== Feature->Label Correlation Changes ===")
    print(f"{'Feature':<30} | {'Corr IS':>8} | {'Corr OOS':>8} | {'dCorr':>8} | {'Direction':>12}")
    print("-" * 75)

    corr_changes = []
    for i, fname in enumerate(model.feature_names_):
        c_is, c_oos, delta = feature_target_corr_change(
            X_is[:, i],
            y_is,
            X_oos[:, i],
            y_oos,
        )
        if not np.isnan(delta):
            direction = (
                "SAME"
                if abs(delta) < 0.05
                else (
                    "FLIPPED" if c_is * c_oos < 0 else "WEAKER" if abs(delta) > 0.05 else "STRONGER"
                )
            )
            corr_changes.append(
                {
                    "feature": fname,
                    "corr_is": c_is,
                    "corr_oos": c_oos,
                    "delta": delta,
                    "direction": direction,
                }
            )
            print(f"{fname:<30} | {c_is:>8.4f} | {c_oos:>8.4f} | {delta:>+8.4f} | {direction:>12}")

    n_flipped = sum(1 for c in corr_changes if c["direction"] == "FLIPPED")
    n_weaker = sum(1 for c in corr_changes if c["direction"] == "WEAKER")
    print(f"\nFlipped: {n_flipped}, Weaker: {n_weaker}")

    # ── Top SHAP features table ──
    print("\n=== Top SHAP Features — Regime Impact ===")
    print(
        f"{'Feature':<25} | {'KS stat':>8} | {'p-value':>10} | "
        f"{'IS corr':>8} | {'OOS corr':>8} | {'d corr':>8} | {'IS mean':>10} | {'OOS mean':>10}"
    )
    print("-" * 110)
    for fname in TOP_FEATURES:
        idx = model.feature_names_.index(fname)
        ks_stat, p_val = ks_test_is_vs_oos(X_is[:, idx], X_oos[:, idx])
        c_is, c_oos, delta = feature_target_corr_change(
            X_is[:, idx],
            y_is,
            X_oos[:, idx],
            y_oos,
        )
        is_mean = np.nanmean(X_is[:, idx])
        oos_mean = np.nanmean(X_oos[:, idx])
        print(
            f"{fname:<25} | {ks_stat:>8.4f} | {p_val:>10.2e} | "
            f"{c_is:>8.4f} | {c_oos:>8.4f} | {delta:>+8.4f} | "
            f"{is_mean:>10.4f} | {oos_mean:>10.4f}"
        )

    # ── Plot: Feature distribution shift for top features ──
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for idx_ax, fname in enumerate(TOP_FEATURES):
        ax = axes[idx_ax // 5, idx_ax % 5]
        feat_idx = model.feature_names_.index(fname)
        is_vals = X_is[:, feat_idx][~np.isnan(X_is[:, feat_idx])]
        oos_vals = X_oos[:, feat_idx][~np.isnan(X_oos[:, feat_idx])]

        ax.hist(
            is_vals, bins=40, alpha=0.5, label="IS (2015-2024)", density=True, color="steelblue"
        )
        ax.hist(
            oos_vals,
            bins=min(40, max(5, len(oos_vals) // 3)),
            alpha=0.5,
            label="OOS (2025-2026)",
            density=True,
            color="darkorange",
        )
        ks_stat, p_val = ks_test_is_vs_oos(is_vals, oos_vals)
        ax.set_title(f"{fname}\nKS={ks_stat:.3f}, p={p_val:.2e}", fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(
        "Feature Distribution Shifts: IS (2015-2024) vs OOS (2025-2026)\nTop 10 SHAP Features",
        fontsize=13,
    )
    fig.tight_layout()
    out_path = OUTPUT_DIR / "regime_shift_features.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved regime shift plot to {out_path}")

    # ── Target rate comparison ──
    print("\n=== Target Distribution ===")
    print(f"  IS  TP rate: {y_is.mean():.4f} ({y_is.mean() * 100:.1f}%)")
    print(f"  OOS TP rate: {y_oos.mean():.4f} ({y_oos.mean() * 100:.1f}%)")
    print(f"  d: {y_oos.mean() - y_is.mean():+.4f}")

    # ── Model confidence shift ──
    probs = model.predict(X)["probability_profitable"].values
    print("\n=== Model Confidence Shift ===")
    print(f"  IS  mean prob: {probs[is_mask].mean():.4f}, std={probs[is_mask].std():.4f}")
    print(f"  OOS mean prob: {probs[oos_mask].mean():.4f}, std={probs[oos_mask].std():.4f}")
    # Fraction of bars above threshold
    for thresh in [0.40, 0.45, 0.50]:
        is_frac = (probs[is_mask] >= thresh).mean()
        oos_frac = (probs[oos_mask] >= thresh).mean()
        print(
            f"  P >= {thresh:.2f}: IS={is_frac:.3f} ({is_frac * 100:.1f}%), "
            f"OOS={oos_frac:.3f} ({oos_frac * 100:.1f}%), d={oos_frac - is_frac:+.3f}"
        )


if __name__ == "__main__":
    main()
