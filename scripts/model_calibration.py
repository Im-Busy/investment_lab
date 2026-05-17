"""Reliability diagram: does P=0.45 signal actually win 45% of the time?

Uses triple-barrier labels matching model training (TP=1.5xATR, SL=1.0xATR, horizon=5).
Compares IS (2015-2024) vs OOS (2025-2026) calibration.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  # noqa: E402

from src.ml.feature_engineering import FeatureExtractor  # noqa: E402
from src.ml.pattern_classifier import PatternClassifier  # noqa: E402
from src.ml.triple_barrier import TripleBarrierLabeler  # noqa: E402

MODEL_PATH = "models/pattern_classifier_v3_SPY_20260514_195235.pkl"
DATA_PATH = "data/raw/SPY_daily.csv"
HORIZON = 5
OUTPUT_DIR = Path("reports/calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Chronological splits
IS_END = "2025-01-01"
OOS_START = "2025-01-01"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0.0 if col == "Volume" else df.iloc[:, 0]
    return df


def reliability_curve(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> tuple:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    frac_positive = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        bin_counts[i] = mask.sum()
        if bin_counts[i] > 0:
            frac_positive[i] = outcomes[mask].mean()
    return bin_centers, frac_positive, bin_counts


def compute_expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """ECE: weighted average of |accuracy - confidence| per bin."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    total = len(y)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        n_bin = mask.sum()
        if n_bin > 0:
            acc = y[mask].mean()
            conf = probs[mask].mean()
            ece += (n_bin / total) * abs(acc - conf)
    return ece


def plot_reliability(probs_is, y_is, probs_oos, y_oos, output_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, probs, y, label in [
        (ax1, probs_is, y_is, "In-Sample (2015-2024)"),
        (ax2, probs_oos, y_oos, "Out-of-Sample (2025-2026)"),
    ]:
        if len(probs) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue

        bin_centers, frac_pos, counts = reliability_curve(probs, y, n_bins=10)
        ece = compute_expected_calibration_error(probs, y)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
        # Baseline: random entry win rate with asymmetric barriers
        baseline_wr = 1.0 / (1.0 + 1.5)
        ax.axhline(
            y=baseline_wr,
            color="orange",
            linestyle=":",
            alpha=0.5,
            label=f"Random baseline ({baseline_wr:.3f})",
        )

        valid = ~np.isnan(frac_pos)
        sizes = np.sqrt(np.maximum(counts[valid], 1)) * 20
        ax.scatter(
            bin_centers[valid],
            frac_pos[valid],
            s=sizes,
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
            label="Model bins",
        )

        for i in range(len(bin_centers)):
            if valid[i]:
                ax.annotate(
                    str(counts[i]),
                    (bin_centers[i], frac_pos[i]),
                    textcoords="offset points",
                    xytext=(0, -14),
                    ha="center",
                    fontsize=7,
                    color="gray",
                )

        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Fraction TP Hit")
        ax.set_title(f"{label}\nECE={ece:.4f}, n={len(probs):,}")
        ax.legend(fontsize=8)
        ax.set_xlim(0.2, 0.7)

    fig.suptitle(
        "Reliability Diagram — CatBoost V3 (SPY) — Triple-Barrier Labels\n"
        "TP=1.5xATR, SL=1.0xATR, Horizon=5 bars",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved reliability diagram to {output_path}")


def threshold_analysis(probs: np.ndarray, y: np.ndarray, label: str) -> None:
    print(f"\n=== Key Threshold Accuracy — {label} ===")
    for thresh in [0.35, 0.40, 0.45, 0.50]:
        mask = probs >= thresh
        n = mask.sum()
        if n > 0:
            wr = y[mask].mean()
            bias = wr - thresh
            status = (
                "OVERCONFIDENT"
                if bias < -0.05
                else "UNDERCONFIDENT"
                if bias > 0.05
                else "CALIBRATED"
            )
            print(
                f"  P >= {thresh:.2f}: {n:>5} bars, actual win rate = {wr:.3f} ({wr * 100:.1f}%), "
                f"bias={bias:+.3f} — {status}"
            )
        else:
            print(f"  P >= {thresh:.2f}: 0 bars (insufficient data)")


def main() -> None:
    print("Loading model...")
    model = PatternClassifier()
    model.load(MODEL_PATH)
    print(f"Model: {len(model.feature_names_)} features, type={model.model_type}")
    print(
        f"Calibrator: {type(model.calibrator_).__name__ if model.calibrator_ is not None else 'None'}"
    )

    print("\nLoading data...")
    df = load_data(DATA_PATH)
    print(f"Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

    print("\nExtracting features...")
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(df)

    # Align with model's feature names
    common_cols = [c for c in model.feature_names_ if c in features.columns]
    missing = [c for c in model.feature_names_ if c not in features.columns]
    print(f"Features matched: {len(common_cols)}/{len(model.feature_names_)}")
    if missing:
        print(f"Missing: {missing}")

    X = features[common_cols].copy()
    for m in missing:
        X[m] = 0.0
    X = X.dropna()
    print(f"Valid bars with features: {len(X)}")

    print("\nGenerating predictions...")
    probs_df = model.predict(X)
    probs = probs_df["probability_profitable"].values
    print(f"Prob range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"Prob mean/median/std: {probs.mean():.4f} / {np.median(probs):.4f} / {probs.std():.4f}")

    # ── Compute triple-barrier labels ──
    print("\nComputing triple-barrier labels (TP=1.5xATR, SL=1.0xATR, horizon=5)...")
    # Align data with features index
    aligned = df.loc[X.index]
    labeler = TripleBarrierLabeler(atr_mult_tp=1.5, atr_mult_sl=1.0)

    # Compute ATR manually
    tr = pd.concat(
        [
            (aligned["High"] - aligned["Low"]).abs(),
            (aligned["High"] - aligned["Close"].shift(1)).abs(),
            (aligned["Low"] - aligned["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14).mean()

    labels = labeler.fit(
        close=aligned["Close"],
        high=aligned["High"],
        low=aligned["Low"],
        take_profit=None,
        stop_loss=None,
        time_limit=HORIZON,
        atr_series=atr,
    )
    binary_labels = (labels == 1).astype(int)
    summary = TripleBarrierLabeler.label_summary(labels)
    print(
        f"Label distribution: TP={summary['n_positive']} ({summary['pct_positive']:.1%}), "
        f"SL={summary['n_negative']} ({summary['pct_negative']:.1%}), "
        f"Timeout={summary['n_timeout']} ({summary['pct_timeout']:.1%})"
    )
    print(f"Baseline random win rate (asymmetric barriers): {1.0 / (1.0 + 1.5):.3f}")

    # ── Split IS vs OOS ──
    is_mask = X.index < IS_END
    oos_mask = X.index >= OOS_START

    probs_is = probs[is_mask]
    y_is = binary_labels.values[is_mask]
    probs_oos = probs[oos_mask]
    y_oos = binary_labels.values[oos_mask]

    print(
        f"\nIS bars: {len(probs_is)} ({X.index[is_mask][0].date()} to {X.index[is_mask][-1].date()})"
    )
    print(
        f"OOS bars: {len(probs_oos)} ({X.index[oos_mask][0].date()} to {X.index[oos_mask][-1].date()})"
    )

    # ── Overall ECE ──
    ece_is = compute_expected_calibration_error(probs_is, y_is)
    ece_oos = compute_expected_calibration_error(probs_oos, y_oos)
    print(f"\nECE IS: {ece_is:.4f}, OOS: {ece_oos:.4f}")

    # ── Reliability diagram ──
    plot_reliability(
        probs_is, y_is, probs_oos, y_oos, OUTPUT_DIR / "reliability_diagram_triple_barrier.png"
    )

    # ── Summary table ──
    print("\n=== Reliability Summary (Triple-Barrier) ===")
    for label_name, p, y_val in [
        ("In-Sample (2015-2024)", probs_is, y_is),
        ("Out-of-Sample (2025-2026)", probs_oos, y_oos),
    ]:
        print(f"\n{label_name}:")
        bin_centers, frac_pos, counts = reliability_curve(p, y_val, n_bins=10)
        print(f"{'Prob bin':>12} | {'Count':>6} | {'TP Rate':>9} | {'Bias':>8}")
        print("-" * 45)
        for i in range(len(bin_centers)):
            if not np.isnan(frac_pos[i]):
                bias = frac_pos[i] - bin_centers[i]
                print(
                    f"  {bin_centers[i]:.3f}      | {counts[i]:>6} | "
                    f"{frac_pos[i]:.3f} ({frac_pos[i] * 100:.1f}%) | {bias:+.4f}"
                )

    # ── Key thresholds ──
    threshold_analysis(probs_is, y_is, "In-Sample (2015-2024)")
    threshold_analysis(probs_oos, y_oos, "Out-of-Sample (2025-2026)")

    # ── Probability distribution shift ──
    print("\n=== Probability Distribution Drift ===")
    print(
        f"  IS  mean={probs_is.mean():.4f}  median={np.median(probs_is):.4f}  "
        f"std={probs_is.std():.4f}  skew={pd.Series(probs_is).skew():.4f}"
    )
    print(
        f"  OOS mean={probs_oos.mean():.4f}  median={np.median(probs_oos):.4f}  "
        f"std={probs_oos.std():.4f}  skew={pd.Series(probs_oos).skew():.4f}"
    )


if __name__ == "__main__":
    main()
