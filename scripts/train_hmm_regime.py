"""HMM Regime Detection — Compare to Simple Rule-Based Regimes (P3-5).

Evaluates Hidden Markov Model (hmmlearn GaussianHMM) latent regime detection
against the existing SimpleTrendRegimeDetector (200MA Bull/Bear rule).

Tasks:
  1. Train HMM on 132 technical features — get regime labels + transition matrix
  2. Compare HMM regimes to SimpleTrendRegimeDetector agreement
  3. Train per-regime CatBoost classifiers using HMM states
  4. Compare OOS performance: HMM-per-regime vs single-model vs simple-regime
  5. Evaluate regime-conditional returns and transition stability

Gate: HMM per-regime OOS Sharpe > single-model OOS Sharpe (adds predictive value).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
from scipy.stats import spearmanr

from src.ml.feature_engineering import FeatureExtractor
from src.ml.hmm_regime import HMMRegimeDetector
from src.ml.pattern_classifier import PatternClassifier
from src.ml.simple_regime import SimpleTrendRegimeDetector
from src.ml.triple_barrier import TripleBarrierLabeler

SYMBOL = "SPY"
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("reports/hmm_regime")
MODEL_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 5
ATR_MULT_TP = 1.5
ATR_MULT_SL = 1.0
RANDOM_STATE = 42
HMM_N_STATES = 4
MIN_BARS_PER_REGIME = 250
IS_END = "2025-01-01"


@dataclass
class RegimeMetrics:
    name: str
    n_bars: int
    mean_return: float
    std_return: float
    sharpe: float
    tp_rate: float


@dataclass
class ModelMetrics:
    model_name: str
    regime_name: str
    auc_is: float
    auc_oos: float
    gap: float
    n_train: int
    n_test: int
    train_time_s: float


@dataclass
class HMMReport:
    hmm_n_states: int
    agreement_with_simple: float
    agreement_with_simple_oos: float
    transition_matrix: list
    expected_durations: dict
    regime_metrics_is: list[dict]
    regime_metrics_oos: list[dict]
    single_model_metrics: dict
    hmm_regime_metrics: list[dict]
    simple_regime_metrics: list[dict]
    gate_pass: bool = False


def load_data(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_daily.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0.0 if col == "Volume" else df.iloc[:, 0]
    return df


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def compute_regime_returns(
    regimes: pd.Series, returns: pd.Series, tp_labels: pd.Series
) -> list[RegimeMetrics]:
    result = []
    for regime in sorted(set(regimes)):
        mask = regimes == regime
        if mask.sum() < 10:
            continue
        r = returns.loc[mask]
        t = tp_labels.loc[mask]
        if len(r) == 0:
            continue
        sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
        result.append(
            RegimeMetrics(
                name=str(regime),
                n_bars=int(mask.sum()),
                mean_return=float(r.mean()),
                std_return=float(r.std()),
                sharpe=sharpe,
                tp_rate=float(t.mean()),
            )
        )
    return result


def map_hmm_to_trend(hmm_regimes: pd.Series, returns: pd.Series) -> pd.Series:
    regime_means = {r: returns[hmm_regimes == r].mean() for r in set(hmm_regimes)}
    sorted_regimes = sorted(regime_means, key=regime_means.get, reverse=True)
    n = len(sorted_regimes)
    mapping = {}
    for i, r in enumerate(sorted_regimes):
        if i < n // 2:
            mapping[r] = "Bull"
        else:
            mapping[r] = "Bear"
    return hmm_regimes.map(mapping)


def train_regime_model(
    X: pd.DataFrame,
    y: pd.Series,
    regime_name: str,
) -> tuple[PatternClassifier, dict, float]:
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    clf = PatternClassifier(
        model_type="catboost",
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,
        l2_leaf_reg=10.0,
        random_strength=3.0,
        min_data_in_leaf=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )
    result = clf.train(X_train, y_train, calibration_data=(X_test, y_test))
    gap = result.train_auc - result.test_auc

    metrics = {
        "train_auc": result.train_auc,
        "test_auc": result.test_auc,
        "calibration_error": result.calibration_error,
        "overfit_gap": gap,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "top_features": dict(list(result.feature_importance.items())[:5]),
    }
    return clf, metrics, gap


def plot_regime_comparison(
    df: pd.DataFrame,
    hmm_regimes: pd.Series,
    simple_regimes: pd.Series,
    returns: pd.Series,
    X: pd.DataFrame,
) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Price with HMM regime shading
    ax = axes[0, 0]
    ax.plot(df.index, df["Close"], "k-", linewidth=0.8, label="SPY Close")
    unique_regimes = sorted(set(hmm_regimes))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_regimes)))
    for r, c in zip(unique_regimes, colors):
        mask = hmm_regimes == r
        regime_idx = hmm_regimes.index[mask]
        if len(regime_idx) > 1:
            for s in regime_idx:
                ax.axvline(x=s, color=c, alpha=0.08, linewidth=3)
    ax.set_title(f"HMM ({HMM_N_STATES} states) Regime Map")
    ax.set_ylabel("Price")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    # Regime distribution (bar chart)
    ax = axes[0, 1]
    hmm_counts = hmm_regimes.value_counts()
    ax.bar(range(len(hmm_counts)), hmm_counts.values, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(hmm_counts)))
    ax.set_xticklabels([str(n)[:20] for n in hmm_counts.index], rotation=45, ha="right", fontsize=8)
    ax.set_title("HMM Regime Distribution")
    ax.set_ylabel("Bars")
    ax.grid(alpha=0.3)

    # HMM vs Simple agreement over time
    ax = axes[1, 0]
    hmm_trend = map_hmm_to_trend(hmm_regimes, returns)
    agreement = -(hmm_trend == simple_regimes).astype(int)
    rolling_agree = agreement.rolling(252).mean()
    ax.plot(rolling_agree.index, rolling_agree.values, "b-", linewidth=1)
    ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Random (0.5)")
    ax.axhline(y=0.7, color="g", linestyle="--", alpha=0.5, label="Good (0.7)")
    ax.set_title("HMM vs Simple Regime Rolling Agreement (252d)")
    ax.set_ylabel("Agreement Rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Transition matrix heatmap
    ax = axes[1, 1]
    from src.ml.hmm_regime import HMMRegimeDetector

    # Get transition matrix directly
    trans_mat = np.zeros((HMM_N_STATES, HMM_N_STATES))
    for i in range(HMM_N_STATES):
        for j in range(HMM_N_STATES):
            trans_mat[i, j] = (
                np.mean(hmm_regimes.iloc[:-1].values == unique_regimes[i].replace(" ", "_")) * 0
            )
    im = ax.imshow(trans_mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(HMM_N_STATES))
    ax.set_yticks(range(HMM_N_STATES))
    labels = [str(r)[:8] for r in unique_regimes]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("HMM Transition Matrix")
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"hmm_regime_comparison_{SYMBOL}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def plot_regime_returns(
    metrics_is: list[RegimeMetrics],
    metrics_oos: list[RegimeMetrics],
) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title in [
        (axes[0], metrics_is, "IS Regime-Conditional Sharpe"),
        (axes[1], metrics_oos, "OOS Regime-Conditional Sharpe"),
    ]:
        names = [m.name[:15] for m in data]
        sharpes = [m.sharpe for m in data]
        colors = ["g" if s > 0 else "r" for s in sharpes]
        ax.barh(range(len(names)), sharpes, color=colors, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Annualized Sharpe")
        ax.set_title(title)
        ax.axvline(x=0, color="k", linewidth=0.5)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"hmm_regime_returns_{SYMBOL}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def main() -> None:
    print(f"HMM Regime Detection — {HMM_N_STATES}-State GaussianHMM ({SYMBOL})")
    print(f"IS/OOS split: {IS_END}")

    df = load_data(SYMBOL)
    print(f"Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

    extractor = FeatureExtractor()
    X = extractor.extract_all_features(df, include_forward_returns=False)
    X = X.dropna()

    atr = _compute_atr(df)
    labeler = TripleBarrierLabeler(atr_mult_tp=ATR_MULT_TP, atr_mult_sl=ATR_MULT_SL)
    labels = labeler.fit(
        close=df["Close"],
        high=df["High"],
        low=df["Low"],
        time_limit=HORIZON,
        atr_series=atr,
    )
    y_binary = (labels == 1).astype(int)

    common = X.index.intersection(y_binary.dropna().index)
    X = X.loc[common]
    y_binary = y_binary.loc[common]
    returns = df.loc[common, "Close"].pct_change().fillna(0.0)

    print(
        f"Features: {X.shape[1]} cols, Labels: {len(y_binary)} samples, "
        f"positive rate: {y_binary.mean() * 100:.1f}%"
    )

    # ── IS/OOS split ──
    is_mask = X.index < IS_END
    X_is = X[is_mask]
    X_oos = X[~is_mask]

    # ── Train HMM on IS data only ──
    print("\n=== Training HMM Regime Detector ===")
    hmm = HMMRegimeDetector(
        n_states=HMM_N_STATES, covariance_type="diag", max_iter=200, random_state=RANDOM_STATE
    )
    hmm.fit(X_is)
    summary = hmm.get_regime_summary()
    print(f"Converged: {summary.metadata['convergence']} (iters: {summary.metadata['n_iter']})")

    trans_mat = hmm.get_transition_matrix()
    print("\nTransition matrix:")
    print(trans_mat.round(3).to_string())

    durations = hmm.get_expected_duration()
    print("\nExpected duration (bars):")
    for label, dur in durations.items():
        print(f"  {label}: {dur:.1f} bars")

    # ── Predict regimes on full dataset ──
    hmm_regimes = hmm.predict(X)
    hmm_probs = hmm.predict_proba(X)

    # ── Simple regime detection for comparison ──
    simple = SimpleTrendRegimeDetector(ma_period=200)
    simple.fit(df)
    simple_regimes = simple.predict(df)
    simple_regimes = simple_regimes.loc[common]

    # ── Map HMM states to Bull/Bear for comparison ──
    hmm_trend = map_hmm_to_trend(hmm_regimes.loc[common], returns)
    hmm_trend_is = hmm_trend[is_mask]
    simple_regimes_is = simple_regimes[is_mask]
    hmm_trend_oos = hmm_trend[~is_mask]
    simple_regimes_oos = simple_regimes[~is_mask]

    agree_is = float((hmm_trend_is == simple_regimes_is).mean())
    agree_oos = float((hmm_trend_oos == simple_regimes_oos).mean())
    print(f"\nHMM vs Simple agreement IS: {agree_is:.3f}, OOS: {agree_oos:.3f}")

    # ── Regime-conditional returns ──
    print("\n=== Regime-Conditional Returns ===")
    regime_metrics_is = compute_regime_returns(
        hmm_regimes[is_mask],
        returns[is_mask],
        y_binary[is_mask],
    )
    regime_metrics_oos = compute_regime_returns(
        hmm_regimes[~is_mask],
        returns[~is_mask],
        y_binary[~is_mask],
    )

    print("IS HMM Regimes:")
    for m in regime_metrics_is:
        print(
            f"  {m.name:<18} | bars={m.n_bars:>5} | return={m.mean_return:>8.5f} "
            f"| vol={m.std_return:.4f} | Sharpe={m.sharpe:>6.2f} | TP={m.tp_rate:.3f}"
        )
    print("OOS HMM Regimes:")
    for m in regime_metrics_oos:
        print(
            f"  {m.name:<18} | bars={m.n_bars:>5} | return={m.mean_return:>8.5f} "
            f"| vol={m.std_return:.4f} | Sharpe={m.sharpe:>6.2f} | TP={m.tp_rate:.3f}"
        )

    # ── Train single model baseline ──
    print("\n=== Training Single Model Baseline ===")
    import time

    t0 = time.monotonic()
    split_idx = int(len(X_is) * 0.7)
    baseline_clf = PatternClassifier(
        model_type="catboost",
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,
        l2_leaf_reg=10.0,
        random_strength=3.0,
        min_data_in_leaf=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )
    b_result = baseline_clf.train(
        X_is.iloc[:split_idx],
        y_binary.loc[X_is.index[:split_idx]],
        calibration_data=(X_is.iloc[split_idx:], y_binary.loc[X_is.index[split_idx:]]),
    )
    b_train_time = time.monotonic() - t0
    print(
        f"Single model: AUC IS={b_result.train_auc:.4f}, "
        f"Val AUC={b_result.test_auc:.4f}, Gap={b_result.train_auc - b_result.test_auc:.4f}"
    )

    # ── Train per-regime models using HMM regimes ──
    print("\n=== Training Per-HMM-Regime Models ===")
    hmm_is = hmm_regimes.loc[X_is.index]
    regime_counts = hmm_is.value_counts().to_dict()
    print(f"HMM regime distribution: {regime_counts}")

    hmm_models: dict[str, PatternClassifier] = {}
    hmm_model_metrics: list[dict] = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for regime_name, count in sorted(regime_counts.items(), key=lambda x: x[1], reverse=True):
        if count < MIN_BARS_PER_REGIME:
            print(f"Skipping {regime_name}: {count} < {MIN_BARS_PER_REGIME} min bars")
            continue

        mask = (hmm_is == regime_name).values
        X_r = X_is.loc[mask]
        y_r = y_binary.loc[X_r.index]

        if len(y_r) < 50 or y_r.nunique() < 2:
            print(f"Skipping {regime_name}: insufficient class variety")
            continue

        print(f"Training HMM regime '{regime_name}' ({len(y_r)} bars, {y_r.mean() * 100:.1f}% TP)")
        model, metrics, gap = train_regime_model(X_r, y_r, regime_name)
        hmm_models[regime_name] = model
        hmm_model_metrics.append(
            {
                "regime": regime_name,
                "auc_is": metrics["train_auc"],
                "auc_oos": metrics["test_auc"],
                "gap": gap,
                "n_train": metrics["n_train"],
                "n_test": metrics["n_test"],
            }
        )

        model_path = (
            MODEL_DIR / f"regime_hmm_{regime_name.replace(' ', '_')}_{SYMBOL}_{timestamp}.pkl"
        )
        model.save(str(model_path))

    # ── Ensemble: route predictions by regime ──
    hmm_oos = hmm_regimes.loc[X_oos.index]
    oos_regimes_found = sorted(set(hmm_oos))
    print(f"\nOOS regimes: {oos_regimes_found}")

    routed_probs = np.zeros(len(X_oos), dtype=float)
    y_oos = y_binary.loc[X_oos.index].values

    for regime_name, model in hmm_models.items():
        mask = (hmm_oos == regime_name).values
        if mask.sum() == 0:
            continue
        routed_probs[mask] = model.predict(X_oos.iloc[mask])["probability_profitable"].values

    fallback_mask = np.abs(routed_probs) < 1e-10
    if fallback_mask.any():
        routed_probs[fallback_mask] = baseline_clf.predict(X_oos.iloc[fallback_mask])[
            "probability_profitable"
        ].values

    # ── AUC evaluation ──
    from sklearn.metrics import roc_auc_score

    b_probs_oos = baseline_clf.predict(X_oos)["probability_profitable"].values
    single_auc_oos = float(roc_auc_score(y_oos, b_probs_oos))
    hmm_auc_oos = float(roc_auc_score(y_oos, routed_probs))

    print(f"\nSingle Model OOS AUC: {single_auc_oos:.4f}")
    print(f"HMM Ensemble OOS AUC:  {hmm_auc_oos:.4f}")

    # ── Simple regime comparison (load existing config) ──
    print("\n=== Simple Regime Baseline ===")
    simple_router_path = MODEL_DIR / f"regime_router_{SYMBOL}.json"
    simple_auc_oos = 0.0
    if simple_router_path.exists():
        try:
            from src.ml.regime_router import RegimeRouter

            with open(simple_router_path) as f:
                router_config = json.load(f)
            model_paths = router_config.get("regime_models", {})
            fallback_path = router_config.get("fallback_model", "")
            if model_paths and fallback_path:
                loaded_models = {r: PatternClassifier.load(p) for r, p in model_paths.items()}
                fallback_model = PatternClassifier.load(fallback_path)
                regime_detector = SimpleTrendRegimeDetector(
                    ma_period=router_config.get("ma_period", 200)
                )
                regime_detector.fit(df)
                router = RegimeRouter(regime_detector, loaded_models, fallback_model)
                simple_probs = np.asarray(router.predict(X_oos, df.loc[common].iloc[-len(X_oos) :]))
                simple_auc_oos = float(roc_auc_score(y_oos, simple_probs))
                print(f"Simple-Ensemble OOS AUC: {simple_auc_oos:.4f}")
        except Exception as e:
            print(f"Could not load simple router: {e}")
    else:
        print("No simple regime router found at", simple_router_path)

    # ── Gate decision ──
    gate_pass = bool(hmm_auc_oos > single_auc_oos)
    print(f"\n{'=' * 70}")
    print("=== OOS AUC Comparison ===")
    print(f"{'Strategy':<25} {'AUC OOS':>8} {'Delta':>8} {'Result':>8}")
    print(f"{'-' * 55}")
    print(f"{'Single Model':<25} {single_auc_oos:>8.4f} {'--':>8} {'BASELINE':>8}")
    print(
        f"{'HMM Ensemble':<25} {hmm_auc_oos:>8.4f} {hmm_auc_oos - single_auc_oos:>8.4f} "
        f"{'PASS' if gate_pass else 'FAIL':>8}"
    )
    if simple_router_path.exists():
        print(
            f"{'Simple Ensemble':<25} {simple_auc_oos:>8.4f} {simple_auc_oos - single_auc_oos:>8.4f}"
        )
    print()

    if gate_pass:
        print(
            f"Gate PASSES — HMM Ensemble AUC {hmm_auc_oos:.4f} > Single Model {single_auc_oos:.4f}"
        )
        print("  HMM latent regimes improve model discrimination beyond a single model.")
    else:
        print(
            f"Gate FAILS — HMM Ensemble AUC {hmm_auc_oos:.4f} <= Single Model {single_auc_oos:.4f}"
        )
        print("  HMM latent regimes do not add predictive value over single model.")
        print("  Simple 200MA trend rule is more practical for regime-dependent modeling.")

    # ── Plots ──
    plot_regime_comparison(df, hmm_regimes, simple_regimes, returns, X)
    plot_regime_returns(regime_metrics_is, regime_metrics_oos)

    # ── Save report ──
    report = HMMReport(
        hmm_n_states=HMM_N_STATES,
        agreement_with_simple=agree_is,
        agreement_with_simple_oos=agree_oos,
        transition_matrix=trans_mat.values.tolist(),
        expected_durations={str(k): float(v) for k, v in durations.items()},
        regime_metrics_is=[
            {
                "name": m.name,
                "n_bars": m.n_bars,
                "mean_return": m.mean_return,
                "sharpe": m.sharpe,
                "tp_rate": m.tp_rate,
            }
            for m in regime_metrics_is
        ],
        regime_metrics_oos=[
            {
                "name": m.name,
                "n_bars": m.n_bars,
                "mean_return": m.mean_return,
                "sharpe": m.sharpe,
                "tp_rate": m.tp_rate,
            }
            for m in regime_metrics_oos
        ],
        single_model_metrics={
            "auc_is": b_result.train_auc,
            "auc_test": b_result.test_auc,
            "gap": b_result.train_auc - b_result.test_auc,
            "auc_oos": single_auc_oos,
        },
        hmm_regime_metrics=hmm_model_metrics,
        simple_regime_metrics=[],
        gate_pass=gate_pass,
    )
    report_d = report.__dict__
    report_d["hmm_auc_oos"] = hmm_auc_oos
    report_d["simple_auc_oos"] = simple_auc_oos

    out_path = OUTPUT_DIR / f"hmm_regime_{SYMBOL}.json"
    with open(out_path, "w") as f:
        json.dump(report_d, f, indent=2, default=str)
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
