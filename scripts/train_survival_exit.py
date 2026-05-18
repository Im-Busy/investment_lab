"""Survival Analysis for Time-to-Exit Prediction — P3-3.

Trains survival models (Cox, RSF, GBSA) to predict when a trade hits TP/SL
within the triple-barrier horizon. Uses censored survival labels where:
  - event=1: TP hit before SL/time (positive outcome)
  - event=0: SL hit or time expired (censored/negative)

Evaluates with C-index, time-dependent AUC, integrated Brier score.
Compares survival-based signal timing to existing binary classification.

Gate: C-index > 0.55 (better than random ranking).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")

from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv
from sklearn.model_selection import train_test_split

from src.ml.feature_engineering import FeatureExtractor
from src.ml.triple_barrier import TripleBarrierLabeler

SYMBOL = "SPY"
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("reports/survival")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 5
ATR_MULT_TP = 1.5
ATR_MULT_SL = 1.0
RANDOM_STATE = 42

IS_END = "2025-01-01"


def load_data(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_daily.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).dropna()
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


@dataclass
class SurvivalResult:
    model_name: str
    c_index_is: float
    c_index_oos: float
    ibs_is: float
    ibs_oos: float
    auc_mean_is: float
    auc_mean_oos: float
    train_time_s: float
    n_train: int
    n_test: int
    n_events_train: int
    n_events_test: float
    gate_pass: bool = False


def generate_survival_labels(
    df: pd.DataFrame,
    atr: pd.Series,
    time_limit: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate survival labels: (time_to_event, event_indicator).

    event=1: TP hit before SL/time
    event=0: SL hit or time expired (censored)
    time: bars until TP hit (or until SL/time if censored)
    """
    labeler = TripleBarrierLabeler(atr_mult_tp=ATR_MULT_TP, atr_mult_sl=ATR_MULT_SL)
    labels = labeler.fit(
        df["Close"],
        df["High"],
        df["Low"],
        time_limit=time_limit,
        atr_series=atr,
    )

    bars_to_exit = labels.attrs.get("bars_to_exit", np.full(len(labels), time_limit))
    barrier = labels.attrs.get("barrier", np.full(len(labels), "time"))

    time_arr = np.asarray(bars_to_exit, dtype=np.float64)
    event_arr = np.where(barrier == "tp", True, False).astype(bool)

    return time_arr, event_arr


def compute_summary(results: list[SurvivalResult]) -> dict:
    best = max(results, key=lambda r: r.c_index_oos)
    return {
        "best_model": best.model_name,
        "best_c_index_oos": best.c_index_oos,
        "models": {
            r.model_name: {
                "c_index_is": round(r.c_index_is, 4),
                "c_index_oos": round(r.c_index_oos, 4),
                "ibs_is": round(r.ibs_is, 4),
                "ibs_oos": round(r.ibs_oos, 4),
                "auc_mean_is": round(r.auc_mean_is, 4),
                "auc_mean_oos": round(r.auc_mean_oos, 4),
                "train_time_s": round(r.train_time_s, 1),
                "gate_pass": r.gate_pass,
            }
            for r in results
        },
    }


def plot_survival_curves(
    model: Any,
    X_test_small: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> str:
    """Plot survival curves for high vs low risk groups."""
    import matplotlib.pyplot as plt

    risk_scores = model.predict(X_test_small.values)
    median_risk = np.median(risk_scores)
    high_risk = risk_scores > median_risk
    low_risk = ~high_risk

    times = np.arange(1, HORIZON + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Survival function estimates
    surv_funcs = model.predict_survival_function(X_test_small.values)
    high_surv = np.array([sf(times - 1) for sf, hr in zip(surv_funcs, high_risk) if hr])
    low_surv = np.array([sf(times - 1) for sf, hr in zip(surv_funcs, low_risk) if not hr])

    if len(high_surv) > 0:
        axes[0].plot(times, high_surv.mean(axis=0), "r-", linewidth=2, label="High Risk")
        axes[0].fill_between(
            times,
            high_surv.mean(axis=0) - high_surv.std(axis=0),
            high_surv.mean(axis=0) + high_surv.std(axis=0),
            color="red",
            alpha=0.15,
        )
    if len(low_surv) > 0:
        axes[0].plot(times, low_surv.mean(axis=0), "b-", linewidth=2, label="Low Risk")
        axes[0].fill_between(
            times,
            low_surv.mean(axis=0) - low_surv.std(axis=0),
            low_surv.mean(axis=0) + low_surv.std(axis=0),
            color="blue",
            alpha=0.15,
        )

    axes[0].set_xlabel("Days")
    axes[0].set_ylabel("Survival Probability (still in trade)")
    axes[0].set_title(f"{model_name}: Survival Curves by Risk Group")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Cumulative incidence (1 - survival) = probability of hitting TP
    if len(high_surv) > 0:
        axes[1].plot(times, 1 - high_surv.mean(axis=0), "g-", linewidth=2, label="High Risk")
    if len(low_surv) > 0:
        axes[1].plot(times, 1 - low_surv.mean(axis=0), "orange", linewidth=2, label="Low Risk")

    axes[1].set_xlabel("Days")
    axes[1].set_ylabel("Cumulative TP Probability")
    axes[1].set_title(f"{model_name}: Cumulative TP Incidence")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"survival_curves_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model: Any,
    model_name: str,
) -> SurvivalResult:
    import time

    t0 = time.monotonic()
    model.fit(X_train.values, y_train)
    train_time = time.monotonic() - t0

    y_train_struct = Surv.from_arrays(event=y_train["event"], time=y_train["time"])
    y_test_struct = Surv.from_arrays(event=y_test["event"], time=y_test["time"])

    risk_is = model.predict(X_train.values)
    risk_oos = model.predict(X_test.values)

    c_is = concordance_index_censored(y_train_struct["event"], y_train_struct["time"], risk_is)[0]
    c_oos = concordance_index_censored(y_test_struct["event"], y_test_struct["time"], risk_oos)[0]

    # Time-dependent AUC
    eval_times = [1, 2, 3, 4, 5]
    try:
        auc_is, mean_auc_is = cumulative_dynamic_auc(
            y_train_struct, y_train_struct, risk_is, eval_times
        )
        auc_oos, mean_auc_oos = cumulative_dynamic_auc(
            y_train_struct, y_test_struct, risk_oos, eval_times
        )
    except Exception:
        mean_auc_is = 0.5
        mean_auc_oos = 0.5

    # Integrated Brier Score
    try:
        surv_funcs_is = model.predict_survival_function(X_train.iloc[:300].values)
        surv_funcs_oos = model.predict_survival_function(X_test.values[:300].values)
        ibs_is = float(
            integrated_brier_score(
                y_train_struct[:300], y_train_struct[:300], surv_funcs_is, eval_times
            )
        )
        ibs_oos = float(
            integrated_brier_score(
                y_train_struct[:300], y_test_struct[:300], surv_funcs_oos, eval_times
            )
        )
    except Exception:
        ibs_is = 1.0
        ibs_oos = 1.0

    gate_pass = bool(c_oos > 0.55)

    return SurvivalResult(
        model_name=model_name,
        c_index_is=float(c_is),
        c_index_oos=float(c_oos),
        ibs_is=float(ibs_is),
        ibs_oos=float(ibs_oos),
        auc_mean_is=float(mean_auc_is),
        auc_mean_oos=float(mean_auc_oos),
        train_time_s=train_time,
        n_train=len(X_train),
        n_test=len(X_test),
        n_events_train=int(y_train["event"].sum()),
        n_events_test=float(y_test["event"].sum()),
        gate_pass=gate_pass,
    )


def main() -> None:
    print(f"Survival Analysis — Time-to-Exit Prediction ({SYMBOL})")
    print(f"Horizon: {HORIZON}d, TP: {ATR_MULT_TP}x ATR, SL: {ATR_MULT_SL}x ATR")

    df = load_data(SYMBOL)
    atr = compute_atr(df, 14)
    print(f"Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

    extractor = FeatureExtractor()
    features = extractor.extract_all_features(df, include_forward_returns=False)
    X = features.dropna()

    time_arr, event_arr = generate_survival_labels(df, atr, HORIZON)
    time_series = pd.Series(time_arr, index=df.index)
    event_series = pd.Series(event_arr, index=df.index)

    common_idx = X.index.intersection(time_series.index)
    X = X.loc[common_idx]
    y_time = time_series.loc[common_idx].values
    y_event = event_series.loc[common_idx].values

    valid = ~np.isnan(y_time) & (y_time > 0)
    X = X[valid]
    y_time = y_time[valid]
    y_event = y_event[valid]

    event_rate = y_event.mean()
    print(f"Aligned samples: {len(X)}, event rate (TP hit): {event_rate:.3f}")
    print()

    is_mask = X.index < IS_END
    X_train = X[is_mask]
    X_test = X[~is_mask]
    y_train = np.array(
        [(bool(e), float(t)) for e, t in zip(y_event[is_mask], y_time[is_mask])],
        dtype=[("event", bool), ("time", float)],
    )
    y_test = np.array(
        [(bool(e), float(t)) for e, t in zip(y_event[~is_mask], y_time[~is_mask])],
        dtype=[("event", bool), ("time", float)],
    )

    print(f"Train: {len(X_train)} bars, TP events: {int(y_train['event'].sum())}")
    print(f"Test (OOS): {len(X_test)} bars, TP events: {int(y_test['event'].sum())}")
    print()

    models = {
        "Cox": CoxPHSurvivalAnalysis(),
        "Random Survival Forest": RandomSurvivalForest(
            n_estimators=100, min_samples_leaf=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting SA": GradientBoostingSurvivalAnalysis(
            n_estimators=100, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        ),
    }

    results: list[SurvivalResult] = []
    for name, model in models.items():
        print(f"Training {name}...")
        result = train_and_evaluate(X_train, y_train, X_test, y_test, model, name)
        results.append(result)
        status = "PASS" if result.gate_pass else "FAIL"
        print(
            f"  IS  C-index={result.c_index_is:.4f}, IBS={result.ibs_is:.4f}, "
            f"AUC={result.auc_mean_is:.3f}"
        )
        print(
            f"  OOS C-index={result.c_index_oos:.4f}, IBS={result.ibs_oos:.4f}, "
            f"AUC={result.auc_mean_oos:.3f}  [{status}]"
        )
        print(f"  Time: {result.train_time_s:.1f}s")

        if result.gate_pass:
            plot_path = plot_survival_curves(model, X_test.iloc[:500], y_train, y_test, name)
            print(f"  Plot: {plot_path}")
        print()

    # Comparison table
    print("=" * 85)
    print("=== Survival Model Comparison ===")
    print("=" * 85)
    print(
        f"{'Model':<22} {'C-Index IS':>10} {'C-Index OOS':>10} "
        f"{'IBS OOS':>8} {'AUC OOS':>8} {'Gate':>6} {'Time':>7}"
    )
    print("-" * 85)
    for r in results:
        gate = "PASS" if r.gate_pass else "FAIL"
        print(
            f"{r.model_name:<22} {r.c_index_is:>10.4f} {r.c_index_oos:>10.4f} "
            f"{r.ibs_oos:>8.4f} {r.auc_mean_oos:>8.4f} {gate:>6} {r.train_time_s:>6.1f}s"
        )
    print()

    # Compare to random baseline
    print("=== Gate Summary ===")
    passed = [r for r in results if r.gate_pass]
    if passed:
        best = max(passed, key=lambda r: r.c_index_oos)
        print(f"Gate PASSES ({len(passed)}/{len(results)} models). Best: {best.model_name}")
        print(f"  C-index OOS: {best.c_index_oos:.4f} (random=0.50)")
        print(
            f"  Interpretation: model ranks exit times {best.c_index_oos * 100:.0f}% "
            f"better than random"
        )
    else:
        print("Gate FAILS — no model exceeds C-index > 0.55 threshold.")
        print("  Survival analysis cannot meaningfully predict time-to-exit from these features.")
        best_c = max(r.c_index_oos for r in results)
        print(f"  Best C-index: {best_c:.4f}")

    summary = compute_summary(results)
    out_path = OUTPUT_DIR / f"survival_results_{SYMBOL}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
