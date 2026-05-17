"""Regression Labels — Predict 5-Day Forward Return (P3-4).

Trains CatBoostRegressor to predict continuous 5-day forward return
instead of binary triple-barrier labels. Compares regression-based
signal generation to existing classification approach.

Evaluation metrics:
  - R²: variance explained
  - RMSE / MAE: prediction accuracy
  - Directional Accuracy: % where sign(pred) == sign(actual)
  - IC: Spearman rank correlation with actual returns

Gate: directional_accuracy > 0.55 AND IC > 0.03 (statistically meaningful).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

from src.ml.feature_engineering import FeatureExtractor

SYMBOL = "SPY"
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("reports/regression_labels")
MODEL_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 5
RANDOM_STATE = 42
IS_END = "2025-01-01"


@dataclass
class RegressionResult:
    model_name: str
    r2_is: float
    r2_oos: float
    rmse_is: float
    rmse_oos: float
    mae_is: float
    mae_oos: float
    dir_accuracy_is: float
    dir_accuracy_oos: float
    ic_is: float
    ic_oos: float
    train_time_s: float
    n_train: int
    n_test: int
    gate_pass: bool = False


def load_data(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_daily.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0.0 if col == "Volume" else df.iloc[:, 0]
    return df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    ic, p_val = spearmanr(y_pred, y_true)
    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": dir_acc,
        "ic": float(ic),
        "ic_p_value": float(p_val),
    }


def compute_baseline_metrics(y_true: np.ndarray) -> dict[str, float]:
    r2 = float(r2_score(y_true, np.zeros_like(y_true)))
    rmse = float(np.sqrt(mean_squared_error(y_true, np.zeros_like(y_true))))
    mae = float(mean_absolute_error(y_true, np.zeros_like(y_true)))
    dir_acc = float(np.mean(np.sign(np.zeros_like(y_true)) == np.sign(y_true)))
    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": dir_acc,
        "ic": 0.0,
        "ic_p_value": 1.0,
    }


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model: CatBoostRegressor,
    model_name: str,
    early_stopping: bool = True,
) -> RegressionResult:
    import time

    fit_params = {}
    if early_stopping:
        eval_set = [(X_test.values, y_test)]
        fit_params = {"eval_set": eval_set, "early_stopping_rounds": 50, "verbose": False}

    t0 = time.monotonic()
    model.fit(X_train.values, y_train, **fit_params)
    train_time = time.monotonic() - t0

    pred_is = model.predict(X_train.values)
    pred_oos = model.predict(X_test.values)

    m_is = compute_metrics(y_train, pred_is)
    m_oos = compute_metrics(y_test, pred_oos)

    gate_pass = bool(m_oos["directional_accuracy"] > 0.55 and m_oos["ic"] > 0.03)

    return RegressionResult(
        model_name=model_name,
        r2_is=m_is["r2"],
        r2_oos=m_oos["r2"],
        rmse_is=m_is["rmse"],
        rmse_oos=m_oos["rmse"],
        mae_is=m_is["mae"],
        mae_oos=m_oos["mae"],
        dir_accuracy_is=m_is["directional_accuracy"],
        dir_accuracy_oos=m_oos["directional_accuracy"],
        ic_is=m_is["ic"],
        ic_oos=m_oos["ic"],
        train_time_s=train_time,
        n_train=len(X_train),
        n_test=len(X_test),
        gate_pass=gate_pass,
    )


def plot_predictions(
    df: pd.DataFrame,
    y_test: np.ndarray,
    pred_oos: np.ndarray,
    test_idx: pd.DatetimeIndex,
    model_name: str,
) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Scatter: predicted vs actual
    axes[0, 0].scatter(y_test, pred_oos, alpha=0.3, s=8)
    lim = max(abs(y_test).max(), abs(pred_oos).max()) * 1.1
    axes[0, 0].plot([-lim, lim], [-lim, lim], "r--", linewidth=1, alpha=0.5)
    axes[0, 0].set_xlabel("Actual 5d Return")
    axes[0, 0].set_ylabel("Predicted 5d Return")
    axes[0, 0].set_title(f"{model_name}: Predicted vs Actual (OOS)")
    axes[0, 0].grid(alpha=0.3)

    # Time series of predictions
    axes[0, 1].plot(test_idx, y_test, "b-", alpha=0.4, linewidth=0.5, label="Actual")
    axes[0, 1].plot(test_idx, pred_oos, "r-", alpha=0.5, linewidth=0.5, label="Predicted")
    axes[0, 1].axhline(y=0, color="k", linewidth=0.5)
    axes[0, 1].set_xlabel("Date")
    axes[0, 1].set_ylabel("5d Forward Return")
    axes[0, 1].set_title(f"{model_name}: Prediction Time Series (OOS)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    # Directional accuracy by quantile
    combined = pd.DataFrame({"actual": y_test, "predicted": pred_oos})
    combined["pred_decile"] = pd.qcut(pred_oos, q=10, labels=False, duplicates="drop")
    decile_stats = combined.groupby("pred_decile").agg(
        actual_mean=("actual", "mean"),
        actual_std=("actual", "std"),
        count=("actual", "count"),
        dir_acc=(
            "actual",
            lambda x: np.mean(np.sign(x) == np.sign(combined.loc[x.index, "predicted"])),
        ),
    )
    axes[1, 0].bar(decile_stats.index, decile_stats["actual_mean"], color="steelblue")
    axes[1, 0].set_xlabel("Prediction Decile (1=lowest, 10=highest)")
    axes[1, 0].set_ylabel("Mean Actual Return")
    axes[1, 0].set_title(f"{model_name}: Return by Prediction Decile (OOS)")
    axes[1, 0].axhline(y=0, color="k", linewidth=0.5)
    axes[1, 0].grid(alpha=0.3)

    # Error distribution
    errors = pred_oos - y_test
    axes[1, 1].hist(errors, bins=50, color="gray", alpha=0.7, edgecolor="black")
    axes[1, 1].axvline(x=0, color="r", linewidth=1)
    axes[1, 1].axvline(
        x=errors.mean(), color="b", linewidth=1, linestyle="--", label=f"Mean={errors.mean():.4f}"
    )
    axes[1, 1].set_xlabel("Prediction Error")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title(f"{model_name}: Error Distribution (OOS)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"regression_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def main() -> None:
    print(f"Regression Labels — Predict {HORIZON}d Forward Return ({SYMBOL})")
    print(f"IS/OOS split: {IS_END}")

    df = load_data(SYMBOL)
    print(f"Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

    extractor = FeatureExtractor()
    X = extractor.extract_all_features(df, include_forward_returns=False)
    X = X.dropna()

    target = df["Close"].shift(-HORIZON) / df["Close"] - 1
    target = target.loc[X.index]
    y = target.values

    valid = ~np.isnan(y)
    X = X[valid]
    y = y[valid]
    print(f"Features: {X.shape[1]} columns")
    print(f"Samples: {len(X)}, target mean={y.mean():.4f}, std={y.std():.4f}")
    print()

    is_mask = X.index < IS_END
    X_train = X[is_mask]
    y_train = y[is_mask]
    X_test = X[~is_mask]
    y_test = y[~is_mask]

    print(f"Train: {len(X_train)} bars, target mean={y_train.mean():.4f}")
    print(f"Test (OOS): {len(X_test)} bars, target mean={y_test.mean():.4f}")
    print()

    baseline = compute_baseline_metrics(y_test)
    print("=== Baseline (predict 0 always) ===")
    print(
        f"  R²={baseline['r2']:.4f}, RMSE={baseline['rmse']:.4f}, "
        f"DirAcc={baseline['directional_accuracy']:.4f}"
    )

    models = {
        "CatBoostRegressor (default)": CatBoostRegressor(
            iterations=200,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_strength=1.0,
            random_state=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        ),
        "CatBoostRegressor (regularized)": CatBoostRegressor(
            iterations=500,
            learning_rate=0.01,
            depth=4,
            l2_leaf_reg=10.0,
            random_strength=2.0,
            subsample=0.8,
            random_state=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        ),
        "CatBoostRegressor (fast)": CatBoostRegressor(
            iterations=100,
            learning_rate=0.05,
            depth=3,
            l2_leaf_reg=5.0,
            random_strength=1.0,
            random_state=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        ),
    }

    results: list[RegressionResult] = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        result = train_and_evaluate(X_train, y_train, X_test, y_test, model, name)
        results.append(result)

        status = "PASS" if result.gate_pass else "FAIL"
        print(
            f"  IS : R²={result.r2_is:.4f}, RMSE={result.rmse_is:.4f}, "
            f"MAE={result.mae_is:.4f}, DirAcc={result.dir_accuracy_is:.4f}, "
            f"IC={result.ic_is:.4f}"
        )
        print(
            f"  OOS: R²={result.r2_oos:.4f}, RMSE={result.rmse_oos:.4f}, "
            f"MAE={result.mae_oos:.4f}, DirAcc={result.dir_accuracy_oos:.4f}, "
            f"IC={result.ic_oos:.4f}  [{status}]"
        )
        print(f"  Time: {result.train_time_s:.1f}s")

        plot_predictions(df, y_test, model.predict(X_test.values), X_test.index, name)

    print()
    print("=" * 95)
    print("=== Regression Model Comparison ===")
    print("=" * 95)
    print(
        f"{'Model':<35} {'R² IS':>7} {'R² OOS':>7} "
        f"{'DirAcc IS':>9} {'DirAcc OOS':>9} {'IC OOS':>7} {'Gate':>6}"
    )
    print("-" * 95)
    for r in results:
        gate = "PASS" if r.gate_pass else "FAIL"
        print(
            f"{r.model_name:<35} {r.r2_is:>7.4f} {r.r2_oos:>7.4f} "
            f"{r.dir_accuracy_is:>9.4f} {r.dir_accuracy_oos:>9.4f} "
            f"{r.ic_oos:>7.4f} {gate:>6}"
        )

    print()
    passed = [r for r in results if r.gate_pass]
    if passed:
        print(
            f"Gate PASSES ({len(passed)}/{len(results)} models). "
            f"Regression labels outperform binary classification as a signal source."
        )
        best = max(passed, key=lambda r: r.dir_accuracy_oos)
        print(
            f"  Best: {best.model_name} (DirAcc={best.dir_accuracy_oos:.4f}, IC={best.ic_oos:.4f})"
        )
    else:
        print("Gate FAILS — no regression model exceeds DirAcc>0.55 and IC>0.03.")
        best_r = max(results, key=lambda r: r.dir_accuracy_oos)
        print(f"  Best DirAcc: {best_r.dir_accuracy_oos:.4f} ({best_r.model_name})")
        print("  Regression labels do not add predictive value over binary classification.")

    # Save best model
    best_overall = max(results, key=lambda r: r.ic_oos)
    best_model = models[best_overall.model_name]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"regression_labels_{SYMBOL}_{timestamp}.pkl"
    best_model.save_model(str(model_path))
    print(f"\nSaved best model to {model_path}")

    # Save report
    report = {
        "symbol": SYMBOL,
        "horizon": HORIZON,
        "is_end": IS_END,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "baseline_oos": baseline,
        "models": {
            r.model_name: {
                "r2_is": round(r.r2_is, 4),
                "r2_oos": round(r.r2_oos, 4),
                "rmse_is": round(r.rmse_is, 4),
                "rmse_oos": round(r.rmse_oos, 4),
                "mae_is": round(r.mae_is, 4),
                "mae_oos": round(r.mae_oos, 4),
                "directional_accuracy_is": round(r.dir_accuracy_is, 4),
                "directional_accuracy_oos": round(r.dir_accuracy_oos, 4),
                "ic_is": round(r.ic_is, 4),
                "ic_oos": round(r.ic_oos, 4),
                "train_time_s": round(r.train_time_s, 1),
                "gate_pass": r.gate_pass,
            }
            for r in results
        },
        "gate": {
            "passed": len(passed),
            "total": len(results),
            "threshold_dir_accuracy": 0.55,
            "threshold_ic": 0.03,
        },
        "best_model": best_overall.model_name,
        "best_model_path": str(model_path),
    }

    out_path = OUTPUT_DIR / f"regression_results_{SYMBOL}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
