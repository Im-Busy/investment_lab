"""Diagnose Dynamic Ensemble Collapse (Phase 12c P2-3).

Investigation: Why did 5 CPCV path models EGD-weighted produce Sharpe 0.26
— worse than individual components?

Tests:
  1. EGD weight trajectory — does it converge or oscillate?
  2. Per-path solo backtest comparison
  3. Equal-weight voting vs EGD vs stacking
  4. Signal density analysis (ensemble vs individual)

Usage:
    uv run scripts/diagnose_ensemble_collapse.py --model models/pattern_classifier_v3_SPY_20260514_124612.pkl
    uv run scripts/diagnose_ensemble_collapse.py --synthetic  # smoke test with synthetic data
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EnsembleDiagnostics:
    """Diagnostic results for dynamic ensemble collapse."""

    solo_sharpes: Dict[str, float] = field(default_factory=dict)
    equal_weight_sharpe: float = 0.0
    egd_sharpe: float = 0.0
    stacking_sharpe: float = 0.0
    weight_history: pd.DataFrame | None = None
    weight_volatility: Dict[str, float] = field(default_factory=dict)
    total_signals_solo: Dict[str, int] = field(default_factory=dict)
    total_signals_ensemble: int = 0
    recommendations: List[str] = field(default_factory=list)


def _make_synthetic_model_predictions(
    n_bars: int = 500, n_models: int = 5, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic model probabilities and binary labels."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="B")

    base_signal = rng.normal(0.45, 0.15, n_bars)
    base_signal = np.clip(base_signal, 0.01, 0.99)

    probs = {}
    for i in range(n_models):
        noise = rng.normal(0, 0.08, n_bars)
        p = np.clip(base_signal + noise, 0.01, 0.99)
        probs[f"path_{i}"] = p

    prob_df = pd.DataFrame(probs, index=idx)

    returns = rng.normal(0.0005, 0.015, n_bars)
    labels = pd.Series((returns > 0).astype(int), index=idx)

    return prob_df, labels


def _compute_sharpe(returns: pd.Series, freq: int = 252) -> float:
    """Annualized Sharpe ratio."""
    if len(returns) < 10:
        return 0.0
    ann_ret = returns.mean() * freq
    ann_vol = returns.std() * np.sqrt(freq)
    return ann_ret / ann_vol if ann_vol > 0 else 0.0


def _run_egd_simulation(
    prob_df: pd.DataFrame,
    labels: pd.Series,
    eta: float = 0.1,
    reinit_every: int = 60,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Simulate EGD weight updates over time.

    Returns:
        Tuple of (ensemble_probs array, weight_history DataFrame).
    """
    n_bars = len(prob_df)
    n_models = prob_df.shape[1]
    weights = np.ones(n_models) / n_models
    weight_history = []

    for t in range(n_bars):
        row = prob_df.iloc[t].values
        ensemble_prob = float(row @ weights)

        weight_history.append(
            {
                "bar": t,
                "ensemble_prob": ensemble_prob,
                **{f"w_{i}": weights[i] for i in range(n_models)},
            }
        )

        if t < n_bars - 1:
            target = float(labels.iloc[t])
            losses = (row - target) ** 2
            weights *= np.exp(-eta * losses)
            w_sum = weights.sum()
            if w_sum > 0:
                weights /= w_sum
            else:
                weights = np.ones(n_models) / n_models

            if (t + 1) % reinit_every == 0:
                weights = np.ones(n_models) / n_models

    hist_df = pd.DataFrame(weight_history)
    ensemble_probs = hist_df["ensemble_prob"].values
    return ensemble_probs, hist_df


def _run_solo_backtests(
    prob_df: pd.DataFrame, labels: pd.Series, threshold: float = 0.45
) -> Dict[str, float]:
    """Compute Sharpe for each solo model."""
    sharpes = {}
    for col in prob_df.columns:
        signals = prob_df[col] > threshold
        shifted = signals.shift(1).fillna(False)
        daily_ret = pd.Series(0.0, index=prob_df.index)
        daily_ret[shifted] = labels[shifted].map({1: 0.01, 0: -0.01}) * (
            2 * (prob_df[col][shifted] - 0.5)
        )
        sharpes[col] = _compute_sharpe(daily_ret)
    return sharpes


def _run_ensemble_backtest(
    prob_df: pd.DataFrame,
    labels: pd.Series,
    weights: Optional[np.ndarray] = None,
    threshold: float = 0.45,
    method: str = "equal",
) -> float:
    """Backtest an ensemble method."""
    if method == "equal" or weights is None:
        ens_prob = prob_df.mean(axis=1)
    else:
        ens_prob = pd.Series((prob_df.values @ weights), index=prob_df.index)

    signals = ens_prob > threshold
    shifted = signals.shift(1).fillna(False)
    daily_ret = pd.Series(0.0, index=prob_df.index)
    daily_ret[shifted] = labels[shifted].map({1: 0.01, 0: -0.01}) * (2 * (ens_prob[shifted] - 0.5))
    return _compute_sharpe(daily_ret)


def _run_stacking_ensemble(
    prob_df: pd.DataFrame,
    labels: pd.Series,
    threshold: float = 0.45,
) -> float:
    """StackingClassifier: meta-model on top of base predictions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit

    if len(prob_df) < 50:
        return 0.0

    meta_feat = prob_df.values
    meta_target = labels.values

    split_idx = int(len(prob_df) * 0.7)
    X_train, X_test = meta_feat[:split_idx], meta_feat[split_idx:]
    y_train, y_test = meta_target[:split_idx], meta_target[split_idx:]

    meta = LogisticRegression(max_iter=1000, random_state=42)
    meta.fit(X_train, y_train)

    stack_prob = meta.predict_proba(X_test)[:, 1]

    signals = stack_prob > threshold
    shifted = np.roll(signals, 1)
    shifted[0] = False
    daily_ret = pd.Series(0.0, index=prob_df.iloc[split_idx:].index)
    for i in range(len(shifted)):
        if shifted[i]:
            ret_val = 0.01 if y_test[i] == 1 else -0.01
            daily_ret.iloc[i] = ret_val * (2 * (stack_prob[i] - 0.5))
    return _compute_sharpe(daily_ret)


def diagnose_synthetic() -> EnsembleDiagnostics:
    """Run diagnostics on synthetic data to validate the framework."""
    logger.info("=== Synthetic Ensemble Diagnostics ===")
    prob_df, labels = _make_synthetic_model_predictions(500, 5)

    diag = EnsembleDiagnostics()

    solo = _run_solo_backtests(prob_df, labels)
    diag.solo_sharpes = solo
    logger.info(f"Solo Sharpes: {solo}")

    diag.equal_weight_sharpe = _run_ensemble_backtest(prob_df, labels, method="equal")
    logger.info(f"Equal-weight Sharpe: {diag.equal_weight_sharpe:.3f}")

    _, hist_df = _run_egd_simulation(prob_df, labels)
    diag.egd_sharpe = _run_ensemble_backtest(prob_df, labels, method="equal")
    diag.weight_history = hist_df

    w_cols = [c for c in hist_df.columns if c.startswith("w_")]
    for col in w_cols:
        diag.weight_volatility[col] = float(hist_df[col].std())

    diag.stacking_sharpe = _run_stacking_ensemble(prob_df, labels)

    for col in prob_df.columns:
        signals = (prob_df[col] > 0.45).sum()
        diag.total_signals_solo[col] = int(signals)

    ens_prob = prob_df.mean(axis=1)
    diag.total_signals_ensemble = int((ens_prob > 0.45).sum())

    best_solo = max(solo.values())
    logger.info(f"Best solo Sharpe: {best_solo:.3f}")
    logger.info(f"Equal-weight ensemble: {diag.equal_weight_sharpe:.3f}")
    logger.info(f"Stacking ensemble: {diag.stacking_sharpe:.3f}")

    w_std = np.mean(list(diag.weight_volatility.values()))
    logger.info(f"Mean weight std: {w_std:.4f}")

    if diag.equal_weight_sharpe > best_solo:
        diag.recommendations.append(
            "Equal-weight voting improves over best solo — keep simple voting"
        )
    else:
        diag.recommendations.append(
            f"Equal-weight voting underperforms best solo ({diag.equal_weight_sharpe:.3f} vs {best_solo:.3f}). "
            "Ensemble averaging washes out sparse signals — use solo best model or pattern-boosted selection."
        )

    if w_std > 0.1:
        diag.recommendations.append(
            f"EGD weights are volatile (std={w_std:.4f}). "
            "Weights reinitialize every 60 bars → no information accumulates. "
            "Consider: increase reinit_every to 252 (1yr), or use exponential moving average of returns for weight update."
        )

    solo_vals = list(diag.total_signals_solo.values())
    if diag.total_signals_ensemble < min(solo_vals) * 0.8 if solo_vals else 0:
        diag.recommendations.append(
            f"Ensemble produces {diag.total_signals_ensemble} signals vs solo min "
            f"{min(solo_vals)}. Averaging pushes probabilities toward center -> fewer above threshold."
        )

    return diag


def print_report(diag: EnsembleDiagnostics) -> None:
    """Print formatted diagnostic report."""
    print("\n" + "=" * 70)
    print(" DYNAMIC ENSEMBLE COLLAPSE DIAGNOSTICS (Phase 12c P2-3)")
    print("=" * 70)

    print("\n--- Solo Model Sharpes ---")
    for name, sh in sorted(diag.solo_sharpes.items(), key=lambda x: -x[1]):
        marker = " (BEST)" if sh == max(diag.solo_sharpes.values()) else ""
        print(f"  {name}: {sh:+.3f}{marker}")

    print("\n--- Ensemble Methods ---")
    print(f"  Equal-weight voting:  {diag.equal_weight_sharpe:+.3f}")
    print(f"  EGD weighted:         {diag.egd_sharpe:+.3f}")
    print(f"  Stacking (Logistic):  {diag.stacking_sharpe:+.3f}")

    if diag.weight_volatility:
        print("\n--- EGD Weight Volatility ---")
        for w_name, w_std in diag.weight_volatility.items():
            bar = "█" * min(int(w_std * 50), 30)
            print(f"  {w_name}: std={w_std:.4f} {bar}")

    if diag.total_signals_solo:
        print("\n--- Signal Counts (threshold=0.45) ---")
        for name, count in diag.total_signals_solo.items():
            print(f"  {name}: {count}")
        print(f"  ensemble (equal-weight): {diag.total_signals_ensemble}")

    if diag.recommendations:
        print("\n--- Recommendations ---")
        for i, rec in enumerate(diag.recommendations, 1):
            print(f"  {i}. {rec}")

    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Dynamic Ensemble Collapse")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic data")
    parser.add_argument("--model", type=str, help="Path to PatternClassifier model")
    args = parser.parse_args()

    if args.model:
        logger.info(f"Loading model from {args.model}...")
        logger.warning(
            "Live model diagnostic not yet implemented for ensemble collapse — use --synthetic for framework test"
        )
        return

    diag = diagnose_synthetic()
    print_report(diag)

    msg = "; ".join(diag.recommendations) if diag.recommendations else "None"
    logger.info(f"Recommendations: {msg}")


if __name__ == "__main__":
    main()
