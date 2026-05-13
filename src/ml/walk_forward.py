"""
Walk-Forward Validation — honest OOS testing by rolling through time.

Implements rolling walk-forward: train on expanding window, predict next step,
advance. This mimics production deployment and is immune to the random-split
optimism of k-fold CV.

Key difference from PurgedKFold:
- PurgedKFold: K random splits within time range, with purging
- Walk-forward: Chronological train → step → expand → repeat
  This is the gold standard. If a signal works in walk-forward, it works.

Usage:
    from src.ml.walk_forward import walk_forward_validation

    results = walk_forward_validation(
        model_class=PatternClassifier,
        X=features,
        y=labels,
        initial_train_days=3 * 252,
        step_days=6 * 21,
        model_type="catboost",
    )
    # results.rank_ic by step, results.predictions vs actuals
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Results from walk-forward validation."""

    steps: list[dict] = field(default_factory=list)
    predictions: pd.Series | None = None
    actuals: pd.Series | None = None

    @property
    def mean_rank_ic(self) -> float:
        ics = [s["rank_ic"] for s in self.steps if not np.isnan(s["rank_ic"])]
        return float(np.mean(ics)) if ics else 0.0

    @property
    def std_rank_ic(self) -> float:
        ics = [s["rank_ic"] for s in self.steps if not np.isnan(s["rank_ic"])]
        return float(np.std(ics)) if len(ics) > 1 else 0.0

    @property
    def mean_hit_rate(self) -> float:
        rates = [s["hit_rate"] for s in self.steps if not np.isnan(s["hit_rate"])]
        return float(np.mean(rates)) if rates else 0.0

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def summary(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "mean_rank_ic": self.mean_rank_ic,
            "std_rank_ic": self.std_rank_ic,
            "mean_hit_rate": self.mean_hit_rate,
            "information_ratio": (
                self.mean_rank_ic / max(self.std_rank_ic, 1e-10) if self.std_rank_ic > 0 else 0.0
            ),
            "min_rank_ic": min((s["rank_ic"] for s in self.steps), default=0.0),
            "max_rank_ic": max((s["rank_ic"] for s in self.steps), default=0.0),
        }


def walk_forward_validation(
    model_class: type,
    X: pd.DataFrame,
    y: pd.Series,
    initial_train_days: int = 3 * 252,
    step_days: int = 6 * 21,
    model_type: str = "catboost",
    model_kwargs: dict[str, Any] | None = None,
    retrain: bool = True,
) -> WalkForwardResult:
    """Rolling walk-forward validation.

    Train on [start, start+initial_train_days], predict next step_days,
    expand window by step_days, repeat. Evaluates predictions vs actuals
    using rank IC and hit rate at each step.

    Args:
        model_class: ML model class with train() and predict() methods.
        X: Feature DataFrame with datetime index, sorted chronologically.
        y: Label series (aligned with X index). Must be pre-shifted
           forward-looking labels (no internal shifting done here).
        initial_train_days: Number of days (rows) for initial training window.
        step_days: Number of days (rows) per forward step.
        model_type: Passed to model_class constructor.
        model_kwargs: Additional kwargs for model_class constructor.
        retrain: If True, retrain at each step (expanding window).
                 If False, train once on initial window and predict all steps.

    Returns:
        WalkForwardResult with per-step metrics and aggregated summary.

    Raises:
        ValueError: If initial_train_days + step_days exceeds data length.
    """
    n_total = len(X)
    if n_total < initial_train_days + step_days:
        raise ValueError(
            f"Not enough data for walk-forward: {n_total} rows, "
            f"need at least {initial_train_days + step_days} "
            f"(initial_train={initial_train_days} + step={step_days})"
        )

    from scipy import stats

    kwargs = model_kwargs or {}
    result = WalkForwardResult()
    all_predictions: list[float] = []
    all_actuals: list[float] = []
    all_dates: list[pd.Timestamp] = []

    step = 0
    train_end = initial_train_days

    while train_end + step_days <= n_total:
        test_start = train_end
        test_end = min(train_end + step_days, n_total)

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        if len(y_train) < 50 or len(y_test) < 10:
            logger.warning(f"Step {step + 1}: insufficient samples, skipping")
            train_end += step_days
            step += 1
            continue

        model = model_class(model_type=model_type, n_estimators=100, **kwargs)
        model.train(X_train, y_train)

        preds_df = model.predict(X_test)
        if isinstance(preds_df, pd.DataFrame) and "probability_profitable" in preds_df.columns:
            preds = preds_df["probability_profitable"]
        elif isinstance(preds_df, pd.Series):
            preds = preds_df
        else:
            preds = pd.Series(preds_df, index=X_test.index)

        # compute rank IC between predictions and actuals
        common = y_test.dropna().index.intersection(preds.dropna().index)
        if len(common) < 5:
            train_end += step_days
            step += 1
            continue

        pred_aligned = preds.loc[common]
        actual_aligned = y_test.loc[common]

        sr_result = stats.spearmanr(pred_aligned, actual_aligned)
        rank_ic = (
            float(sr_result.statistic) if hasattr(sr_result, "statistic") else float(sr_result)
        )
        rank_ic = rank_ic if not np.isnan(rank_ic) else 0.0

        pred_sign = np.sign(pred_aligned.values)
        actual_sign = np.sign(actual_aligned.values)
        hit_rate = float(np.mean(pred_sign == actual_sign))

        test_auc = _compute_auc(actual_aligned > 0, pred_aligned)

        train_start = X_train.index[0]
        train_end = X_train.index[-1]
        test_start = X_test.index[0]
        test_end = X_test.index[-1]

        step_data = {
            "step": step + 1,
            "train_start": str(train_start.date())
            if hasattr(train_start, "date")
            else str(train_start),
            "train_end": str(train_end.date()) if hasattr(train_end, "date") else str(train_end),
            "test_start": str(test_start.date())
            if hasattr(test_start, "date")
            else str(test_start),
            "test_end": str(test_end.date()) if hasattr(test_end, "date") else str(test_end),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "rank_ic": rank_ic,
            "hit_rate": hit_rate,
            "test_auc": test_auc,
            "mean_predicted": float(pred_aligned.mean()),
            "mean_actual": float(actual_aligned.mean()),
        }
        result.steps.append(step_data)

        all_predictions.extend(pred_aligned.tolist())
        all_actuals.extend(actual_aligned.tolist())
        all_dates.extend(common.tolist())

        logger.info(
            f"Step {step + 1}: {step_data['test_start']} → {step_data['test_end']} "
            f"| n={len(y_test)} | rank_IC={rank_ic:.4f} | hit_rate={hit_rate:.3f}"
        )

        if retrain:
            train_end += step_days
        step += 1

    if all_dates:
        result.predictions = pd.Series(all_predictions, index=all_dates, name="predictions")
        result.actuals = pd.Series(all_actuals, index=all_dates, name="actuals")

    summary = result.summary()
    logger.info(
        f"Walk-forward complete: {result.n_steps} steps | "
        f"mean rank_IC={summary['mean_rank_ic']:.4f} ± {summary['std_rank_ic']:.4f} | "
        f"IR={summary['information_ratio']:.2f}"
    )

    return result


def _compute_auc(y_true_bool: pd.Series, y_score: pd.Series) -> float:
    """Compute AUC. Returns 0.5 if only one class present."""
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true_bool, y_score))
    except (ValueError, ImportError):
        return 0.5


@dataclass
class PerTickerWFResult:
    """Walk-forward results for a single ticker using a basket-trained model."""

    ticker: str
    steps: list[dict] = field(default_factory=list)
    predictions: pd.Series | None = None
    actuals: pd.Series | None = None

    @property
    def mean_rank_ic(self) -> float:
        ics = [s["rank_ic"] for s in self.steps if not np.isnan(s["rank_ic"])]
        return float(np.mean(ics)) if ics else 0.0

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def summary(self) -> dict:
        ics = [s["rank_ic"] for s in self.steps if not np.isnan(s["rank_ic"])]
        return {
            "ticker": self.ticker,
            "n_steps": self.n_steps,
            "mean_rank_ic": round(self.mean_rank_ic, 4),
            "std_rank_ic": round(float(np.std(ics)), 4) if len(ics) > 1 else 0.0,
            "min_rank_ic": round(min(ics), 4) if ics else 0.0,
            "max_rank_ic": round(max(ics), 4) if ics else 0.0,
        }


def walk_forward_per_ticker(
    model: Any,
    tickers: list[str],
    data_dir: str = "data/raw",
    initial_train_days: int = 3 * 252,
    step_days: int = 6 * 21,
    horizon: int = 5,
) -> dict[str, PerTickerWFResult]:
    """Run per-ticker walk-forward using a pre-trained basket model.

    For each ticker:
    1. Load OHLCV data, extract features (backward-looking only)
    2. Generate triple-barrier labels
    3. Walk forward chronologically using the basket-trained model (no retraining)
    4. Compute per-step rank IC and aggregate

    Args:
        model: Pre-trained PatternClassifier (basket-trained).
        tickers: List of ticker symbols.
        data_dir: Directory containing {ticker}_daily.csv files.
        initial_train_days: Starting training window size in days.
        step_days: Forward step size in days.
        horizon: Triple-barrier time limit in bars.

    Returns:
        Dict mapping ticker → PerTickerWFResult.
    """
    from pathlib import Path

    from scipy import stats

    from src.ml.feature_engineering import FeatureExtractor
    from src.ml.triple_barrier import TripleBarrierLabeler

    feature_cols = model.feature_names_
    extractor = FeatureExtractor()
    results: dict[str, PerTickerWFResult] = {}

    for ticker in tickers:
        path = Path(data_dir) / f"{ticker}_daily.csv"
        if not path.exists():
            logger.warning(f"No data for {ticker}, skipping")
            continue

        df = pd.read_csv(path, parse_dates=True, index_col=0).dropna()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = 0 if col == "Volume" else df["Close"]

        logger.info(
            f"Walk-forward {ticker}: {len(df)} bars, "
            f"initial_train={initial_train_days}d, step={step_days}d"
        )

        # Extract features (backward-looking only)
        features = extractor.extract_all_features(df, include_forward_returns=False)
        missing = [c for c in feature_cols if c not in features.columns]
        cols = [c for c in feature_cols if c in features.columns]
        fx = features[cols].ffill().bfill()
        # Reindex to match model's expected feature set (fills missing cross-asset
        # features with 0 — unavoidable for per-ticker evaluation)
        for c in missing:
            fx[c] = 0.0
        fx = fx[feature_cols]

        # Generate triple-barrier labels
        atr = _compute_atr(df, period=14)
        labeler = TripleBarrierLabeler(atr_mult_tp=1.5, atr_mult_sl=1.0)
        labels = labeler.fit(df["Close"], df["High"], df["Low"], time_limit=horizon, atr_series=atr)

        ticker_result = PerTickerWFResult(ticker=ticker)
        all_predictions: list[float] = []
        all_actuals: list[float] = []
        all_dates: list[pd.Timestamp] = []

        step = 0
        train_end = initial_train_days

        while train_end + step_days <= len(df) - horizon - 1:
            test_start = train_end
            test_end = min(train_end + step_days, len(df) - horizon - 1)

            y_test = labels.iloc[test_start:test_end]
            X_test = fx.iloc[test_start:test_end]

            valid = y_test.notna() & X_test.notna().all(axis=1)
            X_test = X_test[valid]
            y_test = y_test[valid]

            if len(y_test) < 5:
                train_end += step_days
                step += 1
                continue

            try:
                preds_df = model.predict(X_test)
                if (
                    isinstance(preds_df, pd.DataFrame)
                    and "probability_profitable" in preds_df.columns
                ):
                    preds = preds_df["probability_profitable"]
                elif isinstance(preds_df, pd.Series):
                    preds = preds_df
                else:
                    preds = pd.Series(preds_df, index=X_test.index)
            except Exception:
                train_end += step_days
                step += 1
                continue

            common = preds.dropna().index.intersection(y_test.dropna().index)
            if len(common) < 5:
                train_end += step_days
                step += 1
                continue

            pred_aligned = preds.loc[common]
            actual_aligned = y_test.loc[common]

            sr_result = stats.spearmanr(pred_aligned, actual_aligned)
            rank_ic = (
                float(sr_result.statistic) if hasattr(sr_result, "statistic") else float(sr_result)
            )
            rank_ic = rank_ic if not np.isnan(rank_ic) else 0.0

            step_data = {
                "step": step + 1,
                "test_start": str(X_test.index[0].date())
                if hasattr(X_test.index[0], "date")
                else str(X_test.index[0]),
                "test_end": str(X_test.index[-1].date())
                if hasattr(X_test.index[-1], "date")
                else str(X_test.index[-1]),
                "n_test": len(y_test),
                "rank_ic": rank_ic,
                "mean_predicted": float(pred_aligned.mean()),
                "mean_actual": float(actual_aligned.mean()),
            }
            ticker_result.steps.append(step_data)

            all_predictions.extend(pred_aligned.tolist())
            all_actuals.extend(actual_aligned.tolist())
            all_dates.extend(common.tolist())

            train_end += step_days
            step += 1

        if all_dates:
            ticker_result.predictions = pd.Series(
                all_predictions, index=all_dates, name="predictions"
            )
            ticker_result.actuals = pd.Series(all_actuals, index=all_dates, name="actuals")

        logger.info(
            f"  {ticker}: {ticker_result.n_steps} steps, "
            f"mean rank_IC={ticker_result.mean_rank_ic:.4f}"
        )
        results[ticker] = ticker_result

    return results


def walk_forward_results_to_df(result: WalkForwardResult) -> pd.DataFrame:
    """Convert walk-forward result steps to a DataFrame for plotting."""
    return pd.DataFrame(result.steps)


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    return tr.rolling(period).mean()


def plot_walk_forward(
    result: WalkForwardResult,
    save_path: str | None = None,
) -> None:
    """Plot walk-forward rank IC over time."""
    import matplotlib.pyplot as plt

    df = walk_forward_results_to_df(result)
    if df.empty:
        return

    try:
        test_midpoints = [pd.Timestamp(r["test_start"]) for _, r in df.iterrows()]
    except (ValueError, TypeError):
        test_midpoints = list(range(1, len(df) + 1))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(test_midpoints, df["rank_ic"], "o-", color="steelblue", linewidth=1.5)
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[0].axhline(
        result.mean_rank_ic,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Mean IC = {result.mean_rank_ic:.4f}",
    )
    axes[0].set_ylabel("Rank IC (Spearman)")
    axes[0].set_title("Walk-Forward Rank IC by Step")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].bar(test_midpoints, df["hit_rate"], color="coral", alpha=0.7, width=20)
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_ylabel("Hit Rate")
    axes[1].set_title("Hit Rate by Step")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Walk-forward plot saved to {save_path}")
    plt.close()
