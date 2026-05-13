"""
Signal Evaluation Metrics — IC, rank IC, hit rate, information ratio.

Computes evaluation metrics for ML signal quality beyond simple classification
accuracy. Based on ML4T Ch7:06 + Ch12:06 methodology:
- Pearson IC: correlation between feature/predicted values and forward returns
- Rank IC (Spearman): rank correlation, more robust to outliers
- IC Decay: IC computed over multiple forward horizons
- Hit Rate: fraction of correct directional predictions
- Information Ratio: mean IC / std IC across periods

Usage:
    from src.ml.metrics import compute_ic, compute_rank_ic, compute_ic_decay

    ic = compute_ic(features, forward_returns)
    rank_ic = compute_rank_ic(features, forward_returns)
    decay = compute_ic_decay(feature_matrix, forward_returns, horizons=[1, 5, 10, 20])
    summary = ic_summary(features, forward_returns)
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_ic(
    values: pd.Series | pd.DataFrame,
    forward_returns: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """Compute Pearson Information Coefficient (IC) between values and forward returns.

    IC = Pearson correlation between predicted/feature values and actual
    forward returns. Measures linear predictive power.

    Args:
        values: Feature or prediction series/DataFrame (one per column).
        forward_returns: Forward return series or DataFrame.

    Returns:
        DataFrame with columns: feature, ic, t_stat, p_value, n_samples.
    """
    if isinstance(values, pd.Series):
        values = values.to_frame()
    if isinstance(forward_returns, pd.DataFrame):
        forward_returns = forward_returns.iloc[:, 0]

    results = []
    for col in values.columns:
        aligned = pd.DataFrame(
            {
                "values": values[col],
                "returns": forward_returns,
            }
        ).dropna()

        if len(aligned) < 3:
            continue

        ic, p_value = stats.pearsonr(aligned["values"], aligned["returns"])
        n = len(aligned)
        t_stat = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-10))

        results.append(
            {
                "feature": col,
                "ic": ic,
                "t_stat": t_stat,
                "p_value": p_value,
                "n_samples": n,
            }
        )

    return pd.DataFrame(results)


def compute_rank_ic(
    values: pd.Series | pd.DataFrame,
    forward_returns: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """Compute Spearman Rank IC between values and forward returns.

    Rank IC = Spearman rank correlation. More robust to outliers than
    Pearson IC. Standard metric in quantitative finance.

    Args:
        values: Feature or prediction series/DataFrame.
        forward_returns: Forward return series or DataFrame.

    Returns:
        DataFrame with columns: feature, rank_ic, p_value, n_samples.
    """
    if isinstance(values, pd.Series):
        values = values.to_frame()
    if isinstance(forward_returns, pd.DataFrame):
        forward_returns = forward_returns.iloc[:, 0]

    results = []
    for col in values.columns:
        aligned = pd.DataFrame(
            {
                "values": values[col],
                "returns": forward_returns,
            }
        ).dropna()

        if len(aligned) < 3:
            continue

        rank_ic, p_value = stats.spearmanr(aligned["values"], aligned["returns"])

        results.append(
            {
                "feature": col,
                "rank_ic": rank_ic,
                "p_value": p_value,
                "n_samples": len(aligned),
            }
        )

    return pd.DataFrame(results)


def compute_ic_decay(
    feature: pd.Series,
    forward_returns: pd.Series | pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 10, 20),
) -> pd.DataFrame:
    """Compute IC decay across multiple forward horizons.

    Measures how predictive power diminishes over longer horizons.
    Useful for determining optimal trading horizon.

    Args:
        feature: Feature or prediction values (aligned with forward_returns index).
        forward_returns: Forward return series (for 1-day horizon).
        horizons: List of forward horizons in days.

    Returns:
        DataFrame with columns: horizon, ic, rank_ic, decay_ratio.
    """
    if isinstance(forward_returns, pd.DataFrame):
        forward_returns = forward_returns.iloc[:, 0]

    results = []
    base_ic = None
    for h in horizons:
        fwd_h = forward_returns.rolling(window=h).sum().shift(-h)
        aligned = pd.DataFrame({"values": feature, "returns": fwd_h}).dropna()

        if len(aligned) < 3:
            results.append(
                {
                    "horizon": h,
                    "ic": np.nan,
                    "rank_ic": np.nan,
                    "n_samples": len(aligned),
                }
            )
            continue

        ic, _ = stats.pearsonr(aligned["values"], aligned["returns"])
        rank_ic, _ = stats.spearmanr(aligned["values"], aligned["returns"])

        if base_ic is None:
            base_ic = abs(ic) if abs(ic) > 1e-10 else 1.0

        results.append(
            {
                "horizon": h,
                "ic": ic,
                "rank_ic": rank_ic,
                "n_samples": len(aligned),
                "decay_ratio": abs(ic) / base_ic if base_ic and abs(ic) > 1e-10 else 0.0,
            }
        )

    return pd.DataFrame(results)


def compute_hit_rate(
    predictions: pd.Series | np.ndarray,
    actuals: pd.Series | np.ndarray,
    threshold: float = 0.001,
) -> float:
    """Compute hit rate: percentage of predictions with correct sign.

    Args:
        predictions: Predicted values.
        actuals: Actual values.
        threshold: Minimum absolute actual return to consider.

    Returns:
        Hit rate between 0.0 and 1.0.
    """
    pred = np.asarray(predictions, dtype=np.float64)
    actual = np.asarray(actuals, dtype=np.float64)

    mask = np.abs(actual) > threshold
    if mask.sum() == 0:
        return 0.0

    pred_sign = np.sign(pred[mask])
    actual_sign = np.sign(actual[mask])

    return float(np.mean(pred_sign == actual_sign))


def compute_information_ratio(
    ic_series: pd.Series | list[float],
    periods_per_year: int = 252,
) -> float:
    """Compute Information Ratio from IC time series.

    IR = mean(IC) / std(IC) * sqrt(periods_per_year)

    Args:
        ic_series: Time series of per-period IC values.
        periods_per_year: Number of trading periods per year (252 for daily).

    Returns:
        Annualized Information Ratio.
    """
    ic = np.asarray(ic_series)
    ic = ic[~np.isnan(ic)]

    if len(ic) < 2 or np.std(ic) == 0:
        return 0.0

    return float(np.mean(ic) / np.std(ic) * np.sqrt(periods_per_year))


def ic_summary(
    values: pd.DataFrame,
    forward_returns: pd.Series,
) -> pd.DataFrame:
    """Compute comprehensive IC summary for all feature columns.

    Combines Pearson IC, Rank IC, hit rate, and significance into one table.

    Args:
        values: DataFrame of features or predictions (one per column).
        forward_returns: Forward return series.

    Returns:
        DataFrame: feature, ic, rank_ic, abs_ic, abs_rank_ic,
                   t_stat, p_value, hit_rate, n_samples, significant.
    """
    if values.empty or forward_returns.empty:
        return pd.DataFrame()

    ic_df = compute_ic(values, forward_returns)
    rank_ic_df = compute_rank_ic(values, forward_returns)

    results = ic_df.merge(rank_ic_df[["feature", "rank_ic"]], on="feature", how="left")

    all_results = []
    for _, row in results.iterrows():
        col = row["feature"]
        aligned = pd.DataFrame(
            {
                "values": values[col].astype(np.float64),
                "returns": forward_returns.astype(np.float64),
            }
        ).dropna()

        hit_rate = compute_hit_rate(aligned["values"], aligned["returns"])

        all_results.append(
            {
                "feature": col,
                "ic": row["ic"],
                "rank_ic": row["rank_ic"],
                "abs_ic": abs(row["ic"]),
                "abs_rank_ic": abs(row.get("rank_ic", 0)),
                "t_stat": row.get("t_stat", 0),
                "p_value": row.get("p_value", 1),
                "hit_rate": hit_rate,
                "n_samples": row["n_samples"],
                "significant": row.get("p_value", 1) < 0.05,
            }
        )

    summary = pd.DataFrame(all_results)
    if not summary.empty:
        summary = summary.sort_values("abs_rank_ic", ascending=False).reset_index(drop=True)

    return summary


def evaluate_predictions(
    predictions: pd.Series,
    actuals: pd.Series,
) -> dict[str, float]:
    """Evaluate predictions against actual forward returns.

    Computes rank IC, hit rate, and mean error between predictions and
    actual forward returns. No internal shifting is done — both inputs
    must already be aligned.

    Args:
        predictions: Model predictions (scores or probabilities).
        actuals: Actual forward returns (pre-shifted, aligned by index).

    Returns:
        Dict with rank_ic, ic, hit_rate, mean_error, n_samples.
    """
    common = predictions.dropna().index.intersection(actuals.dropna().index)
    if len(common) < 5:
        return {"rank_ic": 0.0, "ic": 0.0, "hit_rate": 0.0, "mean_error": 0.0, "n_samples": 0}

    pred_aligned = predictions.loc[common]
    actual_aligned = actuals.loc[common]

    rank_ic, _ = stats.spearmanr(pred_aligned, actual_aligned)
    ic, _ = stats.pearsonr(pred_aligned, actual_aligned)
    hit_rate = compute_hit_rate(pred_aligned, actual_aligned)

    return {
        "rank_ic": float(rank_ic) if not np.isnan(rank_ic) else 0.0,
        "ic": float(ic) if not np.isnan(ic) else 0.0,
        "hit_rate": hit_rate,
        "mean_error": float((pred_aligned - actual_aligned).abs().mean()),
        "n_samples": len(common),
    }


def filter_features_by_ic(
    values: pd.DataFrame,
    forward_returns: pd.Series,
    min_abs_ic: float = 0.02,
    min_abs_rank_ic: float = 0.02,
) -> list[str]:
    """Filter features that meet IC thresholds.

    Args:
        values: Feature DataFrame.
        forward_returns: Forward return series.
        min_abs_ic: Minimum absolute Pearson IC.
        min_abs_rank_ic: Minimum absolute Spearman rank IC.

    Returns:
        List of feature names meeting both thresholds.
    """
    summary = ic_summary(values, forward_returns)

    mask = (summary["abs_ic"] >= min_abs_ic) & (summary["abs_rank_ic"] >= min_abs_rank_ic)
    selected = summary[mask]["feature"].tolist()

    logger.info(
        f"Feature selection: {len(selected)}/{len(values.columns)} features pass "
        f"IC >= {min_abs_ic} and rank IC >= {min_abs_rank_ic}"
    )
    return selected
