"""
Regime-Conditional Analysis — does the model work differently by market state?

Decomposes model performance across market regimes to answer:
- Does the signal degrade in ranging markets?
- Is the model just a bull-market beta proxy?
- When should we NOT trade based on the model?

Regime buckets:
  1. ADX regime (Trending/Ranging/Transition) — from rule-based detector
  2. Volatility regime (High/Normal/Low) — ATR percentile
  3. Market direction (SPY 20d return > 0 / < 0)
  4. Calendar year — alpha decay detection

Usage:
    from src.ml.regime_analysis import regime_conditional_analysis

    results = regime_conditional_analysis(
        predictions=probabilities,
        actuals=labels,
        features=X,
        market_data={"SPY": spy_df},
    )
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

REGIME_COLORS = {
    "Trending": "steelblue",
    "Ranging": "coral",
    "Volatile": "darkorange",
    "Transition": "gray",
    "High Vol": "darkred",
    "Normal Vol": "steelblue",
    "Low Vol": "seagreen",
    "Bull": "green",
    "Bear": "red",
}


def _adx_regime(df: pd.DataFrame, adx_period: int = 14) -> pd.Series:
    """Rule-based ADX regime classification."""
    high, low, close = df["High"], df["Low"], df["Close"]

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(adx_period).mean()

    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0)

    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(adx_period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(adx_period).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(adx_period).mean()

    def _classify(adx_val: float) -> str:
        if pd.isna(adx_val):
            return "Unknown"
        if adx_val > 25:
            return "Trending"
        if adx_val < 20:
            return "Ranging"
        return "Transition"

    return adx.apply(_classify)


def _volatility_regime(df: pd.DataFrame, vol_period: int = 20) -> pd.Series:
    """Volatility regime based on ATR percentile."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(vol_period).mean()
    atr_ratio = atr / close

    p80 = atr_ratio.rolling(252).quantile(0.80)
    p20 = atr_ratio.rolling(252).quantile(0.20)

    def _classify(row: pd.Series) -> str:
        if pd.isna(row["atr"]) or pd.isna(row["p80"]):
            return "Unknown"
        if row["atr"] > row["p80"]:
            return "High Vol"
        if row["atr"] < row["p20"]:
            return "Low Vol"
        return "Normal Vol"

    df_regime = pd.DataFrame({"atr": atr_ratio, "p80": p80, "p20": p20})
    return df_regime.apply(_classify, axis=1)


def regime_conditional_analysis(
    predictions: pd.Series,
    actuals: pd.Series,
    features: pd.DataFrame | None = None,
    market_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Analyze model performance across market regimes.

    Args:
        predictions: Model probability scores (aligned index).
        actuals: Binary labels (1 = profitable, 0 = not).
        features: Feature DataFrame (used for feature-regime interaction).
        market_df: OHLCV DataFrame for regime computation (uses Close only).

    Returns:
        DataFrame: regime, n_samples, pct_positive, rank_ic, hit_rate,
                  mean_pred, mean_actual, performance_rating.
    """
    common = predictions.dropna().index.intersection(actuals.dropna().index)
    if len(common) < 10:
        return pd.DataFrame()

    pred = predictions.loc[common]
    act = actuals.loc[common]

    results: list[dict] = []

    # ── ADX Regime ──
    if market_df is not None and all(c in market_df.columns for c in ["High", "Low", "Close"]):
        adx_r = _adx_regime(market_df).reindex(common)
        for regime_name in ["Trending", "Ranging", "Transition"]:
            mask = adx_r == regime_name
            if mask.sum() < 10:
                continue
            _add_regime_row(results, regime_name, pred[mask], act[mask], "ADX")

        # ── Volatility Regime ──
        vol_r = _volatility_regime(market_df).reindex(common)
        for regime_name in ["High Vol", "Normal Vol", "Low Vol"]:
            mask = vol_r == regime_name
            if mask.sum() < 10:
                continue
            _add_regime_row(results, regime_name, pred[mask], act[mask], "Volatility")

        # ── Market Direction ──
        spy_close = market_df["Close"]
        spy_20d = spy_close.pct_change(20).shift(1).reindex(common)
        bull_mask = spy_20d > 0
        bear_mask = spy_20d < 0
        if bull_mask.sum() >= 10:
            _add_regime_row(
                results, "Bull (SPY 20d > 0)", pred[bull_mask], act[bull_mask], "Direction"
            )
        if bear_mask.sum() >= 10:
            _add_regime_row(
                results, "Bear (SPY 20d < 0)", pred[bear_mask], act[bear_mask], "Direction"
            )

    # ── Calendar Year ──
    if hasattr(common[0], "year"):
        years = pd.Series([d.year for d in common], index=common)
        for year in sorted(years.unique()):
            mask = years == year
            if mask.sum() < 10:
                continue
            _add_regime_row(results, str(year), pred[mask], act[mask], "Year")

    df_results = pd.DataFrame(results)
    if df_results.empty:
        return df_results

    df_results = df_results.sort_values(["category", "n_samples"], ascending=[True, False])
    return df_results.reset_index(drop=True)


def _add_regime_row(
    results: list[dict],
    regime_name: str,
    pred: pd.Series,
    act: pd.Series,
    category: str,
) -> None:
    """Compute metrics for a single regime bucket."""
    n = len(pred)
    if n < 5:
        return

    sr = stats.spearmanr(pred, act)
    rank_ic = float(sr.statistic) if hasattr(sr, "statistic") else float(sr)
    rank_ic = rank_ic if not np.isnan(rank_ic) else 0.0

    # Hit rate using binary predictions at 0.5 threshold
    pred_binary = (pred >= 0.5).astype(int)
    hit_rate = float(np.mean(pred_binary == act))

    # AUC
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(act, pred))
    except (ValueError, ImportError):
        auc = 0.5

    pct_positive = float(act.mean())

    performance = (
        "STRONG"
        if rank_ic > 0.05 and n >= 50
        else "WEAK"
        if rank_ic > 0.01
        else "NOISE"
        if abs(rank_ic) < 0.01
        else "ANTI-SIGNAL"
    )

    results.append(
        {
            "category": category,
            "regime": regime_name,
            "n_samples": n,
            "pct_positive": round(pct_positive, 3),
            "rank_ic": round(rank_ic, 4),
            "hit_rate": round(hit_rate, 3),
            "auc": round(auc, 4),
            "mean_prediction": round(float(pred.mean()), 4),
            "performance": performance,
        }
    )


def plot_regime_analysis(
    df_results: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Plot regime-conditional rank IC as a bar chart."""
    import matplotlib.pyplot as plt

    if df_results.empty:
        return

    df = df_results[df_results["n_samples"] >= 10].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: rank IC by regime
    regimes = df["regime"].tolist()
    ics = df["rank_ic"].tolist()
    colors = [REGIME_COLORS.get(r.split("(")[0].strip(), "gray") for r in regimes]

    axes[0].barh(range(len(regimes)), ics, color=colors, alpha=0.8, height=0.6)
    axes[0].set_yticks(range(len(regimes)))
    axes[0].set_yticklabels(regimes, fontsize=9)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Rank IC (Spearman)")
    axes[0].set_title("Rank IC by Market Regime")
    axes[0].grid(axis="x", alpha=0.3)

    # Right: sample count by regime
    axes[1].barh(range(len(regimes)), df["n_samples"].tolist(), color=colors, alpha=0.8, height=0.6)
    axes[1].set_yticks(range(len(regimes)))
    axes[1].set_yticklabels(regimes, fontsize=9)
    axes[1].set_xlabel("Number of Samples")
    axes[1].set_title("Sample Distribution by Regime")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Regime analysis plot saved to {save_path}")
    plt.close()
