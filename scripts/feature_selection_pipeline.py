"""
Feature Selection Pipeline — ML Phase B1

Builds on FeatureStore and FeatureEngineer to:
1. Load multi-asset data
2. Generate 126 features (82 technical + 44 alpha factors)
3. Compute IC summary (Pearson + Rank IC) per feature
4. Filter by IC threshold (|IC| >= 0.02)
5. Run Sequential Feature Importance (SFI) for refined selection
6. Output selected feature list (~40 high-IC features)

Run with: uv run scripts/feature_selection_pipeline.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import yfinance as yf

from src.ml.experiment_logger import ExperimentLogger
from src.ml.feature_selector import SFISelector
from src.ml.features import FeatureEngineer
from src.ml.metrics import ic_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("reports/feature_selection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "XLF", "XLE", "XLK"]


def load_multi_asset_data(
    tickers: list[str] | None = None,
    start: str = "2015-01-01",
    end: str = "2024-12-31",
) -> dict[str, pd.DataFrame]:
    """Load OHLCV data for multiple tickers.

    Args:
        tickers: List of ticker symbols. Defaults to SPY/QQQ/IWM/GLD/TLT/XLF/XLE/XLK.
        start: Start date.
        end: End date.

    Returns:
        Dict of ticker -> OHLCV DataFrame.
    """
    tickers = tickers or DEFAULT_TICKERS
    data: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        logger.info(f"Loading {ticker}...")
        df = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if len(df) > 0:
            data[ticker] = df
            logger.info(f"  {ticker}: {len(df)} bars")
        else:
            logger.warning(f"  {ticker}: no data, skipping")

    return data


def run_ic_analysis(
    data: dict[str, pd.DataFrame],
    horizon: int = 5,
) -> pd.DataFrame:
    """Run IC analysis across all tickers.

    Args:
        data: Dict of ticker -> OHLCV DataFrame.
        horizon: Forward return horizon in days.

    Returns:
        IC summary DataFrame for all features.
    """
    engine = FeatureEngineer()
    all_ic_summaries = []

    for ticker, df in data.items():
        logger.info(f"Generating features for {ticker}...")
        features = engine.generate_features(df)
        forward_returns = df["Close"].pct_change(horizon).shift(-horizon)

        idx = features.index.intersection(forward_returns.dropna().index)
        if len(idx) < 50:
            logger.warning(f"  {ticker}: insufficient aligned samples, skipping")
            continue

        X = features.loc[idx]
        y = forward_returns.loc[idx]

        col_blacklist = [c for c in X.columns if c.startswith("forward_return_")]
        X_clean = X.drop(columns=col_blacklist, errors="ignore")

        ic_df = ic_summary(X_clean, y)
        ic_df["ticker"] = ticker
        all_ic_summaries.append(ic_df)
        logger.info(
            f"  {ticker}: {len(ic_df)} features with mean |IC|={ic_df['abs_ic'].mean():.4f}"
        )

    if not all_ic_summaries:
        raise ValueError("No tickers produced valid features")

    combined = pd.concat(all_ic_summaries, ignore_index=True)
    return combined


def aggregate_ic_across_tickers(ic_by_ticker: pd.DataFrame) -> pd.DataFrame:
    """Aggregate IC results across tickers: mean |IC| and |rank IC| per feature.

    Args:
        ic_by_ticker: IC summary with ticker column.

    Returns:
        Aggregated IC DataFrame sorted by mean |rank IC| descending.
    """
    agg = (
        ic_by_ticker.groupby("feature")
        .agg(
            mean_abs_ic=("abs_ic", "mean"),
            std_abs_ic=("abs_ic", "std"),
            mean_abs_rank_ic=("abs_rank_ic", "mean"),
            std_abs_rank_ic=("abs_rank_ic", "std"),
            mean_hit_rate=("hit_rate", "mean"),
            n_tickers=("ticker", "nunique"),
            significant_pct=("significant", "mean"),
        )
        .sort_values("mean_abs_rank_ic", ascending=False)
        .reset_index()
    )

    agg["consensus_score"] = (
        agg["mean_abs_rank_ic"] * 0.5 + agg["mean_abs_ic"] * 0.3 + agg["significant_pct"] * 0.2
    )

    return agg.sort_values("consensus_score", ascending=False).reset_index(drop=True)


def run_sfi_selection(
    data: dict[str, pd.DataFrame],
    ic_passed_features: list[str],
    horizon: int = 5,
) -> tuple[list[str], pd.DataFrame]:
    """Run SFI on IC-filtered features using aggregated multi-asset data.

    Args:
        data: Dict of ticker -> OHLCV DataFrame.
        ic_passed_features: Features that passed IC threshold.
        horizon: Forward return horizon.

    Returns:
        Tuple of (selected_features, sfi_details).
    """
    engine = FeatureEngineer()

    all_X_parts = []
    all_y_parts = []
    for ticker, df in data.items():
        features = engine.generate_features(df)
        forward_returns = df["Close"].pct_change(horizon).shift(-horizon)

        idx = features.index.intersection(forward_returns.dropna().index)
        if len(idx) < 50:
            continue

        available = [f for f in ic_passed_features if f in features.columns]
        if not available:
            continue

        X_sub = features.loc[idx, available]
        y_sub = forward_returns.loc[idx]
        all_X_parts.append(X_sub)
        all_y_parts.append(y_sub)

    if not all_X_parts:
        logger.warning("No data available for SFI")
        return ic_passed_features, pd.DataFrame()

    X_all = pd.concat(all_X_parts)
    y_all = pd.concat(all_y_parts)

    logger.info(f"SFI: {len(X_all)} samples, {len(X_all.columns)} features")

    sfi = SFISelector(min_ic=0.02, max_features=40, min_ic_improvement=0.005)
    sfi.fit(X_all, y_all, n_splits=3)

    details = sfi.to_dataframe()
    return sfi.get_selected_features(), details


def run_pipeline(
    tickers: list[str] | None = None,
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    horizon: int = 5,
    min_abs_ic: float = 0.02,
) -> dict:
    """Run the full feature selection pipeline.

    Args:
        tickers: List of ticker symbols.
        start: Start date.
        end: End date.
        horizon: Forward return horizon.
        min_abs_ic: Minimum absolute IC threshold.

    Returns:
        Dict with pipeline results.
    """
    tickers = tickers or DEFAULT_TICKERS

    experiment = ExperimentLogger(
        run_id=f"feature_selection_{datetime.now():%Y%m%d_%H%M%S}",
        base_dir="experiments",
        description=f"Feature selection pipeline: {','.join(tickers)} {start}:{end} h={horizon}",
    )

    logger.info("=" * 70)
    logger.info("Feature Selection Pipeline")
    logger.info(f"  Tickers: {tickers}")
    logger.info(f"  Period: {start} to {end}")
    logger.info(f"  Horizon: {horizon}d")
    logger.info("=" * 70)

    # Step 1: Load data
    logger.info("\n[1/5] Loading multi-asset data")
    data = load_multi_asset_data(tickers, start, end)
    logger.info(f"  Loaded {len(data)} tickers")

    # Step 2: Generate features and compute IC
    logger.info("\n[2/5] Generating features and computing IC")
    ic_by_ticker = run_ic_analysis(data, horizon)

    agg_ic = aggregate_ic_across_tickers(ic_by_ticker)
    ic_by_ticker.to_csv(OUTPUT_DIR / "ic_by_ticker.csv", index=False)
    agg_ic.to_csv(OUTPUT_DIR / "aggregated_ic.csv", index=False)

    n_total = len(agg_ic)
    n_pass_ic = int((agg_ic["mean_abs_rank_ic"] >= min_abs_ic).sum())
    logger.info(f"  {n_pass_ic}/{n_total} features pass |IC| >= {min_abs_ic}")

    # Step 3: Filter by IC
    logger.info("\n[3/5] IC-based filtering")
    ic_passed = agg_ic[agg_ic["mean_abs_rank_ic"] >= min_abs_ic]["feature"].tolist()
    logger.info(f"  {len(ic_passed)} features selected by IC filter")

    # Step 4: Run SFI
    logger.info("\n[4/5] Sequential Feature Importance (SFI)")
    if len(ic_passed) <= 40:
        logger.info("  IC-passed features already <= 40, skipping SFI")
        selected_features = ic_passed
        sfi_details = pd.DataFrame({"feature": ic_passed, "method": "ic_filter"})
    else:
        try:
            selected_features, sfi_details = run_sfi_selection(data, ic_passed, horizon)
            if sfi_details is not None:
                sfi_details.to_csv(OUTPUT_DIR / "sfi_details.csv", index=False)
            logger.info(f"  {len(selected_features)} features selected by SFI")
        except Exception as e:
            logger.warning(f"  SFI failed: {e}, falling back to IC filter")
            selected_features = ic_passed[:40]
            sfi_details = pd.DataFrame(
                {"feature": selected_features, "method": "ic_filter_fallback"}
            )

    # Step 5: Save results
    logger.info("\n[5/5] Saving results")

    selected_df = pd.DataFrame({"feature": selected_features})
    selected_df.to_csv(OUTPUT_DIR / "selected_features.csv", index=False)

    top_by_ic = agg_ic.head(20)[
        ["feature", "mean_abs_rank_ic", "mean_abs_ic", "consensus_score", "n_tickers"]
    ]
    logger.info("\nTop 20 features by IC:")
    for _, row in top_by_ic.iterrows():
        marker = " [SELECTED]" if row["feature"] in selected_features else ""
        logger.info(
            f"  {row['feature']:<30} |IC|={row['mean_abs_rank_ic']:.4f} "
            f"IC={row['mean_abs_ic']:.4f} N={int(row['n_tickers'])}{marker}"
        )

    experiment.log_config(
        hyperparams={
            "tickers": tickers,
            "start": start,
            "end": end,
            "horizon": horizon,
            "min_abs_ic": min_abs_ic,
            "n_total_features": n_total,
            "n_selected": len(selected_features),
            "method": "ic_filter_and_sfi",
        }
    )
    experiment.log_summary_verdict(
        mean_oos_metrics={
            "n_total_features": n_total,
            "n_selected": len(selected_features),
            "reduction_pct": round((1 - len(selected_features) / n_total) * 100, 1),
        },
        n_folds=0,
    )

    return {
        "n_total": n_total,
        "n_selected": len(selected_features),
        "selected_features": selected_features,
        "agg_ic": agg_ic,
        "output_dir": str(OUTPUT_DIR),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature Selection Pipeline")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="Ticker symbols")
    parser.add_argument("--start", default="2015-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument("--horizon", type=int, default=5, help="Forward return horizon")
    parser.add_argument("--min-ic", type=float, default=0.02, help="Minimum |IC| threshold")
    args = parser.parse_args()

    result = run_pipeline(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        horizon=args.horizon,
        min_abs_ic=args.min_ic,
    )

    logger.info("=" * 70)
    logger.info(f"Pipeline complete: {result['n_selected']}/{result['n_total']} features selected")
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info("=" * 70)
