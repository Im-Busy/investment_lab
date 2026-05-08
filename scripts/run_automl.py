"""AutoML Baseline — Accuracy ceiling via AutoGluon.

Trains 10+ models simultaneously to find the performance ceiling
for hand-tuned CatBoost classifiers.

Usage:
    uv run scripts/run_automl.py --symbol SPY --time-limit 120
    uv run scripts/run_automl.py --symbol SPY --presets best_quality --time-limit 600
    uv run scripts/run_automl.py --symbol SPY --compare-baseline 0.65
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.automl import AutoMLBaseline
from src.ml.feature_engineering import FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Load OHLCV data."""
    import yfinance as yf

    logger.info(f"Loading {symbol} from {start} to {end}")
    df = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    logger.info(f"Loaded {len(df)} bars")
    return df


def run(
    symbol: str = "SPY",
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    horizon: int = 5,
    time_limit: int = 300,
    presets: str = "medium_quality",
    compare_baseline: float | None = None,
    output_dir: str = "reports/automl",
) -> dict:
    """Run AutoML baseline pipeline.

    Returns:
        Dict with results for JSON output.
    """
    df = load_data(symbol, start, end)
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(df)

    future_returns = df["Close"].shift(-horizon) / df["Close"] - 1
    y = (future_returns > 0).astype(int)
    aligned = features.dropna().index.intersection(y.dropna().index)
    X = features.loc[aligned]
    y = y.loc[aligned]

    logger.info(f"Prepared {len(X)} samples, {X.shape[1]} features, positive ratio {y.mean():.1%}")

    automl = AutoMLBaseline(
        label="target",
        time_limit=time_limit,
        presets=presets,
        eval_metric="roc_auc",
        problem_type="binary",
    )
    result = automl.fit(X, y)

    logger.info(result.summary())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{symbol}"
    output = {
        "run_id": run_id,
        "symbol": symbol,
        "date_range": f"{start}-{end}",
        "n_samples": len(X),
        "n_features": X.shape[1],
        "positive_ratio": float(y.mean()),
        "presets": presets,
        "time_limit_seconds": time_limit,
        "best_model": result.best_model,
        "best_score": float(result.best_score),
        "models_trained": result.models_trained,
        "train_time_seconds": result.train_time_seconds,
        "leaderboard": result.leaderboard.to_dict("records")[:10],
        "top_features": dict(list(result.feature_importance.items())[:15]),
    }

    if compare_baseline is not None:
        comparison = automl.compare_to_baseline(compare_baseline, "Hand-Tuned CatBoost")
        output["baseline_comparison"] = comparison
        logger.info(
            f"Baseline: {compare_baseline:.4f} | AutoML: {result.best_score:.4f} | "
            f"Uplift: {comparison['uplift']:+.4f} ({comparison['uplift_percent']:+.1f}%)"
        )

    results_path = output_path / f"automl_{run_id}.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Results saved to {results_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoML Baseline Benchmark (AutoGluon)")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--time-limit", type=int, default=300, help="Training time limit in seconds"
    )
    parser.add_argument(
        "--presets",
        default="medium_quality",
        choices=["best_quality", "high_quality", "medium_quality", "optimize_for_deployment"],
        help="AutoGluon quality preset",
    )
    parser.add_argument(
        "--compare-baseline",
        type=float,
        default=None,
        help="Hand-tuned CatBoost AUC to compare against",
    )
    parser.add_argument("--output-dir", default="reports/automl")

    args = parser.parse_args()
    run(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        horizon=args.horizon,
        time_limit=args.time_limit,
        presets=args.presets,
        compare_baseline=args.compare_baseline,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
