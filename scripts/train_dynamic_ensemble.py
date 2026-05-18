"""Train a DynamicEnsemble on market data for regime-adaptive prediction.

Usage:
    uv run scripts/train_dynamic_ensemble.py --symbol SPY --fast
    uv run scripts/train_dynamic_ensemble.py --symbol SPY --start 2016-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.dynamic_ensemble import DynamicEnsemble, DynamicEnsembleConfig
from src.ml.feature_engineering import FeatureExtractor
from src.ml.triple_barrier import TripleBarrierLabeler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(symbol: str, start: str = "2015-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    path = Path(f"data/raw/{symbol}_daily.csv")
    if path.exists():
        return pd.read_csv(path, parse_dates=True, index_col=0)
    import yfinance as yf

    return yf.download(symbol, start=start, end=end, progress=False)  # type: ignore[return-value]


def main():
    parser = argparse.ArgumentParser(description="Train DynamicEnsemble for regime adaptation")
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--horizon", type=int, default=5, help="Forward return horizon")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument(
        "--fast", action="store_true", help="Reduce training epochs for quick iteration"
    )
    args = parser.parse_args()

    logger.info(f"Loading {args.symbol} {args.start}→{args.end}")
    df = load_data(args.symbol, args.start, args.end)
    if df.empty:
        logger.error(f"No data for {args.symbol}")
        return

    # Feature extraction
    logger.info("Extracting features")
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(df, include_forward_returns=False)

    # Labels via triple-barrier
    logger.info("Generating triple-barrier labels")
    labeler = TripleBarrierLabeler(atr_mult_tp=3.0, atr_mult_sl=1.5)
    atr = df["High"] - df["Low"]
    atr = atr.rolling(14).mean()
    labels = labeler.fit(
        close=df["Close"],
        high=df["High"],
        low=df["Low"],
        time_limit=args.horizon,
        atr_series=atr,
    )

    common = features.index.intersection(labels.dropna().index)
    X = features.loc[common].ffill().bfill().fillna(0)
    y = labels.loc[common]
    y = (y == 1).astype(int)  # binary: profitable or not

    logger.info(f"Training data: {len(X)} samples, {X.shape[1]} features")
    logger.info(f"Label balance: {y.sum() / len(y) * 100:.1f}% positive")

    # Train DynamicEnsemble
    n_est = 50 if args.fast else 100
    config = DynamicEnsembleConfig(n_estimators=n_est)
    ensemble = DynamicEnsemble(config)
    ensemble.fit(X, y)

    # Save
    output_path = Path(args.output_dir) / f"dynamic_ensemble_{args.symbol}"
    ensemble.save(output_path)
    logger.info(f"Ensemble saved to {output_path}")
    logger.info(f"Models: {len(ensemble.models)}, features: {len(ensemble.feature_names)}")


if __name__ == "__main__":
    main()
