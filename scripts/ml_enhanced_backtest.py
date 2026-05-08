"""
ML-Enhanced Backtesting Script

Trains ML regime classifier and signal scorer, then runs backtest
with ML-enhanced confluence scoring.

Usage:
    uv run scripts/ml_enhanced_backtest.py --symbol SPY --start 2015-01-01 --end 2024-12-31
    uv run scripts/ml_enhanced_backtest.py --symbol SPY --ml-only  # Train ML models without backtest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yfinance as yf

project_root = Path(__file__).parent.parent

from src.backtest.engine import BacktestEngine
from src.indicators.regime_detector import RegimeDetector
from src.ml.features import FeatureEngineer
from src.ml.pipeline import MLPipeline
from src.signals.signal_generator import SignalGenerator


def load_price_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download price data from Yahoo Finance."""
    df = yf.download(symbol, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(
        columns={
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Open": "Open",
            "Volume": "Volume",
        }
    )
    df.index.name = "date"
    return df.sort_index()


def train_ml_models(df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
    """Train ML regime classifier and signal scorer."""
    detector = RegimeDetector()
    regime_result = detector.get_regime_series(df)
    regime_labels = regime_result["regime"].apply(lambda r: r.value)

    engineer = FeatureEngineer()
    features_df = engineer.generate_features(df)
    numeric_features = features_df.select_dtypes(include=[np.number]).ffill().bfill().dropna()

    y = regime_labels.reindex(numeric_features.index).dropna()
    X = numeric_features.reindex(y.index)

    pipeline = MLPipeline(
        regime_model_type="random_forest",
        signal_model_type="gradient_boosting",
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )

    results = pipeline.fit(df, regime_labels=regime_labels)

    if verbose:
        print("\n=== ML Training Results ===")
        if "regime" in results:
            rr = results["regime"]
            print(f"Regime train accuracy: {rr.get('train_accuracy', 0):.4f}")
            print(f"Regime test accuracy: {rr.get('test_accuracy', 0):.4f}")
            if "feature_importance" in rr:
                top5 = list(rr["feature_importance"].items())[:5]
                print(f"Top 5 features: {top5}")

    return {
        "pipeline": pipeline,
        "regime_labels": regime_labels,
        "features": numeric_features,
        "training_results": results,
    }


def run_ml_backtest(
    df: pd.DataFrame, ml_models: Dict[str, Any], verbose: bool = True
) -> Dict[str, Any]:
    """Run backtest with ML-enhanced confluence scoring."""
    pipeline = ml_models["pipeline"]

    regime_pred = pipeline.predict_regime(df)
    regime_proba = pipeline.get_regime_probabilities(df)

    engine = BacktestEngine(initial_capital=100_000)

    signal_gen = SignalGenerator(patterns=[])

    scored_signals = []
    for i in range(len(df)):
        if pd.isna(regime_proba.iloc[i].values).any():
            continue

        signals = signal_gen.generate_signals(df.iloc[: i + 1])
        if not signals:
            continue

        ml_confidence = regime_proba.iloc[i].max()
        for s in signals:
            s.confidence = s.confidence * 0.7 + ml_confidence * 0.3
        scored_signals.extend(signals)

    if verbose:
        print("\n=== ML-Enhanced Backtest ===")
        print(f"Total signals: {len(scored_signals)}")
        print(f"Regime predictions: {regime_pred.value_counts().to_dict()}")

    return {
        "signals": scored_signals,
        "regime_predictions": regime_pred,
        "regime_probabilities": regime_proba,
    }


def main():
    parser = argparse.ArgumentParser(description="ML-Enhanced Backtesting")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol")
    parser.add_argument("--start", default="2015-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument("--ml-only", action="store_true", help="Only train ML models")
    args = parser.parse_args()

    print(f"Loading {args.symbol} data from {args.start} to {args.end}...")
    df = load_price_data(args.symbol, args.start, args.end)
    print(f"Loaded {len(df)} bars")

    print("\nTraining ML models...")
    ml_models = train_ml_models(df)

    if not args.ml_only:
        print("\nRunning ML-enhanced backtest...")
        results = run_ml_backtest(df, ml_models)
        print(f"\nBacktest complete. {len(results['signals'])} signals generated.")
    else:
        print("\nML training complete (--ml-only flag set)")

    output_dir = project_root / "reports" / "ml_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "ml_training_results.json"
    training_report = {
        "training_results": {
            k: {k2: v2 for k2, v2 in v.items() if isinstance(v2, (int, float, str))}
            for k, v in ml_models["training_results"].items()
        }
    }
    report_path.write_text(json.dumps(training_report, indent=2))
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
