"""
Q3: GARCH/EGARCH Volatility Forecasting CLI.

Smoke test and compare GARCH-family volatility forecasters on SPY.

Usage:
    # Run all three GARCH models on SPY
    uv run scripts/garch_forecast.py SPY

    # Compare specific model
    uv run scripts/garch_forecast.py SPY --model egarch

    # Specify date range and horizon
    uv run scripts/garch_forecast.py SPY --start 2020-01-01 --end 2024-12-31 --horizon 10

    # Compare GARCH vs CatBoost VolatilityForecaster
    uv run scripts/garch_forecast.py SPY --compare-catboost
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.garch_forecaster import GARCHForecaster, GARCHForecastResult
from src.ml.volatility_forecaster import VolatilityForecaster


def load_data(symbol: str, start: str = "2016-01-01", end: str | None = None) -> pd.DataFrame:
    path = Path(f"data/raw/{symbol}_daily.csv")
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol} at {path}")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df.dropna()
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]
    return df


def run_full_comparison(
    symbol: str,
    start: str,
    end: str | None,
    horizon: int,
    window: int,
    retrain_every: int,
    compare_catboost: bool,
) -> dict:
    df = load_data(symbol, start=start, end=end)
    returns = df["Close"].pct_change().dropna()

    print(f"\n=== GARCH Volatility Forecast: {symbol} ===\n")
    print(f"Period: {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"Observations: {len(returns)}")
    print(f"Horizon: {horizon} bars, Window: {window}, Retrain: {retrain_every} bars\n")

    # Run all three GARCH models
    forecaster = GARCHForecaster(
        model="egarch",
        window=window,
        retrain_every=retrain_every,
    )
    df_compare = forecaster.compare_models(returns=returns, horizon=horizon)
    print("GARCH Model Comparison:")
    print(df_compare.to_string(index=False))
    print()

    results = {"garch_models": df_compare.to_dict(orient="records")}

    # Compare with CatBoost if requested
    if compare_catboost:
        print("--- CatBoost Volatility Forecaster ---")
        cb = VolatilityForecaster(horizon=horizon)
        cb_result = cb.cross_validate(df, n_splits=3, returns=returns)
        print(
            f"  CatBoost: RMSE={cb_result.rmse:.4f}, R²={cb_result.r2:.4f}, DirAcc={cb_result.direction_accuracy:.3f}"
        )
        results["catboost"] = cb_result.to_dict()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Q3: GARCH/EGARCH volatility forecasting")
    parser.add_argument("symbol", help="Ticker symbol (e.g., SPY)")
    parser.add_argument("--start", default="2016-01-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon (bars)")
    parser.add_argument("--window", type=int, default=252, help="Training window (bars)")
    parser.add_argument("--retrain-every", type=int, default=63, help="Retrain model every N bars")
    parser.add_argument(
        "--compare-catboost", action="store_true", help="Compare against CatBoost forecaster"
    )
    parser.add_argument("--json-output", help="Path to save JSON results")
    args = parser.parse_args()

    results = run_full_comparison(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        horizon=args.horizon,
        window=args.window,
        retrain_every=args.retrain_every,
        compare_catboost=args.compare_catboost,
    )

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"Results saved to {args.json_output}")


if __name__ == "__main__":
    main()
