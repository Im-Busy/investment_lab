"""Walk-forward paper trading with expanding-window retraining.

Trains SPY-only CatBoost model on expanding historical windows,
tests each subsequent year, simulating real-world deployment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.feature_engineering import FeatureExtractor
from src.ml.pattern_classifier import PatternClassifier

DATA_PATH = Path("data/raw/SPY_daily.csv")
OUTPUT_DIR = Path("reports/walk_forward")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df = df.dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0.0 if col == "Volume" else df.iloc[:, 0]
    return df


def train_model(df: pd.DataFrame, extractor: FeatureExtractor) -> PatternClassifier | None:
    """Train a CatBoost model on given data. Returns None if insufficient samples."""
    features = extractor.extract_all_features(df)
    fwd_returns = df["Close"].shift(-5) / df["Close"] - 1
    y = (fwd_returns > 0).astype(int)

    features_clean = features.dropna()
    y_clean = y.reindex(features_clean.index).dropna()
    common_idx = features_clean.index.intersection(y_clean.index)
    X = features_clean.loc[common_idx]
    y_final = y_clean.loc[common_idx]

    if len(X) < 100:
        return None

    model = PatternClassifier(
        model_type="catboost",
        n_estimators=100,
        max_depth=4,
        learning_rate=0.03,
        random_state=42,
    )
    try:
        model.train(X, y_final)
    except ValueError:
        return None

    return model


def simulate_year(
    test_df: pd.DataFrame,
    model: PatternClassifier | None,
    extractor: FeatureExtractor,
    entry_threshold: float = 0.45,
) -> list[dict]:
    """Simulate paper trading for one year: buy if prob > threshold, hold 10 bars."""
    if model is None:
        return []

    features = extractor.extract_all_features(test_df)
    common_cols = [c for c in model.feature_names_ if c in features.columns]
    if len(common_cols) < len(model.feature_names_):
        X = features[common_cols].copy()
        for m in set(model.feature_names_) - set(common_cols):
            X[m] = 0.0
    else:
        X = features[model.feature_names_].copy()

    X = X.dropna()
    if len(X) == 0:
        return []

    probs = model.predict(X)["probability_profitable"]

    trades = []
    in_position = False
    entry_bar = 0

    for i, idx in enumerate(X.index):
        prob = probs.iloc[i]
        if not in_position and prob >= entry_threshold:
            in_position = True
            entry_bar = i
            entry_price = test_df.loc[idx, "Close"]
            trades.append(
                {
                    "entry_date": idx,
                    "entry_price": entry_price,
                    "prob": prob,
                }
            )
        elif in_position and i - entry_bar >= 10:
            in_position = False
            exit_price = test_df.loc[idx, "Close"]
            t = trades[-1]
            t["exit_date"] = idx
            t["exit_price"] = exit_price
            t["return"] = (exit_price - t["entry_price"]) / t["entry_price"]
            t["win"] = t["return"] > 0
            t["bars_held"] = i - entry_bar

    if in_position and len(trades) > 0:
        # Close last trade at end of year
        last_idx = X.index[-1]
        exit_price = test_df.loc[last_idx, "Close"]
        t = trades[-1]
        t["exit_date"] = last_idx
        t["exit_price"] = exit_price
        t["return"] = (exit_price - t["entry_price"]) / t["entry_price"]
        t["win"] = t["return"] > 0
        t["bars_held"] = len(X) - entry_bar

    return [t for t in trades if "return" in t]


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward paper trading")
    parser.add_argument(
        "--years", type=int, default=3, help="Years of training data before first test year"
    )
    parser.add_argument("--entry-threshold", type=float, default=0.45)
    parser.add_argument("--start", type=str, default="2015-01-01", help="Train start date")
    args = parser.parse_args()

    df = load_data()
    print(f"Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

    extractor = FeatureExtractor()

    # Determine test years
    train_start = df.index[0]
    years = sorted(set(df.index.year))
    min_train_years = args.years

    all_trades = []
    year_results = []

    for test_year in years[min_train_years:]:
        train_mask = df.index.year < test_year
        test_mask = df.index.year == test_year

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(test_df) < 50:
            continue

        print(f"\n{'=' * 60}")
        print(
            f"Test year {test_year}: train={train_df.index[0].date()}..{train_df.index[-1].date()} ({len(train_df)} bars), test={len(test_df)} bars"
        )
        print(f"{'=' * 60}")

        model = train_model(train_df, extractor)
        if model is None:
            print("  Skipped: insufficient training data")
            continue

        trades = simulate_year(test_df, model, extractor, entry_threshold=args.entry_threshold)
        all_trades.extend(trades)

        yr_wins = sum(1 for t in trades if t["win"])
        yr_returns = [t["return"] for t in trades]
        yr_total_return = np.prod([1 + r for r in yr_returns]) - 1 if yr_returns else 0

        print(
            f"  Trades: {len(trades)}, Wins: {yr_wins}, Win%: {yr_wins / len(trades) * 100:.1f}%"
            if trades
            else "  No trades"
        )
        if trades:
            print(f"  Return: {yr_total_return * 100:.2f}%, Avg: {np.mean(yr_returns) * 100:.2f}%")

        year_results.append(
            {
                "year": test_year,
                "trades": len(trades),
                "wins": yr_wins,
                "win_rate": yr_wins / len(trades) * 100 if trades else 0,
                "total_return": yr_total_return * 100,
                "avg_return": np.mean(yr_returns) * 100 if yr_returns else 0,
                "train_bars": len(train_df),
                "test_bars": len(test_df),
            }
        )

    # ── Aggregate results ──
    print(f"\n{'=' * 80}")
    print("WALK-FORWARD PAPER TRADING — AGGREGATE RESULTS")
    print(f"{'=' * 80}")

    if year_results:
        yr_df = pd.DataFrame(year_results)
        print(yr_df.to_string(index=False))

    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        total_trades = len(trades_df)
        wins = trades_df["win"].sum()
        win_rate = wins / total_trades * 100
        avg_return = trades_df["return"].mean() * 100
        compound_return = np.prod([1 + r for r in trades_df["return"]]) - 1

        gross_profit = trades_df[trades_df["win"]]["return"].sum()
        gross_loss = abs(trades_df[~trades_df["win"]]["return"].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Annualized Sharpe (approximate)
        returns_series = trades_df["return"]
        sharpe_annual = (
            (returns_series.mean() / returns_series.std()) * np.sqrt(252 / 10)
            if returns_series.std() > 0
            else 0
        )

        # Max drawdown
        equity = (1 + trades_df["return"]).cumprod()
        peak = equity.cummax()
        dd = (equity - peak) / peak
        max_dd = dd.min() * 100

        print(f"\n{'=' * 60}")
        print(f"AGGREGATE METRICS ({len(year_results)} years)")
        print(f"{'=' * 60}")
        print(f"  Total Trades:     {total_trades}")
        print(f"  Win Rate:         {win_rate:.1f}%")
        print(f"  Avg Return/Trade: {avg_return:.2f}%")
        print(f"  Compound Return:  {compound_return * 100:.2f}%")
        print(f"  Profit Factor:    {pf:.2f}")
        print(f"  Approx Sharpe:    {sharpe_annual:.2f}")
        print(f"  Max Drawdown:     {max_dd:.2f}%")

        # Save
        trades_df.to_csv(OUTPUT_DIR / "walk_forward_trades.csv", index=False)
        if year_results:
            pd.DataFrame(year_results).to_csv(OUTPUT_DIR / "walk_forward_yearly.csv", index=False)
        print(f"\nSaved to {OUTPUT_DIR}")

        # Compare to B&H per year
        print(f"\n{'=' * 60}")
        print("SPY Buy & Hold by Year (for context)")
        print(f"{'=' * 60}")
        for yr in sorted(set(df.index.year)):
            yr_mask = df.index.year == yr
            if yr_mask.sum() > 0:
                yr_return = (
                    df.loc[yr_mask, "Close"].iloc[-1] / df.loc[yr_mask, "Close"].iloc[0] - 1
                ) * 100
                print(f"  {yr}: {yr_return:+.1f}%")
    else:
        print("No trades generated.")


if __name__ == "__main__":
    main()
