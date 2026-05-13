"""Walk-Forward Optimization backtest: compare WFO vs single-split on OOS.

Answers: does walk-forward retraining prevent the OOS performance degradation?
Compares:
  1. Single-split: train on 2015-2024, predict 2025-2026 (never retrains)
  2. WFO: train on 5-year expanding window, retrain quarterly, trade next quarter

Uses normalized ATR features (ATR/Close) to fix scale-dependence.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  # noqa: E402

from src.ml.feature_engineering import FeatureExtractor  # noqa: E402
from src.ml.pattern_classifier import PatternClassifier  # noqa: E402
from src.ml.triple_barrier import TripleBarrierLabeler  # noqa: E402

SYMBOL = "SPY"
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("reports/wfo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# WFO parameters
INITIAL_TRAIN_YEARS = 5
RETRAIN_PERIOD_MONTHS = 3
HORIZON = 5
MODEL_KWARGS = {
    "model_type": "catboost",
    "max_depth": 3,
    "n_estimators": 100,
    "learning_rate": 0.03,
    "random_strength": 3.0,
    "l2_leaf_reg": 10.0,
    "min_data_in_leaf": 50,
}
ENTRY_THRESHOLD = 0.45

# Chronological splits
IS_END = "2025-01-01"


def load_data(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_daily.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0.0 if col == "Volume" else df.iloc[:, 0]
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def simulate_trades(
    probs: pd.Series, entry_threshold: float, labels: pd.Series, label_returns: pd.Series
) -> dict:
    """Simulate trades based on model probabilities and triple-barrier labels.

    Only enters when prob >= threshold and no position is open (or previous
    position has timed out past HORIZON). Uses actual label outcomes.
    """
    trades = []
    last_exit_idx = -1

    for i in range(len(probs) - HORIZON):
        prob = probs.iloc[i]
        label = labels.iloc[i] if i < len(labels) else np.nan

        if prob >= entry_threshold and not np.isnan(prob) and i > last_exit_idx:
            if pd.isna(label):
                continue

            entry_date = str(probs.index[i])
            ret = float(label_returns.iloc[i]) if not pd.isna(label_returns.iloc[i]) else 0.0
            exit_reason = "tp" if label == 1 else ("sl" if label == -1 else "time")

            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": str(probs.index[min(i + HORIZON, len(probs) - 1)]),
                    "entry_price": 1.0,
                    "exit_price": 1.0 + ret,
                    "return": float(ret),
                    "exit_reason": exit_reason,
                    "bars_held": HORIZON,
                }
            )
            last_exit_idx = i + HORIZON

    if not trades:
        return {
            "trades": [],
            "return": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "n_trades": 0,
            "max_dd": 0.0,
            "profit_factor": 0.0,
            "exposure": 0.0,
        }

    returns = np.array([t["return"] for t in trades])
    wins = returns > 0

    cumulative = np.cumprod(1 + returns)
    cumulative_return = cumulative[-1] - 1 if len(cumulative) > 0 else 0.0
    peak = np.maximum.accumulate(cumulative)
    dd = (cumulative - peak) / np.maximum(peak, np.finfo(float).eps)
    max_dd = float(dd.min())

    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = mean_ret / std_ret * np.sqrt(len(trades)) if std_ret > 0 else 0.0

    gross_profit = returns[wins].sum() if wins.any() else 0.0
    gross_loss = abs(returns[~wins].sum()) if (~wins).any() else 1e-10
    profit_factor = gross_profit / gross_loss

    total_bars_held = sum(t["bars_held"] for t in trades)
    exposure = total_bars_held / len(probs)

    return {
        "trades": trades,
        "return": float(cumulative_return),
        "sharpe": float(sharpe),
        "win_rate": float(wins.mean()),
        "n_trades": len(trades),
        "max_dd": float(max_dd),
        "profit_factor": float(profit_factor),
        "exposure": float(exposure),
    }


def train_single_split(df: pd.DataFrame, extractor: FeatureExtractor) -> dict:
    """Train model on 2015-2024, simulate trades on 2025-2026."""
    print("\n=== Single-Split Training ===")

    features = extractor.extract_all_features(df)
    X = features.dropna()

    # Train on IS
    is_mask = X.index < IS_END
    oos_mask_df = df.index >= IS_END

    X_train = X[is_mask]
    X_test = X[X.index >= IS_END]

    # Labels
    aligned_is = df.loc[X_train.index]
    atr_is = compute_atr(aligned_is, 14)
    labeler = TripleBarrierLabeler(atr_mult_tp=1.5, atr_mult_sl=1.0)
    y_train = labeler.fit(
        aligned_is["Close"],
        aligned_is["High"],
        aligned_is["Low"],
        time_limit=HORIZON,
        atr_series=atr_is,
    )
    y_train = (y_train == 1).astype(int)
    valid_mask = ~np.isnan(y_train.values)
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]
    labeler = TripleBarrierLabeler(atr_mult_tp=1.5, atr_mult_sl=1.0)
    y_train = labeler.fit(
        aligned_is["Close"],
        aligned_is["High"],
        aligned_is["Low"],
        time_limit=HORIZON,
        atr_series=atr_is,
    )
    y_train = (y_train == 1).astype(int)
    valid_mask = ~np.isnan(y_train.values)
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]

    print(f"Train: {len(X_train)} bars, {len(X_test)} test bars")
    print(f"Train class balance: {y_train.mean():.3f}")

    # Train model
    model = PatternClassifier(**MODEL_KWARGS)
    model.train(X_train, y_train)

    # Evaluate calibration
    from sklearn.metrics import roc_auc_score

    train_preds_df = model.predict(X_train)
    train_probs = train_preds_df["probability_profitable"].values
    try:
        train_auc = float(roc_auc_score(y_train.astype(int), train_probs))
        print(f"Train AUC: {train_auc:.4f}")
    except ValueError:
        print("Train AUC: N/A (single class)")

    # Predict OOS
    probs_oos = model.predict(X_test)["probability_profitable"]
    print(
        f"OOS: {len(probs_oos)} bars, "
        f"mean prob={probs_oos.mean():.4f}, "
        f"frac >= {ENTRY_THRESHOLD}={(probs_oos >= ENTRY_THRESHOLD).mean():.3f}"
    )

    # Generate labels for OOS
    aligned_oos = df.loc[X_test.index]
    atr_oos = compute_atr(aligned_oos, 14)
    labeler_oos = TripleBarrierLabeler(atr_mult_tp=1.5, atr_mult_sl=1.0)
    y_oos = labeler_oos.fit(
        aligned_oos["Close"],
        aligned_oos["High"],
        aligned_oos["Low"],
        time_limit=HORIZON,
        atr_series=atr_oos,
    )
    label_attr = y_oos.attrs.get("return_pct", np.zeros(len(y_oos)))
    label_returns = (
        pd.Series(label_attr, index=y_oos.index)
        if not isinstance(label_attr, pd.Series)
        else label_attr
    )

    # Simulate trades
    result = simulate_trades(probs_oos, ENTRY_THRESHOLD, y_oos, label_returns)

    # BH baseline
    aligned_oos_df = df.loc[oos_mask_df]
    bh_return = aligned_oos_df["Close"].iloc[-1] / aligned_oos_df["Close"].iloc[0] - 1
    result["bh_return"] = float(bh_return)
    result["label"] = "Single-Split"

    print(
        f"Single-split OOS: return={result['return']:.1%}, "
        f"Sharpe={result['sharpe']:.2f}, trades={result['n_trades']}, "
        f"win={result['win_rate']:.1%}, B&H={result['bh_return']:.1%}"
    )
    return result


def train_wfo(df: pd.DataFrame, extractor: FeatureExtractor) -> dict:
    """Walk-forward: retrain quarterly, trade next quarter."""
    print("\n=== Walk-Forward Optimization ===")

    features = extractor.extract_all_features(df)
    X = features.dropna()

    all_trades = []
    retrain_history = []

    # Start at initial_train_years
    train_cutoff_date = pd.Timestamp("2015-01-01") + pd.DateOffset(years=INITIAL_TRAIN_YEARS)
    test_start_date = train_cutoff_date

    dates = X.index[X.index >= test_start_date]
    if len(dates) == 0:
        return {"label": "WFO", "return": 0.0, "sharpe": 0.0, "n_trades": 0, "win_rate": 0.0}

    # Split dates into quarterly chunks
    chunk_start = dates[0]
    chunk_dates = []

    while chunk_start <= dates[-1]:
        chunk_end = chunk_start + pd.DateOffset(months=RETRAIN_PERIOD_MONTHS)
        chunk_mask = (dates >= chunk_start) & (dates < chunk_end)
        chunk_dates.append((chunk_start, chunk_end, dates[chunk_mask]))
        chunk_start = chunk_end

    print(f"WFO: {len(chunk_dates)} quarterly chunks from {dates[0].date()} to {dates[-1].date()}")

    for idx, (c_start, c_end, c_dates) in enumerate(chunk_dates):
        if len(c_dates) < 5:
            continue

        # Train on expanding window up to c_start
        train_mask = X.index < str(c_start.date())
        X_train = X[train_mask]
        if len(X_train) < 100:
            continue

        aligned_train = df.loc[X_train.index]
        atr_train = compute_atr(aligned_train, 14)
        labeler = TripleBarrierLabeler(atr_mult_tp=1.5, atr_mult_sl=1.0)
        y_train = labeler.fit(
            aligned_train["Close"],
            aligned_train["High"],
            aligned_train["Low"],
            time_limit=HORIZON,
            atr_series=atr_train,
        )
        y_train = (y_train == 1).astype(int)
        valid_mask = ~np.isnan(y_train.values)
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]

        if y_train.sum() < 50 or (1 - y_train).sum() < 50:
            continue

        # Train model
        model = PatternClassifier(**MODEL_KWARGS)
        model.train(X_train, y_train)

        # Predict this chunk
        test_mask = X.index.isin(c_dates)
        X_test = X[test_mask]
        if len(X_test) < 3:
            continue

        probs_chunk = model.predict(X_test)["probability_profitable"]
        aligned_chunk = df.loc[probs_chunk.index]

        # Generate labels for this chunk
        aligned_chunk = df.loc[probs_chunk.index]
        atr_chunk = compute_atr(aligned_chunk, 14)
        labeler_chunk = TripleBarrierLabeler(atr_mult_tp=1.5, atr_mult_sl=1.0)
        y_chunk = labeler_chunk.fit(
            aligned_chunk["Close"],
            aligned_chunk["High"],
            aligned_chunk["Low"],
            time_limit=HORIZON,
            atr_series=atr_chunk,
        )
        returns_attr = y_chunk.attrs.get("return_pct", np.zeros(len(y_chunk)))
        returns_chunk = (
            pd.Series(returns_attr, index=y_chunk.index)
            if not isinstance(returns_attr, pd.Series)
            else returns_attr
        )

        chunk_result = simulate_trades(probs_chunk, ENTRY_THRESHOLD, y_chunk, returns_chunk)

        retrain_history.append(
            {
                "chunk": idx + 1,
                "start": str(c_start.date()),
                "end": str(c_end.date()),
                "train_bars": len(X_train),
                "test_bars": len(X_test),
                "n_trades": chunk_result["n_trades"],
                "win_rate": chunk_result["win_rate"],
            }
        )
        all_trades.extend(chunk_result["trades"])

        if idx % 4 == 0:
            print(
                f"  Chunk {idx + 1}: {str(c_start.date())} - {str(c_end.date())}, "
                f"train={len(X_train)}, test={len(X_test)}, "
                f"trades={chunk_result['n_trades']}"
            )

    if not all_trades:
        return {
            "label": "WFO",
            "return": 0.0,
            "sharpe": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
            "max_dd": 0.0,
            "profit_factor": 0.0,
            "exposure": 0.0,
            "retrain_history": retrain_history,
            "bh_return": 0.0,
        }

    returns = np.array([t["return"] for t in all_trades])
    wins = returns > 0

    cumulative = np.cumprod(1 + returns)
    cumulative_return = cumulative[-1] - 1 if len(cumulative) > 0 else 0.0
    peak = np.maximum.accumulate(cumulative)
    dd = (cumulative - peak) / np.maximum(peak, np.finfo(float).eps)
    max_dd = float(dd.min())

    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = mean_ret / std_ret * np.sqrt(len(all_trades)) if std_ret > 0 else 0.0

    gross_profit = returns[wins].sum() if wins.any() else 0.0
    gross_loss = abs(returns[~wins].sum()) if (~wins).any() else 1e-10
    profit_factor = gross_profit / gross_loss

    total_bars = sum(t["bars_held"] for t in all_trades)
    n_oos_bars = len(X[X.index >= str(test_start_date.date())])
    exposure = total_bars / max(n_oos_bars, 1)

    # B&H for same OOS period
    oos_mask_df = df.index >= str(test_start_date.date())
    bh_return = float(
        df.loc[oos_mask_df, "Close"].iloc[-1] / df.loc[oos_mask_df, "Close"].iloc[0] - 1
    )

    result = {
        "label": "WFO",
        "return": float(cumulative_return),
        "sharpe": float(sharpe),
        "win_rate": float(wins.mean()),
        "n_trades": len(all_trades),
        "max_dd": float(max_dd),
        "profit_factor": float(profit_factor),
        "exposure": float(exposure),
        "retrain_history": retrain_history,
        "bh_return": float(bh_return),
    }

    # Split IS vs OOS for reporting
    is_trades = [t for t in all_trades if str(t["entry_date"]) < IS_END]
    oos_trades = [t for t in all_trades if str(t["entry_date"]) >= IS_END]

    if is_trades:
        is_rets = np.array([t["return"] for t in is_trades])
        is_wins = is_rets > 0
        print(
            f"\nWFO IS (before 2025): {len(is_trades)} trades, "
            f"return={np.prod(1 + is_rets) - 1:.1%}, win={is_wins.mean():.1%}"
        )
    if oos_trades:
        oos_rets = np.array([t["return"] for t in oos_trades])
        oos_wins = oos_rets > 0
        print(
            f"WFO OOS (2025+): {len(oos_trades)} trades, "
            f"return={np.prod(1 + oos_rets) - 1:.1%}, win={oos_wins.mean():.1%}"
        )

    print(
        f"\nWFO total: return={result['return']:.1%}, "
        f"Sharpe={result['sharpe']:.2f}, trades={result['n_trades']}, "
        f"win={result['win_rate']:.1%}, B&H={result['bh_return']:.1%}"
    )
    return result


def main() -> None:
    print(f"WFO Backtest — {SYMBOL}")
    print("Normalized ATR features (ATR/Close)")
    print(f"Entry threshold: {ENTRY_THRESHOLD}")
    print(f"Initial train: {INITIAL_TRAIN_YEARS}y, retrain: {RETRAIN_PERIOD_MONTHS}mo")

    df = load_data(SYMBOL)
    print(f"Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

    extractor = FeatureExtractor()

    # Run both methods
    ss_result = train_single_split(df, extractor)
    wfo_result = train_wfo(df, extractor)

    # Compare
    print("\n" + "=" * 60)
    print("=== WFO vs Single-Split Comparison ===")
    print("=" * 60)
    print(f"{'Metric':<20} | {'Single-Split':>12} | {'WFO':>12}")
    print("-" * 50)
    for metric, fmt in [
        ("return", ".1%"),
        ("sharpe", ".2f"),
        ("n_trades", "d"),
        ("win_rate", ".1%"),
        ("max_dd", ".1%"),
        ("profit_factor", ".2f"),
        ("exposure", ".1%"),
        ("bh_return", ".1%"),
    ]:
        ss_val = ss_result.get(metric, 0)
        wfo_val = wfo_result.get(metric, 0)
        if fmt == ".1%":
            print(f"{metric:<20} | {ss_val:>12.1%} | {wfo_val:>12.1%}")
        elif fmt == ".2f":
            print(f"{metric:<20} | {ss_val:>12.2f} | {wfo_val:>12.2f}")
        else:
            print(f"{metric:<20} | {ss_val:>12d} | {wfo_val:>12d}")

    # Save results
    import json

    out_path = OUTPUT_DIR / "wfo_comparison.json"
    with open(out_path, "w") as f:
        json.dump({"single_split": ss_result, "wfo": wfo_result}, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
