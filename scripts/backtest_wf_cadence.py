"""Walk-Forward Retraining Cadence Experiment — P2-4.

Answers: what retraining frequency maximizes OOS performance?
Tests multiple cadences (never, 6mo, 12mo, 24mo) on expanding-window
walk-forward, comparing cumulative returns, Sharpe, trades, and win rate.

Uses the same infrastructure as backtest_wfo.py.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.feature_engineering import FeatureExtractor
from src.ml.pattern_classifier import PatternClassifier
from src.ml.triple_barrier import TripleBarrierLabeler

SYMBOL = "SPY"
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("reports/wf_cadence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 5
INITIAL_TRAIN_YEARS = 5
ENTRY_THRESHOLD = 0.45
MODEL_KWARGS = {
    "model_type": "catboost",
    "max_depth": 3,
    "n_estimators": 100,
    "learning_rate": 0.03,
    "random_strength": 3.0,
    "l2_leaf_reg": 10.0,
    "min_data_in_leaf": 50,
}

CADENCES = {
    "never": None,
    "6mo": 6,
    "12mo": 12,
    "24mo": 24,
    "4mo": 4,
    "3mo": 3,
}


@dataclass
class CadenceResult:
    name: str
    retrain_months: int | None
    return_pct: float
    sharpe: float
    n_trades: int
    win_rate: float
    max_dd: float
    profit_factor: float
    exposure: float
    bh_return: float
    annualized_return: float
    retrain_count: int
    trades_by_chunk: list[int]
    per_chunk_details: list[dict]


def load_data(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_daily.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).dropna()
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
    probs: pd.Series,
    entry_threshold: float,
    labels: pd.Series,
    label_returns: pd.Series,
) -> tuple[list[dict], dict]:
    trades = []
    last_exit_idx = -1

    for i in range(len(probs) - HORIZON):
        prob = probs.iloc[i]
        label = labels.iloc[i] if i < len(labels) else np.nan
        if prob >= entry_threshold and not np.isnan(prob) and i > last_exit_idx:
            if pd.isna(label):
                continue
            ret = float(label_returns.iloc[i]) if not pd.isna(label_returns.iloc[i]) else 0.0
            trades.append(
                {
                    "entry_date": str(probs.index[i]),
                    "exit_date": str(probs.index[min(i + HORIZON, len(probs) - 1)]),
                    "return": float(ret),
                    "exit_reason": "tp" if label == 1 else ("sl" if label == -1 else "time"),
                    "bars_held": HORIZON,
                }
            )
            last_exit_idx = i + HORIZON

    if not trades:
        return [], {
            "return": 0.0,
            "sharpe": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
            "max_dd": 0.0,
            "profit_factor": 0.0,
            "exposure": 0.0,
        }

    returns = np.array([t["return"] for t in trades])
    wins = returns > 0
    cumulative = np.cumprod(1 + returns)
    cumulative_return = cumulative[-1] - 1
    peak = np.maximum.accumulate(cumulative)
    dd = (cumulative - peak) / np.maximum(peak, np.finfo(float).eps)
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = mean_ret / std_ret * np.sqrt(len(trades)) if std_ret > 0 else 0.0
    gross_profit = returns[wins].sum() if wins.any() else 0.0
    gross_loss = abs(returns[~wins].sum()) if (~wins).any() else 1e-10

    return trades, {
        "return": float(cumulative_return),
        "sharpe": float(sharpe),
        "n_trades": len(trades),
        "win_rate": float(wins.mean()),
        "max_dd": float(dd.min()),
        "profit_factor": float(gross_profit / gross_loss),
        "exposure": float(sum(t["bars_held"] for t in trades) / len(probs)),
    }


def train_and_label(
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[Any, pd.Series, pd.Series]:
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

    model = PatternClassifier(**MODEL_KWARGS)
    model.train(X_train, y_train)

    aligned_test = df.loc[X_test.index]
    atr_test = compute_atr(aligned_test, 14)
    labeler_test = TripleBarrierLabeler(atr_mult_tp=1.5, atr_mult_sl=1.0)
    y_test = labeler_test.fit(
        aligned_test["Close"],
        aligned_test["High"],
        aligned_test["Low"],
        time_limit=HORIZON,
        atr_series=atr_test,
    )
    returns_attr = y_test.attrs.get("return_pct", np.zeros(len(y_test)))
    label_returns = (
        pd.Series(returns_attr, index=y_test.index)
        if not isinstance(returns_attr, pd.Series)
        else returns_attr
    )
    return model, y_test, label_returns


def run_cadence(
    df: pd.DataFrame,
    features: pd.DataFrame,
    cadence_name: str,
    retrain_months: int | None,
) -> CadenceResult:
    X = features.dropna()
    all_trades: list[dict] = []
    retrain_count = 0
    trades_by_chunk: list[int] = []
    per_chunk_details: list[dict] = []
    current_model = None

    train_cutoff = pd.Timestamp("2015-01-01") + pd.DateOffset(years=INITIAL_TRAIN_YEARS)
    dates = X.index[X.index >= train_cutoff]
    if len(dates) == 0:
        return CadenceResult(cadence_name, retrain_months, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], [])

    # Single-split: train once, predict all
    if retrain_months is None:
        train_mask = X.index < str(dates[0].date())
        X_train_all = X[train_mask]
        X_test_all = X[X.index >= str(dates[0].date())]
        model, y_all, label_returns = train_and_label(df, X_train_all, X_test_all)
        probs_all = model.predict(X_test_all)["probability_profitable"]
        trades, metrics = simulate_trades(probs_all, ENTRY_THRESHOLD, y_all, label_returns)

        oos_mask = df.index >= str(dates[0].date())
        bh_return = float(
            df.loc[oos_mask, "Close"].iloc[-1] / df.loc[oos_mask, "Close"].iloc[0] - 1
        )
        n_oos_years = (df.index[-1] - pd.Timestamp(str(dates[0].date()))).days / 365.25
        annualized = (1 + metrics["return"]) ** (1 / max(n_oos_years, 0.1)) - 1

        print(
            f"  {cadence_name} (single split): return={metrics['return']:.1%}, "
            f"Sharpe={metrics['sharpe']:.2f}, trades={metrics['n_trades']}, "
            f"win={metrics['win_rate']:.1%}"
        )

        return CadenceResult(
            name=cadence_name,
            retrain_months=None,
            return_pct=metrics["return"],
            sharpe=metrics["sharpe"],
            n_trades=metrics["n_trades"],
            win_rate=metrics["win_rate"],
            max_dd=metrics["max_dd"],
            profit_factor=metrics["profit_factor"],
            exposure=metrics["exposure"],
            bh_return=bh_return,
            annualized_return=annualized,
            retrain_count=0,
            trades_by_chunk=[metrics["n_trades"]],
            per_chunk_details=[],
        )

    # Walk-forward: retrain at each interval
    chunk_start = dates[0]
    chunks = []
    while chunk_start <= dates[-1]:
        chunk_end = chunk_start + pd.DateOffset(months=retrain_months)
        chunk_mask = (dates >= chunk_start) & (dates < chunk_end)
        chunk_dates = dates[chunk_mask]
        if len(chunk_dates) >= 3:
            chunks.append((chunk_start, chunk_end, chunk_dates))
        chunk_start = chunk_end

    next_retrain_at = 0
    for idx, (c_start, c_end, c_dates) in enumerate(chunks):
        if len(c_dates) < 5:
            continue

        if idx >= next_retrain_at:
            train_mask = X.index < str(c_start.date())
            X_train = X[train_mask]
            if len(X_train) < 100:
                continue
            current_model, _, _ = train_and_label(df, X_train, X.iloc[:1])
            retrain_count += 1
            next_retrain_at = idx

        if current_model is None:
            continue

        test_mask = X.index.isin(c_dates)
        X_test = X[test_mask]
        if len(X_test) < 3:
            continue

        probs_chunk = current_model.predict(X_test)["probability_profitable"]
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
        chunk_trades, _ = simulate_trades(probs_chunk, ENTRY_THRESHOLD, y_chunk, returns_chunk)
        all_trades.extend(chunk_trades)
        trades_by_chunk.append(len(chunk_trades))

    if not all_trades:
        return CadenceResult(cadence_name, retrain_months, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], [])

    returns = np.array([t["return"] for t in all_trades])
    wins = returns > 0
    cumulative = np.cumprod(1 + returns)
    cumulative_return = cumulative[-1] - 1
    peak = np.maximum.accumulate(cumulative)
    dd = (cumulative - peak) / np.maximum(peak, np.finfo(float).eps)
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = mean_ret / std_ret * np.sqrt(len(all_trades)) if std_ret > 0 else 0.0
    gross_profit = returns[wins].sum() if wins.any() else 0.0
    gross_loss = abs(returns[~wins].sum()) if (~wins).any() else 1e-10

    oos_mask = df.index >= str(dates[0].date())
    bh_return = float(df.loc[oos_mask, "Close"].iloc[-1] / df.loc[oos_mask, "Close"].iloc[0] - 1)
    n_oos_years = (df.index[-1] - dates[0]).days / 365.25
    annualized = (1 + cumulative_return) ** (1 / max(n_oos_years, 0.1)) - 1
    total_bars = sum(t.get("bars_held", HORIZON) for t in all_trades)
    n_oos_bars = len(X[X.index >= str(dates[0].date())])
    exposure = total_bars / max(n_oos_bars, 1)

    print(
        f"  {cadence_name} ({retrain_count} retrains): return={cumulative_return:.1%}, "
        f"Sharpe={sharpe:.2f}, trades={len(all_trades)}, "
        f"win={wins.mean():.1%}, retrains={retrain_count}"
    )

    return CadenceResult(
        name=cadence_name,
        retrain_months=retrain_months,
        return_pct=float(cumulative_return),
        sharpe=float(sharpe),
        n_trades=len(all_trades),
        win_rate=float(wins.mean()),
        max_dd=float(dd.min()),
        profit_factor=float(gross_profit / gross_loss),
        exposure=float(exposure),
        bh_return=float(bh_return),
        annualized_return=float(annualized),
        retrain_count=retrain_count,
        trades_by_chunk=trades_by_chunk,
        per_chunk_details=[],
    )


def main() -> None:
    print(f"Walk-Forward Retraining Cadence — {SYMBOL}")
    print(f"Entry threshold: {ENTRY_THRESHOLD}, initial train: {INITIAL_TRAIN_YEARS}y")
    print()

    df = load_data(SYMBOL)
    print(f"Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

    extractor = FeatureExtractor()
    features = extractor.extract_all_features(df)

    # Determine OOS period
    train_cutoff = pd.Timestamp("2015-01-01") + pd.DateOffset(years=INITIAL_TRAIN_YEARS)
    oos_dates = features.index[features.index >= train_cutoff]
    n_oos_years = (df.index[-1] - oos_dates[0]).days / 365.25 if len(oos_dates) > 0 else 0
    print(
        f"OOS period: {oos_dates[0].date() if len(oos_dates) > 0 else 'N/A'} "
        f"to {df.index[-1].date()} ({n_oos_years:.1f} years)"
    )
    print()

    results: dict[str, CadenceResult] = {}
    for name, months in CADENCES.items():
        print(f"Running {name} cadence...")
        results[name] = run_cadence(df, features, name, months)
        print()

    # --- Comparison table ---
    print("=" * 90)
    print("=== Walk-Forward Cadence Comparison ===")
    print("=" * 90)
    header = (
        f"{'Cadence':<10} {'Retrains':>8} {'Return':>9} {'Sharpe':>8} "
        f"{'Ann.Ret':>8} {'Trades':>7} {'Win%':>7} {'MaxDD':>7} "
        f"{'PF':>6} {'Exp':>6} {'B&H':>9}"
    )
    print(header)
    print("-" * 90)

    best_sharpe = max(r.sharpe for r in results.values())
    best_return = max(r.return_pct for r in results.values())

    for r in results.values():
        cadence_label = r.name
        retrains = str(r.retrain_count)
        row = (
            f"{cadence_label:<10} {retrains:>8} {r.return_pct:>8.1%} {r.sharpe:>7.2f} "
            f"{r.annualized_return:>7.1%} {r.n_trades:>6d} {r.win_rate:>6.1%} "
            f"{r.max_dd:>6.1%} {r.profit_factor:>5.2f} {r.exposure:>5.1%} "
            f"{r.bh_return:>8.1%}"
        )
        if r.sharpe == best_sharpe:
            row += "  ** BEST Sharpe"
        if r.return_pct == best_return:
            row += "  ** BEST Return"
        print(row)

    print()

    # --- Recommendation ---
    best = max(results.values(), key=lambda r: r.sharpe if r.sharpe > 0 else float("-inf"))
    trades_per_year = best.n_trades / max(n_oos_years, 0.1)

    print("=== Recommendation ===")
    print(f"Optimal cadence: {best.name} ({best.retrain_months or 'never'} months)")
    print(
        f"  Sharpe: {best.sharpe:.2f}, Return: {best.return_pct:.1%}, "
        f"{best.n_trades} trades ({trades_per_year:.1f}/year)"
    )
    print()

    # Cadence efficiency: trades per retrain
    print("=== Retraining Efficiency ===")
    print(f"{'Cadence':<10} {'Retrains':>8} {'Trades':>7} {'Trades/Retrain':>14}")
    print("-" * 45)
    for r in results.values():
        if r.retrain_months is None:
            tpr = "-"
        elif r.retrain_count > 0:
            tpr = f"{r.n_trades / r.retrain_count:.1f}"
        else:
            tpr = "∞"
        print(f"{r.name:<10} {r.retrain_count:>8} {r.n_trades:>7} {tpr:>14}")

    # Save results
    out_path = OUTPUT_DIR / f"cadence_comparison_{SYMBOL}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                name: {
                    "cadence": name,
                    "retrain_months": r.retrain_months,
                    "return": r.return_pct,
                    "sharpe": r.sharpe,
                    "n_trades": r.n_trades,
                    "win_rate": r.win_rate,
                    "max_dd": r.max_dd,
                    "profit_factor": r.profit_factor,
                    "exposure": r.exposure,
                    "bh_return": r.bh_return,
                    "annualized_return": r.annualized_return,
                    "retrain_count": r.retrain_count,
                }
                for name, r in results.items()
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
