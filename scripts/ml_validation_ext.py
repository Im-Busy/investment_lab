"""
ML Validation - Extended: XGBoost comparison and multi-asset signal generation.

Extends the base ML validation with:
1. XGBoost vs sklearn regime classifier comparison
2. Multi-asset backtest to generate 300+ trades for signal scorer
3. Comprehensive validation report
"""

from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indicators.regime_detector import RegimeDetector
from src.ml.features import FeatureEngineer
from src.ml.pipeline import MLPipeline
from src.ml.regime_model import RegimeClassifier
from src.ml.signal_scorer import SignalScorer
from src.strategies.backtest_py.runner import BacktestPyRunner

warnings.filterwarnings("ignore")

REPORT_DIR = Path("reports/ml_validation")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/{name}", dpi=140)
    plt.close()


def download_tickers(tickers: list, start: str, end: str) -> dict:
    """Download OHLCV data for multiple tickers."""
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index.name = "date"
            df = df.sort_index()
            if len(df) > 100:
                data[t] = df
        except Exception:
            pass
    return data


def generate_regime_labels(df):
    """Generate rule-based regime labels."""
    detector = RegimeDetector()
    result = detector.get_regime_series(df)
    labels = result["regime"].apply(lambda r: r.value)
    df["adx"] = result["adx"]
    df["atr"] = result["atr"]
    return labels, df


def encode_labels(y_train, y_test):
    """Encode string labels to integers."""
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc, le


def compare_regime_models(X, y):
    """Compare multiple ML regime classifiers."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    sklearn_configs = [
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42, class_weight="balanced"
            ),
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        ),
        (
            "Logistic Regression",
            LogisticRegression(max_iter=1000, random_state=42),
        ),
    ]

    # Try XGBoost separately
    xgboost_model = None
    try:
        from xgboost import XGBClassifier

        xgboost_model = (
            "XGBoost",
            XGBClassifier(
                n_estimators=100,
                max_depth=4,
                random_state=42,
                eval_metric="mlogloss",
            ),
        )
    except ImportError:
        pass

    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_str, y_test_str = y.iloc[:split_idx], y.iloc[split_idx:]

    # Encode for XGBoost
    y_train, y_test, le = encode_labels(y_train_str, y_test_str)

    results = {}

    for name, model in sklearn_configs:
        model.fit(X_train.fillna(0), y_train_str)
        train_pred = model.predict(X_train.fillna(0))
        test_pred = model.predict(X_test.fillna(0))

        train_acc = accuracy_score(y_train_str, train_pred)
        test_acc = accuracy_score(y_test_str, test_pred)

        results[name] = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "overfit_gap": train_acc - test_acc,
            "classification_report": classification_report(y_test_str, test_pred, output_dict=True),
        }

        print(
            f"  {name}: train={train_acc:.4f}, test={test_acc:.4f}, gap={train_acc - test_acc:.4f}"
        )

    # XGBoost
    if xgboost_model is not None:
        name, model = xgboost_model
        model.fit(X_train.fillna(0), y_train)
        train_pred_enc = model.predict(X_train.fillna(0))
        test_pred_enc = model.predict(X_test.fillna(0))

        train_pred = le.inverse_transform(train_pred_enc)
        test_pred = le.inverse_transform(test_pred_enc)

        train_acc = accuracy_score(y_train_str, train_pred)
        test_acc = accuracy_score(y_test_str, test_pred)

        results[name] = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "overfit_gap": train_acc - test_acc,
            "classification_report": classification_report(y_test_str, test_pred, output_dict=True),
        }

        print(
            f"  {name}: train={train_acc:.4f}, test={test_acc:.4f}, gap={train_acc - test_acc:.4f}"
        )

    return results


def multi_asset_trades(spy_df):
    """Run backtest on SPY + sector ETFs to generate more trades."""
    sector_tickers = [
        "QQQ",
        "IWM",
        "DIA",
        "XLK",
        "XLF",
        "XLE",
        "XLV",
        "XLI",
        "XLP",
        "XLU",
        "XLY",
        "XLB",
    ]

    all_data = {"SPY": spy_df}
    data_download = download_tickers(sector_tickers, "2015-01-01", "2024-12-31")
    all_data.update(data_download)

    print(f"\nDownloading {len(data_download)} sector ETFs...")
    print(f"Tickers: {list(data_download.keys())}")

    all_trades = []

    for ticker, df in all_data.items():
        try:
            runner = BacktestPyRunner(data=df, cash=100000, exclusive_orders=True)
            runner.run(
                min_confidence=0.60,
                min_confluence_count=2,
                risk_per_trade=0.02,
                max_open_positions=5,
                use_regime_filter=False,
            )
            trades_df = runner.get_trades()

            if len(trades_df) > 0:
                trades_df["ticker"] = ticker
                all_trades.append(trades_df)
                print(
                    f"  {ticker}: {len(trades_df)} trades, win_rate={trades_df['pnl'].gt(0).mean():.1%}"
                )
        except Exception as e:
            print(f"  {ticker}: ERROR - {e}")

    if not all_trades:
        return pd.DataFrame()

    combined = pd.concat(all_trades, ignore_index=True)
    print(f"\nTotal trades across all assets: {len(combined)}")
    print(f"Unique tickers: {combined['ticker'].nunique()}")
    print(f"Overall win rate: {combined['pnl'].gt(0).mean():.1%}")

    return combined


def train_signal_scorer_extended(trades_df):
    """Train signal scorer with multi-asset data."""
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    if trades_df.empty or len(trades_df) < 30:
        print("Insufficient trades for signal scorer.")
        return None

    signal_y = (trades_df["pnl"] > 0).astype(int)
    signal_feat_cols = ["entry_price", "exit_price", "size", "return_pct", "duration"]
    available_cols = [c for c in signal_feat_cols if c in trades_df.columns]

    signal_X = trades_df[available_cols].copy()

    # Convert duration to numeric
    if "duration" in signal_X.columns:
        signal_X["duration"] = pd.to_timedelta(signal_X["duration"]).dt.total_seconds()

    signal_X = signal_X.ffill().bfill().dropna()
    signal_y = signal_y.loc[signal_X.index]

    print(f"\nSignal scorer dataset: {len(signal_X)} trades, {len(signal_X.columns)} features")
    print(f"Profitable: {signal_y.sum()}/{len(signal_y)} ({signal_y.mean():.1%})")

    models_to_try = [
        (
            "Gradient Boosting",
            GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=100, max_depth=4, random_state=42, class_weight="balanced"
            ),
        ),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
    ]

    # XGBoost
    try:
        from xgboost import XGBClassifier

        models_to_try.append(
            (
                "XGBoost",
                XGBClassifier(
                    n_estimators=100, max_depth=3, random_state=42, eval_metric="logloss"
                ),
            )
        )
    except ImportError:
        pass

    # Split: 70/30
    split_idx = int(len(signal_X) * 0.7)
    X_train = signal_X.iloc[:split_idx]
    X_test = signal_X.iloc[split_idx:]
    y_train = signal_y.iloc[:split_idx]
    y_test = signal_y.iloc[split_idx:]

    signal_results = {}

    for name, model in models_to_try:
        try:
            m = clone(model)
            m.fit(X_train.fillna(0), y_train)

            train_pred = m.predict(X_train.fillna(0))
            test_pred = m.predict(X_test.fillna(0))

            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)

            try:
                test_proba = m.predict_proba(X_test.fillna(0))[:, 1]
                auc = roc_auc_score(y_test, test_proba)
            except Exception:
                auc = 0.5

            signal_results[name] = {
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "test_auc_roc": auc,
                "overfit_gap": train_acc - test_acc,
                "n_train": len(X_train),
                "n_test": len(X_test),
            }

            print(f"  {name}: train={train_acc:.4f}, test={test_acc:.4f}, AUC={auc:.4f}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    return signal_results


def walk_forward_regime_comparison(X, y):
    """Walk-forward validation comparing multiple models."""
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    sk_configs = [
        (
            "RF",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                class_weight="balanced",
            ),
        ),
        (
            "GB",
            GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        ),
        ("LR", LogisticRegression(max_iter=1000, random_state=42)),
    ]

    xgb_model = None
    try:
        from xgboost import XGBClassifier

        xgb_model = (
            "XGB",
            XGBClassifier(
                n_estimators=100,
                max_depth=4,
                random_state=42,
                eval_metric="mlogloss",
            ),
        )
    except ImportError:
        pass

    train_size = 300
    step_size = 100

    all_wf = {}
    for name, _ in sk_configs:
        all_wf[name] = {"train": [], "test": []}
    if xgb_model:
        all_wf[xgb_model[0]] = {"train": [], "test": []}

    start = 0
    while start + train_size + 50 < len(X):
        end = min(start + train_size + step_size, len(X))

        X_train = X.iloc[start : start + train_size]
        X_test = X.iloc[start + train_size : end]
        y_train = y.iloc[start : start + train_size]
        y_test = y.iloc[start + train_size : end]

        if len(X_test) < 20:
            start += step_size
            continue

        valid_train = X_train.notna().all(axis=1) & y_train.notna()
        valid_test = X_test.notna().all(axis=1) & y_test.notna()

        if valid_train.sum() < 50 or valid_test.sum() < 20:
            start += step_size
            continue

        # sklearn models
        for name, model in sk_configs:
            try:
                m = clone(model)
                m.fit(X_train[valid_train].fillna(0), y_train[valid_train])
                train_pred = m.predict(X_train[valid_train].fillna(0))
                test_pred = m.predict(X_test[valid_test].fillna(0))

                all_wf[name]["train"].append(accuracy_score(y_train[valid_train], train_pred))
                all_wf[name]["test"].append(accuracy_score(y_test[valid_test], test_pred))
            except Exception:
                pass

        # XGBoost model
        if xgb_model is not None:
            try:
                name, xb_model = xgb_model
                m = clone(xb_model)
                y_train_enc = y_train[valid_train].astype("category").cat.codes
                y_test_enc = y_test[valid_test].astype("category").cat.codes
                m.fit(X_train[valid_train].fillna(0), y_train_enc)
                train_pred_enc = m.predict(X_train[valid_train].fillna(0))
                test_pred_enc = m.predict(X_test[valid_test].fillna(0))
                train_pred = y_train[valid_train].cat.categories[train_pred_enc].values
                test_pred = y_test[valid_test].cat.categories[test_pred_enc].values
                all_wf[name]["train"].append(accuracy_score(y_train[valid_train], train_pred))
                all_wf[name]["test"].append(accuracy_score(y_test[valid_test], test_pred))
            except Exception:
                pass

        start += step_size

    return all_wf


def print_summary(report_data):
    """Print comprehensive summary."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ML VALIDATION REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Regime comparison
    print("\n--- REGIME CLASSIFICATION COMPARISON ---")
    for name, r in report_data.get("regime_comparison", {}).items():
        if isinstance(r, dict) and "test_accuracy" in r:
            print(
                f"  {name:20s}: test_acc={r['test_accuracy']:.4f}, overfit={r['overfit_gap']:.4f}"
            )

    # Walk-forward
    print("\n--- WALK-FORWARD (23 folds) ---")
    for name, scores in report_data.get("walk_forward", {}).items():
        if isinstance(scores, dict) and scores.get("test"):
            print(
                f"  {name:20s}: train={np.mean(scores['train']):.4f}, test={np.mean(scores['test']):.4f} (std={np.std(scores['test']):.4f})"
            )

    # Signal scorer
    print("\n--- SIGNAL SCORER (multi-asset) ---")
    for name, r in report_data.get("signal_comparison", {}).items():
        if isinstance(r, dict) and "test_accuracy" in r:
            print(f"  {name:20s}: test_acc={r['test_accuracy']:.4f}, AUC={r['test_auc_roc']:.4f}")

    print("=" * 70)
    print("END OF REPORT")
    print("=" * 70)


def main():
    import matplotlib

    matplotlib.use("Agg")

    # 1. Load SPY
    print("=" * 60)
    print("1. LOADING DATA")
    print("=" * 60)

    spy_df = yf.download("SPY", start="2015-01-01", end="2024-12-31", auto_adjust=True)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.get_level_values(0)
    spy_df.index.name = "date"
    spy_df = spy_df.sort_index()
    print(f"SPY: {len(spy_df)} bars")

    # 2. Regime labels
    print("\n" + "=" * 60)
    print("2. GENERATING REGIME LABELS")
    print("=" * 60)
    regime_labels, spy_df = generate_regime_labels(spy_df)
    print(f"Regimes: {regime_labels.value_counts().sort_index().to_dict()}")

    # 3. Features
    engineer = FeatureEngineer()
    numeric_features = engineer.generate_features(spy_df).select_dtypes(include=[np.number])
    numeric_features = numeric_features.ffill().bfill().dropna()
    print(f"\nFeatures: {numeric_features.shape}")

    y = regime_labels.reindex(numeric_features.index).dropna()
    X = numeric_features.reindex(y.index)

    # 4. Regime model comparison
    print("\n" + "=" * 60)
    print("3. REGIME CLASSIFIER COMPARISON")
    print("=" * 60)
    regime_comparison = compare_regime_models(X, y)

    # 5. Walk-forward
    print("\n" + "=" * 60)
    print("4. WALK-FORWARD VALIDATION")
    print("=" * 60)
    wf_results = walk_forward_regime_comparison(X, y)

    # 6. Multi-asset trades
    print("\n" + "=" * 60)
    print("5. MULTI-ASSET TRADE GENERATION")
    print("=" * 60)
    all_trades = multi_asset_trades(spy_df)

    # 7. Signal scorer
    signal_results = None
    if not all_trades.empty:
        print("\n" + "=" * 60)
        print("6. SIGNAL SCORER (Multi-Asset)")
        print("=" * 60)
        signal_results = train_signal_scorer_extended(all_trades)

    # 8. Summary
    report_data = {
        "regime_comparison": {
            k: {kk: vv for kk, vv in v.items() if kk != "classifier"}
            for k, v in regime_comparison.items()
        },
        "walk_forward": wf_results,
        "signal_comparison": signal_results or {},
    }

    print_summary(report_data)


if __name__ == "__main__":
    main()
