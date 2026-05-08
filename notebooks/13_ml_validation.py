# %% [markdown]
# # ML-Enhanced Regime Detection & Signal Scoring
#
# This notebook validates the ML components (regime classifier, signal scorer) against the existing rule-based baseline.
#
# ## Objectives
# 1. Generate regime labels using rule-based RegimeDetector
# 2. Train ML RegimeClassifier and compare vs rule-based accuracy
# 3. Generate signal features from historical signals
# 4. Train SignalScorer on profitable vs unprofitable trades
# 5. Walk-forward validation with embargo periods
# 6. Compare ML-enhanced vs rule-based regime detection

# %% [markdown]
# ## Configuration

# %%
import yfinance as yf

CONFIG = {
    "data": {
        "ticker": "SPY",
        "start": "2015-01-01",
        "end": "2024-12-31",
    },
    "regime": {
        "adx_threshold": 25,
        "atr_window": 14,
        "atr_percentile": 80,
    },
    "ml": {
        "regime_model": "random_forest",
        "signal_model": "gradient_boosting",
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
        "test_size": 0.3,
    },
    "walk_forward": {
        "train_size": 300,
        "step_size": 100,
        "embargo_days": 10,
    },
    "output": {
        "report_path": "../reports/ml_validation/",
    },
}

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

# %% [markdown]
# ## 1. Load and Prepare Data

# %%
df = yf.download(
    CONFIG["data"]["ticker"],
    start=CONFIG["data"]["start"],
    end=CONFIG["data"]["end"],
    auto_adjust=True,
)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.rename(
    columns={"High": "High", "Low": "Low", "Close": "Close", "Open": "Open", "Volume": "Volume"}
)
df.index.name = "date"
df = df.sort_index()

print(f"Data loaded: {len(df)} bars, columns: {df.columns.tolist()}")
df.head()

# %% [markdown]
# ## 2. Generate Rule-Based Regime Labels

# %%
from src.indicators.regime_detector import RegimeDetector

detector = RegimeDetector()
regime_series = detector.classify(df)
df["regime"] = regime_series

regime_result = detector.get_regime_series(df)
regime_labels = regime_result["regime"].apply(lambda r: r.value)
df["adx"] = regime_result["adx"]
df["atr"] = regime_result["atr"]

print("Regime distribution:")
print(regime_labels.value_counts().sort_index())
print(f"\nRegime classes: {sorted(regime_labels.unique())}")

# %% [markdown]
# ### Regime Time Series Visualization

# %%
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

colors = {"Trending": "blue", "Ranging": "orange", "Volatile": "red", "Transition": "gray"}
regime_colors = regime_labels.map(colors)

axes[0].scatter(df.index, df["Close"], c=regime_colors, s=1)
axes[0].set_title("SPY Close Price with Regime Labels")
axes[0].set_ylabel("Price ($)")

axes[1].plot(df.index, df["adx"], color="purple", linewidth=0.8)
axes[1].axhline(y=25, color="red", linestyle="--", alpha=0.5)
axes[1].axhline(y=20, color="orange", linestyle="--", alpha=0.5)
axes[1].set_ylabel("ADX(14)")

axes[2].plot(df.index, df["atr"], color="green", linewidth=0.8)
axes[2].set_ylabel("ATR(14)")
axes[2].set_xlabel("Date")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Train ML Regime Classifier

# %%
from src.ml.regime_model import RegimeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# %% [markdown]
# ### Generate Features for ML

# %%
from src.ml.features import FeatureEngineer

engineer = FeatureEngineer()
features_df = engineer.generate_features(df)

numeric_features = features_df.select_dtypes(include=[np.number])
numeric_features = numeric_features.ffill().bfill().dropna()

print(f"Generated {len(numeric_features.columns)} features, {len(numeric_features)} samples")
print(f"Features: {list(numeric_features.columns[:20])}...")

# %%
y = regime_labels.reindex(numeric_features.index)
y = y.dropna()

X = numeric_features.reindex(y.index)

print(f"Final dataset: X={X.shape}, y={y.shape}")
print(f"Target distribution:\n{y.value_counts().sort_index()}")

# %% [markdown]
# ### Train/Test Split & Model Training

# %%
test_size = CONFIG["ml"]["test_size"]
split_idx = int(len(X) * (1 - test_size))

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# %%
regime_classifier = RegimeClassifier(
    model_type=CONFIG["ml"]["regime_model"],
    n_estimators=CONFIG["ml"]["n_estimators"],
    max_depth=CONFIG["ml"]["max_depth"],
    random_state=CONFIG["ml"]["random_state"],
)

train_results = regime_classifier.train(X_train, y_train, test_size=0.0)
print("Training Results:")
for k, v in train_results.items():
    if isinstance(v, dict):
        continue
    print(f"  {k}: {v}")

# %% [markdown]
# ### Evaluate ML Regime Classifier

# %%
y_pred_test = regime_classifier.predict(X_test)
ml_accuracy = accuracy_score(y_test, y_pred_test)
print(f"ML Regime Classifier Test Accuracy: {ml_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))

# %%
classes = regime_classifier.classes_
cm = confusion_matrix(y_test, y_pred_test, labels=classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix - ML Regime Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %% [markdown]
# ### Feature Importance

# %%
importance_df = regime_classifier.get_feature_importance(top_n=15)
print(importance_df)

# %%
plt.figure(figsize=(10, 6))
sns.barplot(
    data=importance_df, x="importance", y="feature", palette="viridis", hue="feature", legend=False
)
plt.title("Top 15 Features for Regime Classification")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Compare ML vs Rule-Based Regime Detection

# %%
X_all_filled = X.ffill().bfill()
ml_regime_all = regime_classifier.predict(X_all_filled)

comparison_df = pd.DataFrame(
    {
        "rule_based": y,
        "ml": ml_regime_all,
    }
)

overall_agreement = (comparison_df["rule_based"] == comparison_df["ml"]).mean()
print(f"Overall agreement: {overall_agreement:.2%}")
print("\nML regime distribution:")
print(comparison_df["ml"].value_counts().sort_index())
print("\nRule-based regime distribution:")
print(comparison_df["rule_based"].value_counts().sort_index())

# %% [markdown]
# ## 4. Train Signal Scorer

# %%
from src.ml.signal_scorer import SignalScorer
from src.strategies.backtest_py.runner import BacktestPyRunner
from src.strategies.connors_rsi import ConnorsRSIMeanReversion

# %% [markdown]
# ### Generate Historical Signals & Label Profitable Trades

# %%
runner = BacktestPyRunner(data=df, cash=100000, verbose=False)
backtest_result = runner.run(ConnorsRSIMeanReversion)
trades = backtest_result.get("trades", [])
print(f"Generated {len(trades)} trades")

if len(trades) > 0:
    trades_df = pd.DataFrame(trades) if isinstance(trades, list) else trades
    if "pnl" in trades_df.columns:
        trades_df["is_profitable"] = (trades_df["pnl"] > 0).astype(int)
        print(
            f"Profitable trades: {trades_df['is_profitable'].sum()}/{len(trades_df)} "
            f"({trades_df['is_profitable'].mean():.1%})"
        )
        print("\nTrade P&L summary:")
        print(trades_df["pnl"].describe())
    else:
        print("No pnl column in trades - skipping signal scorer")
else:
    print("No trades generated - skipping signal scorer")

# %%
if len(trades) > 0 and "pnl" in trades_df.columns:
    trades_df = pd.DataFrame(trades) if isinstance(trades, list) else trades

    signal_feat_cols = ["entry_price", "stop_loss", "take_profit", "atr_at_entry"]
    available_cols = [c for c in signal_feat_cols if c in trades_df.columns]

    if available_cols:
        signal_X = trades_df[available_cols].copy()
        signal_y = (trades_df["pnl"] > 0).astype(int)

        signal_X = signal_X.ffill().bfill().dropna()
        signal_y = signal_y.loc[signal_X.index]

        if len(signal_X) >= 10:
            scorer = SignalScorer(
                model_type=CONFIG["ml"]["signal_model"],
                n_estimators=CONFIG["ml"]["n_estimators"],
                max_depth=CONFIG["ml"]["max_depth"],
                random_state=CONFIG["ml"]["random_state"],
            )
            try:
                signal_results = scorer.train(signal_X, signal_y)
                print("Signal Scorer Training Results:")
                for k, v in signal_results.items():
                    if isinstance(v, (int, float)):
                        print(f"  {k}: {v:.4f}")
                    elif isinstance(v, str):
                        print(f"  {k}: {v}")
            except Exception as e:
                print(f"Signal scorer training failed: {e}")
        else:
            print(f"Insufficient trades with features: {len(signal_X)} (need >=10)")
    else:
        print("No signal feature columns available in trades")
else:
    print("Skipping signal scorer - no valid trades")

# %% [markdown]
# ## 5. Complete ML Pipeline

# %%
from src.ml.pipeline import MLPipeline

pipeline = MLPipeline(
    regime_model_type=CONFIG["ml"]["regime_model"],
    signal_model_type=CONFIG["ml"]["signal_model"],
    n_estimators=CONFIG["ml"]["n_estimators"],
    max_depth=CONFIG["ml"]["max_depth"],
    random_state=CONFIG["ml"]["random_state"],
)

# %%
pipeline_result = pipeline.run_pipeline(
    df=df,
    regime_labels=regime_labels,
    signal_features=None,
    signal_labels=None,
)

print("Pipeline Results:")
print(f"  Regime accuracy: {pipeline_result.pipeline_metrics['regime_accuracy']:.4f}")
print(f"  Features used: {pipeline_result.pipeline_metrics['n_features']}")

# %% [markdown]
# ## 6. Walk-Forward Validation

# %%
pipeline2 = MLPipeline(
    regime_model_type=CONFIG["ml"]["regime_model"],
    n_estimators=CONFIG["ml"]["n_estimators"],
    max_depth=CONFIG["ml"]["max_depth"],
    random_state=CONFIG["ml"]["random_state"],
)

wf_results = pipeline2.walk_forward_validation(
    df=df,
    regime_labels=regime_labels,
    signal_features=None,
    signal_labels=None,
)

if "regime" in wf_results:
    print("Walk-Forward Regime Results:")
    regime_wf = wf_results["regime"]
    for k, v in regime_wf.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, list):
            print(f"  {k} ({len(v)} folds): {np.mean(v):.4f} +/- {np.std(v):.4f}")

# %% [markdown]
# ## 7. Summary & Conclusions

# %%
summary = {
    "ML Regime Accuracy": f"{ml_accuracy:.2%}",
    "Rule-Based vs ML Agreement": f"{overall_agreement:.2%}",
    "Features Generated": str(len(numeric_features.columns)),
    "Training Samples": str(len(numeric_features)),
    "Regime Class Distribution": regime_labels.value_counts().to_dict(),
}

print("=" * 60)
print("ML VALIDATION SUMMARY")
print("=" * 60)
for k, v in summary.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("NEXT STEPS")
print("=" * 60)
if ml_accuracy > 0.7:
    print("  ML regime detection shows promise (>70% accuracy)")
    print("  -> Integrate with ConfluenceScorer for ML-enhanced signal scoring")
    print("  -> Run full backtest with ML enhancement")
else:
    print("  ML regime accuracy is not significantly better than random")
    print("  -> Try feature engineering improvements")
    print("  -> Consider different model types (XGBoost, neural nets)")
    print("  -> Add more input features (volume, volatility, cross-asset)")
