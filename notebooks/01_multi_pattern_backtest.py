# %% [markdown]
# # Multi-Pattern Strategy Backtest Analysis
#
# This notebook demonstrates how to run the multi-pattern confluence strategy
# using the backtesting.py framework and generate visualization reports.
#
# ---
#
# ## Quick Configuration Guide
#
# Modify the `CONFIG` dictionary in Section 1 to customize:
# - Data source and date range
# - Backtest parameters (capital, commission, risk)
# - Strategy parameters (confidence threshold, position limits)
# - Output settings

# %% [markdown]
# ---
#
# ## 1. Configuration Section
#
# **Modify parameters below to customize the backtest.**

# %%
# ============================================================
# CONFIGURATION - Modify these parameters to customize analysis
# ============================================================

CONFIG = {
    # ----------------------------------------------------------
    # Data Configuration
    # ----------------------------------------------------------
    "data": {
        "file": "SPY_daily.csv",  # Data file name
        "directory": "data/raw",  # Data directory
        "start_date": None,  # Start date filter (None = all)
        "end_date": None,  # End date filter (None = all)
        "columns": ["Open", "High", "Low", "Close", "Volume"],
    },
    # ----------------------------------------------------------
    # Backtest Configuration
    # ----------------------------------------------------------
    "backtest": {
        "initial_equity": 100000,  # Starting capital
        "commission": 0.001,  # Commission rate (0.1%)
        "exclusive_orders": True,  # Close positions before opening new
    },
    # ----------------------------------------------------------
    # Strategy Configuration
    # ----------------------------------------------------------
    "strategy": {
        "min_confidence": 0.55,  # Minimum signal confidence
        "risk_per_trade": 0.02,  # Risk per trade (2%)
        "max_open_positions": 10,  # Maximum concurrent positions
    },
    # ----------------------------------------------------------
    # Output Configuration
    # ----------------------------------------------------------
    "output": {
        "directory": "reports",  # Output directory
        "save_plots": True,  # Save plots to disk
        "show_plots": True,  # Display plots in notebook
        "dpi": 150,  # Plot resolution
    },
}

# %% [markdown]
# ---
#
# ## 2. Setup and Imports

# %%
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

# Setup project root FIRST (before any src imports)
# Try different possible project root locations
candidates = [
    Path("..").resolve(),
    Path(".").resolve(),
]
for root in candidates:
    if (root / "src").exists():
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        project_root = root
        break
else:
    project_root = Path(".").resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# Import notebook helpers
from src.utils.notebook_helpers import (
    load_price_data,
    print_data_summary,
    print_backtest_summary,
    get_output_path,
    create_runner_kwargs,
    create_strategy_kwargs,
    calculate_performance_metrics,
    calculate_trade_statistics,
)

# Standard imports

# Import backtesting.py integration
from src.strategies.backtest_py import BacktestPyRunner, MultiPatternStrategySimple

# Import visualization

print("✅ Imports successful!")
print(f"Project root: {project_root}")

# %% [markdown]
# ---
#
# ## 3. Load Data

# %%
# Load data using helper function
df = load_price_data(CONFIG, project_root)

# Display data summary
print_data_summary(df, title="SPY Daily Data Summary")

# %%
# Visualize price data
import mplfinance as mpf

mpf.plot(
    df.tail(200),
    type="candle",
    style="charles",
    title="SPY - Last 200 Days",
    volume=True,
    figsize=(14, 8),
)

# %% [markdown]
# ---
#
# ## 4. Run Backtest

# %%
# Initialize runner with config
runner = BacktestPyRunner(data=df, **create_runner_kwargs(CONFIG))

print("✅ Backtest runner initialized")
print(f"  Initial equity: ${CONFIG['backtest']['initial_equity']:,.0f}")
print(f"  Commission: {CONFIG['backtest']['commission'] * 100:.2f}%")

# %%
# Run with simplified strategy for faster backtesting
strategy_kwargs = create_strategy_kwargs(CONFIG)

print("🚀 Running backtest...")
print("  Strategy: MultiPatternStrategySimple")
print(f"  Min confidence: {strategy_kwargs['min_confidence']}")
print(f"  Risk per trade: {strategy_kwargs['risk_per_trade'] * 100:.1f}%")
print(f"  Max positions: {strategy_kwargs['max_open_positions']}")

results = runner.run(strategy_class=MultiPatternStrategySimple, **strategy_kwargs)

print("\n✅ Backtest complete!")

# %%
# Get detailed trades
trades_df = runner.get_trades()
print(f"\nTotal trades: {len(trades_df)}")
if len(trades_df) > 0:
    print(trades_df.head(10))

# %%
# Get equity curve
equity_curve = runner.get_equity_curve()
print(f"Equity curve shape: {equity_curve.shape}")
equity_curve.head()

# %% [markdown]
# ---
#
# ## 5. Generate Reports

# %%
# Generate comprehensive report
files = runner.generate_report(title="Multi-Pattern Strategy", include_plot=True)

print("\n📁 Generated files:")
for name, path in files.items():
    print(f"  {name}: {path}")

# %%
# Display tearsheet (opens in browser)
output_dir = get_output_path(CONFIG, project_root)
runner.plot(filename=str(output_dir / "multi_pattern_plot.html"))

# %% [markdown]
# ---
#
# ## 6. Parameter Optimization (Optional)
#
# Uncomment the cell below to run parameter optimization.

# %%
# Optimize parameters (this may take a while)
# Uncomment to run optimization

# optimized_results = runner.optimize(
#     strategy_class=MultiPatternStrategySimple,
#     max_tries=50,
#     min_confidence=[0.50, 0.55, 0.60, 0.65],
#     risk_per_trade=[0.01, 0.02, 0.03],
#     max_open_positions=[2, 3, 5]
# )

# %% [markdown]
# ---
#
# ## 7. Analyze Results

# %%
# Get backtest statistics
stats = runner.get_stats()

# Print summary using helper function
print_backtest_summary(stats, title="Multi-Pattern Strategy Performance")

# %%
# Analyze trade distribution
if len(trades_df) > 0:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PnL distribution
    axes[0, 0].hist(trades_df["pnl"], bins=30, edgecolor="black", alpha=0.7)
    axes[0, 0].axvline(x=0, color="red", linestyle="--", label="Break-even")
    axes[0, 0].set_title("Trade PnL Distribution")
    axes[0, 0].set_xlabel("PnL ($)")
    axes[0, 0].legend()

    # Win/Loss by direction
    direction_counts = (
        trades_df.groupby(["direction", trades_df["pnl"] > 0]).size().unstack(fill_value=0)
    )
    direction_counts.plot(kind="bar", ax=axes[0, 1], color=["red", "green"])
    axes[0, 1].set_title("Win/Loss by Direction")
    axes[0, 1].set_xlabel("Direction")
    axes[0, 1].legend(["Loss", "Win"])

    # Equity curve
    axes[1, 0].plot(equity_curve.index, equity_curve["Equity"])
    axes[1, 0].set_title("Equity Curve")
    axes[1, 0].set_xlabel("Date")
    axes[1, 0].set_ylabel("Equity ($)")
    axes[1, 0].grid(True, alpha=0.3)

    # Drawdown
    equity = equity_curve["Equity"]
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100
    axes[1, 1].fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)
    axes[1, 1].set_title("Drawdown")
    axes[1, 1].set_xlabel("Date")
    axes[1, 1].set_ylabel("Drawdown (%)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if configured
    if CONFIG["output"]["save_plots"]:
        save_path = get_output_path(CONFIG, project_root, "multi_pattern_analysis.png")
        fig.savefig(save_path, dpi=CONFIG["output"]["dpi"], bbox_inches="tight")

    if CONFIG["output"]["show_plots"]:
        plt.show()
    else:
        plt.close()

# %% [markdown]
# ---
#
# ## 8. Summary Statistics

# %%
# Calculate additional metrics
if len(equity_curve) > 0:
    equity_series = equity_curve["Equity"]
    metrics = calculate_performance_metrics(equity_series)

    print("\n" + "=" * 60)
    print("📊 DETAILED PERFORMANCE METRICS")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("=" * 60)

# Trade statistics
if len(trades_df) > 0:
    trade_stats = calculate_trade_statistics(trades_df)

    print("\n📊 TRADE STATISTICS:")
    for key, value in trade_stats.items():
        print(f"  {key}: {value}")

# %% [markdown]
# # ═══════════════════════════════════════════════════════════
# # 9. META-LABELING FILTER — One-Click Signal Filter
# # ═══════════════════════════════════════════════════════════
# #
# # Meta-labeling trains a second model to predict whether a trade
# # from your primary strategy will actually be profitable.
# #
# # **Just change TICKER and TOP_PCT below, then run all cells.**

# %%
# ╔══════════════════════════════════════════════════════════╗
# ║ CONFIG — Change these                                    ║
# ╚══════════════════════════════════════════════════════════╝
META_CONFIG = {
    "ticker": "SPY",              # <-- Change to your ticker
    "top_pct": 10.0,              # Top N% of signals to trade (more = more trades)
    "horizon": 5,                 # Bars to hold before forced exit
    "atr_mult_tp": 1.5,           # Take-profit ATR multiplier
    "atr_mult_sl": 1.0,           # Stop-loss ATR multiplier
    "meta_threshold": 0.5,        # 0.5=balanced, >0.5=more selective
    "initial_capital": 100_000.0,
    "commission": 0.001,
    "risk_per_trade": 0.02,
}

# %%
import pickle
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from src.ml.feature_engineering import FeatureExtractor
from src.ml.purged_cv import PurgedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("meta_labeler")
logger.setLevel(logging.WARNING)  # Quiet mode for notebook

ticker = META_CONFIG["ticker"]
csv_path = project_root / "data" / "raw" / f"{ticker}_daily.csv"
if csv_path.exists():
    df_meta = pd.read_csv(csv_path, parse_dates=True, index_col=0).sort_index()
else:
    df_meta = yf.download(ticker, start="2017-01-01", end="2024-12-31", progress=False)
    if isinstance(df_meta.columns, pd.MultiIndex):
        df_meta.columns = df_meta.columns.get_level_values(0)

print(f"Loaded {ticker}: {len(df_meta)} bars ({df_meta.index[0].date()} to {df_meta.index[-1].date()})")

# %%
# ── Extract features ──
extractor = FeatureExtractor()
features = extractor.extract_all_features(df_meta)
print(f"Features: {features.shape[1]} columns")

# ── Generate top-N% signals (rank-based, no model needed) ──
# Uses a simple momentum heuristic as a proxy for primary model signals
close = df_meta["Close"]
momentum_5d = close.pct_change(5)  # Simple 5-day momentum
momentum_5d = momentum_5d.dropna()

bt_indices = features.index.intersection(momentum_5d.index)
feat_bt = features.loc[bt_indices].dropna()
mom_bt = momentum_5d.loc[feat_bt.index]

n_signals = max(10, int(len(feat_bt) * META_CONFIG["top_pct"] / 100))
signal_dates = mom_bt.nlargest(n_signals).index
print(f"Top {META_CONFIG['top_pct']:.0f}% = {n_signals} signals (of {len(feat_bt)} bars)")

# %%
# ── ATR calculation ──
high, low, c = df_meta["High"], df_meta["Low"], df_meta["Close"]
tr = pd.concat([high - low, (high - c.shift(1)).abs(), (low - c.shift(1)).abs()], axis=1).max(axis=1)
atr = tr.rolling(14).mean()

# ── Simple backtest engine ──
@dataclass
class MTrade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    return_pct: float
    pnl: float
    exit_reason: str
    bars_held: int

trades: List[MTrade] = []
in_trade = False
entry_price = tp_price = sl_price = 0.0
position_size = bars_in_trade = 0
entry_date: Optional[pd.Timestamp] = None
signal_set = set(signal_dates)
bt_indices_list = list(df_meta.index)

for i in range(len(bt_indices_list)):
    idx = bt_indices_list[i]
    bar = df_meta.iloc[i]
    if in_trade:
        bars_in_trade += 1
        exit_price = None
        exit_reason = "held"
        if bar["Low"] <= sl_price:
            exit_price = sl_price; exit_reason = "SL"
        elif bar["High"] >= tp_price:
            exit_price = tp_price; exit_reason = "TP"
        elif bars_in_trade >= META_CONFIG["horizon"]:
            exit_price = bar["Close"]; exit_reason = "Time"
        if exit_price is not None:
            ret = (exit_price - entry_price) / entry_price
            pnl = position_size * (exit_price - entry_price)
            pnl -= position_size * entry_price * META_CONFIG["commission"] * 2
            trades.append(MTrade(entry_date, idx, entry_price, exit_price, ret*100, pnl, exit_reason, bars_in_trade))
            in_trade = False
        continue
    if idx not in signal_set:
        continue
    c_val = bar["Close"]
    if pd.isna(c_val):
        continue
    _atr = atr.get(idx)
    if _atr is None:
        _atr_idx = atr.index.get_indexer([idx], method="ffill")[0]
        _atr = atr.iloc[_atr_idx] if _atr_idx >= 0 else None
    if _atr is None or pd.isna(_atr) or _atr <= 0:
        continue
    risk = META_CONFIG["initial_capital"] * META_CONFIG["risk_per_trade"]
    sl_dist = META_CONFIG["atr_mult_sl"] * _atr
    pos = int(risk / sl_dist) if sl_dist > 0 else 0
    if pos == 0:
        continue
    entry_price = c_val
    tp_price = c_val + META_CONFIG["atr_mult_tp"] * _atr
    sl_price = c_val - META_CONFIG["atr_mult_sl"] * _atr
    entry_date = idx
    position_size = pos
    bars_in_trade = 0
    in_trade = True

if in_trade:
    last = df_meta.iloc[-1]
    ret = (last["Close"] - entry_price) / entry_price
    pnl = position_size * (last["Close"] - entry_price)
    pnl -= position_size * entry_price * META_CONFIG["commission"] * 2
    trades.append(MTrade(entry_date, df_meta.index[-1], entry_price, last["Close"], ret*100, pnl, "EOD", bars_in_trade))

print(f"Primary model generated {len(trades)} trades")
tp_count = sum(1 for t in trades if t.exit_reason == "TP")
sl_count = sum(1 for t in trades if t.exit_reason == "SL")
print(f"  TP: {tp_count}, SL: {sl_count}, Time/EOD: {len(trades) - tp_count - sl_count}")

# %%
# ── Train Meta-Labeler ──
if len(trades) >= 20:
    entry_dates = [t.entry_date for t in trades]
    meta_X = features.loc[features.index.isin(entry_dates)]
    meta_y = pd.Series([t.exit_reason == "TP" for t in trades], index=entry_dates, dtype=int)
    valid = meta_X.dropna().index.intersection(meta_y.dropna().index)
    X_meta = meta_X.loc[valid]
    y_meta = meta_y.loc[valid]

    print(f"Meta-labeler training: {len(X_meta)} trades, {X_meta.shape[1]} features")
    print(f"Profitable: {y_meta.sum()} ({y_meta.mean()*100:.0f}%)")

    # Use simple 70/30 split
    split = int(len(X_meta) * 0.7)
    X_tr, X_te = X_meta.iloc[:split], X_meta.iloc[split:]
    y_tr, y_te = y_meta.iloc[:split], y_meta.iloc[split:]

    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    meta_model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        min_child_samples=10, random_state=42, verbose=-1,
    )
    meta_model.fit(X_tr, y_tr)
    meta_auc = roc_auc_score(y_te, meta_model.predict_proba(X_te)[:, 1])
    print(f"Meta-label AUC: {meta_auc:.4f}")

    # ── Apply meta-label filter ──
    meta_probs = meta_model.predict_proba(X_meta)[:, 1]
    meta_accept = meta_probs >= META_CONFIG["meta_threshold"]
    accepted_dates = set(X_meta.index[meta_accept])
    filtered_trades = [t for t in trades if t.entry_date in accepted_dates]

    def _metrics(t_list, label):
        if not t_list:
            return {"trades": 0, "return": 0.0, "win_rate": 0.0, "pf": 0.0}
        caps = META_CONFIG["initial_capital"]
        td = pd.DataFrame([{"ret": t.return_pct, "pnl": t.pnl} for t in t_list])
        wins = td[td["ret"] > 0]
        losses = td[td["ret"] <= 0]
        return {
            "trades": len(td),
            "return": td["pnl"].sum() / caps * 100,
            "win_rate": len(wins) / max(len(td), 1) * 100,
            "pf": wins["pnl"].sum() / max(abs(losses["pnl"].sum()), 1.0),
        }

    uf = _metrics(trades, "unfiltered")
    ff = _metrics(filtered_trades, "filtered")

    print(f"\n{'='*60}")
    print(f"  Meta-Labeling Results — {ticker}")
    print(f"{'='*60}")
    print(f"  {'':<20} {'Unfiltered':>15} {'Filtered':>15}")
    print(f"  {'-'*50}")
    print(f"  Trades          {uf['trades']:>15d}   {ff['trades']:>15d}")
    print(f"  Total Return %  {uf['return']:>15.2f}   {ff['return']:>15.2f}")
    print(f"  Win Rate %      {uf['win_rate']:>15.1f}   {ff['win_rate']:>15.1f}")
    print(f"  Profit Factor   {uf['pf']:>15.2f}   {ff['pf']:>15.2f}")
    print(f"{'='*60}")

    if not filtered_trades:
        print("\n⚠️  All trades filtered out. Lower meta_threshold to see results.")
    elif meta_auc < 0.55:
        print(f"\n⚠️  Meta-label AUC={meta_auc:.2f} is low. Results may be unreliable.")
else:
    print(f"⚠️  Only {len(trades)} trades — need 20+. Increase top_pct to get more trades.")

# %% [markdown]
# # ═══════════════════════════════════════════════════════════
# # 10. SHAP MODEL EXPLAINABILITY — One-Click
# # ═══════════════════════════════════════════════════════════
# #
# # Explains what features drive your model's predictions using
# # SHAP (SHapley Additive exPlanations).
# #
# # **Just change MODEL_PATH to your .pkl file and run all cells.**

# %%
# ╔══════════════════════════════════════════════════════════╗
# ║ CONFIG — Change these                                    ║
# ╚══════════════════════════════════════════════════════════╝
SHAP_CONFIG = {
    "model_path": "models/pattern_classifier_20260509_052103_v3_ca_JOE_catboost.pkl",
    "output_dir": "outputs/shap/notebook_analysis",
    "n_background": 50,
    "n_importance": 50,
    "n_beeswarm": 50,
}
# %%
from src.ml.shap_dashboard import SHAPDashboard
from pathlib import Path as _Path

model_path = project_root / SHAP_CONFIG["model_path"]
with open(model_path, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
feature_names = model_data["feature_names"]
print(f"Model: {type(model).__name__}")
print(f"Features expected: {len(feature_names)}")

# %%
# ── Prepare data ──
available = [f for f in feature_names if f in features.columns]
missing = set(feature_names) - set(available)
if missing:
    print(f"Missing features: {sorted(missing)}")

# Add missing features as zero-filled columns (CatBoost requires all features)
X_aligned = features[available].copy()
for mf in sorted(missing):
    X_aligned[mf] = 0.0
# Reorder columns to match model's expected order
X_model = X_aligned[feature_names].dropna()
print(f"Data aligned: {len(X_model)} rows, {len(feature_names)} columns")

# Split into train/test by time
split_idx = int(len(X_model) * 0.7)
X_train = X_model.iloc[:split_idx]
X_test = X_model.iloc[split_idx:]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# %%
import matplotlib
matplotlib.use("Agg")

# ── Run SHAP analysis ──
dashboard = SHAPDashboard(
    model=model,
    X_background=X_train.iloc[:SHAP_CONFIG["n_background"]],
    output_dir=SHAP_CONFIG["output_dir"],
)
importance = dashboard.global_importance(X_test[:SHAP_CONFIG["n_importance"]])
print(importance.summary(top_n=20))

# ── Bar plot ──
dashboard.bar_plot(X_test[:SHAP_CONFIG["n_importance"]], max_display=20, show=True)
print("✅ Bar plot saved")

# ── Beeswarm ──
dashboard.beeswarm(X_test[:SHAP_CONFIG["n_beeswarm"]], max_display=15, show=True)
print("✅ Beeswarm plot saved")

# ── Waterfall for first prediction ──
explanation = dashboard.explain(X_test.iloc[0])
print("\nSingle prediction breakdown:")
print(explanation.summary(top_n=10))

# ── Save CSV for further analysis ──
out_dir = _Path(SHAP_CONFIG["output_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
importance.to_dataframe().to_csv(out_dir / "shap_importance.csv")
print(f"\n✅ SHAP importance CSV saved to {out_dir / 'shap_importance.csv'}")
print(f"✅ All plots saved to {out_dir}/")
