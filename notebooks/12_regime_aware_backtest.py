# %% [markdown]
# # Regime-Aware Backtest Analysis
#
# ## Objectives
# 1. Detect market regimes (Trending/Ranging/Volatile/Transition) on SPY daily data
# 2. Run multi-pattern backtest and attribute trades to regimes
# 3. Compare strategy performance by regime type
# 4. Build regime-filtered strategy allocation recommendations
# 5. Quantify regime misclassification impact and adaptive routing benefits
#
# **Phase:** 4 — Regime Detection & Adaptive Strategy Selection
# **Status:** Notebook created per `plans/phased_implementation_plan.md` line 177

# %% [markdown]
# ## Configuration

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# %matplotlib inline

CONFIG = {
    "data": {"ticker": "SPY", "start": "2015-01-01", "end": "2024-12-31"},
    "regime": {"adx_threshold": 25, "atr_window": 14, "atr_percentile": 80},
    "backtest": {"cash": 100_000, "commission": 0.001},
}

# %% [markdown]
# ## 1. Load Data & Detect Regimes

# %%
df = yf.download(
    CONFIG["data"]["ticker"],
    start=CONFIG["data"]["start"],
    end=CONFIG["data"]["end"],
    auto_adjust=True,
    progress=False,
)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.columns = [
    c.capitalize() if c.lower() in ["open", "high", "low", "close", "volume"] else c
    for c in df.columns
]
df.index.name = "Date"
df.index = pd.to_datetime(df.index)
if "Volume" not in df.columns:
    df["Volume"] = 0
print(f"Loaded {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
df.head()

# %%
from src.indicators.regime_detector import RegimeDetector, RegimeState
from src.strategies.adaptive_router import AdaptiveRouter

detector = RegimeDetector()
regime_info = detector.get_regime_series(df)

regime_labels = regime_info["regime"].apply(lambda r: r.value if hasattr(r, "value") else str(r))
print("\nRegime distribution:")
print(regime_labels.value_counts().sort_index())
print(f"\nRegime pct: {regime_labels.value_counts(normalize=True).sort_index().to_dict()}")

# %% [markdown]
# ### Regime Time Series Visualization

# %%
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
colors = {"Trending": "green", "Ranging": "orange", "Volatile": "red", "Transition": "gray"}
rcolors = regime_labels.map(colors)

axes[0].scatter(df.index, df["Close"], c=rcolors, s=2, alpha=0.7)
axes[0].set_title("SPY Close Price with Regime Labels")
axes[0].set_ylabel("Price ($)")
axes[0].legend(["Trending", "Ranging", "Volatile", "Transition"], loc="upper left", fontsize=7)

axes[1].plot(df.index, regime_info["adx"], color="purple", linewidth=0.8)
axes[1].axhline(y=25, color="green", linestyle="--", alpha=0.5, label="Trending threshold")
axes[1].axhline(y=20, color="orange", linestyle="--", alpha=0.5, label="Ranging threshold")
axes[1].set_ylabel("ADX(14)")
axes[1].legend(fontsize=7)

axes[2].plot(df.index, regime_info["atr"], color="blue", linewidth=0.8)
axes[2].axhline(
    y=regime_info["atr"].quantile(0.8), color="red", linestyle="--", alpha=0.5, label="80th %ile"
)
axes[2].set_ylabel("ATR(14)")
axes[2].set_xlabel("Date")
axes[2].legend(fontsize=7)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Run Multi-Pattern Strategy (backtesting.py)

# %%
from src.strategies.backtest_py.runner import BacktestPyRunner
from src.strategies.backtest_py.multi_pattern_strategy import MultiPatternStrategy

runner = BacktestPyRunner(
    data=df, cash=CONFIG["backtest"]["cash"], commission=CONFIG["backtest"]["commission"]
)
results = runner.run(strategy_class=MultiPatternStrategy)

stats = results["stats"]
trades_df = results["trades"]
print(f"Total trades: {len(trades_df)}")
if "Return [%]" in stats:
    print(f"Return: {stats['Return [%]']:.2f}%")
    print(f"Sharpe: {stats.get('Sharpe Ratio', 'N/A')}")
    print(f"Max DD: {stats.get('Max. Drawdown [%]', 'N/A')}")
    print(f"Win Rate: {stats.get('Win Rate [%]', 'N/A')}%")

# %% [markdown]
# ## 3. Attribute Trades to Regimes

# %%
if not trades_df.empty and len(trades_df) > 0:
    entry_times = pd.to_datetime(trades_df["entry_time"])
    regime_info["regime_str"] = regime_info["regime"].apply(
        lambda r: r.value if hasattr(r, "value") else str(r)
    )

    entry_regimes = []
    for et in entry_times:
        prior = regime_info.loc[regime_info.index <= et]
        if not prior.empty:
            entry_regimes.append(prior.iloc[-1]["regime_str"])
        else:
            entry_regimes.append("Unknown")

    trades_df_copy = trades_df.copy()
    trades_df_copy["regime_at_entry"] = entry_regimes

    print("Trade regime attribution:")
    regime_trade_counts = trades_df_copy["regime_at_entry"].value_counts()
    print(regime_trade_counts.to_string())
else:
    trades_df_copy = pd.DataFrame()
    print("No trades to attribute.")

# %% [markdown]
# ### Performance by Regime

# %%
if not trades_df_copy.empty and len(trades_df_copy) > 0:
    pnl_col = (
        "return_pct"
        if "return_pct" in trades_df_copy.columns
        else "ReturnPct"
        if "ReturnPct" in trades_df_copy.columns
        else None
    )
    if pnl_col:
        regime_perf = (
            trades_df_copy.groupby("regime_at_entry")
            .agg(
                trade_count=(pnl_col, "count"),
                mean_return=(pnl_col, "mean"),
                median_return=(pnl_col, "median"),
                std_return=(pnl_col, "std"),
                win_rate=(pnl_col, lambda x: (x > 0).mean()),
                cum_return=(pnl_col, "sum"),
            )
            .round(2)
        )

        print("\n=== PERFORMANCE BY REGIME ===")
        print(regime_perf.to_string())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].bar(
            regime_perf.index,
            regime_perf["mean_return"],
            color=[colors.get(r, "gray") for r in regime_perf.index],
        )
        axes[0].set_title("Mean Return by Regime")
        axes[0].set_ylabel("Return (%)")
        axes[0].axhline(y=0, color="black", linewidth=0.5)

        axes[1].bar(
            regime_perf.index,
            regime_perf["win_rate"],
            color=[colors.get(r, "gray") for r in regime_perf.index],
        )
        axes[1].set_title("Win Rate by Regime")
        axes[1].set_ylabel("Win Rate")

        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## 4. Regime-Filtered Strategy Simulation

# %%
router = AdaptiveRouter()

if not trades_df_copy.empty and len(trades_df_copy) > 0 and pnl_col:
    simulated_static_return = trades_df_copy[pnl_col].sum()

    filtered_rows = []
    for idx, row in trades_df_copy.iterrows():
        regime_str = row["regime_at_entry"]
        regime_state = (
            RegimeState[regime_str.upper()]
            if regime_str in [r.name for r in RegimeState]
            else RegimeState.TRANSITION
        )

        et = pd.Timestamp(row["entry_time"])
        active = router.get_active_strategies(["MultiPattern"], regime_state)

        if "MultiPattern" in active:
            filtered_rows.append(row)

    if filtered_rows:
        filtered_df = pd.DataFrame(filtered_rows)
        simulated_filtered_return = filtered_df[pnl_col].sum()

        print("\n=== STRATEGY SIMULATION ===")
        print(
            f"Static (no filter): {simulated_static_return:.2f}% cumulative, {len(trades_df_copy)} trades"
        )
        print(
            f"Regime-filtered:    {simulated_filtered_return:.2f}% cumulative, {len(filtered_df)} trades"
        )
        print(f"Difference:         {simulated_filtered_return - simulated_static_return:+.2f}%")
    else:
        print("No regime-filtered trades found.")

# %% [markdown]
# ## 5. Active Strategy Count by Regime Over Time

# %%
regime_ts = router.get_regime_time_series(df)
regime_ts["active_strategy_pct"] = (
    regime_ts["active_strategy_count"] / len(router.strategy_regime_map) * 100
)

print("\nActive strategies by regime:")
regime_summary = regime_ts.copy()
regime_summary["regime"] = regime_summary["regime"].astype(str)
regime_summary = (
    regime_summary.groupby("regime")
    .agg(
        avg_active=("active_strategy_count", "mean"),
        min_active=("active_strategy_count", "min"),
        max_active=("active_strategy_count", "max"),
        bar_count=("regime", "count"),
    )
    .round(1)
)
print(regime_summary.to_string())

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(
    regime_summary.index,
    regime_summary["avg_active"],
    color=[colors.get(r, "gray") for r in regime_summary.index],
)
ax.set_title("Average Active Strategies by Regime")
ax.set_ylabel("Number of Active Strategies")
ax.set_xlabel("Regime")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Equity Curve Split by Regime

# %%
if not trades_df_copy.empty and len(trades_df_copy) > 0 and pnl_col:
    equity_curve = pd.Series(0.0, index=df.index)

    for regime_name in ["Trending", "Ranging", "Volatile", "Transition"]:
        mask = trades_df_copy["regime_at_entry"] == regime_name
        regime_trades = trades_df_copy[mask].copy()
        if regime_trades.empty:
            continue

        cumret = (1 + regime_trades[pnl_col] / 100).cumprod() - 1
        plt.plot(
            regime_trades["entry_time"],
            cumret * 100,
            label=f"{regime_name} ({len(regime_trades)} trades)",
            linewidth=1.5,
        )

    plt.title("Cumulative Return by Regime at Entry")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 7. Summary & Recommendations

# %%
print("=" * 70)
print("REGIME-AWARE BACKTEST SUMMARY")
print("=" * 70)

print("\n1. REGIME DISTRIBUTION")
for regime, count in regime_labels.value_counts().sort_index().items():
    pct = count / len(regime_labels) * 100
    print(f"   {regime:>12s}: {count:>5d} bars ({pct:>5.1f}%)")

if not trades_df_copy.empty and len(trades_df_copy) > 0 and pnl_col:
    print("\n2. TRADE PERFORMANCE BY REGIME")
    for regime_name in ["Trending", "Ranging", "Volatile", "Transition"]:
        sub = trades_df_copy[trades_df_copy["regime_at_entry"] == regime_name]
        if len(sub) == 0:
            continue
        wr = (sub[pnl_col] > 0).mean()
        mr = sub[pnl_col].mean()
        print(f"   {regime_name:>12s}: {len(sub):>3d} trades, WR={wr:.0%}, Mean={mr:+.2f}%")

regime_rec = router.get_regime_recommendations(RegimeState.TRENDING)
print("\n3. ROUTER RECOMMENDATIONS")
print(f"   Enabled: {', '.join(regime_rec['enabled'])}")
print(f"   Disabled: {', '.join(regime_rec['disabled'])}")
print(f"   Advice: {regime_rec['advice']}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
if not trades_df_copy.empty and len(trades_df_copy) > 0 and pnl_col:
    best_regime = trades_df_copy.groupby("regime_at_entry")[pnl_col].mean().idxmax()
    worst_regime = trades_df_copy.groupby("regime_at_entry")[pnl_col].mean().idxmin()
    print(f"Best performing regime: {best_regime}")
    print(f"Worst performing regime: {worst_regime}")
    print("\nRegime-based filtering is most valuable when the strategy has")
    print("significantly different performance across regimes.")
    print("Consider adjusting confluence thresholds or position sizing")
    print("based on detected regime to improve risk-adjusted returns.")

# %% [markdown]
# # ═══════════════════════════════════════════════════════════
# # 8. ML REGIME GATE — One-Click Market Environment Filter
# # ═══════════════════════════════════════════════════════════
# #
# # Uses the project's trained RegimeGate model to predict whether
# # current market conditions are favorable for trading.
# #
# # The model was trained on 5 REITs (JOE+O+PLD+AMT+SPG) across
# # 10,000+ market days. It uses only market-level features:
# #   - SPY trend, volatility, drawdown
# #   - Bond yields (TLT), Gold (GLD), Small-caps (IWM)
# #   - Cross-asset correlation, value/growth spread
# #
# # **Just change TICKER below and run all cells in this section.**

# %%
# ╔══════════════════════════════════════════════════════════╗
# ║ CONFIG — Change these                                    ║
# ╚══════════════════════════════════════════════════════════╝
GATE_CONFIG = {
    "ticker": "SPY",           # <-- Change this to your ticker
    "gate_threshold": 0.6,     # Higher = stricter filtering (0.5-0.7)
    "csv_file": "SPY_daily.csv",
    "data_dir": "data/raw",
}

# %%
import sys
from pathlib import Path
_project = Path().absolute().parent if Path().absolute().name == "notebooks" else Path().absolute()
if str(_project) not in sys.path:
    sys.path.insert(0, str(_project))

from src.ml.regime_gate import RegimeGate

print("[OK] Loaded RegimeGate module")

# %%
gate = RegimeGate.load(f"{GATE_CONFIG['data_dir']}/../models/regime_gate_reit_basket.pkl")
# If model doesn't exist, train it
if gate.model_ is None:
    print("Training RegimeGate (one-time, ~15s)...")
    gate.train(["JOE", "O", "PLD", "AMT", "SPG"], data_dir=GATE_CONFIG["data_dir"])
    gate.save("models/regime_gate_reit_basket.pkl")

print(f"RegimeGate: Test AUC={gate.test_auc_:.3f}, {gate.feature_names_} features")
print("\nTop regime features:")
print(gate.get_feature_importance(10).to_string())

# %%
import pandas as pd
import yfinance as yf

# Load ticker data
ticker = GATE_CONFIG["ticker"]
csv_path = Path(GATE_CONFIG["data_dir"]) / GATE_CONFIG["csv_file"]
if csv_path.exists():
    ticker_df = pd.read_csv(csv_path, parse_dates=True, index_col=0).sort_index()
else:
    ticker_df = yf.download(ticker, start="2017-01-01", end="2024-12-31", progress=False)
    if isinstance(ticker_df.columns, pd.MultiIndex):
        ticker_df.columns = ticker_df.columns.get_level_values(0)

# Get regime gate predictions for every bar
trade_mask, confidence = gate.should_trade(
    ticker_df.index, market_data_dir=GATE_CONFIG["data_dir"],
    threshold=GATE_CONFIG["gate_threshold"],
)

good_bars = trade_mask.sum()
total_bars = len(trade_mask)
print(f"\n{ticker}: {good_bars}/{total_bars} bars favorable ({good_bars/total_bars*100:.0f}%)")
print(f"Confidence mean: {confidence.mean():.3f}, std: {confidence.std():.3f}")

# %%
# ── Visualize: show when the gate says trade vs skip ──
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

close = ticker_df["Close"]
axes[0].plot(close.index, close, color="black", linewidth=0.7, alpha=0.7)
in_favor = trade_mask[trade_mask].index
axes[0].scatter(in_favor, close.loc[in_favor], color="green", s=3, alpha=0.4, label="Favorable")
unfav = trade_mask[~trade_mask].index
axes[0].scatter(unfav, close.loc[unfav], color="red", s=3, alpha=0.4, label="Unfavorable")
axes[0].set_ylabel("Price")
axes[0].legend(loc="upper left", fontsize=8)
axes[0].set_title(f"{ticker} — Regime Gate Favorable vs Unfavorable ({GATE_CONFIG['gate_threshold']*100:.0f}% threshold)")

axes[1].fill_between(confidence.index, confidence, GATE_CONFIG["gate_threshold"],
                     where=(confidence >= GATE_CONFIG["gate_threshold"]),
                     color="green", alpha=0.3, label="Above threshold")
axes[1].fill_between(confidence.index, confidence, GATE_CONFIG["gate_threshold"],
                     where=(confidence < GATE_CONFIG["gate_threshold"]),
                     color="red", alpha=0.3, label="Below threshold")
axes[1].axhline(y=GATE_CONFIG["gate_threshold"], color="black", linestyle="--", alpha=0.5)
axes[1].set_ylabel("Gate Confidence")
axes[1].legend(loc="upper left", fontsize=8)

# Show yearly pass rate
yearly = trade_mask.groupby(trade_mask.index.year).mean() * 100
axes[2].bar(yearly.index.astype(str), yearly.values, color="steelblue")
axes[2].axhline(y=yearly.mean(), color="orange", linestyle="--", linewidth=1.5,
                label=f"Avg: {yearly.mean():.0f}%")
axes[2].set_ylabel("% Favorable")
axes[2].set_xlabel("Year")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()

# %%
# ── Gate Summary ──
print("=" * 60)
print("REGIME GATE SUMMARY")
print("=" * 60)
print(f"  Ticker:        {ticker}")
print(f"  Threshold:     {GATE_CONFIG['gate_threshold']}")
print(f"  Favorable:     {good_bars}/{total_bars} days ({good_bars/total_bars*100:.0f}%)")
print(f"  Confidence:    mean={confidence.mean():.3f}, std={confidence.std():.3f}")
print(f"\n  Use this to gate signals: only trade on green dots above.")
print(f"  Adjust gate_threshold to be more/less selective.")
print("=" * 60)
