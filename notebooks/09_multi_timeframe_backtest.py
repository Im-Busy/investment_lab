# %% [markdown]
# # Multi-Timeframe Multi-Strategy Backtest
#
# Backtests **all 17 Eterna indicator-based strategies** on a given ticker and timeframe, then displays comparative performance metrics.
#
# **Strategies**: SMA_Cross, EMA_Ribbon, VWAP_Bounce, Keltner, Chandelier, ADX, ParabolicSAR, Ichimoku, LinReg, RSI_Div, StochRSI, CCI, WilliamsR, TSI, UltOsc, AO, Chaikin
#
# **Data**: OHLCV CSV from `data/raw/{TICKER}_{TF}.csv` — requires ≥500 bars.

# %% [markdown]
# ## Configuration
#
# Set `TICKER` and `TIMEFRAME`. All strategies will be backtested automatically.

# %%
TICKER = "BTC_USD"
TIMEFRAME = "1h"  # Options: '5m', '1h', 'daily'

CONFIG = {
    "data": {
        "ticker": TICKER,
        "timeframe": TIMEFRAME,
        "filename": f"{TICKER}_{TIMEFRAME}.csv",
        "directory": "data/raw",
    },
    "backtest": {
        "initial_capital": 1_000_000,
        "commission": 0.001,
        "exclusive_orders": True,
    },
    "output": {
        "directory": "reports",
    },
}

# %% [markdown]
# ## 1. Setup and Imports

# %%
import sys
from pathlib import Path
import warnings
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from backtesting import Backtest
from tqdm.notebook import tqdm

try:
    from IPython.display import display
except ImportError:
    display = print

project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# %matplotlib inline
plt.style.use("seaborn-v0_8-whitegrid")

# %% [markdown]
# ## 2. Load Data

# %%
data_path = project_root / CONFIG["data"]["directory"] / CONFIG["data"]["filename"]
print(f"Loading: {data_path}")

df = pd.read_csv(data_path)

# Detect date column
for col in ["Datetime", "date", "Date", "timestamp"]:
    if col in df.columns:
        df = df.set_index(col)
        df.index = pd.to_datetime(df.index)
        break

# Normalize columns
col_map = {c.lower(): c for c in df.columns}
for std, variants in [
    ("Open", ["open"]),
    ("High", ["high"]),
    ("Low", ["low"]),
    ("Close", ["close"]),
    ("Volume", ["volume"]),
]:
    for v in variants:
        if v in col_map and col_map[v] != std:
            df[std] = df[col_map[v]]

if "Volume" not in df.columns:
    df["Volume"] = 0

df = df[["Open", "High", "Low", "Close", "Volume"]]

print(f"Bars: {len(df)}")
print(f"Range: {df.index[0]} to {df.index[-1]}")
print(f"Price: {df['Close'].min():.2f} - {df['Close'].max():.2f}")
df.head()

# %% [markdown]
# ## 3. Strategy Registry

# %%
from src.strategies.sma_crossover import SMACrossoverStrategy
from src.strategies.ema_ribbon import EMARibbonStrategy
from src.strategies.vwap_bounce import VWAPBounceStrategy
from src.strategies.keltner_channel import KeltnerChannelStrategy
from src.strategies.chandelier_exit import ChandelierExitStrategy
from src.strategies.adx_trend_strength import ADXTrendStrengthStrategy
from src.strategies.parabolic_sar import ParabolicSARStrategy
from src.strategies.ichimoku_cloud import IchimokuCloudStrategy
from src.strategies.linear_regression_channel import LinearRegressionChannelStrategy
from src.strategies.rsi_divergence import RSIDivergenceStrategy
from src.strategies.stoch_rsi_crossover import StochRSICrossoverStrategy
from src.strategies.cci_strategy import CCIStrategy
from src.strategies.williams_r_reversal import WilliamsRReversalStrategy
from src.strategies.tsi_strategy import TSIStrategy
from src.strategies.ultimate_oscillator import UltimateOscillatorStrategy
from src.strategies.awesome_oscillator import AwesomeOscillatorStrategy
from src.strategies.chaikin_oscillator import ChaikinOscillatorStrategy

STRATEGY_MAP = {
    "SMA_Cross": SMACrossoverStrategy,
    "EMA_Ribbon": EMARibbonStrategy,
    "VWAP_Bounce": VWAPBounceStrategy,
    "Keltner": KeltnerChannelStrategy,
    "Chandelier": ChandelierExitStrategy,
    "ADX": ADXTrendStrengthStrategy,
    "ParabolicSAR": ParabolicSARStrategy,
    "Ichimoku": IchimokuCloudStrategy,
    "LinReg": LinearRegressionChannelStrategy,
    "RSI_Div": RSIDivergenceStrategy,
    "StochRSI": StochRSICrossoverStrategy,
    "CCI": CCIStrategy,
    "WilliamsR": WilliamsRReversalStrategy,
    "TSI": TSIStrategy,
    "UltOsc": UltimateOscillatorStrategy,
    "AO": AwesomeOscillatorStrategy,
    "Chaikin": ChaikinOscillatorStrategy,
}

print(f"Loaded {len(STRATEGY_MAP)} strategies:")
for name in STRATEGY_MAP.keys():
    print(f"  - {name}")


# %% [markdown]
# ## 4. Run All Strategies

# %%
def safe_get(stats, key, default=0.0):
    """Safely get a value from the stats Series."""
    val = stats.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def extract_metrics(stats, strategy_name, initial_capital):
    """Extract key metrics from backtest stats."""
    return {
        "Strategy": strategy_name,
        "Total Trades": int(safe_get(stats, "# Trades", 0)),
        "Win Rate [%]": safe_get(stats, "Win Rate [%]", 0),
        "Sharpe Ratio": safe_get(stats, "Sharpe Ratio", 0),
        "Profit Factor": safe_get(stats, "Profit Factor", 0),
        "Max Drawdown [%]": safe_get(stats, "Max. Drawdown [%]", 0),
        "Return [%]": safe_get(stats, "Return [%]", 0),
        "Buy & Hold [%]": safe_get(stats, "Buy & Hold Return [%]", 0),
        "Equity Final [$]": safe_get(stats, "Equity Final [$]", initial_capital),
        "Equity Peak [$]": safe_get(stats, "Equity Peak [$]", initial_capital),
        "Exposure Time [%]": safe_get(stats, "Exposure Time [%]", 0),
    }


initial_capital = CONFIG["backtest"]["initial_capital"]
results = []

print(f"\nBacktesting {len(STRATEGY_MAP)} strategies on {TICKER} ({TIMEFRAME})...\n")

for strategy_name, strategy_cls in tqdm(STRATEGY_MAP.items(), desc="Strategies"):
    try:
        bt = Backtest(
            df,
            strategy_cls,
            cash=initial_capital,
            commission=CONFIG["backtest"]["commission"],
            exclusive_orders=CONFIG["backtest"]["exclusive_orders"],
        )
        stats = bt.run()
        metrics = extract_metrics(stats, strategy_name, initial_capital)
        results.append(metrics)
    except Exception as e:
        print(f"\n[ERROR] {strategy_name}: {e}")
        results.append(
            {
                "Strategy": strategy_name,
                "Total Trades": 0,
                "Win Rate [%]": 0,
                "Sharpe Ratio": -999,
                "Profit Factor": 0,
                "Max Drawdown [%]": 0,
                "Return [%]": 0,
                "Buy & Hold [%]": 0,
                "Equity Final [$]": initial_capital,
                "Equity Peak [$]": initial_capital,
                "Exposure Time [%]": 0,
            }
        )

results_df = pd.DataFrame(results)
print(f"\nCompleted: {len(results_df)} strategies backtested.")

# %% [markdown]
# ## 5. Comparison Table — Ranked by Sharpe Ratio

# %%
# Rank by Sharpe Ratio
ranked_df = results_df.sort_values("Sharpe Ratio", ascending=False).reset_index(drop=True)
ranked_df.index += 1  # 1-based ranking
ranked_df.index.name = "Rank"

# Display
print(f"\n{'=' * 80}")
print(f"Strategy Rankings for {TICKER} ({TIMEFRAME}) — Sorted by Sharpe Ratio")
print(f"{'=' * 80}")

display_cols = [
    "Strategy",
    "Total Trades",
    "Win Rate [%]",
    "Sharpe Ratio",
    "Profit Factor",
    "Max Drawdown [%]",
    "Return [%]",
    "Buy & Hold [%]",
]
display(
    ranked_df[display_cols]
    .style.format(
        {
            "Win Rate [%]": "{:.2f}%",
            "Sharpe Ratio": "{:.4f}",
            "Profit Factor": "{:.3f}",
            "Max Drawdown [%]": "{:.2f}%",
            "Return [%]": "{:.2f}%",
            "Buy & Hold [%]": "{:.2f}%",
        }
    )
    .background_gradient(subset=["Sharpe Ratio"], cmap="RdYlGn")
    .bar(subset=["Return [%]"], color="lightgreen", align="mid")
)


# %% [markdown]
# ## 6. Key Metrics — Bar Charts

# %%
def plot_metric_comparison(df, metric, title, higher_is_better=True):
    """Plot horizontal bar chart for a given metric."""
    sorted_df = df.sort_values(metric, ascending=higher_is_better).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(sorted_df)))
    bars = ax.barh(sorted_df["Strategy"], sorted_df[metric], color=colors)
    ax.set_title(f"{title} — Top 10", fontsize=13, fontweight="bold")
    ax.set_xlabel(metric)
    for i, (bar, val) in enumerate(zip(bars, sorted_df[metric])):
        fmt = "{:.2f}" if "Sharpe" not in metric and "Factor" not in metric else "{:.3f}"
        ax.text(
            bar.get_width() + bar.get_width() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(val),
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.show()


plot_metric_comparison(ranked_df, "Sharpe Ratio", "Sharpe Ratio")
plot_metric_comparison(ranked_df, "Return [%]", "Return (%)")
plot_metric_comparison(ranked_df, "Win Rate [%]", "Win Rate (%)")
plot_metric_comparison(ranked_df, "Max Drawdown [%]", "Max Drawdown (%)", higher_is_better=False)

# %% [markdown]
# ## 7. Equity Final vs Buy & Hold Comparison

# %%
fig, ax = plt.subplots(figsize=(12, 6))

# Sort by return
plot_df = ranked_df.sort_values("Return [%]", ascending=True).tail(15)
y_pos = range(len(plot_df))

# Strategy returns
strat_returns = plot_df["Return [%]"].values
bnh_returns = plot_df["Buy & Hold [%]"].values

bars1 = ax.barh(y_pos, strat_returns, height=0.4, label="Strategy Return", color="#2196F3")
bars2 = ax.barh(
    [y + 0.35 for y in y_pos],
    bnh_returns,
    height=0.4,
    label="Buy & Hold Return",
    color="#757575",
    alpha=0.6,
)

ax.set_yticks([y + 0.175 for y in y_pos])
ax.set_yticklabels(plot_df["Strategy"])
ax.set_xlabel("Return (%)")
ax.set_title("Strategy Return vs Buy & Hold Return", fontsize=13, fontweight="bold")
ax.axvline(x=0, color="black", linewidth=0.5)
ax.legend()

# Add value labels
for bar in bars1:
    width = bar.get_width()
    ax.text(
        width + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.1f}%",
        va="center",
        fontsize=8,
        fontweight="bold",
    )

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Composite Score & Final Ranking

# %%
# Filter strategies with at least 5 trades
valid_df = ranked_df[ranked_df["Total Trades"] >= 5].copy()

if len(valid_df) >= 2:
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()

    # Normalize metrics (0-1)
    valid_df["norm_sharpe"] = scaler.fit_transform(valid_df[["Sharpe Ratio"]])
    valid_df["norm_return"] = scaler.fit_transform(valid_df[["Return [%]"]])
    valid_df["norm_dd"] = 1 - scaler.fit_transform(
        valid_df[["Max Drawdown [%]"]]
    )  # Lower DD is better
    valid_df["norm_pf"] = scaler.fit_transform(valid_df[["Profit Factor"]])
    valid_df["norm_wr"] = scaler.fit_transform(valid_df[["Win Rate [%]"]])

    # Composite score: weighted combination
    valid_df["Composite_Score"] = (
        valid_df["norm_sharpe"] * 0.30
        + valid_df["norm_return"] * 0.25
        + valid_df["norm_dd"] * 0.20
        + valid_df["norm_pf"] * 0.15
        + valid_df["norm_wr"] * 0.10
    )

    valid_df = valid_df.sort_values("Composite_Score", ascending=False).reset_index(drop=True)
    valid_df.index += 1
    valid_df.index.name = "Final Rank"

    print(f"\n{'=' * 80}")
    print(f"Composite Ranking (>= 5 trades) — {TICKER} ({TIMEFRAME})")
    print("Weights: Sharpe 30%, Return 25%, Drawdown 20%, Profit Factor 15%, Win Rate 10%")
    print(f"{'=' * 80}")

    display_cols = [
        "Strategy",
        "Total Trades",
        "Win Rate [%]",
        "Sharpe Ratio",
        "Profit Factor",
        "Max Drawdown [%]",
        "Return [%]",
        "Composite_Score",
    ]
    display(
        valid_df[display_cols]
        .style.format(
            {
                "Win Rate [%]": "{:.2f}%",
                "Sharpe Ratio": "{:.4f}",
                "Profit Factor": "{:.3f}",
                "Max Drawdown [%]": "{:.2f}%",
                "Return [%]": "{:.2f}%",
                "Composite_Score": "{:.4f}",
            }
        )
        .background_gradient(subset=["Composite_Score"], cmap="RdYlGn")
    )
else:
    print(f"Only {len(valid_df)} strategy(s) with >= 5 trades. Composite ranking skipped.")
    valid_df = ranked_df.copy()

# %% [markdown]
# ## 9. Top 3 Strategies — Detailed View

# %%
df_for_top = valid_df if len(valid_df) >= 2 else ranked_df
top_strategies = df_for_top.head(3)["Strategy"].tolist()

for strat_name in top_strategies:
    print(f"\n{'=' * 60}")
    print(f"  {strat_name}")
    print(f"{'=' * 60}")

    row = ranked_df[ranked_df["Strategy"] == strat_name].iloc[0]
    print(f"Trades:       {int(row['Total Trades'])}")
    print(f"Win Rate:     {row['Win Rate [%]']:.2f}%")
    print(f"Sharpe:       {row['Sharpe Ratio']:.4f}")
    print(f"Profit Fact:  {row['Profit Factor']:.3f}")
    print(f"Max DD:       {row['Max Drawdown [%]']:.2f}%")
    print(f"Return:       {row['Return [%]']:.2f}%")
    print(f"B&H Return:   {row['Buy & Hold [%]']:.2f}%")
    print(f"Exposure:     {row['Exposure Time [%]']:.1f}%")
    alpha = row["Return [%]"] - row["Buy & Hold [%]"]
    print(f"Alpha vs B&H: {alpha:.2f}%")

    # Assessment
    trades = int(row["Total Trades"])
    sharpe = row["Sharpe Ratio"]
    pf = row["Profit Factor"]

    if trades < 10:
        verdict = f"[CAUTION] Low sample size ({trades} trades)"
    elif sharpe < 0.3:
        verdict = "[WEAK] Sharpe Ratio below 0.3"
    elif pf < 1.0:
        verdict = "[WEAK] Profit Factor below 1.0"
    else:
        verdict = f"[STRONG] Sharpe: {sharpe:.4f}, PF: {pf:.3f}, Trades: {trades}"
    print(f"Verdict:      {verdict}")

# %% [markdown]
# ## 10. Risk-Return Scatter Plot

# %%
fig, ax = plt.subplots(figsize=(12, 8))

for _, row in ranked_df.iterrows():
    size = max(30, min(300, row["Total Trades"] * 10))
    color = (
        "#228B22"
        if row["Sharpe Ratio"] >= 0.5
        else ("#FF8C00" if row["Sharpe Ratio"] >= 0 else "#DC143C")
    )
    ax.scatter(
        abs(row["Max Drawdown [%]"]),
        row["Return [%]"],
        s=size,
        c=[color],
        alpha=0.7,
        edgecolors="white",
        linewidth=0.5,
        label=row["Strategy"] if row["Strategy"] in top_strategies else None,
        zorder=3 if row["Strategy"] in top_strategies else 2,
    )

ax.set_title("Risk (Drawdown) vs Return", fontsize=14, fontweight="bold")
ax.set_xlabel("Max Drawdown (%) (absolute)")
ax.set_ylabel("Return (%)")
ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Add labels for top strategies
for _, row in ranked_df.iterrows():
    if row["Strategy"] in top_strategies:
        ax.annotate(
            row["Strategy"],
            (abs(row["Max Drawdown [%]"]), row["Return [%]"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
            fontweight="bold",
        )

plt.tight_layout()
plt.show()
print("Bubble size = number of trades | Green = Sharpe>=0.5 | Orange = Sharpe>=0 | Red = Sharpe<0")

# %% [markdown]
# ## 11. Save Results

# %%
out_dir = project_root / CONFIG["output"]["directory"]
os.makedirs(out_dir, exist_ok=True)

# Save full comparison table
output_csv = out_dir / f"{TICKER.lower()}_{TIMEFRAME}_strategy_comparison.csv"
ranked_df.to_csv(output_csv, index=False)
print(f"Comparison table saved: {output_csv}")

# Save composite ranking if computed
if len(valid_df) >= 2 and "Composite_Score" in valid_df.columns:
    composite_csv = out_dir / f"{TICKER.lower()}_{TIMEFRAME}_composite_ranking.csv"
    valid_df.to_csv(composite_csv, index=False)
    print(f"Composite ranking saved: {composite_csv}")

print(f"\nDone! All {len(STRATEGY_MAP)} strategies backtested on {TICKER} ({TIMEFRAME}).")
