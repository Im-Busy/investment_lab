# %% [markdown]
# # Strategy Benchmark Ranking
#
# Tests every implemented strategy on every available ticker, ranks them by return %, and produces a clean leaderboard.

# %%
import sys
from pathlib import Path
import pandas as pd
from backtesting import Backtest

project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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

DATA_DIR = project_root / "data" / "raw"

STRATEGIES = {
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


# %%
def load_data(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    for col in ["Datetime", "date", "Date", "timestamp"]:
        if col in df.columns:
            df = df.set_index(col)
            df.index = pd.to_datetime(df.index)
            break
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
    return df[["Open", "High", "Low", "Close", "Volume"]]


# %%
files = sorted(DATA_DIR.glob("*.csv"))
print(
    f"Testing {len(STRATEGIES)} strategies on {len(files)} assets = {len(STRATEGIES) * len(files)} backtests\n"
)

results = []
total = len(STRATEGIES) * len(files)
count = 0

for f in files:
    try:
        df = load_data(f)
    except Exception as e:
        print(f"SKIP {f.name}: {e}")
        continue
    if len(df) < 200:
        print(f"SKIP {f.name}: only {len(df)} bars")
        continue

    for name, cls in STRATEGIES.items():
        count += 1
        try:
            bt = Backtest(df, cls, cash=1_000_000, commission=0.001, exclusive_orders=True)
            stats = bt.run()
            results.append(
                {
                    "Asset": f.name,
                    "Strategy": name,
                    "Trades": int(stats.get("# Trades", 0)),
                    "WR": float(stats.get("Win Rate [%]", 0)),
                    "Sharpe": float(stats.get("Sharpe Ratio", -999)),
                    "PF": float(stats.get("Profit Factor", 0)),
                    "Return": float(stats.get("Return [%]", 0)),
                    "BnH": float(stats.get("Buy & Hold Return [%]", 0)),
                    "MaxDD": float(stats.get("Max. Drawdown [%]", 0)),
                }
            )
        except Exception as e:
            print(f"ERROR {f.name} x {name}: {e}")

df_results = pd.DataFrame(results)
if len(df_results) > 0 and "Return" in df_results.columns:
    df_results = df_results.sort_values("Return", ascending=False).reset_index(drop=True)
print(f"Done: {len(df_results)} results")

# %%
# Top 10 by Return
df_results.head(10).style.format(
    {
        "WR": "{:.1f}%",
        "Sharpe": "{:.2f}",
        "PF": "{:.2f}",
        "Return": "{:.1f}%",
        "BnH": "{:.1f}%",
        "MaxDD": "{:.1f}%",
    }
)

# %%
if len(df_results) > 0 and "Sharpe" in df_results.columns:
    passing = df_results[(df_results["Sharpe"] >= 0.3) & (df_results["PF"] >= 1.0)]
    print(f"Passing: {len(passing)} of {len(df_results)} combos\n")
    passing.style.format(
        {
            "WR": "{:.1f}%",
            "Sharpe": "{:.2f}",
            "PF": "{:.2f}",
            "Return": "{:.1f}%",
            "BnH": "{:.1f}%",
            "MaxDD": "{:.1f}%",
        }
    )
else:
    print("No results to display - skipping")

# %%
if len(df_results) > 0 and "Sharpe" in df_results.columns:
    out_path = project_root / "reports" / "benchmark_ranking.csv"
    df_results.to_csv(out_path, index=False)
    print(f"Saved to: {out_path}")
else:
    print("No results to save")
