"""Quick sweep: test all 3 strategies with default params on all 10 instruments."""

import json
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from backtesting import Backtest
from src.strategies.silver_bullet import SilverBulletStrategy
from src.strategies.turtle_soup import TurtleSoupStrategy
from src.strategies.cameron_model import CameronModelStrategy

IS_END, OOS_START = "2024-09-30", "2024-10-01"
INSTRUMENTS = [
    "EURUSD=X",
    "USDJPY=X",
    "GBPUSD=X",
    "AUDUSD=X",
    "USDCAD=X",
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "XRP-USD",
    "DOGE-USD",
]

DEFAULTS = {
    "SilverBullet": (
        SilverBulletStrategy,
        dict(kill_zone="london_open", trail_stop_atr=2.0, sweep_lookback=12, fvg_min_gap=0.3),
    ),
    "TurtleSoup": (
        TurtleSoupStrategy,
        dict(session_bars=24, breakout_buffer_atr=0.3, reversal_confirm_bars=2, trail_stop_atr=2.0),
    ),
    "CameronModel": (
        CameronModelStrategy,
        dict(swing_lookback=50, sweep_buffer_atr=0.3, trail_stop_atr=2.0),
    ),
}


def fetch(t):
    df = yf.download(t, period="730d", interval="1h", progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    return df if {"Open", "High", "Low", "Close"}.issubset(set(df.columns)) else None


print("=" * 80)
print("  ICT Strategy Baseline — defaults on 10 forex+crypto instruments (IS/OOS)")
print("=" * 80)

all_results = {}
for name, (strat_cls, params) in DEFAULTS.items():
    print(f"\n--- {name} ---")
    strategy_results = {}
    for sym in INSTRUMENTS:
        df = fetch(sym)
        if df is None:
            continue
        df = df.sort_index()
        is_ts = pd.Timestamp(IS_END).tz_localize("UTC")
        oos_ts = pd.Timestamp(OOS_START).tz_localize("UTC")
        is_df = df[df.index <= is_ts]
        oos_df = df[df.index >= oos_ts]
        if len(is_df) < 100 or len(oos_df) < 50:
            continue

        bh = round(float((oos_df["Close"].iloc[-1] / oos_df["Close"].iloc[0] - 1) * 100), 1)

        try:
            bt = Backtest(is_df, strat_cls, cash=100000, commission=0.001)
            stats_is = bt.run(**params)
            bt_oos = Backtest(oos_df, strat_cls, cash=100000, commission=0.001)
            stats_oos = bt_oos.run(**params)
            is_ret = round(float(stats_is.get("Return [%]", 0)), 1)
            is_s = round(float(stats_is.get("Sharpe Ratio", 0) or 0), 2)
            is_t = int(stats_is.get("# Trades", 0))
            oos_ret = round(float(stats_oos.get("Return [%]", 0)), 1)
            oos_s = round(float(stats_oos.get("Sharpe Ratio", 0) or 0), 2)
            oos_t = int(stats_oos.get("# Trades", 0))
            oos_w = round(float(stats_oos.get("Win Rate [%]", 0) or 0), 1)
            oos_pf = round(float(stats_oos.get("Profit Factor", 0) or 0), 2)
            oos_dd = round(float(stats_oos.get("Max. Drawdown [%]", 0) or 0), 1)
            print(
                f"  {sym:12s} | IS: {is_ret:+6.1f}% S={is_s:5.2f} T={is_t:3d} | "
                f"OOS: {oos_ret:+6.1f}% S={oos_s:5.2f} T={oos_t:3d} W={oos_w:4.1f}% PF={oos_pf:4.2f} DD={oos_dd:5.1f}% | "
                f"BH={bh:+6.1f}%"
            )
            strategy_results[sym] = {
                "is_ret": is_ret,
                "is_sharpe": is_s,
                "is_trades": is_t,
                "oos_ret": oos_ret,
                "oos_sharpe": oos_s,
                "oos_trades": oos_t,
                "oos_win": oos_w,
                "oos_pf": oos_pf,
                "oos_dd": oos_dd,
                "bh": bh,
            }
        except Exception as e:
            print(f"  {sym:12s} | FAIL: {e}")

    # Summary
    valid = [r for r in strategy_results.values() if r["oos_trades"] >= 5]
    if valid:
        avg_sharpe = np.mean([r["oos_sharpe"] for r in valid])
        avg_pf = np.mean([r["oos_pf"] for r in valid])
        avg_win = np.mean([r["oos_win"] for r in valid])
        total_trades = sum(r["oos_trades"] for r in valid)
        print(
            f"  SUMMARY ({len(valid)}/{len(strategy_results)} valid): "
            f"avg OOS Sharpe={avg_sharpe:.2f} avg PF={avg_pf:.2f} avg Win={avg_win:.1f}% total trades={total_trades}"
        )
    all_results[name] = strategy_results

# Save
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)
path = out_dir / "ict_baseline.json"
with open(path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved: {path}")
