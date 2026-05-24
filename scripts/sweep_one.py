"""ICT strategy sweep using backtesting.py built-in optimize() for speed."""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
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

IS_END = "2024-09-30"
OOS_START = "2024-10-01"
CASH = 100_000
COMMISSION = 0.001

INSTRUMENTS = [
    ("EURUSD=X", "forex"),
    ("USDJPY=X", "forex"),
    ("GBPUSD=X", "forex"),
    ("AUDUSD=X", "forex"),
    ("USDCAD=X", "forex"),
    ("BTC-USD", "crypto"),
    ("ETH-USD", "crypto"),
    ("SOL-USD", "crypto"),
    ("XRP-USD", "crypto"),
    ("DOGE-USD", "crypto"),
]

STRATEGIES = {
    "SilverBullet": (
        SilverBulletStrategy,
        {
            "kill_zone": ["london_open", "new_york_am", "london_close"],
            "trail_stop_atr": [1.5, 2.5],
            "sweep_lookback": [8, 12],
            "fvg_min_gap": [0.2, 0.3],
        },
    ),
    "TurtleSoup": (
        TurtleSoupStrategy,
        {
            "session_bars": [12, 24],
            "breakout_buffer_atr": [0.2, 0.3],
            "reversal_confirm_bars": [2, 3],
            "trail_stop_atr": [1.5, 2.5],
        },
    ),
    "CameronModel": (
        CameronModelStrategy,
        {
            "swing_lookback": [50, 70],
            "sweep_buffer_atr": [0.2, 0.3],
            "trail_stop_atr": [1.5, 2.0, 2.5],
        },
    ),
}


def fetch(ticker: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period="730d", interval="1h", progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        if not {"Open", "High", "Low", "Close"}.issubset(set(df.columns)):
            return None
        return df
    except:
        return None


def extract_stats(stats) -> dict:
    return {
        "ret": round(float(stats.get("Return [%]", 0)), 1),
        "sharpe": round(float(stats.get("Sharpe Ratio", 0) or 0), 2),
        "trades": int(stats.get("# Trades", 0)),
        "win_rate": round(float(stats.get("Win Rate [%]", 0) or 0), 1),
        "pf": round(float(stats.get("Profit Factor", 0) or 0), 2),
        "max_dd": round(float(stats.get("Max. Drawdown [%]", 0) or 0), 1),
        "exp": round(float(stats.get("Exposure Time [%]", 0) or 0), 1),
    }


def sweep_strategy(name: str) -> Dict:
    strategy_cls, param_grid = STRATEGIES[name]
    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)

    print(f"\n{'=' * 60}")
    print(f"  {name} — {n_combos} combos × {len(INSTRUMENTS)} instruments (optimize + validate)")
    print(f"{'=' * 60}")

    results = {}
    t0 = time.time()

    for sym, cat in INSTRUMENTS:
        df = fetch(sym)
        if df is None:
            continue

        df_sorted = df.sort_index()
        is_ts = pd.Timestamp(IS_END).tz_localize("UTC")
        oos_ts = pd.Timestamp(OOS_START).tz_localize("UTC")
        is_df = df_sorted[df_sorted.index <= is_ts]
        oos_df = df_sorted[df_sorted.index >= oos_ts]

        if len(is_df) < 100 or len(oos_df) < 50:
            continue

        bh_is = round(float((is_df["Close"].iloc[-1] / is_df["Close"].iloc[0] - 1) * 100), 1)
        bh_oos = round(float((oos_df["Close"].iloc[-1] / oos_df["Close"].iloc[0] - 1) * 100), 1)

        # Use backtesting.py optimize() for grid search on IS
        try:
            bt = Backtest(is_df, strategy_cls, cash=CASH, commission=COMMISSION)
            best_result = bt.optimize(
                maximize="Sharpe Ratio",
                method="grid",
                constraint=lambda p: True,
                **param_grid,
            )
            # Parse params from _strategy string: "StrategyName(key=val, ...)"
            strat_str = best_result.get("_strategy", "")
            best_params = {}
            if "(" in strat_str and ")" in strat_str:
                args_str = strat_str[strat_str.index("(") + 1 : strat_str.rindex(")")]
                for part in args_str.split(","):
                    part = part.strip()
                    if "=" in part:
                        k, v = part.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        try:
                            if "." in v:
                                v = float(v)
                            else:
                                v = int(v)
                        except (ValueError, TypeError):
                            pass
                        best_params[k] = v
        except Exception as e:
            print(f"  {sym:12s} | OPTIMIZE FAILED: {e}")
            continue

        is_stats = extract_stats(best_result)

        # Validate best params on OOS
        try:
            bt_oos = Backtest(oos_df, strategy_cls, cash=CASH, commission=COMMISSION)
            oos_result = bt_oos.run(**best_params)
            oos_stats = extract_stats(oos_result)
        except Exception as e:
            print(f"  {sym:12s} | OOS FAILED: {e}")
            continue

        if oos_stats["trades"] < 3:
            continue

        print(
            f"  {sym:10s} [{cat:6s}] | IS: {is_stats['ret']:+6.1f}% S={is_stats['sharpe']:5.2f} T={is_stats['trades']:3d} | "
            f"OOS: {oos_stats['ret']:+6.1f}% S={oos_stats['sharpe']:5.2f} T={oos_stats['trades']:3d} "
            f"W={oos_stats['win_rate']:4.1f}% PF={oos_stats['pf']:4.2f} | "
            f"B&H OOS={bh_oos:+6.1f}% | params={json.dumps(best_params)}"
        )

        results[sym] = {
            "cat": cat,
            "best_params": best_params,
            "is": is_stats,
            "oos": oos_stats,
            "bh_is": bh_is,
            "bh_oos": bh_oos,
        }

    elapsed = time.time() - t0

    # Find universal best params (most common best params with best avg OOS sharpe)
    param_counts: Dict[str, List] = {}
    for sym, info in results.items():
        pk = json.dumps(info["best_params"], sort_keys=True)
        param_counts.setdefault(pk, []).append(info["oos"]["sharpe"])

    best_pk = None
    best_score = -999
    for pk, sharpes in param_counts.items():
        n = len(sharpes)
        avg = np.mean(sharpes)
        score = avg * np.sqrt(n)
        if score > best_score and n >= 3:
            best_score = score
            best_pk = pk

    universal = {
        "params": json.loads(best_pk) if best_pk else {},
        "stats": {
            "n": len(param_counts.get(best_pk or "[]", [])),
            "avg_sharpe": round(np.mean(param_counts.get(best_pk or "[]", [0])), 3)
            if best_pk
            else 0,
        },
    }

    print(
        f"\n  ✓ {name}: {len(results)} instruments, universal params = {universal['params']}, "
        f"avg OOS Sharpe = {universal['stats'].get('avg_sharpe', 0)}, "
        f"time={elapsed:.0f}s"
    )

    return {"strategy": name, "results": results, "universal": universal, "n_tested": len(results)}


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "SilverBullet"
    out = sweep_strategy(name)
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"sweep_{name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")
