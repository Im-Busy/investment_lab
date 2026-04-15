"""
Quick backtest: 5 remaining strategies + optimize 4 passing patterns.
"""

import sys
import json
import time
import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backtesting import Backtest
from src.strategies.backtest_py.multi_pattern_strategy_optimized import MultiPatternStrategyOptimized


def sf(v, d=0):
    try:
        x = float(v); return d if (math.isnan(x) or math.isinf(x)) else x
    except:
        return d


# ============================
# PART 1: Test 5 Remaining Strategies on BTC 1H
# ============================

STRATEGY_IMPORTS = [
    ("Ichimoku Cloud",       "src.strategies.ichimoku_cloud",          "IchimokuCloudStrategy"),
    ("RSI Divergence",       "src.strategies.rsi_divergence",          "RSIDivergenceStrategy"),
    ("MFI Strategy",         "src.strategies.mfi_strategy",            "MFIStrategy"),
    ("Williams %R",          "src.strategies.williams_r_reversal",     "WilliamsRReversalStrategy"),
    ("MACD Histogram",       "src.strategies.macd_histogram",          "MACDHistogramStrategy"),
]


def test_strategies():
    print("\n" + "=" * 100)
    print("PART 1: Backtest 5 Remaining Strategies on BTC 1H")
    print("=" * 100)
    df = pd.read_csv(project_root / "data" / "raw" / "BTC_USD_1h.csv",
                     index_col="Datetime", parse_dates=True)
    if "Volume" not in df.columns:
        df["Volume"] = 0
    print(f"Data: {len(df)} bars\n")

    strat_results = []
    for name, mod_path, cls_name in STRATEGY_IMPORTS:
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            bt = Backtest(df, cls, cash=100000, commission=0.001, exclusive_orders=True)
            stats = bt.run()
            r = {"name": name, "ok": True,
                 "trades": int(stats.get("# Trades", 0) or 0),
                 "wr": sf(stats.get("Win Rate [%]", 0)),
                 "sharpe": sf(stats.get("Sharpe Ratio", -999), -999),
                 "pf": sf(stats.get("Profit Factor", 0)),
                 "return_pct": sf(stats.get("Return [%]", 0)),
                 "dd_pct": sf(stats.get("Max. Drawdown [%]", 0))}
            strat_results.append(r)
            print(f"  {name:<25} t={r['trades']:>5} sh={r['sharpe']:>7.2f} wr={r['wr']:>6.1f}% ret={r['return_pct']:>7.1f}%")
        except Exception as e:
            print(f"  {name:<25} ERROR: {str(e)[:80]}")
            strat_results.append({"name": name, "ok": False, "trades": 0})
    return strat_results


# ============================
# PART 2: Optimize 4 Passing Patterns on SPY and QQQ
# ============================

def load_daily(name):
    df = pd.read_csv(project_root / f"data/raw/{name}_daily.csv", index_col=0, parse_dates=True)
    c = {x.lower(): x for x in df.columns}
    for lo, st in [("open","Open"),("high","High"),("low","Low"),("close","Close"),("volume","Volume")]:
        if st not in df.columns and lo in c:
            df.rename(columns={c[lo]: st}, inplace=True)
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df


def optimize_pattern(aname, df, pat):
    print(f"\n  {aname}: {pat}... ", end="", flush=True)
    try:
        bt = Backtest(df, MultiPatternStrategyOptimized, cash=1000000, commission=0.001, exclusive_orders=True)
        opt = bt.optimize(
            include_patterns_only=pat,
            min_confidence=list(np.arange(0.50, 0.81, 0.05)),
            risk_per_trade=[0.01, 0.02, 0.03, 0.05],
            max_open_positions=[1, 3, 5],
            maximize="Return [%]",
            max_tries=100,
        )
        if opt is not None and len(opt) > 0:
            best = opt.iloc[0]
            trades = int(best.get("# Trades", 0))
            shr = float(best.get("Sharpe Ratio", 0) or 0)
            pf = float(best.get("Profit Factor", 0) or 0)
            ret = float(best.get("Return [%]", 0) or 0)
            wr = float(best.get("Win Rate [%]", 0) or 0)
            dd = float(best.get("Max. Drawdown [%]", 0) or 0)
            conf = best.get("min_confidence", 0.6)
            risk = best.get("risk_per_trade", 0.02)
            pos = best.get("max_open_positions", 1)
            print(f"sh={shr:.2f} pf={pf:.2f} ret={ret:.1f}% wr={wr:.1f}% dd={dd:.1f}% t={trades}")
            return {"pat": pat, "asset": aname,
                    "conf": float(conf), "risk": float(risk), "pos": int(pos),
                    "trades": trades, "sharpe": shr, "pf": pf, "return_pct": ret, "wr": wr, "dd_pct": dd}
        else:
            print("no results")
            return {"pat": pat, "asset": aname, "error": "no results"}
    except Exception as e:
        print(f"ERROR: {str(e)[:80]}")
        return {"pat": pat, "asset": aname, "error": str(e)[:100]}


def optimize_patterns(df_spy, df_qqq):
    print("\n" + "=" * 100)
    print("PART 2: Optimize 4 Passing Patterns on SPY & QQQ")
    print("=" * 100)
    passing = ["Triple Top", "Symmetric Triangle", "Donchian Channel Breakout", "Gap Pattern"]

    all_opt = []
    for aname, df in [("SPY", df_spy), ("QQQ", df_qqq)]:
        for pat in passing:
            t0 = time.time()
            r = optimize_pattern(aname, df, pat)
            r["time_s"] = round(time.time() - t0, 1)
            all_opt.append(r)
    return all_opt


# ============================
# MAIN
# ============================

if __name__ == "__main__":
    strat_results = test_strategies()

    df_spy = load_daily("SPY")
    df_qqq = load_daily("QQQ")
    opt_results = optimize_patterns(df_spy, df_qqq)

    od = project_root / "reports" / "untested_patterns_backtest"
    od.mkdir(parents=True, exist_ok=True)

    with open(od / "remaining_5_strategies.json", "w") as f:
        json.dump(strat_results, f, indent=2, default=str)
    with open(od / "optimization_results.json", "w") as f:
        json.dump(opt_results, f, indent=2, default=str)

    with open(od / "summary.md", "w") as f:
        f.write("# 23 Untested Patterns Backtest + Strategy Tests\n\n")
        f.write("## Pass 1 Result\n\n4 passed: Triple Top, Symmetric Triangle, Donchian Channel Breakout, Gap Pattern\n\n")
        f.write("## 5 Remaining Strategies (BTC 1H)\n\n")
        f.write("| Strategy | Trades | WR% | Sharpe | Return% |\n")
        f.write("|----------|--------|-----|--------|----------|\n")
        for r in strat_results:
            if r.get("ok"):
                f.write(f"| {r['name']} | {r['trades']} | {r['wr']:.1f} | {r['sharpe']:.2f} | {r['return_pct']:.1f} |\n")
        f.write("\n## Optimization (SPY & QQQ)\n\n")
        f.write("| Asset | Pattern | Sharpe | PF | Return% | DD% |\n|-------|---------|--------|----|----------|-----|\n")
        for r in opt_results:
            if "error" not in r:
                f.write(f"| {r['asset']} | {r['pat']} | {r['sharpe']:.2f} | {r['pf']:.2f} | {r['return_pct']:.1f} | {r['dd_pct']:.1f} |\n")

    print(f"\nSaved to {od}")
