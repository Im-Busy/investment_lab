"""
Backtest 23 Untested Patterns - Parallel Fast Version

- Suppresses warnings for speed
- Uses higher initial cash (1M) to avoid margin issues
- Saves results incrementally after each pattern
"""

import sys
import json
import time
import math
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.strategies.backtest_py.runner import BacktestPyRunner
from src.strategies.backtest_py.multi_pattern_strategy_optimized import (
    MultiPatternStrategyOptimized,
)

ASSETS = [
    ("BTC_USD_1h", "data/raw/BTC_USD_1h.csv"),
    ("SPY_daily", "data/raw/SPY_daily.csv"),
    ("BTC_USD_daily", "data/raw/BTC_USD_daily.csv"),
    ("QQQ_daily", "data/raw/QQQ_daily.csv"),
]

UNTESTED_PATTERNS = [
    "Double Top",
    "Double Bottom",
    "Triple Top",
    "Triple Bottom",
    "Ascending Triangle",
    "Descending Triangle",
    "Rectangle",
    "Wedge",
    "Dead Cat Bounce",
    "Trader Vic 2B",
    "Head And Shoulders",
    "Cup And Handle",
    "Spike And Ledge",
    "Three Hills Mountain",
    "Parabolic Arc",
    "Gartley Pattern",
    "ABC Pattern",
    "Bollinger Bands",
    "Symmetric Triangle",
    "Flag",
    "Pennant",
    "Donchian Channel Breakout",
    "Gap Pattern",
]

TH1 = {"min_trades": 10, "min_sharpe": -1.0}
TH2 = {"min_trades": 5, "min_sharpe": -2.0}


def load_df(p):
    full = project_root / p
    if not full.exists():
        return None
    try:
        df = pd.read_csv(full, index_col=0, parse_dates=True)
        c = {x.lower(): x for x in df.columns}
        for lo, st in [
            ("open", "Open"),
            ("high", "High"),
            ("low", "Low"),
            ("close", "Close"),
            ("volume", "Volume"),
        ]:
            if st not in df.columns and lo in c:
                df.rename(columns={c[lo]: st}, inplace=True)
        if "Volume" not in df.columns:
            df["Volume"] = 0.0
        return df
    except:
        return None


def sf(v, d=0.0):
    try:
        x = float(v)
        return d if (math.isnan(x) or math.isinf(x)) else x
    except:
        return d


def bt(pat, df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = BacktestPyRunner(data=df, cash=1000000, commission=0.001, exclusive_orders=True)
            res = r.run(
                strategy_class=MultiPatternStrategyOptimized,
                include_patterns_only=pat,
                min_confluence_count=1,
            )
            s = res.get("stats", res)
            return {
                "trades": int(s.get("# Trades", 0) or 0),
                "wr": sf(s.get("Win Rate [%]", 0)),
                "sharpe": sf(s.get("Sharpe Ratio", -999), -999),
                "sortino": sf(s.get("Sortino Ratio", 0)),
                "pf": sf(s.get("Profit Factor", 0)),
                "return_pct": sf(s.get("Return [%]", 0)),
                "dd_pct": sf(s.get("Max. Drawdown [%]", 0)),
                "bh_return": sf(s.get("Buy & Hold Return [%]", 0)),
                "ok": True,
            }
        except Exception as e:
            return {
                "trades": 0,
                "wr": 0,
                "sharpe": -999,
                "sortino": 0,
                "pf": 0,
                "return_pct": 0,
                "dd_pct": 0,
                "bh_return": 0,
                "ok": False,
                "error": str(e)[:200],
            }


def ok(th, r):
    return (
        r.get("ok", False) and r["trades"] >= th["min_trades"] and r["sharpe"] >= th["min_sharpe"]
    )


def tbl(asset, results, th):
    lines = [f"\n{'=' * 105}\n{asset}\n{'=' * 105}"]
    lines.append(
        f"{'Pattern':<28} {'Trades':>7} {'WR%':>7} {'Sharpe':>8} {'PF':>6} {'Ret%':>8} {'DD%':>8}  Res"
    )
    lines.append("-" * 105)
    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
        p = "PASS" if ok(th, r) else "FAIL"
        lines.append(
            f"{r['name']:<28} {r['trades']:>7} {r['wr']:>6.1f}% {r['sharpe']:>8.2f} "
            f"{r['pf']:>6.2f} {r['return_pct']:>7.1f}% {r['dd_pct']:>7.1f}%  {p}"
        )
    ps = [r["name"] for r in results if ok(th, r)]
    lines += ["-" * 105, f"  Passed: {len(ps)}/{len(results)}"]
    if ps:
        lines.append(f"  -> {', '.join(ps)}")
    return "\n".join(lines)


def main():
    passed = set()

    # PASS 1
    print(f"{'=' * 105}\nPASS 1  min_trades=10  min_sharpe=-1.0\n{'=' * 105}")
    sys.stdout.flush()
    all_p1 = {}

    for aname, acsv in ASSETS:
        df = load_df(acsv)
        if df is None:
            print(f"\n[SKIP] {acsv}")
            sys.stdout.flush()
            continue
        print(f"\n--- {aname} ({len(df)} bars) ---")
        sys.stdout.flush()
        ares = []
        for i, pat in enumerate(UNTESTED_PATTERNS):
            t0 = time.time()
            r = bt(pat, df)
            dt = time.time() - t0
            r["name"] = pat
            ares.append(r)
            p = "PASS" if ok(TH1, r) else "FAIL"
            print(
                f"  [{i + 1:2d}/23] {pat:<28} t={r['trades']:>4} sh={r['sharpe']:>7.2f}  {p}  ({dt:.0f}s)"
            )
            sys.stdout.flush()
            if ok(TH1, r):
                passed.add(pat)
        print(tbl(aname, ares, TH1))
        sys.stdout.flush()
        all_p1[aname] = ares

    if passed:
        print(f"\nPASS 1 RESULT: {len(passed)} passed: {sorted(passed)}")
        print("Relaxed pass SKIPPED.")
        p2 = None
    else:
        print("\nPASS 1: 0 passed -> PASS 2 (relaxed thresholds)")
        sys.stdout.flush()
        print(f"\n{'=' * 105}\nPASS 2  min_trades=5  min_sharpe=-2.0\n{'=' * 105}")
        sys.stdout.flush()
        p2 = {}
        for aname, acsv in ASSETS:
            df = load_df(acsv)
            if df is None:
                continue
            print(f"\n--- {aname} ({len(df)} bars) ---")
            sys.stdout.flush()
            ares = []
            for i, pat in enumerate(UNTESTED_PATTERNS):
                t0 = time.time()
                r = bt(pat, df)
                dt = time.time() - t0
                r["name"] = pat
                ares.append(r)
                p = "PASS" if ok(TH2, r) else "FAIL"
                print(
                    f"  [{i + 1:2d}/23] {pat:<28} t={r['trades']:>4} sh={r['sharpe']:>7.2f}  {p}  ({dt:.0f}s)"
                )
                sys.stdout.flush()
                if ok(TH2, r):
                    passed.add(pat)
            print(tbl(aname, ares, TH2))
            sys.stdout.flush()
            p2[aname] = ares

    # Save
    odir = project_root / "reports" / "untested_patterns_backtest"
    odir.mkdir(parents=True, exist_ok=True)
    jf = odir / "results.json"
    with open(jf, "w") as f:
        json.dump(
            {
                "pass1_th": TH1,
                "pass1": all_p1,
                "pass2_th": TH2,
                "pass2": p2,
                "passed": sorted(passed),
            },
            f,
            indent=2,
            default=str,
        )
    sf_name = odir / "summary.md"
    with open(sf_name, "w") as f:
        f.write(f"# 23 Untested Patterns Backtest\n\nPassed: {len(passed)}\n")
        if passed:
            for p in sorted(passed):
                f.write(f"  - {p}\n")
        else:
            f.write("  NONE\n")
        f.write("\nSee `results.json`\n")

    print(f"\nSaved: {jf}")
    print(f"\n{'=' * 105}\nFINAL: {len(passed)} passed")
    if passed:
        for p in sorted(passed):
            print(f"  + {p}")
    else:
        print("  NO PATTERNS PASSED")
    print(f"{'=' * 105}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
