"""Sweep entry thresholds for ML Strategy backtest and record results."""

import sys
import json
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from backtesting import Backtest
from src.strategies.ml_strategy import MLStrategy


MODEL_PATH = "models/pattern_classifier_v3_SPY_20260511_224704.pkl"
SYMBOL = "SPY"
START = "2016-05-12"
CASH = 100_000


def load_data() -> pd.DataFrame:
    path = Path(f"data/raw/{SYMBOL}_daily.csv")
    df = pd.read_csv(path, parse_dates=True, index_col=0).dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0 if col == "Volume" else df.iloc[:, 0]
    df.columns = [c.capitalize() for c in df.columns]
    if START:
        df = df[df.index >= START]
    return df


def run_config(entry_threshold, use_trail, use_conviction, vol_gate=None):
    df = load_data()
    exit_threshold = round(entry_threshold * 0.7, 2)

    bt = Backtest(df, MLStrategy, cash=CASH, commission=0.001, exclusive_orders=True)
    stats = bt.run(
        model_path=MODEL_PATH,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        tp_atr_mult=3.0,
        sl_atr_mult=1.5,
        risk_pct=0.02,
        vol_gate_threshold=vol_gate,
        confirm_bars=1,
        use_trail_stop=use_trail,
        trail_atr_mult=3.0,
        conviction_scale=use_conviction,
        use_pattern_boost=False,
    )

    return {
        "entry_threshold": entry_threshold,
        "trail_stop": use_trail,
        "conviction": use_conviction,
        "vol_gate": vol_gate,
        "return_pct": round(stats["Return [%]"], 2),
        "buy_hold_return": round(stats["Buy & Hold Return [%]"], 2),
        "return_annual": round(stats["Return (Ann.) [%]"], 2),
        "sharpe": round(stats["Sharpe Ratio"], 2),
        "sortino": round(stats["Sortino Ratio"], 2),
        "calmar": round(stats["Calmar Ratio"], 2),
        "max_drawdown": round(stats["Max. Drawdown [%]"], 2),
        "win_rate": round(stats["Win Rate [%]"], 2),
        "num_trades": stats["# Trades"],
        "profit_factor": round(stats["Profit Factor"], 2),
        "exposure": round(stats["Exposure Time [%]"], 2),
        "avg_trade": round(stats["Avg. Trade [%]"], 2),
    }


def main():
    thresholds = [0.35, 0.40, 0.45]
    all_results = []

    print(
        f"{'Config':<35} {'Return%':>8} {'Sharpe':>7} {'Trades':>7} {'Win%':>7} {'PF':>7} {'MaxDD%':>8} {'Expo%':>7}"
    )
    print("-" * 90)

    for et in thresholds:
        # Trail + conviction
        for trail, conv, label in [
            (True, True, f"et={et} trail+conv"),
            (True, False, f"et={et} trail"),
            (False, False, f"et={et} baseline"),
        ]:
            print(f"  Running {label}...", end=" ", flush=True)
            r = run_config(et, trail, conv)
            all_results.append(r)
            print(
                f"Ret={r['return_pct']:7.1f}% Sh={r['sharpe']:5.2f} Tr={r['num_trades']:4d} "
                f"WR={r['win_rate']:5.1f}% PF={r['profit_factor']:5.2f} DD={r['max_drawdown']:6.1f}% "
                f"Exp={r['exposure']:5.1f}%"
            )

    # Also try vol_gate combos on best-looking threshold
    best = max(all_results, key=lambda x: x["sharpe"])
    best_et = best["entry_threshold"]
    for vg in [1.3, 1.5, 1.8]:
        label = f"et={best_et} trail+conv vg={vg}"
        print(f"  Running {label}...", end=" ", flush=True)
        r = run_config(best_et, True, True, vol_gate=vg)
        all_results.append(r)
        print(
            f"Ret={r['return_pct']:7.1f}% Sh={r['sharpe']:5.2f} Tr={r['num_trades']:4d} "
            f"WR={r['win_rate']:5.1f}% PF={r['profit_factor']:5.2f} DD={r['max_drawdown']:6.1f}% "
            f"Exp={r['exposure']:5.1f}%"
        )

    # Save
    out_dir = Path("reports/sweeps")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"entry_threshold_sweep_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print sorted summary
    print("\n" + "=" * 90)
    print("SORTED BY SHARPE")
    print("=" * 90)
    for r in sorted(all_results, key=lambda x: x["sharpe"], reverse=True):
        label = f"et={r['entry_threshold']}"
        if r["trail_stop"]:
            label += " trail"
        if r["conviction"]:
            label += " conv"
        if r["vol_gate"]:
            label += f" vg={r['vol_gate']}"
        print(
            f"  {label:<30} Ret={r['return_pct']:7.1f}% Sh={r['sharpe']:5.2f} "
            f"Tr={r['num_trades']:4d} WR={r['win_rate']:5.1f}% PF={r['profit_factor']:5.2f} "
            f"DD={r['max_drawdown']:5.1f}% Ann={r['return_annual']:5.1f}% Exp={r['exposure']:5.1f}%"
        )

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
