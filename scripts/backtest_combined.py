"""
AB2: Backtest combined ML + Rules-First strategy.

Usage:
    # Static blend
    uv run scripts/backtest_combined.py SPY --start 2016-01-01 --end 2024-12-31 --weight-mode static --rules-weight 0.5

    # Regime-adaptive (recommended)
    uv run scripts/backtest_combined.py SPY --start 2025-01-01 --end 2026-05-14 --weight-mode regime-adaptive

    # Sweep rules weight
    uv run scripts/backtest_combined.py SPY --weight-mode static --sweep-weight 0.3,0.5,0.7

    # OOS test all modes
    uv run scripts/backtest_combined.py SPY --start 2025-01-01 --weight-mode sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.combined_strategy import CombinedStrategy


def load_data(symbol: str) -> pd.DataFrame:
    path = Path(f"data/raw/{symbol}_daily.csv")
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol}")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df.dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0 if col == "Volume" else df.iloc[:, 0]
    df.columns = [c.capitalize() for c in df.columns]
    return df


def run_single(
    symbol: str,
    cash: float = 10_000,
    start: str | None = None,
    end: str | None = None,
    weight_mode: str = "regime-adaptive",
    rules_weight: float = 0.5,
    rules_weight_min: float = 0.3,
    rules_weight_max: float = 0.8,
    entry_threshold: float = 0.55,
    exit_threshold: float = 0.30,
    trail_stop_atr: float = 3.0,
    min_reliability: float = 0.70,
    confluence_bonus: float = 0.10,
    ml_model_path: str = "models/pattern_classifier_v3_SPY_20260514_195235.pkl",
    use_regime_router: bool = True,
    regime_router_config: str = "models/regime_router_SPY.json",
    use_vix_gate: bool = False,
    use_yield_curve_gate: bool = False,
    use_multi_tp: bool = True,
    tp1_atr: float = 1.5,
    tp1_size: float = 0.5,
    volume_confirm: bool = True,
    use_fuzzy: bool = False,
    fuzzy_weight: float = 0.30,
    use_svm_regime: bool = False,
    svm_regime_window: int = 100,
    svm_regime_down_skip: bool = True,
) -> dict:
    from backtesting import Backtest

    df = load_data(symbol)
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]

    bt = Backtest(
        df,
        CombinedStrategy,
        cash=cash,
        commission=0.001,
        exclusive_orders=True,
        finalize_trades=True,
    )

    stats = bt.run(
        ml_model_path=ml_model_path,
        use_regime_router=use_regime_router,
        regime_router_config=regime_router_config,
        weight_mode=weight_mode,
        rules_weight=rules_weight,
        rules_weight_min=rules_weight_min,
        rules_weight_max=rules_weight_max,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        trail_stop_atr=trail_stop_atr,
        min_reliability=min_reliability,
        confluence_bonus=confluence_bonus,
        use_vix_gate=use_vix_gate,
        use_yield_curve_gate=use_yield_curve_gate,
        use_multi_tp=use_multi_tp,
        tp1_atr=tp1_atr,
        tp1_size=tp1_size,
        volume_confirm=volume_confirm,
        use_fuzzy=use_fuzzy,
        fuzzy_weight=fuzzy_weight,
        use_svm_regime=use_svm_regime,
        svm_regime_window=svm_regime_window,
        svm_regime_down_skip=svm_regime_down_skip,
    )

    result = {
        "symbol": symbol,
        "start": str(df.index[0].date()) if len(df) > 0 else "N/A",
        "end": str(df.index[-1].date()) if len(df) > 0 else "N/A",
        "weight_mode": weight_mode,
        "rules_weight": rules_weight,
        "return_pct": stats["Return [%]"],
        "sharpe": stats["Sharpe Ratio"],
        "sortino": stats.get("Sortino Ratio", 0),
        "calmar": stats.get("Calmar Ratio", 0),
        "max_dd_pct": stats["Max. Drawdown [%]"],
        "trades": stats["# Trades"],
        "win_rate_pct": stats["Win Rate [%]"],
        "profit_factor": stats["Profit Factor"],
        "exposure_pct": stats["Exposure Time [%]"],
    }
    return result


def sweep_weights(
    symbol: str,
    weights: list[float],
    start: str,
    end: str,
    **kwargs,
) -> list[dict]:
    results = []
    for w in weights:
        r = run_single(symbol, start=start, end=end, rules_weight=w, weight_mode="static", **kwargs)
        results.append(r)
        print(
            f"  w_rules={w:.1f}: Sharpe {r['sharpe']:.3f}, "
            f"Return {r['return_pct']:.1f}%, Trades {r['trades']}, "
            f"Win {r['win_rate_pct']:.1f}%, PF {r['profit_factor']:.2f}"
        )
    return results


def sweep_modes(
    symbol: str,
    start: str,
    end: str,
    **kwargs,
) -> list[dict]:
    results = []
    sweep_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ("weight_mode", "rules_weight", "rules_weight_min", "rules_weight_max")
    }
    modes = [
        ("static", 0.5, 0.5, 0.5),
        ("static", 0.7, 0.7, 0.7),
        ("regime-adaptive", 0.5, 0.3, 0.8),
        ("reciprocal-sharpe", 0.5, 0.3, 0.8),
        ("signal-conflict", 0.5, 0.3, 0.8),
    ]
    for mode, w, wmin, wmax in modes:
        r = run_single(
            symbol,
            start=start,
            end=end,
            weight_mode=mode,
            rules_weight=w,
            rules_weight_min=wmin,
            rules_weight_max=wmax,
            **sweep_kwargs,
        )
        results.append(r)
        print(
            f"  {mode}(w={w:.1f}): Sharpe {r['sharpe']:.3f}, "
            f"Return {r['return_pct']:.1f}%, Trades {r['trades']}, "
            f"Win {r['win_rate_pct']:.1f}%, PF {r['profit_factor']:.2f}"
        )
    return results


def print_table(results: list[dict]) -> None:
    cols = [
        "weight_mode",
        "rules_weight",
        "return_pct",
        "sharpe",
        "sortino",
        "calmar",
        "max_dd_pct",
        "trades",
        "win_rate_pct",
        "profit_factor",
        "exposure_pct",
    ]
    header = [c.replace("_pct", "%") for c in cols]
    widths = [max(len(h), 10) for h in header]
    for r in results:
        for i, c in enumerate(cols):
            v = r.get(c)
            if v is not None:
                widths[i] = max(widths[i], len(f"{v}"))

    fmt = "  ".join(f"{{:>{w}}}" for w in widths)
    print(fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for r in results:
        vals = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v) if v is not None else "N/A")
        print(fmt.format(*vals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest combined ML + Rules-First strategy")
    parser.add_argument("symbol", help="Ticker symbol (e.g., SPY)")
    parser.add_argument("--start", default="2016-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument("--cash", type=float, default=10_000, help="Initial cash")
    parser.add_argument(
        "--weight-mode",
        default="regime-adaptive",
        choices=["static", "regime-adaptive", "reciprocal-sharpe", "signal-conflict", "sweep"],
        help="Signal blending mode (default: regime-adaptive)",
    )
    parser.add_argument(
        "--rules-weight", type=float, default=0.5, help="Static rules weight (0-1, default 0.5)"
    )
    parser.add_argument(
        "--rules-weight-min", type=float, default=0.3, help="Minimum rules weight in adaptive modes"
    )
    parser.add_argument(
        "--rules-weight-max", type=float, default=0.8, help="Maximum rules weight in adaptive modes"
    )
    parser.add_argument(
        "--entry-threshold", type=float, default=0.55, help="Entry signal threshold"
    )
    parser.add_argument("--exit-threshold", type=float, default=0.30, help="Exit signal threshold")
    parser.add_argument(
        "--trail-stop-atr", type=float, default=3.0, help="ATR multiplier for trailing stop"
    )
    parser.add_argument(
        "--min-reliability", type=float, default=0.70, help="Minimum pattern reliability"
    )
    parser.add_argument("--confluence-bonus", type=float, default=0.10, help="Confluence bonus")
    parser.add_argument(
        "--ml-model-path",
        default="models/pattern_classifier_v3_SPY_20260514_195235.pkl",
        help="Path to ML model .pkl",
    )
    parser.add_argument(
        "--no-regime-router",
        action="store_true",
        help="Use single ML model instead of RegimeRouter",
    )
    parser.add_argument(
        "--regime-router-config",
        default="models/regime_router_SPY.json",
        help="Path to RegimeRouter JSON config",
    )
    parser.add_argument("--sweep-weight", help="Comma-separated weights to sweep")
    parser.add_argument("--no-vix-gate", action="store_true", help="Disable VIX regime gate")
    parser.add_argument(
        "--no-yield-curve-gate", action="store_true", help="Disable yield curve macro gate"
    )
    parser.add_argument("--no-multi-tp", action="store_true", help="Disable multi-TP exit")
    parser.add_argument("--tp1-atr", type=float, default=1.5, help="TP1 ATR distance")
    parser.add_argument("--tp1-size", type=float, default=0.5, help="Portion to close at TP1")
    parser.add_argument(
        "--no-volume-confirm", action="store_true", help="Disable volume confirmation"
    )
    parser.add_argument("--use-fuzzy", action="store_true", help="Enable fuzzy logic scoring")
    parser.add_argument(
        "--fuzzy-weight", type=float, default=0.30, help="Fuzzy score blend weight (0-1)"
    )
    parser.add_argument(
        "--use-svm-regime", action="store_true", help="Enable SVM regime classification"
    )
    parser.add_argument(
        "--svm-regime-window", type=int, default=100, help="SVM regime lookback window"
    )
    parser.add_argument(
        "--no-svm-down-skip",
        action="store_true",
        help="Don't skip entries when SVM predicts down regime",
    )
    parser.add_argument("--json-output", help="Path to save JSON results")
    args = parser.parse_args()

    kwargs = dict(
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        trail_stop_atr=args.trail_stop_atr,
        min_reliability=args.min_reliability,
        confluence_bonus=args.confluence_bonus,
        ml_model_path=args.ml_model_path,
        use_regime_router=not args.no_regime_router,
        regime_router_config=args.regime_router_config,
        rules_weight_min=args.rules_weight_min,
        rules_weight_max=args.rules_weight_max,
        use_vix_gate=not args.no_vix_gate,
        use_yield_curve_gate=not args.no_yield_curve_gate,
        use_multi_tp=not args.no_multi_tp,
        tp1_atr=args.tp1_atr,
        tp1_size=args.tp1_size,
        volume_confirm=not args.no_volume_confirm,
        use_fuzzy=args.use_fuzzy,
        fuzzy_weight=args.fuzzy_weight,
        use_svm_regime=args.use_svm_regime,
        svm_regime_window=args.svm_regime_window,
        svm_regime_down_skip=not args.no_svm_down_skip,
    )

    all_results: list[dict] = []

    if args.weight_mode == "sweep":
        print("\nSweeping all weight modes")
        print(f"Symbol: {args.symbol}  Period: {args.start} -> {args.end}\n")
        all_results = sweep_modes(args.symbol, args.start, args.end, **kwargs)
    elif args.sweep_weight:
        weights = [float(w.strip()) for w in args.sweep_weight.split(",")]
        print(f"\nSweeping rules weight: {weights}")
        print(f"Symbol: {args.symbol}  Period: {args.start} -> {args.end}\n")
        all_results = sweep_weights(args.symbol, weights, args.start, args.end, **kwargs)
    else:
        result = run_single(
            args.symbol,
            cash=args.cash,
            start=args.start,
            end=args.end,
            weight_mode=args.weight_mode,
            rules_weight=args.rules_weight,
            **kwargs,
        )
        all_results = [result]

    print()
    print_table(all_results)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nResults saved to {args.json_output}")


if __name__ == "__main__":
    main()
