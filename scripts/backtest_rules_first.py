"""
B3-B5: Backtest rules-first pattern strategy on SPY.

Usage:
    # IS backtest (2016-2024)
    uv run scripts/backtest_rules_first.py SPY --start 2016-01-01 --end 2024-12-31

    # OOS backtest (2025-2026)
    uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --end 2026-05-14

    # Sweep entry thresholds
    uv run scripts/backtest_rules_first.py SPY --sweep-entry 0.3,0.4,0.5,0.6,0.7

    # Sweep min reliability
    uv run scripts/backtest_rules_first.py SPY --sweep-reliability 0.3,0.4,0.5,0.6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.rules_first_strategy import RulesFirstStrategy


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
    entry_threshold: float = 0.55,
    exit_threshold: float = 0.30,
    trail_stop_atr: float = 3.0,
    min_reliability: float = 0.40,
    confluence_bonus: float = 0.10,
    volume_confirm: bool = True,
    use_ir_weights: bool = False,
    ir_weighting_window: int = 252,
    ir_weighting_mode: str = "scalar",
    use_short: bool = False,
    use_multi_tp: bool = True,
    use_quality_registry: bool = True,
    quality_registry_path: str = "reports/pattern_gate/all_patterns.json",
    use_multi_factor: bool = False,
    multi_factor_file: str = "",
    multi_factor_weight: float = 0.15,
    use_vix_gate: bool = False,
    vix_gate_stress_mult: float = 0.30,
    vix_gate_elevated_mult: float = 0.75,
    use_yield_curve_gate: bool = False,
    yield_curve_inversion_mult: float = 0.50,
    yield_curve_near_inversion_mult: float = 0.75,
    use_garch_atr: bool = False,
    garch_model: str = "GARCH",
    use_options_sentiment: bool = False,
    options_sentiment_weight: float = 0.3,
    use_kelly_sizing: bool = False,
    kelly_fraction: float = 0.5,
    use_vix_regime_sizing: bool = False,
    vix_regime_size_penalty: float = 0.50,
    vix_regime_high_vol_cap: float = 0.50,
    vix_regime_crisis_cap: float = 0.25,
    use_order_book: bool = False,
    use_signal_strength_sizing: bool = False,
    use_voting_signal: bool = False,
    voting_signal_weight: float = 0.15,
    use_rules_catalog: bool = False,
    rules_catalog_weight: float = 0.10,
    use_divergence: bool = False,
    divergence_weight: float = 0.20,
    use_wm_bollinger: bool = False,
    wm_bollinger_weight: float = 0.15,
) -> dict:
    from backtesting import Backtest

    df = load_data(symbol)
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]

    bt = Backtest(
        df,
        RulesFirstStrategy,
        cash=cash,
        commission=0.001,
        exclusive_orders=True,
        finalize_trades=True,
    )

    stats = bt.run(
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        trail_stop_atr=trail_stop_atr,
        min_reliability=min_reliability,
        confluence_bonus=confluence_bonus,
        volume_confirm=volume_confirm,
        use_ir_weights=use_ir_weights,
        ir_weighting_window=ir_weighting_window,
        ir_weighting_mode=ir_weighting_mode,
        use_short=use_short,
        use_multi_tp=use_multi_tp,
        use_multi_factor=use_multi_factor,
        multi_factor_file=multi_factor_file,
        multi_factor_weight=multi_factor_weight,
        use_quality_registry=use_quality_registry,
        quality_registry_path=quality_registry_path,
        use_vix_gate=use_vix_gate,
        vix_gate_stress_mult=vix_gate_stress_mult,
        vix_gate_elevated_mult=vix_gate_elevated_mult,
        use_yield_curve_gate=use_yield_curve_gate,
        yield_curve_inversion_mult=yield_curve_inversion_mult,
        yield_curve_near_inversion_mult=yield_curve_near_inversion_mult,
        use_garch_atr=use_garch_atr,
        garch_model=garch_model,
        use_options_sentiment=use_options_sentiment,
        options_sentiment_weight=options_sentiment_weight,
        use_kelly_sizing=use_kelly_sizing,
        kelly_fraction=kelly_fraction,
        use_vix_regime_sizing=use_vix_regime_sizing,
        vix_regime_size_penalty=vix_regime_size_penalty,
        vix_regime_high_vol_cap=vix_regime_high_vol_cap,
        vix_regime_crisis_cap=vix_regime_crisis_cap,
        use_order_book=use_order_book,
        use_signal_strength_sizing=use_signal_strength_sizing,
        use_voting_signal=use_voting_signal,
        voting_signal_weight=voting_signal_weight,
        use_rules_catalog=use_rules_catalog,
        rules_catalog_weight=rules_catalog_weight,
        use_divergence=use_divergence,
        divergence_weight=divergence_weight,
        use_wm_bollinger=use_wm_bollinger,
        wm_bollinger_weight=wm_bollinger_weight,
    )

    # Buy & Hold comparison
    if len(df) >= 2:
        bh_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
        daily_ret = df["Close"].pct_change().dropna()
        bh_sharpe = (
            float((daily_ret.mean() / daily_ret.std()) * (252**0.5)) if daily_ret.std() > 0 else 0.0
        )
        bh_max_dd = ((df["Close"] / df["Close"].cummax() - 1).min()) * 100
    else:
        bh_return = bh_sharpe = bh_max_dd = 0.0

    result = {
        "symbol": symbol,
        "start": str(df.index[0].date()) if len(df) > 0 else "N/A",
        "end": str(df.index[-1].date()) if len(df) > 0 else "N/A",
        "entry_threshold": entry_threshold,
        "min_reliability": min_reliability,
        "return_pct": stats["Return [%]"],
        "sharpe": stats["Sharpe Ratio"],
        "sortino": stats.get("Sortino Ratio", 0),
        "calmar": stats.get("Calmar Ratio", 0),
        "max_dd_pct": stats["Max. Drawdown [%]"],
        "trades": stats["# Trades"],
        "win_rate_pct": stats["Win Rate [%]"],
        "profit_factor": stats["Profit Factor"],
        "exposure_pct": stats["Exposure Time [%]"],
        "bh_return_pct": round(bh_return, 2),
        "bh_sharpe": round(bh_sharpe, 3),
        "bh_max_dd_pct": round(bh_max_dd, 2),
    }
    return result


def run_backtest(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    entry_threshold: float = 0.55,
    min_reliability: float = 0.40,
    **kwargs: object,
) -> dict | None:
    """Wrapper around run_single for audit scripts (audit_btc_config)."""
    try:
        return run_single(
            symbol,
            start=start,
            end=end,
            entry_threshold=entry_threshold,
            min_reliability=min_reliability,
            **{k: v for k, v in kwargs.items() if v is not None},
        )
    except Exception as e:
        import logging

        logging.warning("run_backtest(%s) failed: %s", symbol, e)
        return None


def sweep_entry(
    symbol: str,
    thresholds: list[float],
    start: str,
    end: str,
    **kwargs,
) -> list[dict]:
    results = []
    for et in thresholds:
        r = run_single(symbol, start=start, end=end, entry_threshold=et, **kwargs)
        r["entry_threshold"] = et
        results.append(r)
        print(
            f"  et={et:.1f}: Sharpe {r['sharpe']:.3f}, Return {r['return_pct']:.1f}%, "
            f"Trades {r['trades']}, Win {r['win_rate_pct']:.1f}%, PF {r['profit_factor']:.2f}"
            f"  |  B&H Return {r['bh_return_pct']:.1f}%"
        )
    return results


def sweep_reliability(
    symbol: str,
    thresholds: list[float],
    start: str,
    end: str,
    **kwargs,
) -> list[dict]:
    results = []
    for mr in thresholds:
        r = run_single(symbol, start=start, end=end, min_reliability=mr, **kwargs)
        r["min_reliability"] = mr
        results.append(r)
        print(
            f"  mr={mr:.1f}: Sharpe {r['sharpe']:.3f}, Return {r['return_pct']:.1f}%, "
            f"Trades {r['trades']}, Win {r['win_rate_pct']:.1f}%, PF {r['profit_factor']:.2f}"
            f"  |  B&H Return {r['bh_return_pct']:.1f}%"
        )
    return results


def print_table(results: list[dict]) -> None:
    cols = [
        "entry_threshold",
        "min_reliability",
        "return_pct",
        "sharpe",
        "sortino",
        "calmar",
        "max_dd_pct",
        "trades",
        "win_rate_pct",
        "profit_factor",
        "exposure_pct",
        "bh_return_pct",
        "bh_sharpe",
        "bh_max_dd_pct",
    ]
    header = [c.replace("_pct", "%") for c in cols]
    widths = [max(len(h), 8) for h in header]
    for r in results:
        for i, c in enumerate(cols):
            if c in r and r[c] is not None:
                widths[i] = max(widths[i], len(f"{r[c]:.2f}"))

    fmt = "  ".join(f"{{:>{w}}}" for w in widths)
    print(fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for r in results:
        vals = []
        for c in cols:
            v = r.get(c)
            if v is None:
                vals.append("N/A")
            elif isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        print(fmt.format(*vals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest rules-first pattern strategy")
    parser.add_argument("symbol", help="Ticker symbol (e.g., SPY)")
    parser.add_argument("--start", default="2016-01-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date")
    parser.add_argument("--cash", type=float, default=10_000, help="Initial cash")
    parser.add_argument(
        "--entry-threshold", type=float, default=0.55, help="Entry signal threshold"
    )
    parser.add_argument("--exit-threshold", type=float, default=0.30, help="Exit signal threshold")
    parser.add_argument(
        "--trail-stop-atr", type=float, default=3.0, help="ATR multiplier for trail"
    )
    parser.add_argument(
        "--min-reliability", type=float, default=0.40, help="Minimum pattern reliability"
    )
    parser.add_argument("--confluence-bonus", type=float, default=0.10, help="Confluence bonus")
    parser.add_argument(
        "--no-volume-confirm", action="store_true", help="Disable volume confirmation"
    )
    parser.add_argument(
        "--ir-weights",
        action="store_true",
        help="Use rolling IR weights instead of static reliability",
    )
    parser.add_argument(
        "--ir-weighting-window", type=int, default=252, help="IR rolling window size"
    )
    parser.add_argument(
        "--ir-weighting-mode",
        default="scalar",
        choices=["scalar", "gate"],
        help="IR weighting mode: scalar (0.3x-0.7x) or gate (zero out negative IR)",
    )
    parser.add_argument(
        "--use-short",
        action="store_true",
        help="Enable short side (bearish patterns trigger shorts)",
    )
    parser.add_argument(
        "--use-multi-tp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable multi-TP exit (partial TP at 1.5x ATR)",
    )
    parser.add_argument(
        "--use-multi-factor",
        action="store_true",
        help="Use multi-factor fundamental score as signal modifier",
    )
    parser.add_argument(
        "--multi-factor-file", default="", help="Path to precomputed fundamental score CSV"
    )
    parser.add_argument(
        "--multi-factor-weight", type=float, default=0.15, help="Weight of multi-factor modifier"
    )
    parser.add_argument(
        "--no-quality-registry",
        action="store_true",
        help="Disable pattern quality registry weight modulation",
    )
    parser.add_argument(
        "--quality-registry-path",
        default="reports/pattern_gate/all_patterns.json",
        help="Path to pattern gate sweep JSON",
    )
    parser.add_argument(
        "--use-vix-gate",
        action="store_true",
        help="Q1: Enable VIX regime gate (stress/elevated VIX scales signals)",
    )
    parser.add_argument(
        "--vix-stress-mult", type=float, default=0.30, help="Multiplier when VIX > 30 (stress)"
    )
    parser.add_argument(
        "--vix-elevated-mult", type=float, default=0.75, help="Multiplier when VIX 25-30 (elevated)"
    )
    parser.add_argument(
        "--use-yield-curve-gate",
        action="store_true",
        help="Q2: Enable yield curve inversion gate (inversion scales signals)",
    )
    parser.add_argument(
        "--yield-inversion-mult",
        type=float,
        default=0.50,
        help="Multiplier when 2s10s < 0 (inverted)",
    )
    parser.add_argument(
        "--yield-near-inversion-mult",
        type=float,
        default=0.75,
        help="Multiplier when 0 < 2s10s < 0.5%%",
    )
    # RF3.1: GARCH dynamic ATR trail
    parser.add_argument(
        "--use-garch-atr", action="store_true", help="RF3.1: Use GARCH vol for dynamic trail width"
    )
    parser.add_argument(
        "--garch-model", type=str, default="egarch", help="GARCH model (garch/egarch/gjr-garch)"
    )
    # RF3.2: Options sentiment
    parser.add_argument(
        "--use-options-sentiment",
        action="store_true",
        help="RF3.2: Use PC ratio + GEX sentiment modifier",
    )
    parser.add_argument(
        "--options-sentiment-weight",
        type=float,
        default=0.10,
        help="Sentiment modifier weight [0-1]",
    )
    # RF3.3: Kelly sizing
    parser.add_argument(
        "--use-kelly-sizing", action="store_true", help="RF3.3: Use Kelly dynamic position sizing"
    )
    parser.add_argument(
        "--kelly-fraction", type=float, default=0.5, help="Kelly fraction (0.5=half-Kelly)"
    )
    # P1.5: VIX regime-adaptive position sizing
    parser.add_argument(
        "--use-vix-regime-sizing",
        action="store_true",
        help="P1.5: Cap position size by VIX regime (50%% in HIGH_VOL, 25%% in CRISIS)",
    )
    parser.add_argument(
        "--vix-size-high-vol-cap",
        type=float,
        default=0.50,
        help="Max position size in ELEVATED VIX regime [default: 0.50]",
    )
    parser.add_argument(
        "--vix-size-crisis-cap",
        type=float,
        default=0.25,
        help="Max position size in STRESS VIX regime [default: 0.25]",
    )
    # RF3.4: Order book
    parser.add_argument(
        "--use-order-book", action="store_true", help="RF3.4: Use order book microstructure signals"
    )
    parser.add_argument(
        "--use-signal-strength-sizing",
        action="store_true",
        help="P24-17: Signal-strength dynamic position sizing (|score|>=0.45→3 lots, <0.15→2 lots, <0.05→1)",
    )
    parser.add_argument(
        "--use-voting-signal",
        action="store_true",
        help="B6: Enable 6-indicator voting signal (RSI/ROC/SMA/EMA/WMA/MACD majority voting)",
    )
    parser.add_argument(
        "--voting-signal-weight",
        type=float,
        default=0.15,
        help="B6: Weight of voting signal in confluence scoring (default 0.15)",
    )
    parser.add_argument(
        "--use-rules-catalog",
        action="store_true",
        help="B1: Enable 35-rule catalog signals (22 crossover + 6 BB + 7 divergence)",
    )
    parser.add_argument(
        "--rules-catalog-weight",
        type=float,
        default=0.10,
        help="B1: Weight of rules catalog signals (default 0.10)",
    )
    parser.add_argument(
        "--use-divergence",
        action="store_true",
        help="B10: Enable RSI/MFI divergence detection signals",
    )
    parser.add_argument(
        "--divergence-weight",
        type=float,
        default=0.20,
        help="B10: Weight of divergence signals (default 0.20)",
    )
    parser.add_argument(
        "--use-wm-bollinger",
        action="store_true",
        help="B2: Enable W-bottom/M-top Bollinger pattern signals",
    )
    parser.add_argument(
        "--wm-bollinger-weight",
        type=float,
        default=0.15,
        help="B2: Weight of W/M Bollinger signals (default 0.15)",
    )
    parser.add_argument("--sweep-entry", help="Comma-separated entry thresholds to sweep")
    parser.add_argument(
        "--sweep-reliability", help="Comma-separated min reliability values to sweep"
    )
    parser.add_argument("--json-output", help="Path to save JSON results")
    args = parser.parse_args()

    kwargs = dict(
        exit_threshold=args.exit_threshold,
        trail_stop_atr=args.trail_stop_atr,
        confluence_bonus=args.confluence_bonus,
        volume_confirm=not args.no_volume_confirm,
        use_ir_weights=args.ir_weights,
        ir_weighting_window=args.ir_weighting_window,
        ir_weighting_mode=args.ir_weighting_mode,
        use_short=args.use_short,
        use_multi_tp=args.use_multi_tp,
        use_multi_factor=args.use_multi_factor,
        multi_factor_file=args.multi_factor_file,
        multi_factor_weight=args.multi_factor_weight,
        use_quality_registry=not args.no_quality_registry,
        quality_registry_path=args.quality_registry_path,
        use_vix_gate=args.use_vix_gate,
        vix_gate_stress_mult=args.vix_stress_mult,
        vix_gate_elevated_mult=args.vix_elevated_mult,
        use_yield_curve_gate=args.use_yield_curve_gate,
        yield_curve_inversion_mult=args.yield_inversion_mult,
        yield_curve_near_inversion_mult=args.yield_near_inversion_mult,
        use_garch_atr=args.use_garch_atr,
        garch_model=args.garch_model,
        use_options_sentiment=args.use_options_sentiment,
        options_sentiment_weight=args.options_sentiment_weight,
        use_kelly_sizing=args.use_kelly_sizing,
        kelly_fraction=args.kelly_fraction,
        use_vix_regime_sizing=args.use_vix_regime_sizing,
        vix_regime_high_vol_cap=args.vix_size_high_vol_cap,
        vix_regime_crisis_cap=args.vix_size_crisis_cap,
        use_order_book=args.use_order_book,
        use_signal_strength_sizing=args.use_signal_strength_sizing,
        use_voting_signal=args.use_voting_signal,
        voting_signal_weight=args.voting_signal_weight,
        use_rules_catalog=args.use_rules_catalog,
        rules_catalog_weight=args.rules_catalog_weight,
        use_divergence=args.use_divergence,
        divergence_weight=args.divergence_weight,
        use_wm_bollinger=args.use_wm_bollinger,
        wm_bollinger_weight=args.wm_bollinger_weight,
    )
    all_results: list[dict] = []

    if args.sweep_entry:
        thresholds = [float(t.strip()) for t in args.sweep_entry.split(",")]
        print(f"\nSweeping entry thresholds: {thresholds}")
        print(f"Symbol: {args.symbol}  Period: {args.start} -> {args.end}\n")
        results = sweep_entry(
            args.symbol,
            thresholds,
            args.start,
            args.end,
            min_reliability=args.min_reliability,
            **kwargs,
        )
        all_results = results
        print()
        print_table(results)
    elif args.sweep_reliability:
        thresholds = [float(t.strip()) for t in args.sweep_reliability.split(",")]
        print(f"\nSweeping min reliability: {thresholds}")
        print(f"Symbol: {args.symbol}  Period: {args.start} -> {args.end}\n")
        results = sweep_reliability(args.symbol, thresholds, args.start, args.end, **kwargs)
        all_results = results
        print()
        print_table(results)
    else:
        result = run_single(
            args.symbol,
            cash=args.cash,
            start=args.start,
            end=args.end,
            entry_threshold=args.entry_threshold,
            min_reliability=args.min_reliability,
            **kwargs,
        )
        all_results = [result]
        print()
        print_table([result])

        print(
            f"\nBuy & Hold:  Return={result['bh_return_pct']:.1f}%  "
            f"Sharpe={result['bh_sharpe']:.3f}  MaxDD={result['bh_max_dd_pct']:.1f}%"
        )

        print(
            f"\nConfig: et={args.entry_threshold} exit_et={args.exit_threshold} "
            f"trail={args.trail_stop_atr} mr={args.min_reliability} "
            f"confl={args.confluence_bonus} vol={not args.no_volume_confirm} "
            f"ir={args.ir_weights} short={args.use_short} multi_tp={args.use_multi_tp}"
        )

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nResults saved to {args.json_output}")


if __name__ == "__main__":
    main()
