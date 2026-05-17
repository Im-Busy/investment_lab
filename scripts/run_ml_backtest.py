"""
Run ML strategy backtest on one or more tickers using backtesting.py.

Usage:
    uv run scripts/run_ml_backtest.py SPY
    uv run scripts/run_ml_backtest.py SPY --model models/pattern_classifier_v3_SPY_20260511_164601.pkl
    uv run scripts/run_ml_backtest.py SPY,QQQ,XLK,D,SO --compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.ml_strategy import MLStrategy


def load_data(symbol: str) -> pd.DataFrame:
    path = Path(f"data/raw/{symbol}_daily.csv")
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol}")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df.dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0 if col == "Volume" else df.iloc[:, 0]
    # backtesting.py expects specific column name capitalization
    df.columns = [c.capitalize() for c in df.columns]
    return df


def run_single(
    symbol: str,
    model_path: str,
    cash: float,
    start: str | None,
    end: str | None = None,
    entry_threshold: float = 0.50,
    vol_gate: float | None = None,
    confirm: int = 1,
    trail_stop: bool = False,
    trail_atr: float = 3.0,
    conviction: bool = False,
    multi_tp: bool = True,
    use_dynamic_ensemble: bool = False,
    dynamic_ensemble_path: str = "models/dynamic_ensemble_SPY",
    use_meta_label: bool = False,
    meta_label_path: str = "models/meta_labeler_v2_SPY.pkl",
    use_sentiment: bool = False,
    sentiment_weight: float = 0.15,
    use_lm_sentiment: bool = False,
    lm_sentiment_weight: float = 0.15,
    use_event_filter: bool = False,
    use_rl_execution: bool = False,
    rl_model_path: str = "models/rl_executor_dqn.pt",
    use_kelly: bool = False,
    kelly_fraction: float = 0.5,
    use_crash_filter: bool = False,
    crash_threshold: float = 0.50,
    use_defensive: bool = False,
    use_time_reversal: bool = False,
    use_chronos: bool = False,
    chronos_weight: float = 0.30,
    use_cross_asset: bool = False,
    use_regime_router: bool = False,
    regime_router_config: str = "models/regime_router_SPY.json",
    use_fundamentals: bool = False,
) -> dict:
    from backtesting import Backtest

    df = load_data(symbol)
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]

    bt = Backtest(
        df,
        MLStrategy,
        cash=cash,
        commission=0.001,
        exclusive_orders=True,
    )

    stats = bt.run(
        model_path=model_path,
        ticker=symbol,
        entry_threshold=entry_threshold,
        exit_threshold=entry_threshold * 0.7,
        tp_atr_mult=3.0,
        sl_atr_mult=1.5,
        risk_pct=0.02,
        vol_gate_threshold=vol_gate,
        confirm_bars=confirm,
        use_trail_stop=trail_stop,
        trail_atr_mult=trail_atr,
        conviction_scale=conviction,
        use_multi_tp=multi_tp,
        use_dynamic_ensemble=use_dynamic_ensemble,
        dynamic_ensemble_path=dynamic_ensemble_path,
        use_meta_label=use_meta_label,
        meta_label_path=meta_label_path,
        use_sentiment=use_sentiment,
        sentiment_weight=sentiment_weight,
        use_lm_sentiment=use_lm_sentiment,
        lm_sentiment_weight=lm_sentiment_weight,
        use_event_filter=use_event_filter,
        use_rl_execution=use_rl_execution,
        rl_model_path=rl_model_path,
        use_kelly=use_kelly,
        kelly_fraction=kelly_fraction,
        use_crash_filter=use_crash_filter,
        crash_threshold=crash_threshold,
        use_chronos=use_chronos,
        chronos_weight=chronos_weight,
        use_cross_asset=use_cross_asset,
        use_regime_router=use_regime_router,
        regime_router_config=regime_router_config,
        use_fundamentals=use_fundamentals,
    )
    result = {
        "symbol": symbol,
        "start": str(stats["Start"]),
        "end": str(stats["End"]),
        "duration": stats["Duration"],
        "exposure": stats["Exposure Time [%]"],
        "equity_final": round(stats["Equity Final [$]"], 2),
        "equity_peak": round(stats["Equity Peak [$]"], 2),
        "return_pct": round(stats["Return [%]"], 2),
        "buy_hold_return": round(stats["Buy & Hold Return [%]"], 2),
        "return_annual": round(stats["Return (Ann.) [%]"], 2),
        "volatility_annual": round(stats["Volatility (Ann.) [%]"], 2),
        "sharpe": round(stats["Sharpe Ratio"], 2),
        "sortino": round(stats["Sortino Ratio"], 2),
        "calmar": round(stats["Calmar Ratio"], 2),
        "max_drawdown": round(stats["Max. Drawdown [%]"], 2),
        "avg_drawdown": round(stats["Avg. Drawdown [%]"], 2),
        "win_rate": round(stats["Win Rate [%]"], 2),
        "best_trade": round(stats["Best Trade [%]"], 2),
        "worst_trade": round(stats["Worst Trade [%]"], 2),
        "avg_trade": round(stats["Avg. Trade [%]"], 2),
        "num_trades": stats["# Trades"],
        "profit_factor": round(stats["Profit Factor"], 2),
        "expectancy": round(stats["Expectancy [%]"], 2),
    }

    if use_defensive or use_time_reversal:
        rev_df = df.iloc[::-1].copy()
        rev_df.index = df.index
        bt_rev = Backtest(
            rev_df,
            MLStrategy,
            cash=cash,
            commission=0.001,
            exclusive_orders=True,
        )
        rev_stats = bt_rev.run(
            model_path=model_path,
            ticker=symbol,
            entry_threshold=entry_threshold,
            exit_threshold=entry_threshold * 0.7,
            tp_atr_mult=3.0,
            sl_atr_mult=1.5,
            risk_pct=0.02,
            vol_gate_threshold=vol_gate,
            confirm_bars=confirm,
            use_trail_stop=trail_stop,
            trail_atr_mult=trail_atr,
            conviction_scale=conviction,
            use_multi_tp=multi_tp,
            use_dynamic_ensemble=use_dynamic_ensemble,
            dynamic_ensemble_path=dynamic_ensemble_path,
            use_meta_label=use_meta_label,
            meta_label_path=meta_label_path,
            use_sentiment=use_sentiment,
            sentiment_weight=sentiment_weight,
            use_lm_sentiment=use_lm_sentiment,
            lm_sentiment_weight=lm_sentiment_weight,
            use_event_filter=use_event_filter,
            use_rl_execution=use_rl_execution,
            rl_model_path=rl_model_path,
            use_kelly=use_kelly,
            kelly_fraction=kelly_fraction,
            use_crash_filter=use_crash_filter,
            crash_threshold=crash_threshold,
            use_chronos=use_chronos,
            chronos_weight=chronos_weight,
            use_cross_asset=use_cross_asset,
        )
        fwd_ret = result["return_pct"]
        rev_ret = rev_stats["Return [%]"]
        result["defensive"] = {
            "reversed_return_pct": round(rev_ret, 2),
            "reversed_sharpe": round(rev_stats["Sharpe Ratio"], 2),
            "reversed_trades": rev_stats["# Trades"],
            "time_reversal_suspicious": fwd_ret > 0 and rev_ret > 0,
            "interpretation": (
                f"TIME-REVERSAL FLAG: profitable on both forward ({fwd_ret:.1f}%) "
                f"and reversed ({rev_ret:.1f}%) — overfit likely."
                if fwd_ret > 0 and rev_ret > 0
                else f"Time-reversal PASS: forward {fwd_ret:.1f}%, "
                f"reversed {rev_ret:.1f}% — directional edge confirmed."
            ),
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ML strategy backtest")
    parser.add_argument(
        "symbols",
        type=str,
        help="Comma-separated ticker symbols (e.g. SPY,QQQ,XLK)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/pattern_classifier_v3_SPY_20260514_195235.pkl",
        help="Path to trained model pickle",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=100_000,
        help="Initial capital",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Default: use all available data.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Default: use all available data.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run all tickers and print comparison table",
    )
    # ── C7 Refinement parameters ──
    parser.add_argument(
        "--vol-gate",
        type=float,
        default=None,
        help="Volatility gate threshold (e.g. 1.5). Max vol_regime for entry.",
    )
    parser.add_argument(
        "--confirm",
        type=int,
        default=1,
        help="Consecutive bars above entry_threshold required (default 1=off).",
    )
    parser.add_argument(
        "--trail-stop",
        action="store_true",
        help="Use ATR trailing stop instead of fixed take-profit.",
    )
    parser.add_argument(
        "--trail-atr",
        type=float,
        default=3.0,
        help="ATR multiplier for trailing stop distance (default 3.0).",
    )
    parser.add_argument(
        "--conviction",
        action="store_true",
        help="Scale position size by conviction (prob above entry_threshold).",
    )
    parser.add_argument(
        "--multi-tp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use multi-TP exit: partial TP at 50% of target with breakeven SL.",
    )
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=0.50,
        help="Minimum ML probability to enter (default 0.50). Lower = more trades.",
    )
    parser.add_argument(
        "--use-dynamic-ensemble",
        action="store_true",
        help="Use DynamicEnsemble for regime-adaptive prediction (B12).",
    )
    parser.add_argument(
        "--dynamic-ensemble-path",
        type=str,
        default="models/dynamic_ensemble_SPY",
        help="Path to saved DynamicEnsemble directory.",
    )
    parser.add_argument(
        "--use-meta-label",
        action="store_true",
        help="Enable MetaLabelerV2 secondary filter (B13). Requires trained meta-labeler model.",
    )
    parser.add_argument(
        "--meta-label-path",
        type=str,
        default="models/meta_labeler_v2_SPY_20260514_125515.pkl",
        help="Path to trained MetaLabelerV2 model.",
    )
    parser.add_argument(
        "--use-sentiment",
        action="store_true",
        help="Enable sentiment score adjustment to ML probabilities (C2).",
    )
    parser.add_argument(
        "--sentiment-weight",
        type=float,
        default=0.15,
        help="Sentiment influence weight (0-1, default 0.15).",
    )
    parser.add_argument(
        "--use-lm-sentiment",
        action="store_true",
        help="Use Loughran-McDonald financial sentiment dictionary for signal adjustment (Phase N).",
    )
    parser.add_argument(
        "--lm-sentiment-weight",
        type=float,
        default=0.15,
        help="LM sentiment influence weight (0-1, default 0.15).",
    )
    parser.add_argument(
        "--use-event-filter",
        action="store_true",
        help="Suppress entry signals on high-impact event days (FOMC, CPI, NFP, etc.).",
    )
    parser.add_argument(
        "--use-rl-execution",
        action="store_true",
        help="Use RL-trained agent for entry/exit decisions instead of fixed thresholds (C4).",
    )
    parser.add_argument(
        "--rl-model-path",
        type=str,
        default="models/rl_executor_dqn.pt",
        help="Path to trained RL executor model (default: models/rl_executor_dqn.pt).",
    )
    # ── C5 Kelly Allocator + Crash Filter ──
    parser.add_argument(
        "--use-kelly",
        action="store_true",
        help="Use Kelly criterion for position sizing instead of fixed risk_pct (C5).",
    )
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.5,
        help="Fraction of full Kelly to apply (0.25=quarter, 0.5=half, default 0.5).",
    )
    parser.add_argument(
        "--use-crash-filter",
        action="store_true",
        help="Skip entries during behavioral crash regimes (C5).",
    )
    parser.add_argument(
        "--crash-threshold",
        type=float,
        default=0.50,
        help="Crash risk score threshold for entry suppression (default 0.50).",
    )
    # ── E2 Chronos Foundation Model ──
    parser.add_argument(
        "--use-chronos",
        action="store_true",
        help="Enable Chronos-2 zero-shot forecast signal blend with ML probability.",
    )
    parser.add_argument(
        "--chronos-weight",
        type=float,
        default=0.30,
        help="Weight of Chronos signal in blend (0-1, default 0.30).",
    )
    # ── E5 Cross-Asset Inference ──
    parser.add_argument(
        "--cross-asset-inference",
        action="store_true",
        help="Enable cross-asset features (beta, rel_ret, corr) at inference.",
    )
    # ── C6 Defensive Backtest + Time-Reversal ──
    parser.add_argument(
        "--defensive",
        action="store_true",
        help="Run defensive backtest with time-reversal check (C6). "
        "Re-runs strategy on time-reversed data to detect overfit.",
    )
    parser.add_argument(
        "--time-reversal",
        action="store_true",
        help="Run time-reversal heuristic check (Svozil 2026). "
        "If strategy profitable on reversed data, overfit likely.",
    )
    # ── A+B Regime Router ──
    parser.add_argument(
        "--use-regime-router",
        action="store_true",
        help="Route predictions through per-regime ML models via RegimeRouter (A+B).",
    )
    parser.add_argument(
        "--regime-router-config",
        type=str,
        default="models/regime_router_SPY.json",
        help="Path to RegimeRouter config JSON.",
    )
    # ── Q1 Multi-Factor Fundamentals ──
    parser.add_argument(
        "--use-fundamentals",
        action="store_true",
        help="Join fundamental factors (P/E, P/B, ROE, etc.) to feature matrix.",
    )
    args = parser.parse_args()

    tickers = [t.strip() for t in args.symbols.split(",")]

    if args.compare:
        results = []
        for ticker in tickers:
            print(f"Running {ticker}...")
            try:
                r = run_single(
                    ticker,
                    args.model,
                    args.cash,
                    args.start,
                    args.end,
                    args.entry_threshold,
                    args.vol_gate,
                    args.confirm,
                    args.trail_stop,
                    args.trail_atr,
                    args.conviction,
                    args.multi_tp,
                    args.use_dynamic_ensemble,
                    args.dynamic_ensemble_path,
                    args.use_meta_label,
                    args.meta_label_path,
                    args.use_sentiment,
                    args.sentiment_weight,
                    args.use_lm_sentiment,
                    args.lm_sentiment_weight,
                    args.use_event_filter,
                    args.use_rl_execution,
                    args.rl_model_path,
                    args.use_kelly,
                    args.kelly_fraction,
                    args.use_crash_filter,
                    args.crash_threshold,
                    args.defensive,
                    args.time_reversal,
                    args.use_chronos,
                    args.chronos_weight,
                    args.cross_asset_inference,
                    args.use_regime_router,
                    args.regime_router_config,
                    args.use_fundamentals,
                )
                results.append(r)
            except Exception as e:
                print(f"  FAILED: {e}")
                continue

        if not results:
            print("No results.")
            return

        df = pd.DataFrame(results)
        # Print comparison table
        print("\n" + "=" * 120)
        print("ML Strategy Backtest Comparison")
        print("=" * 120)
        cols = [
            "symbol",
            "return_pct",
            "buy_hold_return",
            "sharpe",
            "max_drawdown",
            "win_rate",
            "num_trades",
            "profit_factor",
            "avg_trade",
            "return_annual",
        ]
        print(df[cols].to_string(index=False))

        # Highlight winners
        winners = df[df["return_pct"] > df["buy_hold_return"]]
        if len(winners) > 0:
            print(f"\nBeat buy-and-hold: {', '.join(winners['symbol'].tolist())}")
        print(f"\nMean Sharpe: {df['sharpe'].mean():.2f}")
        print(f"Mean Win Rate: {df['win_rate'].mean():.1f}%")
        print(f"Total Trades: {df['num_trades'].sum()}")

        # Save
        out_path = Path("reports/ml_backtest/comparison.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")
    else:
        ticker = tickers[0]
        r = run_single(
            ticker,
            args.model,
            args.cash,
            args.start,
            args.end,
            args.entry_threshold,
            args.vol_gate,
            args.confirm,
            args.trail_stop,
            args.trail_atr,
            args.conviction,
            args.multi_tp,
            args.use_dynamic_ensemble,
            args.dynamic_ensemble_path,
            args.use_meta_label,
            args.meta_label_path,
            args.use_sentiment,
            args.sentiment_weight,
            args.use_lm_sentiment,
            args.lm_sentiment_weight,
            args.use_event_filter,
            args.use_rl_execution,
            args.rl_model_path,
            args.use_kelly,
            args.kelly_fraction,
            args.use_crash_filter,
            args.crash_threshold,
            args.defensive,
            args.time_reversal,
            args.use_chronos,
            args.chronos_weight,
            args.cross_asset_inference,
            args.use_regime_router,
            args.regime_router_config,
            args.use_fundamentals,
        )
        print("\n" + "=" * 80)
        print(f"ML Strategy Backtest: {ticker}")
        print(
            f"C7 features: vol_gate={args.vol_gate} confirm={args.confirm}"
            f" trail={args.trail_stop} conviction={args.conviction}"
            f" sentiment={args.use_sentiment}(w={args.sentiment_weight})"
            f" event_filter={args.use_event_filter}"
            f" rl_exec={args.use_rl_execution}"
            f" kelly={args.use_kelly}"
            f" crash_filter={args.use_crash_filter}"
            f" defensive={args.defensive}"
            f" chronos={args.use_chronos}(w={args.chronos_weight})"
            f" cross_asset={args.cross_asset_inference}"
            f" regime_router={args.use_regime_router}"
        )
        print("=" * 80)

        # Extract full stats from backtesting.py to produce comprehensive report
        from src.backtest.metrics import PerformanceMetrics

        df = load_data(ticker)
        if args.start:
            df = df[df.index >= args.start]
        if args.end:
            df = df[df.index <= args.end]

        from backtesting import Backtest

        bt_full = Backtest(
            df,
            MLStrategy,
            cash=args.cash,
            commission=0.001,
            exclusive_orders=True,
        )
        full_stats = bt_full.run(
            model_path=args.model,
            ticker=ticker,
            entry_threshold=args.entry_threshold,
            exit_threshold=args.entry_threshold * 0.7,
            tp_atr_mult=3.0,
            sl_atr_mult=1.5,
            risk_pct=0.02,
            vol_gate_threshold=args.vol_gate,
            confirm_bars=args.confirm,
            use_trail_stop=args.trail_stop,
            trail_atr_mult=args.trail_atr,
            conviction_scale=args.conviction,
            use_multi_tp=args.multi_tp,
            use_dynamic_ensemble=args.use_dynamic_ensemble,
            dynamic_ensemble_path=args.dynamic_ensemble_path,
            use_meta_label=args.use_meta_label,
            meta_label_path=args.meta_label_path,
            use_sentiment=args.use_sentiment,
            sentiment_weight=args.sentiment_weight,
            use_lm_sentiment=args.use_lm_sentiment,
            lm_sentiment_weight=args.lm_sentiment_weight,
            use_event_filter=args.use_event_filter,
            use_rl_execution=args.use_rl_execution,
            rl_model_path=args.rl_model_path,
            use_kelly=args.use_kelly,
            kelly_fraction=args.kelly_fraction,
            use_crash_filter=args.use_crash_filter,
            crash_threshold=args.crash_threshold,
            use_chronos=args.use_chronos,
            chronos_weight=args.chronos_weight,
            use_cross_asset=args.cross_asset_inference,
            use_regime_router=args.use_regime_router,
            regime_router_config=args.regime_router_config,
        )
        stats_dict: dict[str, Any] = {}
        try:
            stats_dict = dict(full_stats._asdict())
        except AttributeError:
            for key in full_stats.index:
                try:
                    stats_dict[key] = full_stats[key]
                except (KeyError, TypeError):
                    pass
        stats_dict["_trades"] = list(full_stats._trades) if hasattr(full_stats, "_trades") else []
        stats_dict["_equity_curve"] = (
            full_stats._equity_curve if hasattr(full_stats, "_equity_curve") else None
        )

        metrics = PerformanceMetrics.from_backtesting_stats(stats_dict, initial_equity=args.cash)
        print(PerformanceMetrics.format_report(metrics))

        # Print defensive results if available
        if r.get("defensive"):
            print("\n" + "─" * 60)
            print("DEFENSIVE BACKTEST (Time-Reversal Check)")
            print("─" * 60)
            for k, v in r["defensive"].items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
