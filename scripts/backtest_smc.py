"""
SMC Intraday Strategy Backtest CLI.

Usage:
    uv run scripts/backtest_smc.py --symbol BTC-USD --interval 1h
    uv run scripts/backtest_smc.py --symbol BTC-USD --interval 1h --sweep-entry
    uv run scripts/backtest_smc.py --symbol BTC-USD --interval 1h --use-multi-tp
    uv run scripts/backtest_smc.py --all --interval 1h
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def fetch_data(
    symbol: str, interval: str = "1h", start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """Fetch data from local CSV or yfinance fallback."""
    safe_symbol = symbol.replace("=", "_").replace("/", "_")
    path = Path(f"data/raw/{safe_symbol}_{interval}.csv")

    alt_symbol = symbol.replace("-", "_")
    alt_path = Path(f"data/raw/{alt_symbol}_{interval}.csv")

    read_path = path if path.exists() else (alt_path if alt_path.exists() else None)

    if read_path is not None:
        df = pd.read_csv(read_path, index_col=0, parse_dates=True)
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]
        df = df.dropna()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = 0 if col == "Volume" else df.iloc[:, 0]
        df.columns = [c.capitalize() for c in df.columns]
        return df

    import yfinance as yf

    period = "730d" if interval == "1h" else "60d"
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {symbol} at {interval} interval")
    df.index = pd.to_datetime(df.index)
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def compute_buy_hold_metrics(close_series: pd.Series) -> dict:
    if len(close_series) < 2:
        return {"return_pct": 0.0, "sharpe": 0.0, "max_dd_pct": 0.0}
    bh_return = (close_series.iloc[-1] / close_series.iloc[0] - 1) * 100
    daily_ret = close_series.pct_change().dropna()
    bh_sharpe = (
        float((daily_ret.mean() / daily_ret.std() * (252**0.5))) if daily_ret.std() > 0 else 0.0
    )
    bh_max_dd = float(((close_series / close_series.cummax() - 1).min()) * 100)
    return {"return_pct": bh_return, "sharpe": bh_sharpe, "max_dd_pct": bh_max_dd}


def compute_trade_metrics(trades_df: Optional[pd.DataFrame]) -> dict:
    if trades_df is None or len(trades_df) == 0:
        return {
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "wl_ratio": 0.0,
            "avg_bars_held": 0,
            "best_trade_pct": 0.0,
            "worst_trade_pct": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "max_drawdown_pct": 0.0,
        }
    closed = trades_df.copy()
    closed["pnl_pct"] = closed["ReturnPct"].astype(float)
    wins = closed[closed["pnl_pct"] > 0]
    losses = closed[closed["pnl_pct"] <= 0]
    avg_bars = int(closed["BarsHeld"].mean()) if "BarsHeld" in closed.columns else 0
    equity_curve = (1 + closed["pnl_pct"] / 100).cumprod()
    dd = (equity_curve / equity_curve.cummax() - 1).min() * 100
    return {
        "avg_win_pct": float(wins["pnl_pct"].mean()) if len(wins) > 0 else 0.0,
        "avg_loss_pct": float(losses["pnl_pct"].mean()) if len(losses) > 0 else 0.0,
        "wl_ratio": abs(float(wins["pnl_pct"].mean()) / float(losses["pnl_pct"].mean()))
        if len(losses) > 0 and float(losses["pnl_pct"].mean()) != 0
        else 0.0,
        "avg_bars_held": avg_bars,
        "best_trade_pct": float(closed["pnl_pct"].max()),
        "worst_trade_pct": float(closed["pnl_pct"].min()),
        "long_trades": int(len(closed[closed["Size"] >= 0])) if "Size" in closed.columns else 0,
        "short_trades": int(len(closed[closed["Size"] < 0])) if "Size" in closed.columns else 0,
        "max_drawdown_pct": float(dd) if not pd.isna(dd) else 0.0,
    }


def run_single(
    symbol: str,
    interval: str = "1h",
    cash: float = 10_000,
    start: str | None = None,
    end: str | None = None,
    entry_threshold: float = 0.55,
    exit_threshold: float = 0.30,
    trail_stop_atr: float = 3.0,
    session_bars: int = 24,
    atr_period: int = 14,
    confluence_bonus: float = 0.10,
    volume_confirm: bool = True,
    use_multi_tp: bool = True,
    tp1_atr: float = 1.5,
    tp1_size: float = 0.5,
    move_sl_to_be: bool = True,
    use_short: bool = False,
    sweep_buffer_mult: float = 0.5,
    fvg_proximity_mult: float = 2.0,
    use_vol_gate: bool = False,
    use_session_gate: bool = False,
    use_crash_gate: bool = False,
    use_volume_pressure: bool = False,
    use_order_blocks: bool = False,
    session_trade_start: int = 0,
    session_trade_end: int = 24,
    min_confluence: int = 0,
    htf_ema_period: int = 50,
    crypto_mode: bool = False,
    use_breaker_blocks: bool = False,
    use_mitigation_blocks: bool = False,
    use_rejection_blocks: bool = False,
    use_dow_gate: bool = False,
    use_90min_cycle: bool = False,
    use_frankfurt_gate: bool = False,
    use_smc_sessions: bool = True,
    use_smc_retrace: bool = False,
    crp_lookback: int = 20,
    ob_lookback: int = 5,
    use_vix_gate: bool = False,
    vix_gate_stress_mult: float = 0.30,
    vix_gate_elevated_mult: float = 0.75,
    use_yield_curve_gate: bool = False,
    yield_curve_inversion_mult: float = 0.50,
    yield_curve_near_inversion_mult: float = 0.75,
    use_htf_gate: bool = False,
    use_killzone_gate: bool = False,
    use_swing_points: bool = False,
    swing_point_weight: float = 0.20,
    use_ict_patterns: bool = False,
    ict_pattern_weight: float = 0.15,
) -> dict:
    from backtesting import Backtest
    from backtesting.lib import FractionalBacktest
    from src.strategies.smc_strategy import SMCStrategy

    df = fetch_data(symbol, interval, start, end)
    if len(df) < 50:
        raise ValueError(f"Insufficient data: {len(df)} bars")

    max_price = float(df["Close"].max())
    bt_class = FractionalBacktest if max_price > cash else Backtest

    bt = bt_class(
        df,
        SMCStrategy,
        cash=cash,
        commission=0.001,
        exclusive_orders=True,
        finalize_trades=True,
    )

    stats = bt.run(
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        trail_stop_atr=trail_stop_atr,
        session_bars=session_bars,
        atr_period=atr_period,
        confluence_bonus=confluence_bonus,
        volume_confirm=volume_confirm,
        use_multi_tp=use_multi_tp,
        tp1_atr=tp1_atr,
        tp1_size=tp1_size,
        move_sl_to_be=move_sl_to_be,
        use_short=use_short,
        sweep_buffer_mult=sweep_buffer_mult,
        fvg_proximity_mult=fvg_proximity_mult,
        use_vol_gate=use_vol_gate,
        use_session_gate=use_session_gate,
        use_crash_gate=use_crash_gate,
        use_volume_pressure=use_volume_pressure,
        use_order_blocks=use_order_blocks,
        session_trade_start=session_trade_start,
        session_trade_end=session_trade_end,
        min_confluence=min_confluence,
        htf_ema_period=htf_ema_period,
        crypto_mode=crypto_mode,
        use_breaker_blocks=use_breaker_blocks,
        use_mitigation_blocks=use_mitigation_blocks,
        use_rejection_blocks=use_rejection_blocks,
        use_dow_gate=use_dow_gate,
        use_90min_cycle=use_90min_cycle,
        use_frankfurt_gate=use_frankfurt_gate,
        use_smc_sessions=use_smc_sessions,
        use_smc_retrace=use_smc_retrace,
        crp_lookback=crp_lookback,
        ob_lookback=ob_lookback,
        use_vix_gate=use_vix_gate,
        vix_gate_stress_mult=vix_gate_stress_mult,
        vix_gate_elevated_mult=vix_gate_elevated_mult,
        use_yield_curve_gate=use_yield_curve_gate,
        yield_curve_inversion_mult=yield_curve_inversion_mult,
        yield_curve_near_inversion_mult=yield_curve_near_inversion_mult,
        use_htf_gate=use_htf_gate,
        use_killzone_gate=use_killzone_gate,
        use_swing_points=use_swing_points,
        swing_point_weight=swing_point_weight,
        use_ict_patterns=use_ict_patterns,
        ict_pattern_weight=ict_pattern_weight,
    )

    bh = compute_buy_hold_metrics(df["Close"])
    trade_m = compute_trade_metrics(getattr(stats, "_trades", None))

    result = {
        "symbol": symbol,
        "interval": interval,
        "start": str(df.index[0].date()) if len(df) > 0 else "N/A",
        "end": str(df.index[-1].date()) if len(df) > 0 else "N/A",
        "bars": len(df),
        "entry_threshold": entry_threshold,
        "return_pct": stats["Return [%]"],
        "sharpe": stats["Sharpe Ratio"],
        "sortino": stats.get("Sortino Ratio", 0),
        "calmar": stats.get("Calmar Ratio", 0),
        "max_dd_pct": stats["Max. Drawdown [%]"],
        "trades": stats["# Trades"],
        "win_rate_pct": stats["Win Rate [%]"],
        "profit_factor": stats["Profit Factor"],
        "exposure_pct": stats["Exposure Time [%]"],
        "ann_return_pct": stats.get("Return (Ann.) [%]", 0),
        "ann_vol_pct": stats.get("Volatility (Ann.) [%]", 0),
        "avg_win_pct": trade_m["avg_win_pct"],
        "avg_loss_pct": trade_m["avg_loss_pct"],
        "wl_ratio": trade_m["wl_ratio"],
        "avg_bars_held": trade_m["avg_bars_held"],
        "best_trade_pct": trade_m["best_trade_pct"],
        "worst_trade_pct": trade_m["worst_trade_pct"],
        "long_trades": trade_m["long_trades"],
        "short_trades": trade_m["short_trades"],
        "bh_return_pct": round(bh["return_pct"], 2),
        "bh_sharpe": round(bh["sharpe"], 3),
        "bh_max_dd_pct": round(bh["max_dd_pct"], 2),
    }
    return result


def sweep_entry(
    symbol: str, interval: str, thresholds: list[float], start: str, end: str, **kwargs
) -> list[dict]:
    results = []
    for et in thresholds:
        r = run_single(
            symbol, interval=interval, start=start, end=end, entry_threshold=et, **kwargs
        )
        r["entry_threshold"] = et
        results.append(r)
        print(
            f"  et={et:.2f}: Sharpe {r['sharpe']:.3f}, Return {r['return_pct']:.1f}%, "
            f"Trades {r['trades']}, Win {r['win_rate_pct']:.1f}%, PF {r['profit_factor']:.2f}"
        )
    return results


def sweep_sweep_buffer(
    symbol: str, interval: str, buffers: list[float], start: str, end: str, **kwargs
) -> list[dict]:
    results = []
    for sb in buffers:
        r = run_single(
            symbol, interval=interval, start=start, end=end, sweep_buffer_mult=sb, **kwargs
        )
        r["sweep_buffer_mult"] = sb
        results.append(r)
        print(
            f"  sb={sb:.2f}: Sharpe {r['sharpe']:.3f}, Return {r['return_pct']:.1f}%, "
            f"Trades {r['trades']}, Win {r['win_rate_pct']:.1f}%, PF {r['profit_factor']:.2f}"
        )
    return results


def print_table(results: list[dict], compact: bool = False) -> None:
    coll = [
        "symbol",
        "interval",
        "entry_threshold",
        "return_pct",
        "sharpe",
        "sortino",
        "calmar",
        "max_dd_pct",
        "trades",
        "win_rate_pct",
        "profit_factor",
        "ann_return_pct",
        "ann_vol_pct",
        "avg_win_pct",
        "avg_loss_pct",
        "wl_ratio",
        "avg_bars_held",
        "best_trade_pct",
        "worst_trade_pct",
        "exposure_pct",
        "bh_return_pct",
        "bh_sharpe",
        "bh_max_dd_pct",
    ]
    if compact:
        coll = [
            "symbol",
            "interval",
            "entry_threshold",
            "return_pct",
            "sharpe",
            "sortino",
            "calmar",
            "max_dd_pct",
            "trades",
            "win_rate_pct",
            "profit_factor",
            "ann_return_pct",
            "bh_return_pct",
            "bh_sharpe",
        ]
    header = [c.replace("_pct", "%").replace("_mult", " mult") for c in coll]
    widths = [max(len(h), 8) for h in header]
    for r in results:
        for i, c in enumerate(coll):
            if c in r and r[c] is not None:
                widths[i] = max(
                    widths[i], len(f"{r[c]:.2f}") if isinstance(r[c], float) else len(str(r[c]))
                )

    fmt = "  ".join(f"{{:>{w}}}" for w in widths)
    print(fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for r in results:
        vals = []
        for c in coll:
            v = r.get(c)
            if v is None:
                vals.append("N/A")
            elif isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        print(fmt.format(*vals))


def run_all_instruments(
    symbols: list[str],
    interval: str,
    start: str | None,
    end: str | None,
    **kwargs,
) -> list[dict]:
    results = []
    for sym in symbols:
        try:
            print(f"\n--- {sym} ({interval}) ---")
            r = run_single(sym, interval=interval, start=start, end=end, **kwargs)
            results.append(r)
            print(
                f"  Sharpe {r['sharpe']:.3f}, Return {r['return_pct']:.1f}%, "
                f"Trades {r['trades']}, Win {r['win_rate_pct']:.1f}%, "
                f"PF {r['profit_factor']:.2f}, MaxDD {r['max_dd_pct']:.1f}%  |  "
                f"B&H Return {r['bh_return_pct']:.1f}%"
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"symbol": sym, "interval": interval, "error": str(e)})
    return results


SMC_HOURLY_SYMBOLS: list[str] = [
    "NQ=F",
    "GC=F",
]

SMC_5M_SYMBOLS: list[str] = [
    "NQ=F",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest SMC intraday strategy")
    parser.add_argument("--symbol", default="BTC-USD", help="Ticker symbol")
    parser.add_argument("--all", action="store_true", help="Run on all SMC instruments")
    parser.add_argument("--interval", default="1h", help="Data interval (1h, 5m)")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--cash", type=float, default=10_000, help="Initial cash")
    parser.add_argument(
        "--entry-threshold", type=float, default=0.55, help="Entry signal threshold"
    )
    parser.add_argument("--exit-threshold", type=float, default=0.30, help="Exit signal threshold")
    parser.add_argument(
        "--trail-stop-atr", type=float, default=3.0, help="ATR multiplier for trail"
    )
    parser.add_argument("--session-bars", type=int, default=24, help="Bars per session range")
    parser.add_argument("--atr-period", type=int, default=14, help="ATR period")
    parser.add_argument("--confluence-bonus", type=float, default=0.10, help="Confluence bonus")
    parser.add_argument(
        "--no-volume-confirm", action="store_true", help="Disable volume confirmation"
    )
    parser.add_argument(
        "--use-multi-tp",
        action="store_true",
        default=True,
        help="Enable multi-TP exit (default ON)",
    )
    parser.add_argument("--no-multi-tp", action="store_true", help="Disable multi-TP exit")
    parser.add_argument("--tp1-atr", type=float, default=1.5, help="TP1 ATR distance")
    parser.add_argument("--tp1-size", type=float, default=0.5, help="Portion to close at TP1")
    parser.add_argument("--no-move-be", action="store_true", help="Don't move SL to BE after TP1")
    parser.add_argument(
        "--no-short",
        action="store_true",
        help="Disable short entries (alias; short is OFF by default)",
    )
    parser.add_argument("--use-short", action="store_true", help="Enable short entries")
    parser.add_argument("--sweep-buffer-mult", type=float, default=0.5, help="ATR buffer for sweep")
    parser.add_argument(
        "--fvg-proximity-mult", type=float, default=2.0, help="ATR mult for FVG zone"
    )
    parser.add_argument(
        "--vol-gate", action="store_true", help="Enable volatility gate (default OFF)"
    )
    parser.add_argument(
        "--session-gate", action="store_true", help="Enable session time gate (default OFF)"
    )
    parser.add_argument(
        "--crash-gate", action="store_true", help="Enable crash factor gate (default OFF)"
    )
    parser.add_argument(
        "--no-vol-gate",
        action="store_true",
        help="Disable volatility gate (deprecated; use --vol-gate to enable)",
    )
    parser.add_argument(
        "--no-session-gate", action="store_true", help="Disable session time gate (deprecated)"
    )
    parser.add_argument(
        "--no-crash-gate", action="store_true", help="Disable crash factor gate (deprecated)"
    )
    parser.add_argument(
        "--volume-pressure",
        action="store_true",
        help="Enable volume pressure features (default OFF)",
    )
    parser.add_argument("--no-volume-pressure", action="store_true", help="Disable volume pressure")
    parser.add_argument(
        "--order-blocks", action="store_true", help="Enable order block detection (default OFF)"
    )
    parser.add_argument("--no-order-blocks", action="store_true", help="Disable order blocks")
    parser.add_argument(
        "--session-trade-start", type=int, default=0, help="UTC hour to start trading"
    )
    parser.add_argument(
        "--session-trade-end", type=int, default=24, help="UTC hour to stop trading"
    )
    parser.add_argument(
        "--min-confluence",
        type=int,
        default=0,
        help="Minimum components agreeing for entry (0=no gating)",
    )
    # Phase 22 fixes: Quality gates
    parser.add_argument(
        "--use-htf-gate",
        action="store_true",
        default=False,
        help="Enable hard HTF trend gate (default OFF)",
    )
    parser.add_argument(
        "--no-htf-gate", action="store_true", help="Disable HTF trend gate (redundant)"
    )
    parser.add_argument(
        "--use-killzone-gate",
        action="store_true",
        default=False,
        help="Enable killzone-only trading (default OFF)",
    )
    parser.add_argument("--no-killzone-gate", action="store_true", help="Disable killzone gate")
    parser.add_argument(
        "--htf-ema-period", type=int, default=50, help="HTF daily EMA period for bias"
    )
    parser.add_argument(
        "--crypto-mode", action="store_true", help="Auto-adjust gates for crypto volatility"
    )
    parser.add_argument(
        "--use-breaker-blocks", action="store_true", help="Enable breaker block detection (Phase 6)"
    )
    parser.add_argument(
        "--use-mitigation-blocks",
        action="store_true",
        help="Enable mitigation block detection (Phase 6)",
    )
    parser.add_argument(
        "--use-rejection-blocks",
        action="store_true",
        help="Enable rejection block detection (Phase 6)",
    )
    parser.add_argument(
        "--use-dow-gate", action="store_true", help="Enable day-of-week gate (Phase 12)"
    )
    parser.add_argument(
        "--use-90min-cycle",
        action="store_true",
        help="Enable 90-minute cycle sensitivity (Phase 12)",
    )
    parser.add_argument(
        "--use-frankfurt-gate",
        action="store_true",
        help="Enable Frankfurt fake move gate (Phase 12)",
    )
    parser.add_argument(
        "--use-smc-sessions",
        action="store_true",
        default=True,
        help="Enable smartmoneyconcepts killzone sessions (Phase 9, default ON)",
    )
    parser.add_argument(
        "--use-smc-retrace",
        action="store_true",
        help="Enable smartmoneyconcepts retracements (Phase 9)",
    )
    # Phase 21: New-tech macro regime gates
    parser.add_argument(
        "--use-vix-gate",
        action="store_true",
        default=False,
        help="Enable VIX regime gate (default OFF)",
    )
    parser.add_argument(
        "--no-vix-gate", action="store_true", help="Disable VIX regime gate (redundant)"
    )
    parser.add_argument("--vix-stress-mult", type=float, default=0.30, help="VIX stress multiplier")
    parser.add_argument(
        "--vix-elevated-mult", type=float, default=0.75, help="VIX elevated multiplier"
    )
    parser.add_argument(
        "--use-yield-curve-gate",
        action="store_true",
        default=False,
        help="Enable yield curve macro gate (default OFF)",
    )
    parser.add_argument(
        "--no-yield-curve-gate", action="store_true", help="Disable yield curve macro gate"
    )
    parser.add_argument(
        "--yield-inversion-mult", type=float, default=0.50, help="Yield inversion multiplier"
    )
    parser.add_argument(
        "--yield-near-inversion-mult",
        type=float,
        default=0.75,
        help="Yield near-inversion multiplier",
    )
    parser.add_argument("--crp-lookback", type=int, default=20, help="Crash factor rolling window")
    parser.add_argument("--ob-lookback", type=int, default=5, help="Order block lookback bars")
    parser.add_argument(
        "--use-swing-points",
        action="store_true",
        help="A2: Enable swing point detector (aligns swing highs/lows with sweep direction)",
    )
    parser.add_argument(
        "--swing-point-weight",
        type=float,
        default=0.20,
        help="A2: Weight of swing point alignment bonus",
    )
    parser.add_argument(
        "--use-ict-patterns", action="store_true", help="A1: Enable 12 ICT candlestick patterns"
    )
    parser.add_argument(
        "--ict-pattern-weight", type=float, default=0.15, help="A1: Weight of ICT pattern bonus"
    )
    parser.add_argument("--sweep-entry", help="Comma-separated entry thresholds to sweep")
    parser.add_argument("--sweep-buffer", help="Comma-separated buffer multipliers to sweep")
    parser.add_argument(
        "--sweep-configs", action="store_true", help="Sweep multiple configs per instrument"
    )
    parser.add_argument("--json", "-o", default=None, help="Save results as JSON")

    args = parser.parse_args()

    kwargs = dict(
        exit_threshold=args.exit_threshold,
        trail_stop_atr=args.trail_stop_atr,
        session_bars=args.session_bars,
        atr_period=args.atr_period,
        confluence_bonus=args.confluence_bonus,
        volume_confirm=not args.no_volume_confirm,
        use_multi_tp=args.use_multi_tp and not args.no_multi_tp,
        tp1_atr=args.tp1_atr,
        tp1_size=args.tp1_size,
        move_sl_to_be=not args.no_move_be,
        use_short=args.use_short,
        sweep_buffer_mult=args.sweep_buffer_mult,
        fvg_proximity_mult=args.fvg_proximity_mult,
        use_vol_gate=args.vol_gate and not args.no_vol_gate,
        use_session_gate=args.session_gate and not args.no_session_gate,
        use_crash_gate=args.crash_gate and not args.no_crash_gate,
        use_volume_pressure=args.volume_pressure and not args.no_volume_pressure,
        use_order_blocks=args.order_blocks and not args.no_order_blocks,
        session_trade_start=args.session_trade_start,
        session_trade_end=args.session_trade_end,
        min_confluence=args.min_confluence,
        htf_ema_period=args.htf_ema_period,
        crypto_mode=args.crypto_mode,
        use_breaker_blocks=args.use_breaker_blocks,
        use_mitigation_blocks=args.use_mitigation_blocks,
        use_rejection_blocks=args.use_rejection_blocks,
        use_dow_gate=args.use_dow_gate,
        use_90min_cycle=args.use_90min_cycle,
        use_frankfurt_gate=args.use_frankfurt_gate,
        use_smc_sessions=args.use_smc_sessions,
        use_smc_retrace=args.use_smc_retrace,
        crp_lookback=args.crp_lookback,
        ob_lookback=args.ob_lookback,
        use_vix_gate=args.use_vix_gate and not args.no_vix_gate,
        vix_gate_stress_mult=args.vix_stress_mult,
        vix_gate_elevated_mult=args.vix_elevated_mult,
        use_yield_curve_gate=args.use_yield_curve_gate and not args.no_yield_curve_gate,
        yield_curve_inversion_mult=args.yield_inversion_mult,
        yield_curve_near_inversion_mult=args.yield_near_inversion_mult,
        use_htf_gate=args.use_htf_gate and not args.no_htf_gate,
        use_killzone_gate=args.use_killzone_gate and not args.no_killzone_gate,
        use_swing_points=args.use_swing_points,
        swing_point_weight=args.swing_point_weight,
        use_ict_patterns=args.use_ict_patterns,
        ict_pattern_weight=args.ict_pattern_weight,
    )

    all_results: list[dict] = []

    symbols: list[str]
    if args.all:
        symbols = SMC_5M_SYMBOLS if args.interval == "5m" else SMC_HOURLY_SYMBOLS
    else:
        symbols = [args.symbol]

    if args.sweep_entry:
        thresholds = [float(t.strip()) for t in args.sweep_entry.split(",")]
        for sym in symbols:
            print(f"\n=== Sweeping entry thresholds for {sym} (thresholds={thresholds}) ===")
            results = sweep_entry(
                sym,
                args.interval,
                thresholds,
                args.start,
                args.end,
                **{k: v for k, v in kwargs.items() if k != "entry_threshold"},
            )
            for r in results:
                r["symbol"] = sym
            all_results.extend(results)
            print_table(results, compact=True)
    elif args.sweep_buffer:
        buffers = [float(t.strip()) for t in args.sweep_buffer.split(",")]
        for sym in symbols:
            print(f"\n=== Sweeping buffer multipliers for {sym} ===")
            results = sweep_sweep_buffer(
                sym,
                args.interval,
                buffers,
                args.start,
                args.end,
                **{k: v for k, v in kwargs.items() if k != "sweep_buffer_mult"},
            )
            for r in results:
                r["symbol"] = sym
            all_results.extend(results)
            print_table(results, compact=True)
    elif args.sweep_configs:
        configs = [
            {"entry_threshold": 0.45},
            {"entry_threshold": 0.50},
            {"entry_threshold": 0.55},
            {"entry_threshold": 0.60},
            {"entry_threshold": 0.65},
            {"use_multi_tp": True, "entry_threshold": 0.55},
            {"use_multi_tp": True, "entry_threshold": 0.60},
        ]
        for sym in symbols:
            print(f"\n=== Sweeping configs for {sym} ===")
            for cfg in configs:
                merged = {**kwargs, **cfg}
                try:
                    merged.pop("entry_threshold", None)
                except Exception:
                    pass
                et = cfg.get("entry_threshold", 0.55)
                r = run_single(
                    sym,
                    interval=args.interval,
                    start=args.start,
                    end=args.end,
                    entry_threshold=et,
                    **{k: v for k, v in merged.items() if k != "entry_threshold"},
                )
                r["symbol"] = sym
                r["config"] = str(cfg)
                all_results.append(r)
                print(
                    f"  cfg={cfg}: Sharpe {r['sharpe']:.3f}, Return {r['return_pct']:.1f}%, "
                    f"Trades {r['trades']}, Win {r['win_rate_pct']:.1f}%, PF {r['profit_factor']:.2f}"
                )
        print()
        print_table(all_results, compact=True)
    elif args.all:
        results = run_all_instruments(
            symbols,
            args.interval,
            args.start,
            args.end,
            entry_threshold=args.entry_threshold,
            **{k: v for k, v in kwargs.items() if k != "entry_threshold"},
        )
        all_results = results
        print()
        print_table(results)
    else:
        result = run_single(
            args.symbol,
            interval=args.interval,
            cash=args.cash,
            start=args.start,
            end=args.end,
            entry_threshold=args.entry_threshold,
            **{k: v for k, v in kwargs.items() if k != "entry_threshold"},
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
            f"trail={args.trail_stop_atr} multi_tp={args.use_multi_tp and not args.no_multi_tp} "
            f"short={args.use_short} buffer={args.sweep_buffer_mult} confluence={args.min_confluence} "
            f"vol_gate={args.vol_gate} session={args.session_gate} crash={args.crash_gate} "
            f"breaker={args.use_breaker_blocks} mitigation={args.use_mitigation_blocks} rejection={args.use_rejection_blocks} "
            f"dow={args.use_dow_gate} cycle90={args.use_90min_cycle} frankfurt={args.use_frankfurt_gate} "
            f"smc_sessions={args.use_smc_sessions} smc_retrace={args.use_smc_retrace} htf_period={args.htf_ema_period} crypto={args.crypto_mode} "
            f"vix_gate={args.use_vix_gate and not args.no_vix_gate} yield_gate={args.use_yield_curve_gate and not args.no_yield_curve_gate} "
            f"htf_gate={args.use_htf_gate and not args.no_htf_gate} kz_gate={args.use_killzone_gate and not args.no_killzone_gate} "
            f"min_confl={args.min_confluence}"
        )

    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_results, indent=2, default=str))
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
