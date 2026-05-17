"""
Instrument-Specific Alpha Discovery

Digs unique alphas for 10 instruments across asset classes and liquidity tiers.
Each alpha is designed for the specific characteristics of its instrument.

Usage:
    uv run scripts/dig_alphas.py
    uv run scripts/dig_alphas.py --start 2020-01-01 --end 2024-12-31  # IS only
    uv run scripts/dig_alphas.py --json-output reports/batch/alphas.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ── Instrument ↔ Alpha Definitions ─────────────────────────────────────────

ALPHA_DEFS: dict[str, dict] = {
    # ── Top 3 Most Liquid ──
    "SPY": {
        "name": "Gap Fill",
        "tier": "most-liquid",
        "rationale": "Overnight gaps in SPY tend to fill intraday -- market-maker hedging + mean reversion.",
        "params": {"gap_threshold": 0.005, "max_hold": 5},
    },
    "QQQ": {
        "name": "BB Squeeze Breakout",
        "tier": "most-liquid",
        "rationale": "Tech trends strongly after volatility contraction. Turtle-style BB squeeze breakout.",
        "params": {"bb_period": 20, "bb_std": 2.0, "squeeze_pct": 20},
    },
    "IWM": {
        "name": "RSI(2) Oversold Bounce",
        "tier": "most-liquid",
        "rationale": "Small caps overreact to selling pressure. RSI(2) < 15 signals capitulation bounce.",
        "params": {"rsi_period": 2, "oversold": 15, "max_hold": 3},
    },
    # ── Top 3 Least Liquid ──
    "SO": {
        "name": "Defensive Dip Buy",
        "tier": "least-liquid",
        "rationale": "Utilities are range-bound yield vehicles. 2-sigma dips revert to mean within weeks.",
        "params": {"ma_period": 50, "std_mult": 2.0},
    },
    "JNJ": {
        "name": "Low-Vol Momentum",
        "tier": "least-liquid",
        "rationale": "Defensive healthcare exhibits low-vol anomaly: buy when calm, above trend.",
        "params": {"ma_period": 50, "vol_period": 20, "vol_percentile": 50},
    },
    "XLV": {
        "name": "Sector Oversold Reversal",
        "tier": "least-liquid",
        "rationale": "Healthcare ETF mean-reverts when oversold -- investors rotate back to defense.",
        "params": {"rsi_period": 14, "oversold": 30, "overbought": 55},
    },
    # ── Precious Metals ──
    "GLD": {
        "name": "Trend + Rate Filter",
        "tier": "commodity",
        "rationale": "Gold trends in falling-rate environments. Above 50MA + ADX confirms trend.",
        "params": {"ma_fast": 10, "ma_slow": 50, "adx_period": 14, "adx_min": 20},
        "trail_atr": 6.0,
    },
    "SLV": {
        "name": "Vol Expansion Breakout",
        "tier": "commodity",
        "rationale": "Silver moves in violent bursts. ATR expansion above MA signals trend initiation.",
        "params": {"atr_period": 14, "ma_period": 20, "atr_mult": 3.0},
        "trail_atr": 8.0,
    },
    # ── Fixed Income ──
    "TLT": {
        "name": "Rate Spike Fade",
        "tier": "bond",
        "rationale": "Bond selloffs (yield spikes) are overdone. 1.5-sigma below 50MA + RSI<30 = capitulation buy.",
        "params": {"ma_period": 50, "std_mult": 1.5, "rsi_period": 14, "rsi_oversold": 30},
        "trail_atr": 4.0,
    },
    # ── Crypto ──
    "BTC_USD": {
        "name": "Momentum + Volume Surge",
        "tier": "crypto",
        "rationale": "Bitcoin trends on high-volume breakouts. Volume > 1.5x avg + price > 20MA.",
        "params": {"ma_period": 20, "vol_mult": 1.5, "return_min": 0.02},
        "cash": 100_000,
        "trail_atr": 10.0,
    },
    "ETH_USD": {
        "name": "ETH/BTC Proxy Momentum",
        "tier": "crypto",
        "rationale": "ETH alpha comes from outperforming BTC. Trend + vol expansion = momentum trade.",
        "params": {"ma_period": 20, "atr_period": 14, "atr_mult": 4.0},
        "cash": 50_000,
        "trail_atr": 10.0,
    },
}

YF_TICKERS: dict[str, str] = {
    "BTC_USD": "BTC-USD",
    "ETH_USD": "ETH-USD",
}


def load_data(symbol: str) -> pd.DataFrame:
    path = Path(f"data/raw/{symbol}_daily.csv")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df.dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0 if col == "Volume" else df.iloc[:, 0]
    df.columns = [c.capitalize() for c in df.columns]
    return df


# ── Alpha Signal Generators ─────────────────────────────────────────────────


def alpha_spy_gap_fill(df: pd.DataFrame, params: dict) -> pd.Series:
    gap_pct = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    signal = pd.Series(0, index=df.index)
    signal[gap_pct < -params["gap_threshold"]] = 1
    return signal


def alpha_qqq_squeeze(df: pd.DataFrame, params: dict) -> pd.Series:
    close = df["Close"]
    sma = close.rolling(params["bb_period"]).mean()
    std = close.rolling(params["bb_period"]).std()
    bb_upper = sma + params["bb_std"] * std
    bb_width = (bb_upper - (sma - params["bb_std"] * std)) / sma
    squeeze_thresh = bb_width.rolling(252).quantile(params["squeeze_pct"] / 100)
    signal = pd.Series(0, index=df.index)
    signal[(bb_width < squeeze_thresh) & (close > bb_upper)] = 1
    return signal


def alpha_iwm_rsi(df: pd.DataFrame, params: dict) -> pd.Series:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / params["rsi_period"], adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / params["rsi_period"], adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    signal = pd.Series(0, index=df.index)
    signal[rsi < params["oversold"]] = 1
    return signal


def alpha_so_dip(df: pd.DataFrame, params: dict) -> pd.Series:
    ma = df["Close"].rolling(params["ma_period"]).mean()
    std = df["Close"].rolling(params["ma_period"]).std()
    lower = ma - params["std_mult"] * std
    signal = pd.Series(0, index=df.index)
    signal[df["Close"] < lower] = 1
    return signal


def alpha_jnj_lowvol(df: pd.DataFrame, params: dict) -> pd.Series:
    ret = df["Close"].pct_change()
    vol = ret.rolling(params["vol_period"]).std()
    vol_pct = vol.rolling(252).rank(pct=True)
    ma = df["Close"].rolling(params["ma_period"]).mean()
    signal = pd.Series(0, index=df.index)
    signal[(vol_pct < params["vol_percentile"] / 100) & (df["Close"] > ma)] = 1
    return signal


def alpha_xlv_oversold(df: pd.DataFrame, params: dict) -> pd.Series:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=params["rsi_period"], adjust=False).mean()
    avg_loss = loss.ewm(span=params["rsi_period"], adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    signal = pd.Series(0, index=df.index)
    signal[rsi < params["oversold"]] = 1
    return signal


def alpha_gld_trend(df: pd.DataFrame, params: dict) -> pd.Series:
    ma_fast = df["Close"].rolling(params["ma_fast"]).mean()
    ma_slow = df["Close"].rolling(params["ma_slow"]).mean()
    high = df["High"]
    low = df["Low"]
    tr = pd.concat(
        [
            high - low,
            (high - df["Close"].shift(1)).abs(),
            (low - df["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(span=params["adx_period"], adjust=False).mean()
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    atr_smooth = atr.replace(0, 1e-9)
    plus_di = 100 * plus_dm.ewm(span=params["adx_period"], adjust=False).mean() / atr_smooth
    minus_di = 100 * minus_dm.ewm(span=params["adx_period"], adjust=False).mean() / atr_smooth
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
    adx = dx.ewm(span=params["adx_period"], adjust=False).mean()
    signal = pd.Series(0, index=df.index)
    signal[(ma_fast > ma_slow) & (df["Close"] > ma_slow) & (adx > params["adx_min"])] = 1
    return signal


def alpha_slv_breakout(df: pd.DataFrame, params: dict) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(span=params["atr_period"], adjust=False).mean()
    atr_max = atr.rolling(100).max()
    ma = close.rolling(params["ma_period"]).mean()
    signal = pd.Series(0, index=df.index)
    signal[(atr > atr_max * 0.85) & (close > ma)] = 1
    return signal


def alpha_tlt_fade(df: pd.DataFrame, params: dict) -> pd.Series:
    close = df["Close"]
    ma = close.rolling(params["ma_period"]).mean()
    std = close.rolling(params["ma_period"]).std()
    lower = ma - params["std_mult"] * std
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=params["rsi_period"], adjust=False).mean()
    avg_loss = loss.ewm(span=params["rsi_period"], adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    signal = pd.Series(0, index=df.index)
    signal[(close < lower) & (rsi < params["rsi_oversold"])] = 1
    return signal


def alpha_btc_momentum(df: pd.DataFrame, params: dict) -> pd.Series:
    close = df["Close"]
    ma = close.rolling(params["ma_period"]).mean()
    vol_ma = df["Volume"].rolling(20).mean()
    ret = close.pct_change()
    signal = pd.Series(0, index=df.index)
    signal[
        (close > ma) & (df["Volume"] > vol_ma * params["vol_mult"]) & (ret > params["return_min"])
    ] = 1
    return signal


def alpha_eth_momentum(df: pd.DataFrame, params: dict) -> pd.Series:
    close = df["Close"]
    high, low = df["High"], df["Low"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(span=params["atr_period"], adjust=False).mean()
    ma = close.rolling(params["ma_period"]).mean()
    signal = pd.Series(0, index=df.index)
    signal[(close > ma) & (close > close.shift(5))] = 1
    return signal


ALPHA_GENERATORS: dict[str, Callable] = {
    "SPY": alpha_spy_gap_fill,
    "QQQ": alpha_qqq_squeeze,
    "IWM": alpha_iwm_rsi,
    "SO": alpha_so_dip,
    "JNJ": alpha_jnj_lowvol,
    "XLV": alpha_xlv_oversold,
    "GLD": alpha_gld_trend,
    "SLV": alpha_slv_breakout,
    "TLT": alpha_tlt_fade,
    "BTC_USD": alpha_btc_momentum,
    "ETH_USD": alpha_eth_momentum,
}


# ── Exit Rules ──────────────────────────────────────────────────────────────


def default_exit(
    df: pd.DataFrame, signal: pd.Series, atr_mult: float = 3.0, atr_period: int = 14
) -> pd.DataFrame:
    """Generic ATR trailing stop exit for all strategies."""
    return pd.DataFrame({"signal": signal})


# ── Backtest Runner ─────────────────────────────────────────────────────────


def run_alpha_backtest(
    symbol: str,
    df: pd.DataFrame,
    signal: pd.Series,
    cash: float = 10_000,
    atr_trail: float = 5.0,
    atr_period: int = 14,
) -> dict | None:
    from backtesting import Backtest, Strategy

    class AlphaStrategy(Strategy):
        entry_signal = signal
        trail_atr = atr_trail
        _atr_vals: list[float] = []

        def init(self):
            self._atr_vals = self.I(
                lambda: self._compute_atr_series(),
                name="atr",
                overlay=False,
            ).tolist()

        def _compute_atr_series(self):
            high = pd.Series(self.data.High)
            low = pd.Series(self.data.Low)
            close = pd.Series(self.data.Close)
            tr = pd.concat(
                [
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr = tr.ewm(span=atr_period, adjust=False).mean()
            return atr.values

        def next(self):
            i = len(self.data) - 1
            sig = self.entry_signal.iloc[i] if i < len(self.entry_signal) else 0

            if sig == 1 and not self.position:
                self.buy()
            elif self.position and len(self.trades) > 0:
                entry_bar = self.trades[-1].entry_bar
                if i > entry_bar:
                    atr_val = self._atr_vals[i] if i < len(self._atr_vals) else 0
                    highest_since_entry = max(self.data.Close[entry_bar : i + 1])
                    trail_stop = highest_since_entry - self.trail_atr * atr_val
                    if self.data.Close[-1] <= trail_stop:
                        self.position.close()

    try:
        bt = Backtest(
            df,
            AlphaStrategy,
            cash=cash,
            commission=0.001,
            exclusive_orders=True,
            finalize_trades=True,
        )
        stats = bt.run()
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

    return {
        "symbol": symbol,
        "alpha": ALPHA_DEFS[symbol]["name"],
        "tier": ALPHA_DEFS[symbol]["tier"],
        "return_pct": round(stats["Return [%]"], 2),
        "sharpe": round(stats["Sharpe Ratio"], 3) if stats["# Trades"] > 0 else 0.0,
        "max_dd_pct": round(stats["Max. Drawdown [%]"], 2),
        "trades": stats["# Trades"],
        "win_rate_pct": round(stats["Win Rate [%]"], 1) if stats["# Trades"] > 0 else 0.0,
        "profit_factor": round(stats["Profit Factor"], 2) if stats["# Trades"] > 0 else 0.0,
        "exposure_pct": round(stats["Exposure Time [%]"], 1),
    }


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Dig instrument-specific alphas")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-14")
    parser.add_argument("--symbols", help="Comma-separated symbols (default: all 10)")
    parser.add_argument("--cash", type=float, default=10_000)
    parser.add_argument("--atr-trail", type=float, default=3.0)
    parser.add_argument("--json-output", help="Save results to JSON")
    args = parser.parse_args()

    symbols = (
        [s.strip() for s in args.symbols.split(",")]
        if args.symbols
        else list(ALPHA_GENERATORS.keys())
    )

    print(f"\n{'=' * 70}")
    print(" INSTRUMENT-SPECIFIC ALPHA DISCOVERY")
    print(f" Period: {args.start} -> {args.end}")
    print(f" Instruments: {len(symbols)}")
    print(f"{'=' * 70}\n")

    results = []
    for symbol in symbols:
        if symbol not in ALPHA_GENERATORS:
            print(f"  SKIP {symbol}: no alpha defined")
            continue

        alpha_def = ALPHA_DEFS[symbol]
        gen = ALPHA_GENERATORS[symbol]
        params = alpha_def["params"]

        try:
            df = load_data(symbol)
        except FileNotFoundError:
            print(f"  SKIP {symbol}: no data file")
            continue

        mask = (df.index >= args.start) & (df.index <= args.end)
        df = df[mask]
        if len(df) < 60:
            print(f"  SKIP {symbol}: only {len(df)} bars")
            continue

        signal = gen(df, params)

        n_signals = signal.sum()
        sym_cash = ALPHA_DEFS[symbol].get("cash", args.cash)
        sym_trail = ALPHA_DEFS[symbol].get("trail_atr", args.atr_trail)
        if n_signals == 0:
            print(f"  {symbol:10s} {alpha_def['name']:30s} ZERO signals")
            results.append(
                {
                    "symbol": symbol,
                    "alpha": alpha_def["name"],
                    "tier": alpha_def["tier"],
                    "return_pct": 0,
                    "sharpe": 0,
                    "max_dd_pct": 0,
                    "trades": 0,
                    "win_rate_pct": 0,
                    "profit_factor": 0,
                    "exposure_pct": 0,
                }
            )
            continue

        print(
            f"  {symbol:10s} [{alpha_def['tier']:15s}] {alpha_def['name']:30s} "
            f"signals={n_signals} ...",
            end=" ",
            flush=True,
        )

        r = run_alpha_backtest(symbol, df, signal, cash=sym_cash, atr_trail=sym_trail)
        if r and "error" not in r:
            results.append(r)
            print(
                f"Sharpe={r['sharpe']:.3f}  Return={r['return_pct']:.1f}%  "
                f"Trades={r['trades']}  Win%={r['win_rate_pct']:.1f}"
            )
        elif r:
            print(f"ERROR: {r['error']}")
        else:
            print("FAILED")

    # ── Table ──
    if results:
        print(f"\n{'=' * 70}")
        print(f" RESULTS — {len(results)} instruments")
        print(f"{'=' * 70}")
        header = ["Symbol", "Alpha", "Tier", "Return%", "Sharpe", "MaxDD%", "Trades", "Win%", "PF"]
        cols = [
            "symbol",
            "alpha",
            "tier",
            "return_pct",
            "sharpe",
            "max_dd_pct",
            "trades",
            "win_rate_pct",
            "profit_factor",
        ]
        widths = [max(len(h), 8) for h in header]
        for r in results:
            for i, c in enumerate(cols):
                widths[i] = max(widths[i], len(str(r.get(c, ""))))

        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*header))
        print("  ".join("-" * w for w in widths))
        for r in sorted(results, key=lambda x: x.get("sharpe", -99), reverse=True):
            vals = []
            for c in cols:
                v = r.get(c, "")
                if c in ("sharpe", "profit_factor") and isinstance(v, float):
                    vals.append(f"{v:.3f}")
                elif isinstance(v, float):
                    vals.append(f"{v:.2f}")
                else:
                    vals.append(str(v))
            print(fmt.format(*vals))

        # ── Summary ──
        positive = [r for r in results if r.get("sharpe", 0) > 0]
        profitable = [r for r in results if r.get("return_pct", 0) > 0]
        traded = [r for r in results if r.get("trades", 0) > 0]

        print("\n--- Summary ---")
        print(f"Positive Sharpe: {len(positive)}/{len(results)}")
        print(f"Profitable: {len(profitable)}/{len(results)}")
        print(f"Traded (>0): {len(traded)}/{len(results)}")
        if results:
            avg_sharpe = sum(r.get("sharpe", 0) for r in results) / len(results)
            print(f"Avg Sharpe: {avg_sharpe:.3f}")

        # Tier breakdown
        tiers: dict[str, list] = {}
        for r in results:
            t = r.get("tier", "other")
            tiers.setdefault(t, []).append(r)
        print("\n--- By Tier ---")
        for tier, items in sorted(tiers.items()):
            avg_s = sum(r.get("sharpe", 0) for r in items) / len(items) if items else 0
            avg_r = sum(r.get("return_pct", 0) for r in items) / len(items) if items else 0
            avg_t = sum(r.get("trades", 0) for r in items) / len(items) if items else 0
            syms = ", ".join(r["symbol"] for r in items)
            print(
                f"  {tier:15s}: n={len(items)}  avgSharpe={avg_s:.3f}  "
                f"avgReturn={avg_r:.1f}%  avgTrades={avg_t:.1f}  [{syms}]"
            )

        # Per-alpha rationale
        print("\n--- Alpha Rationale ---")
        for r in sorted(results, key=lambda x: x.get("sharpe", -99), reverse=True):
            sym = r["symbol"]
            if sym in ALPHA_DEFS:
                print(f"  {sym:10s} [{r['alpha']:30s}] Sharpe={r.get('sharpe', 0):.3f}")
                print(f"            {ALPHA_DEFS[sym]['rationale']}")

        print(f"{'=' * 70}\n")

    if args.json_output:
        output = {
            "config": {"start": args.start, "end": args.end},
            "results": results,
        }
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(json.dumps(output, indent=2))
        print(f"Saved to {args.json_output}")


if __name__ == "__main__":
    main()
