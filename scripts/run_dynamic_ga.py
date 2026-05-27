#!/usr/bin/env python3
"""CLI for Dynamic GA with associative memory for regime-adaptive optimization.

Maintains separate GA populations per market regime (up/side/down) with
associative memory to recall good solutions when returning to a known regime.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.optimization.dynamic_ga import DynamicGAOptimizer, DynamicGAResult
from src.optimization.nsga2_optimizer import NSGA2Config, ParamDef

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")


def _load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    logger.info("Loading %s from %s to %s", ticker, start, end)
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    return df.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )


def _compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high, low, close = df["High"].values, df["Low"].values, df["Close"].values
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1)),
        ),
    )
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    return atr


def _generate_labels(df: pd.DataFrame, time_limit: int = 20) -> pd.Series:
    from src.ml.triple_barrier import TripleBarrierLabeler

    atr_series = _compute_atr(df)
    labeler = TripleBarrierLabeler(atr_mult_tp=2.0, atr_mult_sl=1.0)
    labels = labeler.fit(
        close=df["Close"],
        high=df["High"],
        low=df["Low"],
        time_limit=time_limit,
        atr_series=atr_series,
    )
    return labels


def _evaluate_strategy_slice(
    params: dict,
    close_slice: np.ndarray,
    labels_slice: np.ndarray,
) -> list[float]:
    entry_thresh = params.get("entry_threshold", 0.6)
    trail_stop = params.get("trail_stop", 2.0)
    exit_after = int(params.get("exit_after", 10))

    n = len(close_slice)
    trades: list[float] = []
    drawdowns: list[float] = []
    in_position = False
    entry_price = 0.0
    peak_price = 0.0
    bars_held = 0

    for i in range(n - 1):
        if not in_position:
            score = abs(float(labels_slice[i]))
            if score >= entry_thresh:
                in_position = True
                entry_price = float(close_slice[i])
                peak_price = entry_price
                bars_held = 0
        else:
            bars_held += 1
            current_price = float(close_slice[i])
            peak_price = max(peak_price, current_price)
            drawdown = (peak_price - current_price) / peak_price
            drawdowns.append(drawdown)

            exit_signal = drawdown >= trail_stop / 100.0 or bars_held >= exit_after
            if exit_signal or i == n - 2:
                trades.append((current_price - entry_price) / entry_price)
                in_position = False

    if not trades:
        return [0.0, 0.0, 0.0]

    roi = sum(trades)
    max_dd = max(drawdowns) if drawdowns else 0.0
    wins = sum(r for r in trades if r > 0)
    losses = abs(sum(r for r in trades if r < 0))
    profit_factor = wins / losses if losses > 0 else 2.0
    return [roi, max_dd, profit_factor]


def _make_params() -> list[ParamDef]:
    return [
        ParamDef("entry_threshold", 0.3, 0.9),
        ParamDef("trail_stop", 1.0, 5.0),
        ParamDef("exit_after", 3, 25, is_integer=True),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dynamic GA regime-adaptive parameter optimization"
    )
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument(
        "--population", type=int, default=30, help="Population size per regime (default: 30)"
    )
    parser.add_argument(
        "--generations", type=int, default=15, help="Generations per regime (default: 15)"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=252,
        help="Lookback window for regime detection (default: 252)",
    )
    parser.add_argument(
        "--step", type=int, default=21, help="Step size between optimizations (default: 21)"
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=10,
        help="Top-N elite solutions remembered per regime (default: 10)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = _load_data(args.ticker, args.start, args.end)
    labels = _generate_labels(df)
    prices = df["Close"].values
    labels_arr = labels.values

    param_defs = _make_params()
    config = NSGA2Config(
        population_size=args.population,
        generations=args.generations,
        seed=args.seed,
        objectives_minimize=[False, True, False],
    )

    def eval_fn(params: dict, regime: str) -> list[float]:
        n_lookback = min(args.window, len(prices))
        close_slice = prices[-n_lookback:]
        lab_slice = labels_arr[-n_lookback:]
        base = _evaluate_strategy_slice(params, close_slice, lab_slice)

        regime_bias = {"up": 0.05, "side": 0.0, "down": -0.05}
        base[0] += regime_bias.get(regime, 0.0)
        return base

    logger.info(
        "DynamicGA: pop=%d gen=%d window=%d step=%d mem=%d",
        config.population_size,
        config.generations,
        args.window,
        args.step,
        args.memory_size,
    )

    dga = DynamicGAOptimizer(
        param_defs,
        eval_fn,
        config,
        memory_size=args.memory_size,
    )
    result: DynamicGAResult = dga.fit(prices, window=args.window, step=args.step)

    print("\n" + "=" * 60)
    print("Dynamic GA — Regime-Adaptive Optimization Results")
    print("=" * 60)
    print(result.summary())
    print(f"\nRegime transitions detected: {len(result.transitions)}")
    for t in result.transitions[:10]:
        print(f"  {t['from']} -> {t['to']} at bar {t['bar']}")
    if len(result.transitions) > 10:
        print(f"  ... and {len(result.transitions) - 10} more")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"dynamic_ga_{args.ticker.lower()}_{timestamp}.json"

    serializable: dict = {
        "ticker": args.ticker,
        "config": {
            "population_size": config.population_size,
            "generations": config.generations,
            "window": args.window,
            "step": args.step,
            "memory_size": args.memory_size,
            "seed": config.seed,
        },
        "regime_results": {},
        "transitions": result.transitions,
    }
    for regime, nsga_result in result.regime_results.items():
        serializable["regime_results"][regime] = {
            "n_solutions": len(nsga_result.best_params),
            "solutions": [
                {
                    "params": p,
                    "objectives": [float(v) for v in o],
                }
                for p, o in zip(nsga_result.best_params, nsga_result.best_objectives)
            ],
            "generation_history": nsga_result.generation_history,
        }

    out_path.write_text(json.dumps(serializable, indent=2, default=str))
    logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()
