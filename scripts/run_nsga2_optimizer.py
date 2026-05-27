#!/usr/bin/env python3
"""CLI for NSGA2 multi-objective parameter optimization.

Optimizes trading strategy parameters across multiple conflicting objectives:
  - Maximize return
  - Minimize max drawdown
  - Maximize profit factor

Uses triple-barrier labels and features from FeatureExtractor as the
evaluation environment.
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

from src.optimization.nsga2_optimizer import NSGA2Config, NSGA2Optimizer, ParamDef

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


def _generate_labels(df: pd.DataFrame, time_limit: int = 20) -> pd.Series:
    from src.ml.triple_barrier import TripleBarrierLabeler

    atr_series = _compute_atr(df, period=14)
    labeler = TripleBarrierLabeler(atr_mult_tp=2.0, atr_mult_sl=1.0)
    labels = labeler.fit(
        close=df["Close"],
        high=df["High"],
        low=df["Low"],
        time_limit=time_limit,
        atr_series=atr_series,
    )
    return labels


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


def _evaluate_strategy(
    params: dict,
    df: pd.DataFrame,
    labels: pd.Series,
) -> list[float]:
    entry_thresh = params.get("entry_threshold", 0.6)
    trail_stop = params.get("trail_stop", 2.0)
    exit_after = int(params.get("exit_after", 10))

    signals = labels.shift(-1)
    signals = signals.fillna(0)

    close = df["Close"].values
    n = len(close)
    trades: list[float] = []
    drawdowns: list[float] = []
    in_position = False
    entry_price = 0.0
    peak_price = 0.0
    bars_held = 0

    for i in range(n - 1):
        if not in_position:
            score = abs(float(signals.iloc[i]))
            if score >= entry_thresh:
                in_position = True
                entry_price = float(close[i])
                peak_price = entry_price
                bars_held = 0
        else:
            bars_held += 1
            current_price = float(close[i])
            peak_price = max(peak_price, current_price)
            drawdown = (peak_price - current_price) / peak_price
            drawdowns.append(drawdown)

            exit_signal = False
            if drawdown >= trail_stop / 100.0:
                exit_signal = True
            if bars_held >= exit_after:
                exit_signal = True

            if exit_signal or i == n - 2:
                ret = (current_price - entry_price) / entry_price
                trades.append(ret)
                in_position = False

    if not trades:
        return [0.0, 0.0, 0.0]

    roi = sum(trades)
    max_dd = max(drawdowns) if drawdowns else 0.0
    profit_factor = _profit_factor(trades)
    return [roi, max_dd, profit_factor]


def _profit_factor(trades: list[float]) -> float:
    wins = sum(r for r in trades if r > 0)
    losses = abs(sum(r for r in trades if r < 0))
    return wins / losses if losses > 0 else 2.0


def _score_fn(
    params: dict,
    df: pd.DataFrame,
    labels: pd.Series,
) -> list[float]:
    return _evaluate_strategy(params, df, labels)


def _make_params() -> list[ParamDef]:
    return [
        ParamDef("entry_threshold", 0.3, 0.9),
        ParamDef("trail_stop", 1.0, 5.0),
        ParamDef("exit_after", 3, 25, is_integer=True),
        ParamDef("use_rsi", 0, 1, is_binary=True),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="NSGA2 multi-objective parameter optimization")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--population", type=int, default=30, help="Population size (default: 30)")
    parser.add_argument(
        "--generations", type=int, default=20, help="Number of generations (default: 20)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = _load_data(args.ticker, args.start, args.end)
    labels = _generate_labels(df)

    param_defs = _make_params()
    config = NSGA2Config(
        population_size=args.population,
        generations=args.generations,
        seed=args.seed,
        objectives_minimize=[False, True, False, True],
    )

    wrapped_eval = lambda p: _score_fn(p, df, labels)  # noqa: E731

    logger.info(
        "NSGA2: pop=%d gen=%d params=%d objectives=%d",
        config.population_size,
        config.generations,
        len(param_defs),
        len(config.objectives_minimize),
    )

    nsga = NSGA2Optimizer(param_defs, wrapped_eval, config)
    result = nsga.optimize()

    print("\n" + "=" * 60)
    print("NSGA-II Pareto-Optimal Results")
    print("=" * 60)
    print(result.summary())
    print(f"\nPareto front size: {len(result.best_params)}")
    for i, (params, objs) in enumerate(zip(result.best_params, result.best_objectives)):
        if i >= 5:
            print(f"  ... and {len(result.best_params) - 5} more solutions")
            break
        print(f"  [{i}] {params}")
        print(f"      objectives={[round(v, 4) for v in objs]}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"nsga2_{args.ticker.lower()}_{timestamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "ticker": args.ticker,
                "config": {
                    "population_size": config.population_size,
                    "generations": config.generations,
                    "seed": config.seed,
                },
                "pareto_solutions": [
                    {
                        "params": p,
                        "objectives": [float(v) for v in o],
                    }
                    for p, o in zip(result.best_params, result.best_objectives)
                ],
                "generation_history": result.generation_history,
            },
            indent=2,
            default=str,
        )
    )
    logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()
