"""
Train and compare advanced RL algorithms (PPO, SAC, CQL) for trade execution.

Replaces the hand-rolled DQN agent with SB3 PPO/SAC and offline CQL.

Supports:
    - Training PPO and SAC on the TradeExecutionEnv
    - Building offline datasets from historical trade traces
    - Training CQL (Conservative Q-Learning) on offline data
    - Comparing all algorithms against the existing DQN baseline
    - Saving/loading models for backtesting integration

Usage:
    # Compare PPO, SAC, and CQL on SPY IS data
    uv run scripts/train_rl_advanced.py --symbol SPY --compare-all

    # Train PPO only (fast)
    uv run scripts/train_rl_advanced.py --symbol SPY --algo PPO --timesteps 50000

    # Train CQL on offline dataset
    uv run scripts/train_rl_advanced.py --symbol SPY --algo CQL --iterations 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_rl_advanced")

from src.rl.sb3_executors import (
    PPOTradeExecutor,
    SACTradeExecutor,
    RLTrainingResult,
    compare_rl_algorithms,
)
from src.rl.offline_rl import (
    CQLAgent,
    CQLTradeExecutor,
    CQLResult,
    OfflineDataset,
)

REPORTS_DIR = Path("reports/rl_advanced")
MODELS_DIR = Path("models/rl")


def load_spy_data(start: str = "2016-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Load SPY OHLCV and compute derived features."""
    logger.info(f"Fetching SPY {start} to {end}...")
    df = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def build_env_kwargs(df: pd.DataFrame, entry_threshold: float = 0.50) -> dict:
    """Build environment kwargs from SPY DataFrame.

    Computes proxy signals since we don't always have a trained ML model:
        - probabilities: synthetic from close/MA cross
        - vol_regime: ATR(20)/ATR(100) ratio
        - atr_pct: ATR(14)/close
        - sentiment: normalized 5-day return
        - event_days: FOMC-like simulation (every ~6 weeks)
    """
    close = df["Close"].values.astype(np.float32)
    high = df["High"].values.astype(np.float32)
    low = df["Low"].values.astype(np.float32)
    n = len(close)

    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
    tr = np.insert(tr, 0, np.nan)

    def rolling_mean(arr, window):
        result = np.full(len(arr), np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1 : i + 1])
        return result

    atr20 = np.full(n, np.nan)
    atr100 = np.full(n, np.nan)
    atr14 = np.full(n, np.nan)
    for i in range(19, n):
        atr20[i] = np.mean(tr[i - 19 : i + 1])
    for i in range(99, n):
        atr100[i] = np.mean(tr[i - 99 : i + 1])
    for i in range(13, n):
        atr14[i] = np.mean(tr[i - 13 : i + 1])

    vol_regime = atr20 / np.maximum(atr100, 1e-6)
    atr_pct = atr14 / np.maximum(close, 1e-6)

    ma20 = rolling_mean(close, 20)
    ma50 = rolling_mean(close, 50)
    probabilities = np.full(n, 0.45)
    for i in range(50, n):
        probabilities[i] = 0.55 if close[i] > ma20[i] and ma20[i] > ma50[i] else 0.30

    sentiment = np.zeros(n)
    for i in range(5, n):
        sentiment[i] = float(np.tanh((close[i] / close[i - 5] - 1.0) * 20))

    event_days = np.zeros(n, dtype=bool)
    event_days[np.arange(0, n, 30)] = True

    probabilities = np.clip(np.nan_to_num(probabilities, nan=0.45), 0.0, 1.0)
    vol_regime = np.clip(np.nan_to_num(vol_regime, nan=1.0), 0.0, 10.0)
    atr_pct = np.clip(np.nan_to_num(atr_pct, nan=0.02), 0.0, 1.0)
    sentiment = np.clip(np.nan_to_num(sentiment, nan=0.0), -1.0, 1.0)

    return {
        "probabilities": probabilities.astype(np.float32),
        "vol_regime": vol_regime.astype(np.float32),
        "atr_pct": atr_pct.astype(np.float32),
        "sentiment": sentiment.astype(np.float32),
        "prices": close.astype(np.float32),
        "event_days": event_days,
        "entry_threshold": entry_threshold,
        "max_episode_steps": 50,
    }


def train_ppo(env_kwargs: dict, timesteps: int, model_path: str) -> RLTrainingResult:
    logger.info(f"Training PPO for {timesteps} timesteps...")
    ppo = PPOTradeExecutor()
    result = ppo.train(env_kwargs, total_timesteps=timesteps, verbose=1)
    ppo.save_model(model_path + "_ppo")
    return result


def train_sac(env_kwargs: dict, timesteps: int, model_path: str) -> RLTrainingResult:
    logger.info(f"Training SAC for {timesteps} timesteps...")
    sac = SACTradeExecutor()
    result = sac.train(env_kwargs, total_timesteps=timesteps, verbose=1)
    sac.save_model(model_path + "_sac")
    return result


def train_cql(env_kwargs: dict, iterations: int, model_path: str) -> CQLResult:
    logger.info("Building offline dataset...")
    dataset = OfflineDataset.from_env_trajectories(env_kwargs, n_episodes=500, policy="threshold")
    logger.info(f"Dataset: {len(dataset)} transitions")

    logger.info(f"Training CQL for {iterations} iterations...")
    cql = CQLAgent(cql_alpha=5.0, gamma=0.99, lr=3e-4)
    result = cql.train(dataset, n_iterations=iterations, batch_size=256)
    cql.save_model(model_path + "_cql")
    dataset.save(model_path + "_cql_dataset.json")
    return result


def compare_all(
    env_kwargs: dict,
    timesteps: int,
    cql_iterations: int,
    model_dir: str,
) -> Dict:
    """Run PPO, SAC, CQL, and DQN comparison."""
    results = {"ppo": None, "sac": None, "cql": None}

    logger.info("=" * 60)
    logger.info("Training PPO...")
    ppo_result = train_ppo(env_kwargs, timesteps, f"{model_dir}/spy")
    results["ppo"] = {
        "mean_reward": ppo_result.mean_reward,
        "std_reward": ppo_result.std_reward,
        "episode_length": ppo_result.mean_episode_length,
    }

    logger.info("=" * 60)
    logger.info("Training SAC...")
    sac_result = train_sac(env_kwargs, timesteps, f"{model_dir}/spy")
    results["sac"] = {
        "mean_reward": sac_result.mean_reward,
        "std_reward": sac_result.std_reward,
        "episode_length": sac_result.mean_episode_length,
    }

    logger.info("=" * 60)
    logger.info("Training CQL...")
    cql_result = train_cql(env_kwargs, cql_iterations, f"{model_dir}/spy")
    results["cql"] = {
        "cql_alpha": cql_result.cql_alpha,
        "n_iterations": cql_result.n_iterations,
        "n_transitions": cql_result.n_transitions,
        "final_qf_loss": cql_result.final_qf_loss,
        "final_bellman_loss": cql_result.final_bellman_loss,
    }

    logger.info("=" * 60)
    logger.info("Comparison Results:")
    logger.info(
        f"  PPO: reward={results['ppo']['mean_reward']:.4f} len={results['ppo']['episode_length']:.1f}"
    )
    logger.info(
        f"  SAC: reward={results['sac']['mean_reward']:.4f} len={results['sac']['episode_length']:.1f}"
    )
    logger.info(
        f"  CQL: QF={cql_result.final_qf_loss:.4f} Bellman={cql_result.final_bellman_loss:.4f}"
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train advanced RL algorithms (PPO, SAC, CQL) for trade execution"
    )
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument(
        "--algo",
        default="all",
        choices=["all", "PPO", "SAC", "CQL", "compare"],
        help="Algorithm to train (default: all)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Training timesteps for PPO/SAC (default: 50000)",
    )
    parser.add_argument(
        "--iterations", type=int, default=10_000, help="CQL gradient steps (default: 10000)"
    )
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=0.50,
        help="Entry threshold for synthetic signals (default: 0.50)",
    )
    parser.add_argument(
        "--output-dir", default="models/rl", help="Model output directory (default: models/rl)"
    )
    parser.add_argument(
        "--compare-all", action="store_true", help="Compare all algorithms and save results"
    )
    parser.add_argument(
        "--json-output", action="store_true", help="Print results as JSON to stdout"
    )

    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    df = load_spy_data()
    env_kwargs = build_env_kwargs(df, args.entry_threshold)
    logger.info(f"Built env with {len(env_kwargs['prices'])} bars")

    if args.algo == "all" or args.compare_all:
        results = compare_all(env_kwargs, args.timesteps, args.iterations, args.output_dir)

        report_path = REPORTS_DIR / f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.write_text(json.dumps(results, indent=2, default=str))
        logger.info(f"Report saved to {report_path}")

        if args.json_output:
            print(json.dumps(results, indent=2, default=str))
        return

    model_path = f"{args.output_dir}/{args.symbol.lower()}"

    if args.algo == "PPO":
        result = train_ppo(env_kwargs, args.timesteps, model_path)
        if args.json_output:
            print(json.dumps(result.to_dict(), indent=2, default=str))

    elif args.algo == "SAC":
        result = train_sac(env_kwargs, args.timesteps, model_path)
        if args.json_output:
            print(json.dumps(result.to_dict(), indent=2, default=str))

    elif args.algo == "CQL":
        result = train_cql(env_kwargs, args.iterations, model_path)
        if args.json_output:
            print(json.dumps(result.to_dict(), indent=2, default=str))

    elif args.algo == "compare":
        results = compare_all(env_kwargs, args.timesteps, args.iterations, args.output_dir)
        if args.json_output:
            print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
