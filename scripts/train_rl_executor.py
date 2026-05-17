"""
Train RL trade execution agent on historical IS data.

Loads SPY OHLCV data, extracts features, builds the TradeExecutionEnv,
trains a DQN agent, and saves the model for backtesting inference.

Usage:
    uv run scripts/train_rl_executor.py
    uv run scripts/train_rl_executor.py --symbol SPY --episodes 2000
    uv run scripts/train_rl_executor.py --symbol SPY --start 2016-05-12 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rl.trade_execution_env import TradeExecutionEnv
from src.rl.rl_trade_executor import RLTradeExecutor
from src.ml.feature_engineering import FeatureExtractor
from src.ml.pattern_classifier import PatternClassifier


def load_data(symbol: str) -> pd.DataFrame:
    path = Path(f"data/raw/{symbol}_daily.csv")
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol} at {path}")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df.dropna()
    return df


def build_env(
    symbol: str,
    model_path: str,
    start: str | None = None,
    end: str | None = None,
    sentiment_seed: int = 42,
) -> TradeExecutionEnv:
    """Build the gym environment from historical data.

    Loads data, runs the ML model to get probabilities, extracts features,
    and constructs the TradeExecutionEnv.
    """
    df = load_data(symbol)
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]

    ohlcv = df.rename(
        columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        }
    )

    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(ohlcv, include_forward_returns=False)

    # Load model and compute probabilities
    model = PatternClassifier()
    model.load(model_path)
    feature_names = list(model.feature_names_)
    present = [c for c in feature_names if c in features.columns]
    fx = features[present].ffill().bfill().fillna(0)
    for c in feature_names:
        if c not in fx.columns:
            fx[c] = 0.0
    fx = fx[feature_names]
    preds = model.predict(fx)
    probs = preds["probability_profitable"].fillna(0.5).to_numpy(dtype=np.float32)

    # Volatility regime
    if "vol_regime" in features.columns:
        vol_regime = features["vol_regime"].ffill().bfill().fillna(1.0).to_numpy(dtype=np.float32)
    else:
        vol_regime = np.ones(len(fx), dtype=np.float32)

    # ATR as percentage of price
    high, low, close = ohlcv["High"], ohlcv["Low"], ohlcv["Close"]
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_pct = (atr / close.shift(1)).fillna(0.01).clip(0, 1).to_numpy(dtype=np.float32)

    # Synthetic sentiment
    from src.signals.sentiment_scorer import SyntheticSentimentProvider

    provider = SyntheticSentimentProvider(seed=sentiment_seed)
    sentiment_raw = provider.get_sentiment(
        dates=pd.DatetimeIndex(ohlcv.index),
        symbol=symbol,
    )
    sentiment = sentiment_raw.to_numpy(dtype=np.float32)

    # Event days (placeholder — empty if no calendar)
    event_days = np.zeros(len(fx), dtype=bool)

    # Prices
    prices = close.bfill().ffill().to_numpy(dtype=np.float32)

    return TradeExecutionEnv(
        probabilities=probs,
        vol_regime=vol_regime,
        atr_pct=atr_pct,
        sentiment=sentiment,
        prices=prices,
        event_days=event_days,
        entry_threshold=0.50,
        max_episode_steps=50,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL trade execution agent")
    parser.add_argument("--symbol", type=str, default="SPY", help="Ticker symbol for training")
    parser.add_argument(
        "--model",
        type=str,
        default="models/pattern_classifier_v3_SPY_20260514_195235.pkl",
        help="Path to trained ML model for probabilities",
    )
    parser.add_argument("--start", type=str, default="2016-05-12", help="Start date (YYYY-MM-DD)")
    parser.add_argument(
        "--end", type=str, default="2024-12-31", help="End date (YYYY-MM-DD) — IS only"
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--output",
        type=str,
        default="models/rl_executor_dqn.pt",
        help="Output path for trained model",
    )
    parser.add_argument("--device", type=str, default=None, help="Device: cpu or cuda")
    args = parser.parse_args()

    print(f"Building environment for {args.symbol} ({args.start} -> {args.end})...")
    env = build_env(
        symbol=args.symbol,
        model_path=args.model,
        start=args.start,
        end=args.end,
    )

    print(f"Environment: {env._n_bars} bars, {int(env._event_days.sum())} event days")

    executor = RLTradeExecutor(model_path=None, device=args.device)
    executor._agent.epsilon = 1.0
    executor._agent.epsilon_decay = 0.995
    executor._agent.optimizer = type(executor._agent.optimizer)(
        executor._agent.policy_net.parameters(), lr=args.lr
    )

    print(f"\nTraining DQN agent ({args.episodes} episodes)...")
    metrics = executor.train(env, n_episodes=args.episodes, verbose=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    executor.save(output_path)
    print(f"\nModel saved to {output_path}")

    print("\nTraining Results:")
    print(f"  Episodes: {metrics['n_episodes']}")
    print(f"  Final avg reward (last 100): {metrics['final_avg_reward']:.4f}")
    print(f"  Final epsilon: {metrics['final_epsilon']:.4f}")
    if metrics["loss_history"]:
        print(f"  Final loss: {metrics['loss_history'][-1]:.6f}")
        print(f"  Mean loss (last 100): {np.mean(metrics['loss_history'][-100:]):.6f}")


if __name__ == "__main__":
    main()
