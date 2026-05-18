"""
D5: PPO/SAC trade execution via stable-baselines3.

Replaces the hand-rolled DQN agent with production-grade SB3 algorithms.
PPO (Proximal Policy Optimization) and SAC (Soft Actor-Critic) are SOTA
on-policy and off-policy algorithms respectively.

PPO advantages: stable training, good sample efficiency, widely used in finance.
SAC advantages: maximum entropy exploration, continuous action variants possible,
                better in stochastic environments.

Both wrap the existing TradeExecutionEnv (8-dim state, 3 discrete actions)
and expose inference callbacks for integration into the backtesting loop.

Architecture:
    PPOTradeExecutor: SB3 PPO training + inference for backtesting
    SACTradeExecutor: SB3 SAC training + inference for backtesting
    SB3Executor: Base class with shared logic (save/load/evaluate/comparison)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.rl.trade_execution_env import TradeExecutionEnv

logger = logging.getLogger(__name__)


@dataclass
class RLTrainingResult:
    """Results from training an RL agent on the trade execution env."""

    algorithm: str
    total_timesteps: int
    mean_reward: float
    std_reward: float
    mean_episode_length: float
    n_eval_episodes: int
    train_info: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "algorithm": self.algorithm,
            "total_timesteps": self.total_timesteps,
            "mean_reward": round(self.mean_reward, 6),
            "std_reward": round(self.std_reward, 6),
            "mean_episode_length": round(self.mean_episode_length, 1),
            "n_eval_episodes": self.n_eval_episodes,
        }


# ── Base SB3 Executor ────────────────────────────────────────────────────


class SB3Executor:
    """Shared base for SB3-based RL trade executors.

    Args:
        env_kwargs: Passed through to TradeExecutionEnv.__init__.
        tb_log_dir: TensorBoard log directory (optional).
    """

    def __init__(self, tb_log_dir: Optional[str] = None):
        self.model: Optional[Union[PPO, SAC]] = None
        self._tb_log_dir = tb_log_dir

    @staticmethod
    def _make_env(env_kwargs: dict) -> TradeExecutionEnv:
        return TradeExecutionEnv(**env_kwargs)

    @staticmethod
    def _make_vec_env(env_kwargs: dict, n_envs: int = 4) -> DummyVecEnv:
        def _init():
            return TradeExecutionEnv(**env_kwargs)

        return DummyVecEnv([_init for _ in range(n_envs)])

    def save_model(self, path: Union[str, Path]) -> None:
        if self.model is None:
            raise ValueError("No model to save. Call train() first.")
        self.model.save(str(path))
        logger.info(f"Model saved to {path}")

    def load_model(self, path: Union[str, Path]) -> None:
        path_str = str(path)
        if path_str.endswith("_ppo"):
            self.model = PPO.load(path_str)
        elif path_str.endswith("_sac"):
            self.model = SAC.load(path_str)
        else:
            raise ValueError(f"Unknown model type for path: {path_str}")
        logger.info(f"Model loaded from {path}")

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Infer action from observation. For backtesting integration.

        Args:
            observation: 8-dim observation array.
            deterministic: Use greedy action (True) or sample.

        Returns:
            Discrete action: 0=HOLD, 1=ENTER, 2=EXIT.
        """
        if self.model is None:
            return 0
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action[0])


# ── PPO Executor ──────────────────────────────────────────────────────────


class PPOTradeExecutor(SB3Executor):
    """PPO agent for trade execution decisions.

    Trains PPO on the existing TradeExecutionEnv and provides
    inference for backtesting integration.

    Args:
        learning_rate: PPO learning rate.
        n_steps: Steps per update (rollout buffer size).
        batch_size: Mini-batch size.
        n_epochs: Optimization epochs per update.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_range: PPO clipping parameter.
        ent_coef: Entropy coefficient for exploration.
        policy_kwargs: Additional policy kwargs (e.g., net_arch).
        tensorboard_log: TensorBoard log directory.
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        policy_kwargs: Optional[Dict] = None,
        tensorboard_log: Optional[str] = None,
    ):
        super().__init__(tb_log_dir=tensorboard_log)
        self._lr = learning_rate
        self._n_steps = n_steps
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._clip_range = clip_range
        self._ent_coef = ent_coef
        self._policy_kwargs = policy_kwargs or {"net_arch": dict(pi=[64, 64], vf=[64, 64])}

    def train(
        self,
        env_kwargs: dict,
        total_timesteps: int = 100_000,
        n_eval_episodes: int = 100,
        eval_freq: int = 10_000,
        verbose: int = 1,
    ) -> RLTrainingResult:
        """Train PPO on the trade execution environment.

        Args:
            env_kwargs: Keyword arguments for TradeExecutionEnv.
            total_timesteps: Total timesteps to train.
            n_eval_episodes: Number of episodes for evaluation.
            eval_freq: Evaluate every N timesteps.
            verbose: SB3 verbosity (0=none, 1=info).

        Returns:
            RLTrainingResult with training metrics.
        """
        env = self._make_env(env_kwargs)

        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self._lr,
            n_steps=self._n_steps,
            batch_size=self._batch_size,
            n_epochs=self._n_epochs,
            gamma=self._gamma,
            gae_lambda=self._gae_lambda,
            clip_range=self._clip_range,
            ent_coef=self._ent_coef,
            policy_kwargs=self._policy_kwargs,
            tensorboard_log=self._tb_log_dir,
            verbose=verbose,
        )

        eval_callback = EvalCallback(
            env,
            best_model_save_path=None,
            log_path=None,
            eval_freq=eval_freq,
            n_eval_episodes=min(n_eval_episodes, 50),
            deterministic=True,
        )

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=verbose > 0,
        )

        mean_reward, std_reward = evaluate_policy(
            self.model, env, n_eval_episodes=n_eval_episodes, deterministic=True
        )

        info = self.model.logger.name_to_value if hasattr(self.model, "logger") else {}

        return RLTrainingResult(
            algorithm="PPO",
            total_timesteps=total_timesteps,
            mean_reward=float(mean_reward),
            std_reward=float(std_reward),
            mean_episode_length=self._estimate_episode_length(env_kwargs),
            n_eval_episodes=n_eval_episodes,
            train_info={k: float(v) for k, v in info.items() if isinstance(v, (int, float))},
        )

    def _estimate_episode_length(self, env_kwargs: dict) -> float:
        env = self._make_env(env_kwargs)
        lengths = []
        for _ in range(50):
            obs, _ = env.reset()
            for step in range(env_kwargs.get("max_episode_steps", 50)):
                if self.model is None:
                    break
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    lengths.append(step + 1)
                    break
        return float(np.mean(lengths)) if lengths else 0.0


# ── SAC Executor ──────────────────────────────────────────────────────────


class SACTradeExecutor(SB3Executor):
    """SAC agent for trade execution decisions.

    SAC uses entropy maximization for automatic exploration tuning.
    Well-suited for financial environments with stochastic reward distributions.

    Args:
        learning_rate: SAC learning rate.
        buffer_size: Replay buffer size.
        batch_size: Mini-batch size.
        tau: Target network update rate.
        gamma: Discount factor.
        ent_coef: Entropy regularization coefficient ('auto' for automatic tuning).
        policy_kwargs: Additional policy kwargs.
        tensorboard_log: TensorBoard log directory.
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        ent_coef: Union[str, float] = "auto",
        policy_kwargs: Optional[Dict] = None,
        tensorboard_log: Optional[str] = None,
    ):
        super().__init__(tb_log_dir=tensorboard_log)
        self._lr = learning_rate
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._tau = tau
        self._gamma = gamma
        self._ent_coef = ent_coef
        self._policy_kwargs = policy_kwargs or {"net_arch": dict(pi=[64, 64], qf=[64, 64])}

    def train(
        self,
        env_kwargs: dict,
        total_timesteps: int = 100_000,
        n_eval_episodes: int = 100,
        eval_freq: int = 10_000,
        verbose: int = 1,
    ) -> RLTrainingResult:
        """Train SAC on the trade execution environment.

        Args:
            env_kwargs: Keyword arguments for TradeExecutionEnv.
            total_timesteps: Total timesteps to train.
            n_eval_episodes: Number of episodes for evaluation.
            eval_freq: Evaluate every N timesteps.
            verbose: SB3 verbosity.

        Returns:
            RLTrainingResult with training metrics.
        """
        env = self._make_env(env_kwargs)

        self.model = SAC(
            "MlpPolicy",
            env,
            learning_rate=self._lr,
            buffer_size=self._buffer_size,
            batch_size=self._batch_size,
            tau=self._tau,
            gamma=self._gamma,
            ent_coef=self._ent_coef,
            policy_kwargs=self._policy_kwargs,
            tensorboard_log=self._tb_log_dir,
            verbose=verbose,
        )

        eval_callback = EvalCallback(
            env,
            best_model_save_path=None,
            log_path=None,
            eval_freq=eval_freq,
            n_eval_episodes=min(n_eval_episodes, 50),
            deterministic=True,
        )

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=verbose > 0,
        )

        mean_reward, std_reward = evaluate_policy(
            self.model, env, n_eval_episodes=n_eval_episodes, deterministic=True
        )

        info = self.model.logger.name_to_value if hasattr(self.model, "logger") else {}

        return RLTrainingResult(
            algorithm="SAC",
            total_timesteps=total_timesteps,
            mean_reward=float(mean_reward),
            std_reward=float(std_reward),
            mean_episode_length=self._estimate_episode_length(env_kwargs),
            n_eval_episodes=n_eval_episodes,
            train_info={k: float(v) for k, v in info.items() if isinstance(v, (int, float))},
        )

    def _estimate_episode_length(self, env_kwargs: dict) -> float:
        env = self._make_env(env_kwargs)
        lengths = []
        for _ in range(50):
            obs, _ = env.reset()
            for step in range(env_kwargs.get("max_episode_steps", 50)):
                if self.model is None:
                    break
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    lengths.append(step + 1)
                    break
        return float(np.mean(lengths)) if lengths else 0.0


# ── Comparison Utility ────────────────────────────────────────────────────


def compare_rl_algorithms(
    env_kwargs: dict,
    total_timesteps: int = 50_000,
    n_eval_episodes: int = 100,
    algorithms: Optional[List[str]] = None,
) -> List[RLTrainingResult]:
    """Train and compare PPO, SAC on the same environment.

    Args:
        env_kwargs: Keyword arguments for TradeExecutionEnv.
        total_timesteps: Training budget per algorithm.
        n_eval_episodes: Evaluation episodes.
        algorithms: List of algorithms to compare (default: ["PPO", "SAC"]).

    Returns:
        List of RLTrainingResult, one per algorithm.
    """
    if algorithms is None:
        algorithms = ["PPO", "SAC"]

    results = []

    if "PPO" in algorithms:
        logger.info("Training PPO...")
        ppo = PPOTradeExecutor()
        ppo_result = ppo.train(
            env_kwargs,
            total_timesteps=total_timesteps,
            n_eval_episodes=n_eval_episodes,
            verbose=0,
        )
        results.append(ppo_result)
        logger.info(f"PPO mean reward: {ppo_result.mean_reward:.4f} ± {ppo_result.std_reward:.4f}")

    if "SAC" in algorithms:
        logger.info("Training SAC...")
        sac = SACTradeExecutor()
        sac_result = sac.train(
            env_kwargs,
            total_timesteps=total_timesteps,
            n_eval_episodes=n_eval_episodes,
            verbose=0,
        )
        results.append(sac_result)
        logger.info(f"SAC mean reward: {sac_result.mean_reward:.4f} ± {sac_result.std_reward:.4f}")

    return results
