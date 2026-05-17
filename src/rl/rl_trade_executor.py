"""RL-based trade execution agent using DQN with experience replay.

Implements a Double DQN agent in pure PyTorch (no stable-baselines3 dependency)
for deciding optimal entry/exit timing in the backtesting loop.

Architecture:
    DQNAgent: PyTorch neural network + experience replay buffer + epsilon-greedy
    RLTradeExecutor: High-level wrapper for inference during backtesting

Training: IS data only. Evaluation: OOS data. Walk-forward: train on expanding
window, evaluate on next window.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

if TYPE_CHECKING:
    from src.rl.trade_execution_env import TradeExecutionEnv

logger = logging.getLogger(__name__)

NUM_FEATURES = 8
NUM_ACTIONS = 3  # 0=HOLD, 1=ENTER_LONG, 2=EXIT_LONG


class DQNNetwork(nn.Module):
    """Simple feedforward Q-network for DQN."""

    def __init__(self, input_dim: int = NUM_FEATURES, output_dim: int = NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Fixed-size circular replay buffer for experience replay."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self._rng = np.random.RandomState(42)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        indices = self._rng.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Double DQN agent with epsilon-greedy exploration.

    Trains on historical IS data. The trained model can be saved/loaded for
    inference during backtesting.
    """

    def __init__(
        self,
        input_dim: int = NUM_FEATURES,
        output_dim: int = NUM_ACTIONS,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str | None = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self._train_step_count = 0

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQNNetwork(input_dim, output_dim).to(self.device)
        self.target_net = DQNNetwork(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.loss_history: list[float] = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Observation array, shape (input_dim,)
            training: If True, use epsilon-greedy. If False, use greedy.

        Returns:
            Action index: 0=HOLD, 1=ENTER, 2=EXIT
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.output_dim)

        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def update(self) -> float | None:
        """Perform one DQN update step (Double DQN with target network).

        Returns:
            Loss value, or None if buffer too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).unsqueeze(1).to(self.device)

        # Current Q values
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # Double DQN: select action with policy net, evaluate with target net
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.functional.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        loss_val = float(loss.item())
        self.loss_history.append(loss_val)

        self._train_step_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Update target network
        if self._train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss_val

    def save(self, path: str | Path) -> None:
        """Save policy network and training state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_step_count": self._train_step_count,
            "loss_history": self.loss_history,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }
        torch.save(state, path)

    def load(self, path: str | Path) -> None:
        """Load policy network and training state."""
        path = Path(path)
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(state["policy_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epsilon = state["epsilon"]
        self._train_step_count = state["train_step_count"]
        self.loss_history = state.get("loss_history", [])
        self.input_dim = state["input_dim"]
        self.output_dim = state["output_dim"]

    def to_inference_mode(self) -> None:
        """Switch to inference (no gradient, no exploration)."""
        self.policy_net.eval()
        self.epsilon = 0.0


class RLTradeExecutor:
    """High-level wrapper for RL-based trade execution in backtesting.

    Manages training and inference for the DQN agent. During backtesting,
    provides per-bar decision: should we enter, exit, or hold?

    Train on IS data, evaluate on OOS data. Walk-forward training:
    split IS into train/val windows, train on expanding window.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
    ):
        self._agent = DQNAgent(device=device)
        self._is_trained = False
        self._model_path = None

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train(
        self,
        env: "TradeExecutionEnv",
        n_episodes: int = 1000,
        log_interval: int = 100,
        verbose: bool = True,
    ) -> dict:
        """Train the DQN agent on the gym environment.

        Args:
            env: TradeExecutionEnv instance
            n_episodes: Number of training episodes
            log_interval: Print progress every N episodes
            verbose: Whether to print progress

        Returns:
            Training metrics dict
        """
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []

        for episode in range(n_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0.0
            steps = 0

            while not done:
                action = self._agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self._agent.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                steps += 1

                loss = self._agent.update()

            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)

            if verbose and (episode + 1) % log_interval == 0:
                avg_reward = np.mean(episode_rewards[-log_interval:])
                avg_steps = np.mean(episode_lengths[-log_interval:])
                logger.info(
                    f"Episode {episode + 1}/{n_episodes} | "
                    f"Avg Reward: {avg_reward:.4f} | "
                    f"Avg Steps: {avg_steps:.1f} | "
                    f"Epsilon: {self._agent.epsilon:.3f} | "
                    f"Buffer: {len(self._agent.replay_buffer)}"
                )

        self._is_trained = True
        return {
            "n_episodes": n_episodes,
            "final_avg_reward": float(np.mean(episode_rewards[-100:])),
            "final_epsilon": float(self._agent.epsilon),
            "loss_history": self._agent.loss_history,
        }

    def predict(self, state: np.ndarray) -> int:
        """Get RL agent's action for the current state.

        Args:
            state: Observation array, shape (8,)

        Returns:
            0=HOLD, 1=ENTER_LONG, 2=EXIT_LONG
        """
        return self._agent.select_action(state, training=False)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        self._agent.save(path)
        self._model_path = str(path)

    def load(self, path: str | Path) -> None:
        self._agent.load(path)
        self._agent.to_inference_mode()
        self._is_trained = True
        self._model_path = str(path)

    @property
    def is_trained(self) -> bool:
        return self._is_trained
