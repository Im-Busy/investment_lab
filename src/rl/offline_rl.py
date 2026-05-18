"""
D12: Conservative Q-Learning (CQL) for offline trade execution.

Offline/batch RL learns optimal policies from historical data without
environment interaction. Critical for finance where live experimentation
is expensive and historical data is abundant.

CQL (Kumar et al., NeurIPS 2020) adds a conservative penalty to Q-learning:
    L_CQL = α * (logsumexp Q(s,a) - Q(s,a_target)) + standard TD error

This prevents Q-value overestimation on out-of-distribution actions,
a common failure mode when Bellman backups query unseen state-action pairs
in offline settings.

Architecture:
    OfflineDataset: Static replay buffer from historical trade sequences.
    CQLAgent: Double Q-networks + target networks + conservative penalty.
    CQLTradeExecutor: Backtesting inference wrapper.

CQL is safer than DQN for historical data because it explicitly penalizes
extrapolation beyond the dataset's support.

References:
    Kumar et al. "Conservative Q-Learning for Offline Reinforcement Learning"
    https://arxiv.org/abs/2006.04779
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)

NUM_FEATURES = 8
NUM_ACTIONS = 3  # 0=HOLD, 1=ENTER_LONG, 2=EXIT_LONG


# ── Dataset ──────────────────────────────────────────────────────────────


@dataclass
class Transition:
    """Single transition tuple for offline RL."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class OfflineDataset:
    """Static dataset of (s, a, r, s', done) transitions.

    Built from historical trade execution traces without environment
    interaction. Supports weighted sampling and normalization.

    Args:
        transitions: List of Transition objects.
        state_mean: Optional pre-computed state mean for normalization.
        state_std: Optional pre-computed state std for normalization.
    """

    def __init__(
        self,
        transitions: Optional[List[Transition]] = None,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
    ):
        self._transitions: List[Transition] = list(transitions) if transitions else []
        self._state_mean = (
            state_mean if state_mean is not None else np.zeros(NUM_FEATURES, dtype=np.float32)
        )
        self._state_std = (
            state_std if state_std is not None else np.ones(NUM_FEATURES, dtype=np.float32)
        )
        self._normalized = state_mean is not None

    @classmethod
    def from_env_trajectories(
        cls,
        env_kwargs: dict,
        n_episodes: int = 500,
        policy: str = "threshold",
        entry_threshold: float = 0.50,
        normalize: bool = True,
    ) -> OfflineDataset:
        """Build dataset by running episodes in the environment.

        Uses a simple behavior policy (threshold-based) to generate
        trajectories. The resulting dataset captures realistic
        state distributions for offline training.

        Args:
            env_kwargs: Keyword arguments for TradeExecutionEnv.
            n_episodes: Number of episodes to generate.
            policy: Behavior policy: "threshold" (enter on prob>=thresh,
                    exit on profit target or loss limit).
            entry_threshold: Probability threshold for entry signals.
            normalize: If True, compute and store state normalization stats.

        Returns:
            OfflineDataset populated with transitions.
        """
        from src.rl.trade_execution_env import TradeExecutionEnv

        transitions: List[Transition] = []
        all_states = []

        env = TradeExecutionEnv(**env_kwargs)

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False

            while not done:
                state = obs.copy()

                if policy == "threshold":
                    prob = obs[0]
                    if not env._in_position:
                        action = 1 if prob >= entry_threshold else 0
                    else:
                        pnl = obs[5]
                        action = 2 if pnl > 0.01 or pnl < -0.02 else 0
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                transitions.append(
                    Transition(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=obs.copy(),
                        done=done,
                    )
                )
                all_states.append(state)

        all_states_arr = np.array(all_states, dtype=np.float32)
        state_mean = all_states_arr.mean(axis=0) if normalize else None
        state_std = all_states_arr.std(axis=0).clip(1e-6) if normalize else None

        return cls(transitions=transitions, state_mean=state_mean, state_std=state_std)

    def __len__(self) -> int:
        return len(self._transitions)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch of transitions.

        Returns:
            Dict with 'states', 'actions', 'rewards', 'next_states', 'dones'.
        """
        indices = np.random.choice(len(self._transitions), batch_size, replace=True)
        batch = [self._transitions[i] for i in indices]

        states = np.stack([t.state for t in batch])
        next_states = np.stack([t.next_state for t in batch])

        if self._normalized:
            states = (states - self._state_mean) / self._state_std
            next_states = (next_states - self._state_mean) / self._state_std

        return {
            "states": torch.tensor(states, dtype=torch.float32),
            "actions": torch.tensor([t.action for t in batch], dtype=torch.long),
            "rewards": torch.tensor([t.reward for t in batch], dtype=torch.float32),
            "next_states": torch.tensor(next_states, dtype=torch.float32),
            "dones": torch.tensor([float(t.done) for t in batch], dtype=torch.float32),
        }

    def save(self, path: str) -> None:
        data = {
            "transitions": [
                {
                    "state": t.state.tolist(),
                    "action": t.action,
                    "reward": t.reward,
                    "next_state": t.next_state.tolist(),
                    "done": t.done,
                }
                for t in self._transitions
            ],
            "state_mean": self._state_mean.tolist(),
            "state_std": self._state_std.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info(f"Dataset saved ({len(self)} transitions) → {path}")

    @classmethod
    def load(cls, path: str) -> OfflineDataset:
        with open(path) as f:
            data = json.load(f)
        transitions = [
            Transition(
                state=np.array(t["state"], dtype=np.float32),
                action=t["action"],
                reward=t["reward"],
                next_state=np.array(t["next_state"], dtype=np.float32),
                done=t["done"],
            )
            for t in data["transitions"]
        ]
        return cls(
            transitions=transitions,
            state_mean=np.array(data["state_mean"], dtype=np.float32),
            state_std=np.array(data["state_std"], dtype=np.float32),
        )


# ── CQL Agent ─────────────────────────────────────────────────────────────


class QNetwork(nn.Module):
    """Q-network for CQL: maps state → action-values."""

    def __init__(
        self,
        input_dim: int = NUM_FEATURES,
        output_dim: int = NUM_ACTIONS,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class CQLResult:
    """CQL training result."""

    cql_alpha: float
    n_iterations: int
    batch_size: int
    n_transitions: int
    final_qf_loss: float
    final_cql_loss: float
    final_bellman_loss: float

    def to_dict(self) -> Dict:
        return {
            "cql_alpha": self.cql_alpha,
            "n_iterations": self.n_iterations,
            "batch_size": self.batch_size,
            "n_transitions": self.n_transitions,
            "final_qf_loss": round(self.final_qf_loss, 6),
            "final_cql_loss": round(self.final_cql_loss, 6),
            "final_bellman_loss": round(self.final_bellman_loss, 6),
        }


class CQLAgent:
    """Conservative Q-Learning agent for offline trade execution.

    Uses double Q-networks with target networks and a conservative
    regularization term to prevent overestimation on OOD actions.

    Args:
        cql_alpha: Conservative penalty coefficient.
            Higher = more conservative (trusts dataset more).
            Lower = allows more extrapolation.
        gamma: Discount factor.
        tau: Target network soft update rate.
        lr: Learning rate.
        hidden_dim: Hidden layer size.
        device: Torch device.
    """

    def __init__(
        self,
        cql_alpha: float = 5.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        hidden_dim: int = 256,
        device: Optional[str] = None,
    ):
        self._cql_alpha = cql_alpha
        self._gamma = gamma
        self._tau = tau
        self._lr = lr

        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self._qf1 = QNetwork(hidden_dim=hidden_dim).to(self._device)
        self._qf2 = QNetwork(hidden_dim=hidden_dim).to(self._device)
        self._target_qf1 = QNetwork(hidden_dim=hidden_dim).to(self._device)
        self._target_qf2 = QNetwork(hidden_dim=hidden_dim).to(self._device)

        self._target_qf1.load_state_dict(self._qf1.state_dict())
        self._target_qf2.load_state_dict(self._qf2.state_dict())

        self._optimizer = optim.Adam(
            list(self._qf1.parameters()) + list(self._qf2.parameters()),
            lr=self._lr,
        )

        self._train_step: int = 0
        self._loss_history: Dict[str, List[float]] = {
            "qf_loss": [],
            "cql_penalty": [],
            "bellman_error": [],
        }

    def train(
        self,
        dataset: OfflineDataset,
        n_iterations: int = 10_000,
        batch_size: int = 256,
        log_interval: int = 1_000,
    ) -> CQLResult:
        """Train CQL on the offline dataset.

        Args:
            dataset: OfflineDataset of (s, a, r, s', done) transitions.
            n_iterations: Number of gradient steps.
            batch_size: Batch size.
            log_interval: Log every N iterations.

        Returns:
            CQLResult with training metrics.
        """
        final_qf = 0.0
        final_cql = 0.0
        final_bellman = 0.0

        for iteration in range(n_iterations):
            batch = dataset.sample(batch_size)
            states = batch["states"].to(self._device)
            actions = batch["actions"].to(self._device)
            rewards = batch["rewards"].to(self._device)
            next_states = batch["next_states"].to(self._device)
            dones = batch["dones"].to(self._device)

            qf1_loss, cql_penalty, bellman_loss = self._compute_loss(
                states, actions, rewards, next_states, dones
            )

            total_loss = qf1_loss

            self._optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self._qf1.parameters()) + list(self._qf2.parameters()),
                max_norm=10.0,
            )
            self._optimizer.step()

            self._update_targets()
            self._train_step += 1

            self._loss_history["qf_loss"].append(float(qf1_loss))
            self._loss_history["cql_penalty"].append(float(cql_penalty))
            self._loss_history["bellman_error"].append(float(bellman_loss))

            final_qf = float(qf1_loss)
            final_cql = float(cql_penalty)
            final_bellman = float(bellman_loss)

            if iteration % log_interval == 0:
                logger.info(
                    f"CQL iter {iteration}/{n_iterations} | "
                    f"QF={qf1_loss:.4f} CQL={cql_penalty:.4f} "
                    f"Bellman={bellman_loss:.4f}"
                )

        return CQLResult(
            cql_alpha=self._cql_alpha,
            n_iterations=n_iterations,
            batch_size=batch_size,
            n_transitions=len(dataset),
            final_qf_loss=final_qf,
            final_cql_loss=final_cql,
            final_bellman_loss=final_bellman,
        )

    def _compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute CQL loss = Bellman error + conservative penalty.

        Bellman error:
            L_Bellman = MSE(Q(s,a), r + γ * min_i Q_target_i(s', a') * (1-done))

        CQL penalty (Equation 3 in Kumar et al.):
            L_CQL = α * (logsumexp Q(s,·) - Q(s, a_target))
            Minimizing logsumexp pushes ALL Q-values down for unseen actions,
            while the Q(s, a_target) term pulls up the value for in-distribution actions.

        We use the sampled action as a_target (standard CQL implementation).
        """
        batch_size = states.shape[0]

        with torch.no_grad():
            next_q1 = self._target_qf1(next_states)
            next_q2 = self._target_qf2(next_states)
            min_next_q = torch.min(next_q1, next_q2)
            next_actions = min_next_q.argmax(dim=-1, keepdim=True)
            target_q = rewards + self._gamma * (min_next_q.gather(-1, next_actions).squeeze(-1)) * (
                1.0 - dones
            )

        q1_values = self._qf1(states)
        q2_values = self._qf2(states)
        q1_pred = q1_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q2_pred = q2_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        bellman_q1 = F.mse_loss(q1_pred, target_q)
        bellman_q2 = F.mse_loss(q2_pred, target_q)
        bellman_loss = bellman_q1 + bellman_q2

        cql_penalty = self._cql_alpha * (
            torch.logsumexp(q1_values, dim=-1).mean()
            + torch.logsumexp(q2_values, dim=-1).mean()
            - q1_pred.mean()
            - q2_pred.mean()
        )

        total_loss = bellman_loss + cql_penalty
        return total_loss, cql_penalty, bellman_loss

    def _update_targets(self) -> None:
        for target, source in [
            (self._target_qf1, self._qf1),
            (self._target_qf2, self._qf2),
        ]:
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(self._tau * sp.data + (1.0 - self._tau) * tp.data)

    def predict(self, observation: np.ndarray) -> int:
        """Select action from observation (greedy w.r.t. Q-values).

        Args:
            observation: 8-dim observation array (raw, will be normalized if dataset had stats).

        Returns:
            Discrete action 0-2.
        """
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self._device)
        with torch.no_grad():
            q1 = self._qf1(state)
            q2 = self._qf2(state)
            q_values = q1 + q2
        return int(q_values.argmax(dim=-1).item())

    def predict_q(self, observation: np.ndarray) -> np.ndarray:
        """Return Q-values for all actions."""
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self._device)
        with torch.no_grad():
            q1 = self._qf1(state)
            q2 = self._qf2(state)
        return (q1 + q2).cpu().numpy().flatten().astype(np.float32)

    def save_model(self, path: Union[str, Path]) -> None:
        state = {
            "qf1": self._qf1.state_dict(),
            "qf2": self._qf2.state_dict(),
            "target_qf1": self._target_qf1.state_dict(),
            "target_qf2": self._target_qf2.state_dict(),
            "cql_alpha": self._cql_alpha,
            "gamma": self._gamma,
            "train_step": self._train_step,
        }
        torch.save(state, str(path))
        logger.info(f"CQL model saved to {path}")

    def load_model(self, path: Union[str, Path]) -> None:
        state = torch.load(str(path), map_location=self._device, weights_only=True)
        self._qf1.load_state_dict(state["qf1"])
        self._qf2.load_state_dict(state["qf2"])
        self._target_qf1.load_state_dict(state["target_qf1"])
        self._target_qf2.load_state_dict(state["target_qf2"])
        self._cql_alpha = state["cql_alpha"]
        self._gamma = state["gamma"]
        self._train_step = state.get("train_step", 0)
        logger.info(f"CQL model loaded from {path}")

    @property
    def loss_history(self) -> Dict[str, List[float]]:
        return self._loss_history


# ── CQL Executor ──────────────────────────────────────────────────────────


class CQLTradeExecutor:
    """Offline RL inference wrapper for backtesting integration.

    Provides the same interface as DQNAgent/RLTradeExecutor for
    drop-in replacement in existing backtesting infrastructure.

    Args:
        agent: Trained CQLAgent instance.
        dataset: Optional OfflineDataset for state normalization.
    """

    def __init__(
        self,
        agent: CQLAgent,
        dataset: Optional[OfflineDataset] = None,
    ):
        self._agent = agent
        self._dataset = dataset

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Predict action from observation.

        Args:
            observation: 8-dim observation.
            deterministic: Unused (CQL is always greedy). For API compatibility.

        Returns:
            Action 0-2.
        """
        if self._dataset is not None and self._dataset._normalized:
            obs = (observation - self._dataset._state_mean) / self._dataset._state_std
            return self._agent.predict(obs)
        return self._agent.predict(observation)

    def predict_proba(self, observation: np.ndarray) -> np.ndarray:
        """Return Q-values for all actions (proxy for probabilities)."""
        q_values = self._agent.predict_q(observation)
        q_min = q_values.min()
        q_exp = np.exp(q_values - q_min)
        return q_exp / q_exp.sum()
