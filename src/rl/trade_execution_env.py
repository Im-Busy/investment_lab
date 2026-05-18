"""Gymnasium-compatible environment for RL-based trade execution decisions.

Models the per-bar entry/exit decision problem. At each bar, the agent
observes market context and the ML model's probability signal, then decides
whether to enter a long position, exit an existing position, or hold.

State space (8-dim continuous):
    0: ml_probability       - CatBoost probability in [0, 1]
    1: volatility_regime    - ATR(20)/ATR(100) ratio, normalized
    2: atr_pct              - ATR(14) / price, normalized
    3: sentiment_score      - Sentiment score in [-1, 1]
    4: confirm_streak       - Consecutive bars above entry threshold (0-N)
    5: position_pnl_pct     - Unrealized P&L as % of entry price (-inf, +inf)
    6: market_return_5d     - 5-day return
    7: is_event_day         - Boolean: high-impact event day

Action space (3 discrete):
    0: HOLD - do nothing, stay in/out of position as before
    1: ENTER_LONG - open a long position (ignored if already in position)
    2: EXIT_LONG - close the long position (ignored if not in position)

Reward:
    + profit_pct on exit (positive for winning trades, negative for losing)
    - 0.0001 on hold when not in position (small penalty for inaction)
    - 0.001 on entering on event days (penalty for bad timing)

Episode terminates when:
    - Agent exits position (terminal)
    - Max episode length reached
    - Truncation signal (e.g., end of data)
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

NUM_FEATURES = 8


class TradeExecutionEnv(gym.Env):
    """Gym environment for trade execution decisions.

    Operates at the bar level: each step corresponds to one OHLCV bar.
    The agent receives market context + ML probability and decides to
    enter, exit, or hold.

    Designed for training on historical IS data and evaluating on OOS.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        probabilities: np.ndarray,
        vol_regime: np.ndarray,
        atr_pct: np.ndarray,
        sentiment: np.ndarray,
        prices: np.ndarray,
        event_days: np.ndarray,
        entry_threshold: float = 0.50,
        max_episode_steps: int = 50,
        seed: int | None = None,
    ):
        """
        Args:
            probabilities: ML model probability array, shape (n_bars,), in [0, 1]
            vol_regime: Volatility regime ratio, shape (n_bars,)
            atr_pct: ATR as percentage of price, shape (n_bars,)
            sentiment: Sentiment scores, shape (n_bars,), in [-1, 1]
            prices: Close prices, shape (n_bars,)
            event_days: Boolean array, shape (n_bars,), True = high-impact event
            entry_threshold: Minimum probability to consider an entry signal
            max_episode_steps: Maximum bars per episode before truncation
            seed: Random seed
        """
        super().__init__()
        self._probs = np.asarray(probabilities, dtype=np.float32)
        self._vol_regime = np.asarray(vol_regime, dtype=np.float32)
        self._atr_pct = np.asarray(atr_pct, dtype=np.float32)
        self._sentiment = np.asarray(sentiment, dtype=np.float32)
        self._prices = np.asarray(prices, dtype=np.float32)
        self._event_days = np.asarray(event_days, dtype=bool)
        self._entry_threshold = entry_threshold
        self._max_episode_steps = max_episode_steps

        self._n_bars = len(self._probs)
        self._rng = np.random.RandomState(seed)

        # State: [prob, vol_regime, atr_pct, sentiment, confirm_streak,
        #          position_pnl_pct, market_return_5d, is_event_day]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -1.0, 0.0, -np.inf, -np.inf, 0.0], dtype=np.float32),
            high=np.array([1.0, 10.0, 1.0, 1.0, np.inf, np.inf, np.inf, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # Episode state
        self._cursor: int = 0
        self._episode_start: int = 0
        self._step_count: int = 0
        self._in_position: bool = False
        self._entry_price: float = 0.0
        self._entry_bar: int = 0
        self._confirm_streak: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment to start of a new episode.

        Episodes start at bars where probability >= entry_threshold,
        sampled randomly from valid entry bars.
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        # Sample a starting bar with a valid entry signal
        entry_bars = np.where(self._probs >= self._entry_threshold)[0]
        if len(entry_bars) == 0:
            self._episode_start = 0
        else:
            self._episode_start = int(self._rng.choice(entry_bars))

        self._cursor = self._episode_start
        self._step_count = 0
        self._in_position = False
        self._entry_price = 0.0
        self._entry_bar = 0
        self._confirm_streak = 0

        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step.

        Args:
            action: 0=HOLD, 1=ENTER_LONG, 2=EXIT_LONG

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        reward = 0.0
        terminated = False

        # ── Update confirm streak ──
        prob = float(self._probs[self._cursor])
        if prob >= self._entry_threshold:
            self._confirm_streak += 1
        else:
            self._confirm_streak = 0

        # ── Execute action ──
        if action == 1:  # ENTER_LONG
            if not self._in_position:
                self._in_position = True
                self._entry_price = float(self._prices[self._cursor])
                self._entry_bar = self._cursor
                # Penalty for entering on event days
                if self._event_days[self._cursor]:
                    reward -= 0.001
        elif action == 2:  # EXIT_LONG
            if self._in_position:
                exit_price = float(self._prices[self._cursor])
                profit_pct = (exit_price / self._entry_price) - 1.0
                reward += profit_pct
                self._in_position = False
                self._entry_price = 0.0
                terminated = True

        # ── Small penalty for inaction when not positioned ──
        if not self._in_position and action == 0:
            reward -= 0.0001

        # ── Advance cursor ──
        self._cursor += 1
        self._step_count += 1

        # ── Termination checks ──
        if self._cursor >= self._n_bars - 1:
            terminated = True
            # Force-close position at last bar
            if self._in_position:
                exit_price = float(self._prices[-1])
                profit_pct = (exit_price / self._entry_price) - 1.0
                reward += profit_pct
                self._in_position = False

        truncated = self._step_count >= self._max_episode_steps
        if truncated:
            # Close position on truncation
            if self._in_position:
                exit_price = float(self._prices[self._cursor - 1])
                profit_pct = (exit_price / self._entry_price) - 1.0
                reward += profit_pct
                self._in_position = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        """Build observation vector for current cursor position."""
        cursor = min(self._cursor, self._n_bars - 1)

        prob = float(self._probs[cursor])
        vr = float(self._vol_regime[cursor])
        atr_p = float(self._atr_pct[cursor])
        sent = float(self._sentiment[cursor])
        event = 1.0 if self._event_days[cursor] else 0.0

        # Position P&L
        if self._in_position and self._entry_price > 0:
            pnl_pct = (float(self._prices[cursor]) / self._entry_price) - 1.0
        else:
            pnl_pct = 0.0

        # 5-day market return
        if cursor >= 5:
            mkt_5d = (float(self._prices[cursor]) / float(self._prices[cursor - 5])) - 1.0
        else:
            mkt_5d = 0.0

        obs = np.array(
            [
                np.clip(prob, 0.0, 1.0),
                np.clip(vr, 0.0, 10.0),
                np.clip(atr_p, 0.0, 1.0),
                np.clip(sent, -1.0, 1.0),
                float(min(self._confirm_streak, 100)),
                float(pnl_pct),
                float(mkt_5d),
                float(event),
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    def _get_info(self) -> dict:
        return {
            "cursor": self._cursor,
            "in_position": self._in_position,
            "entry_price": self._entry_price,
            "confirm_streak": self._confirm_streak,
        }

    def render(self) -> None:
        info = self._get_info()
        print(
            f"[Bar {info['cursor']}/{self._n_bars}] "
            f"pos={info['in_position']} "
            f"entry=${info['entry_price']:.2f} "
            f"streak={info['confirm_streak']}"
        )

    def seed(self, seed: int | None = None) -> list[int]:
        self._rng = np.random.RandomState(seed)
        return [seed] if seed is not None else [0]
