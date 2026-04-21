"""
Dynamic Rebalancing Frequency

Implements adaptive rebalancing frequency based on signal decay vs. transaction cost.

Research Source: OOM-RL (Out-of-Memory Risk Limitation)
Finding: Signal decay vs. transaction cost tradeoff.
Daily rebalancing destroyed 6700% turnover alpha.
.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class RebalanceFrequency(Enum):
    """Rebalancing frequency options."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class DynamicRebalanceConfig:
    """Configuration for dynamic rebalancing."""

    min_signal_decay_rate: float = 0.02
    """Minimum signal decay rate (2% per day) to consider daily."""

    tx_cost_pct: float = 0.001
    """Transaction cost as % of portfolio value (0.1%)."""

    decay_window: int = 5
    """Window to calculate signal decay rate (bars)."""

    min_days_between_rebalances: int = 1
    """Minimum days between rebalances (prevents over-trading)."""

    max_frequency: RebalanceFrequency = RebalanceFrequency.DAILY
    """Maximum allowed rebalancing frequency."""

    min_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    """Minimum allowed rebalancing frequency."""


class DynamicRebalancer:
    """
    Estimates optimal rebalancing frequency based on signal decay.

    Methods:
    - estimate_optimal_frequency: Determine daily/weekly/monthly
    - calculate_signal_decay: Compute signal decay rate
    - should_rebalance: Check if rebalancing is needed
    - update_signal_history: Track recent signal strengths
    """

    def __init__(self, config: Optional[DynamicRebalanceConfig] = None):
        """
        Initialize dynamic rebalancer.

        Args:
            config: Configuration for rebalancing logic
        """
        self.config = config or DynamicRebalanceConfig()
        self.signal_history = []
        self.last_rebalance_day = None

    def estimate_optimal_frequency(
        self,
        signal_decay_rate: Optional[float] = None,
        tx_cost_pct: Optional[float] = None,
    ) -> RebalanceFrequency:
        """
        Estimate optimal rebalancing frequency.

        Args:
            signal_decay_rate: Rate at which signal strength decays (% per bar)
            tx_cost_pct: Transaction cost (override config)

        Returns:
            Recommended frequency (daily/weekly/monthly)
        """
        if tx_cost_pct is None:
            tx_cost_pct = self.config.tx_cost_pct

        if signal_decay_rate is None:
            signal_decay_rate = self.calculate_signal_decay()

        if signal_decay_rate < self.config.min_signal_decay_rate:
            return self.config.min_frequency

        ratio = signal_decay_rate / tx_cost_pct

        if ratio > 10:
            frequency = RebalanceFrequency.DAILY
        elif ratio > 2:
            frequency = RebalanceFrequency.WEEKLY
        else:
            frequency = RebalanceFrequency.MONTHLY

        frequency = self._clamp_frequency(frequency)
        return frequency

    def calculate_signal_decay(self) -> float:
        """
        Calculate signal decay rate from recent history.

        Returns:
            Average decay rate (% per bar)
        """
        if len(self.signal_history) < 2:
            return 0.0

        window = self.signal_history[-self.config.decay_window :]
        if len(window) < 2:
            window = self.signal_history

        decays = []
        for i in range(1, len(window)):
            prev_strength = window[i - 1]
            curr_strength = window[i]

            if prev_strength > 0:
                decay = (prev_strength - curr_strength) / prev_strength
                decays.append(max(0.0, decay))

        if not decays:
            return 0.0

        return np.mean(decays)

    def update_signal_history(self, signal_strength: float) -> None:
        """
        Add new signal strength to history.

        Args:
            signal_strength: Latest signal confidence/quality score
        """
        self.signal_history.append(signal_strength)

        max_history = 100
        if len(self.signal_history) > max_history:
            self.signal_history = self.signal_history[-max_history:]

    def should_rebalance(
        self,
        current_day: int,
        signal_decay_rate: Optional[float] = None,
        force_rebalance: bool = False,
    ) -> tuple[bool, str, RebalanceFrequency]:
        """
        Check if rebalancing should occur.

        Args:
            current_day: Current backtest day
            signal_decay_rate: Current signal decay rate
            force_rebalance: Force rebalancing regardless of conditions

        Returns:
            Tuple of (should_rebalance, reason, frequency)
        """
        if force_rebalance:
            return True, "Forced rebalance", self.estimate_optimal_frequency()

        if self.last_rebalance_day is None:
            self.last_rebalance_day = current_day
            return True, "First rebalance", self.estimate_optimal_frequency()

        days_since = current_day - self.last_rebalance_day

        if days_since < self.config.min_days_between_rebalances:
            return (
                False,
                f"Too soon (only {days_since} days since last)",
                self.estimate_optimal_frequency(),
            )

        optimal_freq = self.estimate_optimal_frequency(signal_decay_rate)

        min_days_for_freq = self._frequency_to_days(optimal_freq)

        if days_since >= min_days_for_freq:
            self.last_rebalance_day = current_day
            return (
                True,
                f"Days since ({days_since}) >= threshold ({min_days_for_freq})",
                optimal_freq,
            )

        return (
            False,
            f"Wait {min_days_for_freq - days_since} more days",
            optimal_freq,
        )

    def _frequency_to_days(self, frequency: RebalanceFrequency) -> int:
        """Convert frequency to minimum days."""
        mapping = {
            RebalanceFrequency.DAILY: 1,
            RebalanceFrequency.WEEKLY: 5,
            RebalanceFrequency.MONTHLY: 20,
            RebalanceFrequency.QUARTERLY: 60,
        }
        return mapping.get(frequency, 20)

    def _clamp_frequency(self, frequency: RebalanceFrequency) -> RebalanceFrequency:
        """Clamp frequency to configured bounds."""
        freq_order = [
            RebalanceFrequency.DAILY,
            RebalanceFrequency.WEEKLY,
            RebalanceFrequency.MONTHLY,
            RebalanceFrequency.QUARTERLY,
        ]

        max_idx = freq_order.index(self.config.max_frequency)
        min_idx = freq_order.index(self.config.min_frequency)

        freq_idx = freq_order.index(frequency)
        freq_idx = max(min_idx, min(max_idx, freq_idx))

        return freq_order[freq_idx]

    def get_optimal_frequency_name(self, signal_decay_rate: Optional[float] = None) -> str:
        """
        Get human-readable frequency name.

        Args:
            signal_decay_rate: Current signal decay rate

        Returns:
            Frequency name string
        """
        freq = self.estimate_optimal_frequency(signal_decay_rate)
        return freq.value.upper()

    def reset(self) -> None:
        """Reset rebalancer state."""
        self.signal_history = []
        self.last_rebalance_day = None
