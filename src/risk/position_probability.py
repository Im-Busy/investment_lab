"""
Per-Position Success/Failure Probability Model

Implements binomial outcome model per position for risk-adjusted position sizing.

Research Source: Jorion (Value-at-Risk, BET approach)
Finding: Per-position probability > aggregate VaR.
Solution: Position-level success/failure estimates for proper capital allocation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np


class RegimeState(Enum):
    """Market regime states."""

    TRENDING = "Trending"
    RANGING = "Ranging"
    VOLATILE = "Volatile"
    TRANSITION = "Transition"


@dataclass
class PositionRiskConfig:
    """Configuration for position risk estimation."""

    base_success_prob: float = 0.55
    """Base probability of successful trade (from historical win rate)."""

    regime_adjustments: Dict[str, float] = None
    """Regime-specific multipliers for success probability."""

    confidence_weight: float = 0.3
    """How much signal confidence affects position probability (0.0-1.0)."""

    volatility_penalty: float = 0.2
    """Penalty factor for high volatility regimes."""

    def __post_init__(self):
        if self.regime_adjustments is None:
            self.regime_adjustments = {
                RegimeState.TRENDING.value: 1.2,
                RegimeState.RANGING.value: 0.9,
                RegimeState.VOLATILE.value: 0.7,
                RegimeState.TRANSITION.value: 0.8,
            }


class PositionRiskModel:
    """
    Binomial outcome model per position (Jorion BET approach).

    Estimates probability of successful outcome for each position
    based on signal confidence, current regime, and volatility.

    Methods:
    - estimate_success_prob: Calculate success probability
    - calculate_position_size: Risk-adjusted position sizing
    - get_expected_value: EV calculation for position
    """

    def __init__(self, config: Optional[PositionRiskConfig] = None):
        """
        Initialize position risk model.

        Args:
            config: Configuration for risk estimation
        """
        self.config = config or PositionRiskConfig()

    def estimate_success_prob(
        self,
        signal_confidence: float,
        regime: RegimeState,
        volatility: Optional[float] = None,
        base_prob: Optional[float] = None,
    ) -> float:
        """
        Estimate probability of successful outcome.

        Args:
            signal_confidence: Signal confidence score (0.0-1.0)
            regime: Current market regime
            volatility: Current volatility (ATR % or std dev)
            base_prob: Override base probability

        Returns:
            Success probability (0.0-1.0)
        """
        if base_prob is None:
            base_prob = self.config.base_success_prob

        regime_multiplier = self.config.regime_adjustments.get(regime.value, 1.0)

        prob = base_prob * regime_multiplier

        confidence_factor = (signal_confidence - 0.5) * self.config.confidence_weight
        prob += confidence_factor

        if volatility is not None:
            volatility_penalty = min(volatility / 0.05, 1.0) * self.config.volatility_penalty
            prob -= volatility_penalty

        return max(0.1, min(0.9, prob))

    def estimate_failure_prob(
        self,
        success_prob: Optional[float] = None,
        signal_confidence: Optional[float] = None,
        regime: Optional[RegimeState] = None,
    ) -> float:
        """
        Estimate probability of failure (complement of success).

        Args:
            success_prob: Pre-calculated success probability
            signal_confidence: Signal confidence (if success_prob not provided)
            regime: Current regime (if success_prob not provided)

        Returns:
            Failure probability (0.0-1.0)
        """
        if success_prob is not None:
            return 1.0 - success_prob

        if signal_confidence is not None and regime is not None:
            success_prob = self.estimate_success_prob(signal_confidence, regime)
            return 1.0 - success_prob

        return 1.0 - self.config.base_success_prob

    def calculate_position_size(
        self,
        success_prob: float,
        failure_prob: Optional[float] = None,
        max_risk_pct: float = 0.02,
        kelly_criterion: bool = False,
    ) -> float:
        """
        Calculate risk-adjusted position size.

        Args:
            success_prob: Probability of successful outcome
            failure_prob: Probability of failure (if None, calculated as 1-success_prob)
            max_risk_pct: Maximum position size as % of portfolio
            kelly_criterion: Use Kelly Criterion for sizing

        Returns:
            Position size as % of portfolio (0.0-1.0)
        """
        if failure_prob is None:
            failure_prob = 1.0 - success_prob

        if kelly_criterion:
            win_rate = success_prob
            avg_win = 1.0
            avg_loss = -1.0

            b = avg_win / abs(avg_loss)
            kelly_fraction = (b * win_rate - (1 - win_rate)) / b

            position_size = max(0.0, min(kelly_fraction, max_risk_pct))
        else:
            position_size = max_risk_pct * success_prob

        return max(0.0, min(position_size, max_risk_pct))

    def get_expected_value(
        self,
        success_prob: float,
        payoff_ratio: float = 1.5,
    ) -> float:
        """
        Calculate expected value for position.

        Args:
            success_prob: Probability of successful outcome
            payoff_ratio: Win/loss ratio (e.g., 1.5 = 1.5x win for 1x loss)

        Returns:
            Expected value per unit risk
        """
        failure_prob = 1.0 - success_prob
        ev = (success_prob * payoff_ratio) - (failure_prob * 1.0)
        return ev

    def get_var_confidence(
        self,
        success_prob: float,
        n_positions: int,
        confidence_level: float = 0.95,
    ) -> float:
        """
        Calculate VaR for portfolio of positions.

        Args:
            success_prob: Individual position success probability
            n_positions: Number of positions in portfolio
            confidence_level: VaR confidence level (0.0-1.0)

        Returns:
            Value at Risk as % of portfolio
        """
        from scipy.stats import binom

        failure_prob = 1.0 - success_prob
        n_failures = n_positions * failure_prob

        var_pct = binom.ppf(1.0 - confidence_level, n_positions, failure_prob) / n_positions

        return min(var_pct, 1.0)

    def batch_estimate_risks(
        self,
        signals: list[dict],
        regime: RegimeState,
    ) -> list[dict]:
        """
        Batch estimate risks for multiple signals.

        Args:
            signals: List of signal dictionaries
            regime: Current market regime

        Returns:
            List of signal dicts with risk estimates added
        """
        enhanced_signals = []

        for signal in signals:
            confidence = signal.get("confidence", 0.5)
            success_prob = self.estimate_success_prob(confidence, regime)

            risk_estimate = {
                "success_probability": success_prob,
                "failure_probability": 1.0 - success_prob,
                "position_size_risk": self.calculate_position_size(success_prob),
                "expected_value": self.get_expected_value(success_prob),
            }

            enhanced_signal = {**signal, **risk_estimate}
            enhanced_signals.append(enhanced_signal)

        return enhanced_signals
