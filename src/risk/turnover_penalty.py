"""
Turnover Penalty Module

Implements turnover penalty as a hard constraint to prevent excessive trading.

Research Source: OOM-RL (Out-of-Memory Risk Limitation)
Finding: Strategies with 6700% annualized turnover destroyed alpha.
Conservative threshold: 2000% annualized turnover.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TurnoverPenaltyConfig:
    """Configuration for turnover penalty calculation."""

    max_allowed_turnover_pct: float = 2000.0
    """Maximum annualized turnover percentage allowed (hard limit)."""

    penalty_curve: str = "linear"
    """Type of penalty curve: 'linear', 'exponential', or 'step'."""

    warning_threshold_pct: float = 1000.0
    """Turnover level at which to warn (but not penalize)."""


class TurnoverPenalty:
    """
    Penalizes strategies with excessive annualized turnover.

    Methods:
    - calculate_penalty: Returns penalty scaling factor (0.0-1.0)
    - check_constraint: Returns True if turnover exceeds threshold
    """

    def __init__(self, config: Optional[TurnoverPenaltyConfig] = None):
        """
        Initialize turnover penalty.

        Args:
            config: Configuration for penalty calculation
        """
        self.config = config or TurnoverPenaltyConfig()

    def calculate_annualized_turnover(
        self,
        n_trades: int,
        n_days: int,
        portfolio_value: float,
    ) -> float:
        """
        Calculate annualized turnover percentage.

        Args:
            n_trades: Total number of trades executed
            n_days: Number of days in backtest period
            n_portfolio_value: Total portfolio value (for sizing)

        Returns:
            Annualized turnover as percentage
        """
        if n_days == 0:
            return 0.0

        annualized_trades = (n_trades / n_days) * 365

        return annualized_trades

    def calculate_penalty(
        self,
        n_trades: int,
        n_days: int,
        portfolio_value: float = 1.0,
    ) -> float:
        """
        Calculate penalty scaling factor based on turnover.

        Args:
            n_trades: Total number of trades executed
            n_days: Number of days in backtest period
            portfolio_value: Total portfolio value

        Returns:
            Penalty factor from 0.0 (no penalty) to 1.0 (full penalty)
        """
        turnover = self.calculate_annualized_turnover(n_trades, n_days, portfolio_value)

        if turnover <= self.config.max_allowed_turnover_pct:
            return 0.0

        excess = turnover - self.config.max_allowed_turnover_pct
        max_excess = self.config.max_allowed_turnover_pct

        # Guard against zero max_allowed_turnover_pct
        if max_excess <= 0:
            return 1.0 if excess > 0 else 0.0

        if self.config.penalty_curve == "linear":
            return min(excess / max_excess, 1.0)
        elif self.config.penalty_curve == "exponential":
            normalized_excess = excess / max_excess
            return min(normalized_excess**2, 1.0)
        elif self.config.penalty_curve == "step":
            return 1.0 if excess > 0 else 0.0
        else:
            return 0.0

    def check_constraint(
        self,
        n_trades: int,
        n_days: int,
        portfolio_value: float = 1.0,
    ) -> tuple[bool, float, str]:
        """
        Check if turnover exceeds constraint.

        Args:
            n_trades: Total number of trades executed
            n_days: Number of days in backtest period
            portfolio_value: Total portfolio value

        Returns:
            Tuple of (exceeded, turnover_pct, message)
        """
        turnover = self.calculate_annualized_turnover(n_trades, n_days, portfolio_value)

        if turnover > self.config.max_allowed_turnover_pct:
            message = (
                f"Turnover constraint violated: {turnover:.1f}% > "
                f"{self.config.max_allowed_turnover_pct:.1f}%"
            )
            return True, turnover, message

        if turnover > self.config.warning_threshold_pct:
            message = (
                f"Turnover warning: {turnover:.1f}% approaching "
                f"limit of {self.config.max_allowed_turnover_pct:.1f}%"
            )
            return False, turnover, message

        return False, turnover, "Turnover within acceptable limits"

    def apply_penalty_to_returns(
        self,
        returns: float,
        n_trades: int,
        n_days: int,
        portfolio_value: float = 1.0,
    ) -> float:
        """
        Apply penalty to strategy returns.

        Args:
            returns: Raw strategy returns
            n_trades: Total number of trades executed
            n_days: Number of days in backtest period
            portfolio_value: Total portfolio value

        Returns:
            Returns adjusted for turnover penalty
        """
        penalty = self.calculate_penalty(n_trades, n_days, portfolio_value)

        return returns * (1.0 - penalty)
