"""
Portfolio Circuit Breaker Module

Implements portfolio-level drawdown halt mechanism.

Research Source: Jorion (Value-at-Risk)
Finding: Individual position risk < aggregate portfolio risk.
Solution: Portfolio-wide halt to prevent cascade failures.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CircuitBreakerState(Enum):
    """States of the circuit breaker."""

    ACTIVE = "active"
    """Trading is allowed."""

    TRIPPED = "tripped"
    """Trading halted due to max drawdown exceeded."""

    COOLDOWN = "cooldown"
    """Trading halted during cooldown period."""


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    max_drawdown_pct: float = 20.0
    """Maximum portfolio drawdown before tripping breaker (percentage)."""

    cooldown_bars: int = 20
    """Number of bars to stay in cooldown after tripping."""

    reset_on_recovery: bool = True
    """Auto-reset when portfolio recovers above trip threshold."""

    warning_threshold_pct: float = 10.0
    """Drawdown level at which to issue warning (but not halt)."""


class CircuitBreaker:
    """
    Portfolio-wide drawdown halt mechanism.

    Monitors portfolio drawdown and halts new positions when
    maximum acceptable drawdown is exceeded.

    Methods:
    - check_circuit: Check if trading should halt
    - trip: Manually trigger the breaker
    - reset: Reset the breaker to active state
    - get_state: Get current breaker state
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.

        Args:
            config: Configuration for breaker behavior
        """
        self.config = config or CircuitBreakerConfig()

        self.state = CircuitBreakerState.ACTIVE
        self.cooldown_counter = 0
        self.trip_price = None
        self.current_drawdown = 0.0

    def check_circuit(
        self,
        current_price: float,
        peak_price: float,
    ) -> tuple[bool, str, float]:
        """
        Check if trading should be halted.

        Args:
            current_price: Current portfolio value
            peak_price: Historical peak portfolio value

        Returns:
            Tuple of (halt_trading, message, drawdown_pct)
        """
        if peak_price <= 0:
            self.current_drawdown = 0.0
            return False, "No peak established", 0.0

        self.current_drawdown = ((peak_price - current_price) / peak_price) * 100

        if self.state == CircuitBreakerState.COOLDOWN:
            self.cooldown_counter -= 1

            if self.cooldown_counter <= 0:
                self.state = CircuitBreakerState.ACTIVE
                msg = f"Cooldown expired. Trading resumed. DD: {self.current_drawdown:.2f}%"
                return False, msg, self.current_drawdown

            msg = f"In cooldown ({self.cooldown_counter} bars remaining). DD: {self.current_drawdown:.2f}%"
            return True, msg, self.current_drawdown

        if self.current_drawdown >= self.config.max_drawdown_pct:
            self._trip(current_price)
            msg = (
                f"Circuit breaker TRIPPED! Max DD exceeded: "
                f"{self.current_drawdown:.2f}% > {self.config.max_drawdown_pct:.2f}%"
            )
            return True, msg, self.current_drawdown

        if self.state == CircuitBreakerState.TRIPPED:
            if self.config.reset_on_recovery:
                if self.current_drawdown < self.config.warning_threshold_pct:
                    self.state = CircuitBreakerState.ACTIVE
                    msg = f"Portfolio recovered. Trading resumed. DD: {self.current_drawdown:.2f}%"
                    return False, msg, self.current_drawdown

            msg = f"Circuit breaker tripped. Trading halted. DD: {self.current_drawdown:.2f}%"
            return True, msg, self.current_drawdown

        if self.current_drawdown >= self.config.warning_threshold_pct:
            msg = (
                f"Drawdown warning: {self.current_drawdown:.2f}% approaching "
                f"limit of {self.config.max_drawdown_pct:.2f}%"
            )
            return False, msg, self.current_drawdown

        return False, "Trading allowed", self.current_drawdown

    def _trip(self, current_price: float) -> None:
        """
        Internal method to trip the breaker.

        Args:
            current_price: Current portfolio value when tripped
        """
        self.state = CircuitBreakerState.TRIPPED
        self.cooldown_counter = self.config.cooldown_bars
        self.trip_price = current_price

    def trip(self, current_price: Optional[float] = None) -> str:
        """
        Manually trigger the circuit breaker.

        Args:
            current_price: Current portfolio value (optional)

        Returns:
            Message confirming breaker was tripped
        """
        self._trip(current_price or 0.0)
        return "Circuit breaker manually tripped"

    def reset(self) -> str:
        """
        Reset the breaker to active state.

        Returns:
            Message confirming reset
        """
        self.state = CircuitBreakerState.ACTIVE
        self.cooldown_counter = 0
        self.trip_price = None
        self.current_drawdown = 0.0
        return "Circuit breaker reset to active state"

    def get_state(self) -> tuple[CircuitBreakerState, float, Optional[float]]:
        """
        Get current breaker state.

        Returns:
            Tuple of (state, cooldown_counter, trip_price)
        """
        return (
            self.state,
            self.cooldown_counter,
            self.trip_price,
        )

    def is_halted(self) -> bool:
        """
        Check if trading is currently halted.

        Returns:
            True if trading should be halted
        """
        return self.state in (CircuitBreakerState.TRIPPED, CircuitBreakerState.COOLDOWN)

    def is_active(self) -> bool:
        """
        Check if trading is currently allowed.

        Returns:
            True if trading is allowed
        """
        return self.state == CircuitBreakerState.ACTIVE
