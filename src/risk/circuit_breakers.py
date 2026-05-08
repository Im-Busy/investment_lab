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

import numpy as np


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
    cooldown_bars: int = 20
    reset_on_recovery: bool = True
    warning_threshold_pct: float = 10.0
    anomaly_threshold: float = 0.7
    anomaly_lookback: int = 5
    var_95_threshold: float = 0.05
    var_99_threshold: float = 0.10
    cvar_95_threshold: float = 0.08
    """Number of recent bars to check for anomaly persistence."""


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
        if np.isnan(peak_price) or peak_price <= 0:
            self.current_drawdown = 0.0
            return False, "No peak established", 0.0

        self.current_drawdown = ((peak_price - current_price) / peak_price) * 100

        if self.state == CircuitBreakerState.COOLDOWN:
            self.cooldown_counter -= 1

            if self.cooldown_counter <= 0:
                # Check if drawdown is still too high before resuming
                if self.current_drawdown >= self.config.max_drawdown_pct:
                    # Re-trip: extend cooldown instead of letting through
                    self.cooldown_counter = self.config.cooldown_bars
                    msg = (
                        f"Cooldown extended: DD {self.current_drawdown:.2f}% still above "
                        f"limit {self.config.max_drawdown_pct:.2f}%"
                    )
                    return True, msg, self.current_drawdown
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
        self.state = CircuitBreakerState.COOLDOWN
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

    def check_anomaly(
        self,
        anomaly_scores: list[float],
    ) -> tuple[bool, str]:
        """
        Check if HDBSCAN anomaly scores warrant a circuit breaker trip.

        Integrates with HDBSCANAnomalyDetector for flash crash,
        liquidity gap, and spoofing pattern detection.

        Args:
            anomaly_scores: Recent anomaly scores from HDBSCANAnomalyDetector

        Returns:
            Tuple of (trip_breaker, message)

        Example:
            >>> cb = CircuitBreaker()
            >>> trip, msg = cb.check_anomaly(hdbscan_detector.anomaly_score(features).tolist())
            >>> if trip:
            ...     cb.trip()
        """
        if not anomaly_scores:
            return False, "No anomaly data"

        latest = anomaly_scores[-1]
        threshold = self.config.anomaly_threshold
        lookback = min(self.config.anomaly_lookback, len(anomaly_scores))
        recent_mean = sum(anomaly_scores[-lookback:]) / lookback if lookback > 0 else latest

        if latest > threshold:
            msg = f"Anomaly trip: latest score {latest:.3f} exceeds threshold {threshold:.3f}"
            return True, msg

        if recent_mean > threshold:
            msg = (
                f"Anomaly trip: recent mean {recent_mean:.3f} exceeds "
                f"threshold {threshold:.3f} (lookback={lookback})"
            )
            return True, msg

        if latest > 0.5:
            msg = f"Anomaly warning: score {latest:.3f} elevated (threshold={threshold:.3f})"
            return False, msg

        return False, f"Anomaly score normal: {latest:.3f}"

    def check_full(
        self,
        current_price: float,
        peak_price: float,
        anomaly_scores: Optional[list[float]] = None,
        var_95: Optional[float] = None,
        var_99: Optional[float] = None,
        cvar_95: Optional[float] = None,
    ) -> tuple[bool, str, float]:
        """Combined check: drawdown + VaR/CVaR + anomaly detection.

        Args:
            current_price: Current portfolio value
            peak_price: Historical peak portfolio value
            anomaly_scores: Optional HDBSCAN anomaly scores
            var_95: 95% Value-at-Risk (% drop, positive = loss)
            var_99: 99% Value-at-Risk
            cvar_95: 95% Conditional VaR / Expected Shortfall

        Returns:
            Tuple of (halt_trading, message, drawdown_pct)
        """
        halt, msg, dd = self.check_circuit(current_price, peak_price)
        if halt:
            return halt, msg, dd

        if var_95 is not None and var_95 > self.config.var_95_threshold:
            halt_msg = (
                f"VaR trip: VaR95={var_95:.4f} exceeds threshold={self.config.var_95_threshold:.4f}"
            )
            return True, halt_msg, dd

        if var_99 is not None and var_99 > self.config.var_99_threshold:
            halt_msg = (
                f"VaR trip: VaR99={var_99:.4f} exceeds threshold={self.config.var_99_threshold:.4f}"
            )
            return True, halt_msg, dd

        if cvar_95 is not None and cvar_95 > self.config.cvar_95_threshold:
            halt_msg = (
                f"CVaR trip: CVaR95={cvar_95:.4f} exceeds "
                f"threshold={self.config.cvar_95_threshold:.4f}"
            )
            return True, halt_msg, dd

        if anomaly_scores:
            anomaly_halt, anomaly_msg = self.check_anomaly(anomaly_scores)
            if anomaly_halt:
                self._trip(current_price)
                return True, f"ANOMALY: {anomaly_msg}", dd

        return halt, msg, dd
