"""
Portfolio-Level Risk Management

This module provides risk management controls at the portfolio level,
including position sizing, volatility targeting, drawdown limits,
and correlation-based diversification.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    prange = range


class RiskLimitType(Enum):
    """Types of risk limits."""

    MAX_POSITION_SIZE = "max_position_size"
    MAX_PORTFOLIO_EXPOSURE = "max_portfolio_exposure"
    MAX_DRAWDOWN = "max_drawdown"
    MAX_VOLATILITY = "max_volatility"
    MAX_CORRELATION = "max_correlation"
    MAX_TURNOVER = "max_turnover"
    MAX_LOSS_PER_PERIOD = "max_loss_per_period"


@dataclass
class RiskLimit:
    """A single risk limit."""

    limit_type: RiskLimitType
    value: float
    current_value: float = 0.0
    is_breached: bool = False
    warning_threshold: float = 0.8

    def check_breach(self) -> bool:
        """Check if limit is breached."""
        self.is_breached = self.current_value > self.value
        return self.is_breached

    def is_near_limit(self) -> bool:
        """Check if near limit (80% of limit)."""
        return self.current_value > self.value * self.warning_threshold


@jit(nopython=True, cache=True)
def compute_ewma_volatility_numba(
    returns: np.ndarray, span: int = 20, annualize: bool = True
) -> np.ndarray:
    """Compute exponentially weighted moving average volatility."""
    n = len(returns)
    volatility = np.zeros(n)

    alpha = 2.0 / (span + 1)

    if n == 0:
        return volatility

    volatility[0] = np.abs(returns[0])

    for i in range(1, n):
        volatility[i] = np.sqrt(alpha * returns[i] ** 2 + (1 - alpha) * volatility[i - 1] ** 2)

    if annualize:
        volatility = volatility * np.sqrt(252)

    return volatility


@jit(nopython=True, cache=True)
def compute_portfolio_volatility_numba(
    returns_matrix: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Compute portfolio volatility from asset returns."""
    n_days = returns_matrix.shape[0]
    n_assets = returns_matrix.shape[1]

    portfolio_returns = np.zeros(n_days)

    for t in range(n_days):
        for i in range(n_assets):
            portfolio_returns[t] += weights[i] * returns_matrix[t, i]

    return np.std(portfolio_returns) * np.sqrt(252)


@jit(nopython=True, cache=True)
def compute_turnover_numba(positions: np.ndarray, window: int = 20) -> np.ndarray:
    """Compute portfolio turnover (rate of position change)."""
    n = len(positions)
    turnover = np.zeros(n)

    for i in range(1, n):
        turnover[i] = np.abs(positions[i] - positions[i - 1])

    # Rolling sum
    rolling_turnover = np.zeros(n)
    for i in range(window, n):
        rolling_turnover[i] = np.sum(turnover[i - window : i])

    return rolling_turnover


class PortfolioRiskManager:
    """Portfolio-level risk management."""

    def __init__(
        self,
        max_portfolio_exposure: float = 1.0,
        max_drawdown: float = 0.2,
        max_volatility: float = 0.3,
        max_correlation: float = 0.8,
        max_turnover: float = 2.0,
        volatility_target: Optional[float] = None,
    ):
        self.risk_limits = {
            RiskLimitType.MAX_PORTFOLIO_EXPOSURE: RiskLimit(
                RiskLimitType.MAX_PORTFOLIO_EXPOSURE, max_portfolio_exposure
            ),
            RiskLimitType.MAX_DRAWDOWN: RiskLimit(RiskLimitType.MAX_DRAWDOWN, max_drawdown),
            RiskLimitType.MAX_VOLATILITY: RiskLimit(RiskLimitType.MAX_VOLATILITY, max_volatility),
            RiskLimitType.MAX_CORRELATION: RiskLimit(
                RiskLimitType.MAX_CORRELATION, max_correlation
            ),
            RiskLimitType.MAX_TURNOVER: RiskLimit(RiskLimitType.MAX_TURNOVER, max_turnover),
        }

        self.volatility_target = volatility_target
        self.current_exposure = 0.0
        self.current_drawdown = 0.0
        self.current_volatility = 0.0
        self.current_turnover = 0.0
        self.logger = logging.getLogger(__name__)

    def check_risk_limits(self) -> Dict[str, bool]:
        """Check all risk limits."""
        self.risk_limits[RiskLimitType.MAX_PORTFOLIO_EXPOSURE].current_value = self.current_exposure
        self.risk_limits[RiskLimitType.MAX_DRAWDOWN].current_value = self.current_drawdown
        self.risk_limits[RiskLimitType.MAX_VOLATILITY].current_value = self.current_volatility
        self.risk_limits[RiskLimitType.MAX_TURNOVER].current_value = self.current_turnover

        results = {}
        for limit_type, limit in self.risk_limits.items():
            results[limit_type.value] = limit.check_breach()

        return results

    def is_risk_breached(self) -> bool:
        """Check if any risk limit is breached."""
        risk_results = self.check_risk_limits()
        return any(risk_results.values())

    def get_risk_warnings(self) -> List[str]:
        """Get warnings for near-limit conditions."""
        warnings = []

        for limit_type, limit in self.risk_limits.items():
            if limit.is_near_limit() and not limit.is_breached:
                warnings.append(
                    f"Warning: {limit_type.value} at {limit.current_value:.2%} "
                    f"(limit: {limit.value:.2%})"
                )

        return warnings

    def scale_positions_by_risk(
        self, positions: np.ndarray, risk_factor: float = 1.0
    ) -> np.ndarray:
        """Scale positions based on risk factor."""
        return positions * risk_factor

    def apply_volatility_targeting(
        self, positions: np.ndarray, current_volatility: float
    ) -> np.ndarray:
        """Apply volatility targeting to positions."""
        if self.volatility_target is None:
            return positions

        if current_volatility < 1e-10:
            return positions

        # Scale positions to target volatility
        scale_factor = self.volatility_target / current_volatility

        # Limit scaling to avoid drastic changes
        scale_factor = np.clip(scale_factor, 0.5, 2.0)

        return positions * scale_factor

    def limit_portfolio_exposure(
        self, positions: np.ndarray, max_exposure: Optional[float] = None
    ) -> np.ndarray:
        """Limit total portfolio exposure."""
        if max_exposure is None:
            max_exposure = self.risk_limits[RiskLimitType.MAX_PORTFOLIO_EXPOSURE].value

        current_exposure = np.sum(np.abs(positions))

        if current_exposure <= max_exposure:
            return positions

        # Scale down positions
        scale_factor = max_exposure / current_exposure
        return positions * scale_factor

    def apply_circuit_breaker(self, drawdown: float, returns: np.ndarray) -> np.ndarray:
        """Apply circuit breaker on excessive drawdown."""
        max_dd = self.risk_limits[RiskLimitType.MAX_DRAWDOWN].value

        if abs(drawdown) < max_dd:
            return returns

        # Circuit breaker triggered - go to cash
        self.logger.warning(
            f"Circuit breaker triggered: drawdown {drawdown:.2%} exceeds limit {max_dd:.2%}"
        )

        return np.zeros_like(returns)


@jit(nopython=True, cache=True)
def compute_portfolio_drawdown_numba(returns: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute rolling drawdown and max drawdown."""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max

    max_drawdown = np.min(drawdowns)

    return drawdowns, max_drawdown


class PositionSizer:
    """Position sizing at portfolio level."""

    def __init__(self, method: str = "equal_risk", max_position_size: float = 0.3):
        self.method = method
        self.max_position_size = max_position_size
        self.logger = logging.getLogger(__name__)

    def compute_position_sizes(
        self, signals: np.ndarray, n_positions: int, total_capital: float = 1.0
    ) -> np.ndarray:
        """Compute position sizes."""
        if self.method == "equal":
            sizes = np.ones(n_positions) / n_positions

        elif self.method == "equal_risk":
            # Allocate based on signal strength
            signal_strength = np.abs(signals)
            total_strength = np.sum(signal_strength)

            if total_strength > 1e-10:
                sizes = signal_strength / total_strength
            else:
                sizes = np.zeros_like(signals)

        elif self.method == "signal_weighted":
            # Weight by signal magnitude
            weights = np.abs(signals)
            total_weight = np.sum(weights)

            if total_weight > 1e-10:
                sizes = weights / total_weight * total_capital
            else:
                sizes = np.zeros_like(signals)

        else:
            sizes = np.zeros_like(signals)

        # Apply max position size limit
        sizes = np.clip(sizes, 0, self.max_position_size)

        return sizes

    def apply_correlation_limits(
        self, weights: np.ndarray, correlation_matrix: np.ndarray, max_correlation: float = 0.8
    ) -> np.ndarray:
        """Reduce weights for highly correlated assets."""
        n = len(weights)
        adjusted_weights = weights.copy()

        for i in range(n):
            for j in range(i + 1, n):
                if abs(correlation_matrix[i, j]) > max_correlation:
                    # Reduce weights of correlated assets
                    reduction = (abs(correlation_matrix[i, j]) - max_correlation) / (
                        1 - max_correlation
                    )
                    adjusted_weights[i] *= 1 - reduction * 0.5
                    adjusted_weights[j] *= 1 - reduction * 0.5

        # Re-normalize
        if np.sum(adjusted_weights) > 1e-10:
            adjusted_weights = adjusted_weights / np.sum(adjusted_weights)

        return adjusted_weights


class DynamicRiskAdjuster:
    """Dynamically adjusts risk based on market conditions."""

    def __init__(
        self,
        base_risk: float = 1.0,
        volatility_lookback: int = 63,
        max_risk_multiplier: float = 2.0,
        min_risk_multiplier: float = 0.5,
    ):
        self.base_risk = base_risk
        self.volatility_lookback = volatility_lookback
        self.max_risk_multiplier = max_risk_multiplier
        self.min_risk_multiplier = min_risk_multiplier
        self.logger = logging.getLogger(__name__)

    def compute_risk_multiplier(
        self, returns: np.ndarray, current_volatility: Optional[float] = None
    ) -> float:
        """Compute dynamic risk multiplier based on volatility."""
        if current_volatility is None:
            if len(returns) < self.volatility_lookback:
                return 1.0

            recent_returns = returns[-self.volatility_lookback :]
            current_volatility = np.std(recent_returns) * np.sqrt(252)

        # Base multiplier: inverse of volatility
        if current_volatility > 1e-10:
            risk_multiplier = 0.15 / current_volatility
        else:
            risk_multiplier = 1.0

        # Clamp to allowed range
        risk_multiplier = np.clip(
            risk_multiplier, self.min_risk_multiplier, self.max_risk_multiplier
        )

        return risk_multiplier

    def adjust_positions(self, positions: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Adjust positions based on dynamic risk."""
        risk_multiplier = self.compute_risk_multiplier(returns)

        adjusted_positions = positions * risk_multiplier * self.base_risk

        return adjusted_positions
