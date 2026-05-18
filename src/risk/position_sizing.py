# -*- coding: utf-8 -*-
"""
Position Sizing Module

Provides position sizing methods and utilities for risk management.
Implements multiple sizing strategies including fixed fractional, Kelly criterion,
ATR-based sizing, and volatility-adjusted position sizing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SizingMethod(Enum):
    """Position sizing methods."""

    FIXED_FRACTIONAL = "fixed_fractional"
    FIXED_AMOUNT = "fixed_amount"
    KELLY = "kelly"
    KELLY_INFORMATION = "kelly_information"  # C5: info-theoretic Kelly
    ATR_BASED = "atr"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    RISK_PARITY = "risk_parity"


@dataclass
class PositionSizeResult:
    """
    Result of position sizing calculation.

    Attributes:
        size: Position size in shares/contracts
        risk_amount: Dollar amount at risk
        risk_percent: Percentage of equity at risk
        stop_price: Calculated stop loss price
        method: Sizing method used
        metadata: Additional sizing information
    """

    size: float
    risk_amount: float
    risk_percent: float
    stop_price: float
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "size": self.size,
            "risk_amount": self.risk_amount,
            "risk_percent": self.risk_percent,
            "stop_price": self.stop_price,
            "method": self.method,
            "metadata": self.metadata,
        }


@dataclass
class PositionSizer:
    """
    Position Sizer Configuration.

    Supports multiple sizing methods with risk management constraints.

    Attributes:
        method: Primary sizing method
        risk_per_trade: Maximum risk per trade as fraction of equity
        fixed_amount: Fixed dollar amount per trade (for fixed_amount method)
        kelly_fraction: Fraction of Kelly criterion to use (for kelly method)
        atr_multiplier: ATR multiplier for stop distance
        max_position_size: Maximum position size as fraction of equity
        min_position_size: Minimum position size in dollars
        max_risk_per_trade: Maximum risk per trade (hard limit)
        volatility_lookback: Lookback period for volatility calculations
    """

    method: str = "fixed_fractional"
    risk_per_trade: float = 0.02  # 2% risk per trade
    fixed_amount: float = 10000.0
    kelly_fraction: float = 0.25
    atr_multiplier: float = 2.0
    max_position_size: float = 0.20  # 20% of equity max
    min_position_size: float = 100.0
    max_risk_per_trade: float = 0.05  # 5% hard limit
    volatility_lookback: int = 20

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.risk_per_trade <= 0 or self.risk_per_trade > self.max_risk_per_trade:
            raise ValueError(f"risk_per_trade must be between 0 and {self.max_risk_per_trade}")
        if self.kelly_fraction <= 0 or self.kelly_fraction > 1:
            raise ValueError("kelly_fraction must be between 0 and 1")
        if self.atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be positive")

    def calculate(
        self,
        equity: float,
        entry_price: float,
        stop_price: Optional[float] = None,
        atr: Optional[float] = None,
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None,
        direction: str = "long",
        **kwargs,
    ) -> PositionSizeResult:
        """
        Calculate position size based on configured method.

        Args:
            equity: Current account equity
            entry_price: Planned entry price
            stop_price: Stop loss price (optional, calculated if not provided)
            atr: Current ATR value (for atr method)
            volatility: Current volatility (for volatility_adjusted method)
            win_rate: Historical win rate (for kelly method)
            avg_win_loss_ratio: Average win/loss ratio (for kelly method)
            direction: Trade direction - 'long' or 'short'
            **kwargs: Additional parameters for specific methods

        Returns:
            PositionSizeResult with calculated size and metadata
        """
        # Calculate stop price if not provided
        if stop_price is None:
            if atr is not None:
                if direction == "long":
                    stop_price = entry_price - (atr * self.atr_multiplier)
                else:
                    stop_price = entry_price + (atr * self.atr_multiplier)
            else:
                # Default to 5% stop
                if direction == "long":
                    stop_price = entry_price * 0.95
                else:
                    stop_price = entry_price * 1.05

        # Ensure stop price is valid for direction
        if direction == "long":
            if stop_price >= entry_price:
                raise ValueError("Stop price must be below entry price for long positions")
            risk_per_share = entry_price - stop_price
        else:  # short
            if stop_price <= entry_price:
                raise ValueError("Stop price must be above entry price for short positions")
            risk_per_share = stop_price - entry_price

        # Apply sizing method
        if self.method == "fixed_fractional":
            result = self._fixed_fractional(equity, entry_price, stop_price, risk_per_share)
        elif self.method == "fixed_amount":
            result = self._fixed_amount(entry_price, stop_price, risk_per_share)
        elif self.method == "kelly":
            result = self._kelly(
                equity, entry_price, stop_price, risk_per_share, win_rate, avg_win_loss_ratio
            )
        elif self.method == "kelly_information":
            model_prob = kwargs.get("model_probability", 0.50)
            result = self._kelly_information(
                equity, entry_price, stop_price, risk_per_share, model_prob
            )
        elif self.method == "atr":
            result = self._atr_based(equity, entry_price, atr, risk_per_share, direction=direction)
        elif self.method == "volatility_adjusted":
            result = self._volatility_adjusted(
                equity, entry_price, stop_price, risk_per_share, volatility
            )
        elif self.method == "risk_parity":
            result = self._risk_parity(equity, entry_price, stop_price, risk_per_share, **kwargs)
        else:
            raise ValueError(f"Unknown sizing method: {self.method}")

        # Apply constraints
        result = self._apply_constraints(result, equity, entry_price)

        return result

    def _fixed_fractional(
        self, equity: float, entry_price: float, stop_price: float, risk_per_share: float
    ) -> PositionSizeResult:
        """Calculate position size using fixed fractional method."""
        risk_amount = equity * self.risk_per_trade
        size = risk_amount / risk_per_share

        return PositionSizeResult(
            size=size,
            risk_amount=risk_amount,
            risk_percent=self.risk_per_trade,
            stop_price=stop_price,
            method="fixed_fractional",
            metadata={"risk_per_share": risk_per_share},
        )

    def _fixed_amount(
        self, entry_price: float, stop_price: float, risk_per_share: float
    ) -> PositionSizeResult:
        """Calculate position size using fixed dollar amount."""
        size = self.fixed_amount / entry_price
        risk_amount = size * risk_per_share

        return PositionSizeResult(
            size=size,
            risk_amount=risk_amount,
            risk_percent=risk_amount / self.fixed_amount,
            stop_price=stop_price,
            method="fixed_amount",
            metadata={"fixed_amount": self.fixed_amount},
        )

    def _kelly(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        risk_per_share: float,
        win_rate: Optional[float],
        avg_win_loss_ratio: Optional[float],
    ) -> PositionSizeResult:
        """
        Calculate position size using Kelly criterion.

        Kelly % = W - [(1 - W) / R]
        Where:
            W = Win rate
            R = Average win / Average loss

        Uses kelly_fraction to reduce full Kelly for safety.

        Also delegates to KellyAllocator (C5) for full information-theoretic
        Kelly calculation when available.
        """
        # Default values if not provided
        if win_rate is None:
            win_rate = 0.50
        if avg_win_loss_ratio is None:
            avg_win_loss_ratio = 1.5

        # Calculate Kelly percentage (classic formula)
        kelly_pct = win_rate - ((1 - win_rate) / avg_win_loss_ratio)

        # Apply Kelly fraction for safety
        adjusted_kelly = max(0, kelly_pct * self.kelly_fraction)

        # Calculate position size
        risk_amount = equity * adjusted_kelly
        size = risk_amount / risk_per_share if risk_per_share > 0 else 0

        return PositionSizeResult(
            size=size,
            risk_amount=risk_amount,
            risk_percent=adjusted_kelly,
            stop_price=stop_price,
            method="kelly",
            metadata={
                "win_rate": win_rate,
                "avg_win_loss_ratio": avg_win_loss_ratio,
                "full_kelly": kelly_pct,
                "adjusted_kelly": adjusted_kelly,
            },
        )

    def _kelly_information(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        risk_per_share: float,
        model_probability: float,
    ) -> PositionSizeResult:
        """Calculate position size using information-theoretic Kelly (C5).

        Uses KellyAllocator from kelly_allocator.py for full Kelly calculation
        with edge estimation from model probability.

        Args:
            equity: Current account equity.
            entry_price: Planned entry price.
            stop_price: Stop loss price.
            risk_per_share: Dollar risk per share.
            model_probability: ML model probability of profitable outcome.

        Returns:
            PositionSizeResult.
        """
        from src.risk.kelly_allocator import KellyAllocator

        allocator = KellyAllocator(
            kelly_fraction=self.kelly_fraction,
            max_allocation=self.max_risk_per_trade,
            method="classic",
        )
        edge = allocator.estimate_edge_from_probability(
            model_probability=model_probability,
        )
        allocation = allocator.compute(edge, equity)

        risk_amount = equity * allocation.adjusted_fraction
        size = risk_amount / risk_per_share if risk_per_share > 0 else 0

        return PositionSizeResult(
            size=size,
            risk_amount=risk_amount,
            risk_percent=allocation.adjusted_fraction,
            stop_price=stop_price,
            method="kelly_information",
            metadata={
                "full_kelly": allocation.full_kelly_fraction,
                "adjusted_kelly": allocation.adjusted_fraction,
                "model_probability": model_probability,
                "edge": {
                    "win_rate": allocation.edge.win_rate,
                    "avg_win": allocation.edge.avg_win,
                    "avg_loss": allocation.edge.avg_loss,
                    "profit_factor": allocation.edge.profit_factor,
                },
            },
        )

    def _atr_based(
        self,
        equity: float,
        entry_price: float,
        atr: Optional[float],
        risk_per_share: float,
        direction: str = "long",
    ) -> PositionSizeResult:
        """Calculate position size based on ATR."""
        if atr is None:
            # Estimate ATR if not provided
            atr = entry_price * 0.02  # Default 2% ATR

        risk_amount = equity * self.risk_per_trade
        size = risk_amount / risk_per_share if risk_per_share > 0 else 0

        stop_price = (
            entry_price - (atr * self.atr_multiplier)
            if direction == "long"
            else entry_price + (atr * self.atr_multiplier)
        )

        return PositionSizeResult(
            size=size,
            risk_amount=risk_amount,
            risk_percent=self.risk_per_trade,
            stop_price=stop_price,
            method="atr",
            metadata={
                "atr": atr,
                "atr_multiplier": self.atr_multiplier,
                "risk_per_share": risk_per_share,
            },
        )

    def _volatility_adjusted(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        risk_per_share: float,
        volatility: Optional[float],
    ) -> PositionSizeResult:
        """
        Calculate position size adjusted for volatility.

        Higher volatility = smaller position
        Lower volatility = larger position
        """
        if volatility is None:
            volatility = 0.20  # Default 20% annualized volatility

        # Target volatility contribution (e.g., 1% of portfolio vol)
        target_vol_contribution = 0.01

        # Adjust position size based on volatility
        # Scale inversely with volatility
        vol_scalar = target_vol_contribution / volatility if volatility > 0 else 1.0
        vol_scalar = min(max(vol_scalar, 0.1), 2.0)  # Bound between 0.1x and 2.0x

        adjusted_risk = self.risk_per_trade * vol_scalar
        adjusted_risk = min(adjusted_risk, self.max_risk_per_trade)

        risk_amount = equity * adjusted_risk
        size = risk_amount / risk_per_share if risk_per_share > 0 else 0

        return PositionSizeResult(
            size=size,
            risk_amount=risk_amount,
            risk_percent=adjusted_risk,
            stop_price=stop_price,
            method="volatility_adjusted",
            metadata={
                "volatility": volatility,
                "vol_scalar": vol_scalar,
                "target_vol_contribution": target_vol_contribution,
            },
        )

    def _risk_parity(
        self, equity: float, entry_price: float, stop_price: float, risk_per_share: float, **kwargs
    ) -> PositionSizeResult:
        """
        Calculate position size using risk parity approach.

        Allocates risk equally across positions.
        """
        # Get number of positions for risk parity allocation
        num_positions = kwargs.get("num_positions", 1)

        # Allocate equal risk to each position
        risk_allocation = self.risk_per_trade / num_positions
        risk_amount = equity * risk_allocation

        size = risk_amount / risk_per_share if risk_per_share > 0 else 0

        return PositionSizeResult(
            size=size,
            risk_amount=risk_amount,
            risk_percent=risk_allocation,
            stop_price=stop_price,
            method="risk_parity",
            metadata={"num_positions": num_positions, "risk_allocation": risk_allocation},
        )

    def _apply_constraints(
        self, result: PositionSizeResult, equity: float, entry_price: float
    ) -> PositionSizeResult:
        """Apply position size constraints."""
        # Guard against zero or negative inputs
        if entry_price <= 0 or equity <= 0:
            return result

        # Maximum position size
        max_size = (equity * self.max_position_size) / entry_price
        if result.size > max_size:
            result.size = max_size
            result.risk_amount = max_size * abs(entry_price - result.stop_price)
            if equity > 0:
                result.risk_percent = result.risk_amount / equity
            result.metadata["constrained_by"] = "max_position_size"

        # Minimum position size
        min_size = self.min_position_size / entry_price
        if result.size < min_size:
            result.size = min_size
            result.metadata["constrained_by"] = "min_position_size"

        # Maximum risk per trade
        if result.risk_percent > self.max_risk_per_trade:
            result.risk_percent = self.max_risk_per_trade
            result.risk_amount = equity * self.max_risk_per_trade
            stop_distance = abs(entry_price - result.stop_price)
            if stop_distance > 0:
                result.size = result.risk_amount / stop_distance
            result.metadata["constrained_by"] = "max_risk_per_trade"

        # Round size to reasonable precision
        result.size = round(result.size, 2)

        return result


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_per_trade: float = 0.02,
    method: str = "fixed_fractional",
    **kwargs,
) -> PositionSizeResult:
    """
    Convenience function for quick position sizing calculations.

    Args:
        equity: Current account equity
        entry_price: Planned entry price
        stop_price: Stop loss price
        risk_per_trade: Risk per trade as fraction of equity
        method: Sizing method to use
        **kwargs: Additional parameters for PositionSizer

    Returns:
        PositionSizeResult with calculated size
    """
    sizer = PositionSizer(method=method, risk_per_trade=risk_per_trade, **kwargs)
    return sizer.calculate(equity, entry_price, stop_price, **kwargs)


def calculate_stop_loss(
    entry_price: float,
    atr: Optional[float] = None,
    method: str = "atr",
    multiplier: float = 2.0,
    fixed_percent: float = 0.05,
    direction: str = "long",
) -> float:
    """
    Calculate stop loss price.

    Args:
        entry_price: Entry price
        atr: ATR value (for atr method)
        method: Stop loss method ('atr', 'percent', 'support')
        multiplier: ATR multiplier (for atr method)
        fixed_percent: Fixed percentage (for percent method)
        direction: Trade direction - 'long' or 'short'

    Returns:
        Stop loss price
    """
    if method == "atr":
        if atr is None:
            atr = entry_price * 0.02  # Default 2% ATR
        if direction == "short":
            return entry_price + (atr * multiplier)
        return entry_price - (atr * multiplier)
    elif method == "percent":
        if direction == "short":
            return entry_price * (1 + fixed_percent)
        return entry_price * (1 - fixed_percent)
    elif method == "support":
        # Placeholder - actual support level should be passed
        if direction == "short":
            return entry_price * (1 + fixed_percent)
        return entry_price * (1 - fixed_percent)
    else:
        if direction == "short":
            return entry_price * (1 + fixed_percent)
        return entry_price * (1 - fixed_percent)


def calculate_take_profits(
    entry_price: float,
    stop_price: float,
    risk_reward_ratios: Optional[List[float]] = None,
    fibonacci_levels: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Calculate take profit levels.

    Args:
        entry_price: Entry price
        stop_price: Stop loss price
        risk_reward_ratios: List of risk/reward ratios (default: [1.0, 1.5, 2.0])
        fibonacci_levels: Fibonacci extension levels (optional)

    Returns:
        Dictionary with take profit levels
    """
    ratios: List[float] = risk_reward_ratios if risk_reward_ratios is not None else [1.0, 1.5, 2.0]

    risk = entry_price - stop_price

    take_profits: Dict[str, float] = {}
    for i, rr in enumerate(ratios, 1):
        take_profits[f"tp{i}"] = entry_price + (risk * rr)

    if fibonacci_levels:
        for level in fibonacci_levels:
            # Fibonacci extensions: 1.272, 1.414, 1.618
            take_profits[f"fib_{level}"] = entry_price + (risk * (level - 1))

    return take_profits


def calculate_shares_to_trade(
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_percent: float = 0.02,
    max_position_percent: float = 0.20,
) -> Dict[str, float]:
    """
    Calculate number of shares to trade with risk management.

    Args:
        equity: Account equity
        entry_price: Entry price
        stop_price: Stop loss price
        risk_percent: Maximum risk per trade
        max_position_percent: Maximum position size as percent of equity

    Returns:
        Dictionary with shares, risk amount, and position value
    """
    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0:
        raise ValueError("Stop price must be below entry price")

    # Calculate based on risk
    risk_amount = equity * risk_percent
    shares_by_risk = risk_amount / risk_per_share

    # Calculate based on max position
    max_position_value = equity * max_position_percent
    shares_by_position = max_position_value / entry_price

    # Use the smaller of the two
    shares = min(shares_by_risk, shares_by_position)

    # Round down to whole shares
    shares = int(shares)

    return {
        "shares": shares,
        "risk_amount": shares * risk_per_share,
        "position_value": shares * entry_price,
        "risk_percent": (shares * risk_per_share) / equity,
        "position_percent": (shares * entry_price) / equity,
    }
