"""
Position Manager

Handles position sizing, risk management, and portfolio-level controls.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..patterns.base import SignalDirection
from ..risk.crash_factor import CrashFactorModel, CrashFactorConfig


class PositionStatus(Enum):
    """Status of a trading position."""

    PENDING = "Pending"
    OPEN = "Open"
    CLOSED = "Closed"
    CANCELLED = "Cancelled"


@dataclass
class Position:
    """
    Trading position representation.

    Attributes:
        id: Unique position identifier
        pattern_name: Name of pattern that generated the signal
        direction: Long or Short
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit_1: First take profit target
        take_profit_2: Second take profit target (optional)
        take_profit_3: Third take profit target (optional)
        size: Position size in shares/contracts
        status: Current position status
        entry_time: Time when position was opened
        exit_price: Exit price (if closed)
        exit_time: Time when position was closed (if closed)
        pnl: Profit/loss (if closed)
        pnl_pct: Profit/loss percentage (if closed)
        metadata: Additional position information
    """

    id: str
    pattern_name: str
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    size: float = 0.0
    status: PositionStatus = PositionStatus.PENDING
    entry_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    crash_probability: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary format."""
        return {
            "id": self.id,
            "pattern_name": self.pattern_name,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "take_profit_3": self.take_profit_3,
            "size": self.size,
            "status": self.status.value,
            "entry_time": str(self.entry_time) if self.entry_time else None,
            "exit_price": self.exit_price,
            "exit_time": str(self.exit_time) if self.exit_time else None,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "metadata": self.metadata,
        }


@dataclass
class PositionSizer:
    """
    Position sizing configuration.

    Attributes:
        method: Sizing method ('fixed_fractional', 'fixed_amount', 'kelly', 'atr')
        risk_per_trade: Risk per trade as fraction of equity (for fixed_fractional)
        fixed_amount: Fixed dollar amount per trade
        kelly_fraction: Fraction of Kelly criterion to use
        atr_multiplier: ATR multiplier for stop distance (for atr method)
        max_position_size: Maximum position size as fraction of equity
        min_position_size: Minimum position size
    """

    method: str = "fixed_fractional"
    risk_per_trade: float = 0.02  # 2% risk per trade
    fixed_amount: float = 10000.0
    kelly_fraction: float = 0.25
    atr_multiplier: float = 2.0
    max_position_size: float = 0.20  # 20% of equity max
    min_position_size: float = 100.0


class PositionManager:
    """
    Position Manager

    Handles position sizing, risk management, and portfolio controls.
    """

    def __init__(
        self,
        initial_equity: float = 100000.0,
        position_sizer: Optional[PositionSizer] = None,
        max_open_positions: int = 5,
        max_correlated_positions: int = 2,
        max_daily_trades: int = 10,
        max_sector_exposure: float = 0.30,
        use_take_profit_1: bool = True,
        use_take_profit_2: bool = False,
        use_take_profit_3: bool = False,
        use_crash_filter: bool = True,
        crash_threshold: float = 0.10,
    ):
        """
        Initialize Position Manager.

        Args:
            initial_equity: Starting equity amount
            position_sizer: Position sizing configuration
            max_open_positions: Maximum concurrent open positions
            max_correlated_positions: Maximum positions in correlated assets
            max_daily_trades: Maximum trades per day
            max_sector_exposure: Maximum exposure to single sector (fraction)
            use_take_profit_1: Enable exit at TP1 level
            use_take_profit_2: Enable exit at TP2 level
            use_take_profit_3: Enable exit at TP3 level
            use_crash_filter: Enable crash factor pre-trade filter (R17)
            crash_threshold: Crash probability threshold for filtering
        """
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.position_sizer = position_sizer or PositionSizer()
        self.max_open_positions = max_open_positions
        self.max_correlated_positions = max_correlated_positions
        self.max_daily_trades = max_daily_trades
        self.max_sector_exposure = max_sector_exposure
        self.use_take_profit_1 = use_take_profit_1
        self.use_take_profit_2 = use_take_profit_2
        self.use_take_profit_3 = use_take_profit_3
        self.use_crash_filter = use_crash_filter
        self.crash_threshold = crash_threshold
        
        # Crash factor model for pre-trade filtering (R17/H5)
        self.crash_model = CrashFactorModel(CrashFactorConfig(crash_threshold=crash_threshold))

        # Track positions
        self.positions: Dict[str, Position] = {}
        self.open_positions: List[str] = []
        self.closed_positions: List[str] = []
        self.daily_trade_count: Dict[str, int] = {}
        self.position_counter = 0

        # Performance tracking
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        current_equity: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> float:
        """
        Calculate position size based on sizing method.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            current_equity: Current equity (default: self.equity)
            atr: Current ATR value (for ATR-based sizing)

        Returns:
            Position size in shares/contracts
        """
        equity = current_equity or self.equity
        risk_distance = abs(entry_price - stop_loss)

        if risk_distance == 0:
            return self.position_sizer.min_position_size / entry_price

        if self.position_sizer.method == "fixed_fractional":
            # Risk fixed fraction of equity
            risk_amount = equity * self.position_sizer.risk_per_trade
            size = risk_amount / risk_distance

        elif self.position_sizer.method == "fixed_amount":
            # Fixed dollar amount per trade
            size = self.position_sizer.fixed_amount / entry_price

        elif self.position_sizer.method == "kelly":
            # Kelly criterion (requires win rate and avg win/loss)
            win_rate = self.winning_trades / max(1, self.total_trades)
            avg_win = self._get_avg_win()
            avg_loss = self._get_avg_loss()

            if avg_loss > 0:
                kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
                kelly = max(0, kelly) * self.position_sizer.kelly_fraction
                risk_amount = equity * kelly
                size = risk_amount / risk_distance
            else:
                size = self.position_sizer.fixed_amount / entry_price

        elif self.position_sizer.method == "atr":
            # ATR-based position sizing
            if atr is None:
                atr = risk_distance  # Use stop distance as proxy
            risk_amount = equity * self.position_sizer.risk_per_trade
            atr_distance = atr * self.position_sizer.atr_multiplier
            size = risk_amount / atr_distance

        else:
            # Default to fixed fractional
            risk_amount = equity * self.position_sizer.risk_per_trade
            size = risk_amount / risk_distance

        # Apply position limits
        max_size = (equity * self.position_sizer.max_position_size) / entry_price
        min_size = self.position_sizer.min_position_size / entry_price

        size = min(size, max_size)
        size = max(size, min_size)

        return size

    def _get_avg_win(self) -> float:
        """Get average winning trade P&L."""
        wins = [
            p.pnl
            for p in self.positions.values()
            if p.status == PositionStatus.CLOSED and p.pnl and p.pnl > 0
        ]
        return sum(wins) / len(wins) if wins else 0.0

    def _get_avg_loss(self) -> float:
        """Get average losing trade P&L."""
        losses = [
            abs(p.pnl)
            for p in self.positions.values()
            if p.status == PositionStatus.CLOSED and p.pnl and p.pnl < 0
        ]
        return sum(losses) / len(losses) if losses else 0.0

    def can_open_position(self, signal: Any, timestamp: Optional[pd.Timestamp] = None, price_data: Optional[pd.DataFrame] = None) -> tuple:
        """
        Check if a new position can be opened.

        Args:
            signal: Trading signal
            timestamp: Signal timestamp
            price_data: Optional OHLCV price data for crash factor analysis

        Returns:
            Tuple of (can_open: bool, reason: str)
        """
        # Check max open positions
        if len(self.open_positions) >= self.max_open_positions:
            return False, "Maximum open positions reached"

        # Check daily trade limit
        if timestamp:
            date_str = str(timestamp.date()) if hasattr(timestamp, "date") else str(timestamp)[:10]
            daily_count = self.daily_trade_count.get(date_str, 0)
            if daily_count >= self.max_daily_trades:
                return False, "Maximum daily trades reached"

        # H5: Check crash factor filter (R17)
        if self.use_crash_filter and price_data is not None and len(price_data) > 60:
            try:
                crash_result = self.crash_model.predict(price_data, symbol="UNKNOWN")
                if crash_result.crash_probability >= self.crash_threshold:
                    return False, f"Crash risk too high (p={crash_result.crash_probability:.2%})"
            except Exception:
                # If crash model fails, allow trade but log warning
                pass

        return True, "Position allowed"

    def open_position(
        self, signal: Any, timestamp: Optional[pd.Timestamp] = None, atr: Optional[float] = None, price_data: Optional[pd.DataFrame] = None
    ) -> Optional[Position]:
        """
        Open a new position from a signal.

        Args:
            signal: Trading signal
            timestamp: Entry timestamp
            atr: Current ATR value
            price_data: Optional OHLCV data for crash factor analysis

        Returns:
            Position object or None if cannot open
        """
        can_open, reason = self.can_open_position(signal, timestamp, price_data)
        if not can_open:
            return None

        # Generate position ID
        self.position_counter += 1
        position_id = f"POS_{self.position_counter:06d}"

        # Calculate position size
        size = self.calculate_position_size(
            entry_price=signal.entry_price, stop_loss=signal.stop_loss, atr=atr
        )

        # Create position
        pattern_name = (
            signal.pattern_name
            if hasattr(signal, "pattern_name")
            else signal.patterns[0]
            if hasattr(signal, "patterns") and len(signal.patterns) > 0
            else "Unknown"
        )

        # Calculate crash probability for the position
        crash_prob = None
        if self.use_crash_filter and price_data is not None and len(price_data) > 60:
            try:
                crash_result = self.crash_model.predict(price_data, symbol="UNKNOWN")
                crash_prob = crash_result.crash_probability
            except Exception:
                pass

        position = Position(
            id=position_id,
            pattern_name=pattern_name,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            take_profit_3=signal.take_profit_3,
            size=size,
            status=PositionStatus.OPEN,
            entry_time=timestamp,
            crash_probability=crash_prob,
            metadata=signal.metadata.copy() if hasattr(signal, "metadata") else {},
        )

        # Track position
        self.positions[position_id] = position
        self.open_positions.append(position_id)

        # Update daily trade count
        if timestamp:
            date_str = str(timestamp.date()) if hasattr(timestamp, "date") else str(timestamp)[:10]
            self.daily_trade_count[date_str] = self.daily_trade_count.get(date_str, 0) + 1

        return position

    def check_exit(
        self,
        position: Position,
        current_price: float,
        current_high: float,
        current_low: float,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> tuple:
        """
        Check if position should be closed.

        Args:
            position: Position to check
            current_price: Current close price
            current_high: Current bar high
            current_low: Current bar low
            timestamp: Current timestamp

        Returns:
            Tuple of (should_exit: bool, exit_reason: str, exit_price: float)
        """
        if position.status != PositionStatus.OPEN:
            return False, None, None

        if position.direction == SignalDirection.LONG:
            # Check stop loss (always active)
            if current_low <= position.stop_loss:
                return True, "Stop Loss", position.stop_loss

            # Check take profit levels based on configuration
            if (
                self.use_take_profit_1
                and position.take_profit_1
                and current_high >= position.take_profit_1
            ):
                return True, "Take Profit 1", position.take_profit_1
            if (
                self.use_take_profit_2
                and position.take_profit_2
                and current_high >= position.take_profit_2
            ):
                return True, "Take Profit 2", position.take_profit_2
            if (
                self.use_take_profit_3
                and position.take_profit_3
                and current_high >= position.take_profit_3
            ):
                return True, "Take Profit 3", position.take_profit_3

        else:  # SHORT
            # Check stop loss (always active)
            if current_high >= position.stop_loss:
                return True, "Stop Loss", position.stop_loss

            # Check take profit levels based on configuration
            if (
                self.use_take_profit_1
                and position.take_profit_1
                and current_low <= position.take_profit_1
            ):
                return True, "Take Profit 1", position.take_profit_1
            if (
                self.use_take_profit_2
                and position.take_profit_2
                and current_low <= position.take_profit_2
            ):
                return True, "Take Profit 2", position.take_profit_2
            if (
                self.use_take_profit_3
                and position.take_profit_3
                and current_low <= position.take_profit_3
            ):
                return True, "Take Profit 3", position.take_profit_3

        return False, None, None

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> Optional[Position]:
        """
        Close a position.

        Args:
            position_id: Position ID to close
            exit_price: Exit price
            exit_reason: Reason for exit
            timestamp: Exit timestamp

        Returns:
            Updated Position object or None
        """
        if position_id not in self.positions:
            return None

        position = self.positions[position_id]

        if position.status != PositionStatus.OPEN:
            return None

        # Calculate P&L
        if position.direction == SignalDirection.LONG:
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size

        pnl_pct = pnl / (position.entry_price * position.size)

        # Update position
        position.status = PositionStatus.CLOSED
        position.exit_price = exit_price
        position.exit_time = timestamp
        position.pnl = pnl
        position.pnl_pct = pnl_pct
        position.metadata["exit_reason"] = exit_reason

        # Update tracking
        self.open_positions.remove(position_id)
        self.closed_positions.append(position_id)
        self.equity += pnl
        self.total_pnl += pnl
        self.total_trades += 1

        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        return position

    def get_open_positions(self) -> List[Position]:
        """Get list of all open positions."""
        return [self.positions[pid] for pid in self.open_positions]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        if self.total_trades == 0:
            return {"total_trades": 0, "equity": self.equity, "return_pct": 0.0}

        win_rate = self.winning_trades / self.total_trades
        avg_win = self._get_avg_win()
        avg_loss = self._get_avg_loss()

        # Calculate profit factor
        total_wins = sum(
            p.pnl
            for p in self.positions.values()
            if p.status == PositionStatus.CLOSED and p.pnl and p.pnl > 0
        )
        total_losses = abs(
            sum(
                p.pnl
                for p in self.positions.values()
                if p.status == PositionStatus.CLOSED and p.pnl and p.pnl < 0
            )
        )
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        return {
            "initial_equity": self.initial_equity,
            "equity": self.equity,
            "total_pnl": self.total_pnl,
            "return_pct": (self.equity - self.initial_equity) / self.initial_equity * 100,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "open_positions": len(self.open_positions),
        }
