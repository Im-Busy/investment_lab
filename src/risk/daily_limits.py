# -*- coding: utf-8 -*-
"""
Daily Limits Module

Provides daily loss limiting and circuit breaker functionality for risk management.
Implements daily, weekly, and monthly loss limits with automatic trading halts.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


class TradingState(Enum):
    """Trading state based on limits."""

    ACTIVE = "active"
    WARNING = "warning"
    HALTED = "halted"
    COOLING_OFF = "cooling_off"


class LimitType(Enum):
    """Types of trading limits."""

    DAILY_LOSS = "daily_loss"
    DAILY_TRADES = "daily_trades"
    DAILY_DRAWDOWN = "daily_drawdown"
    WEEKLY_LOSS = "weekly_loss"
    WEEKLY_DRAWDOWN = "weekly_drawdown"
    MONTHLY_LOSS = "monthly_loss"
    MONTHLY_DRAWDOWN = "monthly_drawdown"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    PORTFOLIO_HEAT = "portfolio_heat"


@dataclass
class DailyLossState:
    """
    Tracks daily loss state and metrics.

    Attributes:
        date: Current trading date
        starting_equity: Equity at start of day
        current_equity: Current equity
        peak_equity: Highest equity reached today
        realized_pnl: Realized profit/loss today
        unrealized_pnl: Unrealized profit/loss
        trade_count: Number of trades today
        win_count: Number of winning trades
        loss_count: Number of losing trades
        consecutive_losses: Current consecutive loss streak
        max_consecutive_losses: Maximum consecutive losses today
        state: Current trading state
        halt_reason: Reason for trading halt (if halted)
    """

    date: date
    starting_equity: float
    current_equity: float
    peak_equity: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    state: TradingState = TradingState.ACTIVE
    halt_reason: Optional[str] = None
    open_position_risks: Dict[str, float] = field(default_factory=dict)

    @property
    def daily_pnl(self) -> float:
        """Total daily P&L."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def daily_pnl_pct(self) -> float:
        """Daily P&L as percentage of starting equity."""
        if self.starting_equity == 0:
            return 0.0
        return self.daily_pnl / self.starting_equity

    @property
    def daily_loss_pct(self) -> float:
        """Daily loss as percentage (negative value)."""
        return min(0, self.daily_pnl_pct)

    @property
    def drawdown_pct(self) -> float:
        """Drawdown from peak equity."""
        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    @property
    def win_rate(self) -> float:
        """Win rate for the day."""
        if self.trade_count == 0:
            return 0.0
        return self.win_count / self.trade_count

    @property
    def portfolio_heat(self) -> float:
        """
        Calculate total portfolio heat as fraction of equity.

        Portfolio Heat = Sum of (Position Risk / Account Equity)
        Maximum Portfolio Heat: 6%
        Warning Level: 4%
        """
        if self.starting_equity == 0:
            return 0.0
        total_risk = sum(self.open_position_risks.values())
        return total_risk / self.starting_equity

    def update_position_risk(self, position_id: str, risk_amount: float) -> None:
        """Update risk for a specific position."""
        self.open_position_risks[position_id] = risk_amount

    def remove_position_risk(self, position_id: str) -> None:
        """Remove position risk when position is closed."""
        self.open_position_risks.pop(position_id, None)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "date": str(self.date),
            "starting_equity": self.starting_equity,
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl_pct,
            "drawdown_pct": self.drawdown_pct,
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_rate,
            "consecutive_losses": self.consecutive_losses,
            "max_consecutive_losses": self.max_consecutive_losses,
            "state": self.state.value,
            "halt_reason": self.halt_reason,
        }


@dataclass
class DailyLossLimiter:
    """
    Daily Loss Limiter Configuration.

    Implements daily, weekly, and monthly loss limits with automatic
    trading halts and cool-off periods.

    Attributes:
        max_daily_loss_pct: Maximum daily loss as fraction (default: 3%)
        max_daily_trades: Maximum trades per day (default: 10)
        max_daily_drawdown_pct: Maximum drawdown from daily peak (default: 5%)
        max_weekly_loss_pct: Maximum weekly loss as fraction (default: 6%)
        max_weekly_drawdown_pct: Maximum weekly drawdown (default: 10%)
        max_monthly_loss_pct: Maximum monthly loss as fraction (default: 10%)
        max_monthly_drawdown_pct: Maximum monthly drawdown (default: 15%)
        max_consecutive_losses: Maximum consecutive losing trades (default: 5)
        cool_off_hours: Hours to wait after halt before resuming (default: 24)
        warning_threshold_pct: Threshold for warning state (default: 75%)
        position_size_reduction_pct: Reduction when near limits (default: 50%)
    """

    max_daily_loss_pct: float = 0.03
    max_daily_trades: int = 10
    max_daily_drawdown_pct: float = 0.05
    max_weekly_loss_pct: float = 0.06
    max_weekly_drawdown_pct: float = 0.10
    max_monthly_loss_pct: float = 0.10
    max_monthly_drawdown_pct: float = 0.15
    max_consecutive_losses: int = 5
    cool_off_hours: int = 24
    warning_threshold_pct: float = 0.75
    position_size_reduction_pct: float = 0.50

    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.max_daily_loss_pct < 1:
            raise ValueError("max_daily_loss_pct must be between 0 and 1")
        if not 0 < self.max_daily_drawdown_pct < 1:
            raise ValueError("max_daily_drawdown_pct must be between 0 and 1")
        if self.max_daily_trades < 1:
            raise ValueError("max_daily_trades must be at least 1")

    def check_limits(self, state: DailyLossState) -> tuple:
        """
        Check if any limits have been breached.

        Args:
            state: Current daily loss state

        Returns:
            Tuple of (is_breached, limit_type, message)
        """
        # Check daily loss limit
        if abs(state.daily_loss_pct) >= self.max_daily_loss_pct:
            return (
                True,
                LimitType.DAILY_LOSS,
                f"Daily loss limit reached: {state.daily_loss_pct:.2%} >= {self.max_daily_loss_pct:.2%}",
            )

        # Check daily drawdown
        if state.drawdown_pct >= self.max_daily_drawdown_pct:
            return (
                True,
                LimitType.DAILY_DRAWDOWN,
                f"Daily drawdown limit reached: {state.drawdown_pct:.2%} >= {self.max_daily_drawdown_pct:.2%}",
            )

        # Check daily trades
        if state.trade_count >= self.max_daily_trades:
            return (
                True,
                LimitType.DAILY_TRADES,
                f"Daily trade limit reached: {state.trade_count} >= {self.max_daily_trades}",
            )

        # Check consecutive losses
        if state.consecutive_losses >= self.max_consecutive_losses:
            return (
                True,
                LimitType.CONSECUTIVE_LOSSES,
                f"Consecutive loss limit reached: {state.consecutive_losses} >= {self.max_consecutive_losses}",
            )

        # Check warning threshold
        warning_threshold = self.max_daily_loss_pct * self.warning_threshold_pct
        if abs(state.daily_loss_pct) >= warning_threshold:
            state.state = TradingState.WARNING
            state.halt_reason = f"Approaching daily loss limit: {state.daily_loss_pct:.2%}"

        return (False, None, None)

    def should_reduce_position(self, state: DailyLossState) -> bool:
        """Check if position size should be reduced due to approaching limits."""
        warning_threshold = self.max_daily_loss_pct * self.warning_threshold_pct
        return abs(state.daily_loss_pct) >= warning_threshold

    def get_position_multiplier(self, state: DailyLossState) -> float:
        """
        Get position size multiplier based on current state.

        Returns a value between 0 and 1 to reduce position sizes
        when approaching limits.
        """
        if state.state == TradingState.HALTED:
            return 0.0

        if not self.should_reduce_position(state):
            return 1.0

        # Calculate how close we are to the limit
        loss_ratio = abs(state.daily_loss_pct) / self.max_daily_loss_pct

        # Linear reduction from warning threshold to limit
        warning_ratio = self.warning_threshold_pct
        if loss_ratio >= warning_ratio:
            reduction = 1.0 - ((loss_ratio - warning_ratio) / (1.0 - warning_ratio))
            reduction *= 1.0 - self.position_size_reduction_pct
            return max(self.position_size_reduction_pct, reduction)

        return 1.0

    def check_portfolio_heat(self, state: DailyLossState) -> tuple:
        """
        Check if portfolio heat exceeds limits.

        Documented limits (from trading_strategy.md):
        - Maximum Portfolio Heat: 6%
        - Warning Level: 4%

        Args:
            state: Current daily loss state

        Returns:
            Tuple of (is_breached, limit_type, message)
        """
        heat = state.portfolio_heat

        if heat >= 0.06:
            return (
                True,
                LimitType.PORTFOLIO_HEAT,
                f"Portfolio heat limit reached: {heat:.2%} >= 6.00%",
            )

        if heat >= 0.04:
            state.state = TradingState.WARNING
            state.halt_reason = f"Portfolio heat warning: {heat:.2%} >= 4.00%"

        return (False, None, None)


@dataclass
class CircuitBreaker:
    """
    Circuit Breaker for automatic trading halts.

    Implements multiple levels of circuit breakers that halt trading
    when certain conditions are met, with automatic reset mechanisms.

    Attributes:
        level1_threshold: Level 1 circuit breaker threshold (warning)
        level2_threshold: Level 2 circuit breaker threshold (reduce trading)
        level3_threshold: Level 3 circuit breaker threshold (halt trading)
        level1_cool_off_minutes: Cool-off period for level 1
        level2_cool_off_minutes: Cool-off period for level 2
        level3_cool_off_minutes: Cool-off period for level 3
        auto_reset: Whether to automatically reset after cool-off
    """

    level1_threshold: float = 0.02  # 2% loss triggers warning
    level2_threshold: float = 0.03  # 3% loss triggers reduced trading
    level3_threshold: float = 0.05  # 5% loss triggers halt
    level1_cool_off_minutes: int = 30
    level2_cool_off_minutes: int = 60
    level3_cool_off_minutes: int = 120
    auto_reset: bool = True

    # State tracking
    _current_level: int = field(default=0, init=False)
    _triggered_at: Optional[datetime] = field(default=None, init=False)
    _reset_at: Optional[datetime] = field(default=None, init=False)
    _trigger_count: Dict[int, int] = field(default_factory=dict, init=False)

    @property
    def current_level(self) -> int:
        """Current circuit breaker level."""
        return self._current_level

    @property
    def is_triggered(self) -> bool:
        """Whether circuit breaker is currently triggered."""
        return self._current_level > 0

    @property
    def is_halted(self) -> bool:
        """Whether trading is completely halted."""
        return self._current_level >= 3

    def check(
        self, current_loss_pct: float, current_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Check circuit breaker conditions.

        Args:
            current_loss_pct: Current loss as percentage (negative value)
            current_time: Current datetime (defaults to now)

        Returns:
            Dictionary with circuit breaker status
        """
        if current_time is None:
            current_time = datetime.now()

        # Check for auto-reset
        if self.auto_reset and self._reset_at and current_time >= self._reset_at:
            self._reset()

        # Determine new level
        loss_abs = abs(current_loss_pct)
        new_level = 0

        if loss_abs >= self.level3_threshold:
            new_level = 3
        elif loss_abs >= self.level2_threshold:
            new_level = 2
        elif loss_abs >= self.level1_threshold:
            new_level = 1

        # Update state if level increased
        if new_level > self._current_level:
            self._trigger(new_level, current_time)

        return self.get_status(current_time)

    def _trigger(self, level: int, trigger_time: datetime):
        """Trigger circuit breaker at specified level."""
        self._current_level = level
        self._triggered_at = trigger_time

        # Set reset time based on level
        cool_off_minutes = {
            1: self.level1_cool_off_minutes,
            2: self.level2_cool_off_minutes,
            3: self.level3_cool_off_minutes,
        }

        self._reset_at = trigger_time + timedelta(minutes=cool_off_minutes.get(level, 60))

        # Track trigger count
        self._trigger_count[level] = self._trigger_count.get(level, 0) + 1

    def _reset(self):
        """Reset circuit breaker."""
        self._current_level = 0
        self._triggered_at = None
        self._reset_at = None

    def manual_reset(self):
        """Manually reset the circuit breaker."""
        self._reset()

    def get_status(self, current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get current circuit breaker status.

        Returns:
            Dictionary with status information
        """
        if current_time is None:
            current_time = datetime.now()

        status = {
            "level": self._current_level,
            "is_triggered": self.is_triggered,
            "is_halted": self.is_halted,
            "triggered_at": self._triggered_at.isoformat() if self._triggered_at else None,
            "reset_at": self._reset_at.isoformat() if self._reset_at else None,
            "time_until_reset": None,
            "can_trade": not self.is_halted,
            "trigger_count": dict(self._trigger_count),
        }

        if self._reset_at:
            remaining = self._reset_at - current_time
            status["time_until_reset"] = (
                remaining.total_seconds() if remaining.total_seconds() > 0 else 0
            )

        # Add recommended action
        if self._current_level == 0:
            status["action"] = "normal_trading"
        elif self._current_level == 1:
            status["action"] = "caution_recommended"
        elif self._current_level == 2:
            status["action"] = "reduce_position_sizes"
        else:
            status["action"] = "halt_trading"

        return status

    def get_position_multiplier(self) -> float:
        """
        Get position size multiplier based on circuit breaker level.

        Returns:
            Multiplier between 0 and 1
        """
        if self._current_level == 0:
            return 1.0
        elif self._current_level == 1:
            return 0.75
        elif self._current_level == 2:
            return 0.50
        else:
            return 0.0


class RiskMonitor:
    """
    Comprehensive Risk Monitor combining daily limits and circuit breakers.

    Monitors trading activity and enforces risk limits across multiple
    timeframes (daily, weekly, monthly).
    """

    def __init__(
        self,
        initial_equity: float,
        daily_limiter: Optional[DailyLossLimiter] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        """
        Initialize Risk Monitor.

        Args:
            initial_equity: Starting equity
            daily_limiter: Daily loss limiter configuration
            circuit_breaker: Circuit breaker configuration
        """
        self.initial_equity = initial_equity
        self.daily_limiter = daily_limiter or DailyLossLimiter()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

        # State tracking
        self._daily_states: Dict[str, DailyLossState] = {}
        self._current_date: Optional[date] = None
        self._weekly_start_equity: float = initial_equity
        self._monthly_start_equity: float = initial_equity
        self._trade_history: List[Dict[str, Any]] = []

    @property
    def current_state(self) -> Optional[DailyLossState]:
        """Get current day's state."""
        if self._current_date is None:
            return None
        date_key = str(self._current_date)
        return self._daily_states.get(date_key)

    def start_day(self, equity: float, trading_date: Optional[date] = None):
        """
        Start a new trading day.

        Args:
            equity: Current equity
            trading_date: Trading date (defaults to today)
        """
        if trading_date is None:
            trading_date = date.today()

        self._current_date = trading_date
        date_key = str(trading_date)

        # Check if this is a new week
        if trading_date.weekday() == 0:  # Monday
            self._weekly_start_equity = equity

        # Check if this is a new month
        if trading_date.day == 1:
            self._monthly_start_equity = equity

        # Create new state for the day
        self._daily_states[date_key] = DailyLossState(
            date=trading_date, starting_equity=equity, current_equity=equity, peak_equity=equity
        )

    def update_equity(self, equity: float, unrealized_pnl: float = 0.0):
        """
        Update current equity and check limits.

        Args:
            equity: Current equity
            unrealized_pnl: Current unrealized P&L

        Returns:
            Dictionary with limit check results
        """
        state = self.current_state
        if state is None:
            return {"error": "No active trading day"}

        # Update state
        state.current_equity = equity
        state.unrealized_pnl = unrealized_pnl

        # Update peak equity
        if equity > state.peak_equity:
            state.peak_equity = equity

        # Check limits
        limit_breached, limit_type, message = self.daily_limiter.check_limits(state)

        # Check circuit breaker
        cb_status = self.circuit_breaker.check(state.daily_loss_pct)

        # Update state based on checks
        if limit_breached or cb_status["is_halted"]:
            state.state = TradingState.HALTED
            state.halt_reason = message or cb_status.get("action", "Circuit breaker triggered")

        return {
            "state": state.to_dict(),
            "limit_breached": limit_breached,
            "limit_type": limit_type.value if limit_type else None,
            "message": message,
            "circuit_breaker": cb_status,
            "position_multiplier": self.get_position_multiplier(),
        }

    def record_trade(
        self, pnl: float, direction: str, pattern: str, entry_time: Optional[datetime] = None
    ):
        """
        Record a completed trade.

        Args:
            pnl: Realized profit/loss
            direction: Trade direction ('long' or 'short')
            pattern: Pattern that generated the signal
            entry_time: Trade entry time
        """
        state = self.current_state
        if state is None:
            return

        # Update trade counts
        state.trade_count += 1

        if pnl >= 0:
            state.win_count += 1
            state.consecutive_losses = 0
        else:
            state.loss_count += 1
            state.consecutive_losses += 1
            if state.consecutive_losses > state.max_consecutive_losses:
                state.max_consecutive_losses = state.consecutive_losses

        # Update realized P&L
        state.realized_pnl += pnl

        # Record in history
        self._trade_history.append(
            {
                "date": str(state.date),
                "pnl": pnl,
                "direction": direction,
                "pattern": pattern,
                "entry_time": entry_time.isoformat() if entry_time else None,
                "trade_count": state.trade_count,
                "consecutive_losses": state.consecutive_losses,
            }
        )

    def can_open_trade(self) -> tuple:
        """
        Check if opening a new trade is allowed.

        Returns:
            Tuple of (can_trade, reason)
        """
        state = self.current_state
        if state is None:
            return (False, "No active trading day")

        # Check if halted
        if state.state == TradingState.HALTED:
            return (False, state.halt_reason or "Trading halted")

        # Check daily trade limit
        if state.trade_count >= self.daily_limiter.max_daily_trades:
            return (False, "Daily trade limit reached")

        # Check circuit breaker
        if self.circuit_breaker.is_halted:
            return (False, "Circuit breaker triggered")

        # Check consecutive losses
        if state.consecutive_losses >= self.daily_limiter.max_consecutive_losses:
            return (False, "Maximum consecutive losses reached")

        return (True, "Trading allowed")

    def get_position_multiplier(self) -> float:
        """
        Get combined position size multiplier.

        Combines daily limiter and circuit breaker multipliers.
        """
        state = self.current_state
        limiter_mult = 1.0
        cb_mult = self.circuit_breaker.get_position_multiplier()

        if state:
            limiter_mult = self.daily_limiter.get_position_multiplier(state)

        return min(limiter_mult, cb_mult)

    def get_weekly_summary(self) -> Dict[str, Any]:
        """Get weekly performance summary."""
        state = self.current_state
        if state is None:
            return {"error": "No active trading day"}

        weekly_pnl = state.current_equity - self._weekly_start_equity
        weekly_pnl_pct = (
            weekly_pnl / self._weekly_start_equity if self._weekly_start_equity > 0 else 0
        )

        return {
            "start_equity": self._weekly_start_equity,
            "current_equity": state.current_equity,
            "weekly_pnl": weekly_pnl,
            "weekly_pnl_pct": weekly_pnl_pct,
            "max_weekly_loss_pct": self.daily_limiter.max_weekly_loss_pct,
            "max_weekly_drawdown_pct": self.daily_limiter.max_weekly_drawdown_pct,
            "within_limits": abs(weekly_pnl_pct) < self.daily_limiter.max_weekly_loss_pct,
        }

    def get_monthly_summary(self) -> Dict[str, Any]:
        """Get monthly performance summary."""
        state = self.current_state
        if state is None:
            return {"error": "No active trading day"}

        monthly_pnl = state.current_equity - self._monthly_start_equity
        monthly_pnl_pct = (
            monthly_pnl / self._monthly_start_equity if self._monthly_start_equity > 0 else 0
        )

        return {
            "start_equity": self._monthly_start_equity,
            "current_equity": state.current_equity,
            "monthly_pnl": monthly_pnl,
            "monthly_pnl_pct": monthly_pnl_pct,
            "max_monthly_loss_pct": self.daily_limiter.max_monthly_loss_pct,
            "max_monthly_drawdown_pct": self.daily_limiter.max_monthly_drawdown_pct,
            "within_limits": abs(monthly_pnl_pct) < self.daily_limiter.max_monthly_loss_pct,
        }

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        state = self.current_state
        if state is None:
            return {"error": "No active trading day"}

        can_trade, reason = self.can_open_trade()

        return {
            "daily": state.to_dict(),
            "weekly": self.get_weekly_summary(),
            "monthly": self.get_monthly_summary(),
            "circuit_breaker": self.circuit_breaker.get_status(),
            "can_trade": can_trade,
            "trade_reason": reason,
            "position_multiplier": self.get_position_multiplier(),
            "limits": {
                "max_daily_loss_pct": self.daily_limiter.max_daily_loss_pct,
                "max_daily_trades": self.daily_limiter.max_daily_trades,
                "max_daily_drawdown_pct": self.daily_limiter.max_daily_drawdown_pct,
                "max_consecutive_losses": self.daily_limiter.max_consecutive_losses,
            },
        }
