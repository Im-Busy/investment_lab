# -*- coding: utf-8 -*-
"""
SMC/ICT Reversal Strategy

Implements a stateful, multi-phase reversal strategy based on
Smart Money Concepts / ICT methodology.

Strategy Flow:
1. Daily Reset at 00:00 UTC
2. Track Asian Session Range (00:00-08:00 UTC)
3. Detect Liquidity Sweep post-08:00 UTC
4. Confirm Market Structure Shift (MSS)
5. Validate with HTF bias and IFVG proximity
6. Place limit order at IFVG edge
7. Trade management: BE@1R, scale@2R, target@2.5R

Reference: SMC-ICT-ML-Hybrid-Backtester Brief
"""

from dataclasses import dataclass, field
from datetime import time
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd
from loguru import logger

# Import SMC indicators
from ..indicators.asian_range import AsianRange, detect_asian_range
from ..indicators.ifvg import (
    IFVG,
    IFVGProximity,
    detect_ifvg,
    find_nearest_unfilled_ifvg,
    update_ifvg_status,
)
from ..indicators.liquidity_sweep import SweepInfo, detect_liquidity_sweep
from ..indicators.mss import MSSInfo, detect_mss
from ..indicators.technical import atr


class StrategyState(Enum):
    """Strategy state machine states."""

    WAITING_ASIA = "waiting_asia"
    TRACKING_ASIA = "tracking_asia"
    ASIA_COMPLETE = "asia_complete"
    SWEEP_DETECTED = "sweep_detected"
    MSS_CONFIRMED = "mss_confirmed"
    ENTRY_PENDING = "entry_pending"
    IN_TRADE = "in_trade"
    DAILY_RESET = "daily_reset"


class TradeDirection(Enum):
    """Trade direction."""

    LONG = "long"
    SHORT = "short"
    NONE = "none"


@dataclass
class SMCConfig:
    """
    SMC Strategy Configuration.

    Attributes:
        session_start: Asian session start time (UTC)
        session_end: Asian session end time (UTC)
        atr_period: ATR calculation period
        atr_buffer_mult: ATR buffer for sweep detection
        ifvg_atr_mult: ATR multiplier for IFVG detection
        ifvg_proximity_mult: ATR multiplier for IFVG proximity
        risk_per_trade: Risk per trade as fraction of equity
        slippage_buffer: Slippage buffer as fraction of ATR
        target_1r: First target at 1R (breakeven)
        target_2r: Second target at 2R (scale out)
        target_final: Final target at 2.5R
        daily_loss_limit: Daily loss limit as fraction of equity
        max_trades_per_day: Maximum trades per day
        require_volume_confirmation: Require volume confirmation for sweep
        require_mss_confirmation: Require MSS for entry
    """

    session_start: str = "00:00"
    session_end: str = "08:00"
    atr_period: int = 14
    atr_buffer_mult: float = 0.5
    ifvg_atr_mult: float = 1.2
    ifvg_proximity_mult: float = 1.5
    risk_per_trade: float = 0.01
    slippage_buffer: float = 0.1
    target_1r: float = 1.0
    target_2r: float = 2.0
    target_final: float = 2.5
    daily_loss_limit: float = 0.03
    max_trades_per_day: int = 3
    require_volume_confirmation: bool = True
    require_mss_confirmation: bool = True


@dataclass
class TradeSignal:
    """
    SMC Trade Signal.

    Attributes:
        timestamp: Signal timestamp
        direction: Trade direction
        entry_price: Entry price
        stop_loss: Stop loss price
        target_1: First target (breakeven)
        target_2: Second target (scale out)
        target_final: Final target
        position_size: Position size in shares/contracts
        risk_amount: Risk amount in currency
        asia_high: Asian session high
        asia_low: Asian session low
        sweep_price: Sweep price
        ifvg_zone: IFVG zone (low, high)
        confidence: Signal confidence (0.0 to 1.0)
        metadata: Additional information
    """

    timestamp: pd.Timestamp
    direction: Literal["long", "short"]
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    target_final: float
    position_size: float
    risk_amount: float
    asia_high: float
    asia_low: float
    sweep_price: float
    ifvg_zone: Tuple[float, float]
    confidence: float = 0.5
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": str(self.timestamp),
            "direction": self.direction,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_1": self.target_1,
            "target_2": self.target_2,
            "target_final": self.target_final,
            "position_size": self.position_size,
            "risk_amount": self.risk_amount,
            "asia_high": self.asia_high,
            "asia_low": self.asia_low,
            "sweep_price": self.sweep_price,
            "ifvg_zone": self.ifvg_zone,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class DailyState:
    """
    Daily strategy state container.

    Attributes:
        date: Trading date
        state: Current strategy state
        asia_range: Asian session range data
        sweep_info: Sweep detection data
        mss_info: MSS confirmation data
        ifvgs: List of detected IFVGs
        active_signal: Current active signal (if any)
        daily_pnl: Daily profit/loss
        trades_today: Number of trades today
        htf_bias: Higher timeframe bias
    """

    date: pd.Timestamp
    state: StrategyState
    asia_range: Optional[AsianRange] = None
    sweep_info: Optional[SweepInfo] = None
    mss_info: Optional[MSSInfo] = None
    ifvgs: List[IFVG] = field(default_factory=list)
    active_signal: Optional[TradeSignal] = None
    daily_pnl: float = 0.0
    trades_today: int = 0
    htf_bias: Literal["bullish", "bearish", "neutral"] = "neutral"

    def reset(self, date: pd.Timestamp):
        """Reset state for a new trading day."""
        self.date = date
        self.state = StrategyState.TRACKING_ASIA
        self.asia_range = None
        self.sweep_info = None
        self.mss_info = None
        self.ifvgs = []
        self.active_signal = None
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.htf_bias = "neutral"


class SMCReversalStrategy:
    """
    SMC/ICT Reversal Strategy Implementation.

    A stateful strategy that tracks daily sessions, detects liquidity
    sweeps, confirms market structure shifts, and enters on IFVG zones.

    Example:
        >>> config = SMCConfig(risk_per_trade=0.01)
        >>> strategy = SMCReversalStrategy(config)
        >>> signals = strategy.run(df)
        >>> for signal in signals:
        ...     print(f"{signal.direction} @ {signal.entry_price}")
    """

    def __init__(self, config: Optional[SMCConfig] = None, account_equity: float = 100000.0):
        """
        Initialize SMC Strategy.

        Args:
            config: Strategy configuration (uses defaults if None)
            account_equity: Starting account equity
        """
        self.config = config or SMCConfig()
        self.account_equity = account_equity
        self.initial_equity = account_equity
        self.state: Optional[DailyState] = None
        self.signals: List[TradeSignal] = []
        self._atr_series: Optional[pd.Series] = None
        self._current_ifvgs: List[IFVG] = []

        logger.info(
            f"SMCReversalStrategy initialized with config: "
            f"session={self.config.session_start}-{self.config.session_end}, "
            f"risk={self.config.risk_per_trade * 100}%"
        )

    def run(self, df: pd.DataFrame) -> List[TradeSignal]:
        """
        Run strategy on historical data.

        Args:
            df: OHLCV DataFrame with UTC datetime index (5-minute bars)

        Returns:
            List of trade signals

        Raises:
            ValueError: If DataFrame lacks required columns or datetime index
        """
        # Validate DataFrame
        self._validate_dataframe(df)

        # Calculate ATR series
        self._atr_series = atr(df, period=self.config.atr_period)

        # Reset state
        self.signals = []
        self._current_ifvgs = []

        # Ensure UTC timezone
        if df.index.tz is None:
            df = df.tz_localize("UTC")
        elif str(df.index.tz) != "UTC":
            df = df.tz_convert("UTC")

        # Get unique dates
        dates = df.index.normalize().unique()

        logger.info(f"Running SMC strategy on {len(dates)} trading days")

        # Process each day
        for date in dates:
            try:
                self._process_day(df, pd.Timestamp(date))
            except Exception as e:
                logger.error(f"Error processing {date}: {e}")
                continue

        logger.info(f"Strategy complete: {len(self.signals)} signals generated")
        return self.signals

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate DataFrame has required columns and index."""
        if df is None or len(df) == 0:
            raise ValueError("DataFrame is empty or None")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")

        required_cols = ["Open", "High", "Low", "Close"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _process_day(self, df: pd.DataFrame, date: pd.Timestamp) -> None:
        """Process a single trading day."""
        # Initialize or reset daily state
        if self.state is None or self.state.date != date:
            self.state = DailyState(date=date, state=StrategyState.TRACKING_ASIA)

        # Get day's data
        day_start = date
        day_end = date + pd.Timedelta(days=1)
        day_df = df.loc[day_start:day_end]

        if day_df.empty:
            return

        # Process each bar
        for i, (idx, row) in enumerate(day_df.iterrows()):
            bar_time = idx.time()

            # Check for daily reset
            if self._is_session_start(bar_time):
                self._handle_session_start(idx)

            # State machine processing
            self._process_bar(df, idx, i)

    def _is_session_start(self, bar_time: time) -> bool:
        """Check if bar is at session start."""
        start_time = pd.to_datetime(self.config.session_start).time()
        return bar_time.hour == start_time.hour and bar_time.minute == start_time.minute

    def _is_session_end(self, bar_time: time) -> bool:
        """Check if bar is at session end."""
        end_time = pd.to_datetime(self.config.session_end).time()
        return bar_time.hour == end_time.hour and bar_time.minute == end_time.minute

    def _handle_session_start(self, timestamp: pd.Timestamp) -> None:
        """Handle session start - reset state."""
        if self.state is None:
            self.state = DailyState(date=timestamp.normalize(), state=StrategyState.TRACKING_ASIA)
        else:
            self.state.reset(timestamp.normalize())

        logger.debug(f"Session start at {timestamp}")

    def _process_bar(self, df: pd.DataFrame, idx: pd.Timestamp, global_idx: int) -> None:
        """Process a single bar through the state machine."""
        if self.state is None:
            return

        bar_time = idx.time()
        session_end_time = pd.to_datetime(self.config.session_end).time()

        # State machine
        if self.state.state == StrategyState.TRACKING_ASIA:
            # Track Asian session
            if self._is_session_end(bar_time):
                self._complete_asia_session(df, idx)

        elif self.state.state == StrategyState.ASIA_COMPLETE:
            # Look for sweep after Asia
            self._check_for_sweep(df, idx, global_idx)

        elif self.state.state == StrategyState.SWEEP_DETECTED:
            # Look for MSS confirmation
            self._check_for_mss(df, idx, global_idx)

        elif self.state.state == StrategyState.MSS_CONFIRMED:
            # Look for entry
            self._check_for_entry(df, idx, global_idx)

        elif self.state.state == StrategyState.IN_TRADE:
            # Manage active trade
            self._manage_active_trade(df, idx, global_idx)

        # Update IFVGs
        self._update_ifvgs(df, global_idx)

    def _complete_asia_session(self, df: pd.DataFrame, idx: pd.Timestamp) -> None:
        """Complete Asian session tracking."""
        # Get Asia session data
        session_start = pd.to_datetime(self.config.session_start).time()
        session_end = pd.to_datetime(self.config.session_end).time()

        day_start = idx.normalize()
        day_df = df.loc[day_start:idx]

        if day_df.empty:
            logger.warning(f"No data for Asia session at {idx}")
            return

        # Detect Asian range
        asia_range = detect_asian_range(
            day_df,
            session_start=self.config.session_start,
            session_end=self.config.session_end,
            atr_period=self.config.atr_period,
        )

        if asia_range is None:
            logger.warning(f"Could not detect Asian range at {idx}")
            return

        self.state.asia_range = asia_range
        self.state.state = StrategyState.ASIA_COMPLETE

        logger.debug(
            f"Asia session complete: high={asia_range.high:.4f}, "
            f"low={asia_range.low:.4f}, range={asia_range.range_size:.4f}"
        )

    def _check_for_sweep(self, df: pd.DataFrame, idx: pd.Timestamp, global_idx: int) -> None:
        """Check for liquidity sweep."""
        if self.state is None or self.state.asia_range is None:
            return

        # Check daily loss limit
        if self.state.daily_pnl < -self.config.daily_loss_limit * self.initial_equity:
            logger.debug("Daily loss limit reached, skipping sweep check")
            return

        # Check max trades
        if self.state.trades_today >= self.config.max_trades_per_day:
            return

        # Get current ATR
        current_atr = self._get_atr(global_idx)

        # Get post-Asia data
        session_end = pd.to_datetime(self.config.session_end).time()
        post_asia_start = idx.normalize() + pd.Timedelta(
            hours=session_end.hour, minutes=session_end.minute
        )
        post_asia_df = df.loc[post_asia_start:idx]

        if post_asia_df.empty:
            return

        # Detect sweep
        sweep = detect_liquidity_sweep(
            post_asia_df,
            asia_high=self.state.asia_range.high,
            asia_low=self.state.asia_range.low,
            atr=current_atr,
            buffer_mult=self.config.atr_buffer_mult,
        )

        if sweep.detected:
            # Check volume confirmation if required
            if self.config.require_volume_confirmation and not sweep.volume_confirmed:
                logger.debug(f"Sweep detected but volume not confirmed at {idx}")
                return

            self.state.sweep_info = sweep
            self.state.state = StrategyState.SWEEP_DETECTED

            logger.debug(
                f"Sweep detected at {idx}: direction={sweep.direction}, "
                f"price={sweep.sweep_price:.4f}"
            )

    def _check_for_mss(self, df: pd.DataFrame, idx: pd.Timestamp, global_idx: int) -> None:
        """Check for Market Structure Shift confirmation."""
        if self.state is None or self.state.sweep_info is None:
            return

        # Detect MSS
        mss = detect_mss(df, lookback=3, require_close=True, end_bar=global_idx)

        if mss.detected:
            # Validate MSS direction matches sweep
            if mss.direction != self.state.sweep_info.direction:
                logger.debug(
                    f"MSS direction mismatch: sweep={self.state.sweep_info.direction}, "
                    f"mss={mss.direction}"
                )
                return

            self.state.mss_info = mss
            self.state.state = StrategyState.MSS_CONFIRMED

            logger.debug(
                f"MSS confirmed at {idx}: direction={mss.direction}, "
                f"break_price={mss.break_price:.4f}"
            )

    def _check_for_entry(self, df: pd.DataFrame, idx: pd.Timestamp, global_idx: int) -> None:
        """Check for entry signal."""
        if self.state is None or self.state.sweep_info is None or self.state.mss_info is None:
            return

        # Get current ATR
        current_atr = self._get_atr(global_idx)

        # Find nearest IFVG
        ifvg_proximity = find_nearest_unfilled_ifvg(
            df,
            self._current_ifvgs,
            global_idx,
            current_atr,
            proximity_threshold=self.config.ifvg_proximity_mult,
            direction=self.state.sweep_info.direction,
        )

        if not ifvg_proximity.within_proximity or ifvg_proximity.nearest_ifvg is None:
            return

        # Generate signal
        signal = self._generate_signal(df, idx, global_idx, ifvg_proximity)

        if signal is not None:
            self.signals.append(signal)
            self.state.active_signal = signal
            self.state.trades_today += 1
            self.state.state = StrategyState.IN_TRADE

            logger.info(
                f"Signal generated at {idx}: {signal.direction} @ {signal.entry_price:.4f}, "
                f"stop={signal.stop_loss:.4f}"
            )

    def _update_ifvgs(self, df: pd.DataFrame, global_idx: int) -> None:
        """Update IFVG detection."""
        # Detect new IFVGs periodically
        if global_idx % 10 == 0:  # Every 10 bars
            current_atr = self._get_atr(global_idx)

            # Detect IFVGs from recent data
            lookback = min(100, global_idx)
            recent_df = df.iloc[global_idx - lookback : global_idx + 1]
            recent_atr = self._atr_series.iloc[global_idx - lookback : global_idx + 1]

            new_ifvgs = detect_ifvg(recent_df, recent_atr, atr_mult=self.config.ifvg_atr_mult)

            # Merge with existing IFVGs
            for ifvg in new_ifvgs:
                if not any(i.start_bar == ifvg.start_bar for i in self._current_ifvgs):
                    self._current_ifvgs.append(ifvg)

        # Update fill status
        self._current_ifvgs = update_ifvg_status(self._current_ifvgs, df, global_idx)

    def _manage_active_trade(self, df: pd.DataFrame, idx: pd.Timestamp, global_idx: int) -> None:
        """
        Manage active trade with breakeven, scaling, and target exits.

        Rules (from smc_strategy_implementation_plan.md):
        - Move stop to breakeven at 1R profit
        - Scale out 50% at 2R profit
        - Close remaining at 2.5R target
        """
        if self.state is None or self.state.active_signal is None:
            return

        signal = self.state.active_signal
        current_price = df.iloc[global_idx]["Close"]

        # Calculate current profit in R-multiples
        risk_distance = abs(signal.entry_price - signal.stop_loss)

        if signal.direction == "long":
            current_profit = current_price - signal.entry_price
        else:
            current_profit = signal.entry_price - current_price

        r_multiple = current_profit / risk_distance if risk_distance > 0 else 0

        # Breakeven at 1R
        if r_multiple >= self.config.target_1r and not signal.metadata.get("be_moved", False):
            signal.stop_loss = signal.entry_price
            signal.metadata["be_moved"] = True
            logger.info(f"Stop moved to breakeven at {idx}")

        # Scale out at 2R
        if r_multiple >= self.config.target_2r and not signal.metadata.get("scaled", False):
            signal.metadata["scaled"] = True
            signal.metadata["scale_price"] = current_price
            logger.info(f"Scaled out 50% at {idx}, price={current_price:.4f}")

        # Final target at 2.5R
        if r_multiple >= self.config.target_final:
            self.state.active_signal = None
            self.state.state = StrategyState.TRACKING_ASIA
            logger.info(f"Position closed at target {idx}, price={current_price:.4f}")
            return

        # Stop loss hit
        if signal.direction == "long" and current_price <= signal.stop_loss:
            self.state.active_signal = None
            self.state.state = StrategyState.TRACKING_ASIA
            logger.info(f"Stop loss hit at {idx}, price={current_price:.4f}")
        elif signal.direction == "short" and current_price >= signal.stop_loss:
            self.state.active_signal = None
            self.state.state = StrategyState.TRACKING_ASIA
            logger.info(f"Stop loss hit at {idx}, price={current_price:.4f}")

    def _generate_signal(
        self, df: pd.DataFrame, idx: pd.Timestamp, global_idx: int, ifvg_proximity: IFVGProximity
    ) -> Optional[TradeSignal]:
        """Generate trade signal."""
        if self.state is None or self.state.sweep_info is None:
            return None

        current_atr = self._get_atr(global_idx)
        current_price = df.iloc[global_idx]["Close"]
        ifvg = ifvg_proximity.nearest_ifvg

        # Determine entry and stop
        if self.state.sweep_info.direction == "bullish":
            direction = "long"
            entry_price = ifvg.low  # Enter at IFVG low edge
            stop_price = self.state.asia_range.low - (self.config.atr_buffer_mult * current_atr)
        else:
            direction = "short"
            entry_price = ifvg.high  # Enter at IFVG high edge
            stop_price = self.state.asia_range.high + (self.config.atr_buffer_mult * current_atr)

        # Calculate risk
        risk_distance = abs(entry_price - stop_price)
        risk_amount = self.account_equity * self.config.risk_per_trade

        # Calculate position size with slippage buffer
        slippage = current_atr * self.config.slippage_buffer
        effective_risk = risk_distance + slippage
        position_size = risk_amount / effective_risk if effective_risk > 0 else 0

        # Calculate targets
        risk_per_share = abs(entry_price - stop_price)
        if direction == "long":
            target_1 = entry_price + (risk_per_share * self.config.target_1r)
            target_2 = entry_price + (risk_per_share * self.config.target_2r)
            target_final = entry_price + (risk_per_share * self.config.target_final)
        else:
            target_1 = entry_price - (risk_per_share * self.config.target_1r)
            target_2 = entry_price - (risk_per_share * self.config.target_2r)
            target_final = entry_price - (risk_per_share * self.config.target_final)

        # Calculate confidence
        confidence = self._calculate_confidence(ifvg_proximity)

        return TradeSignal(
            timestamp=idx,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_price,
            target_1=target_1,
            target_2=target_2,
            target_final=target_final,
            position_size=position_size,
            risk_amount=risk_amount,
            asia_high=self.state.asia_range.high,
            asia_low=self.state.asia_range.low,
            sweep_price=self.state.sweep_info.sweep_price,
            ifvg_zone=(ifvg.low, ifvg.high),
            confidence=confidence,
            metadata={
                "atr": current_atr,
                "ifvg_distance": ifvg_proximity.distance,
                "ifvg_distance_atr_ratio": ifvg_proximity.distance_atr_ratio,
                "mss_direction": self.state.mss_info.direction if self.state.mss_info else None,
                "mss_break_price": self.state.mss_info.break_price if self.state.mss_info else None,
            },
        )

    def _calculate_confidence(self, ifvg_proximity: IFVGProximity) -> float:
        """Calculate signal confidence score."""
        confidence = 0.5  # Base confidence

        # IFVG proximity bonus
        if ifvg_proximity.distance_atr_ratio < 0.5:
            confidence += 0.15
        elif ifvg_proximity.distance_atr_ratio < 1.0:
            confidence += 0.10
        elif ifvg_proximity.distance_atr_ratio < 1.5:
            confidence += 0.05

        # Volume confirmation bonus
        if self.state and self.state.sweep_info and self.state.sweep_info.volume_confirmed:
            confidence += 0.10

        # MSS confirmation bonus
        if self.state and self.state.mss_info and self.state.mss_info.is_valid:
            confidence += 0.10

        # Low volatility Asia penalty
        if self.state and self.state.asia_range and self.state.asia_range.is_low_vol:
            confidence -= 0.10

        return min(1.0, max(0.0, confidence))

    def _get_atr(self, idx: int) -> float:
        """Get ATR value at given index."""
        if self._atr_series is None or idx >= len(self._atr_series):
            return 0.01

        atr_value = self._atr_series.iloc[idx]
        return atr_value if not pd.isna(atr_value) else 0.01

    def get_state_summary(self) -> Dict:
        """Get current strategy state summary for logging."""
        if self.state is None:
            return {"state": "not_initialized"}

        return {
            "state": self.state.state.value,
            "date": str(self.state.date),
            "asia_range": self.state.asia_range.to_dict() if self.state.asia_range else None,
            "sweep_detected": self.state.sweep_info.detected if self.state.sweep_info else False,
            "mss_detected": self.state.mss_info.detected if self.state.mss_info else False,
            "trades_today": self.state.trades_today,
            "daily_pnl": self.state.daily_pnl,
            "signals_generated": len(self.signals),
        }
