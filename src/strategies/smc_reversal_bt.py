"""
SMC Reversal Strategy for backtesting.py

Simplified Smart Money Concepts reversal strategy adapted for backtesting.py.
Implements a state machine for daily/weekly timeframe use.

Strategy Flow (simplified for daily data):
1. Track N-day range (default: previous 5 days' high/low)
2. Detect liquidity sweep: price breaks outside range by ATR buffer
3. Confirm Market Structure Shift (MSS): break of recent swing after sweep
4. Enter next bar with SL at sweep extreme +/- ATR buffer
5. Manage trade: breakeven at 1R, target at 2.5R

For intraday data (hourly):
1. Track Asian session (first N bars of each day)
2. Detect sweep after session closes
3. Confirm MSS and enter

Parameters:
    session_bars: Number of bars to track as "session" (8 for hourly, 5 for daily)
    atr_period: ATR calculation period (default 14)
    atr_buffer_mult: ATR multiplier for sweep buffer (default 0.5)
    mss_lookback: Lookback bars for MSS swing detection (default 5)
    target_final: R multiple for final target (default 2.5)
    breakeven_at_r: R multiple to move stop to breakeven (default 1.0)
"""

from __future__ import annotations

from enum import Enum, auto

import numpy as np
from backtesting import Strategy


class SMCState(Enum):
    """State machine states for the SMC Reversal strategy."""

    INIT = auto()
    SESSION_COMPLETE = auto()  # N-bar range established
    LOOKING_FOR_SWEEP = auto()
    SWEEP_DETECTED = auto()
    LOOKING_FOR_MSS = auto()
    MSS_CONFIRMED = auto()
    IN_TRADE = auto()


class SMCReversalBacktest(Strategy):
    """
    SMC Reversal strategy for backtesting.py

    Uses a state machine to detect:
    1. Session range (N-bar high/low)
    2. Liquidity sweep beyond session range
    3. Market Structure Shift confirmation
    4. Entry with risk-managed exit (BE at 1R, target at 2.5R)

    Parameters:
        session_bars: Bars to track as session (default 5 for daily)
        atr_period: ATR period (default 14)
        atr_buffer_mult: ATR buffer for sweep (default 0.5)
        mss_lookback: Lookback for MSS swing (default 5)
        target_final: Target R multiple (default 2.5)
        breakeven_at_r: BE trigger R multiple (default 1.0)
    """

    # --- Strategy parameters (overridable on Backtest() call) ---
    session_bars: int = 5
    atr_period: int = 14
    atr_buffer_mult: float = 0.5
    mss_lookback: int = 5
    target_final: float = 2.5
    breakeven_at_r: float = 1.0

    def init(self) -> None:
        """
        Initialize indicators using self.I() for pre-computation.
        Only ATR is needed; all other values are derived from OHLC.
        """

        def _atr(
            close: np.ndarray,
            high: np.ndarray,
            low: np.ndarray,
            period: int,
        ) -> np.ndarray:
            """Average True Range calculation."""
            tr = np.maximum(
                high - low,
                np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))),
            )
            tr[0] = np.nan  # first True Range is undefined
            # Use wilder smoothing (same as backtesting lib's built-in ATR)
            atr = np.empty_like(close)
            atr[:] = np.nan
            if len(close) >= period:
                atr[period - 1] = np.nanmean(tr[:period])
                for i in range(period, len(close)):
                    atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
            return atr

        self.atr_line = self.I(
            lambda: _atr(self.data.Close, self.data.High, self.data.Low, self.atr_period),
            name="ATR",
            color="orange",
        )

    # ── Instance attributes (reset per bar / state tracking) ──

    def _reset_state(self) -> None:
        """Reset tracking state for a new setup cycle."""
        self._state: SMCState = SMCState.INIT
        self._session_high: float = 0.0
        self._session_low: float = 0.0
        self._session_start_idx: int = 0
        self._sweep_high: float = 0.0
        self._sweep_low: float = 0.0
        self._sweep_idx: int = 0
        self._sweep_direction: str = ""  # "long" or "short"
        self._mss_break_level: float = 0.0
        self._entry_price: float = 0.0
        self._stop_loss: float = 0.0
        self._take_profit: float = 0.0
        self._risk_distance: float = 0.0
        self._breakeven_set: bool = False
        self._bars_in_session: int = 0

    def next(self) -> None:
        """Bar-by-bar state machine processing."""
        # Ensure state exists (first bar)
        if not hasattr(self, "_state"):
            self._reset_state()

        current_bar = len(self.data) - 1  # 0-indexed bar count

        # ── If currently in a trade, manage it ──
        if self._state == SMCState.IN_TRADE and self.position:
            self._manage_trade()
            return

        # If an order was open but position is closed (SL/TP hit), reset
        if self._state == SMCState.IN_TRADE and not self.position:
            self._reset_state()

        # ── State machine ──
        if self._state == SMCState.INIT:
            self._track_session()
        elif self._state == SMCState.SESSION_COMPLETE:
            self._check_sweep(current_bar)
        elif self._state == SMCState.SWEEP_DETECTED:
            self._check_mss(current_bar)
        elif self._state == SMCState.MSS_CONFIRMED:
            self._enter_trade()
        # SESSION_COMPLETE transitions to LOOKING_FOR_SWEEP internally

    # ── Phase 1: Track Session Range ──

    def _track_session(self) -> None:
        """
        Track session high/low for the first `session_bars` bars.

        For daily data: tracks the last N days' range.
        For intraday data: tracks the first N bars of each day.

        Once the range is established, move to SESSION_COMPLETE.
        """
        # Initialize on first bar of tracking
        if self._bars_in_session == 0:
            self._session_high = float(self.data.High[-1])
            self._session_low = float(self.data.Low[-1])
            self._session_start_idx = len(self.data) - 1
            self._bars_in_session = 1
            return

        self._bars_in_session += 1
        self._session_high = max(self._session_high, float(self.data.High[-1]))
        self._session_low = min(self._session_low, float(self.data.Low[-1]))

        # Session complete after N bars
        if self._bars_in_session >= self.session_bars:
            self._state = SMCState.SESSION_COMPLETE

    def _rebuild_session(self) -> None:
        """
        Rebuild the session range from the most recent `session_bars` bars.
        Used after a trade completes to establish a new trading range.
        """
        n = self.session_bars
        total = len(self.data)
        if total < n:
            return

        start = total - n
        highs = self.data.High[start:total]
        lows = self.data.Low[start:total]

        self._session_high = float(np.max(highs))
        self._session_low = float(np.min(lows))
        self._session_start_idx = start
        self._bars_in_session = n
        self._state = SMCState.SESSION_COMPLETE

    # ── Phase 2: Liquidity Sweep Detection ──

    def _check_sweep(self, current_bar: int) -> None:
        """
        Detect if price sweeps outside the session range.

        A sweep occurs when:
        - High breaks above session high + ATR buffer (bearish sweep)
        - Low breaks below session low - ATR buffer (bullish sweep)

        The sweep direction is counter-trend: price stops out traders
        on the wrong side before reversing.
        """
        atr = float(self.atr_line[-1])
        if np.isnan(atr) or atr <= 0:
            return

        buffer = self.atr_buffer_mult * atr

        current_high = float(self.data.High[-1])
        current_low = float(self.data.Low[-1])

        # Bearish sweep: price pushes above session high then reverses
        if current_high > self._session_high + buffer:
            self._sweep_direction = "short"
            self._sweep_high = current_high
            self._sweep_low = self._session_low
            self._sweep_idx = current_bar
            self._mss_break_level = self._session_low  # MSS = break session low
            self._state = SMCState.SWEEP_DETECTED
            return

        # Bullish sweep: price pushes below session low then reverses
        if current_low < self._session_low - buffer:
            self._sweep_direction = "long"
            self._sweep_high = self._session_high
            self._sweep_low = current_low
            self._sweep_idx = current_bar
            self._mss_break_level = self._session_high  # MSS = break session high
            self._state = SMCState.SWEEP_DETECTED
            return

    # ── Phase 3: Market Structure Shift (MSS) ──

    def _check_mss(self, current_bar: int) -> None:
        """
        Detect Market Structure Shift after a sweep.

        MSS confirms the reversal direction:
        - After bearish sweep: price breaks below recent swing low
        - After bullish sweep: price breaks above recent swing high

        Uses a simple lookback of `mss_lookback` bars.
        """
        n = self.mss_lookback
        total = len(self.data)
        if total < n + 1:
            return

        if self._sweep_direction == "long":
            # Bullish MSS: break above swing high in lookback window
            window_highs = self.data.High[-(n + 1) : -1]
            if len(window_highs) == 0:
                return
            swing_high = float(np.max(window_highs))
            current_close = float(self.data.Close[-1])

            if current_close > swing_high:
                self._mss_break_level = swing_high
                self._state = SMCState.MSS_CONFIRMED
                return

        elif self._sweep_direction == "short":
            # Bearish MSS: break below swing low in lookback window
            window_lows = self.data.Low[-(n + 1) : -1]
            if len(window_lows) == 0:
                return
            swing_low = float(np.min(window_lows))
            current_close = float(self.data.Close[-1])

            if current_close < swing_low:
                self._mss_break_level = swing_low
                self._state = SMCState.MSS_CONFIRMED
                return

        # Timeout: if too many bars pass without MSS, reset
        bars_since_sweep = current_bar - self._sweep_idx
        if bars_since_sweep > 20:
            self._reset_state()

    # ── Phase 4: Entry ──

    def _enter_trade(self) -> None:
        """
        Enter on the bar after MSS confirmation.

        Position sizing:
        - Long: buy at market
        - Short: sell at market (backtesting.py supports shorts with
          exclusive_orders=True by default, but we only go long for
          this simplified implementation)

        Stop loss:
        - Long: sweep_low - ATR buffer
        - Short: sweep_high + ATR buffer

        Target: 2.5R from entry
        """
        atr = float(self.atr_line[-1])
        if np.isnan(atr) or atr <= 0:
            return

        buffer = self.atr_buffer_mult * atr

        if self._sweep_direction == "long":
            # Long trade
            self._entry_price = float(self.data.Close[-1])
            self._stop_loss = self._sweep_low - buffer
            self._risk_distance = self._entry_price - self._stop_loss

            if self._risk_distance <= 0:
                self._reset_state()
                return

            self._take_profit = self._entry_price + self.target_final * self._risk_distance
            self._breakeven_set = False

            # Enter long
            size = self._calculate_size()
            self.buy(size=size)
            self._state = SMCState.IN_TRADE

        elif self._sweep_direction == "short":
            # Short trade
            self._entry_price = float(self.data.Close[-1])
            self._stop_loss = self._sweep_high + buffer
            self._risk_distance = self._stop_loss - self._entry_price

            if self._risk_distance <= 0:
                self._reset_state()
                return

            self._take_profit = self._entry_price - self.target_final * self._risk_distance
            self._breakeven_set = False

            # Enter short
            size = self._calculate_size()
            self.sell(size=size)
            self._state = SMCState.IN_TRADE

    def _calculate_size(self) -> int:
        """
        Calculate position size to risk ~1% of equity.
        Returns a whole number of shares (required by backtesting.py).
        """
        risk_pct = 0.01  # 1% risk per trade
        risk_capital = float(self.equity) * risk_pct
        risk_per_share = self._risk_distance if self._risk_distance > 0 else 1.0
        size = risk_capital / risk_per_share
        return max(int(round(size)), 1)  # minimum 1 share

    # ── Phase 5: Trade Management ──

    def _manage_trade(self) -> None:
        """
        Manage the active trade:
        - Move stop to breakeven at 1R profit
        - Take profit at 2.5R target
        - Stop loss at sweep extreme
        """
        if not self.position:
            self._reset_state()
            return

        entry = self._entry_price
        sl = self._stop_loss
        tp = self._take_profit
        risk = self._risk_distance

        if self._sweep_direction == "long":
            current = float(self.data.Close[-1])

            # Stop loss hit
            if current <= sl:
                self.position.close()
                self._reset_state()
                return

            # Take profit hit
            if current >= tp:
                self.position.close()
                self._reset_state()
                return

            # Move to breakeven at 1R
            if not self._breakeven_set:
                profit = current - entry
                r_multiple = profit / risk if risk > 0 else 0
                if r_multiple >= self.breakeven_at_r:
                    self._breakeven_set = True
                    self._stop_loss = entry  # move SL to breakeven

        elif self._sweep_direction == "short":
            current = float(self.data.Close[-1])

            # Stop loss hit
            if current >= sl:
                self.position.close()
                self._reset_state()
                return

            # Take profit hit
            if current <= tp:
                self.position.close()
                self._reset_state()
                return

            # Move to breakeven at 1R
            if not self._breakeven_set:
                profit = entry - current
                r_multiple = profit / risk if risk > 0 else 0
                if r_multiple >= self.breakeven_at_r:
                    self._breakeven_set = True
                    self._stop_loss = entry  # move SL to breakeven
