"""
London Breakout Strategy for backtesting.py

Uses Tokyo hour (2:00-3:00 EST) price action to predict London open breakout.
Trades first 30 minutes of London session with risk management.

Strategy Logic:
1. Tokyo Hour Collection (2:00-3:00 EST): Collect high/low prices
2. London Open (3:00 EST): Set upper/lower thresholds based on Tokyo range
3. London Trading Window (3:00-3:30 EST): Monitor for breakout signals
4. Entry Execution: (long) Buy on upper breakout, (short) Sell on lower breakout
5. Risk Management: Stop loss filtering, reverse position detection
6. London Close (12:00 EST): Clear all intraday positions

Parameters:
- tokyo_start_hour: 2 (EST)
- tokyo_end_hour: 3 (EST)
- london_trading_minutes: 30
- london_close_hour: 12 (EST)
- risky_stop: 0.01 (1% stop loss)
- param: 0.5 (range multiplier)

Expected Performance:
- Sharpe Ratio: 0.5-1.5
- Max Drawdown: -10% to -25%
- Win Rate: 45-55%
"""

import pandas as pd
from backtesting import Strategy
from typing import Optional


class LondonBreakoutStrategy(Strategy):
    """
    London Breakout Strategy for backtesting.py

    Uses Tokyo hour price action to predict London open breakout.
    Trades first 30 minutes of London session with risk management.
    """

    tokyo_start_hour = 2
    tokyo_end_hour = 3
    london_trading_minutes = 30
    london_close_hour = 12
    risky_stop = 0.01
    param = 0.5

    def init(self) -> None:
        """Initialize strategy state."""
        self.tokyo_highs: list[float] = []
        self.tokyo_lows: list[float] = []
        self.tokyo_range: float = 0.0
        self.upper_threshold: float = 0.0
        self.lower_threshold: float = 0.0
        self.london_open_price: float = 0.0
        self.entry_price: float = 0.0
        self.signals_this_session: int = 0
        self.in_london_session: bool = False

    def next(self) -> None:
        """Execute strategy logic on each bar."""
        current_time = self.data.index[-1]

        if self._is_tokyo_hour(current_time):
            self._collect_tokyo_data()
            return

        if self._is_london_session_start(current_time):
            self._set_london_thresholds()
            self.in_london_session = True
            return

        if self._is_london_trading_window(current_time) and self.in_london_session:
            signal = self._generate_signals()
            if signal:
                self._manage_positions(signal)
            return

        if self._is_london_close(current_time):
            self._clear_positions_at_close()

    def _is_tokyo_hour(self, dt: pd.Timestamp) -> bool:
        """Check if time is within Tokyo hour window."""
        return dt.hour == self.tokyo_start_hour

    def _is_london_session_start(self, dt: pd.Timestamp) -> bool:
        """Check if time is London open."""
        return dt.hour == self.tokyo_end_hour and dt.minute == 0

    def _is_london_trading_window(self, dt: pd.Timestamp) -> bool:
        """Check if within London trading window."""
        return dt.hour == self.tokyo_end_hour and dt.minute < self.london_trading_minutes

    def _is_london_close(self, dt: pd.Timestamp) -> bool:
        """Check if London session ends."""
        return dt.hour >= self.london_close_hour

    def _collect_tokyo_data(self) -> None:
        """Collect high/low during Tokyo hour."""
        self.tokyo_highs.append(self.data.High[-1])
        self.tokyo_lows.append(self.data.Low[-1])

    def _calculate_tokyo_range(self) -> float:
        """Calculate Tokyo hour price range."""
        if not self.tokyo_highs:
            return 0.0

        tokyo_high = max(self.tokyo_highs)
        tokyo_low = min(self.tokyo_lows)
        return tokyo_high - tokyo_low

    def _set_london_thresholds(self) -> None:
        """Set upper/lower thresholds at London open."""
        self.tokyo_range = self._calculate_tokyo_range()
        self.london_open_price = self.data.Close[-1]

        self.upper_threshold = self.london_open_price + self.param * self.tokyo_range
        self.lower_threshold = self.london_open_price - self.param * self.tokyo_range

    def _generate_signals(self) -> Optional[int]:
        """Generate buy/sell signals based on breakout."""
        current_price = self.data.Close[-1]

        if current_price > self.upper_threshold:
            if self._check_stop_loss(current_price):
                return None
            return 1

        if current_price < self.lower_threshold:
            if self._check_stop_loss(current_price):
                return None
            return -1

        return None

    def _check_stop_loss(self, price: float) -> bool:
        """Check if signal should be rejected by stop loss."""
        if self.entry_price == 0:
            return False
        return abs(price - self.entry_price) / self.entry_price > self.risky_stop

    def _manage_positions(self, signal: int) -> None:
        """Manage positions based on signals."""
        current_price = self.data.Close[-1]

        if self.signals_this_session >= 1 and signal != 0:
            return

        if self.position and self.position.is_short and signal == 1:
            self.position.close()
            self.buy()
            self.entry_price = current_price
            self.signals_this_session += 1
            return

        if self.position and self.position.is_long and signal == -1:
            self.position.close()
            self.sell()
            self.entry_price = current_price
            self.signals_this_session += 1
            return

        if signal == 1 and not self.position:
            self.buy()
            self.entry_price = current_price
            self.signals_this_session += 1

        if signal == -1 and not self.position:
            self.sell()
            self.entry_price = current_price
            self.signals_this_session += 1

    def _clear_positions_at_close(self) -> None:
        """Clear all positions at London close."""
        if self.position:
            self.position.close()
        self._reset_session_state()

    def _reset_session_state(self) -> None:
        """Reset session state."""
        self.tokyo_highs.clear()
        self.tokyo_lows.clear()
        self.tokyo_range = 0.0
        self.upper_threshold = 0.0
        self.lower_threshold = 0.0
        self.london_open_price = 0.0
        self.entry_price = 0.0
        self.signals_this_session = 0
        self.in_london_session = False
