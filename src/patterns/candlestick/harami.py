"""
Harami Pattern

A Harami is a two-candle reversal pattern where a small body candle
is completely contained within the body of the preceding larger candle.

Types:
- Bullish Harami: Large bearish candle followed by small bullish candle
  (Signals potential bullish reversal after downtrend)
- Bearish Harami: Large bullish candle followed by small bearish candle
  (Signals potential bearish reversal after uptrend)

Signal Generation:
- Requires confirmation for higher reliability
- Entry on break of first candle's high/low
- Stop loss on opposite side of pattern

Confidence: 0.50 (medium - requires confirmation)
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class Harami(BasePattern):
    """
    Harami Pattern Detector

    Detects bullish and bearish Harami patterns which signal
    potential reversals when they appear after trends.
    """

    def __init__(
        self,
        body_threshold: float = 0.6,
        small_body_ratio: float = 0.7,
        require_confirmation: bool = True,
        trend_lookback: int = 10,
        confirmation_filter: float = 0.005,
    ):
        """
        Initialize Harami pattern detector.

        Args:
            body_threshold: Minimum body/range ratio for first candle (default 60%)
            small_body_ratio: Maximum ratio of second body to first body (default 70%)
            require_confirmation: Whether to require confirmation candle
            trend_lookback: Number of bars to analyze prior trend
            confirmation_filter: Minimum breakout percentage for confirmation
        """
        super().__init__(
            name="Harami",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=trend_lookback + 3,
        )
        self.body_threshold = body_threshold
        self.small_body_ratio = small_body_ratio
        self.require_confirmation = require_confirmation
        self.trend_lookback = trend_lookback
        self.confirmation_filter = confirmation_filter

    def _get_candle_data(self, df: pd.DataFrame, i: int) -> Tuple[float, float, float, float]:
        """Get OHLC values for a candle."""
        open_price = float(df['Open'].iloc[i])
        high = float(df['High'].iloc[i])
        low = float(df['Low'].iloc[i])
        close = float(df['Close'].iloc[i])
        return open_price, high, low, close

    def _analyze_candle(
        self,
        open_price: float,
        high: float,
        low: float,
        close: float,
    ) -> Dict[str, float]:
        """
        Analyze a single candle.

        Returns:
            Dict with body, range, is_bullish, body_high, body_low
        """
        body = abs(close - open_price)
        total_range = high - low
        is_bullish = close > open_price
        body_high = max(open_price, close)
        body_low = min(open_price, close)

        return {
            'body': body,
            'range': total_range,
            'is_bullish': is_bullish,
            'body_high': body_high,
            'body_low': body_low,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
        }

    def _get_trend_direction(self, df: pd.DataFrame, i: int) -> str:
        """Determine the prior trend direction."""
        if i < self.trend_lookback + 1:
            return 'sideways'

        start_idx = i - self.trend_lookback - 1
        closes = df['Close'].iloc[start_idx:i].values

        if len(closes) < 2:
            return 'sideways'

        # Calculate trend using linear regression slope
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]

        avg_price = np.mean(closes)
        normalized_slope = slope / avg_price if avg_price > 0 else 0

        if normalized_slope > 0.001:
            return 'up'
        elif normalized_slope < -0.001:
            return 'down'
        else:
            return 'sideways'

    def _detect_harami(
        self,
        candle1: Dict[str, float],
        candle2: Dict[str, float],
    ) -> Tuple[bool, Optional[str], Optional[SignalDirection]]:
        """
        Detect Harami pattern from two candles.

        Args:
            candle1: First candle (larger body)
            candle2: Second candle (smaller body)

        Returns:
            Tuple of (is_harami, harami_type, direction)
        """
        # Candle 1 must have substantial body
        if candle1['range'] == 0 or candle1['body'] / candle1['range'] < self.body_threshold:
            return False, None, None

        # Candle 2 body must be smaller than Candle 1 body
        if candle2['body'] >= candle1['body'] * self.small_body_ratio:
            return False, None, None

        # Bodies must be opposite colors
        if candle1['is_bullish'] == candle2['is_bullish']:
            return False, None, None

        # Candle 2 body must be completely inside Candle 1 body
        if candle2['body_high'] > candle1['body_high']:
            return False, None, None
        if candle2['body_low'] < candle1['body_low']:
            return False, None, None

        # Determine Harami type and direction
        if candle1['is_bullish']:
            # First candle bullish, second bearish = Bearish Harami
            return True, 'bearish_harami', SignalDirection.SHORT
        else:
            # First candle bearish, second bullish = Bullish Harami
            return True, 'bullish_harami', SignalDirection.LONG

    def _check_confirmation(
        self,
        df: pd.DataFrame,
        i: int,
        direction: SignalDirection,
        candle1: Dict[str, float],
    ) -> bool:
        """
        Check if pattern is confirmed by next candle.

        Args:
            df: DataFrame with OHLC data
            i: Pattern end bar index (candle 2)
            direction: Signal direction
            candle1: First candle data

        Returns:
            True if confirmed
        """
        if i >= len(df) - 1:
            return False

        next_close = float(df['Close'].iloc[i + 1])

        if direction == SignalDirection.LONG:
            # Bullish Harami - confirm with close above candle 1 high
            return next_close > candle1['body_high'] * (1 + self.confirmation_filter)
        else:
            # Bearish Harami - confirm with close below candle 1 low
            return next_close < candle1['body_low'] * (1 - self.confirmation_filter)

    def _check_volume_spike(self, df: pd.DataFrame, i: int, threshold: float = 1.5) -> bool:
        """Check if volume is above average."""
        if 'Volume' not in df.columns or i < 20:
            return False

        avg_volume = df['Volume'].iloc[i - 20:i].mean()
        if avg_volume == 0:
            return False

        current_volume = float(df['Volume'].iloc[i])
        return current_volume > avg_volume * threshold

    def detect(
        self,
        df: pd.DataFrame,
        i: int,
        window_start: Optional[int] = None,
    ) -> PatternResult:
        """
        Detect Harami pattern at bar index i.

        Args:
            df: DataFrame with OHLC data
            i: Current bar index (second candle of Harami)
            window_start: Optional window start for bounds checking

        Returns:
            PatternResult with detection status and any signal
        """
        # Need at least 2 candles (candle1 at i-1, candle2 at i)
        if i < 2 or i >= len(df):
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
            )

        # Get candle data
        open1, high1, low1, close1 = self._get_candle_data(df, i - 1)
        open2, high2, low2, close2 = self._get_candle_data(df, i)

        candle1 = self._analyze_candle(open1, high1, low1, close1)
        candle2 = self._analyze_candle(open2, high2, low2, close2)

        # Detect Harami
        is_harami, harami_type, direction = self._detect_harami(candle1, candle2)

        if not is_harami:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
            )

        # Get prior trend
        trend = self._get_trend_direction(df, i)

        # Check confirmation
        confirmed = False
        if self.require_confirmation:
            confirmed = self._check_confirmation(df, i, direction, candle1)

        # Calculate confidence
        confidence = 0.50  # Base confidence for Harami

        # Increase confidence if in correct trend context
        if harami_type == 'bullish_harami' and trend == 'down':
            confidence = 0.55
        elif harami_type == 'bearish_harami' and trend == 'up':
            confidence = 0.55

        # Increase confidence with volume spike
        if self._check_volume_spike(df, i):
            confidence = min(confidence + 0.05, 0.65)

        # Increase confidence with confirmation
        if confirmed:
            confidence = min(confidence + 0.10, 0.70)

        # Only generate signal if confirmed
        if not confirmed:
            return PatternResult(
                detected=True,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                signal=None,
                metadata={
                    'harami_type': harami_type,
                    'trend': trend,
                    'confirmed': False,
                    'requires_confirmation': True,
                    'candle1_body': candle1['body'],
                    'candle2_body': candle2['body'],
                },
            )

        # Calculate entry, stop, and target
        atr = self._calculate_atr(df, i)
        if atr is None or atr <= 0:
            atr = (candle1['high'] - candle1['low']) * 0.5

        if direction == SignalDirection.LONG:
            entry_price = candle1['body_high']  # Entry on breakout above candle 1
            stop_loss = min(candle1['low'], candle2['low']) - atr * 0.25
            take_profit_1 = entry_price + atr * 1.5
            take_profit_2 = entry_price + atr * 2.5
            take_profit_3 = entry_price + atr * 4.0
        else:  # SHORT
            entry_price = candle1['body_low']  # Entry on breakout below candle 1
            stop_loss = max(candle1['high'], candle2['high']) + atr * 0.25
            take_profit_1 = entry_price - atr * 1.5
            take_profit_2 = entry_price - atr * 2.5
            take_profit_3 = entry_price - atr * 4.0

        signal = TradeSignal(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            confidence=confidence,
            pattern_name=self.name,
            metadata={
                'harami_type': harami_type,
                'trend': trend,
                'confirmed': True,
                'volume_spike': self._check_volume_spike(df, i),
            },
        )

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            start_index=i - 1,
            end_index=i,
            metadata={
                'harami_type': harami_type,
                'trend': trend,
                'confirmed': True,
            },
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """
        Generate trade signal for Harami pattern.
        
        Args:
            df: DataFrame with OHLC data
            i: Current bar index
            
        Returns:
            TradeSignal if confirmed pattern, None otherwise
        """
        result = self.detect(df, i)
        return result.signal if result.detected else None

    def _calculate_atr(self, df: pd.DataFrame, i: int, period: int = 14) -> Optional[float]:
        """Calculate ATR for position sizing."""
        if i < period + 1:
            return None

        try:
            high = df['High'].iloc[i - period:i + 1].values
            low = df['Low'].iloc[i - period:i + 1].values
            close = df['Close'].iloc[i - period - 1:i].values

            tr_list = []
            for j in range(len(high)):
                if j == 0:
                    tr = high[j] - low[j]
                else:
                    tr = max(
                        high[j] - low[j],
                        abs(high[j] - close[j - 1]),
                        abs(low[j] - close[j - 1]),
                    )
                tr_list.append(tr)

            return np.mean(tr_list)
        except Exception:
            return None
