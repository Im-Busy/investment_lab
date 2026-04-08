"""
Engulfing Pattern

A two-candle reversal pattern where the second candle's body completely
engulfs the first candle's body. This is one of the more reliable
candlestick patterns.

Types:
- Bullish Engulfing: First candle bearish, second candle bullish
  (Signals potential bullish reversal after downtrend)
- Bearish Engulfing: First candle bullish, second candle bearish
  (Signals potential bearish reversal after uptrend)

Signal Generation:
- Higher reliability - can trade without confirmation
- Entry on close of engulfing candle
- Stop loss on opposite side of pattern

Confidence: 0.65 (higher reliability)
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class Engulfing(BasePattern):
    """
    Engulfing Pattern Detector

    Detects bullish and bearish Engulfing patterns which signal
    potential reversals. Higher reliability pattern that can be
    traded without confirmation.
    """

    def __init__(
        self,
        body_threshold: float = 0.3,
        min_engulf_ratio: float = 1.0,
        require_confirmation: bool = False,  # Higher reliability - confirmation optional
        trend_lookback: int = 10,
    ):
        """
        Initialize Engulfing pattern detector.

        Args:
            body_threshold: Minimum body/range ratio for second candle (default 30%)
            min_engulf_ratio: Minimum ratio of second body to first body (default 1.0)
            require_confirmation: Whether to require confirmation candle (default False)
            trend_lookback: Number of bars to analyze prior trend
        """
        super().__init__(
            name="Engulfing",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=trend_lookback + 3,
        )
        self.body_threshold = body_threshold
        self.min_engulf_ratio = min_engulf_ratio
        self.require_confirmation = require_confirmation
        self.trend_lookback = trend_lookback

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
        """Analyze a single candle."""
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
        closes = np.asarray(df['Close'].iloc[start_idx:i].values, dtype=np.float64)

        if len(closes) < 2:
            return 'sideways'

        x = np.arange(len(closes))
        slope = float(np.polyfit(x, closes, 1)[0])

        avg_price = float(np.mean(closes))
        normalized_slope = slope / avg_price if avg_price > 0 else 0

        if normalized_slope > 0.001:
            return 'up'
        elif normalized_slope < -0.001:
            return 'down'
        else:
            return 'sideways'

    def _detect_engulfing(
        self,
        candle1: Dict[str, float],
        candle2: Dict[str, float],
    ) -> Tuple[bool, Optional[str], Optional[SignalDirection]]:
        """
        Detect Engulfing pattern from two candles.

        Args:
            candle1: First candle (smaller, engulfed)
            candle2: Second candle (larger, engulfing)

        Returns:
            Tuple of (is_engulfing, engulfing_type, direction)
        """
        # Candle 2 must have substantial body
        if candle2['range'] == 0 or candle2['body'] / candle2['range'] < self.body_threshold:
            return False, None, None

        # Candle 2 body must be larger than Candle 1 body
        if candle1['body'] == 0:
            # First candle is a doji - still valid engulfing
            pass
        elif candle2['body'] < candle1['body'] * self.min_engulf_ratio:
            return False, None, None

        # Candles must be opposite colors
        if candle1['is_bullish'] == candle2['is_bullish']:
            return False, None, None

        # Candle 2 must engulf Candle 1 body
        # Engulfing means: candle2 body completely contains candle1 body
        if candle2['body_high'] <= candle1['body_high']:
            return False, None, None
        if candle2['body_low'] >= candle1['body_low']:
            return False, None, None

        # Determine type
        if candle2['is_bullish']:
            # Bearish candle engulfed by bullish candle = Bullish Engulfing
            return True, 'bullish_engulfing', SignalDirection.LONG
        else:
            # Bullish candle engulfed by bearish candle = Bearish Engulfing
            return True, 'bearish_engulfing', SignalDirection.SHORT

    def _check_confirmation(
        self,
        df: pd.DataFrame,
        i: int,
        direction: SignalDirection,
        candle2: Dict[str, float],
    ) -> bool:
        """Check if pattern is confirmed by next candle."""
        if i >= len(df) - 1:
            return False

        next_close = float(df['Close'].iloc[i + 1])

        if direction == SignalDirection.LONG:
            return next_close > candle2['close']
        else:
            return next_close < candle2['close']

    def _check_volume_spike(self, df: pd.DataFrame, i: int, threshold: float = 1.5) -> bool:
        """Check if volume is above average."""
        if 'Volume' not in df.columns or i < 20:
            return False

        avg_volume = df['Volume'].iloc[i - 20:i].mean()
        if avg_volume == 0:
            return False

        current_volume = float(df['Volume'].iloc[i])
        return bool(current_volume > avg_volume * threshold)

    def detect(
        self,
        df: pd.DataFrame,
        i: int,
        window_start: Optional[int] = None,
    ) -> PatternResult:
        """
        Detect Engulfing pattern at bar index i.

        Args:
            df: DataFrame with OHLC data
            i: Current bar index (second candle of Engulfing)
            window_start: Optional window start for bounds checking

        Returns:
            PatternResult with detection status and any signal
        """
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

        # Detect Engulfing
        is_engulfing, engulfing_type, direction = self._detect_engulfing(candle1, candle2)

        if not is_engulfing:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
            )

        # Get prior trend
        trend = self._get_trend_direction(df, i)

        # Check confirmation if required
        confirmed = True
        if self.require_confirmation:
            if direction is None:
                return PatternResult(
                    detected=False,
                    pattern_name=self.name,
                    pattern_type=self.pattern_type,
                )
            confirmed = self._check_confirmation(df, i, direction, candle2)

        # Calculate confidence
        confidence = 0.65  # Higher base confidence for Engulfing

        # Increase confidence if in correct trend context
        if engulfing_type == 'bullish_engulfing' and trend == 'down':
            confidence = 0.70
        elif engulfing_type == 'bearish_engulfing' and trend == 'up':
            confidence = 0.70

        # Increase confidence with larger engulfing body
        if candle1['body'] > 0:
            engulf_ratio = candle2['body'] / candle1['body']
            if engulf_ratio > 2.0:
                confidence = min(confidence + 0.05, 0.80)

        # Increase confidence with volume spike
        if self._check_volume_spike(df, i):
            confidence = min(confidence + 0.05, 0.80)

        # Decrease confidence if not confirmed when required
        if not confirmed:
            confidence = 0.50

        # Only generate signal if confirmed
        if not confirmed:
            return PatternResult(
                detected=True,
                pattern_name=engulfing_type or self.name,
                pattern_type=self.pattern_type,
                signal=None,
            )

        # Calculate entry, stop, and target
        atr = self._calculate_atr(df, i)
        if atr is None or atr <= 0:
            atr = candle2['range'] * 0.5

        if direction == SignalDirection.LONG:
            entry_price = candle2['close']  # Entry on close of engulfing candle
            stop_loss = min(candle1['low'], candle2['low']) - atr * 0.25
            take_profit_1 = entry_price + atr * 1.5
            take_profit_2 = entry_price + atr * 2.5
            take_profit_3 = entry_price + atr * 4.0
        else:  # SHORT
            entry_price = candle2['close']  # Entry on close of engulfing candle
            stop_loss = max(candle1['high'], candle2['high']) + atr * 0.25
            take_profit_1 = entry_price - atr * 1.5
            take_profit_2 = entry_price - atr * 2.5
            take_profit_3 = entry_price - atr * 4.0

        assert direction is not None, "direction must be set"
        signal = TradeSignal(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            confidence=confidence,
            pattern_name=engulfing_type or self.name,
        )

        return PatternResult(
            detected=True,
            pattern_name=engulfing_type or self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            start_index=i - 1,
            end_index=i,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """
        Generate trade signal for Engulfing pattern.
        
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

            return float(np.mean(tr_list))
        except Exception:
            return None
