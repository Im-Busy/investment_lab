"""
Dark Cloud Cover / Piercing Line Pattern

Two-candle reversal patterns where the second candle penetrates
the body of the first candle.

Types:
- Dark Cloud Cover (Bearish): 
  - First candle: Strong bullish candle
  - Second candle: Opens above first high, closes below first body midpoint
  - Signals potential bearish reversal after uptrend

- Piercing Line (Bullish):
  - First candle: Strong bearish candle
  - Second candle: Opens below first low, closes above first body midpoint
  - Signals potential bullish reversal after downtrend

Signal Generation:
- Medium reliability - confirmation recommended
- Entry on close of second candle
- Stop loss on opposite side of pattern

Confidence: 0.55 (medium reliability)
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class DarkCloudCover(BasePattern):
    """
    Dark Cloud Cover Pattern Detector

    Detects bearish reversal pattern where a bearish candle opens above
    the prior bullish candle's high and closes below its midpoint.
    """

    def __init__(
        self,
        body_threshold: float = 0.6,
        penetration_threshold: float = 0.5,
        require_confirmation: bool = True,
        trend_lookback: int = 10,
    ):
        """
        Initialize Dark Cloud Cover pattern detector.

        Args:
            body_threshold: Minimum body/range ratio for first candle (default 60%)
            penetration_threshold: How far into body (default 50% = midpoint)
            require_confirmation: Whether to require confirmation candle
            trend_lookback: Number of bars to analyze prior trend
        """
        super().__init__(
            name="Dark Cloud Cover",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=trend_lookback + 3,
        )
        self.body_threshold = body_threshold
        self.penetration_threshold = penetration_threshold
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
        body_mid = (body_high + body_low) / 2

        return {
            'body': body,
            'range': total_range,
            'is_bullish': is_bullish,
            'body_high': body_high,
            'body_low': body_low,
            'body_mid': body_mid,
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

    def _detect_dark_cloud(
        self,
        candle1: Dict[str, float],
        candle2: Dict[str, float],
    ) -> Tuple[bool, Optional[str], Optional[SignalDirection]]:
        """
        Detect Dark Cloud Cover pattern.

        Args:
            candle1: First candle (bullish)
            candle2: Second candle (bearish)

        Returns:
            Tuple of (is_pattern, pattern_type, direction)
        """
        # Candle 1 must have substantial body
        if candle1['range'] == 0 or candle1['body'] / candle1['range'] < self.body_threshold:
            return False, None, None

        # Candle 1 must be bullish
        if not candle1['is_bullish']:
            return False, None, None

        # Candle 2 must be bearish
        if candle2['is_bullish']:
            return False, None, None

        # Candle 2 must open above Candle 1 high
        if candle2['open'] <= candle1['high']:
            return False, None, None

        # Candle 2 must close below Candle 1 midpoint
        if candle2['close'] >= candle1['body_mid']:
            return False, None, None

        # Candle 2 should not close below Candle 1 low (that would be stronger)
        # But we still count it as Dark Cloud Cover

        return True, 'dark_cloud_cover', SignalDirection.SHORT

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

        if direction == SignalDirection.SHORT:
            return next_close < candle2['close']
        else:
            return next_close > candle2['close']

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
        Detect Dark Cloud Cover pattern at bar index i.

        Args:
            df: DataFrame with OHLC data
            i: Current bar index (second candle)
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

        # Detect Dark Cloud Cover
        is_pattern, pattern_type, direction = self._detect_dark_cloud(candle1, candle2)

        if not is_pattern:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
            )

        # Get prior trend
        trend = self._get_trend_direction(df, i)

        # Check confirmation
        confirmed = True
        if self.require_confirmation:
            confirmed = self._check_confirmation(df, i, direction, candle2)

        # Calculate confidence
        confidence = 0.55  # Base confidence

        # Increase confidence if in correct trend context (uptrend before bearish reversal)
        if trend == 'up':
            confidence = 0.60

        # Calculate penetration depth
        penetration = (candle1['body_high'] - candle2['close']) / candle1['body']
        if penetration > 0.6:  # Deeper penetration = stronger signal
            confidence = min(confidence + 0.05, 0.70)

        # Increase confidence with volume spike
        if self._check_volume_spike(df, i):
            confidence = min(confidence + 0.05, 0.70)

        if not confirmed:
            return PatternResult(
                detected=True,
                pattern_name=pattern_type,
                pattern_type=self.pattern_type,
                signal=None,
                metadata={
                    'pattern_type': pattern_type,
                    'trend': trend,
                    'confirmed': False,
                    'requires_confirmation': self.require_confirmation,
                    'penetration': penetration,
                },
            )

        # Calculate entry, stop, and target
        atr = self._calculate_atr(df, i)
        if atr is None or atr <= 0:
            atr = candle2['range'] * 0.5

        # SHORT signal
        entry_price = candle2['close']
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
            pattern_name=pattern_type,
            metadata={
                'pattern_type': pattern_type,
                'trend': trend,
                'confirmed': True,
                'volume_spike': self._check_volume_spike(df, i),
                'penetration': penetration,
            },
        )

        return PatternResult(
            detected=True,
            pattern_name=pattern_type,
            pattern_type=self.pattern_type,
            signal=signal,
            start_index=i - 1,
            end_index=i,
            metadata={
                'pattern_type': pattern_type,
                'trend': trend,
                'confirmed': True,
            },
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """
        Generate trade signal for Dark Cloud Cover pattern.
        
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


class PiercingLine(BasePattern):
    """
    Piercing Line Pattern Detector

    Detects bullish reversal pattern where a bullish candle opens below
    the prior bearish candle's low and closes above its midpoint.
    """

    def __init__(
        self,
        body_threshold: float = 0.6,
        penetration_threshold: float = 0.5,
        require_confirmation: bool = True,
        trend_lookback: int = 10,
    ):
        """
        Initialize Piercing Line pattern detector.

        Args:
            body_threshold: Minimum body/range ratio for first candle (default 60%)
            penetration_threshold: How far into body (default 50% = midpoint)
            require_confirmation: Whether to require confirmation candle
            trend_lookback: Number of bars to analyze prior trend
        """
        super().__init__(
            name="Piercing Line",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=trend_lookback + 3,
        )
        self.body_threshold = body_threshold
        self.penetration_threshold = penetration_threshold
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
        body_mid = (body_high + body_low) / 2

        return {
            'body': body,
            'range': total_range,
            'is_bullish': is_bullish,
            'body_high': body_high,
            'body_low': body_low,
            'body_mid': body_mid,
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

    def _detect_piercing(
        self,
        candle1: Dict[str, float],
        candle2: Dict[str, float],
    ) -> Tuple[bool, Optional[str], Optional[SignalDirection]]:
        """
        Detect Piercing Line pattern.

        Args:
            candle1: First candle (bearish)
            candle2: Second candle (bullish)

        Returns:
            Tuple of (is_pattern, pattern_type, direction)
        """
        # Candle 1 must have substantial body
        if candle1['range'] == 0 or candle1['body'] / candle1['range'] < self.body_threshold:
            return False, None, None

        # Candle 1 must be bearish
        if candle1['is_bullish']:
            return False, None, None

        # Candle 2 must be bullish
        if not candle2['is_bullish']:
            return False, None, None

        # Candle 2 must open below Candle 1 low
        if candle2['open'] >= candle1['low']:
            return False, None, None

        # Candle 2 must close above Candle 1 midpoint
        if candle2['close'] <= candle1['body_mid']:
            return False, None, None

        # Candle 2 should not close above Candle 1 high (that would be stronger)
        # But we still count it as Piercing Line

        return True, 'piercing_line', SignalDirection.LONG

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
        return current_volume > avg_volume * threshold

    def detect(
        self,
        df: pd.DataFrame,
        i: int,
        window_start: Optional[int] = None,
    ) -> PatternResult:
        """
        Detect Piercing Line pattern at bar index i.

        Args:
            df: DataFrame with OHLC data
            i: Current bar index (second candle)
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

        # Detect Piercing Line
        is_pattern, pattern_type, direction = self._detect_piercing(candle1, candle2)

        if not is_pattern:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
            )

        # Get prior trend
        trend = self._get_trend_direction(df, i)

        # Check confirmation
        confirmed = True
        if self.require_confirmation:
            confirmed = self._check_confirmation(df, i, direction, candle2)

        # Calculate confidence
        confidence = 0.55  # Base confidence

        # Increase confidence if in correct trend context (downtrend before bullish reversal)
        if trend == 'down':
            confidence = 0.60

        # Calculate penetration depth
        penetration = (candle2['close'] - candle1['body_low']) / candle1['body']
        if penetration > 0.6:  # Deeper penetration = stronger signal
            confidence = min(confidence + 0.05, 0.70)

        # Increase confidence with volume spike
        if self._check_volume_spike(df, i):
            confidence = min(confidence + 0.05, 0.70)

        if not confirmed:
            return PatternResult(
                detected=True,
                pattern_name=pattern_type,
                pattern_type=self.pattern_type,
                signal=None,
                metadata={
                    'pattern_type': pattern_type,
                    'trend': trend,
                    'confirmed': False,
                    'requires_confirmation': self.require_confirmation,
                    'penetration': penetration,
                },
            )

        # Calculate entry, stop, and target
        atr = self._calculate_atr(df, i)
        if atr is None or atr <= 0:
            atr = candle2['range'] * 0.5

        # LONG signal
        entry_price = candle2['close']
        stop_loss = min(candle1['low'], candle2['low']) - atr * 0.25
        take_profit_1 = entry_price + atr * 1.5
        take_profit_2 = entry_price + atr * 2.5
        take_profit_3 = entry_price + atr * 4.0

        signal = TradeSignal(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            confidence=confidence,
            pattern_name=pattern_type,
            metadata={
                'pattern_type': pattern_type,
                'trend': trend,
                'confirmed': True,
                'volume_spike': self._check_volume_spike(df, i),
                'penetration': penetration,
            },
        )

        return PatternResult(
            detected=True,
            pattern_name=pattern_type,
            pattern_type=self.pattern_type,
            signal=signal,
            start_index=i - 1,
            end_index=i,
            metadata={
                'pattern_type': pattern_type,
                'trend': trend,
                'confirmed': True,
            },
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """
        Generate trade signal for Piercing Line pattern.
        
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
