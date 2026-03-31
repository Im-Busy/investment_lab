"""
Doji Pattern

A Doji is a candlestick pattern where the open and close prices are nearly equal,
indicating market indecision. It's a reversal warning signal that requires confirmation.

Types:
- Standard Doji: Open equals Close
- Long-legged Doji: Long upper and lower shadows
- Dragonfly Doji: Open=Close=High, long lower shadow (bullish)
- Gravestone Doji: Open=Close=Low, long upper shadow (bearish)

Signal Generation:
- Do NOT generate signal on Doji alone (low reliability)
- Flag for watchlist and require confirmation:
  - Next candle breaks Doji high/low with volume
  - At key support/resistance level
  - Confluence with other patterns

Confidence: 0.30 (low - requires confirmation)
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class Doji(BasePattern):
    """
    Doji Pattern Detector

    Detects various types of Doji candles which signal market indecision.
    Doji patterns have very low standalone reliability and should only be
    used as warning signals or in confluence with other patterns.
    """

    def __init__(
        self,
        body_threshold: float = 0.1,
        shadow_threshold: float = 0.1,
        long_leg_threshold: float = 0.3,
        dragonfly_threshold: float = 0.6,
        require_confirmation: bool = True,
        trend_lookback: int = 10,
    ):
        """
        Initialize Doji pattern detector.

        Args:
            body_threshold: Maximum body/range ratio for Doji (default 10%)
            shadow_threshold: Maximum shadow size for body position (default 10%)
            long_leg_threshold: Minimum shadow ratio for long-legged Doji (default 30%)
            dragonfly_threshold: Minimum lower shadow ratio for dragonfly (default 60%)
            require_confirmation: Whether to require confirmation candle
            trend_lookback: Number of bars to analyze prior trend
        """
        super().__init__(
            name="Doji",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=trend_lookback + 2,
        )
        self.body_threshold = body_threshold
        self.shadow_threshold = shadow_threshold
        self.long_leg_threshold = long_leg_threshold
        self.dragonfly_threshold = dragonfly_threshold
        self.require_confirmation = require_confirmation
        self.trend_lookback = trend_lookback

    def _get_candle_data(self, df: pd.DataFrame, i: int) -> Tuple[float, float, float, float]:
        """Get OHLC values for a candle."""
        open_price = float(df['Open'].iloc[i])
        high = float(df['High'].iloc[i])
        low = float(df['Low'].iloc[i])
        close = float(df['Close'].iloc[i])
        return open_price, high, low, close

    def _classify_doji(
        self,
        open_price: float,
        high: float,
        low: float,
        close: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Classify the type of Doji.

        Args:
            open_price: Opening price
            high: High price
            low: Low price
            close: Closing price

        Returns:
            Tuple of (is_doji, doji_type)
        """
        body = abs(close - open_price)
        total_range = high - low

        if total_range == 0:
            # No range - not a valid candle
            return False, None

        body_ratio = body / total_range

        # Check if body is small enough to be Doji
        if body_ratio > self.body_threshold:
            return False, None

        # Calculate shadows
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low

        # Classify Doji type
        # Dragonfly: Open=Close=High, long lower shadow
        if upper_shadow < self.shadow_threshold * total_range and lower_shadow > self.dragonfly_threshold * total_range:
            return True, 'dragonfly'

        # Gravestone: Open=Close=Low, long upper shadow
        if lower_shadow < self.shadow_threshold * total_range and upper_shadow > self.dragonfly_threshold * total_range:
            return True, 'gravestone'

        # Long-legged: Both shadows are significant
        if upper_shadow > self.long_leg_threshold * total_range and lower_shadow > self.long_leg_threshold * total_range:
            return True, 'long_legged'

        # Standard Doji
        return True, 'standard'

    def _get_trend_direction(self, df: pd.DataFrame, i: int) -> str:
        """
        Determine the prior trend direction.

        Args:
            df: DataFrame with OHLC data
            i: Current bar index

        Returns:
            'up', 'down', or 'sideways'
        """
        if i < self.trend_lookback:
            return 'sideways'

        start_idx = i - self.trend_lookback
        closes = df['Close'].iloc[start_idx:i].values

        if len(closes) < 2:
            return 'sideways'

        # Calculate trend using linear regression slope
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]

        # Normalize slope by average price
        avg_price = np.mean(closes)
        normalized_slope = slope / avg_price if avg_price > 0 else 0

        # Threshold for trend determination (0.1% per bar)
        if normalized_slope > 0.001:
            return 'up'
        elif normalized_slope < -0.001:
            return 'down'
        else:
            return 'sideways'

    def _check_confirmation(
        self,
        df: pd.DataFrame,
        i: int,
        doji_type: str,
        trend: str,
    ) -> bool:
        """
        Check if the Doji is confirmed by the next candle.

        Args:
            df: DataFrame with OHLC data
            i: Doji bar index
            doji_type: Type of Doji
            trend: Prior trend direction

        Returns:
            True if confirmed
        """
        if i >= len(df) - 1:
            return False

        # Get Doji candle data
        _, doji_high, doji_low, _ = self._get_candle_data(df, i)

        # Get next candle data
        next_open, next_high, next_low, next_close = self._get_candle_data(df, i + 1)

        # Confirmation depends on Doji type and trend
        if doji_type == 'dragonfly':
            # Bullish signal - confirm with higher close
            return next_close > doji_high

        elif doji_type == 'gravestone':
            # Bearish signal - confirm with lower close
            return next_close < doji_low

        else:
            # Standard/long-legged - confirm with breakout in either direction
            return next_close > doji_high or next_close < doji_low

    def _check_volume_spike(
        self,
        df: pd.DataFrame,
        i: int,
        threshold: float = 1.5,
    ) -> bool:
        """
        Check if volume is above average.

        Args:
            df: DataFrame with OHLCV data
            i: Current bar index
            threshold: Volume multiplier threshold

        Returns:
            True if volume spike detected
        """
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
        Detect Doji pattern at bar index i.

        Args:
            df: DataFrame with OHLCV data
            i: Current bar index
            window_start: Optional window start for bounds checking

        Returns:
            PatternResult with detection status and any signal
        """
        # Validate index
        if i < 1 or i >= len(df):
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
            )

        # Get candle data
        open_price, high, low, close = self._get_candle_data(df, i)

        # Check if it's a Doji
        is_doji, doji_type = self._classify_doji(open_price, high, low, close)

        if not is_doji:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
            )

        # Get prior trend
        trend = self._get_trend_direction(df, i)

        # Check for confirmation if required
        confirmed = False
        if self.require_confirmation:
            confirmed = self._check_confirmation(df, i, doji_type, trend)

        # Determine potential signal direction based on Doji type and trend
        # Dragonfly in downtrend = bullish
        # Gravestone in uptrend = bearish
        # Standard Doji = depends on context

        signal_direction = None
        confidence = 0.30  # Low base confidence for Doji

        if doji_type == 'dragonfly' and trend == 'down':
            signal_direction = SignalDirection.LONG
            confidence = 0.35
        elif doji_type == 'gravestone' and trend == 'up':
            signal_direction = SignalDirection.SHORT
            confidence = 0.35
        elif doji_type in ['standard', 'long_legged']:
            # Indecision - direction depends on confirmation
            if confirmed:
                _, doji_high, doji_low, _ = self._get_candle_data(df, i)
                next_close = float(df['Close'].iloc[i + 1])
                if next_close > doji_high:
                    signal_direction = SignalDirection.LONG
                elif next_close < doji_low:
                    signal_direction = SignalDirection.SHORT

        # Increase confidence if volume spike
        if self._check_volume_spike(df, i):
            confidence = min(confidence + 0.1, 0.5)

        # Only generate signal if confirmed (Doji alone is not tradable)
        if not confirmed:
            return PatternResult(
                detected=True,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                signal=None,
                metadata={
                    'doji_type': doji_type,
                    'trend': trend,
                    'confirmed': False,
                    'requires_confirmation': True,
                },
            )

        # Generate trade signal
        if signal_direction is None:
            return PatternResult(
                detected=True,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                signal=None,
                metadata={
                    'doji_type': doji_type,
                    'trend': trend,
                    'confirmed': True,
                },
            )

        # Calculate entry, stop, and target
        atr = self._calculate_atr(df, i)
        if atr is None or atr <= 0:
            atr = (high - low) * 0.5  # Fallback

        if signal_direction == SignalDirection.LONG:
            entry_price = close
            stop_loss = low - atr * 0.5
            take_profit_1 = entry_price + atr * 1.5
            take_profit_2 = entry_price + atr * 2.5
            take_profit_3 = entry_price + atr * 4.0
        else:  # SHORT
            entry_price = close
            stop_loss = high + atr * 0.5
            take_profit_1 = entry_price - atr * 1.5
            take_profit_2 = entry_price - atr * 2.5
            take_profit_3 = entry_price - atr * 4.0

        signal = TradeSignal(
            direction=signal_direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            confidence=confidence,
            pattern_name=self.name,
            metadata={
                'doji_type': doji_type,
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
                'doji_type': doji_type,
                'trend': trend,
                'confirmed': True,
            },
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """
        Generate trade signal for Doji pattern.
        
        This method is called by detect() when a pattern is found.
        Doji requires confirmation, so this returns the signal from detect().
        
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
