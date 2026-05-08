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

from typing import Optional, Tuple

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
        require_confirmation: bool = False,
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
        open_price = float(df["Open"].iloc[i])
        high = float(df["High"].iloc[i])
        low = float(df["Low"].iloc[i])
        close = float(df["Close"].iloc[i])
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
        if (
            upper_shadow < self.shadow_threshold * total_range
            and lower_shadow > self.dragonfly_threshold * total_range
        ):
            return True, "dragonfly"

        # Gravestone: Open=Close=Low, long upper shadow
        if (
            lower_shadow < self.shadow_threshold * total_range
            and upper_shadow > self.dragonfly_threshold * total_range
        ):
            return True, "gravestone"

        # Long-legged: Both shadows are significant
        if (
            upper_shadow > self.long_leg_threshold * total_range
            and lower_shadow > self.long_leg_threshold * total_range
        ):
            return True, "long_legged"

        # Standard Doji
        return True, "standard"

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
            return "sideways"

        start_idx = i - self.trend_lookback
        closes = np.asarray(df["Close"].iloc[start_idx:i].values, dtype=np.float64)

        if len(closes) < 2:
            return "sideways"

        # Calculate trend using linear regression slope
        x = np.arange(len(closes))
        slope = float(np.polyfit(x, closes, 1)[0])

        # Normalize slope by average price
        avg_price = float(np.mean(closes))
        normalized_slope = slope / avg_price if avg_price > 0 else 0

        # Threshold for trend determination (0.1% per bar)
        if normalized_slope > 0.001:
            return "up"
        elif normalized_slope < -0.001:
            return "down"
        else:
            return "sideways"

    def _check_confirmation(
        self,
        df: pd.DataFrame,
        i: int,
        doji_type: str,
        trend: str,
    ) -> bool:
        """
        Check if the Doji is confirmed using bar-i data only (no look-ahead).

        Confirms via volume spike and trend alignment.
        Does NOT read bar i+1 (eliminates look-ahead bias).

        Args:
            df: DataFrame with OHLC data
            i: Doji bar index
            doji_type: Type of Doji
            trend: Prior trend direction

        Returns:
            True if confirmed
        """
        if trend == "sideways":
            return False
        return self._check_volume_spike(df, i)

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
        if "Volume" not in df.columns or i < 20:
            return False

        avg_volume = df["Volume"].iloc[i - 20 : i].mean()
        if avg_volume == 0:
            return False

        current_volume = float(df["Volume"].iloc[i])
        return bool(current_volume > avg_volume * threshold)

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized detection of Doji signals across the entire DataFrame.

        Args:
            df: DataFrame with 'Open', 'High', 'Low', 'Close' columns

        Returns:
            np.ndarray of np.int8: 0=no signal, 1=LONG, -1=SHORT
        """
        n = len(df)
        result = np.zeros(n, dtype=np.int8)
        if n < self.trend_lookback + 3:
            return result

        open_a = df["Open"].to_numpy(dtype=np.float64)
        high_a = df["High"].to_numpy(dtype=np.float64)
        low_a = df["Low"].to_numpy(dtype=np.float64)
        close_a = df["Close"].to_numpy(dtype=np.float64)

        body = np.abs(close_a - open_a)
        total_range = high_a - low_a
        upper_shadow = high_a - np.maximum(open_a, close_a)
        lower_shadow = np.minimum(open_a, close_a) - low_a

        x_trend = np.arange(self.trend_lookback, dtype=np.float64)

        for i in range(self.trend_lookback + 1, n):
            rng = total_range[i]
            if rng == 0:
                continue

            body_ratio = body[i] / rng
            if body_ratio > self.body_threshold:
                continue

            up_s = upper_shadow[i]
            low_s = lower_shadow[i]

            if up_s < self.shadow_threshold * rng and low_s > self.dragonfly_threshold * rng:
                doji_type = "dragonfly"
            elif low_s < self.shadow_threshold * rng and up_s > self.dragonfly_threshold * rng:
                doji_type = "gravestone"
            elif up_s > self.long_leg_threshold * rng and low_s > self.long_leg_threshold * rng:
                doji_type = "long_legged"
            else:
                doji_type = "standard"

            close_slice = close_a[i - self.trend_lookback : i]
            slope = np.polyfit(x_trend, close_slice, 1)[0]
            avg_price = np.mean(close_slice)
            normalized_slope = slope / avg_price if avg_price > 0 else 0.0

            trend = "sideways"
            if normalized_slope > 0.001:
                trend = "up"
            elif normalized_slope < -0.001:
                trend = "down"

            if doji_type == "dragonfly" and trend == "down":
                result[i] = 1
            elif doji_type == "gravestone" and trend == "up":
                result[i] = -1
            elif doji_type in ("standard", "long_legged"):
                if trend == "down":
                    result[i] = 1
                elif trend == "up":
                    result[i] = -1

        return result

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
        if not self._validate_data(df, i, window_start):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

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
            if doji_type is None:
                return PatternResult(
                    detected=False,
                    pattern_name=self.name,
                    pattern_type=self.pattern_type,
                )
            confirmed = self._check_confirmation(df, i, doji_type, trend)

        # Determine potential signal direction based on Doji type and trend
        # Dragonfly in downtrend = bullish
        # Gravestone in uptrend = bearish
        # Standard Doji = depends on context

        signal_direction = None
        confidence = 0.30  # Low base confidence for Doji

        if doji_type == "dragonfly" and trend == "down":
            signal_direction = SignalDirection.LONG
            confidence = 0.35
        elif doji_type == "gravestone" and trend == "up":
            signal_direction = SignalDirection.SHORT
            confidence = 0.35
        elif doji_type in ["standard", "long_legged"]:
            # Direction from trend context (no look-ahead)
            if confirmed:
                if trend == "down":
                    signal_direction = SignalDirection.LONG
                elif trend == "up":
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
            )

        # Generate trade signal
        if signal_direction is None:
            return PatternResult(
                detected=True,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                signal=None,
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
                "doji_type": doji_type,
                "trend": trend,
                "confirmed": True,
                "volume_spike": self._check_volume_spike(df, i),
            },
        )

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            start_index=i - 1,
            end_index=i,
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
            high = df["High"].iloc[i - period : i + 1].values
            low = df["Low"].iloc[i - period : i + 1].values
            close = df["Close"].iloc[i - period - 1 : i].values

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
