"""
Hammer / Hanging Man Pattern

A single-candle reversal pattern with a small body at the top of the range
and a long lower shadow (at least 2x the body length).

Types:
- Hammer: Appears after downtrend, signals bullish reversal
  (Small body at top, long lower shadow, little/no upper shadow)
- Hanging Man: Appears after uptrend, signals bearish reversal
  (Same structure as Hammer but in different context)

Signal Generation:
- Requires confirmation for higher reliability
- Hammer: Confirm with close above hammer high
- Hanging Man: Confirm with close below hanging man low
- Volume spike increases reliability

Confidence: 0.50 (medium - requires confirmation)
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class Hammer(BasePattern):
    """
    Hammer / Hanging Man Pattern Detector

    Detects hammer (bullish reversal after downtrend) and hanging man
    (bearish reversal after uptrend) patterns.
    """

    def __init__(
        self,
        shadow_ratio: float = 2.0,
        body_threshold: float = 0.35,
        upper_shadow_threshold: float = 0.1,
        require_confirmation: bool = False,
        trend_lookback: int = 10,
        confirmation_filter: float = 0.005,
    ):
        """
        Initialize Hammer pattern detector.

        Args:
            shadow_ratio: Minimum lower shadow / body ratio (default 2.0)
            body_threshold: Maximum body/range ratio (default 35%)
            upper_shadow_threshold: Maximum upper shadow/range ratio (default 10%)
            require_confirmation: Whether to require confirmation candle
            trend_lookback: Number of bars to analyze prior trend
            confirmation_filter: Minimum breakout percentage for confirmation
        """
        super().__init__(
            name="Hammer",
            pattern_type=PatternType.REVERSAL,
            min_bars_required=trend_lookback + 2,
        )
        self.shadow_ratio = shadow_ratio
        self.body_threshold = body_threshold
        self.upper_shadow_threshold = upper_shadow_threshold
        self.require_confirmation = require_confirmation
        self.trend_lookback = trend_lookback
        self.confirmation_filter = confirmation_filter

    def _get_candle_data(self, df: pd.DataFrame, i: int) -> Tuple[float, float, float, float]:
        """Get OHLC values for a candle."""
        open_price = float(df["Open"].iloc[i])
        high = float(df["High"].iloc[i])
        low = float(df["Low"].iloc[i])
        close = float(df["Close"].iloc[i])
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
            Dict with body, range, shadows, and body position
        """
        body = abs(close - open_price)
        total_range = high - low
        is_bullish = close > open_price
        body_high = max(open_price, close)
        body_low = min(open_price, close)

        upper_shadow = high - body_high
        lower_shadow = body_low - low

        return {
            "body": body,
            "range": total_range,
            "is_bullish": is_bullish,
            "body_high": body_high,
            "body_low": body_low,
            "upper_shadow": upper_shadow,
            "lower_shadow": lower_shadow,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
        }

    def _get_trend_direction(self, df: pd.DataFrame, i: int) -> str:
        """Determine the prior trend direction."""
        if i < self.trend_lookback:
            return "sideways"

        start_idx = i - self.trend_lookback
        closes = np.asarray(df["Close"].iloc[start_idx:i].values, dtype=np.float64)

        if len(closes) < 2:
            return "sideways"

        # Calculate trend using linear regression slope
        x = np.arange(len(closes))
        slope = float(np.polyfit(x, closes, 1)[0])

        avg_price = float(np.mean(closes))
        normalized_slope = slope / avg_price if avg_price > 0 else 0

        if normalized_slope > 0.001:
            return "up"
        elif normalized_slope < -0.001:
            return "down"
        else:
            return "sideways"

    def _detect_hammer(
        self,
        candle: Dict[str, float],
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect if candle is a Hammer/Hanging Man.

        Args:
            candle: Candle analysis dict

        Returns:
            Tuple of (is_hammer, pattern_type)
        """
        # Check for valid range
        if candle["range"] == 0:
            return False, None

        # Body must be small relative to range
        body_ratio = candle["body"] / candle["range"]
        if body_ratio > self.body_threshold:
            return False, None

        # Lower shadow must be at least shadow_ratio times body
        if candle["body"] == 0:
            # If body is zero (doji-like), check lower shadow directly
            if candle["lower_shadow"] < candle["range"] * 0.5:
                return False, None
        elif candle["lower_shadow"] < self.shadow_ratio * candle["body"]:
            return False, None

        # Upper shadow should be small
        if candle["upper_shadow"] > self.upper_shadow_threshold * candle["range"]:
            return False, None

        # It's a hammer/hanging man
        return True, "hammer_structure"

    def _check_confirmation(
        self,
        df: pd.DataFrame,
        i: int,
        trend: str,
        candle: Dict[str, float],
    ) -> Tuple[bool, Optional[SignalDirection]]:
        """
        Check if pattern is confirmed using bar-i data only (no look-ahead).

        Confirms via volume spike and trend alignment.
        Does NOT read bar i+1 (eliminates look-ahead bias).

        Args:
            df: DataFrame with OHLC data
            i: Pattern bar index
            trend: Prior trend direction
            candle: Pattern candle data

        Returns:
            Tuple of (confirmed, direction)
        """
        if trend == "sideways":
            return False, None
        if not self._check_volume_spike(df, i):
            return False, None

        if trend == "down":
            return True, SignalDirection.LONG
        elif trend == "up":
            return True, SignalDirection.SHORT
        return False, None

    def _check_volume_spike(self, df: pd.DataFrame, i: int, threshold: float = 1.5) -> bool:
        """Check if volume is above average."""
        if "Volume" not in df.columns or i < 20:
            return False

        avg_volume = df["Volume"].iloc[i - 20 : i].mean()
        if avg_volume == 0:
            return False

        current_volume = float(df["Volume"].iloc[i])
        return bool(current_volume > avg_volume * threshold)

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized detection of Hammer/Hanging Man signals across the entire DataFrame.

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
        is_bullish = close_a > open_a
        body_high = np.maximum(open_a, close_a)
        body_low = np.minimum(open_a, close_a)
        upper_shadow = high_a - body_high
        lower_shadow = body_low - low_a

        x_trend = np.arange(self.trend_lookback, dtype=np.float64)

        for i in range(self.trend_lookback + 1, n):
            rng = total_range[i]
            if rng == 0:
                continue

            bd = body[i]
            body_ratio = bd / rng
            if body_ratio > self.body_threshold:
                continue

            if bd == 0:
                if lower_shadow[i] < rng * 0.5:
                    continue
            elif lower_shadow[i] < self.shadow_ratio * bd:
                continue

            if upper_shadow[i] > self.upper_shadow_threshold * rng:
                continue

            close_slice = close_a[i - self.trend_lookback : i]
            slope = np.polyfit(x_trend, close_slice, 1)[0]
            avg_price = np.mean(close_slice)
            normalized_slope = slope / avg_price if avg_price > 0 else 0.0

            trend = "sideways"
            if normalized_slope > 0.001:
                trend = "up"
            elif normalized_slope < -0.001:
                trend = "down"

            # Use trend context for direction (no look-ahead)
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
        Detect Hammer/Hanging Man pattern at bar index i.

        Args:
            df: DataFrame with OHLC data
            i: Current bar index
            window_start: Optional window start for bounds checking

        Returns:
            PatternResult with detection status and any signal
        """
        if not self._validate_data(df, i, window_start):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        if i < 1 or i >= len(df):
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
            )

        # Get candle data
        open_price, high, low, close = self._get_candle_data(df, i)
        candle = self._analyze_candle(open_price, high, low, close)

        # Check if it's a hammer/hanging man structure
        is_hammer, pattern_type = self._detect_hammer(candle)

        if not is_hammer:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
            )

        # Get prior trend
        trend = self._get_trend_direction(df, i)

        # Determine pattern name based on trend context
        if trend == "down":
            pattern_name = "Hammer"  # Bullish reversal after downtrend
        elif trend == "up":
            pattern_name = "Hanging Man"  # Bearish reversal after uptrend
        else:
            pattern_name = "Hammer"  # Default

        # Check confirmation
        confirmed = False
        direction = None
        if self.require_confirmation:
            confirmed, direction = self._check_confirmation(df, i, trend, candle)
        else:
            # Without confirmation, infer direction from trend
            if trend == "down":
                direction = SignalDirection.LONG
                confirmed = True
            elif trend == "up":
                direction = SignalDirection.SHORT
                confirmed = True

        # Calculate confidence
        confidence = 0.50  # Base confidence

        # Increase confidence with correct trend context
        if trend in ["up", "down"]:
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
                pattern_name=pattern_name or self.name,
                pattern_type=self.pattern_type,
                signal=None,
            )

        if direction is None:
            return PatternResult(
                detected=True,
                pattern_name=pattern_name or self.name,
                pattern_type=self.pattern_type,
                signal=None,
            )

        # Calculate entry, stop, and target
        atr = self._calculate_atr(df, i)
        if atr is None or atr <= 0:
            atr = candle["range"] * 0.5

        if direction == SignalDirection.LONG:
            entry_price = candle["high"]  # Entry on breakout above hammer high
            stop_loss = candle["low"] - atr * 0.25
            take_profit_1 = entry_price + atr * 1.5
            take_profit_2 = entry_price + atr * 2.5
            take_profit_3 = entry_price + atr * 4.0
        else:  # SHORT
            entry_price = candle["low"]  # Entry on breakdown below hanging man low
            stop_loss = candle["high"] + atr * 0.25
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
            pattern_name=pattern_name,
            metadata={
                "pattern_type": pattern_type,
                "trend": trend,
                "confirmed": True,
                "volume_spike": self._check_volume_spike(df, i),
                "lower_shadow_body_ratio": candle["lower_shadow"] / candle["body"]
                if candle["body"] > 0
                else 0,
            },
        )

        return PatternResult(
            detected=True,
            pattern_name=pattern_name or self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            start_index=i,
            end_index=i,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """
        Generate trade signal for Hammer/Hanging Man pattern.

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
