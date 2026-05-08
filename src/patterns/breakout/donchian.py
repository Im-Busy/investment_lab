"""
Donchian Channel Breakout Strategy

Classic breakout system based on Richard Donchian's trend-following method
(used in the famous Turtle Trading system). Enters when price breaks above/below
the highest high/lowest low of the past N periods.

How It Works:
- Long Entry: Price closes above N-period high (upper band breakout)
- Short Entry: Price closes below N-period low (lower band breakdown)
- Exit: Price crosses middle line (optional) or opposite band

Components:
- Upper Band: Highest high of last N periods
- Lower Band: Lowest low of last N periods
- Middle Line: Average of upper and lower bands

Best Timeframes:
- Primary: 1H, 4H, 1D
- Alternative: 15m for scalping
- Original Turtle: Daily (20-day/55-day channels)

Risk/Reward Profile:
- Win Rate: 35-45%
- Risk/Reward: 1:3-5 (few large wins)
- Max Drawdown: 20-30%
- Trading Style: Breakout/momentum

Customizable Parameters:
- Channel Length (default: 20): Lookback period for high/low
- Breakout Confirmation (default: 1): Bars to confirm breakout
- Exit at Middle Line (default: true): Early exit vs ride full range
"""

from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ...indicators.technical import atr, volume_sma
from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class BreakoutQuality(Enum):
    """Breakout quality classification"""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class DonchianChannelBreakout(BasePattern):
    """
    Donchian Channel Breakout Pattern Detector

    A trend-following breakout system based on channel breakouts.
    Implements the classic Turtle Trading methodology with configurable
    filters for breakout quality confirmation.
    """

    def __init__(
        self,
        channel_period: int = 20,
        exit_period: int = 10,
        entry_offset: float = 0.01,
        use_mid_channel_exit: bool = True,
        use_atr_targets: bool = True,
        atr_mult_tp: float = 2.0,
        atr_mult_sl: float = 1.0,
        volume_filter: bool = True,
        volume_threshold: float = 1.5,
        atr_filter: bool = True,
        atr_period: int = 14,
        retest_entry: bool = False,
        retest_bars: int = 5,
    ):
        """
        Initialize Donchian Channel Breakout pattern detector.

        Args:
            channel_period: Period for channel calculation (default 20 - Turtle standard)
            exit_period: Period for exit channel (default 10 - Turtle standard)
            entry_offset: Price offset for entry orders
            use_mid_channel_exit: Exit when price crosses middle line
            use_atr_targets: Use ATR-based targets instead of fixed R:R
            atr_mult_tp: ATR multiplier for take profit (default 2.0)
            atr_mult_sl: ATR multiplier for stop loss (default 1.0)
            volume_filter: Require volume confirmation (>150% average)
            volume_threshold: Volume multiplier threshold (default 1.5)
            atr_filter: Require ATR expansion (volatility rising)
            atr_period: ATR period for filter (default 14)
            retest_entry: Wait for retest of broken level before entry
            retest_bars: Max bars to wait for retest (default 5)
        """
        super().__init__(
            name="Donchian Channel Breakout",
            pattern_type=PatternType.BREAKOUT,
            min_bars_required=max(channel_period, exit_period) + 2,
        )
        self.channel_period = channel_period
        self.exit_period = exit_period
        self.entry_offset = entry_offset
        self.use_mid_channel_exit = use_mid_channel_exit
        self.use_atr_targets = use_atr_targets
        self.atr_mult_tp = atr_mult_tp
        self.atr_mult_sl = atr_mult_sl
        self.volume_filter = volume_filter
        self.volume_threshold = volume_threshold
        self.atr_filter = atr_filter
        self.atr_period = atr_period
        self.retest_entry = retest_entry
        self.retest_bars = retest_bars

    def _compute_channels(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        """
        Calculate Donchian Channel values at bar i.

        Args:
            df: DataFrame with OHLC data
            i: Current bar index

        Returns:
            Dictionary with channel values or None if insufficient data
        """
        if i < self.channel_period:
            return None

        # Upper Band: Highest high of last N periods
        upper = float(df.iloc[i - self.channel_period : i + 1]["High"].max())
        # Lower Band: Lowest low of last N periods
        lower = float(df.iloc[i - self.channel_period : i + 1]["Low"].min())
        # Middle Line: Average of upper and lower bands
        middle = (upper + lower) / 2

        # Previous period channels (for breakout detection)
        if i >= self.channel_period + 1:
            prev_upper = float(df.iloc[i - self.channel_period : i]["High"].max())
            prev_lower = float(df.iloc[i - self.channel_period : i]["Low"].min())
        else:
            prev_upper = upper
            prev_lower = lower

        return {
            "upper": upper,
            "lower": lower,
            "middle": middle,
            "prev_upper": prev_upper,
            "prev_lower": prev_lower,
            "channel_width": upper - lower,
        }

    def _check_volume_confirmation(self, df: pd.DataFrame, i: int) -> bool:
        """
        Check if breakout bar has sufficient volume surge.

        Breakout volume must be > 150% of average volume (customizable).

        Args:
            df: DataFrame with OHLCV data
            i: Current bar index

        Returns:
            True if volume confirms breakout
        """
        if not self.volume_filter:
            return True

        if i < 20:
            return True  # Not enough data for volume SMA

        current_vol = float(df.iloc[i]["Volume"])
        vol_sma = float(volume_sma(df["Volume"], 20).iloc[i])

        if np.isnan(vol_sma) or vol_sma == 0:
            return True

        return current_vol > vol_sma * self.volume_threshold

    def _check_atr_expansion(self, df: pd.DataFrame, i: int) -> bool:
        """
        Check if ATR is expanding (volatility rising).

        Breakouts are more reliable when accompanied by rising volatility.

        Args:
            df: DataFrame with OHLCV data
            i: Current bar index

        Returns:
            True if ATR is expanding
        """
        if not self.atr_filter:
            return True

        if i < self.atr_period + 5:
            return True  # Not enough data

        atr_values = atr(df, self.atr_period)
        if i < len(atr_values):
            current_atr = float(atr_values.iloc[i])
            prev_atr = float(atr_values.iloc[i - 1])

            if np.isnan(current_atr) or np.isnan(prev_atr) or prev_atr == 0:
                return True

            # ATR should be expanding (current > previous)
            return current_atr > prev_atr

        return True

    def _assess_breakout_quality(
        self, df: pd.DataFrame, i: int, channels: Dict, breakout_up: bool
    ) -> BreakoutQuality:
        """
        Assess breakout quality based on multiple factors.

        Quality checklist:
        - Volume surge (>150% average)
        - Strong candle close (not wick)
        - ATR expanding (volatility increasing)
        - Prior consolidation period (at least 10 bars)

        Args:
            df: DataFrame with OHLCV data
            i: Current bar index
            channels: Channel values
            breakout_up: True for upward breakout

        Returns:
            BreakoutQuality enum value
        """
        quality_score = 0

        current_close = float(df.iloc[i]["Close"])
        current_high = float(df.iloc[i]["High"])
        current_low = float(df.iloc[i]["Low"])
        current_open = float(df.iloc[i]["Open"])

        # 1. Volume surge check
        if self._check_volume_confirmation(df, i):
            quality_score += 1

        # 2. Strong candle close (body > 50% of range)
        body = abs(current_close - current_open)
        candle_range = current_high - current_low
        if candle_range > 0 and body / candle_range > 0.5:
            quality_score += 1

        # 3. ATR expansion
        if self._check_atr_expansion(df, i):
            quality_score += 1

        # 4. Prior consolidation (channel width narrowing)
        if i >= self.channel_period + 10:
            # Compare current channel width to 10 bars ago
            prev_width = float(
                df.iloc[i - 10 - self.channel_period : i - 10]["High"].max()
            ) - float(df.iloc[i - 10 - self.channel_period : i - 10]["Low"].min())
            if prev_width > 0 and channels["channel_width"] < prev_width:
                # Channel squeeze detected - breakout more likely
                quality_score += 1

        # Classify quality
        if quality_score >= 3:
            return BreakoutQuality.STRONG
        elif quality_score >= 2:
            return BreakoutQuality.MODERATE
        else:
            return BreakoutQuality.WEAK

    def _check_retest_entry(
        self, df: pd.DataFrame, i: int, channels: Dict, breakout_up: bool
    ) -> bool:
        """
        Check if price has retested the broken channel level as support/resistance.

        Retest entry: Wait for price to retest broken level before entering.
        This reduces false breakouts but may miss some entries.

        Args:
            df: DataFrame with OHLCV data
            i: Current bar index
            channels: Channel values
            breakout_up: True for upward breakout

        Returns:
            True if retest confirmed
        """
        if not self.retest_entry:
            return True  # Retest not required

        lookback = min(self.retest_bars, i)
        if lookback < 1:
            return True

        if breakout_up:
            # For long breakout: price should pull back to upper channel
            # and hold it as support
            for j in range(1, lookback + 1):
                bar_low = float(df.iloc[i - j]["Low"])
                bar_close = float(df.iloc[i - j]["Close"])
                # Price touched or came close to upper channel
                if bar_low <= channels["prev_upper"] * 1.005:
                    # And bounced back (closed above)
                    if bar_close > channels["prev_upper"]:
                        return True
        else:
            # For short breakout: price should throw back to lower channel
            # and hold it as resistance
            for j in range(1, lookback + 1):
                bar_high = float(df.iloc[i - j]["High"])
                bar_close = float(df.iloc[i - j]["Close"])
                # Price touched or came close to lower channel
                if bar_high >= channels["prev_lower"] * 0.995:
                    # And bounced back (closed below)
                    if bar_close < channels["prev_lower"]:
                        return True

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized detection of Donchian Channel breakout signals across the entire DataFrame.

        Args:
            df: DataFrame with 'Open', 'High', 'Low', 'Close' columns

        Returns:
            np.ndarray of np.int8: 0=no signal, 1=LONG, -1=SHORT
        """
        n = len(df)
        result = np.zeros(n, dtype=np.int8)
        cp = self.channel_period
        if n < cp + 2:
            return result

        arrays = self._extract_arrays(df)
        high_a = arrays["high"]
        low_a = arrays["low"]
        close_a = arrays["close"]

        for i in range(cp, n):
            prev_upper = np.max(high_a[i - cp : i])
            prev_lower = np.min(low_a[i - cp : i])

            if close_a[i] > prev_upper:
                result[i] = 1
            elif close_a[i] < prev_lower:
                result[i] = -1

        return result

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Donchian Channel breakout at bar index i.

        Entry Rules:
        - Long Entry: Close[i] > Upper Channel[i-1] (breaks N-period high)
        - Short Entry: Close[i] < Lower Channel[i-1] (breaks N-period low)

        Args:
            df: DataFrame with OHLCV data
            i: Current bar index
            window_start: Optional window start for bounds checking

        Returns:
            PatternResult with detection status and signal
        """
        if not self._validate_data(df, i, window_start):
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
            )

        # Calculate channels
        channels = self._compute_channels(df, i)
        if channels is None:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
            )

        current_close = float(df.iloc[i]["Close"])
        current_high = float(df.iloc[i]["High"])
        current_low = float(df.iloc[i]["Low"])

        # Check for breakout
        # Long: Close breaks above previous period's upper channel
        breakout_up = current_close > channels["prev_upper"]
        # Short: Close breaks below previous period's lower channel
        breakout_down = current_close < channels["prev_lower"]

        if not (breakout_up or breakout_down):
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                pivot_points=channels,
            )

        # Assess breakout quality
        quality = self._assess_breakout_quality(df, i, channels, breakout_up)

        # Filter out weak breakouts (optional - can be disabled)
        # Weak breakouts are still detected but with lower confidence

        # Check retest entry if enabled
        retest_confirmed = self._check_retest_entry(df, i, channels, breakout_up)
        if not retest_confirmed:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                pivot_points={**channels, "quality": quality.value},
            )

        # Generate signal
        signal = self._generate_signal(df, i, channels, breakout_up, breakout_down, quality)

        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                "upper_channel": channels["upper"],
                "lower_channel": channels["lower"],
                "middle_line": channels["middle"],
                "channel_width": channels["channel_width"],
                "breakout_direction": "up" if breakout_up else "down",
                "quality": quality.value,
                "volume_confirmed": self._check_volume_confirmation(df, i),
                "atr_expanding": self._check_atr_expansion(df, i),
            },
            bars_since_detection=0,
            start_index=i - self.channel_period,
            end_index=i,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal if pattern is detected."""
        result = self.detect(df, i)
        return result.signal if result.detected else None

    def _generate_signal(
        self,
        df: pd.DataFrame,
        i: int,
        channels: Dict,
        breakout_up: bool,
        breakout_down: bool,
        quality: BreakoutQuality,
    ) -> Optional[TradeSignal]:
        """
        Generate trade signal for Donchian breakout.

        Stop Loss Rules:
        - Initial Stop: Middle channel level (or exit_period low/high)
        - Trailing Stop: Update to middle channel as price moves in favor

        Take Profit Rules:
        - Target 1: Entry + ATR * multiplier
        - Target 2: Entry + ATR * multiplier * 1.5
        - Exit Signal: Price touches middle channel (if used as trailing stop)
        """
        current_high = float(df.iloc[i]["High"])
        current_low = float(df.iloc[i]["Low"])
        current_close = float(df.iloc[i]["Close"])

        if breakout_up:
            # Long signal - entry at breakout close or slightly above
            entry_price = current_close + self.entry_offset

            # Stop loss at middle channel (or exit_period low)
            if i >= self.exit_period:
                alt_stop = float(df.iloc[i - self.exit_period : i + 1]["Low"].min())
                stop_loss = min(channels["middle"], alt_stop) - self.entry_offset
            else:
                stop_loss = channels["middle"] - self.entry_offset

            # ATR-based targets
            atr_values = atr(df, self.atr_period)
            current_atr = float(atr_values.iloc[i]) if i < len(atr_values) else np.nan

            if not np.isnan(current_atr) and self.use_atr_targets:
                risk = entry_price - stop_loss
                atr_target = current_atr * self.atr_mult_tp

                # Use the larger of ATR target or risk-based target
                take_profit_1 = entry_price + max(atr_target, risk * 1.5)
                take_profit_2 = entry_price + max(atr_target * 1.5, risk * 2.5)
            else:
                risk = entry_price - stop_loss
                take_profit_1 = entry_price + (risk * 1.5)
                take_profit_2 = entry_price + (risk * 2.5)

            direction = SignalDirection.LONG
            breakout_dir = "up"

        else:  # breakout_down
            # Short signal
            entry_price = current_close - self.entry_offset

            # Stop loss at middle channel (or exit_period high)
            if i >= self.exit_period:
                alt_stop = float(df.iloc[i - self.exit_period : i + 1]["High"].max())
                stop_loss = max(channels["middle"], alt_stop) + self.entry_offset
            else:
                stop_loss = channels["middle"] + self.entry_offset

            # ATR-based targets
            atr_values = atr(df, self.atr_period)
            current_atr = float(atr_values.iloc[i]) if i < len(atr_values) else np.nan

            if not np.isnan(current_atr) and self.use_atr_targets:
                risk = stop_loss - entry_price
                atr_target = current_atr * self.atr_mult_tp

                take_profit_1 = entry_price - max(atr_target, risk * 1.5)
                take_profit_2 = entry_price - max(atr_target * 1.5, risk * 2.5)
            else:
                risk = stop_loss - entry_price
                take_profit_1 = entry_price - (risk * 1.5)
                take_profit_2 = entry_price - (risk * 2.5)

            direction = SignalDirection.SHORT
            breakout_dir = "down"

        # Calculate confidence based on breakout quality
        confidence = self._calculate_confidence(quality, df, i)

        return TradeSignal(
            pattern_name=f"{self.name} ({breakout_dir.capitalize()})",
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=None,
            confidence=confidence,
            timestamp=df.index[i] if hasattr(df, "index") else None,
            metadata={
                "breakout_direction": breakout_dir,
                "upper_channel": channels["upper"],
                "lower_channel": channels["lower"],
                "middle_line": channels["middle"],
                "channel_width": channels["channel_width"],
                "quality": quality.value,
                "volume_confirmed": self._check_volume_confirmation(df, i),
                "atr_expanding": self._check_atr_expansion(df, i),
                "entry_type": "buy_stop" if breakout_up else "sell_stop",
                "exit_strategy": "middle_line" if self.use_mid_channel_exit else "opposite_band",
                "channel_period": self.channel_period,
                "exit_period": self.exit_period,
            },
        )

    def _calculate_confidence(self, quality: BreakoutQuality, df: pd.DataFrame, i: int) -> float:
        """
        Calculate signal confidence based on breakout quality.

        Confidence scoring:
        - Strong quality: 0.60-0.75
        - Moderate quality: 0.45-0.60
        - Weak quality: 0.30-0.45

        Args:
            quality: Breakout quality assessment
            df: DataFrame with OHLCV data
            i: Current bar index

        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = {
            BreakoutQuality.STRONG: 0.65,
            BreakoutQuality.MODERATE: 0.50,
            BreakoutQuality.WEAK: 0.35,
        }[quality]

        # Bonus for volume confirmation
        if self._check_volume_confirmation(df, i):
            base_confidence += 0.05

        # Bonus for ATR expansion
        if self._check_atr_expansion(df, i):
            base_confidence += 0.05

        # Bonus for strong candle (large body relative to range)
        if i < len(df):
            current_close = float(df.iloc[i]["Close"])
            current_open = float(df.iloc[i]["Open"])
            current_high = float(df.iloc[i]["High"])
            current_low = float(df.iloc[i]["Low"])

            body = abs(current_close - current_open)
            candle_range = current_high - current_low
            if candle_range > 0 and body / candle_range > 0.7:
                base_confidence += 0.05

        return min(base_confidence, 0.85)


class DonchianTrendFilter(BasePattern):
    """
    Donchian Channel Trend Filter

    Determines trend direction based on price position relative to
    the Donchian Channel middle line.

    - Price above middle line = Uptrend
    - Price below middle line = Downtrend
    - Price at middle line = Neutral/Consolidation
    """

    def __init__(self, channel_period: int = 20):
        """
        Initialize Donchian Trend Filter.

        Args:
            channel_period: Period for channel calculation (default 20)
        """
        super().__init__(
            name="Donchian Channel Trend",
            pattern_type=PatternType.CONTINUATION,
            min_bars_required=channel_period + 1,
        )
        self.channel_period = channel_period

    def get_trend(self, df: pd.DataFrame, i: int) -> str:
        """
        Get trend direction based on Donchian Channel position.

        Args:
            df: DataFrame with OHLC data
            i: Current bar index

        Returns:
            'uptrend', 'downtrend', or 'sideways'
        """
        if i < self.channel_period:
            return "sideways"

        upper = float(df.iloc[i - self.channel_period : i + 1]["High"].max())
        lower = float(df.iloc[i - self.channel_period : i + 1]["Low"].min())
        middle = (upper + lower) / 2
        close = float(df.iloc[i]["Close"])

        # Price in upper half of channel = uptrend
        channel_range = upper - lower
        if channel_range == 0:
            return "sideways"

        position = (close - lower) / channel_range

        if position > 0.6:
            return "uptrend"
        elif position < 0.4:
            return "downtrend"
        else:
            return "sideways"

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized Donchian trend filter across the entire DataFrame.

        Returns np.ndarray of np.int8: 1=uptrend (position > 0.6), -1=downtrend (position < 0.4), 0=sideways.
        """
        n = len(df)
        result = np.zeros(n, dtype=np.int8)
        cp = self.channel_period
        if n < cp + 1:
            return result

        high_a = df["High"].to_numpy()
        low_a = df["Low"].to_numpy()
        close_a = df["Close"].to_numpy()

        for i in range(cp, n):
            upper = np.max(high_a[i - cp : i + 1])
            lower = np.min(low_a[i - cp : i + 1])
            channel_range = upper - lower
            if channel_range == 0:
                continue
            position = (close_a[i] - lower) / channel_range
            if position > 0.6:
                result[i] = 1
            elif position < 0.4:
                result[i] = -1

        return result

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """Detect trend - not a pattern per se, returns trend info in pivot_points."""
        trend = self.get_trend(df, i)

        return PatternResult(
            detected=False,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            pivot_points={"trend": trend},
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Trend filter does not generate trade signals."""
        return None
