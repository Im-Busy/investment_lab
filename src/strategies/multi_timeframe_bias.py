"""
Multi-Timeframe Bias Detector

Detects directional bias across multiple timeframes using the influencer's framework:
- Higher timeframe (4H) determines directional bias
- Lower timeframe (30min) for entry timing
- Volume confirmation required

Key Principles:
1. 4H up → only look for longs on 30min pullbacks
2. 4H down → only look for shorts on 30min rallies
3. Wait for volume spike on reversal confirmation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ..indicators.technical import atr, ema, sma


class BiasState(Enum):
    """Multi-timeframe bias states."""

    BULLISH_ALIGNED = "bullish_aligned"  # 4H up, 30min up
    BULLISH_PULLBACK = "bullish_pullback"  # 4H up, 30min down (entry zone)
    BEARISH_ALIGNED = "bearish_aligned"  # 4H down, 30min down
    BEARISH_RALLY = "bearish_rally"  # 4H down, 30min up (entry zone)
    NEUTRAL = "neutral"  # No clear bias
    CONFLICTING = "conflicting"  # Timeframes disagree


class TrendDirection(Enum):
    """Trend direction."""

    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


@dataclass
class TimeframeBias:
    """
    Bias state for a single timeframe.

    Attributes:
        direction: Trend direction (up/down/sideways)
        confidence: Confidence score (0.0 to 1.0)
        sma_50_above_200: True if SMA50 > SMA200
        price_above_sma50: True if price > SMA50
        price_above_sma200: True if price > SMA200
        vwap_distance: Distance from VWAP (positive = above)
        higher_highs: True if making higher highs
        higher_lows: True if making higher lows
        momentum_score: Momentum score (-1.0 to 1.0)
    """

    direction: TrendDirection
    confidence: float
    sma_50_above_200: bool
    price_above_sma50: bool
    price_above_sma200: bool
    vwap_distance: float
    higher_highs: bool
    higher_lows: bool
    momentum_score: float


@dataclass
class MultiTimeframeSignal:
    """
    Multi-timeframe trading signal.

    Attributes:
        timestamp: Signal timestamp
        htftimeframe: Higher timeframe (e.g., '4H')
        ltftimeframe: Lower timeframe (e.g., '30min')
        htftime_bias: Higher timeframe bias
        ltftime_bias: Lower timeframe bias
        combined_bias: Combined bias state
        entry_direction: Recommended entry direction
        pullback_depth: Pullback depth as fraction (0.0 to 1.0)
        volume_confirmed: True if volume confirms
        confluence_score: Overall confluence score (0.0 to 1.0)
        metadata: Additional information
    """

    timestamp: pd.Timestamp
    htftimeframe: str
    ltftimeframe: str
    htftime_data: TimeframeBias
    ltftime_data: TimeframeBias
    combined_bias: BiasState
    entry_direction: Literal["long", "short", "none"]
    pullback_depth: float
    volume_confirmed: bool
    confluence_score: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class MTFConfig:
    """
    Multi-timeframe bias detector configuration.

    Attributes:
        htftimeframe: Higher timeframe name (e.g., '4H')
        ltftimeframe: Lower timeframe name (e.g., '30min')
        timeframe_ratio: Ratio between timeframes (default 8 for 4H/30min)
        sma_short: Short SMA period (default 50)
        sma_long: Long SMA period (default 200)
        atr_period: ATR calculation period
        pullback_threshold: Minimum pullback depth for entry (default 0.382)
        volume_multiplier: Volume spike threshold (default 2.0x average)
        lookback_highs_lows: Lookback for swing detection
        min_confidence: Minimum confidence for signals
    """

    htftimeframe: str = "4H"
    ltftimeframe: str = "30min"
    timeframe_ratio: int = 8
    sma_short: int = 50
    sma_long: int = 200
    atr_period: int = 14
    pullback_threshold: float = 0.382
    volume_multiplier: float = 2.0
    lookback_highs_lows: int = 20
    min_confidence: float = 0.5


class MultiTimeframeBiasDetector:
    """
    Multi-Timeframe Bias Detector

    Implements the influencer's framework:
    1. Determine 4H bias (directional bias)
    2. Wait for 30min pullback against 4H trend
    3. Enter when volume confirms 4H trend resumption

    Example:
        >>> config = MTFConfig(htftimeframe='4H', ltftimeframe='30min')
        >>> detector = MultiTimeframeBiasDetector(config)
        >>> signals = detector.detect(df_4h, df_30min)
    """

    def __init__(self, config: Optional[MTFConfig] = None):
        """
        Initialize Multi-Timeframe Bias Detector.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or MTFConfig()
        self._htfname = self.config.htftimeframe
        self._ltfname = self.config.ltftimeframe

    def detect(
        self,
        htftime_df: pd.DataFrame,
        ltftime_df: pd.DataFrame,
    ) -> List[MultiTimeframeSignal]:
        """
        Detect multi-timeframe bias signals.

        Args:
            htftime_df: Higher timeframe OHLCV DataFrame
            ltftime_df: Lower timeframe OHLCV DataFrame

        Returns:
            List of MultiTimeframeSignal objects
        """
        self._validate_dataframes(htftime_df, ltftime_df)

        # Calculate indicators for both timeframes
        htftime_bias = self._calculate_timeframe_bias(htftime_df)
        ltftime_bias_series = self._calculate_ltftime_bias_series(ltftime_df, htftime_df)

        # Generate signals
        signals = []
        for idx in range(len(ltftime_df)):
            if idx < self.config.lookback_highs_lows:
                continue

            signal = self._generate_signal(
                htftime_df=htftime_df,
                ltftime_df=ltftime_df,
                htftime_bias=htftime_bias,
                ltftime_idx=idx,
            )

            if signal and signal.confluence_score >= self.config.min_confidence:
                signals.append(signal)

        return signals

    def _validate_dataframes(self, htftime_df: pd.DataFrame, ltftime_df: pd.DataFrame) -> None:
        """Validate DataFrames have required columns."""
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_htf = [c for c in required_cols if c not in htftime_df.columns]
        missing_ltf = [c for c in required_cols if c not in ltftime_df.columns]

        if missing_htf:
            raise ValueError(f"Higher timeframe missing columns: {missing_htf}")
        if missing_ltf:
            raise ValueError(f"Lower timeframe missing columns: {missing_ltf}")

    def _calculate_timeframe_bias(self, df: pd.DataFrame) -> TimeframeBias:
        """
        Calculate bias state for a single timeframe.

        Args:
            df: OHLCV DataFrame

        Returns:
            TimeframeBias object
        """
        # Calculate SMAs
        sma_50 = sma(df, period=self.config.sma_short)
        sma_200 = sma(df, period=self.config.sma_long)

        current_price = df["Close"].iloc[-1]
        current_sma50 = sma_50.iloc[-1] if len(sma_50) > 0 else 0
        current_sma200 = sma_200.iloc[-1] if len(sma_200) > 0 else 0

        # SMA positioning
        sma_50_above_200 = current_sma50 > current_sma200
        price_above_sma50 = current_price > current_sma50
        price_above_sma200 = current_price > current_sma200

        # Detect higher highs/lows
        swing_highs = self._detect_swing_highs(df, lookback=self.config.lookback_highs_lows)
        swing_lows = self._detect_swing_lows(df, lookback=self.config.lookback_highs_lows)

        higher_highs = len(swing_highs) >= 2 and swing_highs[-1] > swing_highs[-2]
        higher_lows = len(swing_lows) >= 2 and swing_lows[-1] > swing_lows[-2]

        # Determine trend direction
        if sma_50_above_200 and price_above_sma50 and price_above_sma200:
            direction = TrendDirection.UP
            confidence = 0.7
        elif not sma_50_above_200 and not price_above_sma50 and not price_above_sma200:
            direction = TrendDirection.DOWN
            confidence = 0.7
        else:
            direction = TrendDirection.SIDEWAYS
            confidence = 0.4

        # Add structure confirmation
        if direction == TrendDirection.UP and higher_highs and higher_lows:
            confidence = min(1.0, confidence + 0.15)
        elif direction == TrendDirection.DOWN and not higher_highs and not higher_lows:
            confidence = min(1.0, confidence + 0.15)

        # Calculate VWAP distance
        vwap = self._calculate_vwap(df)
        vwap_distance = (current_price - vwap.iloc[-1]) / vwap.iloc[-1] if len(vwap) > 0 else 0

        # Momentum score (simplified RSI-based)
        momentum_score = self._calculate_momentum(df)

        # Adjust confidence for sideways
        if direction == TrendDirection.SIDEWAYS:
            confidence = 0.3

        return TimeframeBias(
            direction=direction,
            confidence=confidence,
            sma_50_above_200=sma_50_above_200,
            price_above_sma50=price_above_sma50,
            price_above_sma200=price_above_sma200,
            vwap_distance=vwap_distance,
            higher_highs=higher_highs,
            higher_lows=higher_lows,
            momentum_score=momentum_score,
        )

    def _calculate_ltftime_bias_series(
        self, ltftime_df: pd.DataFrame, htftime_df: pd.DataFrame
    ) -> pd.Series:
        """Calculate lower timeframe bias for each bar."""
        # Implementation for rolling LTf bias
        pass

    def _detect_swing_highs(self, df: pd.DataFrame, lookback: int = 20) -> List[float]:
        """Detect swing highs in DataFrame."""
        highs = df["High"].to_numpy()
        swing_highs = []

        for i in range(lookback, len(highs) - lookback):
            left_max = np.max(highs[i - lookback : i])
            right_max = np.max(highs[i + 1 : i + lookback + 1])

            if highs[i] > left_max and highs[i] > right_max:
                swing_highs.append(highs[i])

        return swing_highs

    def _detect_swing_lows(self, df: pd.DataFrame, lookback: int = 20) -> List[float]:
        """Detect swing lows in DataFrame."""
        lows = df["Low"].to_numpy()
        swing_lows = []

        for i in range(lookback, len(lows) - lookback):
            left_min = np.min(lows[i - lookback : i])
            right_min = np.min(lows[i + 1 : i + lookback + 1])

            if lows[i] < left_min and lows[i] < right_min:
                swing_lows.append(lows[i])

        return swing_lows

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP (session-resetting for intraday)."""
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
        pv = typical_price * df["Volume"]

        # Group by date for session reset
        if df.index.tz is not None:
            sessions = df.index.tz_localize(None).date
        else:
            sessions = df.index.date

        vwap = pv.groupby(sessions).cumsum() / df["Volume"].groupby(sessions).cumsum()
        return vwap

    def _calculate_momentum(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate momentum score (-1.0 to 1.0)."""
        close = df["Close"]
        if len(close) < period:
            return 0.0

        changes = close.diff(period)
        current_change = changes.iloc[-1]

        if pd.isna(current_change):
            return 0.0

        # Normalize by ATR
        current_atr = atr(df, period=period).iloc[-1] if len(df) > period else 1.0

        if current_atr <= 0:
            return 0.0

        momentum = current_change / current_atr
        return np.clip(momentum / 3.0, -1.0, 1.0)  # Normalize to -1 to 1

    def _generate_signal(
        self,
        htftime_df: pd.DataFrame,
        ltftime_df: pd.DataFrame,
        htftime_bias: TimeframeBias,
        ltftime_idx: int,
    ) -> Optional[MultiTimeframeSignal]:
        """
        Generate multi-timeframe signal.

        Args:
            htftime_df: Higher timeframe DataFrame
            ltftime_df: Lower timeframe DataFrame
            htftime_bias: Higher timeframe bias state
            ltftime_idx: Lower timeframe bar index

        Returns:
            MultiTimeframeSignal or None
        """
        # Calculate LTf bias at this index
        ltf_slice = ltftime_df.iloc[: ltftime_idx + 1]
        ltftime_bias = self._calculate_timeframe_bias(ltf_slice)

        # Determine combined bias state
        combined_bias = self._determine_combined_bias(htftime_bias, ltftime_bias)

        # Determine entry direction
        entry_direction = self._determine_entry_direction(combined_bias, htftime_bias, ltftime_bias)

        if entry_direction == "none":
            return None

        # Calculate pullback depth
        pullback_depth = self._calculate_pullback_depth(
            ltftime_df, ltftime_idx, htftime_bias.direction
        )

        # Check volume confirmation
        volume_confirmed = self._check_volume_confirmation(ltftime_df, ltftime_idx)

        # Calculate confluence score
        confluence_score = self._calculate_confluence_score(
            htftime_bias=htftime_bias,
            ltftime_bias=ltftime_bias,
            pullback_depth=pullback_depth,
            volume_confirmed=volume_confirmed,
        )

        timestamp = ltftime_df.index[ltftime_idx]

        return MultiTimeframeSignal(
            timestamp=timestamp,
            htftimeframe=self._htfname,
            ltftimeframe=self._ltfname,
            htftime_data=htftime_bias,
            ltftime_data=ltftime_bias,
            combined_bias=combined_bias,
            entry_direction=entry_direction,
            pullback_depth=pullback_depth,
            volume_confirmed=volume_confirmed,
            confluence_score=confluence_score,
            metadata={
                "htf_confidence": htftime_bias.confidence,
                "ltf_confidence": ltftime_bias.confidence,
                "htf_momentum": htftime_bias.momentum_score,
                "ltf_momentum": ltftime_bias.momentum_score,
            },
        )

    def _determine_combined_bias(
        self, htftime_bias: TimeframeBias, ltftime_bias: TimeframeBias
    ) -> BiasState:
        """Determine combined bias state from both timeframes."""
        htf_up = htftime_bias.direction == TrendDirection.UP
        htf_down = htftime_bias.direction == TrendDirection.DOWN
        ltf_up = ltftime_bias.direction == TrendDirection.UP
        ltf_down = ltftime_bias.direction == TrendDirection.DOWN

        if htf_up and ltf_up:
            return BiasState.BULLISH_ALIGNED
        elif htf_up and ltf_down:
            return BiasState.BULLISH_PULLBACK
        elif htf_down and ltf_down:
            return BiasState.BEARISH_ALIGNED
        elif htf_down and ltf_up:
            return BiasState.BEARISH_RALLY
        elif htftime_bias.direction == TrendDirection.SIDEWAYS:
            return BiasState.NEUTRAL
        else:
            return BiasState.CONFLICTING

    def _determine_entry_direction(
        self,
        combined_bias: BiasState,
        htftime_bias: TimeframeBias,
        ltftime_bias: TimeframeBias,
    ) -> Literal["long", "short", "none"]:
        """Determine entry direction based on bias states."""
        if combined_bias == BiasState.BULLISH_PULLBACK:
            # 4H up, 30min down = buy the dip
            if ltftime_bias.momentum_score < -0.3:  # LTF oversold
                return "long"
        elif combined_bias == BiasState.BEARISH_RALLY:
            # 4H down, 30min up = sell the rally
            if ltftime_bias.momentum_score > 0.3:  # LTF overbought
                return "short"
        elif combined_bias == BiasState.BULLISH_ALIGNED:
            # Both up = look for long continuation
            if htftime_bias.confidence > 0.7 and ltftime_bias.confidence > 0.5:
                return "long"
        elif combined_bias == BiasState.BEARISH_ALIGNED:
            # Both down = look for short continuation
            if htftime_bias.confidence > 0.7 and ltftime_bias.confidence > 0.5:
                return "short"

        return "none"

    def _calculate_pullback_depth(
        self, df: pd.DataFrame, idx: int, htf_direction: TrendDirection
    ) -> float:
        """
        Calculate pullback depth as Fibonacci ratio.

        Args:
            df: DataFrame
            idx: Current bar index
            htf_direction: Higher timeframe direction

        Returns:
            Pullback depth (0.0 to 1.0, where 0.5 = 50% retracement)
        """
        lookback = self.config.lookback_highs_lows
        start_idx = max(0, idx - lookback * 2)

        if htf_direction == TrendDirection.UP:
            # Find recent swing low and high
            slice_df = df.iloc[start_idx : idx + 1]
            swing_low = slice_df["Low"].min()
            swing_high = slice_df["High"].max()

            if swing_high <= swing_low:
                return 0.0

            current_price = df["Close"].iloc[idx]
            retracement = (swing_high - current_price) / (swing_high - swing_low)
            return np.clip(retracement, 0.0, 1.0)
        else:
            # Downtweet pullback (rally)
            slice_df = df.iloc[start_idx : idx + 1]
            swing_low = slice_df["Low"].min()
            swing_high = slice_df["High"].max()

            if swing_high <= swing_low:
                return 0.0

            current_price = df["Close"].iloc[idx]
            retracement = (current_price - swing_low) / (swing_high - swing_low)
            return np.clip(retracement, 0.0, 1.0)

    def _check_volume_confirmation(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Check if volume confirms the signal.

        Volume must be >= threshold × average volume.

        Args:
            df: DataFrame
            idx: Bar index

        Returns:
            True if volume confirms
        """
        lookback = 20
        start_idx = max(0, idx - lookback)

        if idx < lookback:
            return False

        recent_volume = df["Volume"].iloc[start_idx:idx]
        avg_volume = recent_volume.mean()
        current_volume = df["Volume"].iloc[idx]

        return current_volume >= avg_volume * self.config.volume_multiplier

    def _calculate_confluence_score(
        self,
        htftime_bias: TimeframeBias,
        ltftime_bias: TimeframeBias,
        pullback_depth: float,
        volume_confirmed: bool,
    ) -> float:
        """
        Calculate overall confluence score.

        Scoring (from influencer's framework):
        - 4H trend direction: +1 if matches trade direction
        - 30min pullback to Fib level: +1 if 38.2/50/61.8%
        - Volume spike: +1 if 2x+ average
        - VWAP reclaim: +1 if price reclaimed VWAP
        - SMA 50/200 alignment: +1 if price on correct side

        Args:
            htftime_bias: Higher timeframe bias
            ltftime_bias: Lower timeframe bias
            pullback_depth: Pullback depth (Fib ratio)
            volume_confirmed: Volume confirmation status

        Returns:
            Confluence score (0.0 to 1.0)
        """
        score = 0.0
        max_score = 5.0

        # Factor 1: 4H trend alignment
        if htftime_bias.direction == TrendDirection.UP and htftime_bias.sma_50_above_200:
            score += 1.0
        elif htftime_bias.direction == TrendDirection.DOWN and not htftime_bias.sma_50_above_200:
            score += 1.0

        # Factor 2: Pullback to Fib level
        fib_levels = [0.382, 0.5, 0.618]
        if any(abs(pullback_depth - fib) < 0.05 for fib in fib_levels):
            score += 1.0

        # Factor 3: Volume confirmation
        if volume_confirmed:
            score += 1.0

        # Factor 4: VWAP reclaim
        if abs(ltftime_bias.vwap_distance) < 0.01:  # Near VWAP
            score += 1.0

        # Factor 5: SMA alignment
        if ltftime_bias.price_above_sma50 == ltftime_bias.price_above_sma200:
            score += 0.5

        # Normalize
        return score / max_score

    def get_bias_summary(
        self, htftime_df: pd.DataFrame, ltftime_df: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Get current bias summary for logging/display.

        Args:
            htftime_df: Higher timeframe DataFrame
            ltftime_df: Lower timeframe DataFrame

        Returns:
            Dictionary with bias summary
        """
        htftime_bias = self._calculate_timeframe_bias(htftime_df)
        ltftime_bias = self._calculate_timeframe_bias(ltftime_df)
        combined_bias = self._determine_combined_bias(htftime_bias, ltftime_bias)

        return {
            "htftimeframe": self._htfname,
            "htftime_direction": htftime_bias.direction.value,
            "htftime_confidence": htftime_bias.confidence,
            "htftime_sma50_200": "bullish" if htftime_bias.sma_50_above_200 else "bearish",
            "ltftimeframe": self._ltfname,
            "ltftime_direction": ltftime_bias.direction.value,
            "ltftime_confidence": ltftime_bias.confidence,
            "combined_bias": combined_bias.value,
            "recommendation": self._get_recommendation(combined_bias),
        }

    def _get_recommendation(self, bias: BiasState) -> str:
        """Get trading recommendation based on bias state."""
        recommendations = {
            BiasState.BULLISH_ALIGNED: "Look for long entries on minor pullbacks",
            BiasState.BULLISH_PULLBACK: "Wait for volume confirmation, then enter long",
            BiasState.BEARISH_ALIGNED: "Look for short entries on minor rallies",
            BiasState.BEARISH_RALLY: "Wait for volume confirmation, then enter short",
            BiasState.NEUTRAL: "No clear bias - stay out or reduce size",
            BiasState.CONFLICTING: "Timeframes conflict - wait for alignment",
        }
        return recommendations.get(bias, "No recommendation")
