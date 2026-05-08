"""
Volume Confirmation Module

Volume is the only indicator not derived from price data.
This module provides volume-based confirmation signals.

Key Principles:
1. Volume confirms price action (validates moves)
2. Volume spikes indicate institutional participation
3. Low volume moves are suspect (likely to reverse)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..indicators.technical import sma


class VolumeSignalType(Enum):
    """Volume signal types."""

    SPIKE = "spike"  # Abnormal volume surge
    DRY_UP = "dry_up"  # Volume exhaustion
    ACCUMULATION = "accumulation"  # Sustained high volume on up moves
    DISTRIBUTION = "distribution"  # Sustained high volume on down moves
    DIVERGENCE = "divergence"  # Price/volume divergence


@dataclass
class VolumeSignal:
    """
    Volume confirmation signal.

    Attributes:
        timestamp: Signal timestamp
        signal_type: Type of volume signal
        volume_ratio: Current volume vs average volume
        price_change: Price change on the volume
        confirmation_strength: Strength of confirmation (0.0 to 1.0)
        direction: Confirmed direction ('long', 'short', or 'none')
        metadata: Additional information
    """

    timestamp: pd.Timestamp
    signal_type: VolumeSignalType
    volume_ratio: float
    price_change: float
    confirmation_strength: float
    direction: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class VolumeConfig:
    """
    Volume confirmation configuration.

    Attributes:
        volume_period: Period for volume average (default 20)
        spike_threshold: Volume spike multiplier (default 2.0)
        dry_up_threshold: Volume dry-up fraction (default 0.5)
        accumulation_period: Period for accumulation detection
        divergence_lookback: Lookback for divergence detection
        min_volume_ratio: Minimum volume ratio for confirmation
    """

    volume_period: int = 20
    spike_threshold: float = 2.0
    dry_up_threshold: float = 0.5
    accumulation_period: int = 10
    divergence_lookback: int = 14
    min_volume_ratio: float = 1.5


class VolumeConfirmation:
    """
    Volume Confirmation System

    Provides volume-based confirmation for price moves.

    Volume Analysis Rules:
    1. Volume must confirm direction (up + high volume = bullish)
    2. Volume spikes (>2x average) indicate institutional activity
    3. Low volume rallies/declines are suspect
    4. Volume dry-up after trend = potential reversal

    Example:
        >>> config = VolumeConfirmationConfig(spike_threshold=2.0)
        >>> volume = VolumeConfirmation(config)
        >>> signals = volume.confirm(df)
    """

    def __init__(self, config: Optional[VolumeConfig] = None):
        """
        Initialize Volume Confirmation.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or VolumeConfig()
        self._volume_avg: Optional[pd.Series] = None
        self._volume_ratio: Optional[pd.Series] = None

    def analyze(self, df: pd.DataFrame) -> List[VolumeSignal]:
        """
        Analyze volume for confirmation signals.

        Args:
            df: OHLCV DataFrame

        Returns:
            List of VolumeSignal objects
        """
        self._validate_dataframe(df)

        # Calculate volume metrics
        self._calculate_volume_metrics(df)

        signals = []
        for i in range(self.config.volume_period, len(df)):
            signal = self._detect_signals(df, i)
            if signal:
                signals.append(signal)

        return signals

    def confirm_move(
        self, df: pd.DataFrame, direction: str, idx: Optional[int] = None
    ) -> Tuple[bool, float]:
        """
        Confirm a price move with volume.

        Args:
            df: OHLCV DataFrame
            direction: Expected direction ('long' or 'short')
            idx: Bar index (default: last bar)

        Returns:
            Tuple of (confirmed: bool, strength: float)
        """
        if idx is None:
            idx = len(df) - 1

        if idx < self.config.volume_period:
            return False, 0.0

        # Calculate volume ratio
        if self._volume_ratio is None:
            self._calculate_volume_metrics(df)

        volume_ratio = self._volume_ratio.iloc[idx] if self._volume_ratio is not None else 1.0
        price_change = df["Close"].iloc[idx] - df["Close"].iloc[idx - 1]

        # Confirm based on direction
        if direction == "long":
            if price_change > 0 and volume_ratio >= self.config.min_volume_ratio:
                strength = min(1.0, volume_ratio / self.config.spike_threshold)
                return True, strength
        elif direction == "short":
            if price_change < 0 and volume_ratio >= self.config.min_volume_ratio:
                strength = min(1.0, volume_ratio / self.config.spike_threshold)
                return True, strength

        return False, 0.0

    def is_volume_spike(self, df: pd.DataFrame, idx: Optional[int] = None) -> bool:
        """
        Check if current bar has volume spike.

        Args:
            df: OHLCV DataFrame
            idx: Bar index (default: last bar)

        Returns:
            True if volume spike detected
        """
        if idx is None:
            idx = len(df) - 1

        if idx < self.config.volume_period:
            return False

        if self._volume_ratio is None:
            self._calculate_volume_metrics(df)

        volume_ratio = self._volume_ratio.iloc[idx] if self._volume_ratio is not None else 1.0
        return volume_ratio >= self.config.spike_threshold

    def is_volume_dry_up(self, df: pd.DataFrame, idx: Optional[int] = None) -> bool:
        """
        Check if volume has dried up (exhaustion signal).

        Args:
            df: OHLCV DataFrame
            idx: Bar index (default: last bar)

        Returns:
            True if volume dry-up detected
        """
        if idx is None:
            idx = len(df) - 1

        if idx < self.config.volume_period:
            return False

        if self._volume_ratio is None:
            self._calculate_volume_metrics(df)

        volume_ratio = self._volume_ratio.iloc[idx] if self._volume_ratio is not None else 1.0
        return volume_ratio <= self.config.dry_up_threshold

    def detect_accumulation(
        self, df: pd.DataFrame, idx: Optional[int] = None, min_bars: int = 3
    ) -> bool:
        """
        Detect accumulation pattern (high volume on up bars).

        Args:
            df: OHLCV DataFrame
            idx: Bar index (default: last bar)
            min_bars: Minimum consecutive accumulation bars

        Returns:
            True if accumulation detected
        """
        if idx is None:
            idx = len(df) - 1

        if idx < min_bars:
            return False

        # Count up bars with above-average volume
        accum_count = 0
        for i in range(idx - min_bars + 1, idx + 1):
            if i < self.config.volume_period:
                continue

            price_up = df["Close"].iloc[i] > df["Open"].iloc[i]
            volume_ratio = self._volume_ratio.iloc[i] if self._volume_ratio is not None else 1.0

            if price_up and volume_ratio >= 1.2:
                accum_count += 1

        return accum_count >= min_bars

    def detect_distribution(
        self, df: pd.DataFrame, idx: Optional[int] = None, min_bars: int = 3
    ) -> bool:
        """
        Detect distribution pattern (high volume on down bars).

        Args:
            df: OHLCV DataFrame
            idx: Bar index (default: last bar)
            min_bars: Minimum consecutive distribution bars

        Returns:
            True if distribution detected
        """
        if idx is None:
            idx = len(df) - 1

        if idx < min_bars:
            return False

        # Count down bars with above-average volume
        dist_count = 0
        for i in range(idx - min_bars + 1, idx + 1):
            if i < self.config.volume_period:
                continue

            price_down = df["Close"].iloc[i] < df["Open"].iloc[i]
            volume_ratio = self._volume_ratio.iloc[i] if self._volume_ratio is not None else 1.0

            if price_down and volume_ratio >= 1.2:
                dist_count += 1

        return dist_count >= min_bars

    def detect_volume_divergence(
        self, df: pd.DataFrame, idx: Optional[int] = None
    ) -> Optional[str]:
        """
        Detect price/volume divergence.

        Bullish divergence: Price makes lower low, volume makes higher low
        Bearish divergence: Price makes higher high, volume makes lower high

        Args:
            df: OHLCV DataFrame
            idx: Bar index (default: last bar)

        Returns:
            'bullish', 'bearish', or None
        """
        if idx is None:
            idx = len(df) - 1

        lookback = self.config.divergence_lookback
        start_idx = idx - lookback

        if start_idx < 0:
            return None

        # Get price extremes
        price_high = df["High"].iloc[start_idx:idx].max()
        price_low = df["Low"].iloc[start_idx:idx].min()
        volume_high = df["Volume"].iloc[start_idx:idx].max()

        current_price = df["Close"].iloc[idx]
        current_volume = df["Volume"].iloc[idx]
        avg_volume = self._volume_avg.iloc[idx] if self._volume_avg is not None else current_volume

        # Bearish divergence: price at new high, volume declining
        if current_price >= price_high and current_volume < avg_volume * 0.8:
            return "bearish"

        # Bullish divergence: price at new low, volume declining
        if current_price <= price_low and current_volume < avg_volume * 0.8:
            return "bullish"

        return None

    def _calculate_volume_metrics(self, df: pd.DataFrame) -> None:
        """Calculate volume average and ratio series."""
        self._volume_avg = sma(df["Volume"], period=self.config.volume_period)
        self._volume_ratio = df["Volume"] / self._volume_avg

    def _detect_signals(self, df: pd.DataFrame, idx: int) -> Optional[VolumeSignal]:
        """Detect volume signals at a specific bar."""
        if idx < self.config.volume_period:
            return None

        current_volume = df["Volume"].iloc[idx]
        avg_volume = self._volume_avg.iloc[idx] if self._volume_avg is not None else current_volume
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        price_change = df["Close"].iloc[idx] - df["Close"].iloc[idx - 1]
        price_change_pct = (
            price_change / df["Close"].iloc[idx - 1] if df["Close"].iloc[idx - 1] > 0 else 0
        )

        timestamp = df.index[idx]

        # Detect signal type
        signal_type = None
        direction = "none"
        strength = 0.0

        if volume_ratio >= self.config.spike_threshold:
            signal_type = VolumeSignalType.SPIKE
            if price_change > 0:
                direction = "long"
                strength = min(1.0, volume_ratio / self.config.spike_threshold)
            elif price_change < 0:
                direction = "short"
                strength = min(1.0, volume_ratio / self.config.spike_threshold)

        elif volume_ratio <= self.config.dry_up_threshold:
            signal_type = VolumeSignalType.DRY_UP
            strength = 1.0 - volume_ratio

        elif self.detect_accumulation(df, idx):
            signal_type = VolumeSignalType.ACCUMULATION
            direction = "long"
            strength = 0.7

        elif self.detect_distribution(df, idx):
            signal_type = VolumeSignalType.DISTRIBUTION
            direction = "short"
            strength = 0.7

        divergence = self.detect_volume_divergence(df, idx)
        if divergence:
            signal_type = VolumeSignalType.DIVERGENCE
            direction = divergence
            strength = 0.8

        if signal_type is None:
            return None

        return VolumeSignal(
            timestamp=timestamp,
            signal_type=signal_type,
            volume_ratio=volume_ratio,
            price_change=price_change_pct,
            confirmation_strength=strength,
            direction=direction,
            metadata={
                "current_volume": float(current_volume),
                "avg_volume": float(avg_volume) if avg_volume else 0,
                "is_spike": volume_ratio >= self.config.spike_threshold,
                "is_dry_up": volume_ratio <= self.config.dry_up_threshold,
                "divergence": divergence,
            },
        )

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate DataFrame has required columns."""
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

    def get_volume_profile(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Get current volume profile summary.

        Args:
            df: OHLCV DataFrame

        Returns:
            Dictionary with volume metrics
        """
        if len(df) < self.config.volume_period:
            return {"error": "Insufficient data"}

        if self._volume_ratio is None:
            self._calculate_volume_metrics(df)

        current_volume = df["Volume"].iloc[-1]
        current_ratio = self._volume_ratio.iloc[-1] if self._volume_ratio is not None else 1.0

        return {
            "current_volume": float(current_volume),
            "average_volume": float(self._volume_avg.iloc[-1])
            if self._volume_avg is not None
            else 0,
            "volume_ratio": float(current_ratio),
            "is_spike": bool(current_ratio >= self.config.spike_threshold),
            "is_dry_up": bool(current_ratio <= self.config.dry_up_threshold),
            "accumulation_detected": self.detect_accumulation(df),
            "distribution_detected": self.detect_distribution(df),
            "divergence": self.detect_volume_divergence(df),
        }
