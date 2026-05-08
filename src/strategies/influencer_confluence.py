"""
Influencer Confluence Scorer

Combines all influencer framework components into a unified scoring system:
1. Multi-timeframe bias (4H + 30min)
2. Volume confirmation
3. VWAP + SMA confluence
4. Fibonacci pullback levels
5. Pattern compatibility (for SMC integration)

Confluence Scoring (0-5 scale):
- 4H trend direction matches: +1
- 30min pullback to Fib level: +1
- Volume spike (2x+ average): +1
- Price reclaimed VWAP: +1
- Price on correct side of SMA 50/200: +1

Minimum score to trade: 4/5
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

from ..patterns.base import BasePattern
from .multi_timeframe_bias import (
    MTFConfig,
    MultiTimeframeBiasDetector,
    MultiTimeframeSignal,
    TimeframeBias,
    TrendDirection,
)
from .volume_confirmation import VolumeConfirmation, VolumeConfig
from .vwap_sma_confluence import (
    VWAPConfig,
    VWAPConfluenceAnalyzer,
)


class ConfluenceLevel(Enum):
    """Confluence confidence levels."""

    LOW = "low"  # Score < 3
    MEDIUM = "medium"  # Score 3-4
    HIGH = "high"  # Score >= 4


@dataclass
class ConfluenceSignal:
    """
    Combined confluence trading signal.

    Attributes:
        timestamp: Signal timestamp
        direction: Trade direction ('long' or 'short')
        entry_price: Suggested entry price
        stop_loss: Stop loss price
        take_profit: Take profit target
        confluence_score: Overall score (0.0 to 5.0)
        confluence_level: Confidence level
        breakdown: Detailed scoring breakdown
        htf_bias: Higher timeframe bias state
        ltf_bias: Lower timeframe bias state
        volume_confirmed: Volume confirmation status
        vwap_status: VWAP relationship
        fib_level: Fibonacci pullback level
        metadata: Additional information
    """

    timestamp: pd.Timestamp
    direction: Literal["long", "short", "none"]
    entry_price: float
    stop_loss: float
    take_profit: float
    confluence_score: float
    confluence_level: ConfluenceLevel
    breakdown: Dict[str, float]
    htf_bias: TimeframeBias
    ltf_bias: Optional[TimeframeBias]
    volume_confirmed: bool
    vwap_status: str
    fib_level: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class InfluencerConfig:
    """
    Influencer framework configuration.

    Attributes:
        htf_timeframe: Higher timeframe name
        ltf_timeframe: Lower timeframe name
        min_confluence_score: Minimum score to trade (default 4.0)
        volume_multiplier: Volume spike threshold (default 2.0)
        fib_levels: Fibonacci levels to watch
        atr_period: ATR for stop calculation
        risk_per_trade: Risk per trade as fraction
        require_all_factors: Require all 5 factors (strict mode)
    """

    htf_timeframe: str = "4H"
    ltf_timeframe: str = "30min"
    min_confluence_score: float = 4.0
    volume_multiplier: float = 2.0
    fib_levels: List[float] = field(default_factory=lambda: [0.382, 0.5, 0.618])
    atr_period: int = 14
    risk_per_trade: float = 0.01
    require_all_factors: bool = False


class InfluencerConfluenceScorer:
    """
    Influencer Confluence Scorer

    Implements the influencer's complete framework:
    1. 4H determines bias
    2. Wait for 30min inverse move (pullback)
    3. Enter when volume confirms + VWAP reclaimed

    Scoring (5 factors, 1 point each):
    - ✓ 4H trend direction matches trade
    - ✓ 30min pullback to Fib level (38.2/50/61.8%)
    - ✓ Volume spike (2x+ average)
    - ✓ Price reclaimed VWAP
    - ✓ Price on correct side of SMA 50/200

    Minimum score: 4/5 (or 5/5 in strict mode)

    Example:
        >>> config = InfluencerConfig()
        >>> scorer = InfluencerConfluenceScorer(config)
        >>> signals = scorer.scan(df_4h, df_30min)
    """

    def __init__(
        self,
        config: Optional[InfluencerConfig] = None,
        patterns: Optional[List[BasePattern]] = None,
    ):
        """
        Initialize Influencer Confluence Scorer.

        Args:
            config: Configuration (uses defaults if None)
            patterns: Optional list of SMC patterns for additional confluence
        """
        self.config = config or InfluencerConfig()
        self._patterns = patterns or []

        # Initialize sub-modules
        self._mtf_detector = MultiTimeframeBiasDetector(
            MTFConfig(
                htftimeframe=self.config.htf_timeframe,
                ltftimeframe=self.config.ltf_timeframe,
                volume_multiplier=self.config.volume_multiplier,
            )
        )
        self._volume_confirmation = VolumeConfirmation(
            VolumeConfig(
                volume_period=20,
                spike_threshold=self.config.volume_multiplier,
            )
        )
        self._vwap_analyzer = VWAPConfluenceAnalyzer(
            VWAPConfig(
                sma_short=50,
                sma_long=200,
            )
        )

    def scan(
        self,
        htf_df: pd.DataFrame,
        ltf_df: pd.DataFrame,
    ) -> List[ConfluenceSignal]:
        """
        Scan for confluence signals across both timeframes.

        Args:
            htf_df: Higher timeframe OHLCV DataFrame
            ltf_df: Lower timeframe OHLCV DataFrame

        Returns:
            List of ConfluenceSignal objects
        """
        self._validate_dataframes(htf_df, ltf_df)

        # Get MTF signals
        mtf_signals = self._mtf_detector.detect(htf_df, ltf_df)

        # Generate confluence signals
        confluence_signals = []
        for mtf_signal in mtf_signals:
            signal = self._generate_confluence_signal(mtf_signal, htf_df, ltf_df)
            if signal and signal.confluence_score >= self.config.min_confluence_score:
                confluence_signals.append(signal)

        return confluence_signals

    def _generate_confluence_signal(
        self,
        mtf_signal: MultiTimeframeSignal,
        htf_df: pd.DataFrame,
        ltf_df: pd.DataFrame,
    ) -> Optional[ConfluenceSignal]:
        """Generate confluence signal from MTF signal."""
        if mtf_signal.entry_direction == "none":
            return None

        # Calculate individual factor scores
        breakdown = {}
        total_score = 0.0

        # Factor 1: 4H trend alignment
        factor_1 = self._score_htf_alignment(mtf_signal.htftime_data)
        breakdown["htf_alignment"] = factor_1
        total_score += factor_1

        # Factor 2: Fibonacci pullback level
        factor_2, fib_level = self._score_fib_pullback(ltf_df, mtf_signal.htftime_data.direction)
        breakdown["fib_pullback"] = factor_2
        total_score += factor_2

        # Factor 3: Volume confirmation
        factor_3 = self._score_volume_confirm(ltf_df)
        breakdown["volume_confirm"] = factor_3
        total_score += factor_3

        # Factor 4: VWAP reclaim
        factor_4, vwap_status = self._score_vwap_reclaim(ltf_df, mtf_signal.entry_direction)
        breakdown["vwap_reclaim"] = factor_4
        total_score += factor_4

        # Factor 5: SMA 50/200 alignment
        factor_5 = self._score_sma_alignment(mtf_signal.htftime_data, mtf_signal.ltftime_data)
        breakdown["sma_alignment"] = factor_5
        total_score += factor_5

        # Check minimum score
        if total_score < self.config.min_confluence_score:
            return None

        # Determine confluence level
        confluence_level = self._get_confluence_level(total_score)

        # Calculate entry, stop, target
        entry_price = ltf_df["Close"].iloc[-1]
        stop_loss = self._calculate_stop_loss(ltf_df, mtf_signal.entry_direction)
        take_profit = self._calculate_take_profit(
            entry_price, stop_loss, mtf_signal.entry_direction
        )

        # Get current biases
        htf_bias = mtf_signal.htftime_data
        ltf_bias = mtf_signal.ltftime_data

        timestamp = mtf_signal.timestamp

        return ConfluenceSignal(
            timestamp=timestamp,
            direction=mtf_signal.entry_direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confluence_score=total_score,
            confluence_level=confluence_level,
            breakdown=breakdown,
            htf_bias=htf_bias,
            ltf_bias=ltf_bias,
            volume_confirmed=mtf_signal.volume_confirmed,
            vwap_status=vwap_status,
            fib_level=fib_level,
            metadata={
                "pullback_depth": mtf_signal.pullback_depth,
                "htf_confidence": htf_bias.confidence,
                "ltf_confidence": ltf_bias.confidence if ltf_bias else 0,
                "combined_bias": mtf_signal.combined_bias.value,
            },
        )

    def _score_htf_alignment(self, htf_bias: TimeframeBias) -> float:
        """Score factor 1: 4H trend alignment (0 or 1)."""
        if htf_bias.direction == TrendDirection.UP and htf_bias.sma_50_above_200:
            return 1.0
        elif htf_bias.direction == TrendDirection.DOWN and not htf_bias.sma_50_above_200:
            return 1.0
        return 0.0

    def _score_fib_pullback(
        self, df: pd.DataFrame, direction: TrendDirection
    ) -> Tuple[float, float]:
        """Score factor 2: Fibonacci pullback level (0 or 1)."""
        lookback = 20
        start_idx = max(0, len(df) - lookback)

        if direction == TrendDirection.UP:
            swing_high = df["High"].iloc[start_idx:].max()
            swing_low = df["Low"].iloc[start_idx:].min()
            current_price = df["Close"].iloc[-1]

            if swing_high <= swing_low:
                return 0.0, 0.0

            retracement = (swing_high - current_price) / (swing_high - swing_low)

            # Check if near key Fib level
            for fib_level in self.config.fib_levels:
                if abs(retracement - fib_level) < 0.05:  # 5% tolerance
                    return 1.0, fib_level

            return 0.0, retracement
        else:
            swing_high = df["High"].iloc[start_idx:].max()
            swing_low = df["Low"].iloc[start_idx:].min()
            current_price = df["Close"].iloc[-1]

            if swing_high <= swing_low:
                return 0.0, 0.0

            retracement = (current_price - swing_low) / (swing_high - swing_low)

            for fib_level in self.config.fib_levels:
                if abs(retracement - fib_level) < 0.05:
                    return 1.0, fib_level

            return 0.0, retracement

    def _score_volume_confirm(self, df: pd.DataFrame) -> float:
        """Score factor 3: Volume confirmation (0 or 1)."""
        current_volume = df["Volume"].iloc[-1]
        avg_volume = df["Volume"].iloc[-20:].mean() if len(df) >= 20 else current_volume

        if current_volume >= avg_volume * self.config.volume_multiplier:
            return 1.0
        return 0.0

    def _score_vwap_reclaim(self, df: pd.DataFrame, direction: str) -> Tuple[float, str]:
        """Score factor 4: VWAP reclaim (0 or 1)."""
        try:
            vwap_result = self._vwap_analyzer.get_vwap_levels(df)
            price_vs_vwap = vwap_result["price_vs_vwap_pct"]

            if direction == "long":
                # Long: want price reclaiming above VWAP
                if price_vs_vwap > 0.1:  # Above VWAP
                    return 1.0, "above"
                elif price_vs_vwap > -0.1:  # Near VWAP
                    return 0.5, "at"
                else:
                    return 0.0, "below"
            else:
                # Short: want price reclaiming below VWAP
                if price_vs_vwap < -0.1:  # Below VWAP
                    return 1.0, "below"
                elif price_vs_vwap < 0.1:  # Near VWAP
                    return 0.5, "at"
                else:
                    return 0.0, "above"
        except Exception:
            return 0.0, "error"

    def _score_sma_alignment(
        self, htf_bias: TimeframeBias, ltf_bias: Optional[TimeframeBias]
    ) -> float:
        """Score factor 5: SMA 50/200 alignment (0 or 1)."""
        if not ltf_bias:
            return 0.0

        # HTF alignment
        htf_aligned = (
            htf_bias.sma_50_above_200
            if htf_bias.direction == TrendDirection.UP
            else not htf_bias.sma_50_above_200
        )

        # LTF alignment
        ltf_aligned = (
            ltf_bias.sma_50_above_200
            if ltf_bias.direction == TrendDirection.UP
            else not ltf_bias.sma_50_above_200
        )

        if htf_aligned and ltf_aligned:
            return 1.0
        elif htf_aligned or ltf_aligned:
            return 0.5
        return 0.0

    def _get_confluence_level(self, score: float) -> ConfluenceLevel:
        """Get confluence level from score."""
        if score >= 4.5:
            return ConfluenceLevel.HIGH
        elif score >= 3.0:
            return ConfluenceLevel.MEDIUM
        else:
            return ConfluenceLevel.LOW

    def _calculate_stop_loss(self, df: pd.DataFrame, direction: str) -> float:
        """Calculate stop loss based on recent swing."""
        lookback = 10
        start_idx = max(0, len(df) - lookback)

        if direction == "long":
            swing_low = df["Low"].iloc[start_idx:].min()
            atr = self._get_atr(df)
            return swing_low - atr * 0.5
        else:
            swing_high = df["High"].iloc[start_idx:].max()
            atr = self._get_atr(df)
            return swing_high + atr * 0.5

    def _calculate_take_profit(self, entry: float, stop: float, direction: str) -> float:
        """Calculate take profit at 2:1 reward-to-risk."""
        risk = abs(entry - stop)
        if direction == "long":
            return entry + risk * 2.0
        else:
            return entry - risk * 2.0

    def _get_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Get current ATR."""
        from ..indicators.technical import atr

        atr_val = atr(df, period=period).iloc[-1] if len(df) > period else df["Close"].std()
        return float(atr_val) if not pd.isna(atr_val) else 1.0

    def _validate_dataframes(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> None:
        """Validate both DataFrames."""
        required_cols = ["Open", "High", "Low", "Close", "Volume"]

        htf_missing = [c for c in required_cols if c not in htf_df.columns]
        ltf_missing = [c for c in required_cols if c not in ltf_df.columns]

        if htf_missing:
            raise ValueError(f"HTF DataFrame missing: {htf_missing}")
        if ltf_missing:
            raise ValueError(f"LTF DataFrame missing: {ltf_missing}")

    def add_pattern(self, pattern: BasePattern) -> None:
        """Add SMC pattern for additional confluence."""
        if pattern not in self._patterns:
            self._patterns.append(pattern)

    def get_summary(self, signal: ConfluenceSignal) -> Dict[str, Any]:
        """
        Get human-readable summary of a signal.

        Args:
            signal: ConfluenceSignal object

        Returns:
            Dictionary with summary
        """
        return {
            "timestamp": str(signal.timestamp),
            "direction": signal.direction.upper(),
            "entry": f"${signal.entry_price:.2f}",
            "stop_loss": f"${signal.stop_loss:.2f}",
            "take_profit": f"${signal.take_profit:.2f}",
            "confluence_score": f"{signal.confluence_score:.1f}/5.0",
            "level": signal.confluence_level.value,
            "factors": {
                "4H alignment": "✓" if signal.breakdown.get("htf_alignment", 0) > 0 else "✗",
                "Fib pullback": "✓" if signal.breakdown.get("fib_pullback", 0) > 0 else "✗",
                "Volume spike": "✓" if signal.breakdown.get("volume_confirm", 0) > 0 else "✗",
                "VWAP reclaim": "✓" if signal.breakdown.get("vwap_reclaim", 0) > 0 else "✗",
                "SMA alignment": "✓" if signal.breakdown.get("sma_alignment", 0) > 0 else "✗",
            },
            "fib_level": f"{signal.fib_level:.1%}" if signal.fib_level > 0 else "N/A",
            "volume_confirmed": "Yes" if signal.volume_confirmed else "No",
            "vwap_status": signal.vwap_status,
        }
