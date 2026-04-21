# -*- coding: utf-8 -*-
"""
Event-Type Weighted Signal Aggregation (R6)

This module implements event-type weighted signal aggregation based on research findings
from "Event-Based Trading: Building Superior Trading Strategies with IE Tools" (SSRN-2907600).

Key Insights from Research:
- Granular event-type signals beat aggregated scores
- Different events have different informative values for different holding periods
- "Breakout confirmed by volume spike" should weight higher than "breakout alone"
- Event-type specificity improves signal quality vs. aggregated sentiment/confluence

Architecture:
1. Event Taxonomy: Maps patterns to granular event types
2. Event Informativeness: Weights events by historical predictive power
3. Holding Period Alignment: Matches event type to optimal holding horizon
4. Event-Weighted Aggregation: Replaces equal-weight confluence scoring
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..patterns.base import PatternType, TradeSignal


class EventType(Enum):
    """
    Event Type Taxonomy (R6 - Event-Type Weighted Signal Aggregation)
    
    Based on research from SSRN-2907600: granular event types provide superior
    signal quality compared to aggregated confluence scores.
    
    Categories aligned with pattern types:
    - TREND_INITIATION: New trend starting (breakouts, reversals)
    - BREAKOUT: Price breaking key levels
    - REVERSAL: Trend exhaustion and reversal
    - CONTINUATION: Trend continuation patterns
    - MEAN_REVERSION: Counter-trend mean reversion
    - MOMENTUM: Momentum confirmation/acceleration
    - VOLATILITY: Volatility expansion/contraction events
    - STRUCTURAL: Multi-bar structural patterns
    """

    TREND_INITIATION = "trend_initiation"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    STRUCTURAL = "structural"


@dataclass
class EventTypeInfo:
    """
    Event Type Information and Weighting
    
    Attributes:
        event_type: The event type
        base_weight: Base informativeness weight (0.0 to 1.0)
        optimal_holding_period: Optimal holding period in bars
        holding_period_unit: Unit for holding period ('bars', 'days', 'weeks')
        confidence_boost: Confidence boost when multiple patterns confirm same event
        decay_rate: Signal decay rate per bar (for time-weighting)
        volume_confirmation_bonus: Extra weight when volume confirms
        regime_compatibility: Regimes where this event type performs well
    """

    event_type: EventType
    base_weight: float = 0.5
    optimal_holding_period: int = 10
    holding_period_unit: str = "bars"
    confidence_boost: float = 0.15
    decay_rate: float = 0.02
    volume_confirmation_bonus: float = 0.05
    regime_compatibility: List[str] = field(default_factory=list)


EVENT_TYPE_WEIGHTS: Dict[EventType, EventTypeInfo] = {
    EventType.TREND_INITIATION: EventTypeInfo(
        event_type=EventType.TREND_INITIATION,
        base_weight=0.75,
        optimal_holding_period=20,
        holding_period_unit="bars",
        confidence_boost=0.20,
        decay_rate=0.015,
        volume_confirmation_bonus=0.08,
        regime_compatibility=["trending", "transition"],
    ),
    EventType.BREAKOUT: EventTypeInfo(
        event_type=EventType.BREAKOUT,
        base_weight=0.70,
        optimal_holding_period=15,
        holding_period_unit="bars",
        confidence_boost=0.18,
        decay_rate=0.02,
        volume_confirmation_bonus=0.10,
        regime_compatibility=["trending", "volatile"],
    ),
    EventType.REVERSAL: EventTypeInfo(
        event_type=EventType.REVERSAL,
        base_weight=0.65,
        optimal_holding_period=12,
        holding_period_unit="bars",
        confidence_boost=0.15,
        decay_rate=0.025,
        volume_confirmation_bonus=0.06,
        regime_compatibility=["ranging", "transition"],
    ),
    EventType.CONTINUATION: EventTypeInfo(
        event_type=EventType.CONTINUATION,
        base_weight=0.60,
        optimal_holding_period=25,
        holding_period_unit="bars",
        confidence_boost=0.12,
        decay_rate=0.01,
        volume_confirmation_bonus=0.04,
        regime_compatibility=["trending"],
    ),
    EventType.MEAN_REVERSION: EventTypeInfo(
        event_type=EventType.MEAN_REVERSION,
        base_weight=0.55,
        optimal_holding_period=8,
        holding_period_unit="bars",
        confidence_boost=0.10,
        decay_rate=0.03,
        volume_confirmation_bonus=0.03,
        regime_compatibility=["ranging"],
    ),
    EventType.MOMENTUM: EventTypeInfo(
        event_type=EventType.MOMENTUM,
        base_weight=0.65,
        optimal_holding_period=10,
        holding_period_unit="bars",
        confidence_boost=0.15,
        decay_rate=0.025,
        volume_confirmation_bonus=0.07,
        regime_compatibility=["trending", "volatile"],
    ),
    EventType.VOLATILITY: EventTypeInfo(
        event_type=EventType.VOLATILITY,
        base_weight=0.50,
        optimal_holding_period=5,
        holding_period_unit="bars",
        confidence_boost=0.08,
        decay_rate=0.04,
        volume_confirmation_bonus=0.05,
        regime_compatibility=["volatile", "transition"],
    ),
    EventType.STRUCTURAL: EventTypeInfo(
        event_type=EventType.STRUCTURAL,
        base_weight=0.70,
        optimal_holding_period=30,
        holding_period_unit="bars",
        confidence_boost=0.18,
        decay_rate=0.01,
        volume_confirmation_bonus=0.05,
        regime_compatibility=["trending", "ranging"],
    ),
}


@dataclass
class PatternEventMapping:
    """
    Maps pattern names to event types with optional quality modifiers.
    
    Attributes:
        pattern_name: Pattern detector name
        event_type: Mapped event type
        weight_modifier: Optional weight adjustment for this specific pattern
        requires_volume_confirmation: Whether volume confirmation is required
        holding_period_override: Optional override for optimal holding period
    """

    pattern_name: str
    event_type: EventType
    weight_modifier: float = 0.0
    requires_volume_confirmation: bool = False
    holding_period_override: Optional[int] = None


DEFAULT_PATTERN_EVENT_MAPPING: List[PatternEventMapping] = [
    # Breakout patterns
    PatternEventMapping("Donchian Channel Breakout", EventType.BREAKOUT, weight_modifier=0.10, requires_volume_confirmation=True),
    PatternEventMapping("Gap", EventType.BREAKOUT, weight_modifier=0.05),
    
    # Reversal patterns
    PatternEventMapping("Double Bottom", EventType.REVERSAL, weight_modifier=0.08),
    PatternEventMapping("Double Top", EventType.REVERSAL, weight_modifier=0.08),
    PatternEventMapping("Triple Bottom", EventType.REVERSAL, weight_modifier=0.10),
    PatternEventMapping("Triple Top", EventType.REVERSAL, weight_modifier=0.10),
    PatternEventMapping("Head and Shoulders", EventType.REVERSAL, weight_modifier=0.12),
    PatternEventMapping("Inverse Head and Shoulders", EventType.REVERSAL, weight_modifier=0.12),
    
    # Continuation patterns
    PatternEventMapping("Flag", EventType.CONTINUATION),
    PatternEventMapping("Pennant", EventType.CONTINUATION),
    PatternEventMapping("Rectangle", EventType.CONTINUATION, holding_period_override=20),
    
    # Classic patterns
    PatternEventMapping("Ascending Triangle", EventType.BREAKOUT, weight_modifier=0.05),
    PatternEventMapping("Descending Triangle", EventType.BREAKOUT, weight_modifier=0.05),
    PatternEventMapping("Wedge", EventType.REVERSAL),
    PatternEventMapping("Dead Cat Bounce", EventType.REVERSAL, weight_modifier=-0.05),
    
    # Candlestick patterns
    PatternEventMapping("Engulfing", EventType.REVERSAL, weight_modifier=-0.10),
    PatternEventMapping("Harami", EventType.REVERSAL, weight_modifier=-0.15),
    PatternEventMapping("Hammer", EventType.REVERSAL, weight_modifier=-0.08),
    PatternEventMapping("Doji", EventType.REVERSAL, weight_modifier=-0.20),
    PatternEventMapping("Dark Cloud Cover", EventType.REVERSAL, weight_modifier=-0.10),
    
    # Harmonic patterns
    PatternEventMapping("Gartley", EventType.STRUCTURAL, weight_modifier=0.15),
    PatternEventMapping("ABC", EventType.STRUCTURAL, weight_modifier=0.10),
    PatternEventMapping("Symmetric Triangle", EventType.STRUCTURAL),
    PatternEventMapping("Bollinger Band Patterns", EventType.VOLATILITY),
    
    # Basic patterns
    PatternEventMapping("Market Structure Low", EventType.REVERSAL, weight_modifier=-0.05),
    PatternEventMapping("Market Structure High", EventType.REVERSAL, weight_modifier=-0.05),
    PatternEventMapping("Matching Lows", EventType.MEAN_REVERSION),
    PatternEventMapping("Floor Pivot", EventType.STRUCTURAL),
    PatternEventMapping("Two Bar Reversal", EventType.REVERSAL, weight_modifier=-0.15),
    PatternEventMapping("N Bar Decline", EventType.MOMENTUM),
    PatternEventMapping("NR7ID", EventType.VOLATILITY),
    
    # Complex patterns
    PatternEventMapping("Cup and Handle", EventType.STRUCTURAL, weight_modifier=0.12),
    PatternEventMapping("Three Hills", EventType.CONTINUATION, weight_modifier=0.08),
    PatternEventMapping("Parabolic Arc", EventType.MOMENTUM, weight_modifier=0.10),
    PatternEventMapping("Spike and Ledge", EventType.BREAKOUT, weight_modifier=0.08),
    
    # Trend filter (not a trading pattern, but provides context)
    PatternEventMapping("Donchian Channel Trend", EventType.CONTINUATION, weight_modifier=-0.20),
]


@dataclass
class EventWeightedSignal:
    """
    Event-weighted trading signal.
    
    Extends TradeSignal with event-type metadata and weighting information.
    
    Attributes:
        pattern_name: Original pattern name
        direction: Long or Short
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit_1: First take profit target
        take_profit_2: Second take profit target (optional)
        take_profit_3: Third take profit target (optional)
        confidence: Base confidence score
        event_type: Mapped event type
        event_weight: Calculated event weight
        holding_period: Recommended holding period
        volume_confirmed: Whether volume confirms the signal
        metadata: Additional signal information
    """

    pattern_name: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]
    take_profit_3: Optional[float]
    confidence: float
    event_type: EventType
    event_weight: float
    holding_period: int
    volume_confirmed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_trade_signal(
        cls,
        signal: TradeSignal,
        event_type: EventType,
        event_weight: float,
        holding_period: int,
        volume_confirmed: bool = False,
        **kwargs,
    ) -> "EventWeightedSignal":
        """Create EventWeightedSignal from TradeSignal."""
        return cls(
            pattern_name=signal.pattern_name,
            direction=signal.direction.value,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            take_profit_3=signal.take_profit_3,
            confidence=signal.confidence,
            event_type=event_type,
            event_weight=event_weight,
            holding_period=holding_period,
            volume_confirmed=volume_confirmed,
            metadata={**signal.metadata, **kwargs},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_name": self.pattern_name,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "take_profit_3": self.take_profit_3,
            "confidence": self.confidence,
            "event_type": self.event_type.value,
            "event_weight": self.event_weight,
            "holding_period": self.holding_period,
            "volume_confirmed": self.volume_confirmed,
            "metadata": self.metadata,
        }


class EventWeightedAggregator:
    """
    Event-Type Weighted Signal Aggregator (R6 Implementation)
    
    Replaces equal-weight confluence scoring with event-type weighted aggregation.
    
    Key Features:
    1. Maps patterns to granular event types
    2. Weights signals by event-type informativeness
    3. Applies holding-period alignment
    4. Calculates event-weighted confluence scores
    
    Usage:
        aggregator = EventWeightedAggregator()
        signals = [trade_signal1, trade_signal2, ...]
        aggregated = aggregator.aggregate(signals, df, i)
    """

    def __init__(
        self,
        pattern_mappings: Optional[List[PatternEventMapping]] = None,
        use_volume_confirmation: bool = True,
        use_holding_period_alignment: bool = True,
    ):
        """
        Initialize Event-Weighted Aggregator.
        
        Args:
            pattern_mappings: Custom pattern-to-event mappings (uses defaults if None)
            use_volume_confirmation: Apply volume confirmation bonus
            use_holding_period_alignment: Align signals by holding period
        """
        self.pattern_mappings = pattern_mappings or DEFAULT_PATTERN_EVENT_MAPPING
        self.use_volume_confirmation = use_volume_confirmation
        self.use_holding_period_alignment = use_holding_period_alignment
        
        # Build lookup map for fast pattern to event type resolution
        self._pattern_to_event: Dict[str, PatternEventMapping] = {
            m.pattern_name.lower(): m for m in self.pattern_mappings
        }
    
    def get_event_type_for_pattern(self, pattern_name: str) -> Tuple[EventType, float]:
        """
        Get event type and weight modifier for a pattern.
        
        Args:
            pattern_name: Pattern detector name
            
        Returns:
            Tuple of (EventType, weight_modifier)
        """
        pattern_key = pattern_name.lower()
        
        if pattern_key in self._pattern_to_event:
            mapping = self._pattern_to_event[pattern_key]
            return mapping.event_type, mapping.weight_modifier
        
        # Fallback: infer from pattern category keywords
        if any(kw in pattern_key for kw in ["breakout", "gap", "donchian"]):
            return EventType.BREAKOUT, 0.0
        elif any(kw in pattern_key for kw in ["reversal", "top", "bottom", "head"]):
            return EventType.REVERSAL, 0.0
        elif any(kw in pattern_key for kw in ["flag", "pennant", "continuation"]):
            return EventType.CONTINUATION, 0.0
        elif any(kw in pattern_key for kw in ["triangle", "wedge", "rectangle"]):
            return EventType.STRUCTURAL, 0.0
        elif any(kw in pattern_key for kw in ["engulfing", "hammer", "doji", "harami"]):
            return EventType.REVERSAL, -0.05
        else:
            return EventType.REVERSAL, 0.0

    def calculate_event_weight(
        self,
        event_type: EventType,
        pattern_name: str,
        volume_confirmed: bool = False,
        regime: Optional[str] = None,
    ) -> float:
        """
        Calculate event weight for a signal.
        
        Args:
            event_type: Event type
            pattern_name: Pattern name
            volume_confirmed: Whether volume confirms the signal
            regime: Current market regime
            
        Returns:
            Event weight (0.0 to 1.0)
        """
        event_info = EVENT_TYPE_WEIGHTS[event_type]
        
        # Base weight from event type
        weight = event_info.base_weight
        
        # Apply pattern-specific modifier
        _, pattern_modifier = self.get_event_type_for_pattern(pattern_name)
        weight += pattern_modifier
        
        # Volume confirmation bonus
        if self.use_volume_confirmation and volume_confirmed:
            weight += event_info.volume_confirmation_bonus
        
        # Regime compatibility bonus
        if regime and regime in event_info.regime_compatibility:
            weight += 0.05
        
        # Clamp to valid range
        return np.clip(weight, 0.0, 1.0)

    def calculate_confluence_boost(
        self,
        signals: List[EventWeightedSignal],
    ) -> float:
        """
        Calculate confluence boost from multiple signals confirming same event.
        
        Research insight: Multiple patterns confirming the same event type
        should receive a confidence boost, but with diminishing returns.
        
        Args:
            signals: List of event-weighted signals
            
        Returns:
            Confluence boost factor (0.0 to 0.3)
        """
        if len(signals) < 2:
            return 0.0
        
        # Count signals per event type
        event_counts: Dict[EventType, int] = {}
        total_weight = 0.0
        
        for signal in signals:
            event_type = signal.event_type
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            total_weight += signal.event_weight
        
        # Average event weight
        avg_weight = total_weight / len(signals)
        
        # Confluence boost: more signals of same type = higher boost
        # But with diminishing returns
        max_count = max(event_counts.values()) if event_counts else 0
        
        if max_count >= 3:
            boost = 0.20  # Strong confluence
        elif max_count >= 2:
            boost = 0.12  # Moderate confluence
        else:
            boost = 0.0
        
        # Weight by average event quality
        boost *= avg_weight
        
        return min(boost, 0.30)

    def calculate_time_weighted_confidence(
        self,
        signal: EventWeightedSignal,
        bars_held: int = 0,
    ) -> float:
        """
        Calculate time-weighted confidence based on signal decay.
        
        Research insight: Signals decay over time. Event types have different
        decay rates based on optimal holding periods.
        
        Args:
            signal: Event-weighted signal
            bars_held: Number of bars since signal generation
            
        Returns:
            Time-weighted confidence (0.0 to 1.0)
        """
        event_info = EVENT_TYPE_WEIGHTS[signal.event_type]
        
        # Exponential decay based on event type's decay rate
        decay_factor = np.exp(-event_info.decay_rate * bars_held)
        
        # Adjust for holding period alignment
        if self.use_holding_period_alignment:
            optimal_period = signal.holding_period
            if bars_held > optimal_period:
                # Signal is past optimal holding period - extra penalty
                extra_decay = 0.05 * (bars_held - optimal_period) / optimal_period
                decay_factor *= np.exp(-extra_decay)
        
        return signal.confidence * decay_factor

    def aggregate(
        self,
        signals: List[TradeSignal],
        df: pd.DataFrame,
        i: int,
        volume_data: Optional[np.ndarray] = None,
        regime: Optional[str] = None,
    ) -> EventWeightedSignal:
        """
        Aggregate multiple trade signals using event-type weighting.
        
        Args:
            signals: List of TradeSignal objects
            df: DataFrame with OHLCV data
            i: Current bar index
            volume_data: Pre-extracted volume array (optional)
            regime: Current market regime (optional)
            
        Returns:
            EventWeightedSignal with aggregated metrics
        """
        if not signals:
            raise ValueError("No signals to aggregate")
        
        # Convert signals to event-weighted format
        event_signals: List[EventWeightedSignal] = []
        
        for signal in signals:
            # Get event type for this pattern
            event_type, _ = self.get_event_type_for_pattern(signal.pattern_name)
            
            # Check volume confirmation
            volume_confirmed = False
            if self.use_volume_confirmation and volume_data is not None and i > 20:
                current_vol = volume_data[i]
                avg_vol = np.mean(volume_data[i - 20 : i])
                if avg_vol > 0:
                    volume_confirmed = current_vol > avg_vol * 1.5
            
            # Calculate event weight
            event_weight = self.calculate_event_weight(
                event_type, signal.pattern_name, volume_confirmed, regime
            )
            
            # Get holding period from event type info
            event_info = EVENT_TYPE_WEIGHTS[event_type]
            holding_period = event_info.optimal_holding_period
            
            event_signal = EventWeightedSignal.from_trade_signal(
                signal,
                event_type,
                event_weight,
                holding_period,
                volume_confirmed,
            )
            event_signals.append(event_signal)
        
        # Calculate aggregated metrics
        total_weight = sum(s.event_weight for s in event_signals)
        
        if total_weight == 0:
            total_weight = 1.0
        
         # Weighted average entry price
        entry_price = sum(s.entry_price * s.event_weight for s in event_signals) / total_weight
        
        # Use most conservative stop loss (closest to entry for longs, furthest for shorts)
        if event_signals[0].direction == "Long":
            stop_loss = min(s.stop_loss for s in event_signals)
        else:
            stop_loss = max(s.stop_loss for s in event_signals)
        
        # Weighted take profit targets
        tp1 = sum(s.take_profit_1 * s.event_weight for s in event_signals) / total_weight
        tp2_values = [s.take_profit_2 for s in event_signals if s.take_profit_2 is not None]
        tp3_values = [s.take_profit_3 for s in event_signals if s.take_profit_3 is not None]
        take_profit_2 = (
            sum(s.take_profit_2 * s.event_weight for s in event_signals if s.take_profit_2)
            / total_weight
            if tp2_values
            else None
        )
        take_profit_3 = (
            sum(s.take_profit_3 * s.event_weight for s in event_signals if s.take_profit_3)
            / total_weight
            if tp3_values
            else None
        )
        
        # Base confidence weighted by event weights
        base_confidence = (
            sum(s.confidence * s.event_weight for s in event_signals) / total_weight
        )
        
        # Confluence boost for multiple patterns confirming same event
        confluence_boost = self.calculate_confluence_boost(event_signals)
        
        # Time-weighted confidence (all signals assumed to be from same bar, so bars_held=0)
        time_weighted_confidence = self.calculate_time_weighted_confidence(event_signals[0], 0)
        
        # Final confidence
        final_confidence = min(base_confidence + confluence_boost, 1.0)
        final_confidence = max(final_confidence, time_weighted_confidence)
        
        # Determine dominant event type
        event_type_counts: Dict[EventType, float] = {}
        for s in event_signals:
            event_type_counts[s.event_type] = (
                event_type_counts.get(s.event_type, 0.0) + s.event_weight
            )
        dominant_event_type = max(event_type_counts, key=event_type_counts.get)
        
        # Average holding period
        avg_holding_period = int(
            sum(s.holding_period * s.event_weight for s in event_signals) / total_weight
        )
        
        # Aggregate signal
        aggregated = EventWeightedSignal(
            pattern_name="+".join(s.pattern_name for s in event_signals[:3]),
            direction=event_signals[0].direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            confidence=final_confidence,
            event_type=dominant_event_type,
            event_weight=total_weight / len(event_signals),
            holding_period=avg_holding_period,
            volume_confirmed=any(s.volume_confirmed for s in event_signals),
            metadata={
                "individual_signals": [s.to_dict() for s in event_signals],
                "event_type_distribution": {k.value: v for k, v in event_type_counts.items()},
                "confluence_boost": confluence_boost,
                "num_signals": len(event_signals),
            },
        )
        
        return aggregated

    def get_pattern_coverage(self) -> Dict[str, float]:
        """
        Get event type coverage statistics for all mapped patterns.
        
        Returns:
            Dictionary with coverage statistics
        """
        event_type_counts: Dict[EventType, int] = {}
        total_patterns = len(self.pattern_mappings)
        
        for mapping in self.pattern_mappings:
            event_type_counts[mapping.event_type] = (
                event_type_counts.get(mapping.event_type, 0) + 1
            )
        
        return {
            "total_patterns_mapped": total_patterns,
            "event_type_distribution": {k.value: v for k, v in event_type_counts.items()},
            "coverage_percentage": {
                k.value: v / total_patterns * 100 for k, v in event_type_counts.items()
            },
        }
