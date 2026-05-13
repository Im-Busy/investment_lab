"""
Signals Module

Contains signal generation and position management components.
"""

from .signal_generator import SignalGenerator
from .position_manager import PositionManager
from .event_weighting import (
    EventWeightedAggregator,
    EventWeightedSignal,
    EventType,
    EventTypeInfo,
    PatternEventMapping,
    EVENT_TYPE_WEIGHTS,
    DEFAULT_PATTERN_EVENT_MAPPING,
)
from .pattern_boost import PatternBoostFilter, PatternBoost, PATTERN_RELIABILITY

__all__ = [
    "SignalGenerator",
    "PositionManager",
    "EventWeightedAggregator",
    "EventWeightedSignal",
    "EventType",
    "EventTypeInfo",
    "PatternEventMapping",
    "EVENT_TYPE_WEIGHTS",
    "DEFAULT_PATTERN_EVENT_MAPPING",
    "PatternBoostFilter",
    "PatternBoost",
    "PATTERN_RELIABILITY",
]
