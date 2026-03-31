"""
Trading Pattern Detection System

This module provides pattern detection for 20 chart patterns across 4 categories:
- Basic Patterns: MSL, Matching Lows, NR7ID, n-Bar Decline, Floor Pivot
- Harmonic Patterns: Gartley, ABC, Symmetric Triangle, Donchian, Bollinger
- Complex Patterns: Cup and Handle, Head and Shoulders, Spike and Ledge, Three Hills, Parabolic Arc
- Classic Patterns: Double Top/Bottom, 2B, Triple Top, Dead Cat Bounce
"""

from .base import BasePattern, PatternType, SignalDirection, TradeSignal, PatternResult

__all__ = [
    'BasePattern',
    'PatternType',
    'SignalDirection',
    'TradeSignal',
    'PatternResult',
]
