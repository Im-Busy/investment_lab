"""
Trading Pattern Detection System

This module provides pattern detection for 54 chart patterns across 10 categories:
- Basic Patterns (6): MSL, Matching Lows, NR7ID, N-Bar Decline, Floor Pivot, 2-Bar Reversal
- Candlestick Patterns (6): Doji, Harami, Hammer, Engulfing, Dark Cloud Cover, Piercing Line
- Classic Patterns (10): Double Top/Bottom, Triple Top/Bottom, Ascending/Descending Triangle, Rectangle, Wedge, Dead Cat Bounce, Trader Vic 2B
- Complex Patterns (6): Cup & Handle, Head & Shoulders, Spike & Ledge, Three Hills, Parabolic Arc, Pipe
- Harmonic Patterns (9): Gartley, ABC, Symmetric Triangle, Bollinger Bands, Butterfly, Bat, Crab, Cypher, Shark
- Breakout Patterns (2): Donchian Channel, Gap Detection
- Continuation Patterns (2): Flag, Pennant
- FMZ Patterns (6): AlphaBeast, MultiFactorTrend, MomentumZigZag, EMAMACDHF, AdaptiveBollinger, AIVolatilityBreakout
- Technical Patterns (4): Ichimoku Cloud, Keltner Channel, Williams %R, CCI
- Range Persistence (3): PersistentRange, ContractingRange, ExpandingRange
"""

from .base import BasePattern, PatternType, SignalDirection, TradeSignal, PatternResult
from .similarity_search import (
    SimilarPattern,
    SearchResult,
    format_result_table,
    search_rolling,
    search_similar_patterns,
)
from .patternity_wrapper import PatternityMatch, PatternityWrapper

__all__ = [
    "BasePattern",
    "PatternType",
    "SignalDirection",
    "TradeSignal",
    "PatternResult",
    "SimilarPattern",
    "SearchResult",
    "format_result_table",
    "search_rolling",
    "search_similar_patterns",
    "PatternityMatch",
    "PatternityWrapper",
]
