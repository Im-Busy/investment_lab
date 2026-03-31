"""
Candlestick Pattern Module

Contains single-candle and multi-candle reversal patterns.
All patterns require confirmation for signal generation.

Patterns:
- Doji: Indecision pattern (open ≈ close)
- Harami: Two-candle reversal pattern (small body inside large body)
- Hammer: Bullish reversal after downtrend (small body, long lower shadow)
- Engulfing: Two-candle reversal pattern (body engulfs prior)
- DarkCloudCover: Bearish reversal (pierces prior bullish body)
- PiercingLine: Bullish reversal (pierces prior bearish body)
"""

from .doji import Doji
from .harami import Harami
from .hammer import Hammer
from .engulfing import Engulfing
from .dark_cloud import DarkCloudCover, PiercingLine

__all__ = [
    "Doji",
    "Harami",
    "Hammer",
    "Engulfing",
    "DarkCloudCover",
    "PiercingLine",
]
