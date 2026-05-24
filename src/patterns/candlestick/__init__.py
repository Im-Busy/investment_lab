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
- ICT Single Patterns: 12 ICT-named candlestick patterns (WM, CWM, OWM, WDD, WPU, BPU, BM, CBM, OBM, BGD, WSS, BSS)
"""

from .doji import Doji
from .harami import Harami
from .hammer import Hammer
from .engulfing import Engulfing
from .dark_cloud import DarkCloudCover, PiercingLine
from .ict_single_patterns import detect_all_twelve

__all__ = [
    "Doji",
    "Harami",
    "Hammer",
    "Engulfing",
    "DarkCloudCover",
    "PiercingLine",
    "detect_all_twelve",
]
