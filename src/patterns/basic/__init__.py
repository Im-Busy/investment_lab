"""
Basic Patterns Module

Contains fundamental chart patterns:
- Market Structure Low (MSL)
- Matching Lows
- NR7ID (Narrow Range 7 + Inside Day)
- n-Bar Decline
- Floor Pivot Breakout
- Two-Bar Reversal (Pipe Bottom / Pipe Top)
"""

from .msl import MarketStructureLow
from .matching_lows import MatchingLows
from .nr7id import NR7ID
from .n_bar_decline import NBarDecline
from .floor_pivot import FloorPivotBreakout
from .two_bar_reversal import TwoBarReversal

__all__ = [
    'MarketStructureLow',
    'MatchingLows',
    'NR7ID',
    'NBarDecline',
    'FloorPivotBreakout',
    'TwoBarReversal',
]
