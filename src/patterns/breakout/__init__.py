"""
Breakout Patterns Module

Contains breakout chart patterns:
- Gap Pattern (Explosion Gap Pivot)
- Donchian Channel Breakout (Turtle Trading System)
"""

from .donchian import DonchianChannelBreakout, DonchianTrendFilter
from .gap import GapPattern

__all__ = [
    "GapPattern",
    "DonchianChannelBreakout",
    "DonchianTrendFilter",
]
