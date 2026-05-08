"""
Continuation Patterns Module

Contains continuation chart patterns:
- Flag (Bullish/Bearish)
- Pennant
"""

from .flag import Flag
from .pennant import Pennant

__all__ = [
    "Flag",
    "Pennant",
]
