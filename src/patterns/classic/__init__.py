"""
Classic Chart Patterns Module

Contains classic chart patterns:
- Double Top
- Double Bottom
- Trader Vic's 2B
- Triple Top
- Triple Bottom
- Ascending Triangle
- Descending Triangle
- Rectangle (Horizontal Channel)
- Wedge (Rising/Falling)
- Dead Cat Bounce
"""

from .double_top import DoubleTop
from .double_bottom import DoubleBottom
from .trader_vic_2b import TraderVic2B
from .triple_top import TripleTop
from .triple_bottom import TripleBottom
from .ascending_triangle import AscendingTriangle
from .descending_triangle import DescendingTriangle
from .rectangle import Rectangle
from .wedge import Wedge
from .dead_cat_bounce import DeadCatBounce

__all__ = [
    'DoubleTop',
    'DoubleBottom',
    'TraderVic2B',
    'TripleTop',
    'TripleBottom',
    'AscendingTriangle',
    'DescendingTriangle',
    'Rectangle',
    'Wedge',
    'DeadCatBounce',
]
