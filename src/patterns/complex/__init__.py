"""
Complex Patterns Module

Contains complex chart patterns:
- Cup and Handle
- Head and Shoulders
- Spike and Ledge
- Three Hills and Mountain
- Parabolic Arc
"""

from .cup_handle import CupAndHandle
from .head_shoulders import HeadAndShoulders
from .spike_ledge import SpikeAndLedge
from .three_hills import ThreeHillsMountain
from .parabolic_arc import ParabolicArc
from .pipe import PipePattern

__all__ = [
    "CupAndHandle",
    "HeadAndShoulders",
    "SpikeAndLedge",
    "ThreeHillsMountain",
    "ParabolicArc",
    "PipePattern",
]
