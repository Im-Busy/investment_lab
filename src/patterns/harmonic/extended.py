"""
Extended Harmonic Patterns: Butterfly, Bat, Crab, Cypher, Shark.

All five extend from GartleyPattern with different Fibonacci ratio constants.
Each is a 5-point XABCD reversal pattern validated by Fibonacci retracements.

C14: Missing harmonic detectors from Insight #2 (Duddella harmonic guide).
"""

from __future__ import annotations

import numpy as np

from .gartley import GartleyPattern
from ..base import PatternType


class ButterflyPattern(GartleyPattern):
    """Butterfly harmonic pattern.

    Key difference from Gartley: D point extends well beyond X (1.272 XA).
    B retraces 0.786 of XA (deeper pullback), CD extends 1.618-2.618 of BC.
    """

    AB_RETRACEMENT_TARGET = 0.786
    AB_TOLERANCE = 0.05

    BC_RETRACEMENT_MIN = 0.382
    BC_RETRACEMENT_MAX = 0.886

    CD_EXTENSION_MIN = 1.618
    CD_EXTENSION_MAX = 2.618

    XD_RETRACEMENT_TARGET = 1.272
    XD_TOLERANCE = 0.05

    def __init__(
        self,
        lookback: int = 5,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        fib_tolerance: float = 0.05,
        require_confirmation: bool = True,
        volume_filter: bool = True,
    ):
        super().__init__(
            lookback=lookback,
            entry_offset=entry_offset,
            stop_offset=stop_offset,
            fib_tolerance=fib_tolerance,
            require_confirmation=require_confirmation,
            volume_filter=volume_filter,
        )
        self.name = "Butterfly Pattern"


class BatPattern(GartleyPattern):
    """Bat harmonic pattern.

    B retraces 0.382-0.50 XA (shallow pullback), D lands at 0.886 XA.
    CD extends 1.618-2.618 of BC. Most precise harmonic pattern.
    """

    AB_RETRACEMENT_TARGET = 0.50
    AB_TOLERANCE = 0.05

    BC_RETRACEMENT_MIN = 0.382
    BC_RETRACEMENT_MAX = 0.886

    CD_EXTENSION_MIN = 1.618
    CD_EXTENSION_MAX = 2.618

    XD_RETRACEMENT_TARGET = 0.886
    XD_TOLERANCE = 0.05

    def __init__(
        self,
        lookback: int = 5,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        fib_tolerance: float = 0.05,
        require_confirmation: bool = True,
        volume_filter: bool = True,
    ):
        super().__init__(
            lookback=lookback,
            entry_offset=entry_offset,
            stop_offset=stop_offset,
            fib_tolerance=fib_tolerance,
            require_confirmation=require_confirmation,
            volume_filter=volume_filter,
        )
        self.name = "Bat Pattern"


class CrabPattern(GartleyPattern):
    """Crab harmonic pattern.

    Extreme extension pattern. D reaches 1.618 XA. CD extends 2.24-3.618 BC.
    Highest reward/risk of all harmonic patterns.
    """

    AB_RETRACEMENT_TARGET = 0.618
    AB_TOLERANCE = 0.05

    BC_RETRACEMENT_MIN = 0.382
    BC_RETRACEMENT_MAX = 0.886

    CD_EXTENSION_MIN = 2.24
    CD_EXTENSION_MAX = 3.618

    XD_RETRACEMENT_TARGET = 1.618
    XD_TOLERANCE = 0.05

    def __init__(
        self,
        lookback: int = 5,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        fib_tolerance: float = 0.05,
        require_confirmation: bool = True,
        volume_filter: bool = True,
    ):
        super().__init__(
            lookback=lookback,
            entry_offset=entry_offset,
            stop_offset=stop_offset,
            fib_tolerance=fib_tolerance,
            require_confirmation=require_confirmation,
            volume_filter=volume_filter,
        )
        self.name = "Crab Pattern"


class CypherPattern(GartleyPattern):
    """Cypher harmonic pattern.

    Unique: C point extends BEYOND X (1.272-1.414 XA). D retraces to 0.786 of XC.
    Highest win rate claim among harmonic patterns.
    """

    AB_RETRACEMENT_TARGET = 0.618
    AB_TOLERANCE = 0.05

    BC_RETRACEMENT_MIN = 1.272
    BC_RETRACEMENT_MAX = 1.414

    CD_EXTENSION_MIN = 0.786
    CD_EXTENSION_MAX = 0.786

    XD_RETRACEMENT_TARGET = 0.786
    XD_TOLERANCE = 0.05

    def __init__(
        self,
        lookback: int = 5,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        fib_tolerance: float = 0.05,
        require_confirmation: bool = True,
        volume_filter: bool = True,
    ):
        super().__init__(
            lookback=lookback,
            entry_offset=entry_offset,
            stop_offset=stop_offset,
            fib_tolerance=fib_tolerance,
            require_confirmation=require_confirmation,
            volume_filter=volume_filter,
        )
        self.name = "Cypher Pattern"


class SharkPattern(GartleyPattern):
    """Shark harmonic pattern.

    5-point O-X-A-B-C pattern. Entry at C (before D forms).
    O to C forms the impulse. B retraces 0.886-1.13 of XA.
    C extends 1.13-1.618 of OA.
    """

    AB_RETRACEMENT_TARGET = 1.13
    AB_TOLERANCE = 0.05

    BC_RETRACEMENT_MIN = 1.618
    BC_RETRACEMENT_MAX = 2.618

    CD_EXTENSION_MIN = 0.886
    CD_EXTENSION_MAX = 1.13

    XD_RETRACEMENT_TARGET = 0.886
    XD_TOLERANCE = 0.05

    def __init__(
        self,
        lookback: int = 5,
        entry_offset: float = 0.01,
        stop_offset: float = 0.01,
        fib_tolerance: float = 0.05,
        require_confirmation: bool = True,
        volume_filter: bool = True,
    ):
        super().__init__(
            lookback=lookback,
            entry_offset=entry_offset,
            stop_offset=stop_offset,
            fib_tolerance=fib_tolerance,
            require_confirmation=require_confirmation,
            volume_filter=volume_filter,
        )
        self.name = "Shark Pattern"
