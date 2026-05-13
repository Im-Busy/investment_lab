"""
C8: Pattern Boost Filter — wires chart pattern detectors as ML signal confirmation/boost.

At each bar, runs pattern detectors and computes a probability boost (0.0–0.10)
based on pattern reliability weights, confluence count, and direction alignment.

Pattern reliability weights sourced from:
- NCFE study (H&S 86-88%, Symmetric Triangle 76-78%)
- Suri Duddella "Trade Chart Patterns Like The Pros"
- Harmonic trading guides

Usage:
    booster = PatternBoostFilter()
    booster.precompute(ohlcv_df)
    boost = booster.get_boost(bar_idx)  # returns (bull_boost, bear_boost)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

_logger = logging.getLogger(__name__)


# ── Pattern reliability weights from NCFE / Kirkpatrick / Duddella research ──
# Used to weight individual pattern signals when computing boost score.
# Weights > 0.0 = boost-compatible; weights at 0.0 = excluded from boost.
PATTERN_RELIABILITY: Dict[str, float] = {
    # Harmonic patterns — highest confidence, Fibonacci-structured
    "Gartley Pattern": 0.85,
    "ABC Pattern": 0.75,
    "Symmetric Triangle": 0.77,  # NCFE: 76-78%
    "Bollinger Bands": 0.65,
    # Classic reversal patterns
    "Head and Shoulders": 0.87,  # NCFE: 86-88%, highest reliability
    "Inverse Head and Shoulders": 0.87,
    "Double Top": 0.70,
    "Double Bottom": 0.70,
    "Triple Top": 0.75,
    "Triple Bottom": 0.75,
    "Trader Vic 2B": 0.72,
    "Ascending Triangle": 0.78,  # NCFE: 75-80%
    "Descending Triangle": 0.78,  # NCFE: 75-80%
    "Rectangle": 0.68,
    "Wedge": 0.65,
    "Dead Cat Bounce": 0.50,  # High failure rate in source
    # Complex patterns
    "Cup and Handle": 0.80,
    "Spike and Ledge": 0.72,
    "Three Hills and a Mountain": 0.78,
    "Parabolic Arc": 0.70,
    # Breakout patterns
    "Donchian Channel Breakout": 0.62,
    "Gap Pattern": 0.60,
    # Continuation patterns
    "Bull Flag": 0.72,
    "Bear Flag": 0.72,
    "Pennant": 0.68,
    "Flag": 0.68,
    # Basic patterns
    "Market Structure Low": 0.65,
    "Market Structure High": 0.65,
    "Matching Lows": 0.60,
    "NR7ID": 0.55,
    "N-Bar Decline": 0.58,
    "Floor Pivot Breakout": 0.55,
    "Two Bar Reversal": 0.60,
    # Candlestick patterns — lower standalone reliability, used as confirmation
    "Doji": 0.40,
    "Harami": 0.45,
    "Hammer": 0.45,
    "Engulfing": 0.55,
    "Dark Cloud Cover": 0.50,
    "Piercing Line": 0.50,
    # Default fallback
    "default": 0.55,
}


@dataclass
class PatternBoost:
    """Boost result for a single bar."""

    bull_boost: float = 0.0
    bear_boost: float = 0.0
    bull_patterns: List[str] = field(default_factory=list)
    bear_patterns: List[str] = field(default_factory=list)
    bullish_count: int = 0
    bearish_count: int = 0
    max_boost: float = 0.10  # Maximum boost cap


class PatternBoostFilter:
    """Runs pattern detectors each bar and computes an ML probability boost.

    The boost scales from 0.0 (no patterns) to a configurable maximum (default 0.10)
    when multiple high-reliability patterns fire in the same direction.

    Design:
        - Precomputes all pattern signals via vectorized detection at init time
        - At each bar, looks up cached signals and computes a weighted boost
        - Separate bull/bear boosts to allow directional boosting
    """

    def __init__(
        self,
        max_boost: float = 0.10,
        boost_scale: float = 0.05,
        min_patterns: int = 1,
        confluence_bonus: float = 0.02,
        volume_confirm_bonus: float = 0.01,
    ) -> None:
        """Initialize the pattern boost filter.

        Args:
            max_boost: Maximum boost value (capped). Default 0.10.
            boost_scale: Base scaling factor applied to weighted pattern score. Default 0.05.
            min_patterns: Minimum patterns required to produce any boost. Default 1.
            confluence_bonus: Additional boost per extra pattern beyond the first. Default 0.02.
            volume_confirm_bonus: Bonus if volume increases on breakout bar. Default 0.01.
        """
        self._max_boost = max_boost
        self._boost_scale = boost_scale
        self._min_patterns = min_patterns
        self._confluence_bonus = confluence_bonus
        self._volume_confirm_bonus = volume_confirm_bonus

        self._patterns: List[Any] = []
        self._signals_cache: Dict[str, np.ndarray] = {}
        self._reliability: Dict[str, float] = {}
        self._precomputed: bool = False
        self._volume: Optional[np.ndarray] = None
        self._n_bars: int = 0

    def _init_patterns(self) -> List[Any]:
        """Instantiate all pattern detectors used for boosting."""
        from src.patterns.basic.floor_pivot import FloorPivotBreakout
        from src.patterns.basic.matching_lows import MatchingLows
        from src.patterns.basic.msl import MarketStructureLow
        from src.patterns.basic.n_bar_decline import NBarDecline
        from src.patterns.basic.nr7id import NR7ID
        from src.patterns.basic.two_bar_reversal import TwoBarReversal
        from src.patterns.breakout.donchian import DonchianChannelBreakout
        from src.patterns.breakout.gap import GapPattern
        from src.patterns.candlestick.dark_cloud import DarkCloudCover, PiercingLine
        from src.patterns.candlestick.doji import Doji
        from src.patterns.candlestick.engulfing import Engulfing
        from src.patterns.candlestick.hammer import Hammer
        from src.patterns.candlestick.harami import Harami
        from src.patterns.classic.ascending_triangle import AscendingTriangle
        from src.patterns.classic.dead_cat_bounce import DeadCatBounce
        from src.patterns.classic.descending_triangle import DescendingTriangle
        from src.patterns.classic.double_bottom import DoubleBottom
        from src.patterns.classic.double_top import DoubleTop
        from src.patterns.classic.rectangle import Rectangle
        from src.patterns.classic.trader_vic_2b import TraderVic2B
        from src.patterns.classic.triple_bottom import TripleBottom
        from src.patterns.classic.triple_top import TripleTop
        from src.patterns.classic.wedge import Wedge
        from src.patterns.complex.cup_handle import CupAndHandle
        from src.patterns.complex.head_shoulders import HeadAndShoulders
        from src.patterns.complex.parabolic_arc import ParabolicArc
        from src.patterns.complex.spike_ledge import SpikeAndLedge
        from src.patterns.complex.three_hills import ThreeHillsMountain
        from src.patterns.continuation.flag import Flag
        from src.patterns.continuation.pennant import Pennant
        from src.patterns.harmonic.abc import ABCPattern
        from src.patterns.harmonic.bollinger import BollingerBands
        from src.patterns.harmonic.gartley import GartleyPattern
        from src.patterns.harmonic.symmetric_triangle import SymmetricTriangle

        return [
            # Harmonic
            GartleyPattern(),
            ABCPattern(),
            SymmetricTriangle(),
            BollingerBands(),
            # Classic
            DoubleTop(),
            DoubleBottom(),
            TraderVic2B(),
            TripleTop(),
            TripleBottom(),
            AscendingTriangle(),
            DescendingTriangle(),
            Rectangle(),
            Wedge(),
            DeadCatBounce(),
            # Complex
            CupAndHandle(),
            HeadAndShoulders(),
            SpikeAndLedge(),
            ThreeHillsMountain(),
            ParabolicArc(),
            # Breakout
            DonchianChannelBreakout(),
            GapPattern(),
            # Continuation
            Flag(),
            Pennant(),
            # Basic
            MarketStructureLow(),
            MatchingLows(),
            NR7ID(),
            NBarDecline(),
            FloorPivotBreakout(),
            TwoBarReversal(),
            # Candlestick
            Doji(),
            Harami(),
            Hammer(),
            Engulfing(),
            DarkCloudCover(),
            PiercingLine(),
        ]

    def precompute(self, df: pd.DataFrame) -> None:
        """Pre-compute all pattern signals across the dataset.

        Args:
            df: DataFrame with OHLCV columns (Open, High, Low, Close, Volume).
        """
        if len(df) < 5:
            return

        self._n_bars = len(df)
        self._patterns = self._init_patterns()
        self._volume = df["Volume"].to_numpy() if "Volume" in df.columns else None

        for pattern in self._patterns:
            try:
                signals = pattern.detect_vectorized(df)
                self._signals_cache[pattern.name] = signals
                self._reliability[pattern.name] = PATTERN_RELIABILITY.get(
                    pattern.name, PATTERN_RELIABILITY["default"]
                )
            except Exception:
                _logger.debug("Pattern precompute failed for %s", pattern.name, exc_info=True)
                self._signals_cache[pattern.name] = np.zeros(self._n_bars, dtype=np.int8)
                self._reliability[pattern.name] = 0.0

        self._precomputed = True

    def get_boost(self, idx: int) -> PatternBoost:
        """Compute bull/bear boost for a single bar index.

        Args:
            idx: Bar index (0-based, within precomputed range).

        Returns:
            PatternBoost with bull_boost, bear_boost, and pattern lists.
        """
        if not self._precomputed or idx >= self._n_bars:
            return PatternBoost()

        result = PatternBoost()
        bull_weights: List[float] = []
        bear_weights: List[float] = []

        for pattern_name, signals in self._signals_cache.items():
            if idx >= len(signals):
                continue
            sig = signals[idx]
            if sig == 0:
                continue

            reliability = self._reliability.get(pattern_name, PATTERN_RELIABILITY["default"])
            if reliability <= 0:
                continue

            if sig == 1:  # Bullish
                result.bull_patterns.append(pattern_name)
                result.bullish_count += 1
                bull_weights.append(reliability)
            elif sig == -1:  # Bearish
                result.bear_patterns.append(pattern_name)
                result.bearish_count += 1
                bear_weights.append(reliability)

        # Compute boost scores
        result.bull_boost = self._compute_boost(bull_weights, result.bullish_count)
        result.bear_boost = self._compute_boost(bear_weights, result.bearish_count)

        # Volume confirmation bonus
        if self._volume is not None and idx > 0:
            vol_ratio = self._volume[idx] / max(self._volume[idx - 1], 1e-10)
            if vol_ratio > 1.5:
                if result.bull_boost > 0:
                    result.bull_boost = min(
                        result.bull_boost + self._volume_confirm_bonus, self._max_boost
                    )
                if result.bear_boost > 0:
                    result.bear_boost = min(
                        result.bear_boost + self._volume_confirm_bonus, self._max_boost
                    )

        return result

    def _compute_boost(self, weights: List[float], count: int) -> float:
        """Compute boost from pattern weights and confluence count.

        Args:
            weights: List of reliability weights for firing patterns.
            count: Number of patterns firing (should equal len(weights)).

        Returns:
            Boost value clamped to [0, max_boost].
        """
        if count < self._min_patterns or not weights:
            return 0.0

        # Base: average reliability of firing patterns
        avg_reliability = sum(weights) / len(weights)

        # Confluence bonus: each pattern beyond the first adds a small bonus
        confluence = self._confluence_bonus * max(0, count - 1)

        # Scale: base scaling factor × (avg reliability + confluence)
        boost = self._boost_scale * (avg_reliability + confluence)

        # Clamp to [0, max_boost]
        return max(0.0, min(boost, self._max_boost))

    @property
    def pattern_names(self) -> List[str]:
        """List of registered pattern names."""
        return list(self._signals_cache.keys())
