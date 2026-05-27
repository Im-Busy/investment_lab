"""P28-12: Patternity deterministic chart pattern recognition wrapper.

Wraps the patternity library's deterministic pattern recognition algorithm
for comparison against the project's 54 pattern detectors.

Source: awesome-ai-in-finance — patternity provides an alternative
deterministic (non-heuristic) approach to pattern detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.patterns.base import PatternResult, PatternType, SignalDirection, TradeSignal

logger = logging.getLogger(__name__)

_patternity_available = False
try:
    import patternity

    _patternity_available = True
except ImportError:
    logger.info("patternity not installed — PatternityWrapper will produce no signals")


_pat_to_project: dict[str, PatternType] = {
    "head_and_shoulders": PatternType.REVERSAL,
    "inverse_head_and_shoulders": PatternType.REVERSAL,
    "double_top": PatternType.REVERSAL,
    "double_bottom": PatternType.REVERSAL,
    "triple_top": PatternType.REVERSAL,
    "triple_bottom": PatternType.REVERSAL,
    "ascending_triangle": PatternType.CONTINUATION,
    "descending_triangle": PatternType.CONTINUATION,
    "symmetrical_triangle": PatternType.CONTINUATION,
    "flag": PatternType.CONTINUATION,
    "pennant": PatternType.CONTINUATION,
    "wedge": PatternType.REVERSAL,
    "cup_and_handle": PatternType.CONTINUATION,
    "rounding_bottom": PatternType.REVERSAL,
    "rounding_top": PatternType.REVERSAL,
    "broadening_formation": PatternType.BREAKOUT,
    "diamond_top": PatternType.REVERSAL,
    "diamond_bottom": PatternType.REVERSAL,
}


@dataclass
class PatternityMatch:
    """A single patternity detection result."""

    pattern_name: str
    start_idx: int
    end_idx: int
    confidence: float
    target_price: float
    break_price: float = 0.0


class PatternityWrapper:
    """Wrapper for patternity library providing project-compatible output.

    Use `detect()` for a single DataFrame, or `compare_with_internal()` to
    compare patternity results against the project's own detectors.
    """

    def __init__(self) -> None:
        self._patternity = None
        if _patternity_available:
            self._patternity = patternity

    @property
    def available(self) -> bool:
        """Whether the patternity library is installed and usable."""
        return self._patternity is not None

    def detect(self, df: pd.DataFrame) -> list[PatternityMatch]:
        """Detect patterns using patternity's algorithm.

        Args:
            df: OHLCV DataFrame with columns: Open, High, Low, Close, Volume.

        Returns:
            List of PatternityMatch objects for detected patterns.
        """
        if not self.available:
            return []

        matches: list[PatternityMatch] = []
        try:
            result = self._patternity.detect(df)
            if result is None:
                return matches

            for pattern in result.get("patterns", []) if isinstance(result, dict) else result:
                if isinstance(pattern, dict):
                    matches.append(
                        PatternityMatch(
                            pattern_name=str(pattern.get("name", "unknown")),
                            start_idx=int(pattern.get("start", 0)),
                            end_idx=int(pattern.get("end", 0)),
                            confidence=float(pattern.get("confidence", 0.5)),
                            target_price=float(pattern.get("target", 0.0)),
                            break_price=float(pattern.get("break_price", 0.0)),
                        )
                    )
        except Exception:
            logger.debug("patternity detect failed", exc_info=True)

        return matches

    def to_pattern_results(
        self,
        matches: list[PatternityMatch],
        window_size: int,
    ) -> list[PatternResult]:
        """Convert patternity detections to project PatternResult format.

        Args:
            matches: Raw patternity detection results.
            window_size: Number of bars to fill with signals (same length as data).

        Returns:
            List of PatternResult objects compatible with project scoring.
        """
        results: list[PatternResult] = []
        if not matches:
            return results

        closes = np.zeros(window_size, dtype=np.float64)
        signal_flags = np.zeros(window_size, dtype=bool)

        for match in matches:
            idx = match.end_idx
            if 0 <= idx < window_size:
                signal_flags[idx] = True
                closes[idx] = match.target_price

        if signal_flags.any():
            direction = SignalDirection.BULLISH
            results.append(
                PatternResult(
                    pattern_type=PatternType.REVERSAL,
                    signal_direction=direction,
                    signal=np.asarray(signal_flags),
                    signal_strength=np.where(signal_flags, 0.7, 0.0),
                    confidence=self._compute_confidence(matches, window_size),
                    pattern_name="patternity_composite",
                    trade_signal=np.array(
                        [TradeSignal.BUY if s else TradeSignal.NONE for s in signal_flags]
                    ),
                    target_prices=closes,
                )
            )

        return results

    def compare_with_internal(
        self,
        df: pd.DataFrame,
        internal_signals: list[PatternResult],
    ) -> dict:
        """Compare patternity detections against internal project detectors.

        Args:
            df: OHLCV DataFrame.
            internal_signals: PatternResult list from project detectors.

        Returns:
            Dict with overlap statistics: common_bar_count, patternity_only,
            internal_only, agreement_pct.
        """
        patternity_matches = self.detect(df)
        patternity_bars: set[int] = set()
        for m in patternity_matches:
            for i in range(m.start_idx, m.end_idx + 1):
                patternity_bars.add(i)

        internal_bars: set[int] = set()
        for sig in internal_signals:
            for i, flag in enumerate(sig.signal):
                if flag:
                    internal_bars.add(i)

        common = patternity_bars & internal_bars
        union = patternity_bars | internal_bars

        return {
            "patternity_only": len(patternity_bars - internal_bars),
            "internal_only": len(internal_bars - patternity_bars),
            "common": len(common),
            "agreement_pct": len(common) / len(union) * 100 if union else 0.0,
            "patternity_total": len(patternity_bars),
            "internal_total": len(internal_bars),
        }

    @staticmethod
    def _compute_confidence(
        matches: list[PatternityMatch],
        window_size: int,
    ) -> np.ndarray:
        """Compute per-bar confidence from patternity matches."""
        confidence = np.zeros(window_size, dtype=np.float64)
        for match in matches:
            for i in range(max(0, match.start_idx), min(match.end_idx + 1, window_size)):
                confidence[i] = max(confidence[i], match.confidence)
        return confidence
