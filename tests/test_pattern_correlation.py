# -*- coding: utf-8 -*-
"""
Tests for pattern correlation filtering.
"""

from src.patterns.base import PatternResult, PatternType, SignalDirection, TradeSignal
from src.strategies.confluence import ConfluenceScorer


class TestPatternCorrelation:
    """Test pattern correlation filtering."""

    def test_filter_correlated_double_patterns(self):
        """Test that only 1 double pattern is allowed."""
        scorer = ConfluenceScorer()

        # Create mock results for Double Top and Double Bottom (same direction)
        results = [
            self._make_result("Double Top", SignalDirection.SHORT),
            self._make_result("Triple Top", SignalDirection.SHORT),
        ]

        filtered = scorer._filter_correlated_patterns(results)

        # Should only allow 1 from double_patterns group
        assert len(filtered) == 1

    def test_filter_correlated_harmonic(self):
        """Test that only 1 harmonic pattern is allowed."""
        scorer = ConfluenceScorer()

        results = [
            self._make_result("Gartley", SignalDirection.LONG),
            self._make_result("ABC", SignalDirection.LONG),
        ]

        filtered = scorer._filter_correlated_patterns(results)

        # Should only allow 1 from harmonic group
        assert len(filtered) == 1

    def test_filter_breakout_allows_two(self):
        """Test that breakout group allows 2 patterns."""
        scorer = ConfluenceScorer()

        results = [
            self._make_result("NR7 Inside Day", SignalDirection.LONG),
            self._make_result("Donchian Channel", SignalDirection.LONG),
            self._make_result("Bollinger Bands", SignalDirection.LONG),
        ]

        filtered = scorer._filter_correlated_patterns(results)

        # Should allow 2 from breakout group
        assert len(filtered) == 2

    def test_no_correlation_unrelated_patterns(self):
        """Test that unrelated patterns are not filtered."""
        scorer = ConfluenceScorer()

        results = [
            self._make_result("Market Structure Low", SignalDirection.LONG),
            self._make_result("Cup and Handle", SignalDirection.LONG),
        ]

        filtered = scorer._filter_correlated_patterns(results)

        # Both should pass (not in correlation groups)
        assert len(filtered) == 2

    def _make_result(self, name: str, direction: SignalDirection) -> PatternResult:
        """Create a mock PatternResult."""

        signal = TradeSignal(
            pattern_name=name,
            direction=direction,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit_1=110.0,
            take_profit_2=None,
            take_profit_3=None,
            confidence=0.7,
        )

        return PatternResult(
            pattern_name=name,
            pattern_type=PatternType.REVERSAL,
            detected=True,
            signal=signal,
        )
