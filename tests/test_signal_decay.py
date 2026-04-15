# -*- coding: utf-8 -*-
"""
Tests for signal decay functionality.
"""

import pytest

from src.patterns.base import PatternResult, PatternType, SignalDirection, TradeSignal
from src.strategies.confluence import ConfluenceScore, ConfluenceScorer


class TestSignalDecay:
    """Test signal decay functionality."""

    def test_no_decay_within_validity_period(self):
        """Test that confidence doesn't decay within validity period."""
        scorer = ConfluenceScorer()

        score = ConfluenceScore(
            score=0.70,
            direction=SignalDirection.LONG,
            confidence_level=2,
            pattern_count=2,
            patterns=["Double Bottom", "Market Structure Low"],
            regime_alignment=True,
            risk_reward_ratio=2.0,
            quality_score=0.75,
        )

        # Basic patterns have 5 bar validity
        decayed = scorer.apply_signal_decay(score, bars_since_detection=3)
        assert decayed.score == pytest.approx(0.70)

    def test_decay_after_validity_period(self):
        """Test that confidence decays after validity period."""
        scorer = ConfluenceScorer()

        score = ConfluenceScore(
            score=0.70,
            direction=SignalDirection.LONG,
            confidence_level=2,
            pattern_count=2,
            patterns=["Market Structure Low", "NR7 Inside Day"],
            regime_alignment=True,
            risk_reward_ratio=2.0,
            quality_score=0.75,
        )

        # Basic patterns: 5 bar validity, decay rate 0.05 per bar
        # After 8 bars: 3 excess bars * 0.05 = 0.15 decay
        decayed = scorer.apply_signal_decay(score, bars_since_detection=8)
        assert decayed.score == pytest.approx(0.55)

    def test_decay_respects_minimum_confidence(self):
        """Test that decay doesn't go below minimum confidence."""
        scorer = ConfluenceScorer()

        score = ConfluenceScore(
            score=0.50,
            direction=SignalDirection.LONG,
            confidence_level=1,
            pattern_count=1,
            patterns=["Double Bottom"],
            regime_alignment=False,
            risk_reward_ratio=1.5,
            quality_score=0.5,
        )

        # After many bars, should floor at 0.40
        decayed = scorer.apply_signal_decay(score, bars_since_detection=100)
        assert decayed.score == pytest.approx(0.40)

    def test_different_validity_periods_by_category(self):
        """Test that different pattern categories have different validity periods."""
        scorer = ConfluenceScorer()

        # Complex patterns have 20 bar validity
        score = ConfluenceScore(
            score=0.70,
            direction=SignalDirection.LONG,
            confidence_level=3,
            pattern_count=3,
            patterns=["Head and Shoulders"],
            regime_alignment=True,
            risk_reward_ratio=2.0,
            quality_score=0.75,
        )

        # No decay at 15 bars (within 20 bar validity)
        decayed = scorer.apply_signal_decay(score, bars_since_detection=15)
        assert decayed.score == pytest.approx(0.70)

        # Decay at 25 bars (5 excess * 0.05 = 0.25)
        decayed = scorer.apply_signal_decay(score, bars_since_detection=25)
        assert decayed.score == pytest.approx(0.45)

    def test_decay_metadata(self):
        """Test that decay metadata is properly recorded."""
        scorer = ConfluenceScorer()

        score = ConfluenceScore(
            score=0.70,
            direction=SignalDirection.LONG,
            confidence_level=2,
            pattern_count=2,
            patterns=["Market Structure Low"],
            regime_alignment=True,
            risk_reward_ratio=2.0,
            quality_score=0.75,
        )

        decayed = scorer.apply_signal_decay(score, bars_since_detection=8)
        assert decayed.metadata.get("decay_applied", 0) == pytest.approx(0.15)
        assert decayed.metadata.get("bars_since_detection", 0) == 8
