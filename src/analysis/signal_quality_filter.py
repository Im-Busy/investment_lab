"""
Signal Quality Filter for Pattern Selection

Adds a quality gate between pattern detection and confluence scoring.
Validates individual signals before they reach the confluence scorer.

Quality Score Formula:
Quality Score = (Pattern Confidence x 25%) +
                (Risk/Reward Score x 30%) +
                (Historical Performance x 25%) +
                (Regime Alignment x 20%)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SignalQualityConfig:
    """Configuration for signal quality filtering."""

    # Pattern detection confidence
    min_pattern_confidence: float = 0.60

    # Risk/Reward requirements
    min_risk_reward_ratio: float = 1.5
    max_stop_distance_pct: float = 5.0

    # Historical performance thresholds
    min_historical_win_rate: float = 0.35
    min_historical_profit_factor: float = 1.1

    # Regime alignment (optional)
    require_regime_alignment: bool = False

    # Quality score thresholds
    min_quality_score: float = 0.50

    # Component weights (must sum to 1.0)
    weight_confidence: float = 0.25
    weight_risk_reward: float = 0.30
    weight_historical: float = 0.25
    weight_regime: float = 0.20


@dataclass
class SignalQualityResult:
    """Result of signal quality evaluation."""

    signal_id: str
    pattern_name: str
    passes_quality_gate: bool
    quality_score: float

    # Component scores
    confidence_score: float
    risk_reward_score: float
    historical_score: float
    regime_score: float

    # Failure reasons
    failure_reasons: List[str] = field(default_factory=list)

    # Original signal data
    original_confidence: float = 0.0
    risk_reward_ratio: float = 0.0
    stop_distance_pct: float = 0.0
    historical_win_rate: float = 0.0
    historical_profit_factor: float = 0.0
    regime_aligned: bool = True


class SignalQualityFilter:
    """
    Filters signals based on quality criteria before confluence scoring.

    Acts as a gate between pattern detection and confluence scoring to
    ensure only high-quality signals reach the trading decision layer.
    """

    def __init__(self, config: Optional[SignalQualityConfig] = None):
        self.config = config or SignalQualityConfig()

    def evaluate_signal(
        self,
        signal_id: str,
        pattern_name: str,
        confidence: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        historical_win_rate: Optional[float] = None,
        historical_profit_factor: Optional[float] = None,
        regime_aligned: Optional[bool] = None,
        current_price: Optional[float] = None,
    ) -> SignalQualityResult:
        """
        Evaluate a single signal against quality criteria.

        Args:
            signal_id: Unique identifier for the signal
            pattern_name: Name of the pattern that generated the signal
            confidence: Pattern confidence score (0-1)
            entry_price: Suggested entry price
            stop_loss: Suggested stop loss price
            take_profit: Suggested take profit price
            historical_win_rate: Pattern's historical win rate (optional)
            historical_profit_factor: Pattern's historical profit factor (optional)
            regime_aligned: Whether signal aligns with current regime (optional)
            current_price: Current market price for stop distance calculation

        Returns:
            SignalQualityResult with pass/fail and component scores
        """
        failures = []

        # Calculate risk/reward ratio
        if entry_price > 0 and stop_loss > 0 and take_profit > 0:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0.0
        else:
            rr_ratio = 0.0

        # Calculate stop distance percentage
        if current_price and current_price > 0:
            stop_distance_pct = abs(current_price - stop_loss) / current_price * 100
        elif entry_price > 0:
            stop_distance_pct = abs(entry_price - stop_loss) / entry_price * 100
        else:
            stop_distance_pct = float("inf")

        # Check hard thresholds
        if confidence < self.config.min_pattern_confidence:
            failures.append(
                f"Confidence {confidence:.2f} < minimum {self.config.min_pattern_confidence}"
            )

        if rr_ratio < self.config.min_risk_reward_ratio:
            failures.append(
                f"Risk/Reward {rr_ratio:.2f} < minimum {self.config.min_risk_reward_ratio}"
            )

        if stop_distance_pct > self.config.max_stop_distance_pct:
            failures.append(
                f"Stop distance {stop_distance_pct:.1f}% > maximum {self.config.max_stop_distance_pct}%"
            )

        if (
            historical_win_rate is not None
            and historical_win_rate < self.config.min_historical_win_rate
        ):
            failures.append(
                f"Historical win rate {historical_win_rate:.0%} < minimum {self.config.min_historical_win_rate}"
            )

        if (
            historical_profit_factor is not None
            and historical_profit_factor < self.config.min_historical_profit_factor
        ):
            failures.append(
                f"Historical profit factor {historical_profit_factor:.2f} < minimum {self.config.min_historical_profit_factor}"
            )

        if (
            self.config.require_regime_alignment
            and regime_aligned is not None
            and not regime_aligned
        ):
            failures.append("Signal not aligned with current regime")

        # Calculate component scores (all normalized to 0-1)
        confidence_score = confidence

        rr_score = min(rr_ratio / 3.0, 1.0)  # 3:1 RR = perfect score

        if historical_win_rate is not None and historical_profit_factor is not None:
            historical_score = (historical_win_rate * 0.5) + (
                min(historical_profit_factor / 2.0, 1.0) * 0.5
            )
        else:
            historical_score = 0.5  # Neutral when no history available

        if regime_aligned is not None:
            regime_score = 1.0 if regime_aligned else 0.0
        else:
            regime_score = 0.5  # Neutral when regime not specified

        # Calculate weighted quality score
        quality_score = (
            confidence_score * self.config.weight_confidence
            + rr_score * self.config.weight_risk_reward
            + historical_score * self.config.weight_historical
            + regime_score * self.config.weight_regime
        )

        # Check quality score threshold
        if quality_score < self.config.min_quality_score:
            failures.append(
                f"Quality score {quality_score:.2f} < minimum {self.config.min_quality_score}"
            )

        passes = len(failures) == 0

        return SignalQualityResult(
            signal_id=signal_id,
            pattern_name=pattern_name,
            passes_quality_gate=passes,
            quality_score=quality_score,
            confidence_score=confidence_score,
            risk_reward_score=rr_score,
            historical_score=historical_score,
            regime_score=regime_score,
            failure_reasons=failures,
            original_confidence=confidence,
            risk_reward_ratio=rr_ratio,
            stop_distance_pct=stop_distance_pct,
            historical_win_rate=historical_win_rate or 0.0,
            historical_profit_factor=historical_profit_factor or 0.0,
            regime_aligned=regime_aligned if regime_aligned is not None else True,
        )

    def filter_signals(
        self,
        signals: List[Dict[str, Any]],
        historical_stats: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, List[SignalQualityResult]]:
        """
        Filter a batch of signals through the quality gate.

        Args:
            signals: List of signal dicts with keys:
                - signal_id, pattern_name, confidence, entry_price,
                  stop_loss, take_profit, current_price (optional)
            historical_stats: Optional Dict[pattern_name -> {win_rate, profit_factor}]

        Returns:
            Dict with keys 'passed' and 'failed', each containing SignalQualityResult
        """
        passed = []
        failed = []

        for signal in signals:
            pattern_name = signal.get("pattern_name", "")

            # Get historical stats if available
            hist_wr = None
            hist_pf = None
            if historical_stats and pattern_name in historical_stats:
                stats = historical_stats[pattern_name]
                hist_wr = stats.get("win_rate")
                hist_pf = stats.get("profit_factor")

            result = self.evaluate_signal(
                signal_id=signal.get("signal_id", ""),
                pattern_name=pattern_name,
                confidence=signal.get("confidence", 0.0),
                entry_price=signal.get("entry_price", 0.0),
                stop_loss=signal.get("stop_loss", 0.0),
                take_profit=signal.get("take_profit", 0.0),
                historical_win_rate=hist_wr,
                historical_profit_factor=hist_pf,
                regime_aligned=signal.get("regime_aligned"),
                current_price=signal.get("current_price"),
            )

            if result.passes_quality_gate:
                passed.append(result)
            else:
                failed.append(result)

        return {
            "passed": passed,
            "failed": failed,
        }

    def get_quality_summary(
        self,
        results: List[SignalQualityResult],
    ) -> Dict[str, Any]:
        """
        Generate summary statistics for quality filter results.

        Args:
            results: List of SignalQualityResult objects

        Returns:
            Dict with summary statistics
        """
        if not results:
            return {
                "total_signals": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "avg_quality_score": 0.0,
            }

        scores = [r.quality_score for r in results]
        passed_count = sum(1 for r in results if r.passes_quality_gate)

        return {
            "total_signals": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
            "pass_rate": passed_count / len(results) if results else 0.0,
            "avg_quality_score": np.mean(scores),
            "min_quality_score": np.min(scores),
            "max_quality_score": np.max(scores),
            "median_quality_score": np.median(scores),
        }
