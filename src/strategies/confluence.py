"""
Enhanced Confluence Scoring System

Implements multi-pattern confluence scoring with market regime adaptation,
pattern compatibility checking, and signal quality assessment.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..indicators.regime import RegimeState, TrendDirection, VolatilityRegime
from ..patterns.base import PatternResult, PatternType, SignalDirection, TradeSignal


@dataclass
class ConfluenceScore:
    """
    Confluence score for multiple signals.

    Attributes:
        score: Final confluence score (0.0 to 1.0)
        direction: Signal direction (LONG or SHORT)
        confidence_level: Level 1-4 based on number of patterns
        pattern_count: Number of patterns in confluence
        patterns: List of pattern names
        regime_alignment: Whether signals align with market regime
        risk_reward_ratio: Average risk/reward ratio
        quality_score: Signal quality assessment
        metadata: Additional information
    """

    score: float
    direction: SignalDirection
    confidence_level: int
    pattern_count: int
    patterns: List[str]
    regime_alignment: bool
    risk_reward_ratio: float
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "direction": self.direction.value,
            "confidence_level": self.confidence_level,
            "pattern_count": self.pattern_count,
            "patterns": self.patterns,
            "regime_alignment": self.regime_alignment,
            "risk_reward_ratio": self.risk_reward_ratio,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
        }


@dataclass
class PatternCompatibility:
    """Pattern compatibility rules."""

    pattern_a: str
    pattern_b: str
    compatible: bool
    reason: str


class ConfluenceScorer:
    """
    Enhanced Confluence Scoring System

    Calculates confluence scores for multiple pattern signals with
    market regime adaptation and pattern compatibility checking.
    """

    # Pattern category weights
    CATEGORY_BASE_WEIGHTS = {"basic": 1.0, "harmonic": 1.0, "complex": 1.2, "classic": 1.0}

    # Pattern type weights
    TYPE_BASE_WEIGHTS = {
        PatternType.REVERSAL: 1.0,
        PatternType.CONTINUATION: 1.0,
        PatternType.BREAKOUT: 1.0,
        PatternType.COUNTER_TREND: 0.9,
        PatternType.VOLATILITY: 1.0,
    }

    # Confluence bonuses
    CONFLUENCE_BONUSES = {
        1: 0.0,  # Single pattern - no bonus
        2: 0.10,  # 2 patterns - 10% bonus
        3: 0.20,  # 3 patterns - 20% bonus
        4: 0.30,  # 4+ patterns - 30% bonus
    }

    # Minimum confidence by level
    MIN_CONFIDENCE = {
        1: 0.50,  # Level 1 - single pattern
        2: 0.55,  # Level 2 - 2 patterns
        3: 0.65,  # Level 3 - 3 patterns
        4: 0.75,  # Level 4 - 4+ patterns
    }

    # Signal validity periods (bars) - from trading_strategy.md
    SIGNAL_VALIDITY = {
        "basic": 5,
        "harmonic": 10,
        "complex": 20,
        "classic": 15,
    }

    # Confidence decay rate per bar after validity period
    DECAY_RATE = 0.05

    # Minimum confidence threshold (below this, signal is invalid)
    MIN_DECAY_CONFIDENCE = 0.40

    # Pattern correlation groups (from trading_strategy.md)
    CORRELATION_GROUPS = {
        "double_patterns": ["Double Top", "Double Bottom", "Triple Top", "Triple Bottom"],
        "harmonic": ["Gartley", "ABC"],
        "breakout": ["NR7 Inside Day", "Donchian Channel", "Bollinger Bands"],
        "reversal_tops": ["Head and Shoulders", "Double Top", "Trader Vic's 2B"],
        "reversal_bottoms": ["Double Bottom", "Market Structure Low", "Matching Lows"],
    }

    # Maximum active patterns per correlation group
    MAX_CORRELATED = {
        "double_patterns": 1,
        "harmonic": 1,
        "breakout": 2,
        "reversal_tops": 1,
        "reversal_bottoms": 1,
    }

    def __init__(
        self,
        min_confidence: float = 0.50,
        trend_alignment_bonus: float = 0.05,
        regime_adaptation: bool = True,
    ):
        """
        Initialize Confluence Scorer.

        Args:
            min_confidence: Minimum confidence threshold
            trend_alignment_bonus: Bonus for trend alignment
            regime_adaptation: Whether to adapt to market regime
        """
        self.min_confidence = min_confidence
        self.trend_alignment_bonus = trend_alignment_bonus
        self.regime_adaptation = regime_adaptation
        self._compatibility_rules = self._build_compatibility_rules()

    def _build_compatibility_rules(self) -> Dict[Tuple[str, str], PatternCompatibility]:
        """Build pattern compatibility rules."""
        rules = {}

        # Conflicting patterns (same direction, different logic)
        conflicting_pairs = [
            ("Double Top", "Double Bottom"),
            ("Double Top", "Head and Shoulders"),  # Both tops
            ("Double Bottom", "Cup and Handle"),  # Opposite bias
            ("Head and Shoulders", "Cup and Handle"),
            ("Triple Top", "Triple Bottom"),
            ("Triple Top", "Double Top"),  # Similar patterns
            ("Triple Bottom", "Double Bottom"),
            ("Gartley", "ABC"),  # Both harmonic
        ]

        for pattern_a, pattern_b in conflicting_pairs:
            rules[(pattern_a, pattern_b)] = PatternCompatibility(
                pattern_a=pattern_a,
                pattern_b=pattern_b,
                compatible=False,
                reason="Conflicting direction or similar logic",
            )
            rules[(pattern_b, pattern_a)] = PatternCompatibility(
                pattern_a=pattern_b,
                pattern_b=pattern_a,
                compatible=False,
                reason="Conflicting direction or similar logic",
            )

        return rules

    def _get_pattern_category(self, pattern_name: str) -> str:
        """Get pattern category from name."""
        basic_patterns = [
            "Market Structure Low",
            "Matching Lows",
            "NR7 Inside Day",
            "N-Bar Decline",
            "Floor Pivot Breakout",
        ]
        harmonic_patterns = [
            "Gartley",
            "ABC",
            "Symmetric Triangle",
            "Donchian Channel",
            "Bollinger Bands",
        ]
        complex_patterns = [
            "Cup and Handle",
            "Head and Shoulders",
            "Spike and Ledge",
            "Three Hills",
            "Parabolic Arc",
        ]
        classic_patterns = [
            "Double Top",
            "Double Bottom",
            "Trader Vic's 2B",
            "Triple Top",
            "Dead Cat Bounce",
        ]

        for pattern in basic_patterns:
            if pattern.lower() in pattern_name.lower():
                return "basic"
        for pattern in harmonic_patterns:
            if pattern.lower() in pattern_name.lower():
                return "harmonic"
        for pattern in complex_patterns:
            if pattern.lower() in pattern_name.lower():
                return "complex"
        for pattern in classic_patterns:
            if pattern.lower() in pattern_name.lower():
                return "classic"

        return "classic"  # Default

    def _check_compatibility(self, pattern_a: str, pattern_b: str) -> PatternCompatibility:
        """Check if two patterns are compatible."""
        key = (pattern_a, pattern_b)
        if key in self._compatibility_rules:
            return self._compatibility_rules[key]

        # Default to compatible
        return PatternCompatibility(
            pattern_a=pattern_a, pattern_b=pattern_b, compatible=True, reason="No conflict detected"
        )

    def _filter_compatible_patterns(self, results: List[PatternResult]) -> List[PatternResult]:
        """Filter out incompatible patterns."""
        if len(results) <= 1:
            return results

        # Group by direction
        long_results = [
            r for r in results if r.signal and r.signal.direction == SignalDirection.LONG
        ]
        short_results = [
            r for r in results if r.signal and r.signal.direction == SignalDirection.SHORT
        ]

        # Check compatibility within each direction
        def filter_group(group: List[PatternResult]) -> List[PatternResult]:
            if len(group) <= 1:
                return group

            filtered = [group[0]]
            for result in group[1:]:
                is_compatible = True
                for existing in filtered:
                    compat = self._check_compatibility(existing.pattern_name, result.pattern_name)
                    if not compat.compatible:
                        is_compatible = False
                        break

                if is_compatible:
                    filtered.append(result)

            return filtered

        return filter_group(long_results) + filter_group(short_results)

    def _filter_correlated_patterns(self, results: List[PatternResult]) -> List[PatternResult]:
        """
        Filter patterns to enforce correlation limits.

        Only applies to patterns in the same direction.

        Documented correlation limits (from trading_strategy.md):
        - Double Patterns: Max 1 active
        - Harmonic: Max 1 active
        - Breakout: Max 2 active
        - Reversal Tops: Max 1 active
        - Reversal Bottoms: Max 1 active
        """
        if len(results) <= 1:
            return results

        # Group by direction
        long_results = [
            r for r in results if r.signal and r.signal.direction == SignalDirection.LONG
        ]
        short_results = [
            r for r in results if r.signal and r.signal.direction == SignalDirection.SHORT
        ]

        def filter_by_correlation(group: List[PatternResult]) -> List[PatternResult]:
            if len(group) <= 1:
                return group

            # Track which correlation groups have been used
            used_groups: Dict[str, int] = {}
            filtered = []

            for result in group:
                pattern_name = result.pattern_name

                # Find which correlation group this pattern belongs to
                pattern_group = None
                for group_name, patterns in self.CORRELATION_GROUPS.items():
                    if any(p.lower() in pattern_name.lower() for p in patterns):
                        pattern_group = group_name
                        break

                if pattern_group is None:
                    # Not in any correlation group, always include
                    filtered.append(result)
                    continue

                # Check if we've hit the limit for this group
                current_count = used_groups.get(pattern_group, 0)
                max_allowed = self.MAX_CORRELATED.get(pattern_group, 999)

                if current_count < max_allowed:
                    filtered.append(result)
                    used_groups[pattern_group] = current_count + 1

            return filtered

        return filter_by_correlation(long_results) + filter_by_correlation(short_results)

    def _calculate_pattern_weight(
        self, result: PatternResult, regime: Optional[RegimeState]
    ) -> float:
        """Calculate pattern weight with regime adaptation."""
        # Base category weight
        category = self._get_pattern_category(result.pattern_name)
        weight = self.CATEGORY_BASE_WEIGHTS.get(category, 1.0)

        # Pattern type weight
        type_weight = self.TYPE_BASE_WEIGHTS.get(result.pattern_type, 1.0)
        weight *= type_weight

        # Regime adaptation
        if regime and self.regime_adaptation:
            regime_weights = self._get_regime_weights(regime)
            category_mult = regime_weights.get("category", {}).get(category, 1.0)
            type_mult = regime_weights.get("type", {}).get(result.pattern_type.value, 1.0)
            weight *= category_mult * type_mult

        return weight

    def _get_regime_weights(self, regime: RegimeState) -> Dict[str, Dict[str, float]]:
        """
        Get regime-based weight multipliers matching documentation.

        Documented weights from trading_strategy.md:
        | Pattern  | Trending | Ranging | Volatile | Quiet |
        |----------|----------|---------|----------|-------|
        | Basic    | 0.8      | 1.0     | 0.7      | 1.0   |
        | Harmonic | 1.0      | 0.8     | 0.6      | 1.2   |
        | Complex  | 1.2      | 0.6     | 0.8      | 0.7   |
        | Classic  | 1.0      | 1.0     | 1.0      | 1.0   |
        """
        weights = {
            "category": {"basic": 1.0, "harmonic": 1.0, "complex": 1.0, "classic": 1.0},
            "type": {
                "Reversal": 1.0,
                "Continuation": 1.0,
                "Breakout": 1.0,
                "Counter-Trend": 1.0,
                "Volatility": 1.0,
            },
        }

        # Determine regime category
        is_trending = regime.trend_direction in [TrendDirection.UPTREND, TrendDirection.DOWNTREND]
        is_high_vol = regime.volatility_regime == VolatilityRegime.HIGH
        is_low_vol = regime.volatility_regime == VolatilityRegime.LOW

        if is_trending and not is_high_vol:
            # Trending regime
            weights["category"]["basic"] = 0.8
            weights["category"]["harmonic"] = 1.0
            weights["category"]["complex"] = 1.2
            weights["type"]["Continuation"] = 1.3
            weights["type"]["Reversal"] = 0.7
        elif is_high_vol:
            # Volatile regime
            weights["category"]["basic"] = 0.7
            weights["category"]["harmonic"] = 0.6
            weights["category"]["complex"] = 0.8
            weights["type"]["Breakout"] = 1.2
            weights["type"]["Reversal"] = 0.8
        elif is_low_vol:
            # Quiet regime
            weights["category"]["basic"] = 1.0
            weights["category"]["harmonic"] = 1.2
            weights["category"]["complex"] = 0.7
            weights["type"]["Breakout"] = 0.7
        else:
            # Ranging regime
            weights["category"]["basic"] = 1.0
            weights["category"]["harmonic"] = 0.8
            weights["category"]["complex"] = 0.6
            weights["type"]["Reversal"] = 1.2
            weights["type"]["Continuation"] = 0.6

        return weights

    def apply_signal_decay(
        self, score: ConfluenceScore, bars_since_detection: int
    ) -> ConfluenceScore:
        """
        Apply time-based confidence decay to signal.

        Signal Validity Period (from trading_strategy.md):
        - Basic Patterns: 3-5 bars
        - Harmonic Patterns: 5-10 bars
        - Complex Patterns: 10-20 bars
        - Classic Patterns: 5-15 bars

        Confidence Decay: 5% per bar after valid period
        Minimum Confidence: 40% (below this, signal invalid)

        Args:
            score: Original confluence score
            bars_since_detection: Bars elapsed since pattern detected

        Returns:
            Updated ConfluenceScore with decayed confidence
        """
        if bars_since_detection <= 0:
            return score

        # Get validity period based on pattern category
        category = self._get_pattern_category(score.patterns[0]) if score.patterns else "classic"
        validity_period = self.SIGNAL_VALIDITY.get(category, 10)

        # No decay within validity period
        if bars_since_detection <= validity_period:
            return score

        # Apply decay after validity period
        excess_bars = bars_since_detection - validity_period
        decay = excess_bars * self.DECAY_RATE

        new_score = max(self.MIN_DECAY_CONFIDENCE, score.score - decay)

        return ConfluenceScore(
            score=new_score,
            direction=score.direction,
            confidence_level=score.confidence_level,
            pattern_count=score.pattern_count,
            patterns=score.patterns,
            regime_alignment=score.regime_alignment,
            risk_reward_ratio=score.risk_reward_ratio,
            quality_score=score.quality_score,
            metadata={
                **score.metadata,
                "decay_applied": decay,
                "bars_since_detection": bars_since_detection,
            },
        )

    def _calculate_risk_reward(self, signals: List[TradeSignal]) -> float:
        """Calculate average risk/reward ratio."""
        if not signals:
            return 0.0

        rr_ratios = []
        for signal in signals:
            if signal.entry_price and signal.stop_loss and signal.take_profit_1:
                risk = abs(signal.entry_price - signal.stop_loss)
                reward = abs(signal.take_profit_1 - signal.entry_price)
                if risk > 0:
                    rr_ratios.append(reward / risk)

        return np.mean(rr_ratios) if rr_ratios else 0.0

    def _calculate_quality_score(
        self, results: List[PatternResult], regime: Optional[RegimeState]
    ) -> float:
        """Calculate signal quality score."""
        if not results:
            return 0.0

        quality_factors = []

        # Factor 1: Average confidence
        confidences = [r.signal.confidence for r in results if r.signal]
        avg_confidence = np.mean(confidences) if confidences else 0.5
        quality_factors.append(avg_confidence)

        # Factor 2: Pattern diversity (different categories)
        categories = set(self._get_pattern_category(r.pattern_name) for r in results)
        diversity_score = len(categories) / 4.0  # Normalize to 0-1
        quality_factors.append(diversity_score)

        # Factor 3: Risk/reward quality
        signals = [r.signal for r in results if r.signal]
        rr_ratio = self._calculate_risk_reward(signals)
        rr_score = min(1.0, rr_ratio / 2.0)  # Normalize (2:1 RR = 1.0)
        quality_factors.append(rr_score)

        # Factor 4: Regime alignment
        if regime:
            regime_alignment = self._check_regime_alignment(results, regime)
            quality_factors.append(1.0 if regime_alignment else 0.5)

        return np.mean(quality_factors)

    def _check_regime_alignment(self, results: List[PatternResult], regime: RegimeState) -> bool:
        """Check if signals align with market regime."""
        if not results:
            return False

        # Get dominant direction
        long_count = sum(
            1 for r in results if r.signal and r.signal.direction == SignalDirection.LONG
        )
        short_count = sum(
            1 for r in results if r.signal and r.signal.direction == SignalDirection.SHORT
        )

        dominant_direction = (
            SignalDirection.LONG if long_count >= short_count else SignalDirection.SHORT
        )

        # Check alignment with trend
        if regime.trend_direction == TrendDirection.UPTREND:
            return dominant_direction == SignalDirection.LONG
        elif regime.trend_direction == TrendDirection.DOWNTREND:
            return dominant_direction == SignalDirection.SHORT
        else:  # SIDEWAYS
            return True  # Both directions acceptable in ranging market

    def calculate_confluence(
        self, results: List[PatternResult], regime: Optional[RegimeState] = None
    ) -> Optional[ConfluenceScore]:
        """
        Calculate confluence score for multiple pattern results.

        Args:
            results: List of PatternResult objects
            regime: Current market regime state

        Returns:
            ConfluenceScore object or None if no valid signals
        """
        # Filter results with signals
        valid_results = [r for r in results if r.detected and r.signal]

        if not valid_results:
            return None

        # Filter incompatible patterns
        compatible_results = self._filter_compatible_patterns(valid_results)

        if not compatible_results:
            return None

        # Filter correlated patterns
        compatible_results = self._filter_correlated_patterns(compatible_results)

        if not compatible_results:
            return None

        # Separate by direction
        long_results = [r for r in compatible_results if r.signal.direction == SignalDirection.LONG]
        short_results = [
            r for r in compatible_results if r.signal.direction == SignalDirection.SHORT
        ]

        # Determine dominant direction
        if len(long_results) >= len(short_results):
            direction = SignalDirection.LONG
            active_results = long_results
        else:
            direction = SignalDirection.SHORT
            active_results = short_results

        if not active_results:
            return None

        # Calculate weighted confidence
        total_weight = 0.0
        weighted_sum = 0.0

        for result in active_results:
            weight = self._calculate_pattern_weight(result, regime)
            confidence = result.signal.confidence
            weighted_sum += confidence * weight
            total_weight += weight

        base_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Apply confluence bonus
        num_patterns = len(active_results)
        confluence_bonus = self.CONFLUENCE_BONUSES.get(min(num_patterns, 4), 0.30)

        # Trend alignment bonus
        regime_alignment = False
        if regime:
            regime_alignment = self._check_regime_alignment(active_results, regime)
            if regime_alignment:
                confluence_bonus += self.trend_alignment_bonus

        # Calculate final score
        final_score = min(0.95, base_score + confluence_bonus)

        # Determine confidence level
        if num_patterns >= 4:
            confidence_level = 4
        elif num_patterns == 3:
            confidence_level = 3
        elif num_patterns == 2:
            confidence_level = 2
        else:
            confidence_level = 1

        # Check minimum confidence
        min_conf = self.MIN_CONFIDENCE.get(confidence_level, 0.50)
        if final_score < min_conf:
            confidence_level = max(1, confidence_level - 1)

        # Calculate quality metrics
        signals = [r.signal for r in active_results]
        rr_ratio = self._calculate_risk_reward(signals)
        quality_score = self._calculate_quality_score(active_results, regime)

        # Build metadata
        metadata = {
            "base_score": base_score,
            "confluence_bonus": confluence_bonus,
            "pattern_weights": {
                r.pattern_name: self._calculate_pattern_weight(r, regime) for r in active_results
            },
            "categories": list(
                set(self._get_pattern_category(r.pattern_name) for r in active_results)
            ),
        }

        return ConfluenceScore(
            score=final_score,
            direction=direction,
            confidence_level=confidence_level,
            pattern_count=num_patterns,
            patterns=[r.pattern_name for r in active_results],
            regime_alignment=regime_alignment,
            risk_reward_ratio=rr_ratio,
            quality_score=quality_score,
            metadata=metadata,
        )

    def rank_signals(
        self, results: List[PatternResult], regime: Optional[RegimeState] = None, top_n: int = 3
    ) -> List[Tuple[PatternResult, ConfluenceScore]]:
        """
        Rank signals by confluence score.

        Args:
            results: List of PatternResult objects
            regime: Current market regime state
            top_n: Number of top signals to return

        Returns:
            List of (PatternResult, ConfluenceScore) tuples sorted by score
        """
        # Calculate individual scores
        scored_results = []

        for result in results:
            if not result.detected or not result.signal:
                continue

            # Create single-result confluence score
            score = self.calculate_confluence([result], regime)
            if score:
                scored_results.append((result, score))

        # Sort by score descending
        scored_results.sort(key=lambda x: x[1].score, reverse=True)

        return scored_results[:top_n]
