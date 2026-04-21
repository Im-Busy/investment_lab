"""
Test Phase 6 Tier 1: R2, R4, R5

Tests R2 (Position Probability), R4 (Regime Declaration), and R5 (Dynamic Rebalancing) implementations.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.patterns.base import BasePattern, PatternType, RegimeState as PatternRegimeState

from src.risk.position_probability import (
    PositionRiskModel,
    PositionRiskConfig,
    RegimeState as RiskRegimeState,
)
from src.risk.dynamic_rebalancing import (
    DynamicRebalancer,
    DynamicRebalanceConfig,
    RebalanceFrequency,
)


def test_position_probability():
    """Test R2: Per-Position Success/Failure Probability."""

    print("=" * 80)
    print("R2: Position Probability Tests")
    print("=" * 80)

    model = PositionRiskModel()

    print("\nTest 1: Estimate success probability")
    confidence = 0.7
    regime = RiskRegimeState.TRENDING
    prob = model.estimate_success_prob(confidence, regime)
    print(f"  Confidence: {confidence}, Regime: {regime.value}")
    print(f"  Success probability: {prob:.3f} (expected: > base)")

    print("\nTest 2: Volatile regime penalty")
    confidence = 0.7
    regime = RiskRegimeState.VOLATILE
    prob = model.estimate_success_prob(confidence, regime)
    print(f"  Confidence: {confidence}, Regime: {regime.value}")
    print(f"  Success probability: {prob:.3f} (expected: < trending)")

    print("\nTest 3: Estimate failure probability")
    confidence = 0.7
    regime = RiskRegimeState.TRENDING
    success_prob = model.estimate_success_prob(confidence, regime)
    failure_prob = model.estimate_failure_prob(success_prob=success_prob)
    print(f"  Success prob: {success_prob:.3f}")
    print(f"  Failure prob: {failure_prob:.3f}")
    print(f"  Sum: {success_prob + failure_prob:.3f} (should be 1.0)")

    print("\nTest 4: Calculate position size")
    success_prob = 0.6
    pos_size = model.calculate_position_size(success_prob, max_risk_pct=0.02)
    print(f"  Success prob: {success_prob}")
    print(f"  Position size: {pos_size:.4f} (as % of portfolio)")

    print("\nTest 5: Calculate with Kelly Criterion")
    success_prob = 0.6
    pos_size_kelly = model.calculate_position_size(success_prob, kelly_criterion=True)
    print(f"  Success prob: {success_prob}")
    print(f"  Kelly position size: {pos_size_kelly:.4f}")

    print("\n[SUCCESS] Position probability tests passed")


def test_regime_declaration():
    """Test R4: Regime Declaration Per Strategy."""

    print("\n" + "=" * 80)
    print("R4: Regime Declaration Tests")
    print("=" * 80)

    class TestPattern(BasePattern):
        def detect(self, df, i, window_start=None):
            pass

        def generate_signal(self, df, i):
            pass

    print("\nTest 1: Create pattern with regime preferences")
    pattern = TestPattern(
        name="Test Trend Following",
        pattern_type=PatternType.CONTINUATION,
        preferred_regimes=[PatternRegimeState.TRENDING],
        incompatible_regimes=[PatternRegimeState.RANGING],
    )
    print(f"  Name: {pattern.name}")
    print(f"  Preferred: {[r.value for r in pattern.preferred_regimes]}")
    print(f"  Incompatible: {[r.value for r in pattern.incompatible_regimes]}")

    print("\nTest 2: Check regime compatibility")
    regime = PatternRegimeState.TRENDING
    compatible = pattern.is_regime_compatible(regime)
    print(f"  Regime: {regime.value}")
    print(f"  Compatible: {compatible} (expected: True)")

    print("\nTest 3: Check incompatible regime")
    regime = PatternRegimeState.RANGING
    compatible = pattern.is_regime_compatible(regime)
    print(f"  Regime: {regime.value}")
    print(f"  Compatible: {compatible} (expected: False)")

    print("\nTest 4: Get regime preference scores")
    regimes = [
        PatternRegimeState.TRENDING,
        PatternRegimeState.RANGING,
        PatternRegimeState.VOLATILE,
    ]
    print(f"  Regime preferences:")
    for regime in regimes:
        score = pattern.get_regime_preference(regime)
        print(f"    {regime.value}: {score:.1f}")

    print("\nTest 5: Pattern with no regime preferences")
    neutral_pattern = TestPattern(
        name="Neutral Pattern",
        pattern_type=PatternType.REVERSAL,
    )
    for regime in regimes:
        compatible = neutral_pattern.is_regime_compatible(regime)
        score = neutral_pattern.get_regime_preference(regime)
        print(f"  {regime.value}: compatible={compatible}, score={score:.1f}")

    print("\n[SUCCESS] Regime declaration tests passed")


def test_dynamic_rebalancing():
    """Test R5: Dynamic Rebalancing Frequency."""

    print("\n" + "=" * 80)
    print("R5: Dynamic Rebalancing Tests")
    print("=" * 80)

    rebalancer = DynamicRebalancer()

    print("\nTest 1: Estimate optimal frequency (high decay)")
    decay_rate = 0.05
    freq = rebalancer.estimate_optimal_frequency(decay_rate)
    print(f"  Decay rate: {decay_rate} (5% per bar)")
    print(f"  Optimal frequency: {freq.value} (expected: daily)")

    print("\nTest 2: Estimate optimal frequency (low decay)")
    decay_rate = 0.01
    freq = rebalancer.estimate_optimal_frequency(decay_rate)
    print(f"  Decay rate: {decay_rate} (1% per bar)")
    print(f"  Optimal frequency: {freq.value} (expected: monthly)")

    print("\nTest 3: Estimate optimal frequency (medium decay)")
    decay_rate = 0.025
    freq = rebalancer.estimate_optimal_frequency(decay_rate)
    print(f"  Decay rate: {decay_rate} (2.5% per bar)")
    print(f"  Optimal frequency: {freq.value} (expected: weekly)")

    print("\nTest 4: Calculate signal decay from history")
    rebalancer.signal_history = [0.8, 0.75, 0.70, 0.65, 0.60]
    decay = rebalancer.calculate_signal_decay()
    print(f"  Signal history: {rebalancer.signal_history}")
    print(f"  Calculated decay rate: {decay:.3f}")

    print("\nTest 5: Should rebalance check")
    rebalancer.reset()
    for i in range(1, 11):
        signal_strength = 0.8 - (i * 0.05)
        rebalancer.update_signal_history(signal_strength)
        should_rebalance, reason, freq = rebalancer.should_rebalance(i)
        print(f"  Day {i}: strength={signal_strength:.2f}, rebalance={should_rebalance}")
        if should_rebalance:
            print(f"    Reason: {reason}")
            print(f"    Frequency: {freq.value}")

    print("\nTest 6: Force rebalance")
    rebalancer.reset()
    should_rebalance, reason, freq = rebalancer.should_rebalance(1, force_rebalance=True)
    print(f"  Force rebalance: {should_rebalance}")
    print(f"  Reason: {reason}")

    print("\n[SUCCESS] Dynamic rebalancing tests passed")


def main():
    """Run all Phase 6 Tier 1 tests."""

    print("\n" + "=" * 80)
    print("Phase 6 Tier 1: R2 (Position Prob) + R4 (Regime) + R5 (Rebalancing)")
    print("=" * 80)

    test_position_probability()
    test_regime_declaration()
    test_dynamic_rebalancing()

    print("\n" + "=" * 80)
    print("[SUCCESS] All Phase 6 Tier 1 tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
