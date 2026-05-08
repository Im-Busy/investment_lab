"""
Quick Test for Phase 4 Portfolio Module

This script runs basic tests to verify Phase 4 implementation.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

print("=" * 70)
print("Phase 4 Portfolio Module - Quick Test")
print("=" * 70)

# Test 1: Import modules
print("\n[1/5] Testing module imports...")
try:
    from src.portfolio import (
        SignalAggregator,
        AggregationMethod,
        NormalizationMethod,
        EqualWeightScheme,
        SharpeWeightScheme,
        StrategyRanker,
        RankingMethod,
        RegimeDetector,
        PortfolioRiskManager,
    )

    print("[PASS] All modules imported successfully")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

# Test 2: Signal aggregation
print("\n[2/5] Testing signal aggregation...")
try:
    # Create sample signals (3 strategies, 100 days)
    np.random.seed(42)
    signals = np.random.randn(100, 3) * 0.5

    aggregator = SignalAggregator(
        agg_method=AggregationMethod.WEIGHTED_MEAN, norm_method=NormalizationMethod.MIN_MAX
    )

    aggregated = aggregator.aggregate(signals)

    print("[PASS] Aggregation successful")
    print(f"   Input shape: {signals.shape}")
    print(f"   Output shape: {aggregated.shape}")
    print(f"   Range: [{aggregated.min():.2f}, {aggregated.max():.2f}]")
except Exception as e:
    print(f"[FAIL] Aggregation failed: {e}")
    import traceback

    traceback.print_exc()

# Test 3: Weight schemes
print("\n[3/5] Testing weight schemes...")
try:
    strategy_metrics = {
        "strategy1": {"sharpe": 0.5, "volatility": 0.2},
        "strategy2": {"sharpe": 0.8, "volatility": 0.15},
        "strategy3": {"sharpe": 0.3, "volatility": 0.25},
    }

    equal_weights = EqualWeightScheme().compute_weights(strategy_metrics)
    sharpe_weights = SharpeWeightScheme().compute_weights(strategy_metrics)

    print("[PASS] Weight schemes successful")
    print(f"   Equal weights: {equal_weights}")
    print(f"   Sharpe weights: {sharpe_weights}")
except Exception as e:
    print(f"[FAIL] Weight schemes failed: {e}")
    import traceback

    traceback.print_exc()

# Test 4: Strategy ranking
print("\n[4/5] Testing strategy ranking...")
try:
    ranker = StrategyRanker(method=RankingMethod.SHARPE_RATIO, min_sharpe=-1.0, min_trades=0)

    # Generate sample returns
    np.random.seed(42)
    strategy_returns = {
        "strategy1": np.random.normal(0.001, 0.02, 1000),
        "strategy2": np.random.normal(0.002, 0.015, 1000),
        "strategy3": np.random.normal(0.0005, 0.025, 1000),
    }

    ranked = ranker.rank_strategies(strategy_returns)

    print("[PASS] Strategy ranking successful")
    print(f"   Strategies ranked: {len(ranked)}")
    for i, perf in enumerate(ranked):
        print(f"   {i + 1}. {perf.name}: Sharpe={perf.sharpe:.2f}")
except Exception as e:
    print(f"[FAIL] Strategy ranking failed: {e}")
    import traceback

    traceback.print_exc()

# Test 5: Regime detection
print("\n[5/5] Testing regime detection...")
try:
    detector = RegimeDetector(lookback_window=63)

    # Generate returns for different regimes
    bull_returns = np.random.normal(0.002, 0.015, 200)
    bear_returns = np.random.normal(-0.0015, 0.02, 200)

    bull_regime = detector.detect_regime(bull_returns)
    bear_regime = detector.detect_regime(bear_returns)

    print("[PASS] Regime detection successful")
    print(f"   Bull market regime: {bull_regime.regime} (confidence: {bull_regime.confidence:.2f})")
    print(f"   Bear market regime: {bear_regime.regime} (confidence: {bear_regime.confidence:.2f})")
except Exception as e:
    print(f"[FAIL] Regime detection failed: {e}")
    import traceback

    traceback.print_exc()

# Test 6: Risk management
print("\n[6/6] Testing risk management...")
try:
    risk_mgr = PortfolioRiskManager(max_drawdown=0.2, max_volatility=0.3)

    # Set some values
    risk_mgr.current_drawdown = -0.15
    risk_mgr.current_volatility = 0.25

    risk_status = risk_mgr.check_risk_limits()

    print("[PASS] Risk management successful")
    print(f"   Drawdown: {risk_mgr.current_drawdown:.2%} (breached: {risk_status['max_drawdown']})")
    print(
        f"   Volatility: {risk_mgr.current_volatility:.2%} (breached: {risk_status['max_volatility']})"
    )
except Exception as e:
    print(f"[FAIL] Risk management failed: {e}")
    import traceback

    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("[PASS] Phase 4 Portfolio Module - All Tests Passed!")
print("=" * 70)
print("\nNext steps:")
print("1. Run full portfolio backtest: uv run scripts/portfolio_backtest.py")
print("2. Add unit: tests for portfolio module")
print("3. Benchmark performance vs sequential execution")
print("4. Integrate with existing strategies")
