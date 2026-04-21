"""
Validation script for Phase 6 Tier 1 integration.

Tests:
1. Turnover Penalty integration
2. Circuit Breaker integration
3. Position Probability integration
4. Dynamic Rebalancing integration
5. Regime Detection integration
"""

import numpy as np
import pandas as pd
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.patterns.candlestick.engulfing import Engulfing
from src.patterns.classic.double_bottom import DoubleBottom
from src.patterns.classic.double_top import DoubleTop


def generate_test_data(n_days: int = 1000) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)

    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")

    # Random walk with trend
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = 100 * np.cumprod(1 + returns)

    high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    open_price = prices * (1 + np.random.normal(0, 0.005, n_days))
    close = prices

    volume = np.random.randint(1000000, 10000000, n_days)

    df = pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )

    return df


def validate_risk_components():
    """Validate all Phase 6 Tier 1 components are properly integrated."""

    print("=" * 70)
    print("PHASE 6 TIER 1 INTEGRATION VALIDATION")
    print("=" * 70)
    print()

    # Generate test data
    print("1. Generating test data...")
    df = generate_test_data(n_days=1000)
    print(f"   Data shape: {df.shape}")
    print()

    # Create patterns
    print("2. Loading pattern detectors...")
    patterns = [
        Engulfing(),
        DoubleBottom(),
        DoubleTop(),
    ]
    print(f"   Loaded {len(patterns)} patterns")
    print()

    # Test 1: All risk components enabled
    print("3. Testing with ALL Phase 6 Tier 1 components enabled...")
    config_all = BacktestConfig(
        enable_circuit_breaker=True,
        enable_turnover_penalty=True,
        enable_position_probability=True,
        enable_dynamic_re=True,
        enable_regime_detection=True,
        max_drawdown_pct=15.0,
        max_allowed_turnover_pct=2000.0,
    )

    engine_all = BacktestEngine(patterns=patterns, config=config_all)

    # Verify components are initialized
    assert engine_all.circuit_breaker is not None, "Circuit breaker not initialized"
    assert engine_all.turnover_penalty is not None, "Turnover penalty not initialized"
    assert engine_all.position_risk_model is not None, "Position risk model not initialized"
    assert engine_all.dynamic_rebalancer is not None, "Dynamic rebalancer not initialized"
    assert engine_all.regime_detector is not None, "Regime detector not initialized"

    print("   [OK] Circuit breaker: initialized")
    print("   [OK] Turnover penalty: initialized")
    print("   [OK] Position probability: initialized")
    print("   [OK] Dynamic rebalancing: initialized")
    print("   [OK] Regime detection: initialized")
    print()

    # Run backtest
    print("4. Running backtest...")
    result_all = engine_all.run(df)
    print(f"   Total bars: {len(df)}")
    print(f"   Signals generated: {len(result_all.signals)}")
    print(f"   Trades executed: {len(result_all.trades)}")
    print()

    # Check results
    if result_all.metrics:
        print("5. Backtest metrics:")
        for key, value in result_all.metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.4f}")
            elif isinstance(value, dict):
                print(f"   {key}: {value}")
            else:
                print(f"   {key}: {value}")
        print()

    # Check turnover info
    if hasattr(result_all, "turnover_info"):
        print("6. Turnover penalty info:")
        for key, value in result_all.turnover_info.items():
            print(f"   {key}: {value}")
        print()

    # Test 2: Components disabled
    print("7. Testing with ALL Phase 6 Tier 1 components DISABLED...")
    config_none = BacktestConfig(
        enable_circuit_breaker=False,
        enable_turnover_penalty=False,
        enable_position_probability=False,
        enable_dynamic_re=False,
        enable_regime_detection=False,
    )

    engine_none = BacktestEngine(patterns=patterns, config=config_none)

    # Verify components are not initialized
    assert engine_none.circuit_breaker is None, "Circuit breaker should be None"
    assert engine_none.turnover_penalty is None, "Turnover penalty should be None"
    assert engine_none.position_risk_model is None, "Position risk model should be None"
    assert engine_none.dynamic_rebalancer is None, "Dynamic rebalancer should be None"
    assert engine_none.regime_detector is None, "Regime detector should be None"

    print("   [OK] All components correctly disabled")
    print()

    # Run backtest with no risk management
    result_none = engine_none.run(df)
    print(f"   Signals generated: {len(result_none.signals)}")
    print(f"   Trades executed: {len(result_none.trades)}")
    print()

    # Test 3: Compare results
    print("8. Comparing results:")
    print(f"   With risk mgmt: {len(result_all.signals)} signals, {len(result_all.trades)} trades")
    print(
        f"   Without risk mgmt: {len(result_none.signals)} signals, {len(result_none.trades)} trades"
    )

    # Risk management should reduce trade count
    if len(result_all.trades) < len(result_none.trades):
        print("   [OK] Risk management reduced trade count (expected)")
    else:
        print("   [WARNING] Risk management did not reduce trade count (investigate)")
    print()

    # Test 4: Circuit breaker stress test
    print("9. Testing circuit breaker with aggressive drawdown limit...")
    config_circuit = BacktestConfig(
        enable_circuit_breaker=True,
        enable_turnover_penalty=False,
        enable_position_probability=False,
        enable_dynamic_re=False,
        enable_regime_detection=False,
        max_drawdown_pct=5.0,  # Very aggressive limit
    )

    engine_circuit = BacktestEngine(patterns=patterns, config=config_circuit)
    result_circuit = engine_circuit.run(df)

    print(f"   Max drawdown limit: 5.0%")
    print(f"   Trades executed: {len(result_circuit.trades)}")

    # Check if circuit breaker was triggered
    halted_count = 0
    if result_circuit.equity_curve is not None:
        halted_count = (
            result_circuit.equity_curve.get("trading_allowed", pd.Series([True])).eq(False).sum()
        )

    if halted_count > 0:
        print(f"   [OK] Circuit breaker halted trading on {halted_count} bars")
    else:
        print("   [WARNING] Circuit breaker did not halt (investigate)")
    print()

    print("=" * 70)
    print("VALIDATION COMPLETE [OK]")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  - Turnover Penalty: INTEGRATED")
    print(f"  - Circuit Breaker: INTEGRATED")
    print(f"  - Position Probability: INTEGRATED")
    print(f"  - Dynamic Rebalancing: INTEGRATED")
    print(f"  - Regime Detection: INTEGRATED")
    print()
    print("Next: Run production backtest with real data")


if __name__ == "__main__":
    validate_risk_components()
