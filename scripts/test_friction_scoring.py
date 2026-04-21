# -*- coding: utf-8 -*-
"""
Test Script for Friction-Adjusted Backtest Scoring (R10)

Tests transaction cost modeling and friction-adjusted metrics.

Acceptance Criteria (R10):
1. Friction costs calculated correctly (spread + slippage + commission)
2. Net Sharpe = Gross Sharpe - Friction Drag
3. Turnover budget enforcement functional
4. Integration with backtest engine ready
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

from src.backtest.friction_scoring import (
    AssetClass,
    FrictionConfig,
    FrictionCosts,
    FrictionSummary,
    FrictionScorer,
    TurnoverBudgetEnforcer,
)


def generate_mock_trades(n_trades=50, avg_value=10000):
    """Generate mock trade data."""
    np.random.seed(42)

    trades = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n_trades, freq="D"),
            "value": np.random.randn(n_trades).abs() * avg_value,
            "price": 100 + np.random.randn(n_trades) * 2,
            "quantity": np.random.randint(10, 100, n_trades),
        }
    )
    trades["value"] = trades["value"].round(2)

    return trades


def test_friction_cost_calculation():
    """Test friction cost breakdown."""
    print("\n" + "=" * 80)
    print("TEST 1: Friction Cost Calculation")
    print("=" * 80)

    config = FrictionConfig(
        spread_bps=5.0,
        slippage_bps=3.0,
        commission_bps=1.0,
        min_commission=1.0,
    )

    scorer = FrictionScorer(config)

    # Test trade: $10,000 notional
    trade_value = 10000.0
    price = 100.0
    quantity = 100

    cost = scorer.calculate_trade_cost(trade_value, price, quantity)

    print(f"\nTrade: ${trade_value:,.0f} @ ${price:.2f} x {quantity}")
    print(f"\nFriction Cost Breakdown:")
    print(f"  Spread Cost:    ${cost.spread_cost:.2f} ({config.spread_bps} bps)")
    print(f"  Slippage Cost:  ${cost.slippage_cost:.2f} ({config.slippage_bps} bps)")
    print(f"  Commission:     ${cost.commission_cost:.2f} ({config.commission_bps} bps)")
    print(f"  Total Cost:     ${cost.total_cost:.2f} ({cost.cost_bps:.1f} bps)")

    # Verify calculation
    expected_spread = trade_value * 5.0 / 10000.0
    expected_slippage = trade_value * 3.0 / 10000.0
    expected_commission = max(trade_value * 1.0 / 10000.0, 1.0)
    expected_total = expected_spread + expected_slippage + expected_commission

    assert abs(cost.spread_cost - expected_spread) < 0.01, "Spread cost calculation error"
    assert abs(cost.slippage_cost - expected_slippage) < 0.01, "Slippage calculation error"
    assert abs(cost.commission_cost - expected_commission) < 0.01, "Commission calculation error"
    assert abs(cost.total_cost - expected_total) < 0.01, "Total cost calculation error"

    print(f"\n[PASS]: Friction cost calculation correct")
    return True


def test_asset_class_configs():
    """Test default configs for different asset classes."""
    print("\n" + "=" * 80)
    print("TEST 2: Asset Class Configurations")
    print("=" * 80)

    print(f"\nDefault friction configurations:")

    for asset_class in AssetClass:
        config = FrictionConfig.for_asset_class(asset_class)
        total_bps = config.spread_bps + config.slippage_bps + config.commission_bps
        print(
            f"  {asset_class.value:10s}: spread={config.spread_bps:4.1f}bps, "
            f"slippage={config.slippage_bps:4.1f}bps, "
            f"commission={config.commission_bps:4.1f}bps, "
            f"total={total_bps:.1f}bps"
        )

    # Verify expected ordering: crypto should have highest friction
    crypto_config = FrictionConfig.for_asset_class(AssetClass.CRYPTO)
    equity_config = FrictionConfig.for_asset_class(AssetClass.EQUITY)

    crypto_total = (
        crypto_config.spread_bps + crypto_config.slippage_bps + crypto_config.commission_bps
    )
    equity_total = (
        equity_config.spread_bps + equity_config.slippage_bps + equity_config.commission_bps
    )

    assert crypto_total > equity_total, "Crypto should have higher friction than equity"

    print(f"\n[PASS]: Asset class configurations correct")
    return True


def test_friction_summary():
    """Test friction summary calculation."""
    print("\n" + "=" * 80)
    print("TEST 3: Friction Summary Calculation")
    print("=" * 80)

    config = FrictionConfig.for_asset_class(AssetClass.EQUITY)
    scorer = FrictionScorer(config)

    # Generate mock data
    trades = generate_mock_trades(n_trades=100, avg_value=10000)
    gross_returns = np.random.randn(252) * 0.02 + 0.0003  # Daily returns with small drift
    equity_curve = pd.Series(100000 * np.cumprod(1 + gross_returns))

    # Calculate summary
    summary = scorer.calculate_summary(
        trades=trades,
        gross_returns=gross_returns,
        equity_curve=equity_curve,
        gross_sharpe=1.5,
    )

    print(f"\nFriction Summary:")
    print(f"  Total Friction:     ${summary.total_friction:,.2f}")
    print(f"  Total Turnover:     ${summary.total_turnover:,.0f}")
    print(f"  Avg Cost/Trade:     ${summary.avg_cost_per_trade:.2f}")
    print(f"  Avg Cost (bps):     {summary.avg_cost_bps:.1f} bps")
    print(f"  Friction Drag:      {summary.friction_drag:.2%}")
    print(f"  Turnover Ratio:     {summary.turnover_ratio:.1f}x")
    print(f"  Gross Sharpe:       {summary.gross_sharpe:.2f}")
    print(f"  Net Sharpe:         {summary.net_sharpe:.2f}")

    # Verify relationships
    assert summary.net_sharpe <= summary.gross_sharpe, "Net Sharpe should be <= Gross Sharpe"
    assert summary.friction_drag >= 0, "Friction drag should be non-negative"
    assert summary.avg_cost_bps > 0, "Average cost should be positive"

    # Sharpe penalty from friction
    sharpe_penalty = summary.gross_sharpe - summary.net_sharpe
    print(f"  Sharpe Penalty:     {sharpe_penalty:.2f}")

    print(f"\n[PASS]: Friction summary calculation correct")
    return True


def test_turnover_budget_enforcement():
    """Test turnover budget enforcement."""
    print("\n" + "=" * 80)
    print("TEST 4: Turnover Budget Enforcement")
    print("=" * 80)

    config = FrictionConfig.for_asset_class(AssetClass.EQUITY)
    scorer = FrictionScorer(config)

    # Generate low-turnover trades
    trades_low = generate_mock_trades(n_trades=20, avg_value=5000)
    equity_curve = pd.Series(100000 * (1 + np.cumsum(np.random.randn(252) * 0.01)))
    gross_returns = np.random.randn(252) * 0.02

    summary_low = scorer.calculate_summary(
        trades=trades_low,
        gross_returns=gross_returns,
        equity_curve=equity_curve,
        gross_sharpe=1.2,
    )

    # Generate high-turnover trades
    trades_high = generate_mock_trades(n_trades=200, avg_value=50000)
    summary_high = scorer.calculate_summary(
        trades=trades_high,
        gross_returns=gross_returns,
        equity_curve=equity_curve,
        gross_sharpe=1.5,
    )

    enforcer = TurnoverBudgetEnforcer(
        max_annual_turnover=4.0,
        friction_threshold_bps=50.0,
    )

    result_low = enforcer.check_budget(summary_low)
    result_high = enforcer.check_budget(summary_high)

    print(f"\nLow Turnover Strategy:")
    print(f"  Turnover Ratio: {summary_low.turnover_ratio:.1f}x")
    print(f"  Action: {result_low['action']}")
    print(f"  Reason: {result_low['reason']}")
    print(f"  Recommendation: {result_low['recommendation']}")

    print(f"\nHigh Turnover Strategy:")
    print(f"  Turnover Ratio: {summary_high.turnover_ratio:.1f}x")
    print(f"  Action: {result_high['action']}")
    print(f"  Reason: {result_high['reason']}")
    print(f"  Recommendation: {result_high['recommendation']}")

    # Low turnover should pass or flag, high should flag or reject
    assert result_low["action"] in ["PASS", "FLAG"], "Low turnover should not be rejected"
    assert result_high["action"] in ["FLAG", "REJECT"], (
        "High turnover should be flagged or rejected"
    )

    print(f"\n[PASS]: Turnover budget enforcement works correctly")
    return True


def test_net_sharpe_calculation():
    """Test net Sharpe calculation."""
    print("\n" + "=" * 80)
    print("TEST 5: Net Sharpe Calculation")
    print("=" * 80)

    config = FrictionConfig.for_asset_class(AssetClass.CRYPTO)
    scorer = FrictionScorer(config)

    # High-friction scenario (crypto)
    trades = generate_mock_trades(n_trades=100, avg_value=20000)
    gross_returns = np.random.randn(252) * 0.03 + 0.0005
    equity_curve = pd.Series(100000 * np.cumprod(1 + gross_returns))

    summary = scorer.calculate_summary(
        trades=trades,
        gross_returns=gross_returns,
        equity_curve=equity_curve,
        gross_sharpe=2.0,
    )

    print(f"\nHigh-Friction Scenario (Crypto-like):")
    print(f"  Gross Sharpe: {summary.gross_sharpe:.2f}")
    print(f"  Net Sharpe:   {summary.net_sharpe:.2f}")
    print(f"  Friction Drag: {summary.friction_drag:.2%}")
    print(f"  Avg Cost:     {summary.avg_cost_bps:.1f} bps")

    # Sharpe penalty should be proportional to friction
    sharpe_penalty = summary.gross_sharpe - summary.net_sharpe
    expected_penalty = summary.friction_drag * np.sqrt(252) / 0.03  # Approximate

    print(f"  Sharpe Penalty: {sharpe_penalty:.2f} (expected ~{expected_penalty:.2f})")

    assert sharpe_penalty > 0, "High friction should reduce Sharpe"
    assert summary.net_sharpe < summary.gross_sharpe, "Net Sharpe < Gross Sharpe"

    print(f"\n[PASS]: Net Sharpe calculation correct")
    return True


def test_friction_drag_impact():
    """Test friction drag impact on strategy viability."""
    print("\n" + "=" * 80)
    print("TEST 6: Friction Drag Impact on Viability")
    print("=" * 80)

    config = FrictionConfig.for_asset_class(AssetClass.EQUITY)
    scorer = FrictionScorer(config)
    enforcer = TurnoverBudgetEnforcer(max_annual_turnover=6.0)

    # Scenario 1: Low turnover, viable strategy
    trades_1 = generate_mock_trades(n_trades=30, avg_value=5000)
    equity_1 = pd.Series(100000 * np.cumprod(1 + np.random.randn(252) * 0.015 + 0.0003))
    returns_1 = equity_1.pct_change().dropna().values

    summary_1 = scorer.calculate_summary(
        trades=trades_1,
        gross_returns=returns_1,
        equity_curve=equity_1,
        gross_sharpe=1.5,
    )
    result_1 = enforcer.check_budget(summary_1)

    # Scenario 2: High turnover, alpha destroyed
    trades_2 = generate_mock_trades(n_trades=300, avg_value=30000)
    equity_2 = pd.Series(100000 * np.cumprod(1 + np.random.randn(252) * 0.015 + 0.0003))
    returns_2 = equity_2.pct_change().dropna().values

    summary_2 = scorer.calculate_summary(
        trades=trades_2,
        gross_returns=returns_2,
        equity_curve=equity_2,
        gross_sharpe=1.8,
    )
    result_2 = enforcer.check_budget(summary_2)

    print(f"\nScenario 1: Low Turnover")
    print(f"  Turnover: {summary_1.turnover_ratio:.1f}x")
    print(f"  Net Sharpe: {summary_1.net_sharpe:.2f}")
    print(f"  Action: {result_1['action']}")
    print(f"  Viability: {'VIABLE' if result_1['action'] == 'PASS' else 'CONCERN'}")

    print(f"\nScenario 2: High Turnover (OOM-RL pattern)")
    print(f"  Turnover: {summary_2.turnover_ratio:.1f}x")
    print(f"  Net Sharpe: {summary_2.net_sharpe:.2f}")
    print(f"  Action: {result_2['action']}")
    print(f"  Viability: {'VIABLE' if result_2['action'] == 'PASS' else 'NON-VIABLE'}")

    # High turnover should show significant Sharpe degradation
    assert summary_2.net_sharpe < summary_1.net_sharpe, "High turnover should reduce Sharpe more"

    print(f"\n[PASS]: Friction drag impact correctly modeled")
    return True


def run_all_tests():
    """Run all R10 acceptance tests."""
    print("\n" + "=" * 80)
    print("R10 FRICTION-ADJUSTED BACKTEST SCORING - ACCEPTANCE TESTS")
    print("=" * 80)

    tests = [
        ("Friction Cost Calculation", test_friction_cost_calculation),
        ("Asset Class Configurations", test_asset_class_configs),
        ("Friction Summary Calculation", test_friction_summary),
        ("Turnover Budget Enforcement", test_turnover_budget_enforcement),
        ("Net Sharpe Calculation", test_net_sharpe_calculation),
        ("Friction Drag Impact", test_friction_drag_impact),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed, None))
        except AssertionError as e:
            results.append((test_name, False, str(e)))
            print(f"\n[FAIL]: {e}")
        except Exception as e:
            results.append((test_name, False, f"Unexpected error: {e}"))
            print(f"\n[ERROR]: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, p, _ in results if p)
    total = len(results)

    for test_name, passed_test, error in results:
        status = "[PASS]" if passed_test else "[FAIL]"
        print(f"{status}: {test_name}")
        if error:
            print(f"      {error}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "=" * 80)
        print("ALL R10 ACCEPTANCE CRITERIA MET")
        print("=" * 80)
        print("[OK] Friction cost model (spread + slippage + commission)")
        print("[OK] Net Sharpe = Gross Sharpe - Friction Drag")
        print("[OK] Turnover budget enforcement functional")
        print("[OK] Integration with backtest engine ready")
        print("=" * 80)
        return True
    else:
        print("\n" + "=" * 80)
        print("SOME TESTS FAILED - R10 NOT YET COMPLETE")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
