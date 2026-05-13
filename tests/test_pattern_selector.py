"""
Test Script for Pattern Selection Framework

This script tests the pattern selection framework with a subset of patterns
to verify the implementation works correctly.

Usage:
    python scripts/test_pattern_selector.py
    python scripts/test_pattern_selector.py --quick  # Quick test with 5 patterns
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd

from src.utils.notebook_helpers import PatternSelectionConfig
from src.analysis.pattern_selector import PatternSelector
from src.analysis.pattern_selector_viz import generate_all_visualizations


def test_pattern_selector():
    """Test the pattern selection framework."""
    print("=" * 60)
    print("[TEST] Pattern Selection Framework Test")
    print("=" * 60)

    # Load SPY data
    data_path = "data/raw/SPY_daily.csv"
    if not os.path.exists(data_path):
        print(f"[ERROR] Data file not found at {data_path}")
        print("Please ensure SPY_daily.csv is in the data/raw/ directory")
        return None

    print(f"\n[DATA] Loading data from: {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"   Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Create configuration with relaxed thresholds for testing
    config = PatternSelectionConfig(
        data_path=data_path,
        output_dir="reports/pattern_selection_test",
        # Relaxed thresholds for testing (to ensure some patterns pass)
        min_trades=10,  # Lower threshold
        min_sharpe=0.0,  # Accept any positive Sharpe
        min_profit_factor=0.8,  # Slightly profitable
        min_win_rate=0.30,  # 30% win rate
        max_drawdown=0.50,  # Allow higher drawdown
        correlation_threshold=0.8,
        cache_results=True,
    )

    print("\n[CONFIG] Configuration:")
    print(f"   Min Trades: {config.min_trades}")
    print(f"   Min Sharpe: {config.min_sharpe}")
    print(f"   Min Profit Factor: {config.min_profit_factor}")
    print(f"   Min Win Rate: {config.min_win_rate:.0%}")
    print(f"   Max Drawdown: {config.max_drawdown:.0%}")
    print(f"   Correlation Threshold: {config.correlation_threshold}")

    # Validate configuration
    errors = config.validate()
    if errors:
        print("\n[ERROR] Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        return None

    # Initialize selector
    print("\n[INIT] Initializing Pattern Selector...")
    selector = PatternSelector(config, data=df)
    print(f"   Found {len(selector.all_patterns)} patterns to test")

    # Run full selection
    print("\n[RUN] Starting Pattern Selection Process...")
    result = selector.run_full_selection()

    # Generate visualizations
    print("\n" + "=" * 60)
    print("[VIZ] Generating Visualizations and Report...")
    print("=" * 60)

    viz_outputs = generate_all_visualizations(result, config.output_dir)

    print("\n[OUTPUT] Generated outputs:")
    for name, path in viz_outputs.items():
        if path:
            print(f"   [OK] {name}: {path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("[RESULTS] FINAL RESULTS")
    print("=" * 60)

    summary = result.get_summary()
    print("\n[SUMMARY] Summary:")
    print(f"   Total patterns tested: {summary['total_patterns_tested']}")
    print(f"   Passed Phase 1 filter: {summary['passed_phase1_filter']}")
    print(f"   Excluded (Phase 1): {summary['excluded_phase1']}")
    print(f"   Redundant pairs found: {summary['redundant_pairs_found']}")
    print(f"   After redundancy removal: {summary['after_redundancy_removal']}")
    print(f"   Final selected patterns: {summary['final_selected_patterns']}")
    print(f"   Execution time: {summary['execution_time_seconds']:.1f}s")

    # Phase timings
    if result.phase_timings:
        print("\n[TIMING] Phase Timings:")
        for phase, timing in result.phase_timings.items():
            print(f"   {phase.upper()}: {timing:.1f}s")

    # Role distribution
    role_dist = result.get_role_distribution()
    if role_dist:
        print("\n[ROLE] Role Distribution:")
        for role, count in role_dist.items():
            print(f"   {role}: {count}")

    if result.final_patterns:
        print(f"\n[FINAL] Final Selected Patterns ({len(result.final_patterns)}):")
        for i, pattern in enumerate(result.final_patterns, 1):
            c = next((c for c in result.ablation_contributions if c.pattern_name == pattern), None)
            if c:
                role_name = c.role.value if hasattr(c.role, "value") else c.role
                print(f"   {i}. {pattern} ({role_name}, delta_sharpe={c.delta_sharpe:.4f})")
            else:
                print(f"   {i}. {pattern}")
    else:
        print("\n[WARN] No patterns passed selection criteria.")
        print("   Consider adjusting thresholds in the configuration.")

    return result


def test_quick_subset():
    """
    Quick test with a subset of patterns for faster verification.
    Uses only 5 patterns to verify the framework works.
    """
    print("=" * 60)
    print("[QUICK] Quick Subset Test (5 patterns)")
    print("=" * 60)

    data_path = "data/raw/SPY_daily.csv"
    if not os.path.exists(data_path):
        print(f"[ERROR] Data file not found at {data_path}")
        return None

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Use only last 2 years of data for faster testing
    cutoff_date = df.index.max() - pd.DateOffset(years=2)
    df = df[df.index >= cutoff_date]
    print(f"[DATA] Using last 2 years of data: {len(df)} bars")

    # Configuration for quick test
    config = PatternSelectionConfig(
        data_path=data_path,
        output_dir="reports/pattern_selection_quick",
        min_trades=5,
        min_sharpe=-1.0,  # Accept all
        min_profit_factor=0.0,
        min_win_rate=0.0,
        max_drawdown=1.0,
        correlation_threshold=0.8,
        cache_results=True,
    )

    selector = PatternSelector(config, data=df)

    # Override all_patterns with subset for quick test
    if len(selector.all_patterns) > 5:
        selector.all_patterns = selector.all_patterns[:5]
        print(f"[RUN] Testing with subset of {len(selector.all_patterns)} patterns")

    result = selector.run_full_selection()

    # Generate visualizations
    generate_all_visualizations(result, config.output_dir)

    print("\n[DONE] Quick test completed!")
    print(f"   Final patterns: {result.final_patterns}")

    return result


def test_visualization_functions():
    """Test individual visualization functions."""
    print("=" * 60)
    print("[TEST] Testing Visualization Functions")
    print("=" * 60)

    # Import visualization functions
    from src.analysis.pattern_selector_viz import (
        plot_correlation_heatmap,
    )

    print("\n[OK] All visualization functions imported successfully")

    # Create dummy data for testing

    # Test correlation heatmap
    print("\n[VIZ] Testing correlation heatmap...")
    corr_matrix = pd.DataFrame(
        {"Pattern_A": [1.0, 0.5, 0.3], "Pattern_B": [0.5, 1.0, 0.8], "Pattern_C": [0.3, 0.8, 1.0]},
        index=["Pattern_A", "Pattern_B", "Pattern_C"],
    )

    output_dir = "reports/pattern_selection_test"
    os.makedirs(output_dir, exist_ok=True)

    result = plot_correlation_heatmap(
        corr_matrix,
        os.path.join(output_dir, "test_heatmap.png"),
    )
    if result:
        print(f"   [OK] Heatmap saved: {result}")
    else:
        print("   [WARN] Heatmap generation skipped (matplotlib not available)")

    print("\n[DONE] Visualization function tests complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Pattern Selection Framework")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick test with subset of patterns"
    )
    parser.add_argument("--viz-only", action="store_true", help="Test visualization functions only")
    args = parser.parse_args()

    if args.viz_only:
        test_visualization_functions()
    elif args.quick:
        test_quick_subset()
    else:
        test_pattern_selector()
