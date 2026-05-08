"""
Test Reverse Signal Hypothesis

This script tests the hypothesis: "If strategies are unprofitable, can we flip
signals (short when they say buy, buy when they say sell) and be profitable?"

Methodology:
1. Run original strategy backtest (baseline)
2. Run reverse strategy (flip all signals)
3. Compare results

Key insight: If signals have NO edge (random), flipping also has no edge.
             If signals have NEGATIVE edge (consistently wrong), flipping works.

Expected outcomes:
- If Sharpe_original ≈ 0 → Sharpe_reverse ≈ 0 (no edge)
- If Sharpe_original < -0.5 consistently → Sharpe_reverse > 0 (negative edge)

The results will definitively answer whether signal inversion is profitable.
"""

import sys
from pathlib import Path

import pandas as pd
from backtesting import Backtest

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the optimized multi-pattern strategy
from src.strategies.backtest_py.multi_pattern_strategy_optimized import (
    MultiPatternStrategyOptimized,
)


class ReverseSignalStrategy(MultiPatternStrategyOptimized):
    """
    Strategy that FLIPS all signals from MultiPatternStrategyOptimized.

    When original says LONG → We enter SHORT
    When original says SHORT → We enter LONG

    This tests whether the original signals are:
    - Random (no edge) → Flipped also has no edge
    - Consistently wrong (negative edge) → Flipped becomes profitable
    """

    # Override the name for reporting
    name = "Reverse Multi-Pattern Strategy"

    def next(self):
        """Execute REVERSE trading logic - flip all signals."""
        # Check position limit
        if len(self.trades) >= self.max_open_positions:
            return

        # Use pre-computed min_bars
        current_idx = len(self.data.Close) - 1
        if current_idx < self._min_bars:
            return

        # Calculate window start for pattern detectors
        window_start = max(0, current_idx - 200)

        # Detect patterns using parent's method
        signals = self._detect_patterns_sequential(self._df, current_idx, window_start)

        # Check if we have enough confluence
        if len(signals) < self.min_confluence_count:
            return

        # Group signals by direction
        long_signals = [s for s in signals if s["direction"] == "Long"]
        short_signals = [s for s in signals if s["direction"] == "Short"]

        # *** REVERSE LOGIC: Flip the direction ***
        # If original would go LONG → We go SHORT
        # If original would go SHORT → We go LONG

        if (
            len(long_signals) >= len(short_signals)
            and len(long_signals) >= self.min_confluence_count
        ):
            # Original says LONG → We REVERSE to SHORT
            active_signals = long_signals
            direction = "SHORT"  # FLIPPED!
        elif len(short_signals) >= self.min_confluence_count:
            # Original says SHORT → We REVERSE to LONG
            active_signals = short_signals
            direction = "LONG"  # FLIPPED!
        else:
            return

        # Calculate confluence score (same as original)
        avg_confidence = sum(s["confidence"] for s in active_signals) / len(active_signals)

        if avg_confidence < self.min_confidence:
            return

        # Calculate entry and stop (REVERSED logic for stop placement)
        entry_price = self.data.Close[-1]

        # Use weighted average of signal levels
        total_confidence = sum(s["confidence"] for s in active_signals)
        if total_confidence > 0:
            original_stop = (
                sum(s["stop_loss"] * s["confidence"] for s in active_signals) / total_confidence
            )
        else:
            original_stop = active_signals[0]["stop_loss"]

        # *** REVERSE STOP LOGIC ***
        # For LONG signals flipped to SHORT: stop should be ABOVE entry (original was below)
        # For SHORT signals flipped to LONG: stop should be BELOW entry (original was above)

        if direction == "SHORT":
            # We're shorting, stop should be above entry
            # Original stop was below entry (for long), so flip it above entry
            stop_distance = abs(entry_price - original_stop)
            stop_loss = entry_price + stop_distance  # Above entry for short
        else:  # LONG
            # We're going long, stop should be below entry
            # Original stop was above entry (for short), so flip it below entry
            stop_distance = abs(entry_price - original_stop)
            stop_loss = entry_price - stop_distance  # Below entry for long

        # Calculate position size based on risk
        equity = self.equity
        risk_amount = equity * self.risk_per_trade
        stop_distance = abs(entry_price - stop_loss)

        if stop_distance <= 0:
            return

        position_size_frac = (risk_amount / stop_distance) / entry_price
        position_size_frac = min(position_size_frac, 0.20)

        if position_size_frac <= 0 or position_size_frac > 1:
            return

        # Execute REVERSE trade
        if direction == "LONG":
            if any(t.is_long for t in self.trades):
                return
            self.buy(size=position_size_frac, sl=stop_loss)
        else:  # SHORT
            if any(t.is_short for t in self.trades):
                return
            self.sell(size=position_size_frac, sl=stop_loss)

        # Track reversed signals
        self.signals.append(
            {
                "timestamp": self.data.index[current_idx],
                "original_direction": "LONG" if direction == "SHORT" else "SHORT",
                "reversed_direction": direction,
                "confidence": avg_confidence,
                "pattern_count": len(active_signals),
                "patterns": [s["pattern_name"] for s in active_signals],
            }
        )
        self.signal_count += 1


def load_data() -> pd.DataFrame:
    """Load SPY daily data."""
    data_path = project_root / "data" / "raw" / "SPY_daily.csv"
    df = pd.read_csv(
        data_path,
        parse_dates=["Date"],
        index_col="Date",
    )
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={c: c.replace(" ", "") for c in df.columns})

    # Normalize to backtesting.py expected format
    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            if col == "Volume":
                df["Volume"] = 0
            else:
                raise ValueError(f"Missing required column: {col}")

    return df


def run_comparison():
    """Run both original and reverse backtests, compare results."""
    print("=" * 60)
    print("REVERSE SIGNAL HYPOTHESIS TEST")
    print("=" * 60)
    print("\nTesting: Can we profit by flipping unprofitable signals?")
    print("If signals are RANDOM -> Flipped signals also have no edge")
    print("If signals are CONSISTENTLY WRONG -> Flipped signals become profitable")
    print("=" * 60)

    df = load_data()

    print(f"\nData: SPY Daily ({df.index.min().date()} to {df.index.max().date()})")
    print(f"Bars: {len(df)}")

    # Run ORIGINAL strategy
    print("\n" + "-" * 60)
    print("RUNNING ORIGINAL STRATEGY...")
    print("-" * 60)

    bt_original = Backtest(
        df,
        MultiPatternStrategyOptimized,
        cash=1_000_000,
        commission=0.001,
        exclusive_orders=True,
    )
    stats_original = bt_original.run()

    original_results = {
        "trades": int(stats_original.get("# Trades", 0) or 0),
        "win_rate": float(stats_original.get("Win Rate [%]", 0)),
        "sharpe": float(stats_original.get("Sharpe Ratio", -999)),
        "profit_factor": float(stats_original.get("Profit Factor", 0)),
        "return_pct": float(stats_original.get("Return [%]", 0)),
        "max_dd": float(stats_original.get("Max. Drawdown [%]", 0)),
        "bh_return": float(stats_original.get("Buy & Hold Return [%]", 0)),
    }

    print("\nORIGINAL RESULTS:")
    print(f"  Trades:      {original_results['trades']}")
    print(f"  Win Rate:    {original_results['win_rate']:.1f}%")
    print(f"  Sharpe:      {original_results['sharpe']:.2f}")
    print(f"  Profit Factor: {original_results['profit_factor']:.2f}")
    print(f"  Return:      {original_results['return_pct']:.1f}%")
    print(f"  Max DD:      {original_results['max_dd']:.1f}%")
    print(f"  B&H Return:  {original_results['bh_return']:.1f}%")

    # Run REVERSE strategy
    print("\n" + "-" * 60)
    print("RUNNING REVERSE STRATEGY (FLIPPED SIGNALS)...")
    print("-" * 60)

    bt_reverse = Backtest(
        df,
        ReverseSignalStrategy,
        cash=1_000_000,
        commission=0.001,
        exclusive_orders=True,
    )
    stats_reverse = bt_reverse.run()

    reverse_results = {
        "trades": int(stats_reverse.get("# Trades", 0) or 0),
        "win_rate": float(stats_reverse.get("Win Rate [%]", 0)),
        "sharpe": float(stats_reverse.get("Sharpe Ratio", -999)),
        "profit_factor": float(stats_reverse.get("Profit Factor", 0)),
        "return_pct": float(stats_reverse.get("Return [%]", 0)),
        "max_dd": float(stats_reverse.get("Max. Drawdown [%]", 0)),
        "bh_return": float(stats_reverse.get("Buy & Hold Return [%]", 0)),
    }

    print("\nREVERSE RESULTS:")
    print(f"  Trades:      {reverse_results['trades']}")
    print(f"  Win Rate:    {reverse_results['win_rate']:.1f}%")
    print(f"  Sharpe:      {reverse_results['sharpe']:.2f}")
    print(f"  Profit Factor: {reverse_results['profit_factor']:.2f}")
    print(f"  Return:      {reverse_results['return_pct']:.1f}%")
    print(f"  Max DD:      {reverse_results['max_dd']:.1f}%")
    print(f"  B&H Return:  {reverse_results['bh_return']:.1f}%")

    # COMPARISON TABLE
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"\n{'Metric':<20} {'Original':>12} {'Reverse':>12} {'Delta':>12}")
    print("-" * 56)

    metrics = [
        ("Trades", original_results["trades"], reverse_results["trades"]),
        ("Win Rate %", original_results["win_rate"], reverse_results["win_rate"]),
        ("Sharpe", original_results["sharpe"], reverse_results["sharpe"]),
        ("Profit Factor", original_results["profit_factor"], reverse_results["profit_factor"]),
        ("Return %", original_results["return_pct"], reverse_results["return_pct"]),
        ("Max DD %", original_results["max_dd"], reverse_results["max_dd"]),
    ]

    for name, orig_val, rev_val in metrics:
        if isinstance(orig_val, int):
            delta = rev_val - orig_val
            print(f"{name:<20} {orig_val:>12} {rev_val:>12} {delta:>12}")
        else:
            delta = rev_val - orig_val
            print(f"{name:<20} {orig_val:>12.1f} {rev_val:>12.1f} {delta:>12.1f}")

    # INTERPRETATION
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    orig_sharpe = original_results["sharpe"]
    rev_sharpe = reverse_results["sharpe"]

    # Determine signal quality
    print("\nSignal Edge Analysis:")

    if orig_sharpe > 0.5:
        print(f"  ✓ Original has POSITIVE edge (Sharpe = {orig_sharpe:.2f})")
        print("  → Flipping would DESTROY profitable edge")
        print("  → Recommendation: KEEP original signals, DO NOT flip")
    elif orig_sharpe < -0.5:
        print(f"  ⚠ Original has NEGATIVE edge (Sharpe = {orig_sharpe:.2f})")
        if rev_sharpe > 0:
            print(f"  ✓ Reverse is PROFITABLE (Sharpe = {rev_sharpe:.2f})")
            print("  → Signals are CONSISTENTLY WRONG")
            print("  → Recommendation: FLIP signals could work (but verify costs)")
        else:
            print(f"  ✗ Reverse STILL unprofitable (Sharpe = {rev_sharpe:.2f})")
            print("  → Short execution costs may exceed any reversal gain")
            print("  → Recommendation: AVOID both strategies")
    else:
        print(f"  — Original has NO edge (Sharpe = {orig_sharpe:.2f})")
        print(f"  — Reverse has NO edge (Sharpe = {rev_sharpe:.2f})")
        print("  → Signals are RANDOM, not consistently wrong")
        print("  → Recommendation: AVOID both strategies, focus on signal improvement")

    # Additional analysis
    print("\nTrade Analysis:")

    if original_results["trades"] > 0:
        orig_pf = original_results["profit_factor"]
        rev_pf = reverse_results["profit_factor"]

        if orig_pf < 1.0 and rev_pf > 1.0:
            print(f"  ✓ PF: {orig_pf:.2f} → {rev_pf:.2f} (improvement)")
            print("  → Reversing improved win rate or average win/loss ratio")
        elif orig_pf < 1.0 and rev_pf < 1.0:
            print(f"  ✗ PF: {orig_pf:.2f} → {rev_pf:.2f} (still losing)")
            print("  → Neither direction has edge")
        else:
            print(f"  PF: {orig_pf:.2f} → {rev_pf:.2f}")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if orig_sharpe < -0.5 and rev_sharpe > 0.3:
        print("\n✓ HYPOTHESIS CONFIRMED: Signals have negative edge.")
        print("  Flipping signals could be profitable.")
        print("  BUT: Consider short selling costs (borrow fees, dividends).")
        print("  AND: Test on multiple assets/periods before trading.")
    elif abs(orig_sharpe) < 0.3 and abs(rev_sharpe) < 0.3:
        print("\n✗ HYPOTHESIS REJECTED: Signals have NO edge (random).")
        print("  Flipping random signals produces random results.")
        print("  Focus on: improving signal quality, regime filtering,")
        print("  or reducing transaction costs instead.")
    else:
        print("\n? INCONCLUSIVE: Mixed results.")
        print("  Test on more assets/time periods for definitive answer.")

    print("\n" + "=" * 60)

    return {
        "original": original_results,
        "reverse": reverse_results,
    }


if __name__ == "__main__":
    results = run_comparison()
