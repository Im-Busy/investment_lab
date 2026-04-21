"""
Test Reverse Signal Hypothesis - CORRECTED VERSION

Properly processes all pre-computed signals by converting them to a time series
that backtesting.py can use efficiently.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


SIGNAL_PATH = project_root / "output" / "data" / "signals.parquet"


class SignalBasedStrategy(Strategy):
    """
    Strategy that uses pre-computed signal direction.

    Parameters:
        reverse: If True, flip all signals (LONG->SHORT, SHORT->LONG)
    """

    # Class-level config (set before run)
    reverse = False

    def init(self):
        # Get the signals DataFrame
        if not hasattr(self, "_signals_df"):
            if SIGNAL_PATH.exists():
                self._signals_df = pd.read_parquet(SIGNAL_PATH)
                self._signals_df = self._signals_df.set_index("timestamp")
            else:
                self._signals_df = None

        # Track last processed signal index to avoid re-processing
        self._last_signal_idx = -1

        mode = "REVERSED" if self.reverse else "ORIGINAL"
        print(f"[{mode}] Strategy initialized")

    def next(self):
        # Skip if not enough bars
        if len(self.data.Close) < 5:
            return

        # Close any existing positions after 5 bars (simple exit rule)
        if len(self.trades) > 0:
            for trade in self.trades:
                # Hold for 5 bars max
                if len(self.data.Close) - self._last_signal_idx > 5:
                    trade.close()
            return

        # Get current date
        current_date = self.data.index[-1]

        # Find the most recent signal before or at current date
        if self._signals_df is None:
            return

        # Get signals up to current date, only look at new signals
        recent_signals = self._signals_df[self._signals_df.index <= current_date]

        if len(recent_signals) == 0:
            return

        # Get the latest signal
        latest_signal = recent_signals.iloc[-1]
        latest_idx = recent_signals.index[-1]

        # Skip if we already processed this signal
        if hasattr(self, "_processed_signals") and latest_idx in self._processed_signals:
            return

        # Track processed signals
        if not hasattr(self, "_processed_signals"):
            self._processed_signals = set()
        self._processed_signals.add(latest_idx)

        direction = latest_signal["direction"]
        confidence = latest_signal.get("confidence", 0.7)

        # Calculate position size based on confidence
        position_size = min(0.95, confidence)

        # Apply reversal if configured
        actual_direction = direction
        if self.reverse:
            actual_direction = "Short" if direction == "Long" else "Long"

        # Execute trade
        if actual_direction == "Long":
            stop_loss = self.data.Close[-1] * 0.97  # 3% stop
            self.buy(size=position_size, sl=stop_loss)
        else:  # Short
            stop_loss = self.data.Close[-1] * 1.03  # 3% stop
            self.sell(size=position_size, sl=stop_loss)

        self._last_signal_idx = len(self.data.Close)


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

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            if col == "Volume":
                df["Volume"] = 0
            else:
                raise ValueError(f"Missing required column: {col}")

    return df


def run_backtest_with_class(df: pd.DataFrame, strategy_class, is_reverse: bool) -> dict:
    """Run backtest with properly configured strategy class."""

    # Create a new class with reverse parameter set
    class ConfiguredStrategy(strategy_class):
        reverse = is_reverse

    ConfiguredStrategy.__name__ = f"{'Reverse' if is_reverse else 'Original'}Strategy"

    bt = Backtest(
        df,
        ConfiguredStrategy,
        cash=100_000,
        commission=0.001,
        exclusive_orders=True,
        finalize_trades=True,
    )

    stats = bt.run()

    return {
        "trades": int(stats.get("# Trades", 0) or 0),
        "win_rate": float(stats.get("Win Rate [%]", 0) or 0),
        "sharpe": float(stats.get("Sharpe Ratio", -999) or -999),
        "profit_factor": float(stats.get("Profit Factor", 0) or 0),
        "return_pct": float(stats.get("Return [%]", 0) or 0),
        "max_dd": float(stats.get("Max. Drawdown [%]", 0) or 0),
        "bh_return": float(stats.get("Buy & Hold Return [%]", 0) or 0),
        "avg_trade": float(stats.get("Avg. Trade [%]", 0) or 0),
    }


def run_comparison():
    """Run both original and reverse backtests."""
    print("=" * 60)
    print("REVERSE SIGNAL HYPOTHESIS TEST")
    print("=" * 60)
    print("\nTesting: Can we profit by flipping signals?")
    print("If signals are RANDOM  -> Flipped also has no edge")
    print("If signals are WRONG   -> Flipped becomes profitable")
    print("=" * 60)

    df = load_data()

    # Load signals for analysis
    if SIGNAL_PATH.exists():
        signals_df = pd.read_parquet(SIGNAL_PATH)
        signals_df = signals_df.set_index("timestamp")

        longs = (signals_df["direction"] == "Long").sum()
        shorts = (signals_df["direction"] == "Short").sum()
        print(f"\nPre-computed signals: {len(signals_df)} total")
        print(f"  Long:  {longs} ({100 * longs / len(signals_df):.1f}%)")
        print(f"  Short: {shorts} ({100 * shorts / len(signals_df):.1f}%)")
        print(f"  Date range: {signals_df.index.min()} to {signals_df.index.max()}")

    print(f"\nData: SPY Daily ({df.index.min().date()} to {df.index.max().date()})")
    print(f"Bars: {len(df)}")

    # Run ORIGINAL strategy
    print("\n" + "-" * 60)
    print("RUNNING ORIGINAL STRATEGY...")
    print("-" * 60)

    original_results = run_backtest_with_class(df, SignalBasedStrategy, is_reverse=False)

    print(f"\nORIGINAL RESULTS:")
    print(f"  Trades:        {original_results['trades']}")
    print(f"  Win Rate:      {original_results['win_rate']:.1f}%")
    print(f"  Sharpe:        {original_results['sharpe']:.2f}")
    print(f"  Profit Factor: {original_results['profit_factor']:.2f}")
    print(f"  Return:        {original_results['return_pct']:.1f}%")
    print(f"  Max DD:        {original_results['max_dd']:.1f}%")
    print(f"  B&H Return:    {original_results['bh_return']:.1f}%")
    print(f"  Avg Trade:     {original_results['avg_trade']:.2f}%")

    # Run REVERSE strategy
    print("\n" + "-" * 60)
    print("RUNNING REVERSE STRATEGY...")
    print("-" * 60)

    reverse_results = run_backtest_with_class(df, SignalBasedStrategy, is_reverse=True)

    print(f"\nREVERSE RESULTS:")
    print(f"  Trades:        {reverse_results['trades']}")
    print(f"  Win Rate:      {reverse_results['win_rate']:.1f}%")
    print(f"  Sharpe:        {reverse_results['sharpe']:.2f}")
    print(f"  Profit Factor: {reverse_results['profit_factor']:.2f}")
    print(f"  Return:        {reverse_results['return_pct']:.1f}%")
    print(f"  Max DD:        {reverse_results['max_dd']:.1f}%")
    print(f"  B&H Return:    {reverse_results['bh_return']:.1f}%")
    print(f"  Avg Trade:     {reverse_results['avg_trade']:.2f}%")

    # COMPARISON TABLE
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"\n{'Metric':<18} {'Original':>12} {'Reverse':>12} {'Delta':>12}")
    print("-" * 54)

    metrics = [
        ("Trades", original_results["trades"], reverse_results["trades"]),
        ("Win Rate %", original_results["win_rate"], reverse_results["win_rate"]),
        ("Sharpe", original_results["sharpe"], reverse_results["sharpe"]),
        ("Profit Factor", original_results["profit_factor"], reverse_results["profit_factor"]),
        ("Return %", original_results["return_pct"], reverse_results["return_pct"]),
        ("Max DD %", original_results["max_dd"], reverse_results["max_dd"]),
        ("Avg Trade %", original_results["avg_trade"], reverse_results["avg_trade"]),
    ]

    for name, orig_val, rev_val in metrics:
        if isinstance(orig_val, int):
            delta = rev_val - orig_val
            print(f"{name:<18} {orig_val:>12} {rev_val:>12} {delta:>12}")
        else:
            delta = rev_val - orig_val
            print(f"{name:<18} {orig_val:>12.1f} {rev_val:>12.1f} {delta:>12.1f}")

    # INTERPRETATION
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    orig_sharpe = original_results["sharpe"]
    rev_sharpe = reverse_results["sharpe"]
    orig_pf = original_results["profit_factor"]
    rev_pf = reverse_results["profit_factor"]

    print("\nSignal Edge Analysis:")

    # Determine if reversal would help
    reversal_improves = rev_sharpe > orig_sharpe and rev_pf > orig_pf

    if orig_sharpe > 0.5:
        print(f"  [+] Original has POSITIVE edge (Sharpe = {orig_sharpe:.2f})")
        print("  -> DO NOT flip - would destroy profitable edge")
        conclusion = "ORIGINAL strategy is profitable - keep as-is"
    elif orig_sharpe < -0.5:
        print(f"  [!] Original has NEGATIVE edge (Sharpe = {orig_sharpe:.2f})")
        if rev_sharpe > 0.3:
            print(f"  [+] Reverse is PROFITABLE (Sharpe = {rev_sharpe:.2f})")
            print("  -> Signals appear CONSISTENTLY WRONG")
            conclusion = "FLIPPING could work - but verify costs"
        else:
            print(f"  [X] Reverse STILL unprofitable (Sharpe = {rev_sharpe:.2f})")
            print("  -> Short costs likely exceed any reversal gain")
            conclusion = "NEITHER DIRECTION works - avoid both"
    elif abs(orig_sharpe - rev_sharpe) < 0.2:
        print(f"  [-] Both have similar low edge")
        print(f"      Original: Sharpe = {orig_sharpe:.2f}, PF = {orig_pf:.2f}")
        print(f"      Reverse:  Sharpe = {rev_sharpe:.2f}, PF = {rev_pf:.2f}")
        print("  -> Signals appear RANDOM (no consistent edge)")
        conclusion = "SIGNALS ARE RANDOM - flipping doesn't help"
    else:
        print(f"  [-] Original: Sharpe = {orig_sharpe:.2f}, PF = {orig_pf:.2f}")
        print(f"  [-] Reverse:  Sharpe = {rev_sharpe:.2f}, PF = {rev_pf:.2f}")
        if reversal_improves:
            print("  -> Reversal improves results but still marginal")
            conclusion = "FLIPPING HELPS but not enough to trade"
        else:
            print("  -> Results are inconclusive")
            conclusion = "INCONCLUSIVE - test more assets/periods"

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(f"\n{conclusion}")

    # Additional analysis
    if original_results["trades"] > 5:
        print("\nTrade Statistics:")
        print(f"  Original win rate: {original_results['win_rate']:.1f}%")
        print(f"  Reverse win rate:  {reverse_results['win_rate']:.1f}%")

        # If original win rate is around 50%, signals are random
        wr = original_results["win_rate"]
        if 45 <= wr <= 55:
            print("\n  Note: ~50% win rate suggests RANDOM signals")
            print("        Flipping random signals yields random results")

    print("\n" + "=" * 60)

    return {"original": original_results, "reverse": reverse_results}


if __name__ == "__main__":
    results = run_comparison()
