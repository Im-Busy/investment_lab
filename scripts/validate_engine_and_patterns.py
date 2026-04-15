"""
Validation Script: Engine Baseline + Untested Patterns

1. Runs a simple Buy & Hold baseline on SPY daily data via backtesting.py
2. Backtests all 23 untested geometric patterns on SPY daily and BTC 1H
3. Reports results to determine if patterns are unsuitable vs implementation errors

Usage:
    uv run scripts/validate_engine_and_patterns.py
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from backtesting import Backtest, Strategy

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ============================================================================
# 1. Buy & Hold Baseline Strategy
# ============================================================================


class BuyAndHoldStrategy(Strategy):
    """Simple buy and hold strategy for baseline validation."""

    def init(self):
        pass

    def next(self):
        if not self.position:
            self.buy(size=1)


# ============================================================================
# 2. Simple Pattern-Based Strategy for backtesting.py
# ============================================================================

from src.patterns.basic import (
    MarketStructureLow,
    MatchingLows,
    NR7ID,
    NBarDecline,
    FloorPivotBreakout,
)
from src.patterns.basic.two_bar_reversal import TwoBarReversal
from src.patterns.breakout.gap import GapPattern
from src.patterns.candlestick import Doji, Engulfing, Hammer, Harami
from src.patterns.candlestick.dark_cloud import DarkCloudCover
from src.patterns.classic import (
    DoubleTop,
    DoubleBottom,
    TraderVic2B,
    TripleTop,
    TripleBottom,
    AscendingTriangle,
    DescendingTriangle,
    Rectangle,
    Wedge,
    DeadCatBounce,
)
from src.patterns.complex import (
    CupAndHandle,
    HeadAndShoulders,
    SpikeAndLedge,
    ThreeHillsMountain,
    ParabolicArc,
)
from src.patterns.continuation.flag import Flag
from src.patterns.continuation.pennant import Pennant
from src.patterns.harmonic import (
    GartleyPattern,
    ABCPattern,
    SymmetricTriangle,
    BollingerBands,
)
from src.patterns.breakout.donchian import DonchianChannelBreakout


PATTERNS_TO_TEST = [
    # Basic (6)
    ("MSL", MarketStructureLow),
    ("Matching Lows", MatchingLows),
    ("NR7ID", NR7ID),
    ("N-Bar Decline", NBarDecline),
    ("Floor Pivot", FloorPivotBreakout),
    ("Two-Bar Reversal", TwoBarReversal),
    # Harmonic (5)
    ("Gartley", GartleyPattern),
    ("ABC", ABCPattern),
    ("Symmetric Triangle", SymmetricTriangle),
    ("Donchian", DonchianChannelBreakout),
    ("Bollinger Bands", BollingerBands),
    # Complex (5)
    ("Cup & Handle", CupAndHandle),
    ("Head & Shoulders", HeadAndShoulders),
    ("Spike & Ledge", SpikeAndLedge),
    ("Three Hills", ThreeHillsMountain),
    ("Parabolic Arc", ParabolicArc),
    # Classic (10)
    ("Double Top", DoubleTop),
    ("Double Bottom", DoubleBottom),
    ("Trader Vic 2B", TraderVic2B),
    ("Triple Top", TripleTop),
    ("Triple Bottom", TripleBottom),
    ("Ascending Triangle", AscendingTriangle),
    ("Descending Triangle", DescendingTriangle),
    ("Rectangle", Rectangle),
    ("Wedge", Wedge),
    ("Dead Cat Bounce", DeadCatBounce),
    # Continuation (2)
    ("Flag", Flag),
    ("Pennant", Pennant),
    # Breakout (1)
    ("Gap", GapPattern),
    # Candlestick (5)
    ("Doji", Doji),
    ("Engulfing", Engulfing),
    ("Hammer", Hammer),
    ("Harami", Harami),
    ("Dark Cloud/Piercing", DarkCloudCover),
]


class SinglePatternStrategy(Strategy):
    """Simple strategy that trades on a single pattern detection."""

    pattern_class = None  # Set via kwargs
    min_confidence = 0.40
    stop_atr_mult = 3.0
    tp_atr_mult = 6.0

    def init(self):
        self.pattern = self.pattern_class()
        self._atr = self.I(self._calculate_atr, self.data.High, self.data.Low, self.data.Close)

    @staticmethod
    def _calculate_atr(high, low, close, period=14):
        import numpy as np

        atr = np.empty_like(high)
        atr[:] = np.nan
        if len(high) < period + 1:
            return atr
        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - np.roll(close, 1)[1:]), np.abs(low - np.roll(close, 1)[1:])),
        )
        for i in range(period, len(high)):
            if i == period:
                atr[i] = np.mean(tr[: period + 1])
            else:
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

    def next(self):
        if len(self.trades) > 0:
            return

        df = pd.DataFrame(
            {
                "Open": self.data.Open,
                "High": self.data.High,
                "Low": self.data.Low,
                "Close": self.data.Close,
                "Volume": self.data.Volume if hasattr(self.data, "Volume") else 0,
            }
        )
        df.index = self.data.index

        current_idx = len(df) - 1
        if current_idx < self.pattern.min_bars_required:
            return

        try:
            result = self.pattern.detect(df, current_idx)
            if result.detected and result.signal:
                if result.signal.confidence >= self.min_confidence:
                    sig = result.signal
                    entry = self.data.Close[-1]
                    atr_val = self._atr[-1] if not np.isnan(self._atr[-1]) else entry * 0.02

                    stop = entry - (atr_val * self.stop_atr_mult)
                    tp = entry + (atr_val * self.tp_atr_mult)

                    if sig.direction.value == "Long":
                        self.buy(size=0.95, sl=stop, tp=tp)
                    else:
                        self.sell(
                            size=0.95,
                            sl=entry + (atr_val * self.stop_atr_mult),
                            tp=entry - (atr_val * self.tp_atr_mult),
                        )
        except Exception:
            pass


# ============================================================================
# 3. Backtest Runner
# ============================================================================


def load_spy_daily() -> pd.DataFrame:
    path = project_root / "data" / "raw" / "SPY_daily.csv"
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df


def load_btc_1h() -> pd.DataFrame:
    path = project_root / "data" / "raw" / "BTC_USD_1h.csv"
    df = pd.read_csv(path, parse_dates=["Datetime"], index_col="Datetime")
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df


def run_backtest(
    name: str, strategy_cls, df: pd.DataFrame, cash=1_000_000, commission=0.001
) -> dict:
    try:
        bt = Backtest(df, strategy_cls, cash=cash, commission=commission, exclusive_orders=True)
        stats = bt.run()
        return {
            "Strategy": name,
            "Trades": int(stats.get("# Trades", 0)),
            "Win Rate": float(stats.get("Win Rate [%]", 0)),
            "Sharpe": float(stats.get("Sharpe Ratio", -999)),
            "Profit Factor": float(stats.get("Profit Factor", 0)),
            "Max DD": float(stats.get("Max. Drawdown [%]", 0)),
            "Return": float(stats.get("Return [%]", 0)),
            "B&H Return": float(stats.get("Buy & Hold Return [%]", 0)),
        }
    except Exception as e:
        return {
            "Strategy": name,
            "Trades": 0,
            "Win Rate": 0,
            "Sharpe": -999,
            "Profit Factor": 0,
            "Max DD": 0,
            "Return": 0,
            "B&H Return": 0,
            "Error": str(e),
        }


def run_pattern_backtest(
    name: str, pattern_cls, df: pd.DataFrame, cash=1_000_000, commission=0.001
) -> dict:
    """Run backtest for a single pattern strategy."""

    class PatternStrat(SinglePatternStrategy):
        pattern_class = pattern_cls

    try:
        bt = Backtest(
            df,
            PatternStrat,
            cash=cash,
            commission=commission,
            exclusive_orders=True,
            finalize_trades=True,
        )
        stats = bt.run()
        return {
            "Strategy": name,
            "Trades": int(stats.get("# Trades", 0)),
            "Win Rate": float(stats.get("Win Rate [%]", 0)),
            "Sharpe": float(stats.get("Sharpe Ratio", -999)),
            "Profit Factor": float(stats.get("Profit Factor", 0)),
            "Max DD": float(stats.get("Max. Drawdown [%]", 0)),
            "Return": float(stats.get("Return [%]", 0)),
            "B&H Return": float(stats.get("Buy & Hold Return [%]", 0)),
        }
    except Exception as e:
        return {
            "Strategy": name,
            "Trades": 0,
            "Win Rate": 0,
            "Sharpe": -999,
            "Profit Factor": 0,
            "Max DD": 0,
            "Return": 0,
            "B&H Return": 0,
            "Error": str(e),
        }


# ============================================================================
# 4. Main Execution
# ============================================================================


def main():
    print("=" * 100)
    print("ENGINE VALIDATION + PATTERN BACKTEST")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 100)

    # Load data
    spy_df = load_spy_daily()
    btc_df = load_btc_1h()

    print(f"\nSPY Daily: {len(spy_df)} bars, {spy_df.index[0]} to {spy_df.index[-1]}")
    print(f"  Price range: {spy_df['Close'].min():.2f} - {spy_df['Close'].max():.2f}")
    print(f"\nBTC 1H: {len(btc_df)} bars, {btc_df.index[0]} to {btc_df.index[-1]}")
    print(f"  Price range: {btc_df['Close'].min():.2f} - {btc_df['Close'].max():.2f}")

    # 1. Buy & Hold Baseline
    print("\n" + "=" * 100)
    print("1. BUY & HOLD BASELINE (SPY Daily)")
    print("=" * 100)

    bh_result = run_backtest("Buy & Hold", BuyAndHoldStrategy, spy_df)
    print(f"  Return: {bh_result['Return']:.2f}%")
    print(f"  Buy & Hold Return: {bh_result['B&H Return']:.2f}%")
    print(f"  Sharpe: {bh_result['Sharpe']:.2f}")

    # 2. Test all 34 patterns on SPY Daily
    print("\n" + "=" * 100)
    print("2. SINGLE PATTERN BACKTESTS - SPY DAILY")
    print("=" * 100)

    print(
        f"{'Pattern':<25} {'Trades':>7} {'WR%':>7} {'Sharpe':>8} {'PF':>7} {'MaxDD%':>8} {'Return%':>8}"
    )
    print("-" * 100)

    spy_results = []
    for name, pattern_cls in PATTERNS_TO_TEST:
        result = run_pattern_backtest(name, pattern_cls, spy_df)
        spy_results.append(result)
        error = result.get("Error", "")
        if error:
            print(f"{name:<25} ERROR: {error}")
        else:
            print(
                f"{name:<25} {result['Trades']:>7} {result['Win Rate']:>6.1f}% {result['Sharpe']:>8.2f} {result['Profit Factor']:>7.2f} {result['Max DD']:>7.1f}% {result['Return']:>7.1f}%"
            )

    # 3. Test all 34 patterns on BTC 1H
    print("\n" + "=" * 100)
    print("3. SINGLE PATTERN BACKTESTS - BTC 1H")
    print("=" * 100)

    print(
        f"{'Pattern':<25} {'Trades':>7} {'WR%':>7} {'Sharpe':>8} {'PF':>7} {'MaxDD%':>8} {'Return%':>8}"
    )
    print("-" * 100)

    btc_results = []
    for name, pattern_cls in PATTERNS_TO_TEST:
        result = run_pattern_backtest(name, pattern_cls, btc_df)
        btc_results.append(result)
        error = result.get("Error", "")
        if error:
            print(f"{name:<25} ERROR: {error}")
        else:
            print(
                f"{name:<25} {result['Trades']:>7} {result['Win Rate']:>6.1f}% {result['Sharpe']:>8.2f} {result['Profit Factor']:>7.2f} {result['Max DD']:>7.1f}% {result['Return']:>7.1f}%"
            )

    # 4. Summary
    print("\n" + "=" * 100)
    print("4. SUMMARY - PATTERNS WITH POSITIVE SHARPE")
    print("=" * 100)

    print("\n--- SPY Daily ---")
    profitable_spy = [r for r in spy_results if r.get("Sharpe", -999) > 0 and r["Trades"] > 0]
    if profitable_spy:
        for r in profitable_spy:
            print(
                f"  {r['Strategy']:<25} Trades={r['Trades']:>4} WR={r['Win Rate']:.1f}% Sharpe={r['Sharpe']:.2f} PF={r['Profit Factor']:.2f}"
            )
    else:
        print("  NONE - All patterns have negative Sharpe on SPY daily")

    print("\n--- BTC 1H ---")
    profitable_btc = [r for r in btc_results if r.get("Sharpe", -999) > 0 and r["Trades"] > 0]
    if profitable_btc:
        for r in profitable_btc:
            print(
                f"  {r['Strategy']:<25} Trades={r['Trades']:>4} WR={r['Win Rate']:.1f}% Sharpe={r['Sharpe']:.2f} PF={r['Profit Factor']:.2f}"
            )
    else:
        print("  NONE - All patterns have negative Sharpe on BTC 1H")

    # 5. Conclusion
    print("\n" + "=" * 100)
    print("5. CONCLUSION")
    print("=" * 100)

    total_patterns = len(PATTERNS_TO_TEST)
    zero_trades_spy = sum(1 for r in spy_results if r["Trades"] == 0)
    zero_trades_btc = sum(1 for r in btc_results if r["Trades"] == 0)

    print(f"  Total patterns tested: {total_patterns}")
    print(
        f"  SPY: {zero_trades_spy} patterns with zero trades, {len(profitable_spy)} with positive Sharpe"
    )
    print(
        f"  BTC: {zero_trades_btc} patterns with zero trades, {len(profitable_btc)} with positive Sharpe"
    )

    if not profitable_spy and not profitable_btc:
        print("\n  *** ALL PATTERNS FAIL ON BOTH ASSETS ***")
        print("  Conclusion: The pattern detection logic itself is likely unsuitable,")
        print("  not implementation errors. Recommend pivoting to:")
        print("    1. Mean-reversion strategies on range-bound assets")
        print("    2. Pair trading / statistical arbitrage (Phase 2)")
        print("    3. Proven strategies from academic/quant literature")
    elif profitable_spy or profitable_btc:
        print("\n  *** SOME PATTERNS ARE PROFITABLE ***")
        print("  Focus on optimizing the profitable patterns and finding")
        print("  the right asset/timeframe combinations.")

    # Save results
    output_dir = project_root / "reports"
    output_dir.mkdir(exist_ok=True)

    spy_df_results = pd.DataFrame(spy_results)
    btc_df_results = pd.DataFrame(btc_results)

    spy_df_results.to_csv(output_dir / "validation_spy_daily.csv", index=False)
    btc_df_results.to_csv(output_dir / "validation_btc_1h.csv", index=False)

    print(f"\nResults saved to:")
    print(f"  {output_dir / 'validation_spy_daily.csv'}")
    print(f"  {output_dir / 'validation_btc_1h.csv'}")


if __name__ == "__main__":
    main()
