"""
H7: Batch evaluate all 45+ pattern detectors through the 4-step gate.

Runs PatternEvaluationGate (t-stat, return/risk classification, IC, quantile spread)
on every registered pattern detector against SPY OHLCV data. Outputs JSON report
consumed by H2 (Pattern Quality Registry).

Usage:
    uv run scripts/sweep_pattern_gates.py --symbol SPY --output reports/pattern_gate/all_patterns.json
    uv run scripts/sweep_pattern_gates.py --symbol SPY --start 2016-01-01 --output reports/pattern_gate/all_patterns.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.signals.evaluation_gate import PatternEvaluationGate, EvaluationReport
from scripts.evaluate_pattern import run_pattern_detector, report_to_dict


@dataclass
class PatternEntry:
    """Registry entry for a pattern detector."""

    name: str
    cls_path: str


# ── All pattern detectors registered in RulesFirstStrategy + FMZ ──
ALL_PATTERNS: list[PatternEntry] = [
    PatternEntry("Gartley Pattern", "src.patterns.harmonic.gartley.GartleyPattern"),
    PatternEntry("ABC Pattern", "src.patterns.harmonic.abc.ABCPattern"),
    PatternEntry(
        "Symmetric Triangle", "src.patterns.harmonic.symmetric_triangle.SymmetricTriangle"
    ),
    PatternEntry("Bollinger Bands", "src.patterns.harmonic.bollinger.BollingerBands"),
    PatternEntry("Double Top", "src.patterns.classic.double_top.DoubleTop"),
    PatternEntry("Double Bottom", "src.patterns.classic.double_bottom.DoubleBottom"),
    PatternEntry("Trader Vic 2B", "src.patterns.classic.trader_vic_2b.TraderVic2B"),
    PatternEntry("Triple Top", "src.patterns.classic.triple_top.TripleTop"),
    PatternEntry("Triple Bottom", "src.patterns.classic.triple_bottom.TripleBottom"),
    PatternEntry("Ascending Triangle", "src.patterns.classic.ascending_triangle.AscendingTriangle"),
    PatternEntry(
        "Descending Triangle", "src.patterns.classic.descending_triangle.DescendingTriangle"
    ),
    PatternEntry("Rectangle", "src.patterns.classic.rectangle.Rectangle"),
    PatternEntry("Wedge", "src.patterns.classic.wedge.Wedge"),
    PatternEntry("Dead Cat Bounce", "src.patterns.classic.dead_cat_bounce.DeadCatBounce"),
    PatternEntry("Cup and Handle", "src.patterns.complex.cup_handle.CupAndHandle"),
    PatternEntry("Head and Shoulders", "src.patterns.complex.head_shoulders.HeadAndShoulders"),
    PatternEntry(
        "Inverse Head and Shoulders", "src.patterns.complex.head_shoulders.InverseHeadAndShoulders"
    ),
    PatternEntry("Spike and Ledge", "src.patterns.complex.spike_ledge.SpikeAndLedge"),
    PatternEntry(
        "Three Hills and a Mountain", "src.patterns.complex.three_hills.ThreeHillsMountain"
    ),
    PatternEntry("Parabolic Arc", "src.patterns.complex.parabolic_arc.ParabolicArc"),
    PatternEntry("Pipe Pattern", "src.patterns.complex.pipe.PipePattern"),
    PatternEntry(
        "Donchian Channel Breakout", "src.patterns.breakout.donchian.DonchianChannelBreakout"
    ),
    PatternEntry("Gap Pattern", "src.patterns.breakout.gap.GapPattern"),
    PatternEntry("Bull Flag", "src.patterns.continuation.flag.Flag"),
    PatternEntry("Pennant", "src.patterns.continuation.pennant.Pennant"),
    PatternEntry("Market Structure Low", "src.patterns.basic.msl.MarketStructureLow"),
    PatternEntry("Market Structure High", "src.patterns.basic.msl.MarketStructureHigh"),
    PatternEntry("Matching Lows", "src.patterns.basic.matching_lows.MatchingLows"),
    PatternEntry("NR7ID", "src.patterns.basic.nr7id.NR7ID"),
    PatternEntry("N-Bar Decline", "src.patterns.basic.n_bar_decline.NBarDecline"),
    PatternEntry("Floor Pivot Breakout", "src.patterns.basic.floor_pivot.FloorPivotBreakout"),
    PatternEntry("Two Bar Reversal", "src.patterns.basic.two_bar_reversal.TwoBarReversal"),
    PatternEntry("Doji", "src.patterns.candlestick.doji.Doji"),
    PatternEntry("Harami", "src.patterns.candlestick.harami.Harami"),
    PatternEntry("Hammer", "src.patterns.candlestick.hammer.Hammer"),
    PatternEntry("Engulfing", "src.patterns.candlestick.engulfing.Engulfing"),
    PatternEntry("Dark Cloud Cover", "src.patterns.candlestick.dark_cloud.DarkCloudCover"),
    PatternEntry("Piercing Line", "src.patterns.candlestick.dark_cloud.PiercingLine"),
    PatternEntry("Butterfly Pattern", "src.patterns.harmonic.extended.ButterflyPattern"),
    PatternEntry("Bat Pattern", "src.patterns.harmonic.extended.BatPattern"),
    PatternEntry("Crab Pattern", "src.patterns.harmonic.extended.CrabPattern"),
    PatternEntry("Cypher Pattern", "src.patterns.harmonic.extended.CypherPattern"),
    PatternEntry("Shark Pattern", "src.patterns.harmonic.extended.SharkPattern"),
    # FMZ detectors
    PatternEntry("Alpha Beast", "src.patterns.fmz.alpha_beast.AlphaBeast"),
    PatternEntry("Multi Factor Trend", "src.patterns.fmz.multi_factor_trend.MultiFactorTrend"),
    PatternEntry("Momentum ZigZag", "src.patterns.fmz.momentum_zigzag.MomentumZigZag"),
    PatternEntry("EMA MACD HF", "src.patterns.fmz.ema_macd_hf.EMAMACDHF"),
    PatternEntry("Adaptive Bollinger", "src.patterns.fmz.adaptive_bollinger.AdaptiveBollinger"),
    PatternEntry(
        "AI Volatility Breakout", "src.patterns.fmz.ai_volatility_breakout.AIVolatilityBreakout"
    ),
]


def load_data(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Load OHLCV data, preferring local CSV then yfinance."""
    data_dir = project_root / "data" / "raw"
    csv_path = data_dir / f"{symbol}_daily.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        import yfinance as yf

        ticker = symbol.replace("_USD", "-USD")
        df = yf.download(ticker, start=start or "2015-01-01", auto_adjust=False)
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]
    return df


@dataclass
class SweepResult:
    pattern_name: str
    cls_path: str
    passed: bool
    steps_passed: int
    total_steps: int
    step1_tstat: float | None
    step1_pvalue: float | None
    step2_factor_type: str
    step3_ic: float | None
    step3_ic_pvalue: float | None
    step4_spread: float | None
    step_details: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    bars: int = 0


def sweep_patterns(
    df: pd.DataFrame,
    significance: float = 0.05,
    tstat_threshold: float = 2.0,
    n_quantiles: int = 5,
    skip_fmz: bool = True,
) -> list[SweepResult]:
    """Run 4-step gate on all registered patterns."""
    ret = df["Close"].pct_change().shift(-1).dropna().to_numpy(dtype=float)
    ohlcv_df = df.iloc[:-1]

    gate = PatternEvaluationGate(
        significance=significance,
        tstat_threshold=tstat_threshold,
        n_quantiles=n_quantiles,
    )

    results: list[SweepResult] = []
    for entry in ALL_PATTERNS:
        if skip_fmz and "fmz" in entry.cls_path:
            continue

        print(f"  Evaluating: {entry.name}...", end=" ", flush=True)
        try:
            signal = run_pattern_detector(entry.cls_path, ohlcv_df)
            report: EvaluationReport = gate.evaluate(
                entry.name,
                signal,
                ret,
                sectors=None,
                log_mcap=None,
            )
            bar_count = len(signal)
            steps = [
                {
                    "step": s.step,
                    "name": s.name,
                    "value": s.value,
                    "threshold": s.threshold,
                    "passed": s.passed,
                }
                for s in report.steps
            ]

            sr = SweepResult(
                pattern_name=entry.name,
                cls_path=entry.cls_path,
                passed=report.passed,
                steps_passed=sum(1 for s in report.steps if s.passed),
                total_steps=len(report.steps),
                step1_tstat=report.step1_tstat,
                step1_pvalue=report.step1_pvalue,
                step2_factor_type=report.step2_factor_type.value,
                step3_ic=report.step3_ic,
                step3_ic_pvalue=report.step3_ic_pvalue,
                step4_spread=report.step4_spread,
                step_details=steps,
                bars=bar_count,
            )
            status = "PASS" if sr.passed else f"FAIL ({sr.steps_passed}/{sr.total_steps})"
            print(status)
        except Exception as e:
            sr = SweepResult(
                pattern_name=entry.name,
                cls_path=entry.cls_path,
                passed=False,
                steps_passed=0,
                total_steps=4,
                step1_tstat=None,
                step1_pvalue=None,
                step2_factor_type="unknown",
                step3_ic=None,
                step3_ic_pvalue=None,
                step4_spread=None,
                error=f"{type(e).__name__}: {e}",
            )
            print(f"ERROR: {sr.error}")
        results.append(sr)

    return results


def format_summary_table(results: list[SweepResult]) -> str:
    """Format a summary table of all results."""
    lines = []
    lines.append(f"\n{'=' * 90}")
    lines.append("  Pattern Gate Sweep Summary")
    lines.append(f"{'=' * 90}")
    header = f"  {'Pattern':<32} {'T-stat':>8} {'P-val':>8} {'IC':>8} {'Spread':>8} {'Result':>10}"
    lines.append(header)
    lines.append(f"  {'-' * 32} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")

    results = sorted(results, key=lambda r: r.step3_ic or -999, reverse=True)
    for r in results:
        tstat = f"{r.step1_tstat:.2f}" if r.step1_tstat is not None else "N/A"
        pval = f"{r.step1_pvalue:.3f}" if r.step1_pvalue is not None else "N/A"
        ic = f"{r.step3_ic:.4f}" if r.step3_ic is not None else "N/A"
        spread = f"{r.step4_spread:.6f}" if r.step4_spread is not None else "N/A"
        result = "PASS" if r.passed else f"FAIL ({r.steps_passed}/4)"
        if r.error:
            result = "ERROR"
        lines.append(
            f"  {r.pattern_name:<32} {tstat:>8} {pval:>8} {ic:>8} {spread:>8} {result:>10}"
        )

    lines.append("")
    passed_count = sum(1 for r in results if r.passed)
    err_count = sum(1 for r in results if r.error)
    lines.append(f"  Passed: {passed_count}/{len(results)}  Errors: {err_count}")
    lines.append(f"{'=' * 90}\n")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H7: Sweep all pattern detectors through 4-step evaluation gate"
    )
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--output", default="reports/pattern_gate/all_patterns.json", help="JSON output path"
    )
    parser.add_argument(
        "--no-skip-fmz", action="store_true", help="Include FMZ patterns (may hang)"
    )
    parser.add_argument("--tstat-threshold", type=float, default=2.0)
    parser.add_argument("--significance", type=float, default=0.05)
    parser.add_argument("--n-quantiles", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading {args.symbol} data...")
    df = load_data(args.symbol, args.start, args.end)
    print(f"Loaded {len(df)} bars ({df.index[0].date()} -> {df.index[-1].date()})")

    print(f"\nEvaluating {len(ALL_PATTERNS)} pattern detectors...")
    results = sweep_patterns(
        df,
        significance=args.significance,
        tstat_threshold=args.tstat_threshold,
        n_quantiles=args.n_quantiles,
        skip_fmz=not args.no_skip_fmz,
    )

    print(format_summary_table(results))

    # Save JSON
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "symbol": args.symbol,
            "start": str(df.index[0].date()),
            "end": str(df.index[-1].date()),
            "bars": len(df),
            "tstat_threshold": args.tstat_threshold,
            "significance": args.significance,
            "n_quantiles": args.n_quantiles,
        },
        "results": [
            {
                "pattern_name": r.pattern_name,
                "cls_path": r.cls_path,
                "passed": r.passed,
                "steps_passed": r.steps_passed,
                "total_steps": r.total_steps,
                "step1_tstat": r.step1_tstat,
                "step1_pvalue": r.step1_pvalue,
                "step2_factor_type": r.step2_factor_type,
                "step3_ic": r.step3_ic,
                "step3_ic_pvalue": r.step3_ic_pvalue,
                "step4_spread": r.step4_spread,
                "step_details": r.step_details,
                "error": r.error,
                "bars": r.bars,
            }
            for r in results
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
