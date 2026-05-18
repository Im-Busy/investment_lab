"""
R3: CLI for 4-step pattern evaluation gate.

Evaluates pattern signal quality against forward returns.  Applies
sector/size purification and reports per-step gate results.

Usage:
    # Quick evaluation with close prices
    uv run scripts/evaluate_pattern.py --name "Head and Shoulders" \\
        --cls src.patterns.complex.head_shoulders.HeadAndShoulders \\
        --ticker data/raw/SPY_daily.csv

    # With sector + market cap data
    uv run scripts/evaluate_pattern.py --name "K Line Power" \\
        --signals-file outputs/pattern_signals.csv \\
        --returns-file outputs/forward_returns.csv \\
        --sectors-file outputs/sectors.csv
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.signals.evaluation_gate import PatternEvaluationGate, EvaluationReport
from src.signals.factor_purification import PurificationReport


def load_csv(path: str, col: str | None = None) -> np.ndarray:
    """Load a column from CSV. If col is None, uses the first column."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if col is not None:
        return df[col].to_numpy(dtype=float)
    return df.iloc[:, 0].to_numpy(dtype=float)


def run_pattern_detector(
    cls_path: str,
    ohlcv_df: pd.DataFrame,
) -> np.ndarray:
    """Instantiate a pattern detector and run detect_vectorized()."""
    module_path, class_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    detector_cls = getattr(module, class_name)
    detector = detector_cls()
    signals = detector.detect_vectorized(ohlcv_df)
    return signals.astype(np.float64)


def format_report(report: EvaluationReport) -> str:
    """Pretty-print evaluation report."""
    lines = []
    lines.append(f"\n{'=' * 60}")
    lines.append(f"  Pattern Evaluation: {report.pattern_name}")
    lines.append(f"{'=' * 60}")

    # Step results table
    lines.append(f"\n  {'Step':<8} {'Gate':<24} {'Value':>10} {'Threshold':>10} {'Result':>8}")
    lines.append(f"  {'-' * 8} {'-' * 24} {'-' * 10} {'-' * 10} {'-' * 8}")
    for step in report.steps:
        status = "PASS" if step.passed else "FAIL"
        thresh = f"{step.threshold}" if step.threshold is not None else "N/A"
        lines.append(
            f"  {step.step:<8} {step.name:<24} {step.value:>10.4f} {thresh:>10} {status:>8}"
        )

    # Overall
    overall = "PASS" if report.passed else "FAIL"
    lines.append(f"\n  OVERALL: {overall}")

    # Classification
    lines.append(f"  Factor Type: {report.step2_factor_type.value}")
    if report.step1_pvalue is not None:
        stars = ""
        if report.step1_pvalue < 0.001:
            stars = "***"
        elif report.step1_pvalue < 0.01:
            stars = "**"
        elif report.step1_pvalue < 0.05:
            stars = "*"
        lines.append(f"  T-stat: {report.step1_tstat:.4f} (p={report.step1_pvalue:.4f}{stars})")
    lines.append(f"  Rank IC: {report.step3_ic:.4f}")
    if report.step3_ic_pvalue is not None:
        lines.append(f"  IC p-value: {report.step3_ic_pvalue:.4f}")
    lines.append(f"  Quantile Spread (top-bot): {report.step4_spread:.6f}")
    lines.append(f"  Top Q return: {report.step4_top_ret:.6f}")
    lines.append(f"  Bottom Q return: {report.step4_bottom_ret:.6f}")

    # Purification
    if report.purification is not None:
        p = report.purification
        lines.append("\n  Purification:")
        lines.append(
            f"    Purity Ratio: {p.purity_ratio:.4f} {'(CONTAMINATED)' if p.is_contaminated else ''}"
        )
        lines.append(f"    Sector R²:    {p.sector_r2:.4f}")
        lines.append(f"    Market Cap R²:{p.market_cap_r2:.4f}")
        lines.append(f"    Total R²:     {p.total_r2:.4f}")

    if report.warnings:
        lines.append("\n  Warnings:")
        for w in report.warnings:
            lines.append(f"    - {w}")

    lines.append("")
    return "\n".join(lines)


def report_to_dict(report: EvaluationReport) -> dict:
    """Convert report to JSON-serializable dict."""
    result: dict = {
        "pattern_name": report.pattern_name,
        "passed": report.passed,
        "step1_tstat": report.step1_tstat,
        "step1_pvalue": report.step1_pvalue,
        "step2_factor_type": report.step2_factor_type.value,
        "step2_tstat_ratio": report.step2_tstat_ratio,
        "step3_ic": report.step3_ic,
        "step3_ic_pvalue": report.step3_ic_pvalue,
        "step4_top_ret": report.step4_top_ret,
        "step4_bottom_ret": report.step4_bottom_ret,
        "step4_spread": report.step4_spread,
        "steps": [
            {
                "step": s.step,
                "name": s.name,
                "value": s.value,
                "threshold": s.threshold,
                "passed": s.passed,
            }
            for s in report.steps
        ],
        "warnings": report.warnings,
    }
    if report.purification is not None:
        result["purification"] = {
            "purity_ratio": report.purification.purity_ratio,
            "sector_r2": report.purification.sector_r2,
            "market_cap_r2": report.purification.market_cap_r2,
            "total_r2": report.purification.total_r2,
            "is_contaminated": report.purification.is_contaminated,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="4-step pattern evaluation gate (华泰多因子 §1.3)")
    parser.add_argument("--name", required=True, help="Pattern name for reporting")

    # Data sources
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--ticker", help="CSV path with OHLCV data")
    src_group.add_argument("--signals-file", help="CSV with precomputed pattern signals")

    parser.add_argument(
        "--cls",
        help="Dotted path to pattern detector class (e.g. src.patterns.harmonic.gartley.GartleyPattern)",
    )
    parser.add_argument("--returns-file", help="CSV with forward returns")
    parser.add_argument("--sectors-file", help="CSV with sector labels per bar")
    parser.add_argument("--mcap-file", help="CSV with log(market cap) per bar")
    parser.add_argument("--signal-col", help="Column name in signals file")
    parser.add_argument("--returns-col", help="Column name in returns file")

    # Gate params
    parser.add_argument(
        "--tstat-threshold",
        type=float,
        default=2.0,
        help="Minimum |t| for significance (default: 2.0)",
    )
    parser.add_argument(
        "--significance", type=float, default=0.05, help="P-value threshold (default: 0.05)"
    )
    parser.add_argument(
        "--n-quantiles", type=int, default=5, help="Number of quantile bins (default: 5)"
    )

    # Output
    parser.add_argument("--json-output", help="Save JSON report to file")

    args = parser.parse_args()

    # ── Load data ──
    if args.ticker:
        if not args.cls:
            parser.error("--cls is required with --ticker")
        ohlcv_df = pd.read_csv(args.ticker, index_col=0, parse_dates=True)
        ret = ohlcv_df["Close"].pct_change().shift(-1).dropna().to_numpy(dtype=float)
        ohlcv_df = ohlcv_df.iloc[:-1]
        signal = run_pattern_detector(args.cls, ohlcv_df)
    else:
        signal = load_csv(args.signals_file, args.signal_col)
        if args.returns_file:
            ret = load_csv(args.returns_file, args.returns_col)
        else:
            parser.error("--returns-file is required with --signals-file")

    # Load sector/mcap if provided
    sectors: np.ndarray | None = None
    log_mcap: np.ndarray | None = None
    if args.sectors_file:
        sectors = np.array(pd.read_csv(args.sectors_file, index_col=0).iloc[:, 0].astype(str))
    if args.mcap_file:
        log_mcap = pd.read_csv(args.mcap_file, index_col=0).iloc[:, 0].to_numpy(dtype=float)

    # ── Run gate ──
    gate = PatternEvaluationGate(
        significance=args.significance,
        tstat_threshold=args.tstat_threshold,
        n_quantiles=args.n_quantiles,
    )
    report = gate.evaluate(args.name, signal, ret, sectors, log_mcap)

    # ── Output ──
    print(format_report(report))

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report_to_dict(report), indent=2, default=str))
        print(f"Report saved to {args.json_output}")


if __name__ == "__main__":
    main()
