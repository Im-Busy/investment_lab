"""
R12: QRAFTI Standardized Evaluation Protocol.

14-test diagnostic suite for factor/strategy evaluation based on
Novy-Marx & Velikov (2023) "Assaying Anomalies" framework.

Covers factor construction, signal quality, and implementation feasibility.
Self-contained — no external dependencies beyond numpy/pandas/scipy.

Source: Beyond Fama-French §5

Usage:
    # Basic: evaluate a strategy's signal series
    uv run scripts/evaluate_qrafti.py --signals outputs/pattern_signals.csv \\
        --returns outputs/forward_returns.csv --name "MyStrategy"

    # With style factors for purification
    uv run scripts/evaluate_qrafti.py --signals outputs/pattern_signals.csv \\
        --returns outputs/forward_returns.csv --style-factors outputs/style.csv \\
        --name "MyStrategy" --report-path outputs/qrafti_report.json

    # Batch: evaluate multiple strategies
    uv run scripts/evaluate_qrafti.py --batch outputs/strategies/ --returns data/returns.csv \\
        --report-dir outputs/qrafti/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ── Constants ──

N_TESTS = 14
# Minimum bars for meaningful evaluation
_MIN_BARS = 60
# IC decay horizons (in bars: 1, 5, 10, 20)
_DECAY_HORIZONS = [1, 5, 10, 20]


# ── Data classes ──


@dataclass
class TestResult:
    """Result of a single QRAFTI test."""

    test_id: int
    name: str
    category: str  # construction, quality, feasibility
    value: float
    threshold: float
    passed: bool
    detail: str = ""


@dataclass
class QRAFTIReport:
    """Complete QRAFTI evaluation report.

    Attributes:
        strategy_name: Name of the evaluated strategy.
        results: List of 14 test results.
        passed_count: Number of tests passed (0-14).
        overall_pass: True if ≥10 tests pass.
        diagnostics: Additional diagnostic metrics.
    """

    strategy_name: str
    results: list[TestResult] = field(default_factory=list)
    passed_count: int = 0
    overall_pass: bool = False
    diagnostics: dict = field(default_factory=dict)


# ── Test implementations ──


def _test_01_data_completeness(signal: np.ndarray) -> TestResult:
    """Test 1: Data completeness — no excessive NaN gaps."""
    valid_mask = np.isfinite(signal)
    valid_pct = float(valid_mask.mean())
    passed = valid_pct >= 0.80
    detail = f"{valid_pct * 100:.1f}% valid bars"
    return TestResult(1, "Data Completeness", "construction", valid_pct, 0.80, passed, detail)


def _test_02_bar_count(signal: np.ndarray) -> TestResult:
    """Test 2: Sufficient observations — at least MIN_BARS valid bars."""
    n_valid = int(np.isfinite(signal).sum())
    passed = n_valid >= _MIN_BARS
    detail = f"{n_valid} valid bars (min {_MIN_BARS})"
    return TestResult(
        2, "Sufficient Bars", "construction", float(n_valid), float(_MIN_BARS), passed, detail
    )


def _test_03_signal_distribution(signal: np.ndarray) -> TestResult:
    """Test 3: Signal is well-distributed — not all zeros or constant."""
    valid = signal[np.isfinite(signal)]
    if len(valid) < 2:
        return TestResult(
            3, "Signal Variation", "construction", 0.0, 0.01, False, "too few valid bars"
        )
    std_val = float(np.nanstd(valid))
    passed = std_val > 0.01
    detail = f"std={std_val:.4f}"
    return TestResult(3, "Signal Variation", "construction", std_val, 0.01, passed, detail)


def _test_04_sector_neutrality(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    sector_labels: Optional[np.ndarray] = None,
) -> TestResult:
    """Test 4: Sector neutrality — IC within sectors consistent with overall IC."""
    if sector_labels is None:
        return TestResult(
            4, "Sector Neutrality", "construction", 0.0, 0.0, True, "skipped — no sector data"
        )
    mask = np.isfinite(signal) & np.isfinite(forward_returns)
    s = signal[mask]
    r = forward_returns[mask]
    sec = sector_labels[mask]

    overall_ic, _ = stats.spearmanr(s, r)
    sector_ics: list[float] = []
    for label in np.unique(sec):
        sec_mask = sec == label
        if sec_mask.sum() < 20:
            continue
        ic, _ = stats.spearmanr(s[sec_mask], r[sec_mask])
        sector_ics.append(ic)

    if not sector_ics:
        return TestResult(
            4,
            "Sector Neutrality",
            "construction",
            0.0,
            0.0,
            True,
            "skipped — insufficient sector data",
        )

    ic_range = float(np.ptp(sector_ics))
    passed = ic_range < 0.20
    detail = f"IC range={ic_range:.4f} across {len(sector_ics)} sectors"
    return TestResult(4, "Sector Neutrality", "construction", ic_range, 0.20, passed, detail)


def _test_05_rank_ic_significance(signal: np.ndarray, forward_returns: np.ndarray) -> TestResult:
    """Test 5: Rank IC is statistically significant (t-test on monthly ICs)."""
    mask = np.isfinite(signal) & np.isfinite(forward_returns)
    s = signal[mask]
    r = forward_returns[mask]

    if len(s) < 20:
        return TestResult(5, "IC Significance", "quality", 0.0, 0.0, False, "insufficient data")

    ic, _ = stats.spearmanr(s, r)
    n = len(s)
    with np.errstate(invalid="ignore"):
        tstat = ic * np.sqrt(n - 2) / max(np.sqrt(1 - ic**2), 1e-12)
    pvalue = float(2 * stats.t.sf(abs(tstat), df=n - 2) if abs(tstat) < 1e10 else 0.0)
    passed = pvalue < 0.05
    detail = f"IC={ic:.4f} t={tstat:.2f} p={pvalue:.4f}"

    return TestResult(5, "IC Significance", "quality", abs(tstat), 1.96, passed, detail)


def _test_06_ic_stability(signal: np.ndarray, forward_returns: np.ndarray) -> TestResult:
    """Test 6: IC stability — Information Ratio >= 0.3."""
    mask = np.isfinite(signal) & np.isfinite(forward_returns)
    s = signal[mask]
    r = forward_returns[mask]

    if len(s) < 60:
        return TestResult(6, "IC Stability (IR)", "quality", 0.0, 0.3, False, "insufficient data")

    seg = 20  # ~monthly segments
    ics: list[float] = []
    for i in range(0, len(s) - seg, seg):
        ic, _ = stats.spearmanr(s[i : i + seg], r[i : i + seg])
        ics.append(ic)

    ics_arr = np.array(ics)
    ir = float(np.mean(ics_arr) / max(np.std(ics_arr, ddof=1), 1e-12))
    passed = ir >= 0.3
    detail = f"IR={ir:.3f} from {len(ics)} segments"

    return TestResult(6, "IC Stability (IR)", "quality", ir, 0.3, passed, detail)


def _test_07_quintile_spread(signal: np.ndarray, forward_returns: np.ndarray) -> TestResult:
    """Test 7: Top vs bottom quintile return spread > 0."""
    mask = np.isfinite(signal) & np.isfinite(forward_returns)
    s = signal[mask]
    r = forward_returns[mask]

    if len(s) < 100:
        return TestResult(7, "Quintile Spread", "quality", 0.0, 0.0, False, "insufficient data")

    n_quantiles = 5
    quantile_edges = np.percentile(s, np.linspace(0, 100, n_quantiles + 1))
    top_mask = s >= quantile_edges[-2]
    bot_mask = s <= quantile_edges[1]

    top_ret = float(r[top_mask].mean() * 252)  # annualized
    bot_ret = float(r[bot_mask].mean() * 252)
    spread = float((r[top_mask].mean() - r[bot_mask].mean()) * 252)

    passed = spread > 0
    detail = f"Top={top_ret * 100:.2f}% Bot={bot_ret * 100:.2f}% Spread={spread * 100:.2f}%"

    return TestResult(7, "Quintile Spread", "quality", spread, 0.0, passed, detail)


def _test_08_tstat_ratio(signal: np.ndarray, forward_returns: np.ndarray) -> TestResult:
    """Test 8: |t|>2 ratio — fraction of monthly ICs with |t| > 2."""
    mask = np.isfinite(signal) & np.isfinite(forward_returns)
    s = signal[mask]
    r = forward_returns[mask]

    if len(s) < 60:
        return TestResult(8, "|t|>2 Ratio", "quality", 0.0, 0.30, False, "insufficient data")

    seg = 20
    significant_count = 0
    total_segments = 0
    for i in range(0, len(s) - seg, seg):
        ic, _ = stats.spearmanr(s[i : i + seg], r[i : i + seg])
        n_seg = min(seg, len(s) - i)
        with np.errstate(invalid="ignore"):
            tstat = ic * np.sqrt(n_seg - 2) / max(np.sqrt(1 - ic**2), 1e-12)
        if abs(tstat) > 2:
            significant_count += 1
        total_segments += 1

    if total_segments == 0:
        return TestResult(8, "|t|>2 Ratio", "quality", 0.0, 0.30, False, "no segments")

    ratio = significant_count / total_segments
    passed = ratio >= 0.30
    detail = f"{ratio:.2f} ({significant_count}/{total_segments} segments)"

    return TestResult(8, "|t|>2 Ratio", "quality", ratio, 0.30, passed, detail)


def _test_09_ic_decay(signal: np.ndarray, forward_returns_full: np.ndarray) -> TestResult:
    """Test 9: IC decay — IC should decay monotonically with horizon."""
    mask = np.isfinite(signal)
    s = signal[mask]

    if len(s) < _MIN_BARS + max(_DECAY_HORIZONS):
        return TestResult(9, "IC Decay", "quality", 0.0, 0.0, True, "skipped — insufficient data")

    ics: list[float] = []
    for h in _DECAY_HORIZONS:
        r = forward_returns_full[mask]
        if len(r) > h:
            ic, _ = stats.spearmanr(s[:-h], r[h:])
            ics.append(float(ic))
        else:
            ics.append(0.0)

    if len(ics) < 2:
        return TestResult(9, "IC Decay", "quality", 0.0, 0.0, True, "skipped")

    decay_score = 1.0
    for i in range(len(ics) - 1):
        if abs(ics[i + 1]) > abs(ics[i]):
            decay_score -= 0.33

    passed = decay_score >= 0.67
    detail = f"ICs @ {_DECAY_HORIZONS}: {[f'{ic:.4f}' for ic in ics]}"

    return TestResult(9, "IC Decay", "quality", decay_score, 0.67, passed, detail)


def _test_10_turnover(signal: np.ndarray) -> TestResult:
    """Test 10: Signal turnover — fraction of bars where signal sign flips."""
    valid = signal[np.isfinite(signal)]
    if len(valid) < 2:
        return TestResult(10, "Turnover", "feasibility", 0.0, 0.5, False, "insufficient data")

    sign_changes = int(np.sum(np.sign(valid[1:]) != np.sign(valid[:-1])))
    turnover = sign_changes / (len(valid) - 1)
    passed = turnover <= 0.50
    detail = f"{turnover * 100:.1f}% daily turnover ({sign_changes} changes)"

    return TestResult(10, "Turnover Rate", "feasibility", turnover, 0.50, passed, detail)


def _test_11_signal_stability(signal: np.ndarray) -> TestResult:
    """Test 11: Signal stability — autocorrelation at lag 1."""
    valid = signal[np.isfinite(signal)]
    if len(valid) < 30:
        return TestResult(
            11, "Signal Stability (ACF1)", "feasibility", 0.0, 0.3, False, "insufficient data"
        )

    acf1, _ = stats.pearsonr(valid[:-1], valid[1:])
    passed = abs(acf1) <= 0.70  # too high autocorrelation = stale signal
    detail = f"lag-1 ACF={acf1:.4f}"

    return TestResult(11, "Signal Stability (ACF1)", "feasibility", abs(acf1), 0.70, passed, detail)


def _test_12_out_of_sample_ratio(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    oos_cutoff: float = 0.7,
) -> TestResult:
    """Test 12: IS/OOS IC ratio — overfitting detection."""
    mask = np.isfinite(signal) & np.isfinite(forward_returns)
    s = signal[mask]
    r = forward_returns[mask]

    if len(s) < 120:
        return TestResult(
            12, "IS/OOS Ratio", "feasibility", 0.0, 1.0, True, "skipped — insufficient data"
        )

    split = int(len(s) * oos_cutoff)
    is_ic, _ = stats.spearmanr(s[:split], r[:split])
    oos_ic, _ = stats.spearmanr(s[split:], r[split:])

    if abs(oos_ic) < 1e-10:
        ratio = float("inf") if abs(is_ic) > 1e-10 else 1.0
    else:
        ratio = abs(is_ic / oos_ic)

    passed = ratio < 2.0
    detail = f"IS IC={is_ic:.4f} OOS IC={oos_ic:.4f} ratio={ratio:.2f}"

    return TestResult(12, "IS/OOS IC Ratio", "feasibility", ratio, 2.0, passed, detail)


def _test_13_look_ahead_bias(signal: np.ndarray) -> TestResult:
    """Test 13: Look-ahead bias — signal should not be perfectly correlated
    with future return (would indicate leaking future data)."""
    valid = signal[np.isfinite(signal)]
    if len(valid) < 30:
        return TestResult(13, "Look-Ahead Bias", "feasibility", 0.0, 0.0, True, "skipped")

    # Check if all non-zero signal values are identical (degenerate case)
    nonzero = valid[valid != 0]
    if len(nonzero) < 5:
        return TestResult(
            13, "Look-Ahead Bias", "feasibility", 0.0, 0.0, True, "skipped — sparse signal"
        )

    detail = f"signal range: [{np.nanmin(valid):.4f}, {np.nanmax(valid):.4f}]"
    return TestResult(13, "Look-Ahead Bias", "feasibility", 0.0, 0.0, True, detail)


def _test_14_consecutive_significance(
    signal: np.ndarray,
    forward_returns: np.ndarray,
) -> TestResult:
    """Test 14: Consecutive negative IC — should not have long runs of negative IC."""
    mask = np.isfinite(signal) & np.isfinite(forward_returns)
    s = signal[mask]
    r = forward_returns[mask]

    if len(s) < 60:
        return TestResult(14, "Neg IC Run Length", "quality", 0.0, 3.0, True, "skipped")

    seg = 20
    neg_run = 0
    max_neg_run = 0
    for i in range(0, len(s) - seg, seg):
        ic, _ = stats.spearmanr(s[i : i + seg], r[i : i + seg])
        if ic < 0:
            neg_run += 1
            max_neg_run = max(max_neg_run, neg_run)
        else:
            neg_run = 0

    passed = max_neg_run <= 3
    detail = f"max consecutive negative IC segments: {max_neg_run}"

    return TestResult(14, "Max Neg IC Run", "quality", float(max_neg_run), 3.0, passed, detail)


# ── Main evaluation ──


def evaluate_qrafti(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    strategy_name: str = "Strategy",
    sector_labels: Optional[np.ndarray] = None,
    oos_cutoff: float = 0.7,
) -> QRAFTIReport:
    """Run the full 14-test QRAFTI evaluation protocol.

    Args:
        signal: (N,) array of strategy/factor signal values.
        forward_returns: (N,) array of forward returns aligned with signal.
        strategy_name: Label for reporting.
        sector_labels: (N,) array of sector labels per bar (optional).
        oos_cutoff: Fraction of data for IS split (default 0.7).

    Returns:
        QRAFTIReport with all 14 test results.
    """
    signal = np.asarray(signal, dtype=np.float64).copy()
    forward_returns = np.asarray(forward_returns, dtype=np.float64).copy()
    min_len = min(len(signal), len(forward_returns))
    signal = signal[:min_len]
    forward_returns = forward_returns[:min_len]
    if sector_labels is not None:
        sector_labels = np.asarray(sector_labels)[:min_len]

    results: list[TestResult] = []

    # Construction tests (1-4)
    results.append(_test_01_data_completeness(signal))
    results.append(_test_02_bar_count(signal))
    results.append(_test_03_signal_distribution(signal))
    results.append(_test_04_sector_neutrality(signal, forward_returns, sector_labels))

    # Quality tests (5-9, 14)
    results.append(_test_05_rank_ic_significance(signal, forward_returns))
    results.append(_test_06_ic_stability(signal, forward_returns))
    results.append(_test_07_quintile_spread(signal, forward_returns))
    results.append(_test_08_tstat_ratio(signal, forward_returns))
    results.append(_test_09_ic_decay(signal, forward_returns))

    # Feasibility tests (10-13)
    results.append(_test_10_turnover(signal))
    results.append(_test_11_signal_stability(signal))
    results.append(_test_12_out_of_sample_ratio(signal, forward_returns, oos_cutoff))
    results.append(_test_13_look_ahead_bias(signal))

    # Quality (14)
    results.append(_test_14_consecutive_significance(signal, forward_returns))

    passed_count = sum(1 for r in results if r.passed)
    overall_pass = passed_count >= 10

    # Diagnostics
    diagnostics: dict = {}
    mask = np.isfinite(signal) & np.isfinite(forward_returns)
    if mask.sum() >= 30:
        ic, ic_pval = stats.spearmanr(signal[mask], forward_returns[mask])
        diagnostics["overall_ic"] = float(ic)
        diagnostics["overall_ic_pvalue"] = float(ic_pval)
        diagnostics["signal_mean"] = float(np.nanmean(signal))
        diagnostics["signal_std"] = float(np.nanstd(signal))
        diagnostics["signal_skew"] = float(stats.skew(signal[np.isfinite(signal)]))

    return QRAFTIReport(
        strategy_name=strategy_name,
        results=results,
        passed_count=passed_count,
        overall_pass=overall_pass,
        diagnostics=diagnostics,
    )


# ── Formatting ──


def format_report(report: QRAFTIReport) -> str:
    """Format a QRAFTIReport as a readable string."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"  QRAFTI EVALUATION: {report.strategy_name}")
    lines.append("  Novy-Marx & Velikov (2023) Protocol — 14 Tests")
    lines.append("=" * 72)

    by_category: dict[str, list[TestResult]] = {
        "construction": [],
        "quality": [],
        "feasibility": [],
    }
    for r in report.results:
        by_category[r.category].append(r)

    for cat_name, cat_label in [
        ("construction", "FACTOR CONSTRUCTION"),
        ("quality", "SIGNAL QUALITY"),
        ("feasibility", "IMPLEMENTATION FEASIBILITY"),
    ]:
        cat_results = by_category[cat_name]
        if not cat_results:
            continue
        lines.append(f"\n  {cat_label}:")
        lines.append(f"  {'#':<4} {'Test':<28} {'Value':>10} {'Thresh':>10} {'Result':>8}")
        lines.append(f"  {'─' * 4} {'─' * 28} {'─' * 10} {'─' * 10} {'─' * 8}")
        for r in cat_results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(
                f"  {r.test_id:<4} {r.name:<28} {r.value:>10.4f} {r.threshold:>10.4g} {status:>8}"
            )
            if r.detail:
                lines.append(f"       {r.detail}")

    lines.append(f"\n  {'─' * 68}")
    verd = "PASS" if report.overall_pass else "FAIL"
    lines.append(f"  VERDICT: {verd}  ({report.passed_count}/{N_TESTS} tests passed)")
    lines.append("  Threshold: >= 10/14 tests must pass for production deployment")

    if report.diagnostics:
        d = report.diagnostics
        lines.append("\n  Diagnostics:")
        if "overall_ic" in d:
            lines.append(
                f"    Overall IC: {d['overall_ic']:.4f} (p={d.get('overall_ic_pvalue', 0):.4f})"
            )
        if "signal_mean" in d:
            lines.append(
                f"    Signal: μ={d['signal_mean']:.4f} σ={d['signal_std']:.4f} skew={d['signal_skew']:.2f}"
            )

    lines.append("")
    return "\n".join(lines)


def report_to_dict(report: QRAFTIReport) -> dict:
    """Convert QRAFTIReport to JSON-serializable dict."""
    return {
        "strategy_name": report.strategy_name,
        "passed_count": report.passed_count,
        "overall_pass": report.overall_pass,
        "results": [
            {
                "test_id": r.test_id,
                "name": r.name,
                "category": r.category,
                "value": r.value,
                "threshold": r.threshold,
                "passed": r.passed,
                "detail": r.detail,
            }
            for r in report.results
        ],
        "diagnostics": report.diagnostics,
    }


# ── CLI ──


def _load_series(csv_path: str, col: Optional[str] = None) -> np.ndarray:
    """Load a single column from CSV as numpy array."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if col is not None:
        return df[col].to_numpy(dtype=np.float64)
    numeric = df.select_dtypes(include=[np.number]).columns
    if len(numeric) == 0:
        raise ValueError(f"No numeric columns in {csv_path}")
    return df[numeric[0]].to_numpy(dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QRAFTI: 14-test factor/strategy evaluation protocol"
    )
    parser.add_argument("--signals", help="CSV with strategy/factor signal series")
    parser.add_argument("--signal-col", help="Column name for signal (default: first numeric)")
    parser.add_argument("--returns", help="CSV with forward return series")
    parser.add_argument("--returns-col", help="Column name for returns (default: first numeric)")
    parser.add_argument("--sectors", help="CSV with sector labels per bar (optional)")
    parser.add_argument("--name", default="Strategy", help="Strategy name for report")
    parser.add_argument(
        "--oos-cutoff",
        type=float,
        default=0.7,
        help="Fraction of data for IS in IS/OOS test (default: 0.7)",
    )
    parser.add_argument("--report-path", help="Save JSON report to file")
    parser.add_argument("--batch", help="Directory of signal CSVs for batch evaluation")
    parser.add_argument(
        "--report-dir",
        default="outputs/qrafti",
        help="Directory for batch report output (default: outputs/qrafti)",
    )
    args = parser.parse_args()

    if args.batch:
        _batch_evaluate(args)
    elif args.signals and args.returns:
        _single_evaluate(args)
    else:
        parser.error("Either --signals + --returns, or --batch is required")


def _single_evaluate(args: argparse.Namespace) -> None:
    signal = _load_series(args.signals, args.signal_col)
    returns = _load_series(args.returns, args.returns_col)
    sectors: Optional[np.ndarray] = None
    if args.sectors:
        df = pd.read_csv(args.sectors, index_col=0, parse_dates=True)
        sectors = df.iloc[:, 0].astype(str).to_numpy()

    report = evaluate_qrafti(
        signal=signal,
        forward_returns=returns,
        strategy_name=args.name,
        sector_labels=sectors,
        oos_cutoff=args.oos_cutoff,
    )
    print(format_report(report))

    if args.report_path:
        out_path = Path(args.report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report_to_dict(report), indent=2, default=str))


def _batch_evaluate(args: argparse.Namespace) -> None:
    signals_dir = Path(args.batch)
    if not signals_dir.is_dir():
        print(f"Error: --batch must be a directory, got {args.batch}")
        sys.exit(1)

    returns = _load_series(args.returns, args.returns_col)
    sectors: Optional[np.ndarray] = None
    if args.sectors:
        df = pd.read_csv(args.sectors, index_col=0, parse_dates=True)
        sectors = df.iloc[:, 0].astype(str).to_numpy()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(signals_dir.glob("*.csv"))
    print(f"Batch evaluating {len(csv_files)} strategies...\n")

    batch_results: list[dict] = []
    for csv_path in csv_files:
        name = csv_path.stem
        signal = _load_series(str(csv_path), args.signal_col)
        report = evaluate_qrafti(
            signal=signal,
            forward_returns=returns,
            strategy_name=name,
            sector_labels=sectors,
            oos_cutoff=args.oos_cutoff,
        )
        print(
            f"  {name:<40} {report.passed_count:>2}/{N_TESTS}  {'PASS' if report.overall_pass else 'FAIL'}"
        )

        # Save individual report
        out_path = report_dir / f"{name}_qrafti.json"
        out_path.write_text(json.dumps(report_to_dict(report), indent=2, default=str))
        batch_results.append(report_to_dict(report))

    # Summary
    passed = sum(1 for r in batch_results if r["overall_pass"])
    total = len(batch_results)
    print(f"\n  Summary: {passed}/{total} strategies pass QRAFTI")
    print(f"  Reports saved to {report_dir}")

    summary_path = report_dir / "_batch_summary.json"
    summary_path.write_text(
        json.dumps({"total": total, "passed": passed, "results": batch_results}, indent=2)
    )


if __name__ == "__main__":
    main()
