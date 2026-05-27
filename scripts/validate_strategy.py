#!/usr/bin/env python3
"""Comprehensive Strategy Validation Suite.

Runs the full Phase 25 post-backtest validation pipeline:
  P0.1: DSR/PBO — Deflated Sharpe Ratio & statistical significance
  P0.2: Purged Walk-Forward Analysis — WFE, majority-pass, catastrophic veto
  P0.3 + P1.6: Regime Audit — Per-regime Sharpe/DSR/consistency score
  P1.4: Monte Carlo Robustness — Return reshuffling + param perturbation

Usage:
    # From backtest JSON output
    uv run scripts/validate_strategy.py --json reports/batch/rules_first_OOS_2025_2026.json

    # From return series CSV
    uv run scripts/validate_strategy.py --csv data/strategy_returns.csv --column returns

    # All tests for a single symbol
    uv run scripts/validate_strategy.py --json reports/batch/rules_first_OOS_2025_2026.json \\
        --symbol SPY --n-trials 200 --full

    # Quick check (DSR + WFA only, fast)
    uv run scripts/validate_strategy.py --returns 0.005,-0.003,0.012,0.008,-0.001,... \\
        --n-trials 150 --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.analysis.deflated_sharpe import (
    compute_dsr_from_returns,
    compute_psr_from_returns,
    format_significance_summary,
    print_significance_report,
    _sharpe_ratio,
)
from src.analysis.purged_walk_forward import (
    DEFAULT_CATASTROPHIC_THRESHOLD,
    DEFAULT_WFE_THRESHOLD,
    PurgedWalkForwardValidator,
)
from src.analysis.regime_audit import (
    audit_regimes,
    format_regime_report,
)
from src.analysis.monte_carlo_robustness import (
    run_full_robustness_check,
    format_mc_report,
)

logger = logging.getLogger(__name__)


def load_returns_from_json(
    filepath: str,
    symbol: str | None = None,
    return_key: str = "returns",
) -> tuple[np.ndarray, dict]:
    """Load return series from a backtest JSON file.

    Supports multiple formats:
      1. Direct array under key: {"returns": [0.01, -0.02, ...]}
      2. Per-symbol dict: {"SPY": {"returns": [...]}, "QQQ": {...}}
      3. Nested: {"results": [{"symbol": "SPY", "returns": [...]}]}
      4. Backtest stats dict: {"sharpe": 1.5, "returns": [...]}
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    metadata = {}

    # Case 1: Direct array
    if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data[:5]):
        return np.array(data, dtype=np.float64), metadata

    # Case 2: Dict with returns key
    if isinstance(data, dict) and return_key in data:
        rets = data[return_key]
        if isinstance(rets, list):
            metadata = {k: v for k, v in data.items() if k != return_key}
            return np.array(rets, dtype=np.float64), metadata

    # Case 3: Per-symbol dict
    if symbol and isinstance(data, dict) and symbol in data:
        sub = data[symbol]
        if isinstance(sub, dict) and return_key in sub:
            metadata = {k: v for k, v in sub.items() if k != return_key}
            return np.array(sub[return_key], dtype=np.float64), metadata
        if isinstance(sub, list):
            return np.array(sub, dtype=np.float64), metadata

    # Case 4: List of per-symbol results
    if symbol and isinstance(data, list) and isinstance(data[0], dict):
        for entry in data:
            if entry.get("symbol") == symbol or entry.get("ticker") == symbol:
                if return_key in entry:
                    metadata = {k: v for k, v in entry.items() if k != return_key}
                    return np.array(entry[return_key], dtype=np.float64), metadata
                if "daily_returns" in entry:
                    metadata = {k: v for k, v in entry.items() if k != "daily_returns"}
                    return np.array(entry["daily_returns"], dtype=np.float64), metadata

    # Case 5: First entry
    if isinstance(data, dict):
        sharpe = data.get("sharpe_ratio") or data.get("sharpe") or data.get("oos_sharpe")
        returns_val = data.get(return_key) or data.get("daily_returns") or data.get("oos_returns")
        if returns_val and isinstance(returns_val, list):
            metadata = {"sharpe": sharpe} if sharpe else {}
            metadata["n_trades"] = data.get("trades") or data.get("n_trades") or 0
            return np.array(returns_val, dtype=np.float64), metadata

    raise ValueError(
        f"Could not extract returns from {filepath}. Try --symbol <NAME> or --return-key <KEY>."
    )


def load_returns_from_csv(
    filepath: str,
    column: str = "returns",
) -> tuple[np.ndarray, dict]:
    """Load return series from a CSV file."""
    df = pd.read_csv(filepath)
    if column not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(f"Column '{column}' not found. Available: {available}")
    returns = df[column].dropna().to_numpy(dtype=np.float64)
    return returns, {}


def parse_returns_arg(returns_str: str) -> np.ndarray:
    """Parse comma-separated returns string."""
    parts = returns_str.strip().split(",")
    return np.array([float(p.strip()) for p in parts], dtype=np.float64)


def run_dsr_analysis(
    returns: np.ndarray,
    n_trials: int,
    periods_per_year: int = 252,
) -> str:
    """Run DSR + PSR + permutation + bootstrap CI analysis."""
    report = print_significance_report(returns, n_trials=n_trials, n_trades=0)
    return report


def run_wfa_analysis(
    returns: np.ndarray,
    is_days: int = 4 * 252,
    oos_days: int = 252,
    purge_days: int = 21,
    step_days: int = 126,
    periods_per_year: int = 252,
) -> str:
    """Run purged walk-forward analysis."""
    validator = PurgedWalkForwardValidator(
        is_days=is_days,
        oos_days=oos_days,
        purge_days=purge_days,
        step_days=step_days,
        periods_per_year=periods_per_year,
    )
    report = validator.validate(returns)
    return validator.format_report(report)


def run_regime_analysis(
    returns: np.ndarray,
    vix_path: str | None = None,
    spy_path: str | None = None,
    periods_per_year: int = 252,
    n_trials: int = 100,
    use_svm: bool = False,
) -> str:
    """Run regime audit analysis."""
    vix = None
    spy_ret = None

    if vix_path:
        vix_df = pd.read_csv(vix_path, parse_dates=True, index_col=0)
        if "Close" in vix_df.columns:
            vix = vix_df["Close"].to_numpy()
        elif len(vix_df.columns) > 0:
            vix = vix_df.iloc[:, 0].to_numpy()
        vix = vix[-len(returns) :]

    if spy_path:
        spy_df = pd.read_csv(spy_path, parse_dates=True, index_col=0)
        if "Close" in spy_df.columns:
            spy_prices = spy_df["Close"].to_numpy()
            spy_ret = np.diff(np.log(spy_prices))
            spy_ret = np.insert(spy_ret, 0, 0)
            spy_ret = spy_ret[-len(returns) :]

    report = audit_regimes(
        returns,
        vix=vix,
        spy_returns=spy_ret,
        periods_per_year=periods_per_year,
        n_trials=n_trials,
        use_svm=use_svm,
    )
    return format_regime_report(report)


def run_mc_robustness(
    returns: np.ndarray,
    params: dict[str, float] | None = None,
    n_simulations: int = 5000,
    n_perturbations: int = 100,
    periods_per_year: int = 252,
) -> str:
    """Run Monte Carlo robustness analysis."""
    report = run_full_robustness_check(
        returns,
        params=params,
        n_simulations=n_simulations,
        n_perturbations=n_perturbations,
        periods_per_year=periods_per_year,
    )
    return format_mc_report(report)


def run_comprehensive_validation(
    returns: np.ndarray,
    n_trials: int = 200,
    params: dict[str, float] | None = None,
    vix_path: str | None = None,
    spy_path: str | None = None,
    is_days: int = 4 * 252,
    oos_days: int = 252,
    purge_days: int = 21,
    step_days: int = 126,
    n_mc_simulations: int = 5000,
    n_perturbations: int = 100,
    periods_per_year: int = 252,
    quick: bool = False,
    use_svm_regime: bool = False,
) -> str:
    """Run the complete validation suite and return a combined report."""
    if quick:
        n_mc_simulations = 1000
        n_perturbations = 30
        step_days = oos_days * 2

    lines = [
        "",
        "#" * 72,
        "#" + " " * 22 + "STRATEGY VALIDATION REPORT" + " " * 22 + "#",
        "#" + " " * 72 + "#",
        f"#  Days: {len(returns):>5}  |  Sharpe: {_sharpe_ratio(returns, periods_per_year):.3f}  |  "
        f"Return: {float(np.prod(1.0 + returns) - 1.0):.1%}"
        + " " * (40 - len(f"{float(np.prod(1.0 + returns) - 1.0):.1%}"))
        + "#",
        "#" * 72,
        "",
    ]

    # P0.1: DSR Analysis
    lines.append("## P0.1: DEFLATED SHARPE RATIO / STATISTICAL SIGNIFICANCE")
    lines.append("")
    try:
        dsr_report = print_significance_report(returns, n_trials=n_trials, n_trades=0)
        lines.append(dsr_report)
    except Exception as e:
        lines.append(f"  ERROR: {e}")
    lines.append("")

    # P0.2: Purged WFA
    lines.append("## P0.2: PURGED WALK-FORWARD ANALYSIS")
    lines.append("")
    try:
        wfa_report = run_wfa_analysis(
            returns,
            is_days=is_days,
            oos_days=oos_days,
            purge_days=purge_days,
            step_days=step_days,
            periods_per_year=periods_per_year,
        )
        lines.append(wfa_report)
    except Exception as e:
        lines.append(f"  ERROR: {e}")
    lines.append("")

    # P0.3 + P1.6: Regime Audit
    lines.append("## P0.3 + P1.6: REGIME AUDIT")
    lines.append("")
    try:
        regime_report = run_regime_analysis(
            returns,
            vix_path=vix_path,
            spy_path=spy_path,
            periods_per_year=periods_per_year,
            n_trials=n_trials,
            use_svm=use_svm_regime,
        )
        lines.append(regime_report)
    except Exception as e:
        lines.append(f"  ERROR: {e}")
    lines.append("")

    # P1.4: Monte Carlo Robustness
    lines.append("## P1.4: MONTE CARLO ROBUSTNESS")
    lines.append("")
    try:
        mc_report = run_mc_robustness(
            returns,
            params=params,
            n_simulations=n_mc_simulations,
            n_perturbations=n_perturbations,
            periods_per_year=periods_per_year,
        )
        lines.append(mc_report)
    except Exception as e:
        lines.append(f"  ERROR: {e}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive strategy validation suite (DSR + WFA + Regime Audit + Monte Carlo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run scripts/validate_strategy.py --json reports/batch/rules_first_OOS_2025_2026.json --symbol SPY --n-trials 200
  uv run scripts/validate_strategy.py --csv results.csv --column daily_returns --quick
  uv run scripts/validate_strategy.py --returns "0.01,-0.005,0.02,0.008,-0.003" --n-trials 150
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--json", type=str, help="Path to backtest JSON results file")
    input_group.add_argument("--csv", type=str, help="Path to CSV file with returns column")
    input_group.add_argument("--returns", type=str, help="Comma-separated returns string")

    parser.add_argument("--symbol", type=str, help="Symbol to extract from multi-symbol JSON")
    parser.add_argument(
        "--return-key",
        type=str,
        default="returns",
        help="Key for returns in JSON (default: 'returns')",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="returns",
        help="Column name for returns in CSV (default: 'returns')",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=200,
        help="Number of strategy trials for DSR correction (default: 200)",
    )
    parser.add_argument("--periods", type=int, default=252, help="Periods per year (default: 252)")

    # WFA params
    parser.add_argument(
        "--is-days", type=int, default=4 * 252, help="WFA training window in bars (default: 1008)"
    )
    parser.add_argument(
        "--oos-days", type=int, default=252, help="WFA test window in bars (default: 252)"
    )
    parser.add_argument(
        "--purge-days", type=int, default=21, help="WFA purge gap in bars (default: 21)"
    )
    parser.add_argument(
        "--step-days", type=int, default=126, help="WFA step size in bars (default: 126)"
    )

    # Regime audit params
    parser.add_argument("--vix", type=str, help="Path to VIX CSV for regime classification")
    parser.add_argument("--spy", type=str, help="Path to SPY CSV for drawdown calculation")
    parser.add_argument(
        "--use-svm-regime", action="store_true", help="Use SVM classifier for regime labeling"
    )

    # Monte Carlo params
    parser.add_argument(
        "--mc-simulations",
        type=int,
        default=5000,
        help="Monte Carlo simulation count (default: 5000)",
    )
    parser.add_argument(
        "--mc-perturbations", type=int, default=100, help="Param perturbation draws (default: 100)"
    )
    parser.add_argument(
        "--params-json", type=str, help="JSON file with parameter dict for perturbation testing"
    )

    # Analysis selection
    parser.add_argument("--full", action="store_true", help="Run all validation tests (default)")
    parser.add_argument(
        "--quick", action="store_true", help="Fast run: reduced simulations, larger steps"
    )
    parser.add_argument("--dsr-only", action="store_true", help="Run only DSR/PBO analysis")
    parser.add_argument("--wfa-only", action="store_true", help="Run only walk-forward analysis")
    parser.add_argument("--regime-only", action="store_true", help="Run only regime audit")
    parser.add_argument("--mc-only", action="store_true", help="Run only Monte Carlo robustness")

    # Output
    parser.add_argument("--output", "-o", type=str, help="Save report to file")
    parser.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    args = parser.parse_args()

    # Load returns
    if args.json:
        returns, metadata = load_returns_from_json(args.json, args.symbol, args.return_key)
    elif args.csv:
        returns, metadata = load_returns_from_csv(args.csv, args.column)
    elif args.returns:
        returns = parse_returns_arg(args.returns)
        metadata = {}
    else:
        parser.error("No input specified")

    if len(returns) < 20:
        print(f"ERROR: Only {len(returns)} data points — need at least 20 for meaningful analysis")
        sys.exit(1)

    # Load params for perturbation
    params: dict[str, float] | None = None
    if args.params_json:
        with open(args.params_json) as f:
            params = json.load(f)

    # Determine which analyses to run
    run_all = args.full or not (args.dsr_only or args.wfa_only or args.regime_only or args.mc_only)

    reports: list[str] = []

    if run_all or args.dsr_only:
        reports.append("\n=== DSR / PBO / STATISTICAL SIGNIFICANCE ===")
        reports.append(
            print_significance_report(
                returns,
                n_trials=args.n_trials,
                n_trades=metadata.get("n_trades", 0),
            )
        )

    if run_all or args.wfa_only:
        reports.append(
            f"\n=== PURGED WALK-FORWARD (IS={args.is_days}d, OOS={args.oos_days}d, purge={args.purge_days}d) ==="
        )
        reports.append(
            run_wfa_analysis(
                returns,
                is_days=args.is_days,
                oos_days=args.oos_days,
                purge_days=args.purge_days,
                step_days=args.step_days,
                periods_per_year=args.periods,
            )
        )

    if run_all or args.regime_only:
        reports.append("\n=== REGIME AUDIT ===")
        reports.append(
            run_regime_analysis(
                returns,
                vix_path=args.vix,
                spy_path=args.spy,
                periods_per_year=args.periods,
                n_trials=args.n_trials,
                use_svm=args.use_svm_regime,
            )
        )

    if run_all or args.mc_only:
        reports.append(f"\n=== MONTE CARLO ROBUSTNESS ({args.mc_simulations} simulations) ===")
        mc_sims = min(args.mc_simulations, 1000) if args.quick else args.mc_simulations
        mc_pert = min(args.mc_perturbations, 30) if args.quick else args.mc_perturbations
        reports.append(
            run_mc_robustness(
                returns,
                params=params,
                n_simulations=mc_sims,
                n_perturbations=mc_pert,
                periods_per_year=args.periods,
            )
        )

    output = "\n".join(reports)

    if not args.quiet:
        print(output)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        if not args.quiet:
            print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
