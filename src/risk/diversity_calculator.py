# -*- coding: utf-8 -*-
"""
Diversity Score Calculator - Q2 Implementation

Utility to calculate portfolio diversity score using Jorion's BET formula
and answer Q2: "What's our portfolio's effective diversity score (ρ)?"

From handover document:
- Q2 Target Date: 2026-05-01
- Priority: 🟠 High
- Related: P3 (Jorion), R7, H4

Usage:
    uv run python src/risk/diversity_calculator.py --data data/symbols/*.csv
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.risk.diversity_score import DiversityConfig, DiversityMethod, DiversityScorer


@dataclass
class DiversityReport:
    """Report on portfolio diversity analysis."""

    total_patterns: int
    effective_bets: float
    average_correlation: float
    diversification_ratio: float
    concentration_risk: float
    implied_capital_reduction: float
    recommendation: str
    metadata: Dict


def load_pattern_signals(
    data_dir: str,
    pattern_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load pattern signals from CSV files.

    Args:
        data_dir: Directory containing pattern signal CSVs
        pattern_names: Optional list of patterns to load

    Returns:
        DataFrame with columns as patterns, rows as time periods
    """
    signals = {}

    if pattern_names is None:
        pattern_files = list(Path(data_dir).glob("*.csv"))
        pattern_names = [f.stem for f in pattern_files]

    for pattern in pattern_names:
        filepath = os.path.join(data_dir, f"{pattern}.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            if "signal" in df.columns:
                signals[pattern] = df["signal"]
            elif "value" in df.columns:
                signals[pattern] = df["value"]

    if not signals:
        raise ValueError(f"No pattern signals found in {data_dir}")

    returns_df = pd.DataFrame(signals)
    returns_df = returns_df.dropna()

    return returns_df


def calculate_portfolio_diversity(
    returns: pd.DataFrame,
    method: DiversityMethod = DiversityMethod.SIMPLIFIED_BET,
) -> DiversityReport:
    """
    Calculate portfolio diversity metrics.

    Args:
        returns: DataFrame of pattern returns/signals
        method: Diversity calculation method

    Returns:
        DiversityReport with all metrics
    """
    scorer = DiversityScorer(DiversityConfig(method=method))
    result = scorer.calculate(returns)

    # Calculate implied capital reduction (Jorion's finding: ρ=0.03 → 6x reduction)
    # If ρ_avg is our average correlation, then capital reduction factor is:
    # reduction = 1 / diversification_ratio
    implied_capital_reduction = 1.0 / max(result.diversification_ratio, 0.01)

    # Generate recommendation
    if result.diversification_ratio < 0.3:
        recommendation = (
            f"CRITICAL: Only {result.diversification_ratio:.1%} diversification. "
            f"Effective bets: {result.diversity_score:.1f} of {result.position_count}. "
            "Consider reducing concentration or adding uncorrelated strategies."
        )
    elif result.diversification_ratio < 0.5:
        recommendation = (
            f"WARNING: Moderate diversification ({result.diversification_ratio:.1%}). "
            f"Effective bets: {result.diversity_score:.1f}. "
            "Monitor correlation drift."
        )
    else:
        recommendation = (
            f"GOOD: Well diversified ({result.diversification_ratio:.1%}). "
            f"Effective bets: {result.diversity_score:.1f} of {result.position_count}."
        )

    return DiversityReport(
        total_patterns=result.position_count,
        effective_bets=result.diversity_score,
        average_correlation=result.average_correlation,
        diversification_ratio=result.diversification_ratio,
        concentration_risk=result.concentration_risk,
        implied_capital_reduction=implied_capital_reduction,
        recommendation=recommendation,
        metadata={
            "largest_eigenvalue": result.metadata.get("largest_eigenvalue", 0),
            "variance_explained": result.metadata.get("variance_explained", 0),
            "condition_number": result.metadata.get("condition_number", 0),
        },
    )


def run_sensitivity_analysis(
    returns: pd.DataFrame,
    rho_range: np.ndarray = None,
) -> pd.DataFrame:
    """
    Run sensitivity analysis on ρ assumption.

    Args:
        returns: Pattern returns DataFrame
        rho_range: Range of ρ values to test (default: 0.00 to 0.10)

    Returns:
        DataFrame with sensitivity results
    """
    if rho_range is None:
        rho_range = np.linspace(0.00, 0.10, 11)

    n_patterns = returns.shape[1]
    results = []

    for rho in rho_range:
        denominator = 1.0 + (n_patterns - 1) * rho
        diversity = n_patterns / denominator if denominator > 0 else n_patterns
        diversity_ratio = diversity / n_patterns
        capital_reduction = 1.0 / max(diversity_ratio, 0.01)

        results.append(
            {
                "rho_assumption": rho,
                "effective_bets": diversity,
                "diversification_ratio": diversity_ratio,
                "capital_reduction_factor": capital_reduction,
            }
        )

    return pd.DataFrame(results)


def calculate_pattern_correlation_matrix(
    returns: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Calculate 34×34 pattern signal correlation matrix.

    Args:
        returns: Pattern returns/signals DataFrame

    Returns:
        Tuple of (correlation_matrix, summary_stats)
    """
    corr_matrix = returns.corr(method="pearson")

    # Extract summary statistics
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    correlations = upper_tri.stack()

    summary = {
        "mean_correlation": correlations.mean(),
        "median_correlation": correlations.median(),
        "max_correlation": correlations.max(),
        "min_correlation": correlations.min(),
        "std_correlation": correlations.std(),
        "high_corr_pairs": len(correlations[correlations.abs() > 0.7]),
        "negative_corr_pairs": len(correlations[correlations < 0]),
    }

    return corr_matrix, summary


def generate_diversity_dashboard(
    data_dir: str,
    output_dir: str = "reports/diversity",
) -> Dict:
    """
    Generate comprehensive diversity dashboard.

    Args:
        data_dir: Directory with pattern signal CSVs
        output_dir: Directory to save reports

    Returns:
        Dictionary with all analysis results
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading pattern signals...")
    returns = load_pattern_signals(data_dir)
    print(f"  Loaded {returns.shape[1]} patterns, {returns.shape[0]} observations")

    print("\nCalculating diversity score...")
    report = calculate_portfolio_diversity(returns)

    print("\nCalculating correlation matrix...")
    corr_matrix, corr_summary = calculate_pattern_correlation_matrix(returns)

    print("\nRunning sensitivity analysis...")
    sensitivity = run_sensitivity_analysis(returns)

    print("\n=== DIVERSITY ANALYSIS RESULTS ===\n")
    print(f"Total Patterns: {report.total_patterns}")
    print(f"Effective Bets (D): {report.effective_bets:.2f}")
    print(f"Average Correlation (ρ): {report.average_correlation:.4f}")
    print(f"Diversification Ratio: {report.diversification_ratio:.2%}")
    print(f"Concentration Risk: {report.concentration_risk:.2%}")
    print(f"\nImplied Capital Reduction: {report.implied_capital_reduction:.1f}x")
    print(f"\n{report.recommendation}")

    print("\nCorrelation Summary:")
    for key, value in corr_summary.items():
        print(f"  {key}: {value:.4f}")

    print("\nSensitivity Analysis (ρ assumption → effective bets):")
    print(
        sensitivity.to_string(
            index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
        )
    )

    # Save results
    results = {
        "summary": asdict(report),
        "correlation_matrix": corr_matrix.to_dict(),
        "correlation_summary": corr_summary,
        "sensitivity_analysis": sensitivity.to_dict(orient="records"),
    }

    # Save JSON report
    json_path = os.path.join(output_dir, "diversity_report.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nReport saved to: {json_path}")

    # Save correlation matrix as CSV
    corr_path = os.path.join(output_dir, "pattern_correlation_matrix.csv")
    corr_matrix.to_csv(corr_path)
    print(f"Correlation matrix saved to: {corr_path}")

    # Save sensitivity analysis
    sens_path = os.path.join(output_dir, "sensitivity_analysis.csv")
    sensitivity.to_csv(sens_path, index=False)
    print(f"Sensitivity analysis saved to: {sens_path}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate portfolio diversity score (Q2)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing pattern signal CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/diversity",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["simplified_bet", "full_bet", "effective_n", "correlation_weighted"],
        default="simplified_bet",
        help="Diversity calculation method",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    method_map = {
        "simplified_bet": DiversityMethod.SIMPLIFIED_BET,
        "full_bet": DiversityMethod.FULL_BET,
        "effective_n": DiversityMethod.EFFECTIVE_N,
        "correlation_weighted": DiversityMethod.CORRELATION_WEIGHTED,
    }

    results = generate_diversity_dashboard(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    print("\n✅ Q2 Analysis complete!")
    print("\nKey Finding:")
    print(f"  Your 34-pattern portfolio has effective diversity of ~{results['summary']['effective_bets']:.1f} independent bets")
    print(f"  This implies a capital requirement multiplier of {results['summary']['implied_capital_reduction']:.1f}x")


if __name__ == "__main__":
    main()
