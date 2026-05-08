# -*- coding: utf-8 -*-
"""
Ablation Study Runner - Q7 Implementation

Answer Q7: "What's the marginal benefit of 34 patterns vs. top 10?"

From handover document:
- Q7 Target Date: 2026-06-15
- Priority: 🟠 High
- Related: R4, I4.1, I5.3

Runs ablation studies with pattern subsets:
- Top 10 patterns
- Top 20 patterns
- Top 30 patterns
- All 34 patterns

Usage:
    uv run python src/analysis/ablation_study_runner.py --data data/symbols/*.csv --output reports/ablation_study
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class AblationStudyConfig:
    """Configuration for ablation study."""

    # Pattern subsets to test
    subsets: List[int] = None

    # Minimum observations per pattern
    min_observations: int = 100

    # Random seed for reproducibility
    random_seed: int = 42

    # Output directory
    output_dir: str = "reports/ablation_study"

    def __post_init__(self):
        if self.subsets is None:
            self.subsets = [10, 20, 30, 34]


@dataclass
class SubsetResult:
    """Result for one pattern subset."""

    subset_size: int
    patterns_used: List[str]
    total_trades: int
    win_rate: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    turnover: float
    concentration_hhi: float
    runtime_seconds: float


class AblationStudyRunner:
    """
    Run ablation studies to answer Q7.

    Tests whether 34 patterns provide marginal benefit over
    smaller subsets (10, 20, 30).
    """

    def __init__(
        self,
        signal_matrix: pd.DataFrame,
        returns: pd.DataFrame,
        config: Optional[AblationStudyConfig] = None,
    ):
        """
        Args:
            signal_matrix: Binary signal matrix (bars x patterns)
            returns: Pattern returns DataFrame
            config: Configuration
        """
        self.signal_matrix = signal_matrix
        self.returns = returns
        self.config = config or AblationStudyConfig()

        np.random.seed(self.config.random_seed)

    def rank_patterns_by_sharpe(
        self,
        returns: Optional[pd.DataFrame] = None,
    ) -> List[Tuple[str, float]]:
        """
        Rank patterns by individual Sharpe ratio.

        Args:
            returns: Returns DataFrame (uses self.returns if None)

        Returns:
            List of (pattern_name, sharpe) tuples, sorted descending
        """
        if returns is None:
            returns = self.returns

        sharpe_ratios = []
        for pattern in returns.columns:
            pattern_returns = returns[pattern]
            if pattern_returns.std() > 0:
                sharpe = pattern_returns.mean() / pattern_returns.std() * np.sqrt(252)
            else:
                sharpe = 0.0
            sharpe_ratios.append((pattern, sharpe))

        return sorted(sharpe_ratios, key=lambda x: x[1], reverse=True)

    def select_top_patterns(self, n: int) -> List[str]:
        """
        Select top N patterns by Sharpe ratio.

        Args:
            n: Number of patterns to select

        Returns:
            List of pattern names
        """
        ranked = self.rank_patterns_by_sharpe()
        return [pattern for pattern, _ in ranked[:n]]

    def calculate_concentration_hhi(
        self,
        weights: np.ndarray,
    ) -> float:
        """
        Calculate Herfindahl-Hirschman Index (concentration measure).

        Args:
            weights: Portfolio weights

        Returns:
            HHI (0 = perfectly diversified, 1 = single position)
        """
        return float(np.sum(weights**2))

    def simulate_subset_performance(
        self,
        pattern_subset: List[str],
        returns: pd.DataFrame,
        signal_matrix: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Simulate performance of a pattern subset.

        This is a simplified simulation - in production would run
        full backtest with these patterns only.

        Args:
            pattern_subset: List of patterns to include
            returns: Full returns DataFrame
            signal_matrix: Full signal matrix

        Returns:
            Dictionary with performance metrics
        """
        subset_returns = returns[pattern_subset]
        subset_signals = signal_matrix[pattern_subset]

        equal_weight = 1.0 / len(pattern_subset)
        portfolio_returns = subset_returns.mean(axis=1)

        total_return = float((1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1)
        sharpe = float(portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252))
        sortino = self._calculate_sortino(portfolio_returns)
        max_dd = self._calculate_max_drawdown(portfolio_returns)

        win_rate = float((portfolio_returns > 0).mean())
        profit_factor = self._calculate_profit_factor(portfolio_returns)

        turnover = float(subset_signals.sum().sum() / len(subset_signals))

        concentration = self.calculate_concentration_hhi(
            np.ones(len(pattern_subset)) * equal_weight
        )

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "turnover": turnover,
            "concentration_hhi": concentration,
        }

    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            return float(returns.mean() / downside.std() * np.sqrt(252))
        return 0.0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min())

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor."""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses == 0:
            return float("inf") if gains > 0 else 0.0
        return float(gains / losses)

    def run_ablation_study(
        self,
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        Run ablation study across all subsets.

        Args:
            progress_callback: Optional callback for progress

        Returns:
            DataFrame with results for each subset
        """
        results = []

        for i, subset_size in enumerate(self.config.subsets):
            print(f"\nTesting subset: Top {subset_size} patterns")

            start_time = time.time()

            patterns = self.select_top_patterns(subset_size)
            metrics = self.simulate_subset_performance(
                pattern_subset=patterns,
                returns=self.returns,
                signal_matrix=self.signal_matrix,
            )

            runtime = time.time() - start_time

            result = SubsetResult(
                subset_size=subset_size,
                patterns_used=patterns,
                total_trades=int(metrics["turnover"]),
                win_rate=metrics["win_rate"],
                total_return_pct=metrics["total_return"],
                sharpe_ratio=metrics["sharpe_ratio"],
                sortino_ratio=metrics["sortino_ratio"],
                max_drawdown_pct=metrics["max_drawdown"],
                profit_factor=metrics["profit_factor"],
                turnover=metrics["turnover"],
                concentration_hhi=metrics["concentration_hhi"],
                runtime_seconds=runtime,
            )

            results.append(asdict(result))

            print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}")
            print(f"  Return: {metrics['total_return_pct']:.2%}")
            print(f"  MaxDD: {metrics['max_drawdown_pct']:.2%}")

            if progress_callback:
                progress_callback(i + 1, len(self.config.subsets))

        df = pd.DataFrame(results)
        return df

    def analyze_marginal_contribution(
        self,
        results_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate marginal contribution of each pattern group.

        Args:
            results_df: Results from run_ablation_study()

        Returns:
            DataFrame with marginal contributions
        """
        contributions = []

        for i in range(1, len(results_df)):
            prev = results_df.iloc[i - 1]
            curr = results_df.iloc[i]

            delta_patterns = curr["subset_size"] - prev["subset_size"]
            delta_sharpe = curr["sharpe_ratio"] - prev["sharpe_ratio"]
            delta_return = curr["total_return_pct"] - prev["total_return_pct"]
            delta_mdd = curr["max_drawdown_pct"] - prev["max_drawdown_pct"]

            marginal_sharpe_per_pattern = delta_sharpe / delta_patterns
            marginal_return_per_pattern = delta_return / delta_patterns

            contributions.append(
                {
                    "from_subset": int(prev["subset_size"]),
                    "to_subset": int(curr["subset_size"]),
                    "patterns_added": delta_patterns,
                    "delta_sharpe": delta_sharpe,
                    "delta_return": delta_return,
                    "delta_mdd": delta_mdd,
                    "marginal_sharpe_per_pattern": marginal_sharpe_per_pattern,
                    "marginal_return_per_pattern": marginal_return_per_pattern,
                    "diminishing_returns": delta_sharpe < 0,
                }
            )

        return pd.DataFrame(contributions)

    def get_optimal_subset(
        self,
        results_df: pd.DataFrame,
        max_subset_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Determine optimal subset size based on Sharpe ratio.

        Args:
            results_df: Results from run_ablation_study()
            max_subset_size: Maximum allowable subset size

        Returns:
            Dictionary with optimal subset recommendation
        """
        if max_subset_size:
            candidates = results_df[results_df["subset_size"] <= max_subset_size]
        else:
            candidates = results_df

        if len(candidates) == 0:
            return {"error": "No valid subsets"}

        best_idx = candidates["sharpe_ratio"].idxmax()
        best = results_df.loc[best_idx]

        return {
            "optimal_subset_size": int(best["subset_size"]),
            "sharpe_ratio": float(best["sharpe_ratio"]),
            "total_return": float(best["total_return_pct"]),
            "max_drawdown": float(best["max_drawdown_pct"]),
            "patterns": best["patterns_used"],
            "recommendation": f"Use top {int(best['subset_size'])} patterns (Sharpe={best['sharpe_ratio']:.3f})",
        }


def run_full_study(
    signal_matrix_path: str,
    Returns_path: str,
    output_dir: str,
    subsets: List[int] = None,
) -> Dict[str, Any]:
    """
    Run complete ablation study.

    Args:
        signal_matrix_path: Path to signal matrix CSV
        returns_path: Path to returns CSV
        output_dir: Output directory
        subsets: Subset sizes to test

    Returns:
        Dictionary with all results
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    signal_matrix = pd.read_csv(signal_matrix_path, index_col=0, parse_dates=True)
    returns = pd.read_csv(Returns_path, index_col=0, parse_dates=True)

    print(f"  Signals: {signal_matrix.shape}")
    print(f"  Returns: {returns.shape}")

    config = AblationStudyConfig(subsets=subsets, output_dir=output_dir)
    runner = AblationStudyRunner(signal_matrix, returns, config)

    print("\nRunning ablation study...")
    results_df = runner.run_ablation_study()

    print("\nAnalyzing marginal contributions...")
    marginal_df = runner.analyze_marginal_contribution(results_df)

    print("\nFinding optimal subset...")
    optimal = runner.get_optimal_subset(results_df)

    print("\n=== ABLATION STUDY RESULTS ===\n")
    print("Performance by subset size:")
    print(
        results_df[
            ["subset_size", "sharpe_ratio", "total_return_pct", "max_drawdown_pct"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nMarginal contribution analysis:")
    print(
        marginal_df[
            [
                "from_subset",
                "to_subset",
                "delta_sharpe",
                "marginal_sharpe_per_pattern",
                "diminishing_returns",
            ]
        ].to_string(index=False)
    )

    print(f"\nOptimal subset: {optimal['recommendation']}")

    # Save results
    results = {
        "subset_results": results_df.to_dict(orient="records"),
        "marginal_contribution": marginal_df.to_dict(orient="records"),
        "optimal_subset": optimal,
    }

    json_path = os.path.join(output_dir, "ablation_study_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {json_path}")

    results_df.to_csv(os.path.join(output_dir, "subset_performance.csv"), index=False)
    marginal_df.to_csv(os.path.join(output_dir, "marginal_contribution.csv"), index=False)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Q7 ablation study (34 vs 10 patterns)")
    parser.add_argument(
        "--signal-matrix",
        type=str,
        required=True,
        help="Path to signal matrix CSV (bars x patterns)",
    )
    parser.add_argument(
        "--returns",
        type=str,
        required=True,
        help="Path to returns CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/ablation_study",
        help="Output directory",
    )
    parser.add_argument(
        "--subsets",
        type=int,
        nargs="+",
        default=[10, 20, 30, 34],
        help="Subset sizes to test",
    )

    args = parser.parse_args()

    if not os.path.exists(args.signal_matrix):
        print(f"Error: Signal matrix not found: {args.signal_matrix}")
        sys.exit(1)

    if not os.path.exists(args.returns):
        print(f"Error: Returns not found: {args.returns}")
        sys.exit(1)

    run_full_study(
        signal_matrix_path=args.signal_matrix,
        returns_path=args.returns,
        output_dir=args.output_dir,
        subsets=args.subsets,
    )

    print("\n✅ Q7 Ablation study complete!")


if __name__ == "__main__":
    main()
