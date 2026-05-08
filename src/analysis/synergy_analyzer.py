"""
Synergy Analyzer - Layer 4 of Contribution Analysis

Analyze pairwise (and optionally higher-order) interactions between patterns
to identify complementary pairs and conflicting pairs.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from io import StringIO
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from src.strategies.backtest_py.multi_pattern_strategy_optimized import (
    MultiPatternStrategyOptimized,
)


@dataclass
class SynergyResult:
    """Result of a pairwise synergy analysis."""

    pattern_a: str
    pattern_b: str
    solo_a_return: float
    solo_b_return: float
    pair_return: float
    expected_return: float  # solo_a + solo_b (if independent)
    synergy_score: float  # pair_return - expected_return
    solo_a_sharpe: float
    solo_b_sharpe: float
    pair_sharpe: float
    synergy_sharpe: float  # pair_sharpe - max(solo_a, solo_b)
    co_occurrence_count: int  # how many bars both fired
    co_trade_count: int  # how many trades included both
    co_trade_win_rate: float


class SynergyAnalyzer:
    """Analyze pairwise pattern interactions."""

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_class: Type = MultiPatternStrategyOptimized,
        cash: float = 100000,
        commission: float = 0.001,
        base_params: Optional[Dict[str, Any]] = None,
        output_dir: str = "reports/synergy",
        max_workers: int = 1,
        verbose: bool = False,
        quick_test: bool = False,
        quick_test_patterns: int = 5,
    ):
        """
        Args:
            data: OHLCV DataFrame
            strategy_class: Strategy class to use
            cash: Initial cash
            commission: Commission rate
            base_params: Base strategy parameters
            output_dir: Directory to cache results
            max_workers: Max parallel threads for backtest execution
            verbose: Print per-backtest details
            quick_test: If True, only test top N patterns
            quick_test_patterns: Number of patterns to test in quick_test mode
        """
        self.data = data
        self.strategy_class = strategy_class
        self.cash = cash
        self.commission = commission
        self.base_params = base_params or {}
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.verbose = verbose
        self.quick_test = quick_test
        self.quick_test_patterns = quick_test_patterns

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Results storage
        self.solo_results: Dict[str, Dict[str, Any]] = {}
        self.synergy_results: List[SynergyResult] = []

        # Get all pattern names
        self.all_patterns = self._get_all_pattern_names()

        if quick_test:
            self.all_patterns = self.all_patterns[:quick_test_patterns]

    def _get_all_pattern_names(self) -> List[str]:
        """
        Get all pattern names from the strategy.

        Returns:
            List of pattern names
        """
        # Create a mock strategy instance with just the attributes needed for _init_patterns
        # This avoids needing broker/data from backtesting.py
        temp_strategy = object.__new__(self.strategy_class)
        temp_strategy.include_patterns_only = ""
        temp_strategy.exclude_patterns = ""
        # Call _init_patterns on the mock instance
        patterns = temp_strategy._init_patterns()
        return [p.name for p in patterns]

    def _run_backtest(
        self,
        include_patterns_only: Optional[List[str]] = None,
        min_confluence_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a single backtest with specified pattern configuration.

        Args:
            include_patterns_only: Patterns to include exclusively
            min_confluence_count: Override min_confluence_count

        Returns:
            Dictionary with backtest results
        """
        from src.strategies.backtest_py.runner import BacktestPyRunner

        # Build strategy parameters
        params = self.base_params.copy()

        if include_patterns_only:
            params["include_patterns_only"] = ",".join(include_patterns_only)

        if min_confluence_count is not None:
            params["min_confluence_count"] = min_confluence_count

        # Run backtest
        start_time = time.time()
        runner = BacktestPyRunner(
            data=self.data,
            cash=self.cash,
            commission=self.commission,
            exclusive_orders=True,
            verbose=self.verbose,
        )

        results = runner.run(strategy_class=self.strategy_class, **params)
        duration = time.time() - start_time

        # Extract metrics
        return {
            "total_trades": results.get("total_trades", 0),
            "win_rate": results.get("win_rate", 0.0),
            "total_return_pct": results.get("total_return", 0.0),
            "sharpe_ratio": results.get("sharpe_ratio", 0.0),
            "sortino_ratio": results.get("sortino_ratio", 0.0),
            "max_drawdown_pct": results.get("max_drawdown", 0.0),
            "profit_factor": results.get("profit_factor", 0.0),
            "equity_final": results.get("equity_final", self.cash),
            "duration_seconds": duration,
        }

    def run_solo(self, pattern_name: str) -> Dict[str, Any]:
        """
        Run backtest with ONLY this pattern active (threshold=1).

        Args:
            pattern_name: Pattern to run solo

        Returns:
            Dictionary with solo backtest results
        """
        print(f"Running solo backtest for: {pattern_name}")
        results = self._run_backtest(
            include_patterns_only=[pattern_name],
            min_confluence_count=1,  # Solo runs need threshold=1
        )

        self.solo_results[pattern_name] = results

        print(f"  Return: {results['total_return_pct']:.2%}")
        print(f"  Sharpe: {results['sharpe_ratio']:.3f}")
        print(f"  Trades: {results['total_trades']}")

        return results

    def run_pair(self, pattern_a: str, pattern_b: str) -> Dict[str, Any]:
        """
        Run backtest with ONLY these two patterns active.

        Args:
            pattern_a: First pattern
            pattern_b: Second pattern

        Returns:
            Dictionary with pair backtest results
        """
        print(f"Running pair backtest for: {pattern_a} + {pattern_b}")
        results = self._run_backtest(
            include_patterns_only=[pattern_a, pattern_b],
            min_confluence_count=1,  # Pair runs need threshold=1
        )

        print(f"  Return: {results['total_return_pct']:.2%}")
        print(f"  Sharpe: {results['sharpe_ratio']:.3f}")
        print(f"  Trades: {results['total_trades']}")

        return results

    def _compute_synergy(
        self,
        pattern_a: str,
        pattern_b: str,
        solo_a: Dict[str, Any],
        solo_b: Dict[str, Any],
        pair: Dict[str, Any],
    ) -> SynergyResult:
        """
        Compute synergy score for a pattern pair.

        Args:
            pattern_a: First pattern name
            pattern_b: Second pattern name
            solo_a: Solo results for pattern A
            solo_b: Solo results for pattern B
            pair: Pair results for patterns A and B

        Returns:
            SynergyResult with computed metrics
        """
        # Calculate expected return (if independent)
        expected_return = solo_a["total_return_pct"] + solo_b["total_return_pct"]

        # Calculate synergy score
        synergy_score = pair["total_return_pct"] - expected_return

        # Calculate synergy Sharpe
        max_solo_sharpe = max(solo_a["sharpe_ratio"], solo_b["sharpe_ratio"])
        synergy_sharpe = pair["sharpe_ratio"] - max_solo_sharpe

        return SynergyResult(
            pattern_a=pattern_a,
            pattern_b=pattern_b,
            solo_a_return=solo_a["total_return_pct"],
            solo_b_return=solo_b["total_return_pct"],
            pair_return=pair["total_return_pct"],
            expected_return=expected_return,
            synergy_score=synergy_score,
            solo_a_sharpe=solo_a["sharpe_ratio"],
            solo_b_sharpe=solo_b["sharpe_ratio"],
            pair_sharpe=pair["sharpe_ratio"],
            synergy_sharpe=synergy_sharpe,
            co_occurrence_count=0,  # Would need signal log data
            co_trade_count=0,  # Would need signal log data
            co_trade_win_rate=0.0,  # Would need signal log data
        )

    def run_full_pairwise(
        self,
        top_n: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        """
        Run pairwise analysis for all pattern pairs.

        Args:
            top_n: Only analyze top N patterns by ablation rank (saves time)
            progress_callback: Progress callback (current, total)

        Returns:
            DataFrame with synergy scores for all pairs
        """
        # Auto-cache: skip if results already exist on disk
        if self.load_results():
            return self.get_synergy_matrix()

        # Determine which patterns to analyze
        patterns_to_analyze = self.all_patterns
        if top_n is not None:
            patterns_to_analyze = self.all_patterns[:top_n]

        # Run solo backtests in parallel
        print("Running solo backtests...")

        def _run_solo(pattern: str) -> Tuple[str, Optional[Dict[str, Any]]]:
            if pattern in self.solo_results:
                return (pattern, self.solo_results[pattern])
            try:
                with redirect_stdout(StringIO()):
                    result = self.run_solo(pattern)
                return (pattern, result)
            except Exception as e:
                print(f"  [SKIP] Solo for {pattern} failed: {e}")
                return (pattern, None)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_run_solo, p) for p in patterns_to_analyze]
            for future in as_completed(futures):
                name, result = future.result()
                if result is not None:
                    self.solo_results[name] = result

        # Run pair backtests in parallel
        pairs = list(combinations(patterns_to_analyze, 2))
        total_pairs = len(pairs)

        print(f"\nRunning {total_pairs} pair backtests...")
        self.synergy_results = []

        def _run_pair(pa: str, pb: str) -> SynergyResult:
            with redirect_stdout(StringIO()):
                pair_result = self.run_pair(pa, pb)
            solo_a = self.solo_results.get(pa, {})
            solo_b = self.solo_results.get(pb, {})
            return self._compute_synergy(pa, pb, solo_a, solo_b, pair_result)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_run_pair, pa, pb): (pa, pb) for pa, pb in pairs}
            for i, future in enumerate(as_completed(futures)):
                if progress_callback:
                    progress_callback(i + 1, total_pairs)
                try:
                    self.synergy_results.append(future.result())
                except Exception as e:
                    pa, pb = futures[future]
                    print(f"  [SKIP] Pair {pa} x {pb} failed: {e}")

        # Convert to DataFrame
        df = self.get_synergy_matrix()

        # Save results
        self.save_results()

        return df

    def get_synergy_matrix(self) -> pd.DataFrame:
        """
        NxN matrix of synergy scores.

        Returns:
            DataFrame with synergy scores
        """
        if not self.synergy_results:
            return pd.DataFrame()

        # Get unique patterns
        patterns = sorted(
            set(
                [r.pattern_a for r in self.synergy_results]
                + [r.pattern_b for r in self.synergy_results]
            )
        )
        n = len(patterns)

        # Initialize matrix
        matrix = np.zeros((n, n))

        # Build pattern name to index mapping
        pattern_to_idx = {name: idx for idx, name in enumerate(patterns)}

        # Fill matrix
        for result in self.synergy_results:
            idx_a = pattern_to_idx[result.pattern_a]
            idx_b = pattern_to_idx[result.pattern_b]
            matrix[idx_a, idx_b] = result.synergy_score
            matrix[idx_b, idx_a] = result.synergy_score

        # Convert to DataFrame
        df = pd.DataFrame(matrix, index=patterns, columns=patterns)
        return df

    def get_complementary_pairs(self, min_synergy: float = 0.0) -> List[Tuple[str, str]]:
        """
        Pairs that work better together than expected.

        Args:
            min_synergy: Minimum synergy score to be considered complementary

        Returns:
            List of (pattern_a, pattern_b) tuples
        """
        if not self.synergy_results:
            return []

        complementary = []
        for result in self.synergy_results:
            if result.synergy_score > min_synergy:
                complementary.append((result.pattern_a, result.pattern_b))

        return complementary

    def get_conflicting_pairs(self, max_synergy: float = 0.0) -> List[Tuple[str, str]]:
        """
        Pairs that conflict (combined worse than solo).

        Args:
            max_synergy: Maximum synergy score to be considered conflicting

        Returns:
            List of (pattern_a, pattern_b) tuples
        """
        if not self.synergy_results:
            return []

        conflicting = []
        for result in self.synergy_results:
            if result.synergy_score < max_synergy:
                conflicting.append((result.pattern_a, result.pattern_b))

        return conflicting

    def get_best_synergy_pairs(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N pairs by synergy score.

        Args:
            n: Number of top pairs to return

        Returns:
            DataFrame with best synergy pairs
        """
        if not self.synergy_results:
            return pd.DataFrame()

        # Sort by synergy_score descending
        sorted_results = sorted(self.synergy_results, key=lambda r: r.synergy_score, reverse=True)
        top_results = sorted_results[:n]

        data = []
        for result in top_results:
            data.append(
                {
                    "pattern_a": result.pattern_a,
                    "pattern_b": result.pattern_b,
                    "synergy_score": result.synergy_score,
                    "synergy_sharpe": result.synergy_sharpe,
                    "pair_return": result.pair_return,
                    "expected_return": result.expected_return,
                    "solo_a_return": result.solo_a_return,
                    "solo_b_return": result.solo_b_return,
                }
            )

        return pd.DataFrame(data)

    def get_worst_synergy_pairs(self, n: int = 10) -> pd.DataFrame:
        """
        Get bottom N pairs by synergy score.

        Args:
            n: Number of worst pairs to return

        Returns:
            DataFrame with worst synergy pairs
        """
        if not self.synergy_results:
            return pd.DataFrame()

        # Sort by synergy_score ascending
        sorted_results = sorted(self.synergy_results, key=lambda r: r.synergy_score)
        worst_results = sorted_results[:n]

        data = []
        for result in worst_results:
            data.append(
                {
                    "pattern_a": result.pattern_a,
                    "pattern_b": result.pattern_b,
                    "synergy_score": result.synergy_score,
                    "synergy_sharpe": result.synergy_sharpe,
                    "pair_return": result.pair_return,
                    "expected_return": result.expected_return,
                    "solo_a_return": result.solo_a_return,
                    "solo_b_return": result.solo_b_return,
                }
            )

        return pd.DataFrame(data)

    def save_results(self, path: Optional[str] = None) -> None:
        """
        Cache synergy results to disk.

        Args:
            path: Optional custom path, defaults to output_dir/synergy_results.json
        """
        if path is None:
            path = os.path.join(self.output_dir, "synergy_results.json")

        # Convert results to serializable format
        data = {
            "solo_results": self.solo_results,
            "synergy_results": [asdict(r) for r in self.synergy_results],
            "all_patterns": self.all_patterns,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Results saved to {path}")

        # Also save solo results separately
        solo_path = os.path.join(self.output_dir, "solo_results.json")
        with open(solo_path, "w") as f:
            json.dump(self.solo_results, f, indent=2, default=str)

        print(f"Solo results saved to {solo_path}")

    def load_results(self, path: Optional[str] = None) -> bool:
        """
        Load cached synergy results.

        Args:
            path: Optional custom path, defaults to output_dir/synergy_results.json

        Returns:
            True if loaded successfully, False otherwise
        """
        if path is None:
            path = os.path.join(self.output_dir, "synergy_results.json")

        if not os.path.exists(path):
            return False

        try:
            with open(path, "r") as f:
                data = json.load(f)

            self.solo_results = data.get("solo_results", {})
            self.synergy_results = [SynergyResult(**r) for r in data.get("synergy_results", [])]
            self.all_patterns = data.get("all_patterns", self.all_patterns)

            print(f"Loaded results from {path}")
            return True
        except Exception as e:
            print(f"Error loading results: {e}")
            return False

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for synergy analysis.

        Returns:
            Dictionary with summary statistics
        """
        if not self.synergy_results:
            return {
                "total_pairs": 0,
                "complementary_pairs": 0,
                "conflicting_pairs": 0,
                "avg_synergy_score": 0.0,
            }

        total_pairs = len(self.synergy_results)
        complementary = len([r for r in self.synergy_results if r.synergy_score > 0])
        conflicting = len([r for r in self.synergy_results if r.synergy_score < 0])
        avg_synergy = np.mean([r.synergy_score for r in self.synergy_results])

        return {
            "total_pairs": total_pairs,
            "complementary_pairs": complementary,
            "conflicting_pairs": conflicting,
            "avg_synergy_score": avg_synergy,
        }

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_summary_stats()
        return (
            f"SynergyAnalyzer("
            f"patterns={len(self.all_patterns)}, "
            f"pairs={stats['total_pairs']}, "
            f"complementary={stats['complementary_pairs']}, "
            f"conflicting={stats['conflicting_pairs']})"
        )
