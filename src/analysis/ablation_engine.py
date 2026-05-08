"""
Ablation Engine - Layer 3 of Contribution Analysis

Quantify each pattern's marginal contribution to the system by running leave-one-out ablation.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Type

import pandas as pd

from src.strategies.backtest_py.multi_pattern_strategy_optimized import (
    MultiPatternStrategyOptimized,
)


@dataclass
class AblationResult:
    """Result of a single ablation run."""

    excluded_pattern: str
    total_trades: int
    win_rate: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    equity_final: float
    duration_seconds: float


class AblationEngine:
    """Leave-one-out ablation analysis for pattern contribution."""

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_class: Type = MultiPatternStrategyOptimized,
        cash: float = 100000,
        commission: float = 0.001,
        base_params: Optional[Dict[str, Any]] = None,
        output_dir: str = "reports/ablation",
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
        self.baseline_result: Optional[Dict[str, Any]] = None
        self.ablation_results: List[AblationResult] = []

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
        exclude_patterns: Optional[List[str]] = None,
        include_patterns_only: Optional[List[str]] = None,
        min_confluence_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a single backtest with specified pattern configuration.

        Args:
            exclude_patterns: Patterns to exclude
            include_patterns_only: Patterns to include exclusively
            min_confluence_count: Override min_confluence_count

        Returns:
            Dictionary with backtest results
        """
        from src.strategies.backtest_py.runner import BacktestPyRunner

        # Build strategy parameters
        params = self.base_params.copy()

        if exclude_patterns:
            params["exclude_patterns"] = ",".join(exclude_patterns)

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

    def run_baseline(self) -> Dict[str, Any]:
        """
        Run full system (all patterns) as baseline.

        Returns:
            Dictionary with baseline results
        """
        if self.verbose:
            print("Running baseline backtest (all patterns)...")
        self.baseline_result = self._run_backtest()
        if self.verbose:
            print(f"  Return: {self.baseline_result['total_return_pct']:.2%}")
            print(f"  Sharpe: {self.baseline_result['sharpe_ratio']:.3f}")
            print(f"  Trades: {self.baseline_result['total_trades']}")
        return self.baseline_result

    def run_ablation(self, exclude_pattern: str) -> AblationResult:
        """
        Run system with one pattern excluded.

        Args:
            exclude_pattern: Pattern name to exclude

        Returns:
            AblationResult with ablation metrics
        """
        if self.verbose:
            print(f"Running ablation for: {exclude_pattern}")
        results = self._run_backtest(exclude_patterns=[exclude_pattern])

        ablation_result = AblationResult(
            excluded_pattern=exclude_pattern,
            total_trades=results["total_trades"],
            win_rate=results["win_rate"],
            total_return_pct=results["total_return_pct"],
            sharpe_ratio=results["sharpe_ratio"],
            sortino_ratio=results.get("sortino_ratio", 0.0),
            max_drawdown_pct=results["max_drawdown_pct"],
            profit_factor=results["profit_factor"],
            equity_final=results["equity_final"],
            duration_seconds=results["duration_seconds"],
        )

        if self.verbose:
            print(f"  Return: {ablation_result.total_return_pct:.2%}")
            print(f"  Sharpe: {ablation_result.sharpe_ratio:.3f}")
            print(f"  Trades: {ablation_result.total_trades}")

        return ablation_result

    def run_full_ablation(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        """
        Run leave-one-out for ALL patterns.

        Args:
            progress_callback: Callback for progress updates (current, total)

        Returns:
            DataFrame sorted by marginal contribution
        """
        # Auto-cache: skip if results already exist on disk
        if self.load_results():
            return self.get_contribution_report()

        # Run baseline if not already done
        if self.baseline_result is None:
            self.run_baseline()

        # Run ablation for each pattern in parallel
        self.ablation_results = []
        total_patterns = len(self.all_patterns)

        def _run_one(pattern: str) -> AblationResult:
            with redirect_stdout(StringIO()):
                return self.run_ablation(pattern)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_run_one, p): p for p in self.all_patterns}
            for i, future in enumerate(as_completed(futures)):
                if progress_callback:
                    progress_callback(i + 1, total_patterns)
                try:
                    result = future.result()
                    self.ablation_results.append(result)
                except Exception as e:
                    pattern = futures[future]
                    print(f"  [SKIP] Ablation for {pattern} failed: {e}")

        # Convert to DataFrame (re-sort by pattern order for consistency)
        df = self.get_contribution_report()

        # Save results
        self.save_results()

        return df

    def get_contribution_report(self) -> pd.DataFrame:
        """
        Compare each ablation to baseline.

        Returns:
            DataFrame with columns: pattern, delta_return, delta_sharpe, delta_win_rate,
                     delta_trades, contribution_rank
        """
        if not self.ablation_results or self.baseline_result is None:
            return pd.DataFrame()

        baseline_return = self.baseline_result["total_return_pct"]
        baseline_sharpe = self.baseline_result["sharpe_ratio"]
        baseline_win_rate = self.baseline_result["win_rate"]
        baseline_trades = self.baseline_result["total_trades"]

        results = []
        for ablation in self.ablation_results:
            delta_return = baseline_return - ablation.total_return_pct
            delta_sharpe = baseline_sharpe - ablation.sharpe_ratio
            delta_win_rate = baseline_win_rate - ablation.win_rate
            delta_trades = baseline_trades - ablation.total_trades

            results.append(
                {
                    "pattern_name": ablation.excluded_pattern,
                    "baseline_return": baseline_return,
                    "ablated_return": ablation.total_return_pct,
                    "delta_return": delta_return,
                    "baseline_sharpe": baseline_sharpe,
                    "ablated_sharpe": ablation.sharpe_ratio,
                    "delta_sharpe": delta_sharpe,
                    "baseline_win_rate": baseline_win_rate,
                    "ablated_win_rate": ablation.win_rate,
                    "delta_win_rate": delta_win_rate,
                    "baseline_trades": baseline_trades,
                    "ablated_trades": ablation.total_trades,
                    "delta_trades": delta_trades,
                }
            )

        df = pd.DataFrame(results)

        # Sort by delta_sharpe (most positive = most important pattern)
        df = df.sort_values("delta_sharpe", ascending=False).reset_index(drop=True)

        # Add contribution rank
        df["contribution_rank"] = range(1, len(df) + 1)

        return df

    def save_results(self, path: Optional[str] = None) -> None:
        """
        Cache ablation results to disk (JSON).

        Args:
            path: Optional custom path, defaults to output_dir/ablation_results.json
        """
        if path is None:
            path = os.path.join(self.output_dir, "ablation_results.json")

        # Convert results to serializable format
        data = {
            "baseline": self.baseline_result,
            "ablation_results": [asdict(r) for r in self.ablation_results],
            "all_patterns": self.all_patterns,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Results saved to {path}")

    def load_results(self, path: Optional[str] = None) -> bool:
        """
        Load cached ablation results.

        Args:
            path: Optional custom path, defaults to output_dir/ablation_results.json

        Returns:
            True if loaded successfully, False otherwise
        """
        if path is None:
            path = os.path.join(self.output_dir, "ablation_results.json")

        if not os.path.exists(path):
            return False

        try:
            with open(path, "r") as f:
                data = json.load(f)

            self.baseline_result = data.get("baseline")
            self.ablation_results = [AblationResult(**r) for r in data.get("ablation_results", [])]
            self.all_patterns = data.get("all_patterns", self.all_patterns)

            print(f"Loaded results from {path}")
            return True
        except Exception as e:
            print(f"Error loading results: {e}")
            return False

    def run_solo_backtest(
        self,
        pattern_name: str,
        min_confluence_count: int = 1,
    ) -> Dict[str, Any]:
        """
        Run backtest with ONLY this pattern active (threshold=1).

        Args:
            pattern_name: Pattern to run solo
            min_confluence_count: Minimum confluence count (should be 1 for solo)

        Returns:
            Dictionary with solo backtest results
        """
        if self.verbose:
            print(f"Running solo backtest for: {pattern_name}")
        results = self._run_backtest(
            include_patterns_only=[pattern_name],
            min_confluence_count=min_confluence_count,
        )

        if self.verbose:
            print(f"  Return: {results['total_return_pct']:.2%}")
            print(f"  Sharpe: {results['sharpe_ratio']:.3f}")
            print(f"  Trades: {results['total_trades']}")

        return results

    def run_all_solo_backtests(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        """
        Run solo backtests for all patterns.

        Args:
            progress_callback: Callback for progress updates (current, total)

        Returns:
            DataFrame with solo backtest results for each pattern
        """
        # Auto-cache: try loading cached results first
        solo_path = os.path.join(self.output_dir, "solo_results.json")
        if os.path.exists(solo_path):
            try:
                df = pd.read_json(solo_path)
                if len(df) > 0:
                    print(f"Loaded {len(df)} solo results from cache")
                    return df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
            except Exception:
                pass

        results = []
        total_patterns = len(self.all_patterns)

        def _run_one(pattern: str) -> Optional[Dict[str, Any]]:
            try:
                with redirect_stdout(StringIO()):
                    result = self.run_solo_backtest(pattern)
                result["pattern_name"] = pattern
                return result
            except Exception as e:
                print(f"  [SKIP] Solo backtest for {pattern} failed: {e}")
                return None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_run_one, p): p for p in self.all_patterns}
            for i, future in enumerate(as_completed(futures)):
                if progress_callback:
                    progress_callback(i + 1, total_patterns)
                r = future.result()
                if r is not None:
                    results.append(r)

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
            self.save_solo_results(df)

        return df

    def get_solo_results(self) -> Optional[pd.DataFrame]:
        """
        Get solo backtest results if available.

        Returns:
            DataFrame with solo results or None
        """
        solo_path = os.path.join(self.output_dir, "solo_results.json")
        if not os.path.exists(solo_path):
            return None

        try:
            with open(solo_path, "r") as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except Exception:
            return None

    def save_solo_results(self, results: pd.DataFrame) -> None:
        """
        Save solo backtest results to disk.

        Args:
            results: DataFrame with solo results
        """
        path = os.path.join(self.output_dir, "solo_results.json")
        results.to_json(path, orient="records", indent=2)
        print(f"Solo results saved to {path}")

    def get_top_contributors(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N patterns by contribution.

        Args:
            n: Number of top patterns to return

        Returns:
            DataFrame with top contributors
        """
        report = self.get_contribution_report()
        if report.empty:
            return pd.DataFrame()

        return report.head(n)

    def get_bottom_contributors(self, n: int = 10) -> pd.DataFrame:
        """
        Get bottom N patterns by contribution (negative contributors).

        Args:
            n: Number of bottom patterns to return

        Returns:
            DataFrame with bottom contributors
        """
        report = self.get_contribution_report()
        if report.empty:
            return pd.DataFrame()

        return report.tail(n)

    def get_removal_candidates(self, min_delta_sharpe: float = 0.0) -> pd.DataFrame:
        """
        Get patterns that could be removed (positive delta_sharpe = removal improves system).

        Args:
            min_delta_sharpe: Minimum delta_sharpe to be considered a removal candidate

        Returns:
            DataFrame with removal candidates
        """
        report = self.get_contribution_report()
        if report.empty:
            return pd.DataFrame()

        candidates = report[report["delta_sharpe"] > min_delta_sharpe]
        return candidates.sort_values("delta_sharpe", ascending=False)

    def __repr__(self) -> str:
        """String representation."""
        n_patterns = len(self.all_patterns)
        n_ablations = len(self.ablation_results)
        return f"AblationEngine(patterns={n_patterns}, ablations={n_ablations})"
