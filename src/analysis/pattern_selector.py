"""
Pattern Selection Framework - Main Orchestrator

Three-phase systematic pattern selection:
- Phase 1: Isolated Performance Baseline (solo backtests + noise filtering)
- Phase 2: Statistical Correlation Analysis (redundancy detection)
- Phase 3: Ablation Testing (marginal contribution analysis)

Author: Pattern Selection Framework
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PatternRole(Enum):
    """Classification of pattern roles based on contribution."""

    PRIMARY_SIGNAL = "Primary Signal"
    CONFIRMATION_FILTER = "Confirmation Filter"
    NEUTRAL = "Neutral"
    NOISE_GENERATOR = "Noise Generator"


@dataclass
class PatternSelectionConfig:
    """
    Configuration for the pattern selection framework.

    All parameters are configurable for different datasets and requirements.
    """

    # Data configuration
    data_path: str = "data/raw/SPY_daily.csv"
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Backtest configuration
    initial_equity: float = 100000.0
    commission: float = 0.001
    min_confluence_count: int = 2

    # Phase 1 thresholds (Noise Filtering)
    min_trades: int = 30
    min_sharpe: float = 0.5
    min_profit_factor: float = 1.0
    min_win_rate: float = 0.40
    max_drawdown: float = 0.30

    # Phase 2 thresholds (Correlation Analysis)
    correlation_threshold: float = 0.8

    # Phase 3 thresholds (Ablation Testing)
    positive_contribution_threshold: float = 0.0
    noise_threshold: float = -0.05
    primary_signal_threshold: float = 0.1

    # Output configuration
    output_dir: str = "reports/pattern_selection"
    cache_results: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def validate(self) -> List[str]:
        """Validate configuration parameters. Returns list of errors."""
        errors = []
        if self.min_trades < 1:
            errors.append("min_trades must be >= 1")
        if self.initial_equity <= 0:
            errors.append("initial_equity must be positive")
        if not 0 <= self.min_win_rate <= 1:
            errors.append("min_win_rate must be between 0 and 1")
        if not 0 <= self.max_drawdown <= 1:
            errors.append("max_drawdown must be between 0 and 1")
        if not -1 <= self.correlation_threshold <= 1:
            errors.append("correlation_threshold must be between -1 and 1")
        return errors


@dataclass
class SoloBacktestResult:
    """Result of a single pattern's solo backtest."""

    pattern_name: str
    total_trades: int
    win_rate: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    equity_final: float
    equity_curve: Optional[np.ndarray] = None
    passed_filter: bool = False
    filter_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            "pattern_name": self.pattern_name,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "total_return_pct": self.total_return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "profit_factor": self.profit_factor,
            "equity_final": self.equity_final,
            "passed_filter": self.passed_filter,
            "filter_reasons": self.filter_reasons,
        }
        if self.equity_curve is not None:
            d["equity_curve"] = self.equity_curve.tolist()
        return d

    def get_summary(self) -> str:
        """Get a one-line summary of the result."""
        status = "[PASS]" if self.passed_filter else "[FAIL]"
        return f"{self.pattern_name}: {status} | Trades={self.total_trades}, Sharpe={self.sharpe_ratio:.3f}, Return={self.total_return_pct:.2%}"


@dataclass
class CorrelationResult:
    """Result of correlation analysis between two patterns."""

    pattern_a: str
    pattern_b: str
    correlation: float
    is_redundant: bool
    selected_pattern: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_a": self.pattern_a,
            "pattern_b": self.pattern_b,
            "correlation": self.correlation,
            "is_redundant": self.is_redundant,
            "selected_pattern": self.selected_pattern,
            "reason": self.reason,
        }


@dataclass
class AblationContribution:
    """Result of ablation analysis for a single pattern."""

    pattern_name: str
    baseline_sharpe: float
    ablated_sharpe: float
    delta_sharpe: float
    baseline_return: float
    ablated_return: float
    delta_return: float
    contribution_rank: int = 0
    role: PatternRole = PatternRole.NEUTRAL
    keep: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_name": self.pattern_name,
            "baseline_sharpe": self.baseline_sharpe,
            "ablated_sharpe": self.ablated_sharpe,
            "delta_sharpe": self.delta_sharpe,
            "baseline_return": self.baseline_return,
            "ablated_return": self.ablated_return,
            "delta_return": self.delta_return,
            "contribution_rank": self.contribution_rank,
            "role": self.role.value,
            "keep": self.keep,
        }


@dataclass
class SelectionResult:
    """Final result of the pattern selection process."""

    # Phase 1 results
    solo_results: List[SoloBacktestResult]
    filtered_patterns: List[str]
    excluded_patterns: List[str]

    # Phase 2 results
    correlation_matrix: Optional[pd.DataFrame]
    redundant_pairs: List[CorrelationResult]
    non_redundant_patterns: List[str]

    # Phase 3 results
    ablation_contributions: List[AblationContribution]
    final_patterns: List[str]

    # Metadata
    config: PatternSelectionConfig
    execution_time_seconds: float = 0.0
    phase_timings: Dict[str, float] = field(default_factory=dict)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_patterns_tested": len(self.solo_results),
            "passed_phase1_filter": len(self.filtered_patterns),
            "excluded_phase1": len(self.excluded_patterns),
            "redundant_pairs_found": len(self.redundant_pairs),
            "after_redundancy_removal": len(self.non_redundant_patterns),
            "final_selected_patterns": len(self.final_patterns),
            "execution_time_seconds": self.execution_time_seconds,
            "phase_timings": self.phase_timings,
        }

    def get_role_distribution(self) -> Dict[str, int]:
        """Get distribution of pattern roles."""
        roles = {}
        for c in self.ablation_contributions:
            role_name = c.role.value
            roles[role_name] = roles.get(role_name, 0) + 1
        return roles


class PatternSelector:
    """
    Main orchestrator for the three-phase pattern selection framework.

    Usage:
        config = PatternSelectionConfig(data_path="data/raw/SPY_daily.csv")
        selector = PatternSelector(config)
        result = selector.run_full_selection()
    """

    def __init__(
        self,
        config: PatternSelectionConfig,
        data: Optional[pd.DataFrame] = None,
        strategy_class: Optional[Type] = None,
    ):
        """
        Initialize the pattern selector.

        Args:
            config: Configuration for the selection process
            data: Optional pre-loaded DataFrame (overrides data_path)
            strategy_class: Strategy class to use for backtests
        """
        self.config = config

        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {errors}")

        # Import strategy class if not provided
        if strategy_class is None:
            from src.strategies.backtest_py.multi_pattern_strategy_optimized import (
                MultiPatternStrategyOptimized,
            )

            strategy_class = MultiPatternStrategyOptimized

        self.strategy_class = strategy_class

        # Load data
        if data is not None:
            self.data = data
        else:
            self.data = self._load_data()

        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)

        # Get all pattern names
        self.all_patterns = self._get_all_pattern_names()

        # Results storage
        self.solo_results: List[SoloBacktestResult] = []
        self.correlation_results: List[CorrelationResult] = []
        self.ablation_results: List[AblationContribution] = []

        logger.info(f"PatternSelector initialized with {len(self.all_patterns)} patterns")

    def _load_data(self) -> pd.DataFrame:
        """Load data from configured path."""
        logger.info(f"Loading data from {self.config.data_path}")

        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"Data file not found: {self.config.data_path}")

        df = pd.read_csv(self.config.data_path, index_col=0, parse_dates=True)

        # Apply date filters
        if self.config.start_date:
            df = df[df.index >= self.config.start_date]
        if self.config.end_date:
            df = df[df.index <= self.config.end_date]

        # Ensure proper column casing
        column_mapping = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        df = df.rename(columns=column_mapping)

        logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        return df

    def _get_all_pattern_names(self) -> List[str]:
        """Get all pattern names from the strategy."""
        try:
            temp_strategy = object.__new__(self.strategy_class)
            temp_strategy.include_patterns_only = ""
            temp_strategy.exclude_patterns = ""
            patterns = temp_strategy._init_patterns()
            return [p.name for p in patterns]
        except Exception as e:
            logger.error(f"Failed to get pattern names: {e}")
            return []

    def _run_backtest(
        self,
        include_patterns_only: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        min_confluence_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a single backtest with specified pattern configuration.

        Args:
            include_patterns_only: Patterns to include exclusively
            exclude_patterns: Patterns to exclude
            min_confluence_count: Override min_confluence_count

        Returns:
            Dictionary with backtest results with normalized keys
        """
        from src.strategies.backtest_py.runner import BacktestPyRunner

        # Build strategy parameters
        params = {}

        if exclude_patterns:
            params["exclude_patterns"] = ",".join(exclude_patterns)

        if include_patterns_only:
            params["include_patterns_only"] = ",".join(include_patterns_only)

        if min_confluence_count is not None:
            params["min_confluence_count"] = min_confluence_count

        # Run backtest
        runner = BacktestPyRunner(
            data=self.data,
            cash=self.config.initial_equity,
            commission=self.config.commission,
            exclusive_orders=True,
        )

        try:
            results = runner.run(strategy_class=self.strategy_class, **params)
            
            # Extract stats from nested structure
            # BacktestPyRunner.run() returns {"stats": {...}, "equity_curve": ..., "trades": ...}
            stats = results.get("stats", results)
            
            # Extract equity curve - backtesting.py returns a DataFrame with 'Equity' column
            equity_curve_raw = results.get("equity_curve")
            equity_curve = None
            if equity_curve_raw is not None:
                # Handle DataFrame format from backtesting.py
                if hasattr(equity_curve_raw, 'Equity'):
                    # It's a DataFrame with 'Equity' column
                    equity_curve = np.array(equity_curve_raw['Equity'])
                elif isinstance(equity_curve_raw, pd.DataFrame):
                    # Try to get Equity column
                    if 'Equity' in equity_curve_raw.columns:
                        equity_curve = np.array(equity_curve_raw['Equity'])
                    else:
                        # Use first column
                        equity_curve = np.array(equity_curve_raw.iloc[:, 0])
                elif isinstance(equity_curve_raw, np.ndarray):
                    # Already an array - check dimensions
                    if equity_curve_raw.ndim == 2:
                        equity_curve = equity_curve_raw.flatten()
                    else:
                        equity_curve = equity_curve_raw
                elif isinstance(equity_curve_raw, pd.Series):
                    equity_curve = np.array(equity_curve_raw)
            
            # Map backtesting.py keys to normalized keys expected by SoloBacktestResult
            # backtesting.py uses keys like "# Trades", "Sharpe Ratio", "Win Rate [%]", etc.
            normalized = {
                "total_trades": stats.get("# Trades", 0),
                "win_rate": stats.get("Win Rate [%]", 0) / 100.0 if stats.get("Win Rate [%]", 0) != 0 else 0.0,
                "total_return": stats.get("Return [%]", 0),
                "sharpe_ratio": stats.get("Sharpe Ratio", 0),
                "sortino_ratio": stats.get("Sortino Ratio", 0),
                "max_drawdown": stats.get("Max. Drawdown [%]", 0),
                "profit_factor": stats.get("Profit Factor", 0),
                "equity_final": stats.get("Equity Final [$]", self.config.initial_equity),
                "_equity": equity_curve,
            }
            
            return normalized
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {"error": str(e)}

    # ==================== PHASE 1: Isolated Performance Baseline ====================

    def run_solo_backtest(self, pattern_name: str) -> SoloBacktestResult:
        """
        Run a solo backtest for a single pattern.

        Args:
            pattern_name: Name of the pattern to test

        Returns:
            SoloBacktestResult with performance metrics
        """
        logger.info(f"Running solo backtest for: {pattern_name}")

        results = self._run_backtest(
            include_patterns_only=[pattern_name],
            min_confluence_count=1,  # Critical: threshold=1 for solo
        )

        # Check for errors
        if "error" in results:
            logger.error(f"Backtest error for {pattern_name}: {results['error']}")
            return SoloBacktestResult(
                pattern_name=pattern_name,
                total_trades=0,
                win_rate=0.0,
                total_return_pct=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown_pct=0.0,
                profit_factor=0.0,
                equity_final=self.config.initial_equity,
                filter_reasons=[f"Backtest error: {results['error']}"],
            )

        # Extract equity curve if available
        equity_curve = None
        if "_equity" in results:
            equity_curve = np.array(results["_equity"])

        solo_result = SoloBacktestResult(
            pattern_name=pattern_name,
            total_trades=results.get("total_trades", 0),
            win_rate=results.get("win_rate", 0.0),
            total_return_pct=results.get("total_return", 0.0),
            sharpe_ratio=results.get("sharpe_ratio", 0.0),
            sortino_ratio=results.get("sortino_ratio", 0.0),
            max_drawdown_pct=results.get("max_drawdown", 0.0),
            profit_factor=results.get("profit_factor", 0.0),
            equity_final=results.get("equity_final", self.config.initial_equity),
            equity_curve=equity_curve,
        )

        logger.info(f"  {solo_result.get_summary()}")
        return solo_result

    def run_all_solo_backtests(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[SoloBacktestResult]:
        """
        Run solo backtests for all patterns.

        Args:
            progress_callback: Callback for progress updates (current, total)

        Returns:
            List of SoloBacktestResult for all patterns
        """
        print(f"\n{'=' * 60}")
        print("PHASE 1: Running Solo Backtests for All Patterns")
        print(f"{'=' * 60}")

        self.solo_results = []
        total_patterns = len(self.all_patterns)

        for i, pattern_name in enumerate(self.all_patterns):
            if progress_callback:
                progress_callback(i + 1, total_patterns)

            solo_result = self.run_solo_backtest(pattern_name)
            self.solo_results.append(solo_result)

        return self.solo_results

    def apply_noise_filter(
        self,
        solo_results: Optional[List[SoloBacktestResult]] = None,
    ) -> tuple[List[str], List[str]]:
        """
        Apply noise filter to solo backtest results.

        Args:
            solo_results: Optional override for solo results

        Returns:
            Tuple of (filtered_patterns, excluded_patterns)
        """
        if solo_results is None:
            solo_results = self.solo_results

        print(f"\n{'=' * 60}")
        print("PHASE 1: Applying Noise Filter")
        print(f"{'=' * 60}")
        print(
            f"  Thresholds: min_trades={self.config.min_trades}, "
            f"min_sharpe={self.config.min_sharpe}, "
            f"min_profit_factor={self.config.min_profit_factor}, "
            f"min_win_rate={self.config.min_win_rate:.0%}, "
            f"max_drawdown={self.config.max_drawdown:.0%}"
        )

        filtered_patterns = []
        excluded_patterns = []

        for result in solo_results:
            reasons = []

            # Check each threshold
            if result.total_trades < self.config.min_trades:
                reasons.append(
                    f"Insufficient trades ({result.total_trades} < {self.config.min_trades})"
                )

            if result.sharpe_ratio < self.config.min_sharpe:
                reasons.append(f"Low Sharpe ({result.sharpe_ratio:.3f} < {self.config.min_sharpe})")

            if result.profit_factor < self.config.min_profit_factor:
                reasons.append(
                    f"Low Profit Factor ({result.profit_factor:.3f} < {self.config.min_profit_factor})"
                )

            if result.win_rate < self.config.min_win_rate:
                reasons.append(
                    f"Low Win Rate ({result.win_rate:.1%} < {self.config.min_win_rate:.0%})"
                )

            if result.max_drawdown_pct > self.config.max_drawdown * 100:
                reasons.append(
                    f"High Drawdown ({result.max_drawdown_pct:.1%} > {self.config.max_drawdown:.0%})"
                )

            # Update result
            result.filter_reasons = reasons
            result.passed_filter = len(reasons) == 0

            if result.passed_filter:
                filtered_patterns.append(result.pattern_name)
                print(f"  [PASS] {result.pattern_name}: PASSED")
            else:
                excluded_patterns.append(result.pattern_name)
                print(f"  [FAIL] {result.pattern_name}: EXCLUDED - {'; '.join(reasons)}")

        print(f"\n  Summary: {len(filtered_patterns)} passed, {len(excluded_patterns)} excluded")

        return filtered_patterns, excluded_patterns

    # ==================== PHASE 2: Statistical Correlation Analysis ====================

    def compute_correlation_matrix(
        self,
        patterns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute correlation matrix for pattern equity curves.

        Args:
            patterns: Optional subset of patterns to analyze

        Returns:
            DataFrame with correlation matrix
        """
        if patterns is None:
            patterns = [r.pattern_name for r in self.solo_results if r.passed_filter]

        print(f"\n{'=' * 60}")
        print("PHASE 2: Computing Correlation Matrix")
        print(f"{'=' * 60}")

        # Build returns DataFrame from equity curves
        returns_dict = {}

        for result in self.solo_results:
            if result.pattern_name not in patterns:
                continue
            if result.equity_curve is None:
                logger.warning(f"No equity curve for {result.pattern_name}")
                continue

            # Calculate returns from equity curve
            equity = pd.Series(result.equity_curve)
            returns = equity.pct_change().dropna()
            returns_dict[result.pattern_name] = returns

        if not returns_dict:
            print("  Warning: No equity curves available for correlation analysis")
            return pd.DataFrame()

        # Align all returns to same index
        returns_df = pd.DataFrame(returns_dict)

        # Compute correlation matrix
        correlation_matrix = returns_df.corr(method="pearson")

        print(f"  Computed {len(correlation_matrix)}x{len(correlation_matrix)} correlation matrix")

        return correlation_matrix

    def identify_redundant_pairs(
        self,
        correlation_matrix: pd.DataFrame,
    ) -> List[CorrelationResult]:
        """
        Identify redundant pattern pairs based on correlation threshold.

        Args:
            correlation_matrix: Correlation matrix from equity curves

        Returns:
            List of CorrelationResult for redundant pairs
        """
        print(
            f"\n  Identifying redundant pairs (correlation > {self.config.correlation_threshold})..."
        )

        redundant_pairs = []
        patterns = correlation_matrix.columns.tolist()

        # Find pairs above threshold
        for i, pattern_a in enumerate(patterns):
            for j, pattern_b in enumerate(patterns):
                if i >= j:  # Skip diagonal and lower triangle
                    continue

                corr_value = correlation_matrix.loc[pattern_a, pattern_b]
                corr = float(corr_value) if hasattr(corr_value, "__float__") else 0.0

                if abs(corr) > self.config.correlation_threshold:
                    # Determine which pattern to keep
                    result_a = next(
                        (r for r in self.solo_results if r.pattern_name == pattern_a), None
                    )
                    result_b = next(
                        (r for r in self.solo_results if r.pattern_name == pattern_b), None
                    )

                    selected = None
                    reason = ""

                    if result_a and result_b:
                        # Select pattern with higher Sharpe ratio
                        if result_a.sharpe_ratio > result_b.sharpe_ratio:
                            selected = pattern_a
                            reason = f"Higher Sharpe ({result_a.sharpe_ratio:.3f} > {result_b.sharpe_ratio:.3f})"
                        else:
                            selected = pattern_b
                            reason = f"Higher Sharpe ({result_b.sharpe_ratio:.3f} > {result_a.sharpe_ratio:.3f})"

                    redundant_pairs.append(
                        CorrelationResult(
                            pattern_a=pattern_a,
                            pattern_b=pattern_b,
                            correlation=corr,
                            is_redundant=True,
                            selected_pattern=selected,
                            reason=reason,
                        )
                    )

                    print(
                        f"    Redundant: {pattern_a} ~ {pattern_b} (corr={corr:.3f}) -> Keep {selected}"
                    )

        if not redundant_pairs:
            print("    No redundant pairs found")

        return redundant_pairs

    def resolve_redundancy(
        self,
        redundant_pairs: List[CorrelationResult],
        patterns: List[str],
    ) -> List[str]:
        """
        Resolve redundancy by removing lower-performing patterns.

        Args:
            redundant_pairs: List of redundant pairs
            patterns: Current list of patterns

        Returns:
            List of non-redundant patterns
        """
        print("\n  Resolving redundancy...")

        # Build set of patterns to remove
        to_remove = set()

        for pair in redundant_pairs:
            if pair.selected_pattern:
                # Remove the pattern that was NOT selected
                if pair.pattern_a != pair.selected_pattern:
                    to_remove.add(pair.pattern_a)
                else:
                    to_remove.add(pair.pattern_b)

        non_redundant = [p for p in patterns if p not in to_remove]

        print(f"    Removed {len(to_remove)} redundant patterns: {sorted(to_remove)}")
        print(f"    Remaining: {len(non_redundant)} patterns")

        return non_redundant

    # ==================== PHASE 3: Ablation Testing ====================

    def run_ablation_analysis(
        self,
        patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[AblationContribution]:
        """
        Run leave-one-out ablation analysis.

        Args:
            patterns: Optional subset of patterns to analyze
            progress_callback: Progress callback (current, total)

        Returns:
            List of AblationContribution for each pattern
        """
        if patterns is None:
            patterns = [r.pattern_name for r in self.solo_results if r.passed_filter]

        print(f"\n{'=' * 60}")
        print("PHASE 3: Running Ablation Analysis (Leave-One-Out)")
        print(f"{'=' * 60}")

        if not patterns:
            print("  No patterns to analyze")
            return []

        # Run baseline (all patterns)
        print("  Running baseline backtest (all patterns)...")
        baseline = self._run_backtest(
            include_patterns_only=patterns,
            min_confluence_count=self.config.min_confluence_count,
        )

        baseline_sharpe = baseline.get("sharpe_ratio", 0.0)
        baseline_return = baseline.get("total_return", 0.0)

        print(f"    Baseline: Sharpe={baseline_sharpe:.3f}, Return={baseline_return:.2%}")

        # Run ablation for each pattern
        self.ablation_results = []
        total_patterns = len(patterns)

        for i, pattern_name in enumerate(patterns):
            if progress_callback:
                progress_callback(i + 1, total_patterns)

            print(f"  Running ablation for: {pattern_name}")

            # Run backtest excluding this pattern
            ablated = self._run_backtest(
                include_patterns_only=[p for p in patterns if p != pattern_name],
                min_confluence_count=self.config.min_confluence_count,
            )

            ablated_sharpe = ablated.get("sharpe_ratio", 0.0)
            ablated_return = ablated.get("total_return", 0.0)

            # Calculate delta (positive = pattern helps, negative = pattern hurts)
            delta_sharpe = baseline_sharpe - ablated_sharpe
            delta_return = baseline_return - ablated_return

            contribution = AblationContribution(
                pattern_name=pattern_name,
                baseline_sharpe=baseline_sharpe,
                ablated_sharpe=ablated_sharpe,
                delta_sharpe=delta_sharpe,
                baseline_return=baseline_return,
                ablated_return=ablated_return,
                delta_return=delta_return,
            )

            self.ablation_results.append(contribution)

            print(f"    Delta Sharpe: {delta_sharpe:.4f}, Delta Return: {delta_return:.2%}")

        # Rank by contribution
        self.ablation_results.sort(key=lambda x: x.delta_sharpe, reverse=True)
        for i, result in enumerate(self.ablation_results):
            result.contribution_rank = i + 1

        return self.ablation_results

    def classify_pattern_roles(self) -> None:
        """
        Classify each pattern into a role based on ablation results.

        Roles:
        - Primary Signal: delta_sharpe > primary_signal_threshold AND win_rate > 50%
        - Confirmation Filter: delta_sharpe > 0 AND win_rate > 45%
        - Noise Generator: delta_sharpe < noise_threshold
        - Neutral: otherwise
        """
        print("\n  Classifying pattern roles...")

        for contribution in self.ablation_results:
            # Get win rate from solo results
            solo = next(
                (r for r in self.solo_results if r.pattern_name == contribution.pattern_name), None
            )
            win_rate = solo.win_rate if solo else 0.0

            if contribution.delta_sharpe > self.config.primary_signal_threshold and win_rate > 0.5:
                contribution.role = PatternRole.PRIMARY_SIGNAL
            elif contribution.delta_sharpe > 0 and win_rate > 0.45:
                contribution.role = PatternRole.CONFIRMATION_FILTER
            elif contribution.delta_sharpe < self.config.noise_threshold:
                contribution.role = PatternRole.NOISE_GENERATOR
            else:
                contribution.role = PatternRole.NEUTRAL

            # Determine if pattern should be kept
            contribution.keep = (
                contribution.delta_sharpe > self.config.positive_contribution_threshold
            )

            print(
                f"    {contribution.pattern_name}: {contribution.role.value} "
                f"(delta_sharpe={contribution.delta_sharpe:.4f}, keep={contribution.keep})"
            )

    def get_final_selection(self) -> List[str]:
        """
        Get final list of selected patterns.

        Returns:
            List of pattern names to keep
        """
        return [c.pattern_name for c in self.ablation_results if c.keep]

    # ==================== Main Orchestrator ====================

    def run_full_selection(
        self,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> SelectionResult:
        """
        Run the complete three-phase pattern selection process.

        Args:
            progress_callback: Callback for progress updates (phase, current, total)

        Returns:
            SelectionResult with all analysis results
        """
        start_time = time.time()
        phase_timings = {}

        # Phase 1: Isolated Performance Baseline
        phase1_start = time.time()
        self.run_all_solo_backtests(
            progress_callback=lambda c, t: (
                progress_callback("phase1", c, t) if progress_callback else None
            )
        )
        filtered_patterns, excluded_patterns = self.apply_noise_filter()
        phase_timings["phase1"] = time.time() - phase1_start

        if not filtered_patterns:
            print("\nERROR: No patterns passed Phase 1 filter. Adjust thresholds and try again.")
            return SelectionResult(
                solo_results=self.solo_results,
                filtered_patterns=[],
                excluded_patterns=excluded_patterns,
                correlation_matrix=None,
                redundant_pairs=[],
                non_redundant_patterns=[],
                ablation_contributions=[],
                final_patterns=[],
                config=self.config,
                phase_timings=phase_timings,
            )

        # Phase 2: Statistical Correlation Analysis
        phase2_start = time.time()
        correlation_matrix = self.compute_correlation_matrix(filtered_patterns)

        if not correlation_matrix.empty:
            redundant_pairs = self.identify_redundant_pairs(correlation_matrix)
            non_redundant_patterns = self.resolve_redundancy(redundant_pairs, filtered_patterns)
        else:
            redundant_pairs = []
            non_redundant_patterns = filtered_patterns
        phase_timings["phase2"] = time.time() - phase2_start

        if not non_redundant_patterns:
            print(
                "\nWARNING: All patterns were removed as redundant. Using filtered patterns instead."
            )
            non_redundant_patterns = filtered_patterns

        # Phase 3: Ablation Testing
        phase3_start = time.time()
        self.run_ablation_analysis(
            non_redundant_patterns,
            progress_callback=lambda c, t: (
                progress_callback("phase3", c, t) if progress_callback else None
            ),
        )
        self.classify_pattern_roles()
        final_patterns = self.get_final_selection()
        phase_timings["phase3"] = time.time() - phase3_start

        execution_time = time.time() - start_time

        result = SelectionResult(
            solo_results=self.solo_results,
            filtered_patterns=filtered_patterns,
            excluded_patterns=excluded_patterns,
            correlation_matrix=correlation_matrix,
            redundant_pairs=redundant_pairs,
            non_redundant_patterns=non_redundant_patterns,
            ablation_contributions=self.ablation_results,
            final_patterns=final_patterns,
            config=self.config,
            execution_time_seconds=execution_time,
            phase_timings=phase_timings,
        )

        # Save results
        if self.config.cache_results:
            self.save_results(result)

        # Print summary
        self.print_summary(result)

        return result

    def save_results(self, result: SelectionResult) -> None:
        """Save all results to output directory."""
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save solo results
        solo_df = pd.DataFrame([r.to_dict() for r in result.solo_results])
        solo_df.to_csv(os.path.join(output_dir, "solo_results.csv"), index=False)

        # Save correlation matrix
        if result.correlation_matrix is not None and not result.correlation_matrix.empty:
            result.correlation_matrix.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))

        # Save ablation results
        ablation_df = pd.DataFrame([c.to_dict() for c in result.ablation_contributions])
        ablation_df.to_csv(os.path.join(output_dir, "ablation_results.csv"), index=False)

        # Save final selection
        selection_data = {
            "final_patterns": result.final_patterns,
            "filtered_patterns": result.filtered_patterns,
            "excluded_patterns": result.excluded_patterns,
            "non_redundant_patterns": result.non_redundant_patterns,
            "summary": result.get_summary(),
            "role_distribution": result.get_role_distribution(),
        }
        with open(os.path.join(output_dir, "final_selection.json"), "w") as f:
            json.dump(selection_data, f, indent=2, default=str)

        # Save config
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        print(f"\nResults saved to {output_dir}")

    def print_summary(self, result: SelectionResult) -> None:
        """Print summary of selection results."""
        summary = result.get_summary()

        print(f"\n{'=' * 60}")
        print("PATTERN SELECTION SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Total patterns tested: {summary['total_patterns_tested']}")
        print(f"  Phase 1 - Passed filter: {summary['passed_phase1_filter']}")
        print(f"  Phase 1 - Excluded: {summary['excluded_phase1']}")
        print(f"  Phase 2 - Redundant pairs: {summary['redundant_pairs_found']}")
        print(f"  Phase 2 - After redundancy removal: {summary['after_redundancy_removal']}")
        print(f"  Phase 3 - Final selected: {summary['final_selected_patterns']}")
        print(f"  Total execution time: {summary['execution_time_seconds']:.1f}s")

        # Phase timings
        if result.phase_timings:
            print("\n  Phase Timings:")
            for phase, timing in result.phase_timings.items():
                print(f"    {phase}: {timing:.1f}s")

        print(f"\n{'=' * 60}")
        print("FINAL SELECTED PATTERNS")
        print(f"{'=' * 60}")
        for i, pattern in enumerate(result.final_patterns, 1):
            contribution = next(
                (c for c in result.ablation_contributions if c.pattern_name == pattern), None
            )
            if contribution:
                print(
                    f"  {i}. {pattern} ({contribution.role.value}, delta_sharpe={contribution.delta_sharpe:.4f})"
                )
            else:
                print(f"  {i}. {pattern}")

        # Show excluded patterns
        if result.excluded_patterns:
            print(f"\n{'=' * 60}")
            print("EXCLUDED PATTERNS (Phase 1)")
            print(f"{'=' * 60}")
            for pattern in result.excluded_patterns:
                solo = next((r for r in result.solo_results if r.pattern_name == pattern), None)
                if solo:
                    print(f"  - {pattern}: {'; '.join(solo.filter_reasons)}")

        # Show noise generators
        noise_generators = [
            c for c in result.ablation_contributions if c.role == PatternRole.NOISE_GENERATOR
        ]
        if noise_generators:
            print(f"\n{'=' * 60}")
            print("NOISE GENERATORS (Phase 3)")
            print(f"{'=' * 60}")
            for c in noise_generators:
                print(f"  - {c.pattern_name}: delta_sharpe={c.delta_sharpe:.4f}")


def main():
    """Main entry point for pattern selection."""
    import argparse

    parser = argparse.ArgumentParser(description="Pattern Selection Framework")
    parser.add_argument(
        "--data", type=str, default="data/raw/SPY_daily.csv", help="Path to OHLCV data"
    )
    parser.add_argument(
        "--output", type=str, default="reports/pattern_selection", help="Output directory"
    )
    parser.add_argument("--min-trades", type=int, default=30, help="Minimum trades threshold")
    parser.add_argument(
        "--min-sharpe", type=float, default=0.5, help="Minimum Sharpe ratio threshold"
    )
    parser.add_argument(
        "--min-profit-factor", type=float, default=1.0, help="Minimum profit factor threshold"
    )
    parser.add_argument(
        "--min-win-rate", type=float, default=0.4, help="Minimum win rate threshold"
    )
    parser.add_argument(
        "--max-drawdown", type=float, default=0.3, help="Maximum drawdown threshold"
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.8,
        help="Correlation threshold for redundancy",
    )

    args = parser.parse_args()

    config = PatternSelectionConfig(
        data_path=args.data,
        output_dir=args.output,
        min_trades=args.min_trades,
        min_sharpe=args.min_sharpe,
        min_profit_factor=args.min_profit_factor,
        min_win_rate=args.min_win_rate,
        max_drawdown=args.max_drawdown,
        correlation_threshold=args.correlation_threshold,
    )

    selector = PatternSelector(config)
    result = selector.run_full_selection()

    return result


if __name__ == "__main__":
    main()
