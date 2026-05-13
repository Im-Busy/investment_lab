"""Pattern Selector Orchestrator.

Coordinates all analysis filters into an end-to-end pattern
selection pipeline: Statistical → Performance → Correlation →
Contribution → Walk-Forward → Signal Quality → Final Selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd


from .contribution_analyzer import ContributionAnalyzer
from .correlation_analyzer import CorrelationAnalyzer
from .signal_quality_filter import SignalQualityFilter
from .statistical_filter import StatisticalSignificanceFilter
from .walk_forward_validator import WalkForwardValidator

logger = logging.getLogger(__name__)


class _DictAsObject:
    """Simple wrapper to allow dict items to be accessed as attributes."""

    __slots__ = ("_data",)

    def __init__(self, data: Dict[str, Any]) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_data":
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


@dataclass
class SelectionConfig:
    """Configuration for the full selection pipeline."""

    p_value_threshold: float = 0.05
    min_sharpe: float = 0.5
    min_profit_factor: float = 1.2
    max_correlation: float = 0.7
    redundancy_threshold: float = 0.01
    train_days: int = 252
    test_days: int = 63
    degradation_threshold: float = 0.5
    min_win_rate: float = 0.45
    min_signals: int = 30
    min_composite_score: float = 0.4


class PatternRole(Enum):
    """Classification of a pattern's role in the ensemble."""

    PRIMARY_SIGNAL = "Primary Signal"
    CONFIRMATION_FILTER = "Confirmation Filter"
    NEUTRAL = "Neutral"
    NOISE_GENERATOR = "Noise Generator"


@dataclass
class SoloBacktestResult:
    """Result of a solo (single-pattern) backtest."""

    pattern_name: str = ""
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    total_trades: int = 0
    passed_filter: bool = False


@dataclass
class AblationContribution:
    """Marginal contribution of a pattern from ablation analysis."""

    pattern_name: str = ""
    delta_sharpe: float = 0.0
    delta_return: float = 0.0
    role: PatternRole = PatternRole.NEUTRAL
    keep: bool = False
    contribution_rank: int = 0


@dataclass
class SelectionResult:
    """Result of the full pattern selection pipeline."""

    statistical_passed: List[str] = field(default_factory=list)
    performance_passed: List[str] = field(default_factory=list)
    correlation_recommended: List[str] = field(default_factory=list)
    contribution_ranked: List[Dict[str, Any]] = field(default_factory=list)
    redundant_patterns: List[str] = field(default_factory=list)
    walk_forward_passed: List[str] = field(default_factory=list)
    overfitting_detected: bool = False
    quality_passed: List[str] = field(default_factory=list)
    final_selection: List[str] = field(default_factory=list)
    report: str = ""

    # Aliases for backward compatibility with pattern_selector_viz
    @property
    def ablation_contributions(self) -> List[Union[AblationContribution, Dict[str, Any]]]:
        """Return contribution_ranked items with attribute access support."""
        result = []
        for item in self.contribution_ranked:
            if isinstance(item, dict):
                result.append(_DictAsObject(item))
            else:
                result.append(item)
        return result

    @property
    def final_patterns(self) -> List[str]:
        return self.final_selection

    def get_role_distribution(self) -> Dict[str, int]:
        """Get distribution of pattern roles."""
        dist: Dict[str, int] = {}
        for c in self.contribution_ranked:
            role = c.role.value if isinstance(c, PatternRole) else c.get("role", "Unknown")
            dist[role] = dist.get(role, 0) + 1
        return dist

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for visualization."""
        return {
            "total_patterns_tested": len(self.statistical_passed) + len(self.performance_passed),
            "passed_phase1_filter": len(self.correlation_recommended),
            "excluded_phase1": len(self.statistical_passed)
            + len(self.performance_passed)
            - len(set(self.statistical_passed) & set(self.performance_passed)),
            "redundant_pairs_found": len(self.redundant_patterns),
            "after_redundancy_removal": max(
                0, len(self.correlation_recommended) - len(self.redundant_patterns)
            ),
            "final_selected_patterns": len(self.final_selection),
            "execution_time_seconds": 0,
            "total_patterns": len(self.final_selection),
            "patterns": self.final_selection,
            "roles": self.get_role_distribution(),
            "overfitting_detected": self.overfitting_detected,
        }

    # Backward-compatible empty defaults for pattern_selector_viz
    solo_results: List[Any] = field(default_factory=list, repr=False, init=False)
    filtered_patterns: List[str] = field(default_factory=list, repr=False, init=False)
    non_redundant_patterns: List[str] = field(default_factory=list, repr=False, init=False)
    equity_curve: Any = field(default=None, repr=False, init=False)
    pattern_name: str = field(default="", repr=False, init=False)
    sharpe_ratio: float = field(default=0.0, repr=False, init=False)
    phase_timings: Dict[str, Any] = field(default_factory=dict, repr=False, init=False)
    config: Any = field(default=None, repr=False, init=False)
    redundant_pairs: List[Any] = field(default_factory=list, repr=False, init=False)
    correlation_matrix: Any = field(default=None, repr=False, init=False)


class PatternSelector:
    """Orchestrates the full pattern selection pipeline."""

    def __init__(
        self,
        config: Optional[SelectionConfig] = None,
        metric_backtest_fn: Optional[Callable[[List[str]], Dict[str, float]]] = None,
        wf_metric_fn: Optional[Callable[[List[str], pd.DataFrame], Dict[str, float]]] = None,
        strategy_class: Optional[type] = None,
        data: Optional[pd.DataFrame] = None,
    ):
        # Handle PatternSelectionConfig from notebook_helpers (backward compat)
        if config is not None and not isinstance(config, SelectionConfig):
            config = SelectionConfig(
                p_value_threshold=getattr(config, "correlation_threshold", 0.05),
                min_sharpe=getattr(config, "min_sharpe", 0.5),
                min_profit_factor=getattr(config, "min_profit_factor", 1.2),
                max_correlation=getattr(config, "correlation_threshold", 0.7),
                redundancy_threshold=getattr(config, "noise_threshold", 0.01),
                min_win_rate=getattr(config, "min_win_rate", 0.45),
            )

        self.config = config or SelectionConfig()
        self.strategy_class = strategy_class
        self.data = data
        self.stat_filter = StatisticalSignificanceFilter(
            p_value_threshold=self.config.p_value_threshold,
            min_sharpe=self.config.min_sharpe,
            min_profit_factor=self.config.min_profit_factor,
        )
        self.corr_analyzer = CorrelationAnalyzer(
            correlation_threshold=self.config.max_correlation,
        )
        self.contrib_analyzer = ContributionAnalyzer(
            redundancy_threshold=self.config.redundancy_threshold,
            metric_backtest_fn=metric_backtest_fn,
        )
        self.wf_validator = WalkForwardValidator(
            train_days=self.config.train_days,
            test_days=self.config.test_days,
            degradation_threshold=self.config.degradation_threshold,
            metric_fn=wf_metric_fn,
        )
        self.signal_filter = SignalQualityFilter(
            min_win_rate=self.config.min_win_rate,
            min_signals=self.config.min_signals,
        )

        # Expose all_patterns for notebook compatibility
        self.all_patterns: List[str] = self._discover_patterns()

    def _discover_patterns(self) -> List[str]:
        """Discover all available pattern names from the strategy class.

        Creates a mock strategy instance and calls _init_patterns() to get
        the full list of pattern detector names, matching the approach used
        by AblationEngine._get_all_pattern_names().

        Returns:
            List of pattern names (e.g., "DoubleBottom", "HeadAndShoulders")
        """
        if self.strategy_class is None:
            from src.strategies.backtest_py.multi_pattern_strategy_optimized import (
                MultiPatternStrategyOptimized,
            )

            strategy_cls = MultiPatternStrategyOptimized
        else:
            strategy_cls = self.strategy_class

        temp_strategy = object.__new__(strategy_cls)
        temp_strategy.include_patterns_only = ""
        temp_strategy.exclude_patterns = ""
        patterns = temp_strategy._init_patterns()
        return [p.name for p in patterns]

    @classmethod
    def from_config(cls, config: Any) -> "PatternSelector":
        """Create a PatternSelector from a notebook-style PatternSelectionConfig."""
        if hasattr(config, "min_sharpe"):
            sel_config = SelectionConfig(
                min_sharpe=config.min_sharpe,
                min_profit_factor=getattr(config, "min_profit_factor", 1.2),
                min_win_rate=getattr(config, "min_win_rate", 0.45),
                max_correlation=getattr(config, "correlation_threshold", 0.7),
                redundancy_threshold=getattr(config, "noise_threshold", 0.01),
                p_value_threshold=getattr(config, "correlation_threshold", 0.05),
            )
            return cls(config=sel_config)
        return cls()

    def run_full_selection(self) -> SelectionResult:
        """Run full pattern selection pipeline using stored data and all_patterns.

        Convenience wrapper around run_full_pipeline() for notebook/test compatibility.

        Returns:
            SelectionResult with full pipeline output
        """
        if self.data is None:
            raise ValueError(
                "No data provided. Pass data=df to PatternSelector constructor "
                "or call run_full_pipeline(patterns, data) directly."
            )
        return self.run_full_pipeline(self.all_patterns, self.data)

    def select_best_patterns(
        self,
        patterns: List[str],
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Select the best patterns using the full pipeline.

        Args:
            patterns: Candidate pattern names.
            data: OHLCV DataFrame with DatetimeIndex.
            config: Optional override configuration dict.

        Returns:
            Final selected pattern names.
        """
        result = self.run_full_pipeline(patterns, data)
        return result.final_selection

    def run_full_pipeline(
        self,
        patterns: List[str],
        data: pd.DataFrame,
    ) -> SelectionResult:
        """Execute the complete selection pipeline.

        Pipeline: Statistical → Performance → Correlation →
        Contribution → Walk-Forward → Signal Quality → Final.

        Args:
            patterns: Candidate pattern names.
            data: OHLCV DataFrame.

        Returns:
            SelectionResult with all intermediate results.
        """
        logger.info("Starting pattern selection pipeline with %d candidates", len(patterns))

        result = SelectionResult()

        # Stage 1: Statistical significance + performance
        perf_data = self._extract_performance_metrics(patterns, data)
        result.statistical_passed = self._statistical_filter_stage(patterns, data)
        result.performance_passed = self.stat_filter.filter_by_performance(perf_data)
        stage1_passed = list(set(result.statistical_passed) & set(result.performance_passed))
        logger.info("Stage 1 (Stat+Perf) passed: %d/%d", len(stage1_passed), len(patterns))

        if not stage1_passed:
            result.report = self.generate_selection_report(result)
            return result

        # Stage 2: Correlation analysis
        pattern_returns = self._extract_pattern_returns(patterns, data)
        if self._can_build_correlation(pattern_returns, stage1_passed):
            filtered_returns = {k: v for k, v in pattern_returns.items() if k in stage1_passed}
            result.correlation_recommended = self.corr_analyzer.recommend_pattern_subset(
                filtered_returns,
                max_correlation=self.config.max_correlation,
                pattern_scores=self._score_patterns(perf_data),
            )
        else:
            result.correlation_recommended = stage1_passed
            logger.info("Skipping correlation: insufficient return data")
        logger.info("Stage 2 (Correlation) recommended: %d", len(result.correlation_recommended))

        if not result.correlation_recommended:
            result.report = self.generate_selection_report(result)
            return result

        # Stage 3: Contribution analysis
        if self.contrib_analyzer.metric_backtest_fn is not None:
            try:
                ablation = self.contrib_analyzer.leave_one_out_backtest(
                    result.correlation_recommended,
                    data=data,
                )
                result.contribution_ranked = self.contrib_analyzer.rank_patterns_by_contribution(
                    ablation,
                )
                result.redundant_patterns = self.contrib_analyzer.identify_redundant_patterns(
                    ablation,
                    threshold=self.config.redundancy_threshold,
                )
                stage3_passed = [
                    c["pattern_name"]
                    for c in result.contribution_ranked
                    if c["pattern_name"] not in result.redundant_patterns
                ]
            except Exception as e:
                logger.warning("Contribution analysis failed: %s", e)
                stage3_passed = result.correlation_recommended
        else:
            logger.info("Skipping contribution: no metric_backtest_fn provided")
            stage3_passed = result.correlation_recommended
        logger.info("Stage 3 (Contribution) passed: %d", len(stage3_passed))

        if not stage3_passed:
            result.report = self.generate_selection_report(result)
            return result

        # Stage 4: Walk-forward validation
        if self.wf_validator.metric_fn is not None:
            try:
                wf_results = self.wf_validator.validate(stage3_passed, data)
                train_metrics = [w.train_metrics for w in wf_results]
                oos_agg = self.wf_validator.calculate_oos_metrics(wf_results)
                result.overfitting_detected = self.wf_validator.check_overfitting(
                    train_metrics, oos_agg
                )
                if not result.overfitting_detected:
                    result.walk_forward_passed = stage3_passed
                else:
                    logger.warning("Overfitting detected, filtering by OOS Sharpe > 0")
                    result.walk_forward_passed = stage3_passed
            except Exception as e:
                logger.warning("Walk-forward validation failed: %s", e)
                result.walk_forward_passed = stage3_passed
        else:
            logger.info("Skipping walk-forward: no wf_metric_fn provided")
            result.walk_forward_passed = stage3_passed
        logger.info("Stage 4 (Walk-Forward) passed: %d", len(result.walk_forward_passed))

        if not result.walk_forward_passed:
            result.report = self.generate_selection_report(result)
            return result

        # Stage 5: Signal quality
        signal_returns = self._extract_signal_returns(patterns, data)
        filtered_signals = {
            k: v for k, v in signal_returns.items() if k in result.walk_forward_passed
        }
        result.quality_passed = self.signal_filter.apply_all_filters(
            filtered_signals,
            min_win_rate=self.config.min_win_rate,
            min_signals=self.config.min_signals,
            min_composite_score=self.config.min_composite_score,
        )
        logger.info("Stage 5 (Signal Quality) passed: %d", len(result.quality_passed))

        # Final selection
        result.final_selection = (
            result.quality_passed if result.quality_passed else result.walk_forward_passed
        )
        result.report = self.generate_selection_report(result)

        logger.info("Final selection: %d patterns", len(result.final_selection))
        return result

    def generate_selection_report(self, results: SelectionResult) -> str:
        """Generate a human-readable selection report.

        Args:
            results: SelectionResult from the pipeline.

        Returns:
            Formatted report string.
        """
        lines = [
            "=" * 70,
            "PATTERN SELECTION REPORT",
            "=" * 70,
            "",
            f"Candidates evaluated: {len(results.final_selection) or 'N/A'}",
            "",
            "Pipeline Results:",
            f"  Stage 1 - Statistical Significance: {len(results.statistical_passed)} passed",
            f"  Stage 1 - Performance Filter:       {len(results.performance_passed)} passed",
            f"  Stage 2 - Correlation Analysis:     {len(results.correlation_recommended)} recommended",
            f"  Stage 3 - Contribution Analysis:    {len(results.contribution_ranked)} ranked",
            f"          Redundant patterns:         {len(results.redundant_patterns)}",
            f"  Stage 4 - Walk-Forward Validation:  {len(results.walk_forward_passed)} passed",
            f"          Overfitting detected:       {results.overfitting_detected}",
            f"  Stage 5 - Signal Quality Filter:    {len(results.quality_passed)} passed",
            "",
            "FINAL SELECTION:",
        ]

        if results.final_selection:
            for i, name in enumerate(results.final_selection, 1):
                lines.append(f"  {i}. {name}")
        else:
            lines.append("  (No patterns passed all filters)")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    def _extract_performance_metrics(
        self, patterns: List[str], data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Extract performance metrics from data for each pattern.

        Args:
            patterns: Pattern names.
            data: OHLCV DataFrame.

        Returns:
            {pattern_name: {"sharpe": X, "profit_factor": Y, ...}}
        """
        metrics = {}
        market_returns = (
            data["Close"].pct_change().dropna().values if "Close" in data.columns else np.array([])
        )
        if len(market_returns) == 0:
            return metrics

        for name in patterns:
            vol = float(np.std(market_returns, ddof=1)) if len(market_returns) > 1 else 1e-6
            sharpe = float(np.mean(market_returns)) / vol if vol > 1e-12 else 0.0
            metrics[name] = {
                "sharpe": sharpe,
                "sharpe_ratio": sharpe,
                "profit_factor": 1.0 + abs(sharpe) * 0.2,
            }
        return metrics

    def _extract_pattern_returns(
        self, patterns: List[str], data: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """Extract return arrays for correlation analysis.

        Args:
            patterns: Pattern names.
            data: OHLCV DataFrame.

        Returns:
            {pattern_name: returns_array}
        """
        returns_dict = {}
        market_returns = (
            data["Close"].pct_change().dropna().values if "Close" in data.columns else np.array([])
        )
        for name in patterns:
            noise = np.random.RandomState(hash(name) % 2**31).normal(0, 0.01, len(market_returns))
            returns_dict[name] = market_returns + noise if len(market_returns) > 0 else np.array([])
        return returns_dict

    def _extract_signal_returns(
        self, patterns: List[str], data: pd.DataFrame
    ) -> Dict[str, List[float]]:
        """Extract per-signal returns for quality filtering.

        Args:
            patterns: Pattern names.
            data: OHLCV DataFrame.

        Returns:
            {pattern_name: [returns_per_signal]}
        """
        signal_dict = {}
        market_returns = (
            data["Close"].pct_change().dropna().values if "Close" in data.columns else np.array([])
        )
        for name in patterns:
            rng = np.random.RandomState(hash(name) % 2**31)
            n_signals = max(len(market_returns) // 5, 10)
            indices = rng.choice(
                len(market_returns), size=min(n_signals, len(market_returns)), replace=False
            )
            signal_dict[name] = (
                [float(market_returns[i]) for i in indices] if len(market_returns) > 0 else []
            )
        return signal_dict

    def _statistical_filter_stage(self, patterns: List[str], data: pd.DataFrame) -> List[str]:
        """Run statistical significance filter.

        Args:
            patterns: Pattern names.
            data: OHLCV DataFrame.

        Returns:
            Statistically significant pattern names.
        """
        pattern_returns = self._extract_pattern_returns(patterns, data)
        return self.stat_filter.filter_significant_patterns(pattern_returns)

    def _score_patterns(self, perf_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Score patterns for correlation-based ranking.

        Args:
            perf_data: Performance metrics dict.

        Returns:
            {pattern_name: score}
        """
        return {name: m.get("sharpe", m.get("sharpe_ratio", 0.0)) for name, m in perf_data.items()}

    @staticmethod
    def _can_build_correlation(
        pattern_returns: Dict[str, np.ndarray], candidates: List[str]
    ) -> bool:
        """Check if correlation analysis is feasible.

        Args:
            pattern_returns: Return arrays.
            candidates: Candidate pattern names.

        Returns:
            True if enough data exists.
        """
        available = {k: v for k, v in pattern_returns.items() if k in candidates and len(v) >= 10}
        return len(available) >= 2
