"""
Performance Filter for Pattern Selection

Provides comprehensive performance threshold filtering for patterns.
Tests patterns against multiple performance criteria to ensure only
profitable patterns are included in the portfolio.

Key Metrics:
- Sharpe Ratio (risk-adjusted return)
- Profit Factor (gross wins / gross losses)
- Win Rate (percentage of winning trades)
- Expectancy (average profit per trade in R-multiples)
- Maximum Drawdown (largest peak-to-trough decline)
- Annualized Return
- Calmar Ratio (return / max drawdown)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PerformanceFilterConfig:
    """Configuration for performance filter thresholds."""
    
    # Risk-adjusted return thresholds
    min_sharpe_ratio: float = 0.5
    min_sortino_ratio: float = 0.5
    min_calmar_ratio: float = 0.5
    
    # Profitability thresholds
    min_profit_factor: float = 1.2
    min_win_rate: float = 0.35
    min_expectancy_r: float = 0.1  # In R-multiples
    
    # Risk thresholds
    max_drawdown_pct: float = 25.0
    max_ulcer_index: float = 10.0
    
    # Return thresholds
    min_annualized_return: float = 0.05  # 5% annual
    min_total_return: float = 0.0  # No minimum total return


@dataclass
class PerformanceFilterResult:
    """Result of performance filtering for a single pattern."""
    
    pattern_name: str
    
    # Core metrics
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    total_return_pct: float
    max_drawdown_pct: float
    
    # Extended metrics
    expectancy_r: float = 0.0
    annualized_return: float = 0.0
    calmar_ratio: float = 0.0
    ulcer_index: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Filter result
    passed: bool = False
    failure_reasons: List[str] = field(default_factory=list)
    metric_values: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_name": self.pattern_name,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "profit_factor": self.profit_factor,
            "total_return_pct": self.total_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "expectancy_r": self.expectancy_r,
            "annualized_return": self.annualized_return,
            "calmar_ratio": self.calmar_ratio,
            "ulcer_index": self.ulcer_index,
            "passed": self.passed,
            "failure_reasons": self.failure_reasons,
        }


class PerformanceFilter:
    """
    Performance filter for pattern selection.
    
    Tests patterns against comprehensive performance thresholds
    to ensure only profitable patterns are included.
    
    Usage:
        config = PerformanceFilterConfig(
            min_sharpe_ratio=0.5,
            min_profit_factor=1.2,
            min_win_rate=0.35,
            max_drawdown_pct=25.0,
        )
        filter = PerformanceFilter(config)
        result = filter.test_pattern(
            pattern_name="Double Bottom",
            total_trades=45,
            win_rate=0.52,
            sharpe_ratio=0.85,
            profit_factor=1.45,
            total_return_pct=15.5,
            max_drawdown_pct=12.5,
        )
    """
    
    def __init__(self, config: Optional[PerformanceFilterConfig] = None):
        """
        Initialize performance filter.
        
        Args:
            config: Filter configuration (uses defaults if None)
        """
        self.config = config or PerformanceFilterConfig()
    
    def calculate_expectancy(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate expectancy in R-multiples.
        
        Expectancy = (Win_Rate * Avg_Win) - (Loss_Rate * Avg_Loss)
        
        Args:
            win_rate: Win rate (0.0 to 1.0)
            avg_win: Average win amount
            avg_loss: Average loss amount (positive)
            
        Returns:
            Expectancy in same units as avg_win/avg_loss
        """
        loss_rate = 1 - win_rate
        return (win_rate * avg_win) - (loss_rate * avg_loss)
    
    def calculate_expectancy_r(
        self,
        win_rate: float,
        avg_win_r: float,
        avg_loss_r: float,
    ) -> float:
        """
        Calculate expectancy in R-multiples.
        
        Args:
            win_rate: Win rate (0.0 to 1.0)
            avg_win_r: Average win in R-multiples
            avg_loss_r: Average loss in R-multiples (positive)
            
        Returns:
            Expectancy in R-multiples
        """
        return self.calculate_expectancy(win_rate, avg_win_r, avg_loss_r)
    
    def calculate_calmar_ratio(
        self,
        annualized_return: float,
        max_drawdown_pct: float,
    ) -> float:
        """
        Calculate Calmar Ratio (return / max drawdown).
        
        Args:
            annualized_return: Annualized return (decimal)
            max_drawdown_pct: Maximum drawdown (percentage)
            
        Returns:
            Calmar Ratio
        """
        if max_drawdown_pct == 0:
            return float('inf') if annualized_return > 0 else 0.0
        return annualized_return / (max_drawdown_pct / 100)
    
    def calculate_ulcer_index(self, drawdown_series: np.ndarray) -> float:
        """
        Calculate Ulcer Index (measures drawdown severity).
        
        Ulcer Index = sqrt(mean(drawdown^2))
        
        Args:
            drawdown_series: Array of drawdown percentages
            
        Returns:
            Ulcer Index value
        """
        if len(drawdown_series) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.array(drawdown_series)**2)))
    
    def test_pattern(
        self,
        pattern_name: str,
        total_trades: int,
        win_rate: float,
        sharpe_ratio: float,
        profit_factor: float,
        total_return_pct: float,
        max_drawdown_pct: float,
        sortino_ratio: float = 0.0,
        expectancy_r: float = 0.0,
        annualized_return: float = 0.0,
        avg_win: float = 0.0,
        avg_loss: float = 0.0,
        largest_win: float = 0.0,
        largest_loss: float = 0.0,
        calmar_ratio: float = 0.0,
        ulcer_index: float = 0.0,
    ) -> PerformanceFilterResult:
        """
        Test a pattern against performance thresholds.
        
        Args:
            pattern_name: Name of the pattern
            total_trades: Number of trades
            win_rate: Win rate (0.0 to 1.0)
            sharpe_ratio: Sharpe ratio
            profit_factor: Profit factor
            total_return_pct: Total return percentage
            max_drawdown_pct: Maximum drawdown percentage
            sortino_ratio: Sortino ratio (optional)
            expectancy_r: Expectancy in R-multiples (optional)
            annualized_return: Annualized return (optional)
            avg_win: Average win amount (optional)
            avg_loss: Average loss amount (optional)
            largest_win: Largest win (optional)
            largest_loss: Largest loss (optional)
            calmar_ratio: Calmar ratio (optional)
            ulcer_index: Ulcer index (optional)
            
        Returns:
            PerformanceFilterResult with test results
        """
        failure_reasons = []
        
        # Calculate derived metrics if not provided
        if calmar_ratio == 0 and annualized_return > 0 and max_drawdown_pct > 0:
            calmar_ratio = self.calculate_calmar_ratio(annualized_return, max_drawdown_pct)
        
        if expectancy_r == 0 and avg_win > 0 and avg_loss > 0:
            expectancy_r = self.calculate_expectancy_r(win_rate, avg_win, avg_loss)
        
        # Test 1: Sharpe Ratio
        if sharpe_ratio < self.config.min_sharpe_ratio:
            failure_reasons.append(
                f"Low Sharpe ({sharpe_ratio:.3f} < {self.config.min_sharpe_ratio})"
            )
        
        # Test 2: Sortino Ratio
        if sortino_ratio > 0 and sortino_ratio < self.config.min_sortino_ratio:
            failure_reasons.append(
                f"Low Sortino ({sortino_ratio:.3f} < {self.config.min_sortino_ratio})"
            )
        
        # Test 3: Profit Factor
        if profit_factor < self.config.min_profit_factor:
            failure_reasons.append(
                f"Low Profit Factor ({profit_factor:.3f} < {self.config.min_profit_factor})"
            )
        
        # Test 4: Win Rate
        if win_rate < self.config.min_win_rate:
            failure_reasons.append(
                f"Low Win Rate ({win_rate:.1%} < {self.config.min_win_rate:.0%})"
            )
        
        # Test 5: Expectancy
        if expectancy_r != 0 and expectancy_r < self.config.min_expectancy_r:
            failure_reasons.append(
                f"Low Expectancy ({expectancy_r:.3f}R < {self.config.min_expectancy_r}R)"
            )
        
        # Test 6: Maximum Drawdown
        if max_drawdown_pct > self.config.max_drawdown_pct:
            failure_reasons.append(
                f"High Drawdown ({max_drawdown_pct:.1f}% > {self.config.max_drawdown_pct:.0f}%)"
            )
        
        # Test 7: Annualized Return
        if annualized_return > 0 and annualized_return < self.config.min_annualized_return:
            failure_reasons.append(
                f"Low Annualized Return ({annualized_return:.1%} < {self.config.min_annualized_return:.0%})"
            )
        
        # Test 8: Ulcer Index
        if ulcer_index > 0 and ulcer_index > self.config.max_ulcer_index:
            failure_reasons.append(
                f"High Ulcer Index ({ulcer_index:.2f} > {self.config.max_ulcer_index:.0f})"
            )
        
        # Test 9: Calmar Ratio
        if calmar_ratio > 0 and calmar_ratio < self.config.min_calmar_ratio:
            failure_reasons.append(
                f"Low Calmar ({calmar_ratio:.3f} < {self.config.min_calmar_ratio})"
            )
        
        passed = len(failure_reasons) == 0
        
        return PerformanceFilterResult(
            pattern_name=pattern_name,
            total_trades=total_trades,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            profit_factor=profit_factor,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            expectancy_r=expectancy_r,
            annualized_return=annualized_return,
            calmar_ratio=calmar_ratio,
            ulcer_index=ulcer_index,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            passed=passed,
            failure_reasons=failure_reasons,
            metric_values={
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "profit_factor": profit_factor,
                "win_rate": win_rate,
                "expectancy_r": expectancy_r,
                "max_drawdown_pct": max_drawdown_pct,
                "annualized_return": annualized_return,
                "calmar_ratio": calmar_ratio,
                "ulcer_index": ulcer_index,
            },
        )
    
    def test_pattern_from_result(self, result: dict) -> PerformanceFilterResult:
        """
        Test pattern performance from a dictionary result.
        
        Args:
            result: Dictionary with pattern performance metrics
            
        Returns:
            PerformanceFilterResult
        """
        return self.test_pattern(
            pattern_name=result.get("pattern_name", "Unknown"),
            total_trades=result.get("total_trades", 0),
            win_rate=result.get("win_rate", 0.0),
            sharpe_ratio=result.get("sharpe_ratio", 0.0),
            sortino_ratio=result.get("sortino_ratio", 0.0),
            profit_factor=result.get("profit_factor", 0.0),
            total_return_pct=result.get("total_return_pct", 0.0),
            max_drawdown_pct=result.get("max_drawdown_pct", 0.0),
            expectancy_r=result.get("expectancy_r", 0.0),
            annualized_return=result.get("annualized_return", 0.0),
            avg_win=result.get("avg_win", 0.0),
            avg_loss=result.get("avg_loss", 0.0),
            largest_win=result.get("largest_win", 0.0),
            largest_loss=result.get("largest_loss", 0.0),
        )
    
    def evaluate_patterns(
        self, results: List[dict]
    ) -> List[PerformanceFilterResult]:
        """
        Evaluate multiple patterns against performance thresholds.
        
        Args:
            results: List of dictionaries with pattern performance metrics
            
        Returns:
            List of PerformanceFilterResult for each pattern
        """
        return [self.test_pattern_from_result(r) for r in results]
    
    def get_passed_patterns(self, results: List[PerformanceFilterResult]) -> List[str]:
        """
        Get list of patterns that passed performance filtering.
        
        Args:
            results: List of PerformanceFilterResult
            
        Returns:
            List of pattern names that passed
        """
        return [r.pattern_name for r in results if r.passed]
    
    def get_failed_patterns(self, results: List[PerformanceFilterResult]) -> List[str]:
        """
        Get list of patterns that failed performance filtering.
        
        Args:
            results: List of PerformanceFilterResult
            
        Returns:
            List of pattern names that failed
        """
        return [r.pattern_name for r in results if not r.passed]
    
    def generate_report(self, results: List[PerformanceFilterResult]) -> str:
        """
        Generate a human-readable performance filter report.
        
        Args:
            results: List of PerformanceFilterResult
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("PERFORMANCE FILTER REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Summary
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        
        lines.append(f"Total Patterns Evaluated: {len(results)}")
        lines.append(f"Passed: {len(passed)}")
        lines.append(f"Failed: {len(failed)}")
        lines.append("")
        
        # Passed patterns
        if passed:
            lines.append("-" * 80)
            lines.append("PASSED PATTERNS")
            lines.append("-" * 80)
            for r in passed:
                lines.append(
                    f"  {r.pattern_name}: "
                    f"Sharpe={r.sharpe_ratio:.3f}, "
                    f"PF={r.profit_factor:.2f}, "
                    f"WR={r.win_rate:.1%}, "
                    f"Return={r.total_return_pct:.2f}%"
                )
            lines.append("")
        
        # Failed patterns
        if failed:
            lines.append("-" * 80)
            lines.append("FAILED PATTERNS")
            lines.append("-" * 80)
            for r in failed:
                lines.append(f"  {r.pattern_name}:")
                for reason in r.failure_reasons:
                    lines.append(f"    - {reason}")
            lines.append("")
        
        return "\n".join(lines)
