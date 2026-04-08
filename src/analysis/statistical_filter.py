"""
Statistical Significance Filter for Pattern Selection

Provides statistical tests to ensure pattern performance results are 
statistically significant and not due to random chance.

Key Tests:
- Wilson Score Confidence Interval for Win Rate
- Sharpe Ratio Standard Error
- Minimum Sample Size Calculation
- Statistical Significance Testing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class StatisticalFilterConfig:
    """Configuration for statistical significance filters."""
    
    # Minimum sample size requirements
    min_trades: int = 30
    min_sample_period_days: int = 365
    
    # Confidence level for tests
    confidence_level: float = 0.95
    
    # Maximum acceptable confidence interval width
    max_win_rate_ci_width: float = 0.20  # 20 percentage points
    
    # Minimum acceptable Sharpe ratio standard error
    max_sharpe_se: float = 0.5


@dataclass
class StatisticalFilterResult:
    """Result of statistical significance testing for a pattern."""
    
    pattern_name: str
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    
    # Wilson Score Confidence Interval for Win Rate
    win_rate_ci_lower: float = 0.0
    win_rate_ci_upper: float = 0.0
    win_rate_ci_width: float = 0.0
    
    # Sharpe Ratio Standard Error
    sharpe_se: float = 0.0
    
    # Statistical significance
    is_significant: bool = False
    failure_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pattern_name": self.pattern_name,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate_ci_lower": self.win_rate_ci_lower,
            "win_rate_ci_upper": self.win_rate_ci_upper,
            "win_rate_ci_width": self.win_rate_ci_width,
            "sharpe_se": self.sharpe_se,
            "is_significant": self.is_significant,
            "failure_reasons": self.failure_reasons,
        }


class StatisticalFilter:
    """
    Statistical significance filter for pattern selection.
    
    Tests whether pattern performance is statistically significant
    using Wilson Score Confidence Intervals and Sharpe Ratio Standard Errors.
    
    Usage:
        config = StatisticalFilterConfig(min_trades=30)
        filter = StatisticalFilter(config)
        result = filter.test_pattern(
            pattern_name="Double Bottom",
            total_trades=45,
            win_rate=0.52,
            sharpe_ratio=0.85
        )
    """
    
    def __init__(self, config: Optional[StatisticalFilterConfig] = None):
        """
        Initialize statistical filter.
        
        Args:
            config: Filter configuration (uses defaults if None)
        """
        self.config = config or StatisticalFilterConfig()
    
    def wilson_score_interval(
        self, successes: int, trials: int, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate Wilson Score Confidence Interval for a proportion.
        
        The Wilson score interval is more accurate than the normal approximation
        for proportions, especially with small sample sizes.
        
        Args:
            successes: Number of successful trials (wins)
            trials: Total number of trials (trades)
            confidence: Confidence level (default 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if trials == 0:
            return (0.0, 0.0)
        
        p_hat = successes / trials
        
        # Z-score for confidence level
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        
        # Wilson score interval formula
        denominator = 1 + z**2 / trials
        center = (p_hat + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials) / denominator
        
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        
        return (float(lower), float(upper))
    
    def sharpe_standard_error(self, sharpe: float, n: int) -> float:
        """
        Calculate standard error of Sharpe ratio.
        
        Formula: SE = sqrt((1 + 0.5 * Sharpe^2) / n)
        
        Args:
            sharpe: Sharpe ratio estimate
            n: Number of observations
            
        Returns:
            Standard error of Sharpe ratio
        """
        if n <= 1:
            return float('inf')
        
        se = np.sqrt((1 + 0.5 * sharpe**2) / n)
        return float(se)  # type: ignore[arg-type]
    
    def minimum_sample_size(
        self, margin_of_error: float = 0.10, p_estimate: float = 0.50, confidence: float = 0.95
    ) -> int:
        """
        Calculate minimum sample size for desired margin of error.
        
        Args:
            margin_of_error: Desired maximum margin of error (e.g., 0.10 = 10%)
            p_estimate: Estimated proportion (use 0.5 for conservative estimate)
            confidence: Confidence level
            
        Returns:
            Minimum sample size required
        """
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        n = (z / margin_of_error)**2 * p_estimate * (1 - p_estimate)
        return int(np.ceil(n))
    
    def test_pattern(
        self,
        pattern_name: str,
        total_trades: int,
        win_rate: float,
        sharpe_ratio: float,
        confidence: Optional[float] = None,
    ) -> StatisticalFilterResult:
        """
        Test whether a pattern's performance is statistically significant.
        
        Args:
            pattern_name: Name of the pattern
            total_trades: Number of trades
            win_rate: Win rate (0.0 to 1.0)
            sharpe_ratio: Sharpe ratio
            confidence: Override confidence level (uses config default if None)
            
        Returns:
            StatisticalFilterResult with test results
        """
        conf = confidence or self.config.confidence_level
        failure_reasons = []
        
        # Test 1: Minimum sample size
        if total_trades < self.config.min_trades:
            failure_reasons.append(
                f"Insufficient trades ({total_trades} < {self.config.min_trades})"
            )
        
        # Test 2: Wilson Score Confidence Interval for Win Rate
        successes = int(total_trades * win_rate)
        ci_lower, ci_upper = self.wilson_score_interval(successes, total_trades, conf)
        ci_width = ci_upper - ci_lower
        
        if total_trades >= self.config.min_trades:
            if ci_width > self.config.max_win_rate_ci_width:
                failure_reasons.append(
                    f"Win rate CI too wide ({ci_width:.1%} > {self.config.max_win_rate_ci_width:.0%})"
                )
        
        # Test 3: Sharpe Ratio Standard Error
        sharpe_se = self.sharpe_standard_error(sharpe_ratio, total_trades)
        
        if total_trades >= self.config.min_trades:
            if sharpe_se > self.config.max_sharpe_se:
                failure_reasons.append(
                    f"Sharpe SE too high ({sharpe_se:.3f} > {self.config.max_sharpe_se})"
                )
        
        # Determine overall significance
        is_significant = len(failure_reasons) == 0
        
        return StatisticalFilterResult(
            pattern_name=pattern_name,
            total_trades=total_trades,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            win_rate_ci_lower=ci_lower,
            win_rate_ci_upper=ci_upper,
            win_rate_ci_width=ci_width,
            sharpe_se=sharpe_se,
            is_significant=is_significant,
            failure_reasons=failure_reasons,
        )
    
    def test_pattern_from_result(self, result: dict) -> StatisticalFilterResult:
        """
        Test pattern performance from a dictionary result.
        
        Args:
            result: Dictionary with pattern performance metrics
            
        Returns:
            StatisticalFilterResult
        """
        return self.test_pattern(
            pattern_name=result.get("pattern_name", "Unknown"),
            total_trades=result.get("total_trades", 0),
            win_rate=result.get("win_rate", 0.0),
            sharpe_ratio=result.get("sharpe_ratio", 0.0),
        )


def calculate_statistical_power(
    win_rate: float, n_trades: int, alpha: float = 0.05
) -> float:
    """
    Calculate statistical power of a win rate test.
    
    Statistical power is the probability of correctly rejecting
    the null hypothesis when it is false.
    
    Args:
        win_rate: Observed win rate
        n_trades: Number of trades
        alpha: Significance level
        
    Returns:
        Statistical power (0.0 to 1.0)
    """
    if n_trades == 0:
        return 0.0
    
    # Null hypothesis: win_rate = 0.5 (random)
    p0 = 0.5
    
    # Effect size (Cohen's h)
    h = 2 * np.arcsin(np.sqrt(win_rate)) - 2 * np.arcsin(np.sqrt(p0))
    
    # Non-centrality parameter
    ncp = h * np.sqrt(n_trades)
    
    # Critical value
    z_crit = stats.norm.ppf(1 - alpha / 2)
    
    # Power calculation
    power = stats.norm.cdf(-z_crit + ncp) + stats.norm.cdf(-z_crit - ncp)
    
    return float(power)


def get_minimum_trades_for_power(
    win_rate: float, power: float = 0.80, alpha: float = 0.05
) -> int:
    """
    Calculate minimum trades needed for desired statistical power.
    
    Args:
        win_rate: Expected win rate
        power: Desired statistical power (default 0.80)
        alpha: Significance level
        
    Returns:
        Minimum number of trades required
    """
    # Null hypothesis
    p0 = 0.5
    
    # Effect size
    h = 2 * np.arcsin(np.sqrt(win_rate)) - 2 * np.arcsin(np.sqrt(p0))
    
    if abs(h) < 1e-10:
        return int(1e9)  # Return large number instead of inf for int return type
    
    # Z-scores
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    
    # Sample size formula
    n = ((z_alpha + z_beta) / h)**2
    
    return int(np.ceil(n))
