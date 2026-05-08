# -*- coding: utf-8 -*-
"""
Diversity Score for Portfolio Sizing (R7)

Implements Jorion's Binomial Expansion Technique (BET) for measuring
effective portfolio diversification. Based on credit risk methodology
adapted for trading portfolios.

Key Insight:
- Count positions != diversification
- N correlated positions ≈ D uncorrelated equivalents
- Formula: D = N / (1 + (N-1)*ρ_avg) where ρ = average correlation

Research Reference:
- Jorion, Value-at-Risk (Third Edition) - Chapter 11
- SSRN-1018281: "Diversification in Trading Portfolios"

Integration Points:
- Connects to position_sizing.py for diversity-adjusted sizing
- Used in signal_aggregator.py for confluence scoring
- Uses correlation matrix from position_probability.py
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class DiversityMethod(Enum):
    """Methods for calculating diversity score."""

    SIMPLIFIED_BET = "simplified_bet"
    FULL_BET = "full_bet"
    EFFECTIVE_N = "effective_n"
    CORRELATION_WEIGHTED = "correlation_weighted"


@dataclass
class DiversityConfig:
    """
    Configuration for diversity score calculation.

    Attributes:
        method: Diversity calculation method
        confidence_level: Confidence level for tail risk (0.95 to 0.999)
        correlation_window: Rolling window for correlation estimation
        min_observations: Minimum observations for valid correlation
        use_rank_correlation: Use Spearman rank correlation instead of Pearson
    """

    method: DiversityMethod = DiversityMethod.SIMPLIFIED_BET
    confidence_level: float = 0.99
    correlation_window: int = 60
    min_observations: int = 20
    use_rank_correlation: bool = False


@dataclass
class DiversityResult:
    """
    Result of diversity score calculation.

    Attributes:
        diversity_score: Effective number of independent bets (D)
        position_count: Actual number of positions (N)
        average_correlation: Average pairwise correlation (ρ_avg)
        correlation_matrix: Full correlation matrix
        diversification_ratio: D / N ratio (0.0 to 1.0)
        concentration_risk: 1 - diversification_ratio
        metadata: Additional statistics
    """

    diversity_score: float
    position_count: int
    average_correlation: float
    correlation_matrix: np.ndarray
    diversification_ratio: float
    concentration_risk: float
    metadata: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "diversity_score": self.diversity_score,
            "position_count": self.position_count,
            "average_correlation": self.average_correlation,
            "diversification_ratio": self.diversification_ratio,
            "concentration_risk": self.concentration_risk,
            **self.metadata,
        }


class DiversityScorer:
    """
    Calculate portfolio diversity score using Jorion's BET approach.

    The key insight is that N correlated positions are equivalent to
    D uncorrelated positions, where D ≤ N.

    Formula (Simplified BET):
        D = N / (1 + (N - 1) * ρ_avg)

    Where:
        D = diversity score (effective independent bets)
        N = number of positions
        ρ_avg = average pairwise correlation

    When all positions are uncorrelated (ρ = 0): D = N
    When all positions perfectly correlated (ρ = 1): D = 1

    Usage:
        scorer = DiversityScorer()
        result = scorer.calculate(returns_df)
        print(f"Effective bets: {result.diversity_score:.2f} of {result.position_count}")
    """

    def __init__(self, config: Optional[DiversityConfig] = None):
        """
        Initialize diversity scorer.

        Args:
            config: Configuration for diversity calculation
        """
        self.config = config or DiversityConfig()

    def calculate_correlation_matrix(
        self,
        returns: pd.DataFrame,
        window: Optional[int] = None,
    ) -> np.ndarray:
        """
        Calculate correlation matrix from returns.

        Args:
            returns: DataFrame with columns as positions, rows as time periods
            window: Optional rolling window (uses config default if None)

        Returns:
            Correlation matrix (K x K where K = number of positions)
        """
        if window is None:
            window = self.config.correlation_window

        # Drop columns with insufficient data
        min_obs = self.config.min_observations
        valid_cols = returns.columns[returns.count() >= min_obs]
        valid_returns = returns[valid_cols]

        if len(valid_cols) < 2:
            # Single position or insufficient data
            return np.array([[1.0]])

        # Choose correlation method
        if self.config.use_rank_correlation:
            # Spearman rank correlation (more robust to outliers)
            corr_matrix = valid_returns.corr(method="spearman").values
        else:
            # Pearson correlation (standard)
            corr_matrix = valid_returns.corr(method="pearson").values

        # Handle NaN values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Ensure diagonal is 1.0
        np.fill_diagonal(corr_matrix, 1.0)

        # Symmetrize (in case of numerical issues)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2.0

        return corr_matrix

    def calculate_average_correlation(
        self,
        correlation_matrix: np.ndarray,
    ) -> float:
        """
        Calculate average pairwise correlation (excluding diagonal).

        Args:
            correlation_matrix: K x K correlation matrix

        Returns:
            Average off-diagonal correlation
        """
        n = correlation_matrix.shape[0]

        if n < 2:
            return 0.0

        # Sum all elements, subtract diagonal, divide by off-diagonal count
        total_sum = np.sum(correlation_matrix)
        diagonal_sum = np.trace(correlation_matrix)
        off_diagonal_sum = total_sum - diagonal_sum

        # Number of off-diagonal elements
        off_diagonal_count = n * n - n

        if off_diagonal_count == 0:
            return 0.0

        avg_corr = off_diagonal_sum / off_diagonal_count

        # Clamp to valid range
        return np.clip(avg_corr, -1.0, 1.0)

    def calculate_diversity_score(
        self,
        correlation_matrix: np.ndarray,
        n_positions: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Calculate diversity score using BET formula.

        Args:
            correlation_matrix: K x K correlation matrix
            n_positions: Number of positions (uses matrix size if None)

        Returns:
            Tuple of (diversity_score, average_correlation)
        """
        n = n_positions or correlation_matrix.shape[0]

        if n < 2:
            return (float(n) if n > 0 else 0.0, 0.0)

        # Calculate average correlation
        avg_corr = self.calculate_average_correlation(correlation_matrix)

        # Apply BET formula
        # D = N / (1 + (N - 1) * ρ_avg)
        denominator = 1.0 + (n - 1) * avg_corr

        # Avoid division by zero or negative denominator
        if denominator <= 0:
            # Extreme case: negative average correlation
            diversity = float(n)
        else:
            diversity = n / denominator

        # Clamp: diversity cannot exceed position count or go below 1
        diversity = np.clip(diversity, 1.0, float(n))

        return (diversity, avg_corr)

    def calculate(
        self,
        returns: pd.DataFrame,
        position_weights: Optional[np.ndarray] = None,
    ) -> DiversityResult:
        """
        Calculate full diversity score with all metrics.

        Args:
            returns: DataFrame of position returns (T x K)
            position_weights: Optional weights for each position

        Returns:
            DiversityResult with all metrics
        """
        n_positions = returns.shape[1]

        if n_positions == 0:
            return DiversityResult(
                diversity_score=0.0,
                position_count=0,
                average_correlation=0.0,
                correlation_matrix=np.array([[]]),
                diversification_ratio=0.0,
                concentration_risk=0.0,
                metadata={"error": "No positions"},
            )

        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(returns)

        # Calculate diversity score
        diversity, avg_corr = self.calculate_diversity_score(corr_matrix, n_positions)

        # Calculate diversification ratio
        diversification_ratio = diversity / n_positions if n_positions > 0 else 0.0

        # Concentration risk (inverse of diversification)
        concentration_risk = 1.0 - diversification_ratio

        # Additional statistics
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        largest_eigenvalue = np.max(eigenvalues)
        variance_explained = (
            largest_eigenvalue / np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else 1.0
        )

        metadata = {
            "largest_eigenvalue": float(largest_eigenvalue),
            "variance_explained": float(variance_explained),
            "condition_number": float(np.max(eigenvalues) / np.max([np.min(eigenvalues), 1e-10])),
            "effective_n": float(1.0 / np.sum(eigenvalues**2))
            if np.sum(eigenvalues**2) > 0
            else float(n_positions),
        }

        return DiversityResult(
            diversity_score=diversity,
            position_count=n_positions,
            average_correlation=avg_corr,
            correlation_matrix=corr_matrix,
            diversification_ratio=diversification_ratio,
            concentration_risk=concentration_risk,
            metadata=metadata,
        )

    def calculate_rolling_diversity(
        self,
        returns: pd.DataFrame,
        window: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Calculate rolling diversity score over time.

        Args:
            returns: DataFrame of position returns (T x K)
            window: Rolling window size

        Returns:
            DataFrame with rolling diversity metrics (T x 4)
        """
        if window is None:
            window = self.config.correlation_window

        results = []
        dates = []

        for i in range(window, len(returns) + 1):
            window_returns = returns.iloc[i - window : i]

            try:
                result = self.calculate(window_returns)
                results.append(
                    [
                        result.diversity_score,
                        result.position_count,
                        result.average_correlation,
                        result.diversification_ratio,
                        result.concentration_risk,
                    ]
                )
                dates.append(
                    returns.index[i - 1] if hasattr(returns.index, "__getitem__") else i - 1
                )
            except Exception:
                results.append([np.nan] * 5)
                dates.append(
                    returns.index[i - 1] if hasattr(returns.index, "__getitem__") else i - 1
                )

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(
            results,
            index=dates,
            columns=[
                "diversity_score",
                "position_count",
                "average_correlation",
                "diversification_ratio",
                "concentration_risk",
            ],
        )


class DiversityAdjustedSizer:
    """
    Apply diversity score to position sizing decisions.

    Integrates diversity score with position sizing to adjust
    position sizes based on portfolio concentration risk.

    Key formula:
        adjusted_size = base_size * (1 - concentration_penalty)

    Where:
        concentration_penalty = concentration_risk * penalty_factor
    """

    def __init__(
        self,
        diversity_scorer: Optional[DiversityScorer] = None,
        concentration_penalty_factor: float = 0.5,
        min_diversity_threshold: float = 2.0,
    ):
        """
        Initialize diversity-adjusted sizer.

        Args:
            diversity_scorer: Diversity scorer instance
            concentration_penalty_factor: How much concentration penalizes size (0.0-1.0)
            min_diversity_threshold: Minimum acceptable diversity score
        """
        self.diversity_scorer = diversity_scorer or DiversityScorer()
        self.concentration_penalty_factor = concentration_penalty_factor
        self.min_diversity_threshold = min_diversity_threshold

    def adjust_position_size(
        self,
        base_size: float,
        returns: pd.DataFrame,
        current_returns: Optional[pd.DataFrame] = None,
    ) -> Tuple[float, DiversityResult]:
        """
        Adjust position size based on diversity score.

        Args:
            base_size: Base position size (before diversity adjustment)
            returns: Historical returns DataFrame for correlation estimation
            current_returns: Optional current period returns (uses historical if None)

        Returns:
            Tuple of (adjusted_size, diversity_result)
        """
        # Calculate current diversity
        diversity_result = self.diversity_scorer.calculate(current_returns or returns)

        # Check minimum diversity threshold
        if diversity_result.diversity_score < self.min_diversity_threshold:
            # Below minimum diversity - reduce size significantly
            adjustment_factor = diversity_result.diversity_score / self.min_diversity_threshold
            adjusted_size = base_size * adjustment_factor
        else:
            # Apply concentration penalty
            penalty = diversity_result.concentration_risk * self.concentration_penalty_factor
            adjustment_factor = max(1.0 - penalty, 0.5)  # Max 50% reduction
            adjusted_size = base_size * adjustment_factor

        return (adjusted_size, diversity_result)

    def get_portfolio_constraints(
        self,
        returns: pd.DataFrame,
        max_positions: int = 10,
    ) -> Dict[str, float]:
        """
        Get portfolio constraints based on diversity analysis.

        Args:
            returns: Historical returns for correlation analysis
            max_positions: Maximum number of simultaneous positions

        Returns:
            Dictionary with recommended constraints
        """
        diversity_result = self.diversity_scorer.calculate(returns)

        # Calculate optimal number of positions
        # More positions only add value if they increase diversity
        optimal_n = min(
            max_positions,
            int(diversity_result.diversity_score * 1.5),  # Allow some overlap
        )

        # Per-position risk limit based on concentration
        base_risk = 0.02  # 2% base risk per trade
        concentration_adjustment = 1.0 + diversity_result.concentration_risk

        # Higher concentration = lower per-position risk
        risk_per_position = base_risk / concentration_adjustment

        return {
            "max_positions": optimal_n,
            "risk_per_position": risk_per_position,
            "diversity_score": diversity_result.diversity_score,
            "concentration_risk": diversity_result.concentration_risk,
            "recommended_action": "reduce_concentration"
            if diversity_result.concentration_risk > 0.5
            else "maintain",
        }


def calculate_diversity_benefit(
    portfolio_returns: pd.Series,
    component_returns: pd.DataFrame,
) -> float:
    """
    Calculate diversification benefit as variance reduction.

    Diversification Benefit = 1 - (Portfolio Variance / Sum of Component Variances)

    Args:
        portfolio_returns: Portfolio returns (weighted combination)
        component_returns: Individual component returns (T x K)

    Returns:
        Diversification benefit (0.0 to 1.0, higher = better diversification)
    """
    portfolio_var = portfolio_returns.var()
    component_vars = component_returns.var()
    sum_component_vars = component_vars.sum()

    if sum_component_vars == 0:
        return 0.0

    benefit = 1.0 - (portfolio_var / sum_component_vars)
    return max(0.0, min(1.0, benefit))
