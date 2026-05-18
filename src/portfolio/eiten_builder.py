"""Portfolio optimization builder wrapping eiten strategies.

Provides a unified interface to the 4 portfolio construction strategies
(Eigen, MVP, MSR, GA) with optional RMT covariance denoising and
compatibility with the project's signal pipeline.

Usage:
    from src.portfolio.eiten_builder import EitenPortfolioBuilder

    builder = EitenPortfolioBuilder(use_rmt=True)
    weights = builder.optimize(
        symbols=["SPY", "QQQ", "TLT", "GLD", "XLK"],
        returns_df=df,
        strategy="eigen",
    )
    # weights = {"SPY": 0.25, "QQQ": 0.30, "TLT": 0.15, ...}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .eiten_adapters import (
    EigenPortfolio,
    GeneticAlgorithmPortfolio,
    MaximumSharpePortfolio,
    MinimumVariancePortfolio,
    RMTFiltering,
)

logger = logging.getLogger(__name__)

VALID_STRATEGIES = ("eigen", "mvp", "msr", "ga")


@dataclass
class PortfolioOptimizationResult:
    """Result of a portfolio optimization run."""

    strategy: str
    """Strategy name used."""
    weights: dict[str, float]
    """Asset -> weight mapping."""
    expected_return: float | None = None
    """Expected annualized portfolio return."""
    expected_volatility: float | None = None
    """Expected annualized portfolio volatility."""
    expected_sharpe: float | None = None
    """Expected annualized Sharpe ratio."""
    rmt_applied: bool = False
    """Whether RMT denoising was applied."""
    rmt_n_kept: int | None = None
    """Number of eigenvalues surviving RMT filter."""
    ga_history: list[float] | None = None
    """GA best-Sharpe per generation (if GA strategy)."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional strategy-specific metadata."""


class EitenPortfolioBuilder:
    """Portfolio optimization orchestrator.

    Parameters:
        use_rmt: Apply RMT covariance denoising (default True).
        rmt_sigma: RMT standard deviation parameter (default 1.0).
        ga_population_size: GA population size.
        ga_generations: GA generations.
        ga_seed: GA random seed.
        eigen_number: Which eigen portfolio (2 = orthogonal to market).
    """

    def __init__(
        self,
        use_rmt: bool = True,
        rmt_sigma: float = 1.0,
        ga_population_size: int = 200,
        ga_generations: int = 50,
        ga_seed: int | None = None,
        eigen_number: int = 2,
    ) -> None:
        self.use_rmt = use_rmt
        self.rmt = RMTFiltering(sigma=rmt_sigma)
        self.eigen = EigenPortfolio(eigen_number=eigen_number)
        self.mvp = MinimumVariancePortfolio()
        self.msr = MaximumSharpePortfolio()
        self.ga = GeneticAlgorithmPortfolio(
            population_size=ga_population_size,
            generations=ga_generations,
            random_seed=ga_seed,
        )

    def optimize(
        self,
        symbols: list[str],
        returns_df: pd.DataFrame,
        strategy: str = "msr",
    ) -> PortfolioOptimizationResult:
        """Run portfolio optimization.

        Args:
            symbols: List of asset symbols (must be columns in returns_df).
            returns_df: DataFrame with daily return columns per symbol.
            strategy: One of 'eigen', 'mvp', 'msr', 'ga'.

        Returns:
            PortfolioOptimizationResult with weights and metrics.

        Raises:
            ValueError: If strategy is invalid or symbols are missing.
        """
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. Must be one of: {', '.join(VALID_STRATEGIES)}"
            )

        missing = [s for s in symbols if s not in returns_df.columns]
        if missing:
            raise ValueError(f"Symbols not found in returns_df columns: {missing}")

        returns_subset = returns_df[symbols].dropna()
        if len(returns_subset) < 30:
            logger.warning(
                "Only %d valid return rows — results may be unstable", len(returns_subset)
            )

        returns_matrix = returns_subset.values
        returns_matrix_pct = returns_matrix * 100  # GA uses percentage returns

        cov_matrix = np.cov(returns_matrix, rowvar=False)
        rmt_n_kept: int | None = None

        if self.use_rmt:
            rmt_result = self.rmt.denoise(returns_matrix)
            cov_matrix = self.rmt.ensure_psd(rmt_result.filtered_cov)
            rmt_n_kept = rmt_result.n_kept

        exp_returns = np.mean(returns_matrix, axis=0)

        if strategy == "eigen":
            weights_dict = self.eigen.build(symbols, cov_matrix)
        elif strategy == "mvp":
            weights_dict = self.mvp.build(symbols, cov_matrix)
        elif strategy == "msr":
            weights_dict = self.msr.build(symbols, cov_matrix, exp_returns)
        elif strategy == "ga":
            ga_result = self.ga.optimize(symbols, returns_matrix_pct)
            weights_dict = ga_result.weights
        else:
            weights_dict = {}

        weights_array = np.array([weights_dict.get(s, 0.0) for s in symbols])
        port_return = float(weights_array @ exp_returns) * 252
        port_vol = float(
            np.sqrt(weights_array @ np.cov(returns_matrix, rowvar=False) @ weights_array)
        ) * np.sqrt(252)
        port_sharpe = port_return / port_vol if port_vol > 1e-12 else 0.0

        metadata: dict[str, Any] = {}
        ga_history: list[float] | None = None

        if strategy == "ga":
            ga_history = ga_result.history

        if strategy == "eigen":
            metadata["eigen_number"] = self.eigen.eigen_number

        return PortfolioOptimizationResult(
            strategy=strategy,
            weights=weights_dict,
            expected_return=port_return,
            expected_volatility=port_vol,
            expected_sharpe=port_sharpe,
            rmt_applied=self.use_rmt,
            rmt_n_kept=rmt_n_kept,
            ga_history=ga_history,
            metadata=metadata,
        )

    def compare(
        self,
        symbols: list[str],
        returns_df: pd.DataFrame,
    ) -> dict[str, PortfolioOptimizationResult]:
        """Run all 4 strategies and return comparison.

        Args:
            symbols: List of asset symbols.
            returns_df: DataFrame with daily return columns.

        Returns:
            Dict mapping strategy name to result.
        """
        results: dict[str, PortfolioOptimizationResult] = {}
        for strategy in VALID_STRATEGIES:
            try:
                results[strategy] = self.optimize(symbols, returns_df, strategy=strategy)
            except Exception as exc:
                logger.error("Strategy %s failed: %s", strategy, exc)
        return results

    def optimize_from_signals(
        self,
        signals: dict[str, float],
        returns_df: pd.DataFrame,
        strategy: str = "msr",
        min_abs_signal: float = 0.05,
    ) -> PortfolioOptimizationResult:
        """Optimize weights using signal scores as expected return proxies.

        Active instruments are filtered to those with |signal| >= min_abs_signal.
        Signal strengths are used as expected return estimates for MSR/GA.

        Args:
            signals: Dict of symbol -> signal score (-1.0 to 1.0).
            returns_df: DataFrame with daily return columns.
            strategy: Optimization strategy.
            min_abs_signal: Minimum absolute signal to include.

        Returns:
            PortfolioOptimizationResult.
        """
        active_symbols = [
            s for s, v in signals.items() if abs(v) >= min_abs_signal and s in returns_df.columns
        ]

        if not active_symbols:
            logger.warning("No symbols meet min_abs_signal threshold")
            return PortfolioOptimizationResult(
                strategy=strategy,
                weights={},
            )

        return self.optimize(active_symbols, returns_df, strategy=strategy)
