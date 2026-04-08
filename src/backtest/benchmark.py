"""
Benchmark Comparison Module

Provides metrics to compare strategy performance against a benchmark.
Answers the question: "Is this the market growing or is my strategy being smart?"
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]


@dataclass
class BenchmarkComparison:
    """
    Calculate benchmark comparison metrics.

    Compares strategy returns against a benchmark (e.g., SPY buy-and-hold)
    to determine if the strategy adds alpha or simply captures market beta.

    Key Metrics:
        - Alpha: Excess return after adjusting for market exposure
        - Beta: Sensitivity to market movements
        - Information Ratio: Risk-adjusted excess return
        - Tracking Error: Volatility of excess returns
        - R-squared: How much strategy variance is explained by benchmark
    """

    @staticmethod
    def calculate(
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> Dict[str, Any]:
        """
        Calculate benchmark comparison metrics.

        Args:
            strategy_returns: Daily returns of the strategy (as decimals, e.g., 0.01 for 1%)
            benchmark_returns: Daily returns of the benchmark (as decimals)
            risk_free_rate: Annual risk-free rate (default 2%)
            periods_per_year: Trading periods per year (252 for daily, 12 for monthly)

        Returns:
            Dictionary with benchmark comparison metrics
        """
        # Align returns by index
        aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 30:
            return {
                "error": "Insufficient data for benchmark comparison",
                "min_observations": 30,
                "actual_observations": len(aligned),
            }

        strat = aligned.iloc[:, 0]
        bench = aligned.iloc[:, 1]

        metrics = {}

        # Daily risk-free rate
        daily_rf = risk_free_rate / periods_per_year

        # =================================================================
        # Basic Return Comparison
        # =================================================================

        # Total returns (compound)
        metrics["strategy_total_return"] = (1 + strat).prod() - 1  # type: ignore[operator, arg-type]
        metrics["benchmark_total_return"] = (1 + bench).prod() - 1  # type: ignore[operator, arg-type]
        metrics["excess_return"] = (
            metrics["strategy_total_return"] - metrics["benchmark_total_return"]
        )  # type: ignore[operator]

        # Annualized returns
        total_days = len(strat)
        years = total_days / periods_per_year
        metrics["strategy_annualized_return"] = (1 + metrics["strategy_total_return"]) ** (
            1 / years
        ) - 1  # type: ignore[operator, arg-type]
        metrics["benchmark_annualized_return"] = (1 + metrics["benchmark_total_return"]) ** (
            1 / years
        ) - 1  # type: ignore[operator, arg-type]

        # Relative strength
        if metrics["benchmark_total_return"] != 0:
            metrics["relative_strength"] = (1 + metrics["strategy_total_return"]) / (
                1 + metrics["benchmark_total_return"]
            ) - 1  # type: ignore[operator, arg-type]
        else:
            metrics["relative_strength"] = 0

        # =================================================================
        # Beta: Market Sensitivity
        # =================================================================

        # Beta = Cov(Strat, Bench) / Var(Bench)
        covariance = np.cov(strat, bench, ddof=1)[0, 1]
        benchmark_variance = np.var(bench, ddof=1)
        metrics["beta"] = covariance / benchmark_variance if benchmark_variance > 0 else 0

        # =================================================================
        # Alpha: Excess Return (Jensen's Alpha)
        # =================================================================

        # Annualized mean returns
        annualized_strat = np.mean(strat) * periods_per_year
        annualized_bench = np.mean(bench) * periods_per_year

        # Jensen's Alpha = Rp - [Rf + β × (Rm - Rf)]
        metrics["alpha"] = annualized_strat - (
            risk_free_rate + metrics["beta"] * (annualized_bench - risk_free_rate)
        )

        # Alpha as percentage
        metrics["alpha_pct"] = metrics["alpha"] * 100

        # =================================================================
        # Tracking Error & Information Ratio
        # =================================================================

        # Excess returns over benchmark
        excess_returns = strat - bench

        # Tracking Error: StdDev of excess returns (annualized)
        metrics["tracking_error"] = np.std(excess_returns, ddof=1) * np.sqrt(periods_per_year)

        # Information Ratio: (Return - Benchmark) / Tracking Error
        if metrics["tracking_error"] > 0:
            # Annualized excess return divided by tracking error
            annualized_excess = (
                metrics["strategy_annualized_return"] - metrics["benchmark_annualized_return"]
            )
            metrics["information_ratio"] = annualized_excess / metrics["tracking_error"]
        else:
            metrics["information_ratio"] = 0

        # =================================================================
        # Correlation & R-Squared
        # =================================================================

        # Correlation coefficient
        correlation = np.corrcoef(strat, bench)[0, 1]
        metrics["correlation"] = correlation if not np.isnan(correlation) else 0

        # R-squared: Proportion of variance explained by benchmark
        metrics["r_squared"] = metrics["correlation"] ** 2

        # =================================================================
        # Risk-Adjusted Performance vs Benchmark
        # =================================================================

        # Strategy Sharpe Ratio
        strat_std = np.std(strat, ddof=1)
        if strat_std > 0:
            metrics["strategy_sharpe"] = (
                (np.mean(strat) - daily_rf) / strat_std * np.sqrt(periods_per_year)
            )
        else:
            metrics["strategy_sharpe"] = 0

        # Benchmark Sharpe Ratio
        bench_std = np.std(bench, ddof=1)
        if bench_std > 0:
            metrics["benchmark_sharpe"] = (
                (np.mean(bench) - daily_rf) / bench_std * np.sqrt(periods_per_year)
            )
        else:
            metrics["benchmark_sharpe"] = 0

        # Treynor Ratio: (Rp - Rf) / β
        if metrics["beta"] > 0:
            metrics["treynor_ratio"] = (annualized_strat - risk_free_rate) / metrics["beta"]
        else:
            metrics["treynor_ratio"] = 0

        # M2 Modigliani-Modigliani Measure
        if bench_std > 0 and strat_std > 0:
            strat_sharpe = metrics["strategy_sharpe"]
            metrics["m2_measure"] = (
                strat_sharpe * bench_std * np.sqrt(periods_per_year) + risk_free_rate
            )
        else:
            metrics["m2_measure"] = 0

        # =================================================================
        # Statistical Significance Tests
        # =================================================================

        # T-test for alpha significance
        # Standard error of alpha
        if metrics["beta"] != 0 and len(excess_returns) > 2:
            # Regression-based standard error
            residuals = excess_returns - (metrics["alpha"] / periods_per_year)  # type: ignore[operator]
            residual_std = np.std(residuals, ddof=2)
            se_alpha = residual_std / np.sqrt(len(excess_returns))

            if se_alpha > 0:
                metrics["alpha_t_stat"] = metrics["alpha"] / (se_alpha * np.sqrt(periods_per_year))
                metrics["alpha_p_value"] = 2 * (
                    1 - stats.t.cdf(abs(metrics["alpha_t_stat"]), len(excess_returns) - 2)
                )
            else:
                metrics["alpha_t_stat"] = 0
                metrics["alpha_p_value"] = 1
        else:
            metrics["alpha_t_stat"] = 0
            metrics["alpha_p_value"] = 1

        # Alpha significance flag
        metrics["alpha_significant"] = metrics["alpha_p_value"] < 0.05

        # =================================================================
        # Market Regime Analysis
        # =================================================================

        # Up vs Down market performance
        up_market = bench > 0
        down_market = bench < 0

        if up_market.sum() > 0:
            metrics["up_capture"] = (
                strat[up_market].mean() / bench[up_market].mean()
                if bench[up_market].mean() != 0
                else 0
            )
            metrics["up_market_return"] = strat[up_market].mean() * periods_per_year
        else:
            metrics["up_capture"] = 0
            metrics["up_market_return"] = 0

        if down_market.sum() > 0:
            metrics["down_capture"] = (
                strat[down_market].mean() / bench[down_market].mean()
                if bench[down_market].mean() != 0
                else 0
            )
            metrics["down_market_return"] = strat[down_market].mean() * periods_per_year
        else:
            metrics["down_capture"] = 0
            metrics["down_market_return"] = 0

        # Batting average: % of periods strategy beats benchmark
        metrics["batting_average"] = (excess_returns > 0).sum() / len(excess_returns)

        # =================================================================
        # Summary Interpretation
        # =================================================================

        metrics["interpretation"] = BenchmarkComparison._interpret_metrics(metrics)

        return metrics

    @staticmethod
    def _interpret_metrics(metrics: Dict[str, Any]) -> str:
        """Generate human-readable interpretation of metrics."""
        lines = []

        # Alpha interpretation
        alpha = metrics.get("alpha", 0)
        if alpha > 0.02:
            lines.append(
                f"Strong positive alpha ({alpha * 100:.2f}%): Strategy outperforms benchmark significantly."
            )
        elif alpha > 0:
            lines.append(
                f"Positive alpha ({alpha * 100:.2f}%): Strategy adds value above benchmark."
            )
        elif alpha > -0.02:
            lines.append(
                f"Near-zero alpha ({alpha * 100:.2f}%): Strategy tracks benchmark closely."
            )
        else:
            lines.append(f"Negative alpha ({alpha * 100:.2f}%): Strategy underperforms benchmark.")

        # Beta interpretation
        beta = metrics.get("beta", 1)
        if beta < 0.5:
            lines.append(
                f"Low beta ({beta:.2f}): Strategy is defensive, less volatile than market."
            )
        elif beta < 0.9:
            lines.append(f"Moderate beta ({beta:.2f}): Strategy has lower volatility than market.")
        elif beta < 1.1:
            lines.append(f"Market-neutral beta ({beta:.2f}): Strategy moves closely with market.")
        elif beta < 1.5:
            lines.append(f"Elevated beta ({beta:.2f}): Strategy is more aggressive than market.")
        else:
            lines.append(
                f"High beta ({beta:.2f}): Strategy is highly aggressive, amplifies market moves."
            )

        # Information Ratio interpretation
        ir = metrics.get("information_ratio", 0)
        if ir > 0.5:
            lines.append(
                f"Good information ratio ({ir:.2f}): Consistent excess returns vs benchmark."
            )
        elif ir > 0:
            lines.append(
                f"Positive information ratio ({ir:.2f}): Some consistency in excess returns."
            )
        else:
            lines.append(
                f"Negative information ratio ({ir:.2f}): Inconsistent performance vs benchmark."
            )

        # R-squared interpretation
        r2 = metrics.get("r_squared", 0)
        if r2 > 0.8:
            lines.append(f"High R-squared ({r2:.2f}): Most strategy variance explained by market.")
        elif r2 > 0.5:
            lines.append(f"Moderate R-squared ({r2:.2f}): Some correlation with market.")
        else:
            lines.append(f"Low R-squared ({r2:.2f}): Strategy is largely independent of market.")

        return "\n".join(lines)

    @staticmethod
    def calculate_from_equity(
        strategy_equity: pd.Series,
        benchmark_prices: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> Dict[str, Any]:
        """
        Calculate benchmark comparison from equity curve and benchmark prices.

        Args:
            strategy_equity: Strategy equity curve over time
            benchmark_prices: Benchmark price series (e.g., SPY Close prices)
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            Dictionary with benchmark comparison metrics
        """
        # Calculate returns
        strategy_returns = strategy_equity.pct_change().dropna()
        benchmark_returns = benchmark_prices.pct_change().dropna()

        return BenchmarkComparison.calculate(
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )

    @staticmethod
    def calculate_buy_hold_comparison(
        trades: list,
        benchmark_prices: pd.Series,
        initial_equity: float = 100000.0,
        risk_free_rate: float = 0.02,
    ) -> Dict[str, Any]:
        """
        Calculate benchmark comparison from trades vs buy-and-hold.

        Args:
            trades: List of trade dictionaries with 'entry_time', 'exit_time', 'pnl'
            benchmark_prices: Benchmark price series (e.g., SPY Close prices)
            initial_equity: Starting equity
            risk_free_rate: Annual risk-free rate

        Returns:
            Dictionary with benchmark comparison metrics
        """
        if not trades:
            return {"error": "No trades provided"}

        # Build strategy equity curve from trades
        equity = initial_equity
        equity_curve = {}

        for trade in trades:
            exit_time = trade.get("exit_time")
            pnl = trade.get("pnl", 0)

            if exit_time:
                equity += pnl
                equity_curve[exit_time] = equity

        # Convert to Series
        strategy_equity = pd.Series(equity_curve)
        strategy_equity = strategy_equity.sort_index()

        # Filter benchmark to match strategy period
        start_date = min(trades, key=lambda x: x.get("entry_time", pd.Timestamp.max)).get(
            "entry_time"
        )
        end_date = max(trades, key=lambda x: x.get("exit_time", pd.Timestamp.min)).get("exit_time")

        if start_date and end_date:
            benchmark_filtered = benchmark_prices.loc[start_date:end_date]
        else:
            benchmark_filtered = benchmark_prices

        return BenchmarkComparison.calculate_from_equity(
            strategy_equity=strategy_equity,
            benchmark_prices=benchmark_filtered,
            risk_free_rate=risk_free_rate,
        )
