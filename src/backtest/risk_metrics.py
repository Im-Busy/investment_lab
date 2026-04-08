"""
Advanced Risk Metrics Module

Provides VaR, CVaR, Ulcer Index, and other risk measurements.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class RiskMetrics:
    """
    Calculate advanced risk metrics for trading strategies.

    Key Metrics:
        - Value at Risk (VaR): Maximum expected loss at given confidence
        - Conditional VaR (CVaR): Expected loss beyond VaR (tail risk)
        - Ulcer Index: Measures severity and duration of drawdowns
        - Recovery Factor: How well strategy recovers from drawdowns
        - Kelly Criterion: Optimal position sizing
    """

    @staticmethod
    def calculate(
        returns: pd.Series,
        equity_curve: pd.Series,
        trades: Optional[Union[List[Dict[str, Any]], pd.DataFrame]] = None,
        initial_equity: float = 100000.0,
        confidence_levels: List[float] = [0.95, 0.99],
    ) -> Dict[str, Any]:
        """
        Calculate advanced risk metrics.

        Args:
            returns: Daily returns series (as decimals, e.g., 0.01 for 1%)
            equity_curve: Equity values over time
            trades: Optional list of trade dictionaries for Kelly calculation
            initial_equity: Starting equity
            confidence_levels: Confidence levels for VaR calculation

        Returns:
            Dictionary with risk metrics
        """
        metrics = {}

        # Ensure returns is a Series
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        # Clean returns
        returns = returns.dropna()

        if len(returns) == 0:
            return {"error": "No valid returns data"}

        # =================================================================
        # Value at Risk (VaR)
        # =================================================================

        # Historical VaR
        for conf in confidence_levels:
            var_level = int((1 - conf) * 100)
            metrics[f"var_{var_level}"] = np.percentile(returns, (1 - conf) * 100)
            metrics[f"var_{var_level}_pct"] = metrics[f"var_{var_level}"] * 100

        # Parametric VaR (assuming normal distribution)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        from scipy import stats  # noqa: F811

        for conf in confidence_levels:
            var_level = int((1 - conf) * 100)
            z_score = stats.norm.ppf(1 - conf)
            metrics[f"var_{var_level}_parametric"] = mean_return + z_score * std_return
            metrics[f"var_{var_level}_parametric_pct"] = (
                metrics[f"var_{var_level}_parametric"] * 100
            )

        # =================================================================
        # Conditional VaR (Expected Shortfall)
        # =================================================================

        for conf in confidence_levels:
            var_level = int((1 - conf) * 100)
            var_key = f"var_{var_level}"
            if var_key in metrics:
                # CVaR is the mean of returns below VaR
                cvar = returns[returns <= metrics[var_key]].mean()
                metrics[f"cvar_{var_level}"] = cvar if not np.isnan(cvar) else metrics[var_key]
                metrics[f"cvar_{var_level}_pct"] = metrics[f"cvar_{var_level}"] * 100

        # =================================================================
        # Drawdown-Based Metrics
        # =================================================================

        # Calculate drawdown series
        if isinstance(equity_curve, pd.Series):
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak
            drawdown_pct = drawdown * 100
        else:
            # Build from returns
            equity = initial_equity * (1 + returns).cumprod()
            peak = pd.Series(equity).expanding().max()
            drawdown = (equity - peak.values) / peak.values
            drawdown_pct = drawdown * 100

        # Maximum Drawdown
        metrics["max_drawdown"] = drawdown.min()
        metrics["max_drawdown_pct"] = metrics["max_drawdown"] * 100

        # Average Drawdown
        metrics["avg_drawdown"] = drawdown.mean()
        metrics["avg_drawdown_pct"] = metrics["avg_drawdown"] * 100

        # =================================================================
        # Ulcer Index
        # =================================================================
        # Ulcer Index = sqrt(mean(drawdown_pct^2))
        # More weight on larger drawdowns and longer duration

        squared_drawdown = drawdown_pct**2
        metrics["ulcer_index"] = np.sqrt(squared_drawdown.mean())

        # Ulcer Performance Index (UPI) = (Return - RiskFree) / Ulcer Index
        total_return = (
            (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 1 else 0
        )
        years = len(returns) / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        if metrics["ulcer_index"] > 0:
            metrics["ulcer_performance_index"] = (annualized_return - 0.02) / (
                metrics["ulcer_index"] / 100
            )
        else:
            metrics["ulcer_performance_index"] = 0

        # =================================================================
        # Pain Index
        # =================================================================
        # Average drawdown over time (simpler than Ulcer Index)

        metrics["pain_index"] = np.abs(drawdown_pct).mean()

        # Pain Ratio = (Return - RiskFree) / Pain Index
        if metrics["pain_index"] > 0:
            metrics["pain_ratio"] = (annualized_return - 0.02) / (metrics["pain_index"] / 100)
        else:
            metrics["pain_ratio"] = 0

        # =================================================================
        # Recovery Factor
        # =================================================================
        # Recovery Factor = Net Profit / Max Drawdown

        net_profit = equity_curve.iloc[-1] - initial_equity if len(equity_curve) > 0 else 0
        max_dd_abs = initial_equity * abs(metrics["max_drawdown"])

        if max_dd_abs > 0:
            metrics["recovery_factor"] = net_profit / max_dd_abs
        else:
            metrics["recovery_factor"] = float("inf") if net_profit > 0 else 0

        # =================================================================
        # Kelly Criterion
        # =================================================================

        if trades is not None and len(trades) > 0:
            # Convert to DataFrame if needed
            if isinstance(trades, pd.DataFrame):
                trades_df = trades
            else:
                trades_df = pd.DataFrame(trades)

            # Get wins and losses using DataFrame operations
            if "pnl" in trades_df.columns:
                wins = trades_df[trades_df["pnl"] > 0]["pnl"].values
                losses = np.abs(trades_df[trades_df["pnl"] < 0]["pnl"].values)
            else:
                wins = np.array([])
                losses = np.array([])

            if len(wins) > 0 and len(losses) > 0:
                win_rate = len(wins) / len(trades)
                avg_win = float(np.mean(wins))  # type: ignore[arg-type]
                avg_loss = float(np.mean(losses))  # type: ignore[arg-type]

                if avg_loss > 0:
                    # Kelly = W - (1-W) / (Avg Win / Avg Loss)
                    win_loss_ratio = avg_win / avg_loss
                    metrics["kelly_fraction"] = win_rate - (1 - win_rate) / win_loss_ratio
                    metrics["kelly_fraction"] = max(
                        0, min(1, metrics["kelly_fraction"])
                    )  # Cap at 0-1
                else:
                    metrics["kelly_fraction"] = 0

                metrics["win_rate"] = win_rate
                metrics["avg_win"] = avg_win
                metrics["avg_loss"] = avg_loss
                metrics["win_loss_ratio"] = win_loss_ratio if avg_loss > 0 else 0
            else:
                metrics["kelly_fraction"] = 0
        else:
            # Calculate Kelly from returns
            wins = returns[returns > 0]
            losses = returns[returns < 0]

            if len(wins) > 0 and len(losses) > 0:
                win_rate = len(wins) / len(returns)
                avg_win = np.mean(wins)
                avg_loss = abs(np.mean(losses))

                if avg_loss > 0:
                    win_loss_ratio = avg_win / avg_loss
                    metrics["kelly_fraction"] = win_rate - (1 - win_rate) / win_loss_ratio
                    metrics["kelly_fraction"] = max(0, min(1, metrics["kelly_fraction"]))
                else:
                    metrics["kelly_fraction"] = 0

                metrics["win_rate"] = win_rate
                metrics["avg_win"] = avg_win
                metrics["avg_loss"] = avg_loss
                metrics["win_loss_ratio"] = win_loss_ratio if avg_loss > 0 else 0
            else:
                metrics["kelly_fraction"] = 0

        # =================================================================
        # Tail Risk Metrics
        # =================================================================

        # Skewness and Kurtosis
        metrics["skewness"] = returns.skew()
        metrics["kurtosis"] = returns.kurtosis()

        # Excess kurtosis (kurtosis - 3)
        metrics["excess_kurtosis"] = metrics["kurtosis"] - 3  # type: ignore[operator]

        # Tail ratio (95th percentile / 5th percentile)
        percentile_95 = np.percentile(returns, 95)
        percentile_5 = np.percentile(returns, 5)
        if percentile_5 != 0:
            metrics["tail_ratio"] = abs(percentile_95 / percentile_5)
        else:
            metrics["tail_ratio"] = 0

        # =================================================================
        # Volatility Metrics
        # =================================================================

        # Annualized volatility
        metrics["annualized_volatility"] = std_return * np.sqrt(252) * 100

        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            metrics["downside_deviation"] = np.std(downside_returns, ddof=1) * np.sqrt(252) * 100
        else:
            metrics["downside_deviation"] = 0

        # Upside deviation (only positive returns)
        upside_returns = returns[returns > 0]
        if len(upside_returns) > 0:
            metrics["upside_deviation"] = np.std(upside_returns, ddof=1) * np.sqrt(252) * 100
        else:
            metrics["upside_deviation"] = 0

        # =================================================================
        # Risk Summary
        # =================================================================

        metrics["risk_summary"] = RiskMetrics._generate_summary(metrics)

        return metrics

    @staticmethod
    def _generate_summary(metrics: Dict[str, Any]) -> str:
        """Generate human-readable risk summary."""
        lines = []

        # VaR summary
        var_95 = metrics.get("var_95_pct", 0)
        lines.append(
            f"Value at Risk (95%): {var_95:.2f}% - Expected maximum daily loss under normal conditions."
        )

        # CVaR summary
        cvar_95 = metrics.get("cvar_95_pct", 0)
        lines.append(
            f"Expected Shortfall (95%): {cvar_95:.2f}% - Average loss in worst 5% of days."
        )

        # Ulcer Index
        ulcer = metrics.get("ulcer_index", 0)
        if ulcer < 5:
            ulcer_interp = "Low drawdown stress"
        elif ulcer < 10:
            ulcer_interp = "Moderate drawdown stress"
        elif ulcer < 20:
            ulcer_interp = "High drawdown stress"
        else:
            ulcer_interp = "Extreme drawdown stress"
        lines.append(f"Ulcer Index: {ulcer:.2f} - {ulcer_interp}.")

        # Recovery Factor
        recovery = metrics.get("recovery_factor", 0)
        if recovery > 3:
            rec_interp = "Excellent recovery ability"
        elif recovery > 2:
            rec_interp = "Good recovery ability"
        elif recovery > 1:
            rec_interp = "Moderate recovery ability"
        else:
            rec_interp = "Poor recovery ability"
        lines.append(f"Recovery Factor: {recovery:.2f} - {rec_interp}.")

        # Kelly
        kelly = metrics.get("kelly_fraction", 0)
        lines.append(
            f"Kelly Fraction: {kelly * 100:.1f}% - Suggested maximum position size for optimal growth."
        )

        # Tail risk
        skew = metrics.get("skewness", 0)
        kurt = metrics.get("excess_kurtosis", 0)
        if skew < -0.5:
            skew_interp = "left-skewed (more extreme losses)"
        elif skew > 0.5:
            skew_interp = "right-skewed (more extreme gains)"
        else:
            skew_interp = "approximately symmetric"

        if kurt > 1:
            kurt_interp = "fat tails (more extreme outcomes than normal)"
        elif kurt < -1:
            kurt_interp = "thin tails (fewer extreme outcomes)"
        else:
            kurt_interp = "near-normal tails"

        lines.append(f"Return Distribution: {skew_interp}, {kurt_interp}.")

        return "\n".join(lines)

    @staticmethod
    def calculate_from_trades(
        trades: List[Dict[str, Any]], initial_equity: float = 100000.0
    ) -> Dict[str, Any]:
        """
        Calculate risk metrics from trade list.

        Args:
            trades: List of trade dictionaries with 'pnl' and optionally 'entry_time', 'exit_time'
            initial_equity: Starting equity

        Returns:
            Dictionary with risk metrics
        """
        if not trades:
            return {"error": "No trades provided"}

        # Build equity curve from trades
        equity = initial_equity
        equity_values = [initial_equity]
        timestamps = []

        for trade in trades:
            pnl = trade.get("pnl", 0)
            equity += pnl
            equity_values.append(equity)

            exit_time = trade.get("exit_time")
            if exit_time:
                timestamps.append(exit_time)

        # Create Series
        if timestamps:
            equity_curve = pd.Series(equity_values[1:], index=timestamps)
        else:
            equity_curve = pd.Series(equity_values)

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        return RiskMetrics.calculate(
            returns=returns, equity_curve=equity_curve, trades=trades, initial_equity=initial_equity
        )


@dataclass
class DrawdownAnalyzer:
    """
    Detailed drawdown analysis.
    """

    @staticmethod
    def analyze(equity_curve: pd.Series) -> Dict[str, Any]:
        """
        Perform detailed drawdown analysis.

        Args:
            equity_curve: Equity values over time

        Returns:
            Dictionary with drawdown analysis
        """
        if len(equity_curve) < 2:
            return {"error": "Insufficient data for drawdown analysis"}

        # Calculate drawdown series
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        drawdown_pct = drawdown * 100

        metrics = {}

        # Basic drawdown stats
        metrics["max_drawdown"] = drawdown.min()
        metrics["max_drawdown_pct"] = drawdown_pct.min()
        metrics["avg_drawdown"] = drawdown.mean()
        metrics["avg_drawdown_pct"] = drawdown_pct.mean()

        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_starts = []
        drawdown_ends = []
        drawdown_depths = []
        drawdown_durations = []

        current_start = None
        current_depth = 0

        for i, (idx, dd) in enumerate(drawdown.items()):
            if dd < 0 and current_start is None:
                current_start = idx
                current_depth = dd
            elif dd < 0 and current_start is not None:
                current_depth = min(current_depth, dd)
            elif dd >= 0 and current_start is not None:
                drawdown_starts.append(current_start)
                drawdown_ends.append(idx)
                drawdown_depths.append(current_depth)
                drawdown_durations.append(len(equity_curve.loc[current_start:idx]) - 1)
                current_start = None
                current_depth = 0

        # Handle if still in drawdown at end
        if current_start is not None:
            drawdown_starts.append(current_start)
            drawdown_ends.append(equity_curve.index[-1])
            drawdown_depths.append(current_depth)
            drawdown_durations.append(len(equity_curve.loc[current_start:]) - 1)

        # Drawdown statistics
        if drawdown_durations:
            metrics["num_drawdowns"] = len(drawdown_durations)
            metrics["avg_drawdown_duration"] = np.mean(drawdown_durations)
            metrics["max_drawdown_duration"] = max(drawdown_durations)
            metrics["avg_drawdown_depth"] = np.mean(drawdown_depths)
        else:
            metrics["num_drawdowns"] = 0
            metrics["avg_drawdown_duration"] = 0
            metrics["max_drawdown_duration"] = 0
            metrics["avg_drawdown_depth"] = 0

        # Time in drawdown
        metrics["time_in_drawdown"] = (drawdown < 0).sum() / len(drawdown)

        # Worst drawdown details
        if drawdown_depths:
            worst_idx = np.argmin(drawdown_depths)
            metrics["worst_drawdown_start"] = drawdown_starts[worst_idx]
            metrics["worst_drawdown_end"] = drawdown_ends[worst_idx]
            metrics["worst_drawdown_depth"] = drawdown_depths[worst_idx]
            metrics["worst_drawdown_duration"] = drawdown_durations[worst_idx]

        return metrics
