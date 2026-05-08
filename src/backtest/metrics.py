"""
Performance Metrics

Calculates comprehensive performance metrics for backtesting.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """
    Performance Metrics Calculator

    Calculates and stores comprehensive performance metrics.
    """

    @staticmethod
    def calculate(
        trades: Union[List[Dict[str, Any]], pd.DataFrame],
        equity_curve: Optional[pd.DataFrame] = None,
        initial_equity: float = 100000.0,
        risk_free_rate: float = 0.02,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Args:
            trades: List of trade dictionaries
            equity_curve: DataFrame with equity over time
            initial_equity: Starting equity
            risk_free_rate: Annual risk-free rate for Sharpe calculation

        Returns:
            Dictionary with performance metrics
        """
        if trades is None or len(trades) == 0:
            return {
                "total_trades": 0,
                "initial_equity": initial_equity,
                "final_equity": initial_equity,
                "total_return": 0.0,
                "message": "No trades to analyze",
            }

        metrics: Dict[str, Any] = {}  # type: ignore[assignment]

        # Basic trade statistics
        metrics["total_trades"] = len(trades)

        # Convert to DataFrame if needed for consistent processing
        if isinstance(trades, pd.DataFrame):
            trades_df = trades
        else:
            trades_df = pd.DataFrame(trades)

        # Separate winning and losing trades using DataFrame operations
        if "pnl" in trades_df.columns:
            wins = trades_df[trades_df["pnl"] > 0]
            losses = trades_df[trades_df["pnl"] < 0]
            breakeven = trades_df[trades_df["pnl"] == 0]

            metrics["winning_trades"] = len(wins)
            metrics["losing_trades"] = len(losses)
            metrics["breakeven_trades"] = len(breakeven)

            # Win rate
            metrics["win_rate"] = len(wins) / len(trades) if len(trades) > 0 else 0

            # Profit/Loss metrics
            total_pnl = float(trades_df["pnl"].sum())
            metrics["total_pnl"] = total_pnl
            metrics["total_return"] = (total_pnl / initial_equity) * 100

            # Average win/loss
            win_amounts = wins["pnl"].values if len(wins) > 0 else np.array([])
            loss_amounts = np.abs(losses["pnl"].values) if len(losses) > 0 else np.array([])

            metrics["avg_win"] = float(np.mean(win_amounts)) if len(win_amounts) > 0 else 0  # type: ignore[arg-type]
            metrics["avg_loss"] = float(np.mean(loss_amounts)) if len(loss_amounts) > 0 else 0  # type: ignore[arg-type]
            metrics["largest_win"] = float(np.max(win_amounts)) if len(win_amounts) > 0 else 0  # type: ignore[arg-type]
            metrics["largest_loss"] = float(np.max(loss_amounts)) if len(loss_amounts) > 0 else 0  # type: ignore[arg-type]

            # Profit factor
            total_wins = float(np.sum(win_amounts))  # type: ignore[arg-type]
            total_losses = float(np.sum(loss_amounts))  # type: ignore[arg-type]
            metrics["profit_factor"] = (
                total_wins / total_losses if total_losses > 0 else float("inf")
            )
        else:
            total_pnl = 0
            metrics["winning_trades"] = 0
            metrics["losing_trades"] = 0
            metrics["breakeven_trades"] = 0
            metrics["win_rate"] = 0
            metrics["total_pnl"] = 0
            metrics["total_return"] = 0
            metrics["avg_win"] = 0
            metrics["avg_loss"] = 0
            metrics["largest_win"] = 0
            metrics["largest_loss"] = 0
            metrics["profit_factor"] = 0

        # Expectancy
        if metrics["win_rate"] > 0 and metrics["avg_loss"] > 0:
            expectancy = (metrics["win_rate"] * metrics["avg_win"]) - (
                (1 - metrics["win_rate"]) * metrics["avg_loss"]
            )
            metrics["expectancy"] = expectancy
        else:
            metrics["expectancy"] = 0

        # Risk/Reward ratio
        metrics["risk_reward_ratio"] = (
            metrics["avg_win"] / metrics["avg_loss"] if metrics["avg_loss"] > 0 else 0
        )

        # Final equity
        metrics["initial_equity"] = initial_equity
        metrics["final_equity"] = initial_equity + total_pnl

        # Calculate equity curve metrics if available
        if equity_curve is not None and len(equity_curve) > 0:
            metrics.update(
                PerformanceMetrics._calculate_equity_metrics(
                    equity_curve, initial_equity, risk_free_rate
                )
            )

        # Calculate drawdown metrics
        metrics.update(PerformanceMetrics._calculate_drawdown_metrics(trades_df, initial_equity))

        # Calculate streak metrics
        metrics.update(PerformanceMetrics._calculate_streak_metrics(trades_df))

        # Calculate holding period metrics
        metrics.update(PerformanceMetrics._calculate_holding_metrics(trades_df))

        return metrics

    @staticmethod
    def _calculate_equity_metrics(
        equity_curve: pd.DataFrame, initial_equity: float, risk_free_rate: float
    ) -> Dict[str, Any]:
        """Calculate equity curve based metrics."""
        metrics: Dict[str, Any] = {}  # type: ignore[assignment]

        if "equity" not in equity_curve.columns:
            return metrics

        equity = np.asarray(equity_curve["equity"].values)

        # Returns
        returns = np.diff(equity) / equity[:-1]  # type: ignore[arg-type]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

        if len(returns) > 0:
            # Annualized return
            total_days = len(equity_curve)
            annual_factor = 252 / total_days if total_days > 0 else 1
            metrics["annualized_return"] = ((equity[-1] / equity[0]) ** annual_factor - 1) * 100

            # Volatility
            metrics["annualized_volatility"] = np.std(returns) * np.sqrt(252) * 100

            # Sharpe Ratio
            excess_returns = returns - (risk_free_rate / 252)
            if np.std(returns) > 0:
                metrics["sharpe_ratio"] = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
            else:
                metrics["sharpe_ratio"] = 0

            # Sortino Ratio
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and np.std(downside_returns) > 0:
                metrics["sortino_ratio"] = (
                    np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
                )
            else:
                metrics["sortino_ratio"] = 0

            # Calmar Ratio
            if "max_drawdown_pct" in metrics and metrics["max_drawdown_pct"] != 0:
                metrics["calmar_ratio"] = metrics["annualized_return"] / abs(
                    metrics["max_drawdown_pct"]
                )
            else:
                metrics["calmar_ratio"] = 0

        return metrics

    @staticmethod
    def _calculate_drawdown_metrics(
        trades: Union[List[Dict[str, Any]], pd.DataFrame], initial_equity: float
    ) -> Dict[str, Any]:
        """Calculate drawdown metrics from trades."""
        metrics: Dict[str, Any] = {}  # type: ignore[assignment]

        # Handle empty trades
        if trades is None or len(trades) == 0:
            metrics["max_drawdown"] = 0
            metrics["max_drawdown_pct"] = 0
            metrics["avg_drawdown"] = 0
            metrics["avg_drawdown_duration"] = 0
            return metrics

        # Convert to DataFrame if needed
        if isinstance(trades, pd.DataFrame):
            trades_df = trades
        else:
            trades_df = pd.DataFrame(trades)

        # Build equity curve from trades
        equity = initial_equity
        equity_values = [initial_equity]
        peak = initial_equity
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        drawdowns = []

        # Get pnl values
        if "pnl" in trades_df.columns:
            pnl_values = trades_df["pnl"].values
        else:
            pnl_values = []

        for pnl in pnl_values:
            pnl_val = pnl if pd.notna(pnl) else 0
            equity += pnl_val
            equity_values.append(equity)

            if equity > peak:
                peak = equity

            drawdown = peak - equity
            drawdown_pct = (drawdown / peak * 100) if peak > 0 else 0

            drawdowns.append(drawdown)

            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct

        metrics["max_drawdown"] = max_drawdown
        metrics["max_drawdown_pct"] = max_drawdown_pct
        metrics["avg_drawdown"] = np.mean(drawdowns) if drawdowns else 0

        # Calculate average drawdown duration (simplified)
        in_drawdown = False
        drawdown_start = 0
        drawdown_durations = []

        for i, (eq, dd) in enumerate(zip(equity_values, drawdowns + [0])):
            if dd > 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
            elif dd == 0 and in_drawdown:
                in_drawdown = False
                drawdown_durations.append(i - drawdown_start)

        metrics["avg_drawdown_duration"] = np.mean(drawdown_durations) if drawdown_durations else 0

        return metrics

    @staticmethod
    def _calculate_streak_metrics(
        trades: Union[List[Dict[str, Any]], pd.DataFrame],
    ) -> Dict[str, Any]:
        """Calculate winning/losing streak metrics."""
        metrics: Dict[str, Any] = {}  # type: ignore[assignment]

        if trades is None or len(trades) == 0:
            return metrics

        # Convert to DataFrame if needed
        if isinstance(trades, pd.DataFrame):
            trades_df = trades
        else:
            trades_df = pd.DataFrame(trades)

        # Calculate streaks
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        current_type = None

        # Get pnl values
        if "pnl" in trades_df.columns:
            pnl_values = trades_df["pnl"].values
        else:
            return metrics

        for pnl in pnl_values:
            pnl_val = pnl if pd.notna(pnl) else 0

            if pnl_val > 0:
                if current_type == "win":
                    current_streak += 1
                else:
                    current_type = "win"
                    current_streak = 1
                max_win_streak = max(max_win_streak, current_streak)
            elif pnl_val < 0:
                if current_type == "loss":
                    current_streak += 1
                else:
                    current_type = "loss"
                    current_streak = 1
                max_loss_streak = max(max_loss_streak, current_streak)
            else:
                current_streak = 0
                current_type = None

        metrics["max_win_streak"] = max_win_streak
        metrics["max_loss_streak"] = max_loss_streak

        return metrics

    @staticmethod
    def _calculate_holding_metrics(
        trades: Union[List[Dict[str, Any]], pd.DataFrame],
    ) -> Dict[str, Any]:
        """Calculate holding period metrics."""
        metrics: Dict[str, Any] = {}  # type: ignore[assignment]

        holding_periods = []

        # Handle empty trades
        if trades is None or len(trades) == 0:
            return metrics

        # Convert to DataFrame if needed
        if isinstance(trades, pd.DataFrame):
            trades_df = trades
        else:
            trades_df = pd.DataFrame(trades)

        # Check if required columns exist
        if "entry_time" not in trades_df.columns or "exit_time" not in trades_df.columns:
            return metrics

        for _, trade in trades_df.iterrows():
            entry_time = trade.get("entry_time")
            exit_time = trade.get("exit_time")

            if entry_time and exit_time and pd.notna(entry_time) and pd.notna(exit_time):
                try:
                    if isinstance(entry_time, str):
                        entry = pd.to_datetime(entry_time)
                    else:
                        entry = entry_time

                    if isinstance(exit_time, str):
                        exit = pd.to_datetime(exit_time)
                    else:
                        exit = exit_time

                    holding = (exit - entry).total_seconds() / 3600  # hours
                    holding_periods.append(holding)
                except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
                    pass

        if holding_periods:
            metrics["avg_holding_hours"] = np.mean(holding_periods)
            metrics["min_holding_hours"] = min(holding_periods)
            metrics["max_holding_hours"] = max(holding_periods)

        return metrics

    @staticmethod
    def calculate_monthly_returns(
        trades: List[Dict[str, Any]], initial_equity: float = 100000.0
    ) -> pd.DataFrame:
        """
        Calculate monthly returns from trades.

        Args:
            trades: List of trade dictionaries
            initial_equity: Starting equity

        Returns:
            DataFrame with monthly returns
        """
        if not trades:
            return pd.DataFrame()

        # Build monthly P&L
        monthly_pnl: Dict[str, float] = {}
        equity = initial_equity

        for trade in trades:
            exit_time = trade.get("exit_time")
            pnl = trade.get("pnl", 0)

            if exit_time:
                try:
                    if isinstance(exit_time, str):
                        dt = pd.to_datetime(exit_time)
                    else:
                        dt = exit_time

                    month_key = dt.strftime("%Y-%m")
                    monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + pnl
                except (ValueError, TypeError, AttributeError):
                    pass

        if not monthly_pnl:
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame([{"month": k, "pnl": v} for k, v in monthly_pnl.items()])
        df = df.sort_values("month")
        df["return_pct"] = (df["pnl"] / initial_equity) * 100

        return df

    @staticmethod
    def calculate_pattern_statistics(trades: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Calculate statistics broken down by pattern.

        Args:
            trades: List of trade dictionaries

        Returns:
            DataFrame with pattern statistics
        """
        if not trades:
            return pd.DataFrame()

        pattern_stats: Dict[str, Any] = {}  # type: ignore[assignment]

        for trade in trades:
            pattern = str(trade.get("pattern", "Unknown"))

            if pattern not in pattern_stats:
                pattern_stats[pattern] = {
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0.0,
                    "pnl_list": [],
                }

            pattern_stats[pattern]["trades"] += 1
            pnl = float(trade.get("pnl", 0))
            pattern_stats[pattern]["total_pnl"] += pnl
            pattern_stats[pattern]["pnl_list"].append(pnl)

            if pnl > 0:
                pattern_stats[pattern]["wins"] += 1
            elif pnl < 0:
                pattern_stats[pattern]["losses"] += 1

        # Calculate metrics for each pattern
        rows = []
        for pattern, stats in pattern_stats.items():
            win_rate = float(stats["wins"]) / float(stats["trades"]) if stats["trades"] > 0 else 0.0
            avg_pnl = (
                float(stats["total_pnl"]) / float(stats["trades"]) if stats["trades"] > 0 else 0.0
            )

            pnl_list: list = stats.get("pnl_list", [])  # type: ignore[assignment]
            wins = [p for p in pnl_list if p > 0]
            losses = [abs(p) for p in pnl_list if p < 0]

            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0

            profit_factor = sum(wins) / sum(losses) if sum(losses) > 0 else float("inf")

            rows.append(
                {
                    "pattern": pattern,
                    "trades": stats["trades"],
                    "wins": stats["wins"],
                    "losses": stats["losses"],
                    "win_rate": win_rate,
                    "total_pnl": stats["total_pnl"],
                    "avg_pnl": avg_pnl,
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "profit_factor": profit_factor,
                }
            )

        return pd.DataFrame(rows).sort_values("total_pnl", ascending=False)

    @staticmethod
    def format_report(metrics: Dict[str, Any]) -> str:
        """
        Format metrics as a readable report.

        Args:
            metrics: Dictionary of performance metrics

        Returns:
            Formatted string report
        """
        lines = []
        lines.append("=" * 60)
        lines.append("PERFORMANCE REPORT")
        lines.append("=" * 60)

        # Trade Statistics
        lines.append("\n--- Trade Statistics ---")
        lines.append(f"Total Trades: {metrics.get('total_trades', 0)}")
        lines.append(f"Winning Trades: {metrics.get('winning_trades', 0)}")
        lines.append(f"Losing Trades: {metrics.get('losing_trades', 0)}")
        lines.append(f"Win Rate: {metrics.get('win_rate', 0):.2%}")

        # Profit/Loss
        lines.append("\n--- Profit/Loss ---")
        lines.append(f"Initial Equity: ${metrics.get('initial_equity', 0):,.2f}")
        lines.append(f"Final Equity: ${metrics.get('final_equity', 0):,.2f}")
        lines.append(f"Total P&L: ${metrics.get('total_pnl', 0):,.2f}")
        lines.append(f"Total Return: {metrics.get('total_return', 0):.2f}%")
        lines.append(f"Average Win: ${metrics.get('avg_win', 0):,.2f}")
        lines.append(f"Average Loss: ${metrics.get('avg_loss', 0):,.2f}")
        lines.append(f"Largest Win: ${metrics.get('largest_win', 0):,.2f}")
        lines.append(f"Largest Loss: ${metrics.get('largest_loss', 0):,.2f}")

        # Risk Metrics
        lines.append("\n--- Risk Metrics ---")
        lines.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        lines.append(f"Expectancy: ${metrics.get('expectancy', 0):,.2f}")
        lines.append(f"Risk/Reward Ratio: {metrics.get('risk_reward_ratio', 0):.2f}")
        lines.append(f"Max Drawdown: ${metrics.get('max_drawdown', 0):,.2f}")
        lines.append(f"Max Drawdown %: {metrics.get('max_drawdown_pct', 0):.2f}%")

        # Risk-Adjusted Returns
        if "sharpe_ratio" in metrics:
            lines.append("\n--- Risk-Adjusted Returns ---")
            lines.append(f"Annualized Return: {metrics.get('annualized_return', 0):.2f}%")
            lines.append(f"Annualized Volatility: {metrics.get('annualized_volatility', 0):.2f}%")
            lines.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            lines.append(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
            lines.append(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")

        # Streak Statistics
        lines.append("\n--- Streak Statistics ---")
        lines.append(f"Max Win Streak: {metrics.get('max_win_streak', 0)}")
        lines.append(f"Max Loss Streak: {metrics.get('max_loss_streak', 0)}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)
