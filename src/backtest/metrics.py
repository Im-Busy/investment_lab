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

    Calculates and stores comprehensive performance metrics covering:
    - Capital & equity overview
    - Return metrics (total, annualized simple/CAGR, monthly)
    - Risk & ratios (Sharpe, Sortino, equity volatility)
    - Drawdown analysis (max, date, duration)
    - Trade statistics (win rate, profit factor, avg per trade)
    - Streaks & duration (winning/losing days, profit/loss increments)
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
            trades: List of trade dictionaries or DataFrame
            equity_curve: DataFrame with equity over time (daily bars)
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
            metrics["net_profit"] = total_pnl
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

            # Average profit per trade
            metrics["avg_profit_per_trade"] = total_pnl / len(trades) if len(trades) > 0 else 0
        else:
            total_pnl = 0
            metrics["winning_trades"] = 0
            metrics["losing_trades"] = 0
            metrics["breakeven_trades"] = 0
            metrics["win_rate"] = 0
            metrics["total_pnl"] = 0
            metrics["net_profit"] = 0
            metrics["total_return"] = 0
            metrics["avg_win"] = 0
            metrics["avg_loss"] = 0
            metrics["largest_win"] = 0
            metrics["largest_loss"] = 0
            metrics["profit_factor"] = 0
            metrics["avg_profit_per_trade"] = 0

        # Expectancy
        expectancy = (metrics["win_rate"] * metrics["avg_win"]) - (
            (1 - metrics["win_rate"]) * metrics["avg_loss"]
        )
        metrics["expectancy"] = expectancy

        # Risk/Reward ratio
        metrics["risk_reward_ratio"] = (
            metrics["avg_win"] / metrics["avg_loss"] if metrics["avg_loss"] > 0 else 0
        )

        # Final equity
        metrics["initial_equity"] = initial_equity
        metrics["final_equity"] = initial_equity + total_pnl

        # Build equity from trades
        trade_equity_values = PerformanceMetrics._build_equity_from_trades(
            trades_df, initial_equity
        )
        metrics["max_equity"] = max(trade_equity_values) if trade_equity_values else initial_equity
        metrics["min_equity"] = min(trade_equity_values) if trade_equity_values else initial_equity

        # Max capital used ratio (peak equity / initial)
        if initial_equity > 0:
            metrics["max_capital_used"] = metrics["max_equity"]
            metrics["max_capital_used_ratio"] = (metrics["max_equity"] / initial_equity) * 100
        else:
            metrics["max_capital_used"] = 0
            metrics["max_capital_used_ratio"] = 0

        # Capital risk rate (max drawdown / initial capital)
        # calculated after drawdown metrics

        # Calculate equity curve metrics if available
        if equity_curve is not None and len(equity_curve) > 0:
            metrics.update(
                PerformanceMetrics._calculate_equity_metrics(
                    equity_curve, initial_equity, risk_free_rate
                )
            )
            # Calculate equity max/min from equity curve if available (more accurate)
            if "equity" in equity_curve.columns:
                eq = equity_curve["equity"].values
                metrics["max_equity"] = float(np.max(eq))
                metrics["min_equity"] = float(np.min(eq))
                metrics["max_capital_used"] = float(np.max(eq))
                if initial_equity > 0:
                    metrics["max_capital_used_ratio"] = (float(np.max(eq)) / initial_equity) * 100

                # Calculate drawdowns from equity curve (captures intra-trade drawdowns)
                dd_metrics = PerformanceMetrics._calculate_drawdowns_from_equity(
                    equity_curve, trades_df, initial_equity
                )
                metrics.update(dd_metrics)

            # Calculate daily streak metrics from equity curve
            metrics.update(PerformanceMetrics._calculate_daily_streaks(equity_curve))

        # Fallback: calculate drawdown from trades if no equity curve
        if equity_curve is None or len(equity_curve) == 0:
            metrics.update(
                PerformanceMetrics._calculate_drawdown_metrics(trades_df, initial_equity)
            )

        # Capital risk rate (max drawdown amount / initial capital)
        if initial_equity > 0:
            metrics["capital_risk_rate"] = (metrics.get("max_drawdown", 0) / initial_equity) * 100
        else:
            metrics["capital_risk_rate"] = 0

        # Calmar Ratio (post-drawdown)
        cagr = metrics.get("annualized_compound_return", 0)
        dd_pct = metrics.get("max_drawdown_pct", 0)
        if dd_pct != 0:
            metrics["calmar_ratio"] = cagr / abs(dd_pct)
        else:
            metrics["calmar_ratio"] = 0 if dd_pct == 0 else float("inf")

        # Annualized return/risk ratio
        ann_ret = metrics.get("annualized_compound_return", 0)
        ann_vol = metrics.get("annualized_volatility", 0)
        if ann_vol and ann_vol > 0:
            metrics["annualized_return_risk_ratio"] = ann_ret / ann_vol
        else:
            metrics["annualized_return_risk_ratio"] = 0

        # Calculate streak metrics (trade-based)
        metrics.update(PerformanceMetrics._calculate_streak_metrics(trades_df))

        # Calculate holding period metrics
        metrics.update(PerformanceMetrics._calculate_holding_metrics(trades_df))

        return metrics

    @staticmethod
    def _build_equity_from_trades(trades_df: pd.DataFrame, initial_equity: float) -> List[float]:
        """Build equity values from trade PnL."""
        equity = initial_equity
        values = [initial_equity]
        if "pnl" in trades_df.columns:
            for pnl in trades_df["pnl"].values:
                val = pnl if pd.notna(pnl) else 0
                equity += val
                values.append(equity)
        return values

    @staticmethod
    def _calculate_equity_metrics(
        equity_curve: pd.DataFrame, initial_equity: float, risk_free_rate: float
    ) -> Dict[str, Any]:
        """Calculate equity curve based metrics including returns, risk, and ratios."""
        metrics: Dict[str, Any] = {}  # type: ignore[assignment]

        if "equity" not in equity_curve.columns:
            return metrics

        equity = np.asarray(equity_curve["equity"].values)

        # Daily returns
        returns = np.diff(equity) / equity[:-1]  # type: ignore[arg-type]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

        if len(returns) == 0:
            return metrics

        total_days = len(equity_curve)
        years = total_days / 252 if total_days > 0 else 1
        start_eq = equity[0] if equity[0] > 0 else 1
        end_eq = equity[-1]

        # ---- Return Metrics ----
        # Annualized compound return (CAGR)
        if years > 0 and start_eq > 0:
            cagr = ((end_eq / start_eq) ** (1 / years) - 1) * 100
        else:
            cagr = 0
        metrics["annualized_compound_return"] = cagr
        # Keep backward compat key
        metrics["annualized_return"] = cagr

        # Annualized simple return (total return / years)
        total_return_pct = ((end_eq - start_eq) / start_eq) * 100
        metrics["annualized_simple_return"] = total_return_pct / years if years > 0 else 0

        # Cumulative return
        metrics["cumulative_return"] = total_return_pct
        metrics["total_return_pct"] = total_return_pct

        # Monthly returns
        if isinstance(equity_curve.index, pd.DatetimeIndex) and len(equity_curve) > 1:
            monthly = PerformanceMetrics._calculate_monthly_metrics(
                equity_curve, returns, total_days, years
            )
            metrics.update(monthly)

        # ---- Volatility & Risk ----
        metrics["annualized_volatility"] = np.std(returns) * np.sqrt(252) * 100

        # Equity dispersion (standard deviation of equity values, as percentage of mean)
        equity_mean = np.mean(equity)
        if equity_mean > 0:
            metrics["equity_dispersion"] = (np.std(equity) / equity_mean) * 100
        else:
            metrics["equity_dispersion"] = 0

        # ---- Sharpe Ratio ----
        excess_returns = returns - (risk_free_rate / 252)
        if np.std(returns) > 0:
            metrics["sharpe_ratio"] = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
        else:
            metrics["sharpe_ratio"] = 0

        # ---- Sortino Ratio ----
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            metrics["sortino_ratio"] = (
                np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
            )
        else:
            metrics["sortino_ratio"] = 0

        return metrics

    @staticmethod
    def _calculate_monthly_metrics(
        equity_curve: pd.DataFrame, returns: np.ndarray, total_days: int, years: float
    ) -> Dict[str, Any]:
        """Calculate monthly return metrics from equity curve."""
        metrics: Dict[str, Any] = {}  # type: ignore[assignment]

        try:
            # Resample equity to monthly and compute monthly returns
            monthly_eq = equity_curve["equity"].resample("ME").last()
            if len(monthly_eq) < 2:
                return metrics

            monthly_returns = monthly_eq.pct_change().dropna()
            if len(monthly_returns) == 0:
                return metrics

            # Average monthly simple return (arithmetic mean)
            metrics["monthly_simple_return"] = monthly_returns.mean() * 100

            # Monthly compound return (geometric mean equivalent)
            if len(monthly_returns) > 0:
                full_return = monthly_eq.iloc[-1] / monthly_eq.iloc[0]
                metrics["monthly_compound_return"] = (
                    full_return ** (1 / len(monthly_returns)) - 1
                ) * 100
            else:
                metrics["monthly_compound_return"] = 0

            # Count profitable months
            profitable_months = (monthly_returns > 0).sum()
            total_months = len(monthly_returns)
            metrics["profitable_months"] = int(profitable_months)
            metrics["total_months"] = total_months

        except (ValueError, TypeError, AttributeError):
            pass

        return metrics

    @staticmethod
    def _calculate_drawdowns_from_equity(
        equity_curve: pd.DataFrame,
        trades_df: pd.DataFrame,
        initial_equity: float,
    ) -> Dict[str, Any]:
        """Calculate drawdown metrics from equity curve (captures intra-trade drawdowns)."""
        metrics: Dict[str, Any] = {}  # type: ignore[assignment]

        if "equity" not in equity_curve.columns or len(equity_curve) < 2:
            return PerformanceMetrics._calculate_drawdown_metrics(trades_df, initial_equity)

        equity = equity_curve["equity"].values
        dates = equity_curve.index if isinstance(equity_curve.index, pd.DatetimeIndex) else None

        # Running peak and drawdown
        peak = float(equity[0])
        max_dd = 0.0
        max_dd_pct = 0.0
        max_dd_date = None
        max_dd_peak_idx = 0
        max_dd_trough_idx = 0
        peak_idx = 0
        drawdowns = []

        for i, eq in enumerate(equity):
            if eq > peak:
                peak = float(eq)
                peak_idx = i
            dd = peak - eq
            dd_pct = (dd / peak * 100) if peak > 0 else 0
            drawdowns.append(dd)

            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
                max_dd_peak_idx = peak_idx
                max_dd_trough_idx = i

        metrics["max_drawdown"] = max_dd
        metrics["max_drawdown_pct"] = max_dd_pct
        metrics["avg_drawdown"] = float(np.mean(drawdowns)) if drawdowns else 0.0

        # Max drawdown date
        if dates is not None and max_dd_trough_idx < len(dates):
            metrics["max_drawdown_date"] = dates[max_dd_trough_idx].strftime("%Y-%m-%d")

        # Max drawdown duration (bars from peak to recovery)
        max_duration = 0
        in_dd = False
        dd_start = 0
        dd_peak_val = float(equity[0])
        for i, eq in enumerate(equity):
            if not in_dd and eq < dd_peak_val:
                in_dd = True
                dd_start = i
            elif in_dd and eq >= dd_peak_val:
                in_dd = False
                duration = i - dd_start
                max_duration = max(max_duration, duration)
                dd_peak_val = float(eq)
            if eq > dd_peak_val:
                dd_peak_val = float(eq)

        metrics["max_drawdown_duration_days"] = max_duration

        # Number of drawdown episodes
        num_episodes = 0
        in_dd = False
        running_peak = float(equity[0])
        for i, eq in enumerate(equity):
            if eq > running_peak:
                running_peak = float(eq)
            if not in_dd and eq < running_peak:
                in_dd = True
                num_episodes += 1
            elif in_dd and eq >= running_peak:
                in_dd = False
        metrics["num_drawdown_episodes"] = num_episodes

        # Max P/L drawdown (from trades)
        pl_dd, pl_dd_pct = PerformanceMetrics._calc_pl_drawdown(trades_df, initial_equity)
        metrics["max_pl_drawdown"] = pl_dd
        metrics["max_pl_drawdown_pct"] = pl_dd_pct

        # Average drawdown duration (trades-based)
        avg_dd = PerformanceMetrics._calculate_drawdown_metrics(trades_df, initial_equity)
        metrics["avg_drawdown_duration"] = avg_dd.get("avg_drawdown_duration", 0)

        return metrics

    @staticmethod
    def _calculate_drawdown_metrics(
        trades: Union[List[Dict[str, Any]], pd.DataFrame], initial_equity: float
    ) -> Dict[str, Any]:
        """Calculate drawdown metrics from trades including date and duration."""
        metrics: Dict[str, Any] = {}  # type: ignore[assignment]

        # Handle empty trades
        if trades is None or len(trades) == 0:
            metrics["max_drawdown"] = 0
            metrics["max_drawdown_pct"] = 0
            metrics["avg_drawdown"] = 0
            metrics["avg_drawdown_duration"] = 0
            metrics["max_drawdown_date"] = None
            metrics["max_drawdown_duration_days"] = 0
            metrics["max_pl_drawdown"] = 0
            metrics["max_pl_drawdown_pct"] = 0
            return metrics

        # Convert to DataFrame if needed
        if isinstance(trades, pd.DataFrame):
            trades_df = trades
        else:
            trades_df = pd.DataFrame(trades)

        # Build equity curve from trades with timestamps
        equity = initial_equity
        equity_values = [initial_equity]
        peak = initial_equity
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        max_dd_idx = 0
        drawdowns = []
        peak_values = []

        # Get pnl values and exit times
        if "pnl" in trades_df.columns:
            pnl_values = trades_df["pnl"].values
        else:
            pnl_values = []

        for i, pnl in enumerate(pnl_values):
            pnl_val = pnl if pd.notna(pnl) else 0
            equity += pnl_val
            equity_values.append(equity)

            if equity > peak:
                peak = equity

            peak_values.append(peak)

            drawdown = peak - equity
            drawdown_pct = (drawdown / peak * 100) if peak > 0 else 0

            drawdowns.append(drawdown)

            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
                max_dd_idx = i

        metrics["max_drawdown"] = max_drawdown
        metrics["max_drawdown_pct"] = max_drawdown_pct
        metrics["avg_drawdown"] = np.mean(drawdowns) if drawdowns else 0

        # Max drawdown date (from exit_time of the trade at the drawdown trough)
        metrics["max_drawdown_date"] = None
        if max_dd_idx < len(trades_df) and "exit_time" in trades_df.columns:
            try:
                exit_t = trades_df.iloc[max_dd_idx].get("exit_time")
                if exit_t and pd.notna(exit_t):
                    if hasattr(exit_t, "strftime"):
                        metrics["max_drawdown_date"] = exit_t.strftime("%Y-%m-%d")
                    elif hasattr(exit_t, "date"):
                        metrics["max_drawdown_date"] = exit_t.date().isoformat()
                    else:
                        metrics["max_drawdown_date"] = str(exit_t)[:10]
            except (IndexError, KeyError, AttributeError):
                pass

        # Max drawdown duration (max days from peak to trough, plus recovery to new peak)
        metrics["max_drawdown_duration_days"] = PerformanceMetrics._calc_max_dd_duration(
            equity_values, peak_values, drawdowns
        )

        # Trade count per drawdown (average)
        dd_count = 0
        in_dd = False
        for dd_val in drawdowns:
            if dd_val > 0 and not in_dd:
                in_dd = True
                dd_count += 1
            elif dd_val == 0:
                in_dd = False
        metrics["num_drawdown_episodes"] = dd_count

        # Calculate average drawdown duration (simplified)
        in_drawdown = False
        drawdown_start = 0
        drawdown_durations: list[int] = []

        for i, (eq, dd) in enumerate(zip(equity_values, drawdowns + [0])):
            if dd > 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
            elif dd == 0 and in_drawdown:
                in_drawdown = False
                drawdown_durations.append(i - drawdown_start)

        metrics["avg_drawdown_duration"] = (
            float(np.mean(drawdown_durations)) if drawdown_durations else 0.0
        )

        # Max P/L drawdown (largest cumulative loss in a sequence of consecutive losing trades)
        pl_dd, pl_dd_pct = PerformanceMetrics._calc_pl_drawdown(trades_df, initial_equity)
        metrics["max_pl_drawdown"] = pl_dd
        metrics["max_pl_drawdown_pct"] = pl_dd_pct

        return metrics

    @staticmethod
    def _calc_max_dd_duration(
        equity_values: List[float], peak_values: List[float], drawdowns: List[float]
    ) -> int:
        """Calculate maximum drawdown duration in trade-count units (time from peak to recovery)."""
        max_duration = 0
        current_duration = 0
        in_dd = False

        for i, dd in enumerate(drawdowns):
            if dd > 0:
                if not in_dd:
                    in_dd = True
                    current_duration = 1
                else:
                    current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                in_dd = False
                current_duration = 0

        return max_duration

    @staticmethod
    def _calc_pl_drawdown(trades_df: pd.DataFrame, initial_equity: float) -> tuple:
        """Calculate maximum P/L drawdown from consecutive losing trades."""
        max_pl_dd = 0.0
        cumulative_loss = 0.0

        if "pnl" not in trades_df.columns:
            return 0.0, 0.0

        for pnl in trades_df["pnl"].values:
            val = pnl if pd.notna(pnl) else 0
            if val < 0:
                cumulative_loss += val
                max_pl_dd = min(max_pl_dd, cumulative_loss)
            else:
                cumulative_loss = 0

        max_pl_dd_abs = abs(max_pl_dd)
        max_pl_dd_pct = (max_pl_dd_abs / initial_equity * 100) if initial_equity > 0 else 0

        return max_pl_dd_abs, max_pl_dd_pct

    @staticmethod
    def _calculate_daily_streaks(
        equity_curve: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Calculate daily win/loss and profit/loss increment streaks from equity curve."""
        metrics: Dict[str, Any] = {}  # type: ignore[assignment]

        if "equity" not in equity_curve.columns or len(equity_curve) < 2:
            return metrics

        equity = equity_curve["equity"].values
        daily_pnl = np.diff(equity)

        # Consecutive winning/losing days
        max_win_days = 0
        max_loss_days = 0
        curr_win = 0
        curr_loss = 0

        for pnl in daily_pnl:
            if pnl > 0:
                curr_win += 1
                curr_loss = 0
                max_win_days = max(max_win_days, curr_win)
            elif pnl < 0:
                curr_loss += 1
                curr_win = 0
                max_loss_days = max(max_loss_days, curr_loss)

        metrics["max_consecutive_winning_days"] = max_win_days
        metrics["max_consecutive_losing_days"] = max_loss_days

        # Consecutive profit/loss increments (each day's PnL greater than previous)
        max_profit_inc = 0
        max_loss_inc = 0
        curr_profit_inc = 0
        curr_loss_inc = 0

        for i in range(1, len(daily_pnl)):
            if daily_pnl[i] > daily_pnl[i - 1] > 0:
                curr_profit_inc += 1
                max_profit_inc = max(max_profit_inc, curr_profit_inc)
            else:
                curr_profit_inc = 0

            if daily_pnl[i] < daily_pnl[i - 1] < 0:
                curr_loss_inc += 1
                max_loss_inc = max(max_loss_inc, curr_loss_inc)
            else:
                curr_loss_inc = 0

        metrics["max_consecutive_profit_increments"] = max_profit_inc
        metrics["max_consecutive_loss_increments"] = max_loss_inc

        return metrics

    @staticmethod
    def _calculate_streak_metrics(
        trades: Union[List[Dict[str, Any]], pd.DataFrame],
    ) -> Dict[str, Any]:
        """Calculate winning/losing streak metrics from trades."""
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
        metrics["current_streak"] = current_streak
        metrics["current_streak_type"] = current_type

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
    def from_backtesting_stats(
        stats: Dict[str, Any],
        initial_equity: float = 100000.0,
        risk_free_rate: float = 0.02,
    ) -> Dict[str, Any]:
        """
        Convert backtesting.py stats dict to comprehensive PerformanceMetrics dict.

        Extracts trades and equity curve from backtesting.py Results object keys
        (_trades, _equity_curve) and runs the full PerformanceMetrics calculation.

        Args:
            stats: Dictionary from backtesting.py Results (must contain _trades,
                   _equity_curve, or the standard backtesting.py metrics)
            initial_equity: Starting capital
            risk_free_rate: Annual risk-free rate

        Returns:
            Dictionary with comprehensive performance metrics
        """
        raw_trades = stats.get("_trades", pd.DataFrame())
        equity_curve = stats.get("_equity_curve", pd.DataFrame())

        # Convert trades from backtesting.py format to our format
        trades_list: List[Dict[str, Any]] = []
        if isinstance(raw_trades, list):
            for t in raw_trades:
                if hasattr(t, "_asdict"):
                    trades_list.append(t._asdict())
                elif isinstance(t, dict):
                    trades_list.append(t)
        elif isinstance(raw_trades, pd.DataFrame) and not raw_trades.empty:
            trades_list = raw_trades.to_dict("records")

        # Normalize trade keys to 'pnl' and 'exit_time'
        normalized_trades = []
        for t in trades_list:
            nt = dict(t)
            if "pnl" not in nt:
                if "PnL" in nt:
                    nt["pnl"] = float(nt["PnL"])
                elif "ReturnPct" in nt and "Size" in nt:
                    entry_price = float(nt.get("EntryPrice", 0))
                    size = float(nt.get("Size", 0))
                    if entry_price > 0:
                        nt["pnl"] = float(nt["ReturnPct"]) * 0.01 * entry_price * abs(size)
                    else:
                        nt["pnl"] = 0.0
                else:
                    nt["pnl"] = 0.0
            if "exit_time" not in nt:
                if "ExitTime" in nt:
                    nt["exit_time"] = nt["ExitTime"]
                elif "ExitDate" in nt:
                    nt["exit_time"] = nt["ExitDate"]
            normalized_trades.append(nt)

        # If no trades from backtesting, derive from stats
        if not normalized_trades:
            total_trades = int(stats.get("# Trades", 0))
            if total_trades > 0:
                # Create synthetic minimal trades list from stats alone
                win_rate = float(stats.get("Win Rate [%]", 0)) / 100.0
                avg_trade_pct = float(stats.get("Avg. Trade [%]", 0))
                profit_factor = float(stats.get("Profit Factor", 0))

                num_wins = int(round(total_trades * win_rate))
                num_losses = total_trades - num_wins

                avg_trade_dollar = avg_trade_pct * 0.01 * initial_equity
                total_pnl_est = avg_trade_dollar * total_trades

                if profit_factor > 0 and num_losses > 0:
                    avg_loss = (
                        abs(total_pnl_est) / (profit_factor * num_losses - num_losses)
                        if profit_factor != float("inf")
                        else 0
                    )
                    avg_win = avg_loss * profit_factor if avg_loss > 0 else avg_trade_dollar * 2
                else:
                    avg_win = avg_trade_dollar * 1.5 if avg_trade_dollar > 0 else 50
                    avg_loss = abs(avg_trade_dollar) * 1.5 if avg_trade_dollar < 0 else 50

                for _ in range(num_wins):
                    normalized_trades.append({"pnl": float(avg_win)})
                for _ in range(num_losses):
                    normalized_trades.append({"pnl": -float(avg_loss)})

        if not normalized_trades:
            return {
                "total_trades": 0,
                "initial_equity": initial_equity,
                "final_equity": initial_equity,
                "total_return": 0.0,
                "message": "No trades to analyze",
            }

        # Prepare equity curve as DataFrame with "equity" column
        eq_df: Optional[pd.DataFrame] = None
        if isinstance(equity_curve, pd.DataFrame) and not equity_curve.empty:
            eq_df = equity_curve.copy()
            if "equity" not in eq_df.columns and "Equity" in eq_df.columns:
                eq_df["equity"] = eq_df["Equity"]
            # Ensure datetime index
            if not isinstance(eq_df.index, pd.DatetimeIndex):
                try:
                    eq_df.index = pd.to_datetime(eq_df.index)
                except (ValueError, TypeError):
                    pass
        elif isinstance(equity_curve, pd.Series) and not equity_curve.empty:
            eq_df = pd.DataFrame({"equity": equity_curve.values}, index=equity_curve.index)

        # If no equity curve from backtesting, build one from trades
        if eq_df is None and normalized_trades:
            equity_vals = [initial_equity]
            for t in normalized_trades:
                equity_vals.append(equity_vals[-1] + t.get("pnl", 0))
            exit_times = [t.get("exit_time") for t in normalized_trades if t.get("exit_time")]
            if exit_times:
                eq_df = pd.DataFrame({"equity": equity_vals[1:]}, index=pd.to_datetime(exit_times))
            else:
                eq_df = pd.DataFrame({"equity": equity_vals[1:]})

        return PerformanceMetrics.calculate(
            trades=normalized_trades,
            equity_curve=eq_df,
            initial_equity=initial_equity,
            risk_free_rate=risk_free_rate,
        )

    @staticmethod
    def format_report(metrics: Dict[str, Any]) -> str:
        """
        Format metrics as a comprehensive professional report.

        Args:
            metrics: Dictionary of performance metrics

        Returns:
            Formatted string report
        """
        lines: list[str] = []
        W = 62  # report width

        lines.append("=" * W)
        lines.append("PERFORMANCE REPORT")
        lines.append("=" * W)

        # -- 1. Capital and Equity Overview --
        lines.append("\n" + "-" * W)
        lines.append("CAPITAL & EQUITY OVERVIEW")
        lines.append("-" * W)
        init_eq = metrics.get("initial_equity", 0)
        final_eq = metrics.get("final_equity", 0)
        max_eq = metrics.get("max_equity", 0)
        min_eq = metrics.get("min_equity", 0)
        net_profit = metrics.get("net_profit", metrics.get("total_pnl", 0))

        lines.append(f"  Initial Capital:        ${init_eq:>15,.2f}")
        lines.append(
            f"  Max Capital Used:        ${metrics.get('max_capital_used', max_eq):>15,.2f}"
        )
        lines.append(
            f"  Max Capital Used Ratio:   {metrics.get('max_capital_used_ratio', 0):>14.2f}%"
        )
        lines.append(f"  Final Equity:            ${final_eq:>15,.2f}")
        lines.append(f"  Net Profit:              ${net_profit:>15,.2f}")
        lines.append(f"  Max Equity (Peak):       ${max_eq:>15,.2f}")
        lines.append(f"  Min Equity (Trough):     ${min_eq:>15,.2f}")

        # -- 2. Return Metrics --
        lines.append("\n" + "-" * W)
        lines.append("RETURN METRICS")
        lines.append("-" * W)
        lines.append(
            f"  Total Return:             {metrics.get('total_return_pct', metrics.get('total_return', 0)):>14.2f}%"
        )
        lines.append(
            f"  Cumulative Return:        {metrics.get('cumulative_return', metrics.get('total_return', 0)):>14.2f}%"
        )
        lines.append(
            f"  Annualized Simple Return: {metrics.get('annualized_simple_return', 0):>14.2f}%"
        )
        lines.append(
            f"  Annualized Compound (CAGR):{metrics.get('annualized_compound_return', 0):>13.2f}%"
        )

        monthly_simple = metrics.get("monthly_simple_return")
        monthly_compound = metrics.get("monthly_compound_return")
        if monthly_simple is not None:
            lines.append(f"  Monthly Simple Return:    {monthly_simple:>14.2f}%")
        if monthly_compound is not None:
            lines.append(f"  Monthly Compound Return:  {monthly_compound:>14.2f}%")

        # -- 3. Risk and Ratios --
        lines.append("\n" + "-" * W)
        lines.append("RISK & RATIOS")
        lines.append("-" * W)
        lines.append(f"  Sharpe Ratio:             {metrics.get('sharpe_ratio', 0):>14.2f}")
        lines.append(f"  Sortino Ratio:            {metrics.get('sortino_ratio', 0):>14.2f}")
        lines.append(f"  Calmar Ratio:             {metrics.get('calmar_ratio', 0):>14.2f}")
        lines.append(f"  Equity Dispersion (CV):   {metrics.get('equity_dispersion', 0):>14.2f}%")
        lines.append(
            f"  Annualized Volatility:    {metrics.get('annualized_volatility', 0):>14.2f}%"
        )
        lines.append(
            f"  Annual Return/Risk Ratio: {metrics.get('annualized_return_risk_ratio', 0):>14.2f}"
        )
        lines.append(f"  Capital Risk Rate:        {metrics.get('capital_risk_rate', 0):>14.2f}%")

        # -- 4. Drawdown Analysis --
        lines.append("\n" + "-" * W)
        lines.append("DRAWDOWN ANALYSIS")
        lines.append("-" * W)
        dd = metrics.get("max_drawdown", 0)
        dd_pct = metrics.get("max_drawdown_pct", 0)
        lines.append(f"  Max Equity Drawdown:     ${dd:>15,.2f}")
        lines.append(f"  Max Equity Drawdown %:    {dd_pct:>14.2f}%")
        dd_date = metrics.get("max_drawdown_date")
        if dd_date is not None:
            lines.append(f"  Max Drawdown Date:        {str(dd_date):>14s}")
        lines.append(
            f"  Max Drawdown Duration:    {metrics.get('max_drawdown_duration_days', 0):>14} trades"
        )
        lines.append(f"  Number of DD Episodes:    {metrics.get('num_drawdown_episodes', 0):>14}")
        lines.append(f"  Max P/L Drawdown:        ${metrics.get('max_pl_drawdown', 0):>15,.2f}")
        lines.append(f"  Max P/L Drawdown %:       {metrics.get('max_pl_drawdown_pct', 0):>14.2f}%")

        # -- 5. Trade Statistics --
        lines.append("\n" + "-" * W)
        lines.append("TRADE STATISTICS")
        lines.append("-" * W)
        total_trades = metrics.get("total_trades", 0)
        wins = metrics.get("winning_trades", 0)
        losses = metrics.get("losing_trades", 0)
        be = metrics.get("breakeven_trades", 0)
        wr = metrics.get("win_rate", 0) * 100

        lines.append(f"  Win Rate:                 {wr:>14.2f}%")
        lines.append(f"  Total Trades:             {total_trades:>14}")
        lines.append(f"  Winning Trades:           {wins:>14}")
        lines.append(f"  Losing Trades:            {losses:>14}")
        lines.append(f"  Breakeven Trades:         {be:>14}")
        lines.append(f"  Profit Factor:            {metrics.get('profit_factor', 0):>14.2f}")
        lines.append(
            f"  Average Profit/Trade:    ${metrics.get('avg_profit_per_trade', 0):>15,.2f}"
        )
        lines.append(f"  Average Win:             ${metrics.get('avg_win', 0):>15,.2f}")
        lines.append(f"  Average Loss:            ${metrics.get('avg_loss', 0):>15,.2f}")
        lines.append(f"  Largest Win:             ${metrics.get('largest_win', 0):>15,.2f}")
        lines.append(f"  Largest Loss:            ${metrics.get('largest_loss', 0):>15,.2f}")
        lines.append(f"  Expectancy:              ${metrics.get('expectancy', 0):>15,.2f}")
        lines.append(f"  Risk/Reward Ratio:        {metrics.get('risk_reward_ratio', 0):>14.2f}")

        # -- 6. Streaks and Duration --
        lines.append("\n" + "-" * W)
        lines.append("STREAKS & DURATION")
        lines.append("-" * W)
        lines.append(f"  Trade Win Streak (Max):   {metrics.get('max_win_streak', 0):>14}")
        lines.append(f"  Trade Loss Streak (Max):  {metrics.get('max_loss_streak', 0):>14}")
        lines.append(
            f"  Max Consecutive Win Days: {metrics.get('max_consecutive_winning_days', 0):>14}"
        )
        lines.append(
            f"  Max Consecutive Loss Days:{metrics.get('max_consecutive_losing_days', 0):>14}"
        )
        lines.append(
            f"  Max Profit Increments:    {metrics.get('max_consecutive_profit_increments', 0):>14}"
        )
        lines.append(
            f"  Max Loss Increments:      {metrics.get('max_consecutive_loss_increments', 0):>14}"
        )

        # -- 7. Monthly Breakdown (if available) --
        profitable_months = metrics.get("profitable_months")
        total_months = metrics.get("total_months")
        if profitable_months is not None and total_months and total_months > 0:
            lines.append("\n" + "-" * W)
            lines.append("MONTHLY BREAKDOWN")
            lines.append("-" * W)
            pct_pm = (profitable_months / total_months) * 100
            lines.append(
                f"  Profitable Months:        {profitable_months:>14} / {total_months} ({pct_pm:.1f}%)"
            )

        lines.append("\n" + "=" * W)

        return "\n".join(lines)
