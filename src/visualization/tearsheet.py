"""
quantstats Tearsheet Integration

Generates professional performance reports from backtest results.
Works with both backtesting.py and custom engine results.
"""

import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path

try:
    import quantstats as qs

    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False


class TearsheetGenerator:
    """
    Generate quantstats tearsheets from backtest results.

    Works with both backtesting.py and custom engine results.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize tearsheet generator.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        if not QUANTSTATS_AVAILABLE:
            raise ImportError("quantstats is not installed. Install with: pip install quantstats")

        self.risk_free_rate = risk_free_rate

    def from_custom_engine(
        self,
        equity_curve: pd.DataFrame,
        trades: Optional[list] = None,
        benchmark: Optional[pd.Series] = None,
        title: str = "Strategy Performance",
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate tearsheet from custom engine results.

        Args:
            equity_curve: DataFrame with 'timestamp' and 'equity' columns
            trades: List of trade dictionaries (optional, for trade analysis)
            benchmark: Benchmark returns series (optional)
            title: Report title
            output_path: Path to save HTML report (optional)

        Returns:
            Path to generated HTML file
        """
        # Convert equity curve to returns
        if isinstance(equity_curve, pd.DataFrame):
            if "equity" in equity_curve.columns:
                equity = equity_curve["equity"]
            else:
                # Assume last column is equity
                equity = equity_curve.iloc[:, -1]

            # Set index if timestamp column exists
            if "timestamp" in equity_curve.columns:
                equity.index = pd.to_datetime(equity_curve["timestamp"])
        else:
            equity = equity_curve

        # Calculate returns
        returns = equity.pct_change().dropna()
        returns.name = "Strategy"

        # Generate output path if not provided
        if output_path is None:
            output_path = f"reports/{title.lower().replace(' ', '_')}_tearsheet.html"

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Generate tearsheet
        if benchmark is not None:
            qs.reports.html(
                returns,
                benchmark=benchmark,
                title=title,
                output=output_path,
                risk_free=self.risk_free_rate,
            )
        else:
            qs.reports.html(returns, title=title, output=output_path, risk_free=self.risk_free_rate)

        return output_path

    def from_backtesting_py(
        self,
        stats: Dict[str, Any],
        title: str = "Strategy Performance",
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate tearsheet from backtesting.py results.

        Args:
            stats: Statistics dictionary from backtesting.py Backtest.run()
            title: Report title
            output_path: Path to save HTML report (optional)

        Returns:
            Path to generated HTML file
        """
        # Extract equity curve from backtesting.py stats
        if "_equity_curve" in stats:
            equity = stats["_equity_curve"]["Equity"]
        elif hasattr(stats, "_equity_curve"):
            equity = stats._equity_curve["Equity"]
        else:
            raise ValueError("Could not extract equity curve from backtesting.py results")

        # Calculate returns
        returns = equity.pct_change().dropna()
        returns.name = "Strategy"

        # Generate output path if not provided
        if output_path is None:
            output_path = f"reports/{title.lower().replace(' ', '_')}_tearsheet.html"

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Generate tearsheet
        qs.reports.html(returns, title=title, output=output_path, risk_free=self.risk_free_rate)

        return output_path

    def from_returns(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        title: str = "Strategy Performance",
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate tearsheet from returns series.

        Args:
            returns: Daily returns series
            benchmark: Benchmark returns series (optional)
            title: Report title
            output_path: Path to save HTML report (optional)

        Returns:
            Path to generated HTML file
        """
        # Generate output path if not provided
        if output_path is None:
            output_path = f"reports/{title.lower().replace(' ', '_')}_tearsheet.html"

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Generate tearsheet
        if benchmark is not None:
            qs.reports.html(
                returns,
                benchmark=benchmark,
                title=title,
                output=output_path,
                risk_free=self.risk_free_rate,
            )
        else:
            qs.reports.html(returns, title=title, output=output_path, risk_free=self.risk_free_rate)

        return output_path

    def generate_comparison(
        self,
        returns_dict: Dict[str, pd.Series],
        benchmark: Optional[pd.Series] = None,
        title: str = "Strategy Comparison",
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate comparison tearsheet for multiple strategies.

        Args:
            returns_dict: Dictionary of strategy name -> returns series
            benchmark: Benchmark returns series (optional)
            title: Report title
            output_path: Path to save HTML report (optional)

        Returns:
            Path to generated HTML file
        """
        # Generate output path if not provided
        if output_path is None:
            output_path = f"reports/{title.lower().replace(' ', '_')}_comparison.html"

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Use first strategy as primary
        primary_name = list(returns_dict.keys())[0]
        primary_returns = returns_dict[primary_name]

        # Generate comparison report
        # Note: quantstats supports comparison via benchmark parameter
        # For multiple strategies, we'll use the primary with benchmark
        if benchmark is not None:
            qs.reports.html(
                primary_returns,
                benchmark=benchmark,
                title=title,
                output=output_path,
                risk_free=self.risk_free_rate,
            )
        else:
            qs.reports.html(
                primary_returns, title=title, output=output_path, risk_free=self.risk_free_rate
            )

        return output_path

    @staticmethod
    def get_available_metrics() -> list:
        """
        Get list of metrics available in quantstats reports.

        Returns:
            List of metric names
        """
        return [
            "Cumulative Return",
            "CAGR﹪(Ann.)",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Max Drawdown",
            "Longest DD Days",
            "Volatility (Ann.)",
            "Calmar Ratio",
            "Skew",
            "Kurtosis",
            "Expected Daily",
            "Expected Monthly",
            "Expected Yearly",
            "Kelly Criterion",
            "Risk of Ruin",
            "Daily Value-at-Risk",
            "Expected Shortfall (cVaR)",
            "Max Consecutive Wins",
            "Max Consecutive Losses",
            "Win Days",
            "Win Month",
            "Win Year",
            "Best Day",
            "Worst Day",
            "Best Month",
            "Worst Month",
            "Best Year",
            "Worst Year",
            "Avg. Up Month",
            "Avg. Down Month",
            "Profit Factor",
            "Common Sense Ratio",
            "Outlier Win",
            "Outlier Loss",
        ]
