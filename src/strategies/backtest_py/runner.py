"""
backtesting.py Runner

Executes strategies using the backtesting.py framework with integrated
visualization support.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import pandas as pd

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backtesting import Backtest  # noqa: E402

from src.visualization.report import ReportGenerator  # noqa: E402

BACKTESTING_AVAILABLE = True


def _extract_stats(results: Any) -> Dict[str, Any]:
    """Extract stats from backtesting.py Results object."""
    try:
        return dict(results._asdict())
    except AttributeError:
        stats: Dict[str, Any] = {}
        for key in results.index:
            try:
                stats[key] = results[key]
            except (KeyError, TypeError):
                pass
        return stats


class BacktestPyRunner:
    """
    Runner for executing strategies with backtesting.py framework.

    Provides a simple interface to run backtests, optimize parameters,
    and generate reports.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        cash: float = 100000,
        commission: float = 0.001,
        exclusive_orders: bool = True,
        output_dir: str = "reports",
    ):
        """
        Initialize the backtest runner.

        Args:
            data: OHLCV DataFrame with datetime index
                  Required columns: Open, High, Low, Close, Volume (optional)
            cash: Initial capital
            commission: Commission rate (default 0.1%)
            exclusive_orders: Close existing position before opening new
            output_dir: Directory for generated reports
        """
        if not BACKTESTING_AVAILABLE:
            raise ImportError(
                "backtesting.py is not installed. Install with: pip install backtesting"
            )

        # Validate and prepare data
        self.data = self._prepare_data(data)
        self.cash = cash
        self.commission = commission
        self.exclusive_orders = exclusive_orders
        self.output_dir = output_dir

        self.bt: Any = None
        self.results: Any = None
        self.report_gen = ReportGenerator(output_dir=output_dir)

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for backtesting.py.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with proper format for backtesting.py
        """
        df = data.copy()

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df = df.set_index("date")
            elif "timestamp" in df.columns:
                df = df.set_index("timestamp")
            elif "Date" in df.columns:
                df = df.set_index("Date")
            df.index = pd.to_datetime(df.index)

        # Ensure required columns exist with proper casing
        column_mapping = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }

        df.columns = [column_mapping.get(c.lower(), c) for c in df.columns]

        # Add Volume if missing
        if "Volume" not in df.columns:
            df["Volume"] = 0

        return df

    def run(
        self,
        strategy_class=None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run backtest with specified strategy.

        Args:
            strategy_class: Strategy class to use (default: MultiPatternStrategy)
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            **kwargs: Strategy parameters to override

        Returns:
            Dictionary with backtest results
        """
        # Import default strategy if not provided
        if strategy_class is None:
            from .multi_pattern_strategy import MultiPatternStrategy

            strategy_class = MultiPatternStrategy

        # Filter data by date range
        df = self.data.copy()
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        # Create backtest instance
        self.bt = Backtest(
            df,
            strategy_class,
            cash=self.cash,
            commission=self.commission,
            exclusive_orders=self.exclusive_orders,
        )

        # Run backtest
        print(f"Running backtest on {len(df)} bars...")
        self.results = self.bt.run(**kwargs)  # type: ignore[union-attr]

        # Print summary
        self._print_summary()

        return self.get_results()

    def optimize(
        self,
        strategy_class=None,
        max_tries: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **params,
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters.

        Args:
            strategy_class: Strategy class to use
            max_tries: Maximum optimization iterations
            start_date: Start date filter
            end_date: End date filter
            **params: Parameter ranges to optimize
                      e.g., min_confidence=[0.5, 0.55, 0.6, 0.65, 0.7]

        Returns:
            Dictionary with optimization results
        """
        if strategy_class is None:
            from .multi_pattern_strategy import MultiPatternStrategy

            strategy_class = MultiPatternStrategy

        # Filter data
        df = self.data.copy()
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        # Create backtest
        self.bt = Backtest(
            df,
            strategy_class,
            cash=self.cash,
            commission=self.commission,
            exclusive_orders=self.exclusive_orders,
        )

        print(f"Optimizing parameters with {max_tries} iterations...")

        # Run optimization
        self.results = self.bt.optimize(  # type: ignore[union-attr]
            max_tries=max_tries,
            maximize="Return [%]",  # Maximize return
            **params,
        )

        print("Optimization complete!")
        print(f"Best parameters: {self.results._strategy}")  # type: ignore[union-attr]

        return self.get_stats()

    def plot(
        self, filename: Optional[str] = None, open_browser: bool = True, show_legend: bool = True
    ) -> None:
        """
        Plot backtest results.

        Args:
            filename: Save to file if provided
            open_browser: Open in browser
            show_legend: Show legend on plot
        """
        if self.bt is None:
            raise ValueError("Run backtest first before plotting")

        assert self.bt is not None
        self.bt.plot(filename=filename, open_browser=open_browser)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get backtest statistics.

        Returns:
            Dictionary with backtest statistics
        """
        if self.results is None:
            raise ValueError("Run backtest first")

        # Convert to dictionary
        assert self.results is not None
        stats: Dict[str, Any] = dict(self.results)

        # Add equity curve and trades with proper keys for notebook compatibility
        stats["_equity_curve"] = self.results._equity_curve
        stats["_trades"] = list(self.results._trades)

        return stats

    def get_results(self) -> Dict[str, Any]:
        """
        Get backtest results in a format compatible with notebooks.

        Returns:
            Dictionary with 'stats', 'equity_curve', and 'trades' keys
        """
        if self.results is None:
            raise ValueError("Run backtest first")

        # Convert stats to dictionary - handle backtesting.py Results object properly
        # The Results object supports _asdict() for conversion to OrderedDict
        assert self.results is not None
        stats: Dict[str, Any] = _extract_stats(self.results)

        return {
            "stats": stats,
            "equity_curve": self.results._equity_curve,
            "trades": self.get_trades(),
        }

    def get_trades(self) -> pd.DataFrame:
        """
        Get list of trades as DataFrame.

        Returns:
            DataFrame with trade details
        """
        if self.results is None:
            raise ValueError("Run backtest first")

        assert self.results is not None
        trades = list(self.results._trades)
        if not trades:
            return pd.DataFrame()

        # Handle different trade formats from backtesting.py
        trade_records = []
        for t in trades:
            # Check if t is a string (column name) or a trade object
            if isinstance(t, str):
                # Skip string entries (these are column names in some versions)
                continue

            # Try to access trade attributes
            try:
                trade_records.append(
                    {
                        "entry_time": getattr(t, "entry_time", None),
                        "exit_time": getattr(t, "exit_time", None),
                        "entry_price": getattr(t, "entry_price", None),
                        "exit_price": getattr(t, "exit_price", None),
                        "size": getattr(t, "size", 0),
                        "pnl": getattr(t, "pl", getattr(t, "pnl", 0)),
                        "pnl_pct": getattr(t, "pl_pct", getattr(t, "pnl_pct", 0)),
                        "return_pct": getattr(t, "return_pct", 0),
                        "duration": getattr(t, "duration", None),
                        "direction": "LONG" if getattr(t, "size", 0) > 0 else "SHORT",
                    }
                )
            except Exception:
                # Skip trades that can't be processed
                continue

        if not trade_records:
            return pd.DataFrame()

        return pd.DataFrame(trade_records)

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve as DataFrame.

        Returns:
            DataFrame with equity over time
        """
        if self.results is None:
            raise ValueError("Run backtest first")

        assert self.results is not None
        return self.results._equity_curve

    def generate_report(
        self, title: str = "Strategy Report", include_plot: bool = True
    ) -> Dict[str, Path]:
        """
        Generate comprehensive report with visualization.

        Args:
            title: Report title
            include_plot: Whether to include HTML plot

        Returns:
            Dictionary with paths to generated files
        """
        if self.results is None:
            raise ValueError("Run backtest first")

        assert self.results is not None
        assert self.bt is not None
        files: Dict[str, Path] = {}

        # Generate tearsheet
        print("Generating tearsheet...")
        tearsheet_path = self.report_gen.tearsheet_gen.from_backtesting_py(
            stats=self.results,
            title=title,
            output_path=str(
                Path(self.output_dir) / f"{title.lower().replace(' ', '_')}_tearsheet.html"
            ),
        )
        files["tearsheet"] = Path(tearsheet_path)

        # Generate plot
        if include_plot:
            print("Generating plot...")
            plot_path = Path(self.output_dir) / f"{title.lower().replace(' ', '_')}_plot.html"
            self.bt.plot(filename=str(plot_path), open_browser=False)
            files["plot"] = plot_path

        print(f"Report generated: {len(files)} files")
        return files

    def _print_summary(self) -> None:
        """Print backtest summary."""
        if self.results is None:
            return

        assert self.results is not None

        print("\n" + "=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)

        # Key metrics
        print(f"Start Date: {self.results['Start']}")
        print(f"End Date: {self.results['End']}")
        print(f"Duration: {self.results['Duration']}")
        print(f"\nInitial Capital: ${self.cash:,.2f}")
        print(f"Final Equity: ${self.results['Equity Final [$]']:,.2f}")
        print(f"Final Return: {self.results['Return [%]']:.2f}%")
        print(f"\nTotal Trades: {self.results['# Trades']}")
        print(f"Win Rate: {self.results['Win Rate [%]']:.2f}%")
        print(f"Best Trade: {self.results['Best Trade [%]']:.2f}%")
        print(f"Worst Trade: {self.results['Worst Trade [%]']:.2f}%")
        print(f"Avg Trade: {self.results['Avg. Trade [%]']:.2f}%")
        print(f"\nMax Drawdown: {self.results['Max. Drawdown [%]']:.2f}%")
        print(f"Sharpe Ratio: {self.results['Sharpe Ratio']:.2f}")
        print(f"Sortino Ratio: {self.results['Sortino Ratio']:.2f}")
        print(f"Calmar Ratio: {self.results['Calmar Ratio']:.2f}")
        print("=" * 60 + "\n")


def run_backtest(
    data: pd.DataFrame,
    strategy_class=None,
    cash: float = 100000,
    commission: float = 0.001,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: str = "reports",
    generate_report: bool = True,
    title: str = "Strategy Backtest",
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to run a backtest with one call.

    Args:
        data: OHLCV DataFrame
        strategy_class: Strategy class to use
        cash: Initial capital
        commission: Commission rate
        start_date: Start date filter
        end_date: End date filter
        output_dir: Directory for reports
        generate_report: Whether to generate report
        title: Report title
        **kwargs: Strategy parameters

    Returns:
        Dictionary with results and file paths
    """
    runner = BacktestPyRunner(data=data, cash=cash, commission=commission, output_dir=output_dir)

    stats = runner.run(
        strategy_class=strategy_class, start_date=start_date, end_date=end_date, **kwargs
    )

    result = {
        "stats": stats,
        "trades": runner.get_trades(),
        "equity_curve": runner.get_equity_curve(),
        "files": {},
    }

    if generate_report:
        result["files"] = runner.generate_report(title=title)

    return result


def compare_strategies(
    data: pd.DataFrame,
    strategies: List[Type],
    names: List[str],
    cash: float = 100000,
    commission: float = 0.001,
    output_dir: str = "reports",
) -> Dict[str, Any]:
    """
    Compare multiple strategies.

    Args:
        data: OHLCV DataFrame
        strategies: List of strategy classes
        names: List of strategy names
        cash: Initial capital
        commission: Commission rate
        output_dir: Directory for reports

    Returns:
        Dictionary with comparison results
    """
    results = {}

    for strategy, name in zip(strategies, names):
        print(f"\nRunning {name}...")
        runner = BacktestPyRunner(
            data=data, cash=cash, commission=commission, output_dir=output_dir
        )

        stats = runner.run(strategy_class=strategy)
        results[name] = {
            "stats": stats,
            "trades": runner.get_trades(),
            "equity_curve": runner.get_equity_curve(),
        }

    # Generate comparison report
    report_gen = ReportGenerator(output_dir=output_dir)

    comparison_files = report_gen.generate_comparison_report(
        results_list=[results[name]["stats"] for name in names],
        names=names,
        title="Strategy Comparison",
    )

    return {"results": results, "comparison_files": comparison_files}
