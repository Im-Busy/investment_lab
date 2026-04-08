"""
Unified Report Generation

Generates comprehensive reports for both backtesting.py and custom engine results.
Combines quantstats tearsheets with mplfinance charts.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .charts import ChartGenerator
from .pattern_markers import PatternMarkerGenerator
from .tearsheet import TearsheetGenerator


class ReportGenerator:
    """
    Generate unified reports from any backtest engine.

    Works with:
    - Custom BacktestEngine results
    - backtesting.py results
    - Raw equity curves and trade lists
    """

    def __init__(self, output_dir: str = "reports", risk_free_rate: float = 0.02):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tearsheet_gen = TearsheetGenerator(risk_free_rate=risk_free_rate)
        self.chart_gen = ChartGenerator()
        self.marker_gen = PatternMarkerGenerator()

    def generate_full_report(
        self,
        results: Union[Dict[str, Any], Any],
        df: pd.DataFrame,
        title: str = "Strategy Report",
        benchmark: Optional[pd.Series] = None,
        include_trades: bool = True,
        include_patterns: bool = True,
    ) -> Dict[str, Path]:
        """
        Generate complete report with tearsheet and charts.

        Args:
            results: Backtest results (from either engine)
            df: OHLCV DataFrame
            title: Report title
            benchmark: Optional benchmark returns series
            include_trades: Whether to include trade charts
            include_patterns: Whether to include pattern charts

        Returns:
            Dictionary with paths to generated files
        """
        files = {}
        base_name = title.lower().replace(" ", "_")

        # Extract data based on result type
        equity_curve, trades, signals = self._extract_results(results)

        # 1. Generate tearsheet
        print(f"Generating tearsheet for {title}...")
        tearsheet_path = self.output_dir / f"{base_name}_tearsheet.html"

        try:
            if isinstance(results, dict):
                self.tearsheet_gen.from_custom_engine(
                    equity_curve=equity_curve,
                    trades=trades,
                    benchmark=benchmark,
                    title=title,
                    output_path=str(tearsheet_path),
                )
            else:
                self.tearsheet_gen.from_backtesting_py(
                    stats=results, title=title, output_path=str(tearsheet_path)
                )
            files["tearsheet"] = tearsheet_path
            print(f"  ✓ Tearsheet saved: {tearsheet_path}")
        except Exception as e:
            print(f"  ✗ Tearsheet failed: {e}")

        # 2. Generate equity curve chart
        print("Generating equity curve chart...")
        equity_path = self.output_dir / f"{base_name}_equity.png"

        try:
            self.chart_gen.plot_equity_curve(
                equity_curve=equity_curve,
                benchmark=benchmark,
                title=f"{title} - Equity Curve",
                save_path=str(equity_path),
                show=False,
            )
            files["equity_curve"] = equity_path
            print(f"  ✓ Equity curve saved: {equity_path}")
        except Exception as e:
            print(f"  ✗ Equity curve failed: {e}")

        # 3. Generate drawdown chart
        print("Generating drawdown chart...")
        drawdown_path = self.output_dir / f"{base_name}_drawdown.png"

        try:
            self.chart_gen.plot_drawdown(
                equity_curve=equity_curve,
                title=f"{title} - Drawdown",
                save_path=str(drawdown_path),
                show=False,
            )
            files["drawdown"] = drawdown_path
            print(f"  ✓ Drawdown chart saved: {drawdown_path}")
        except Exception as e:
            print(f"  ✗ Drawdown chart failed: {e}")

        # 4. Generate pattern chart (if signals available)
        if include_patterns and signals:
            print("Generating pattern chart...")
            pattern_path = self.output_dir / f"{base_name}_patterns.png"

            try:
                self.chart_gen.plot_with_patterns(
                    df=df,
                    signals=signals,
                    title=f"{title} - Pattern Signals",
                    save_path=str(pattern_path),
                    show=False,
                )
                files["patterns"] = pattern_path
                print(f"  ✓ Pattern chart saved: {pattern_path}")
            except Exception as e:
                print(f"  ✗ Pattern chart failed: {e}")

        # 5. Generate trade chart (if trades available)
        if include_trades and trades:
            print("Generating trade chart...")
            trade_path = self.output_dir / f"{base_name}_trades.png"

            try:
                self.chart_gen.plot_with_trades(
                    df=df,
                    trades=trades,
                    title=f"{title} - Trades",
                    save_path=str(trade_path),
                    show=False,
                )
                files["trades"] = trade_path
                print(f"  ✓ Trade chart saved: {trade_path}")
            except Exception as e:
                print(f"  ✗ Trade chart failed: {e}")

        # 6. Generate summary JSON
        print("Generating summary...")
        summary_path = self.output_dir / f"{base_name}_summary.json"

        try:
            summary = self._generate_summary(equity_curve, trades, signals)
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            files["summary"] = summary_path
            print(f"  ✓ Summary saved: {summary_path}")
        except Exception as e:
            print(f"  ✗ Summary failed: {e}")

        print(f"\nReport generation complete. {len(files)} files generated.")
        return files

    def generate_comparison_report(
        self,
        results_list: List[Union[Dict[str, Any], Any]],
        names: List[str],
        title: str = "Strategy Comparison",
        benchmark: Optional[pd.Series] = None,
    ) -> Dict[str, Path]:
        """
        Generate comparison report for multiple strategies.

        Args:
            results_list: List of backtest results
            names: Strategy names
            title: Report title
            benchmark: Optional benchmark returns series

        Returns:
            Dictionary with paths to generated files
        """
        files = {}
        base_name = title.lower().replace(" ", "_")

        # Extract returns for each strategy
        returns_dict = {}
        for results, name in zip(results_list, names):
            equity_curve, _, _ = self._extract_results(results)

            if isinstance(equity_curve, pd.DataFrame):
                equity = equity_curve["equity"]
            else:
                equity = equity_curve

            returns_dict[name] = equity.pct_change().dropna()

        # Generate comparison tearsheet
        print(f"Generating comparison report for {len(names)} strategies...")
        comparison_path = self.output_dir / f"{base_name}_comparison.html"

        try:
            self.tearsheet_gen.generate_comparison(
                returns_dict=returns_dict,
                benchmark=benchmark,
                title=title,
                output_path=str(comparison_path),
            )
            files["comparison"] = comparison_path
            print(f"  ✓ Comparison saved: {comparison_path}")
        except Exception as e:
            print(f"  ✗ Comparison failed: {e}")

        # Generate comparison chart
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 6))

        for name, returns in returns_dict.items():
            equity = (1 + returns).cumprod()
            ax.plot(equity.index, equity.values, label=name, linewidth=2)

        if benchmark is not None:
            benchmark_equity = (1 + benchmark).cumprod()
            ax.plot(
                benchmark_equity.index,
                benchmark_equity.values,  # type: ignore[arg-type]
                label="Benchmark",
                linewidth=1.5,
                linestyle="--",
                alpha=0.7,
            )

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        ax.grid(True, alpha=0.3)

        chart_path = self.output_dir / f"{base_name}_comparison.png"
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        files["comparison_chart"] = chart_path
        print(f"  ✓ Comparison chart saved: {chart_path}")

        return files

    def generate_quick_report(
        self, equity_curve: pd.DataFrame, title: str = "Quick Report"
    ) -> Path:
        """
        Generate a quick tearsheet from equity curve only.

        Args:
            equity_curve: DataFrame with 'timestamp' and 'equity' columns
            title: Report title

        Returns:
            Path to generated tearsheet
        """
        base_name = title.lower().replace(" ", "_")
        tearsheet_path = self.output_dir / f"{base_name}_tearsheet.html"

        return Path(self.tearsheet_gen.from_custom_engine(
            equity_curve=equity_curve, title=title, output_path=str(tearsheet_path)
        ))

    def _extract_results(self, results: Union[Dict[str, Any], Any]) -> tuple:
        """
        Extract equity curve, trades, and signals from results.

        Args:
            results: Backtest results (dict or object)

        Returns:
            Tuple of (equity_curve, trades, signals)
        """
        if isinstance(results, dict):
            # Custom engine results
            equity_curve = results.get("equity_curve")
            trades = results.get("trades", [])
            signals = results.get("signals", [])

            # Handle case where equity_curve is a DataFrame
            if equity_curve is None and "metrics" in results:
                # Try to reconstruct from trades
                pass

            return equity_curve, trades, signals

        else:
            # backtesting.py results
            try:
                equity_curve = results._equity_curve
                trades = list(results._trades) if hasattr(results, "_trades") else []
                signals = []
                return equity_curve, trades, signals
            except AttributeError:
                # Try dictionary-like access
                equity_curve = results.get("_equity_curve", results.get("equity_curve"))
                trades = results.get("_trades", results.get("trades", []))
                signals = results.get("signals", [])
                return equity_curve, trades, signals

    def _generate_summary(
        self,
        equity_curve: pd.DataFrame,
        trade_data: Union[List[Dict[str, Any]], pd.DataFrame],
        signal_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate summary statistics from backtest results.

        Args:
            equity_curve: DataFrame with equity data
            trades: List of trade dictionaries or DataFrame of trades
            signals: List of signal dictionaries

        Returns:
            Dictionary with summary statistics
        """
        metrics_dict: Dict[str, Any] = {}
        trade_stats: Dict[str, Any] = {}
        sig_stats: Dict[str, Any] = {}
        summary: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "metrics": metrics_dict,
            "trades": trade_stats,
            "signals": sig_stats,
        }

        # Equity metrics
        if equity_curve is not None:
            if isinstance(equity_curve, pd.DataFrame):
                equity = equity_curve["equity"]
            else:
                equity = equity_curve

            metrics_dict["initial_equity"] = float(equity.iloc[0])
            metrics_dict["final_equity"] = float(equity.iloc[-1])
            metrics_dict["total_return"] = float((equity.iloc[-1] / equity.iloc[0] - 1) * 100)
            metrics_dict["max_drawdown"] = float(self._calculate_max_drawdown(equity))

            returns = equity.pct_change().dropna()
            if len(returns) > 0:
                metrics_dict["sharpe_ratio"] = (
                    float(returns.mean() / returns.std() * np.sqrt(252))
                    if returns.std() > 0
                    else 0.0
                )

        # Trade statistics
        if trade_data is not None and len(trade_data) > 0:
            # Convert to DataFrame if needed
            trades_df: pd.DataFrame
            if isinstance(trade_data, pd.DataFrame):
                trades_df = trade_data
            else:
                trades_df = pd.DataFrame(trade_data)

            trade_stats["total_trades"] = len(trades_df)

            if "pnl" in trades_df.columns:
                pnl_col = trades_df["pnl"]
                winning = trades_df[pnl_col > 0]  # type: ignore[index]
                losing = trades_df[pnl_col < 0]  # type: ignore[index]

                trade_stats["winning_trades"] = len(winning)
                trade_stats["losing_trades"] = len(losing)
                trade_stats["win_rate"] = (
                    len(winning) / len(trades_df) if len(trades_df) > 0 else 0
                )
                trade_stats["avg_win"] = (
                    float(winning["pnl"].mean()) if len(winning) > 0 else 0
                )
                trade_stats["avg_loss"] = (
                    float(losing["pnl"].mean()) if len(losing) > 0 else 0
                )
                trade_stats["total_pnl"] = float(trades_df["pnl"].sum())

                total_wins = winning["pnl"].sum() if len(winning) > 0 else 0
                total_losses = abs(losing["pnl"].sum()) if len(losing) > 0 else 0
                trade_stats["profit_factor"] = (
                    float(total_wins / total_losses) if total_losses > 0 else 0
                )

            if "pattern" in trades_df.columns:
                pattern_counts = trades_df["pattern"].value_counts().to_dict()
                trade_stats["by_pattern"] = pattern_counts

            if "direction" in trades_df.columns:
                direction_counts = trades_df["direction"].value_counts().to_dict()
                trade_stats["by_direction"] = direction_counts

        # Signal statistics
        if signal_data:
            signals_df = pd.DataFrame(signal_data)
            sig_stats["total_signals"] = len(signal_data)

            if "pattern_name" in signals_df.columns:
                pattern_counts = signals_df["pattern_name"].value_counts().to_dict()
                sig_stats["by_pattern"] = pattern_counts

            if "direction" in signals_df.columns:
                direction_counts = signals_df["direction"].value_counts().to_dict()
                sig_stats["by_direction"] = direction_counts

            if "confidence" in signals_df.columns:
                sig_stats["avg_confidence"] = float(signals_df["confidence"].mean())

        return summary

    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown percentage."""
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        return float(drawdown.min() * 100)

    def list_reports(self) -> List[Path]:
        """
        List all generated reports.

        Returns:
            List of report file paths
        """
        return list(self.output_dir.glob("*.html")) + list(self.output_dir.glob("*.png"))

    def clean_reports(self, older_than_days: Optional[int] = None) -> int:
        """
        Clean old reports.

        Args:
            older_than_days: Delete reports older than this many days (None = all)

        Returns:
            Number of files deleted
        """
        deleted = 0

        for file_path in self.output_dir.glob("*"):
            if file_path.is_file():
                if older_than_days is None:
                    file_path.unlink()
                    deleted += 1
                else:
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if (datetime.now() - file_mtime).days > older_than_days:
                        file_path.unlink()
                        deleted += 1

        return deleted
