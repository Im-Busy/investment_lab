"""
Trading Pattern Detection System - Main Entry Point

This module provides the main interface for running pattern detection,
signal generation, and backtesting.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import pattern modules
from .backtest import (
    BacktestAdapter,
    BacktestEngine,
    PerformanceMetrics,
    UnifiedBacktestConfig,
    UnifiedBacktestResult,
)
from .backtest.engine import BacktestConfig
from .patterns.base import BasePattern
from .patterns.basic import NR7ID, FloorPivotBreakout, MarketStructureLow, MatchingLows, NBarDecline
from .patterns.breakout import (
    DonchianChannelBreakout,
)
from .patterns.classic import DeadCatBounce, DoubleBottom, DoubleTop, TraderVic2B, TripleTop
from .patterns.complex import (
    CupAndHandle,
    HeadAndShoulders,
    ParabolicArc,
    SpikeAndLedge,
    ThreeHillsMountain,
)
from .patterns.harmonic import ABCPattern, BollingerBands, GartleyPattern, SymmetricTriangle

# Import other modules
from .signals import SignalGenerator
from .utils import load_data, save_results, validate_ohlcv_data


def get_all_patterns() -> List[BasePattern]:
    """
    Get all available pattern detectors with default parameters.

    Returns:
        List of pattern detector instances
    """
    patterns = [
        # Basic patterns
        MarketStructureLow(),
        MatchingLows(),
        NR7ID(),
        NBarDecline(),
        FloorPivotBreakout(),
        # Harmonic patterns
        GartleyPattern(),
        ABCPattern(),
        SymmetricTriangle(),
        BollingerBands(),
        # Breakout patterns
        DonchianChannelBreakout(),
        # Complex patterns
        CupAndHandle(),
        HeadAndShoulders(),
        SpikeAndLedge(),
        ThreeHillsMountain(),
        ParabolicArc(),
        # Classic patterns
        DoubleTop(),
        DoubleBottom(),
        TraderVic2B(),
        TripleTop(),
        DeadCatBounce(),
    ]

    return patterns


def get_patterns_by_category(category: str) -> List[BasePattern]:
    """
    Get pattern detectors by category.

    Args:
        category: Pattern category ('basic', 'harmonic', 'complex', 'classic')

    Returns:
        List of pattern detector instances
    """
    if category == "basic":
        return [
            MarketStructureLow(),
            MatchingLows(),
            NR7ID(),
            NBarDecline(),
            FloorPivotBreakout(),
        ]
    elif category == "harmonic":
        return [
            GartleyPattern(),
            ABCPattern(),
            SymmetricTriangle(),
            BollingerBands(),
        ]
    elif category == "complex":
        return [
            CupAndHandle(),
            HeadAndShoulders(),
            SpikeAndLedge(),
            ThreeHillsMountain(),
            ParabolicArc(),
        ]
    elif category == "classic":
        return [
            DoubleTop(),
            DoubleBottom(),
            TraderVic2B(),
            TripleTop(),
            DeadCatBounce(),
        ]
    else:
        logger.warning(f"Unknown category: {category}. Returning all patterns.")
        return get_all_patterns()


def get_pattern_by_name(name: str) -> Optional[BasePattern]:
    """
    Get a specific pattern detector by name.

    Args:
        name: Pattern name (case-insensitive)

    Returns:
        Pattern detector instance or None if not found
    """
    pattern_map = {
        "msl": MarketStructureLow,
        "market structure low": MarketStructureLow,
        "matching lows": MatchingLows,
        "matching_lows": MatchingLows,
        "nr7": NR7ID,
        "nr7id": NR7ID,
        "nr7 inside day": NR7ID,
        "n bar decline": NBarDecline,
        "n_bar_decline": NBarDecline,
        "floor pivot": FloorPivotBreakout,
        "floor_pivot": FloorPivotBreakout,
        "gartley": GartleyPattern,
        "abc": ABCPattern,
        "symmetric triangle": SymmetricTriangle,
        "symmetric_triangle": SymmetricTriangle,
        "donchian": DonchianChannelBreakout,
        "donchian channel": DonchianChannelBreakout,
        "donchian channel breakout": DonchianChannelBreakout,
        "bollinger": BollingerBands,
        "bollinger bands": BollingerBands,
        "cup and handle": CupAndHandle,
        "cup_handle": CupAndHandle,
        "head and shoulders": HeadAndShoulders,
        "head_shoulders": HeadAndShoulders,
        "spike and ledge": SpikeAndLedge,
        "spike_ledge": SpikeAndLedge,
        "three hills": ThreeHillsMountain,
        "three_hills": ThreeHillsMountain,
        "three hills mountain": ThreeHillsMountain,
        "parabolic arc": ParabolicArc,
        "parabolic_arc": ParabolicArc,
        "double top": DoubleTop,
        "double_top": DoubleTop,
        "double bottom": DoubleBottom,
        "double_bottom": DoubleBottom,
        "2b": TraderVic2B,
        "trader vic 2b": TraderVic2B,
        "trader_vic_2b": TraderVic2B,
        "triple top": TripleTop,
        "triple_top": TripleTop,
        "dead cat bounce": DeadCatBounce,
        "dead_cat_bounce": DeadCatBounce,
    }

    name_lower = name.lower().strip()

    if name_lower in pattern_map:
        result = pattern_map[name_lower]()
        return result if result is not None else None

    return None


class PatternDetectionSystem:
    """
    Main class for the Trading Pattern Detection System.

    Provides a unified interface for:
    - Loading and validating data
    - Running pattern detection
    - Generating trading signals
    - Running backtests
    - Analyzing results
    """

    def __init__(
        self,
        patterns: Optional[List[BasePattern]] = None,
        min_confidence: float = 0.5,
        max_open_positions: int = 5,
        initial_equity: float = 100000.0,
        risk_per_trade: float = 0.02,
        engine: str = "backtesting.py",
    ):
        """
        Initialize the Pattern Detection System.

        Args:
            patterns: List of pattern detectors (default: all patterns)
            min_confidence: Minimum signal confidence threshold
            max_open_positions: Maximum concurrent positions
            initial_equity: Starting equity for backtests
            risk_per_trade: Risk per trade as fraction of equity
            engine: Backtest engine to use ('backtesting.py', 'homemade', or 'auto')
        """
        self.patterns = patterns if patterns is not None else get_all_patterns()
        self.min_confidence = min_confidence
        self.max_open_positions = max_open_positions
        self.initial_equity = initial_equity
        self.risk_per_trade = risk_per_trade
        self.engine_type = engine

        # Initialize components
        self.signal_generator = SignalGenerator(
            patterns=self.patterns,
            min_confidence=min_confidence,
            max_signals_per_bar=1,
            combine_same_direction=True,
        )

        self.backtest_engine = None
        self.backtest_adapter: Optional["BacktestAdapter"] = None
        self.last_results: Optional[UnifiedBacktestResult] = None

        logger.info(
            f"Initialized Pattern Detection System with {len(self.patterns)} patterns (engine: {engine})"
        )

    def load_data(
        self,
        filepath: str,
        date_column: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load and validate OHLCV data.

        Args:
            filepath: Path to data file
            date_column: Name of date column
            start_date: Filter start date
            end_date: Filter end date

        Returns:
            DataFrame with OHLCV data
        """
        df = load_data(filepath, date_column=date_column)

        # Validate data
        is_valid, issues = validate_ohlcv_data(df)
        if not is_valid:
            logger.warning(f"Data validation issues: {issues}")

        # Filter by date range
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        logger.info(f"Loaded {len(df)} rows of data from {filepath}")

        return df

    def scan_patterns(
        self, df: pd.DataFrame, start_index: Optional[int] = None, end_index: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Scan data for all patterns.

        Args:
            df: DataFrame with OHLCV data
            start_index: Starting index
            end_index: Ending index

        Returns:
            List of detected signals
        """
        signals = self.signal_generator.scan_dataframe(df, start_index, end_index)

        # Convert to dictionaries
        results = [s.to_dict() for s in signals]

        logger.info(f"Found {len(results)} signals")

        return results

    def run_backtest(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        engine: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a backtest on the data.

        Args:
            df: DataFrame with OHLCV data
            start_date: Backtest start date
            end_date: Backtest end date
            commission_pct: Commission percentage
            slippage_pct: Slippage percentage
            engine: Override engine type ('backtesting.py', 'homemade', or 'auto')

        Returns:
            Dictionary with backtest results
        """
        # Use provided engine or default from init
        engine_type = engine or self.engine_type

        # Create unified configuration
        unified_config = UnifiedBacktestConfig(
            initial_equity=self.initial_equity,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            risk_per_trade=self.risk_per_trade,
            max_open_positions=self.max_open_positions,
            min_confidence=self.min_confidence,
            start_date=start_date,
            end_date=end_date,
        )

        # Create adapter and run backtest
        self.backtest_adapter = BacktestAdapter(engine=engine_type, default_config=unified_config)
        assert self.backtest_adapter is not None

        result = self.backtest_adapter.run_backtest(
            data=df, patterns=self.patterns, config=unified_config
        )

        # Store results in compatible format
        self.last_results = result

        total_trades = result.metrics.get("total_trades", 0)
        logger.info(
            f"Backtest complete: {total_trades} trades (engine: {result.engine_type.value})"
        )

        return result.to_dict()

    def get_performance_report(self) -> str:
        """
        Get a formatted performance report from the last backtest.

        Returns:
            Formatted performance report string
        """
        if self.last_results is None or self.last_results.metrics is None:
            return "No backtest results available. Run run_backtest() first."

        return PerformanceMetrics.format_report(self.last_results.metrics)

    def get_pattern_statistics(self) -> pd.DataFrame:
        """
        Get performance statistics broken down by pattern.

        Returns:
            DataFrame with pattern statistics
        """
        if self.last_results is None:
            return pd.DataFrame()

        return PerformanceMetrics.calculate_pattern_statistics(
            self.last_results.trades.to_dict("records")
        )  # type: ignore[union-attr]

    def export_results(self, filepath: str, format: str = "json") -> None:
        """
        Export backtest results to file.

        Args:
            filepath: Output file path
            format: Output format ('json', 'csv')
        """
        if self.last_results is None:
            logger.warning("No results to export")
            return

        save_results(self.last_results.to_dict(), filepath, format)
        logger.info(f"Results exported to {filepath}")


def main():
    """Command-line interface for the Pattern Detection System."""
    parser = argparse.ArgumentParser(
        description="Trading Pattern Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest on data file
  python -m src.main backtest data/SPY_historical.csv --start 2020-01-01 --end 2023-12-31

  # Scan for patterns only
  python -m src.main scan data/SPY_historical.csv --category classic

  # Run with specific patterns
  python -m src.main backtest data/SPY_historical.csv --patterns "Double Top,Double Bottom"

  # Export results
  python -m src.main backtest data/SPY_historical.csv --output results.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest on data")
    backtest_parser.add_argument("datafile", help="Path to OHLCV data file")
    backtest_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--equity", type=float, default=100000, help="Initial equity")
    backtest_parser.add_argument("--risk", type=float, default=0.02, help="Risk per trade")
    backtest_parser.add_argument(
        "--confidence", type=float, default=0.5, help="Min confidence threshold"
    )
    backtest_parser.add_argument(
        "--category",
        choices=["basic", "harmonic", "complex", "classic", "all"],
        default="all",
        help="Pattern category to use",
    )
    backtest_parser.add_argument("--patterns", help="Comma-separated list of specific patterns")
    backtest_parser.add_argument("--output", help="Output file for results")
    backtest_parser.add_argument("--report", action="store_true", help="Print performance report")
    backtest_parser.add_argument(
        "--engine",
        choices=["backtesting.py", "homemade", "auto"],
        default="backtesting.py",
        help="Backtest engine to use (default: backtesting.py)",
    )

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan for patterns without backtest")
    scan_parser.add_argument("datafile", help="Path to OHLCV data file")
    scan_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    scan_parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    scan_parser.add_argument(
        "--category",
        choices=["basic", "harmonic", "complex", "classic", "all"],
        default="all",
        help="Pattern category to use",
    )
    scan_parser.add_argument(
        "--confidence", type=float, default=0.5, help="Min confidence threshold"
    )
    scan_parser.add_argument("--output", help="Output file for signals")

    # List patterns command
    list_parser = subparsers.add_parser("list", help="List available patterns")

    args = parser.parse_args()

    if args.command == "list":
        print("\nAvailable Patterns:")
        print("=" * 50)
        for category in ["basic", "harmonic", "complex", "classic"]:
            patterns = get_patterns_by_category(category)
            print(f"\n{category.upper()}:")
            for p in patterns:
                print(f"  - {p.name}")
        return

    if args.command == "scan":
        # Initialize system
        if args.category == "all":
            patterns = get_all_patterns()
        else:
            patterns = get_patterns_by_category(args.category)

        system = PatternDetectionSystem(patterns=patterns, min_confidence=args.confidence)

        # Load data
        df = system.load_data(args.datafile, start_date=args.start, end_date=args.end)

        # Scan for patterns
        signals = system.scan_patterns(df)

        print(f"\nFound {len(signals)} signals")

        if args.output:
            save_results({"signals": signals}, args.output)
            print(f"Signals saved to {args.output}")
        else:
            for signal in signals[:10]:  # Show first 10
                print(f"\n{signal['timestamp']}: {signal['patterns']}")
                print(f"  Direction: {signal['direction']}")
                print(f"  Entry: {signal['entry_price']:.2f}")
                print(f"  Stop: {signal['stop_loss']:.2f}")
                print(f"  Confidence: {signal['confidence']:.2%}")

        return

    if args.command == "backtest":
        # Get patterns
        if args.patterns:
            pattern_names = [p.strip() for p in args.patterns.split(",")]
            patterns = [get_pattern_by_name(name) for name in pattern_names]
            patterns = [p for p in patterns if p is not None]
        elif args.category == "all":
            patterns = get_all_patterns()
        else:
            patterns = get_patterns_by_category(args.category)

        print(f"\nUsing {len(patterns)} patterns")

        # Initialize system with selected engine
        system = PatternDetectionSystem(
            patterns=patterns,
            min_confidence=args.confidence,
            initial_equity=args.equity,
            risk_per_trade=args.risk,
            engine=args.engine,
        )

        # Load data
        df = system.load_data(args.datafile, start_date=args.start, end_date=args.end)

        # Run backtest
        print(f"\nRunning backtest with {args.engine} engine...")
        results = system.run_backtest(df)

        # Print report
        if args.report:
            print(system.get_performance_report())
        else:
            metrics = results.get("metrics", {})
            print("\nBacktest Results:")
            print(f"  Total Trades: {metrics.get('total_trades', 0)}")
            print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"  Total Return: {metrics.get('total_return', 0):.2f}%")
            print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")

        # Export results
        if args.output:
            system.export_results(args.output)
            print(f"\nResults saved to {args.output}")

        return

    # Visualization command
    viz_parser = subparsers.add_parser("visualize", help="Generate visualization reports")
    viz_parser.add_argument("datafile", help="Path to OHLCV data file")
    viz_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    viz_parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    viz_parser.add_argument("--output-dir", default="reports", help="Output directory for reports")
    viz_parser.add_argument("--title", default="Strategy Report", help="Report title")
    viz_parser.add_argument(
        "--tearsheet", action="store_true", help="Generate quantstats tearsheet"
    )
    viz_parser.add_argument(
        "--chart", action="store_true", help="Generate price chart with patterns"
    )
    viz_parser.add_argument("--all", action="store_true", help="Generate all reports")

    if args.command == "visualize":
        from .strategies.backtest_py import BacktestPyRunner, MultiPatternStrategySimple
        from .visualization import ReportGenerator

        # Load data
        df = load_data(args.datafile)
        if args.start:
            df = df[df.index >= args.start]
        if args.end:
            df = df[df.index <= args.end]

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run quick backtest
        print("Running backtest for visualization...")
        runner = BacktestPyRunner(data=df, cash=100000, commission=0.001)
        results = runner.run(strategy_class=MultiPatternStrategySimple)

        # Generate reports
        report_gen = ReportGenerator(output_dir=str(output_dir))

        if args.all or args.tearsheet:
            print("Generating tearsheet...")
            tearsheet_path = output_dir / f"{args.title.lower().replace(' ', '_')}_tearsheet.html"
            report_gen.tearsheet_gen.from_backtesting_py(
                stats=results, title=args.title, output_path=str(tearsheet_path)
            )
            print(f"  Tearsheet saved: {tearsheet_path}")

        if args.all or args.chart:
            print("Generating chart...")
            chart_path = output_dir / f"{args.title.lower().replace(' ', '_')}_chart.html"
            runner.plot(filename=str(chart_path), open_browser=False)
            print(f"  Chart saved: {chart_path}")

        if not args.all and not args.tearsheet and not args.chart:
            # Generate full report by default
            print("Generating full report...")
            files = report_gen.generate_full_report(results=results, df=df, title=args.title)
            print("\nGenerated files:")
            for name, path in files.items():
                print(f"  {name}: {path}")

        return

    # SMC backtest command
    smc_parser = subparsers.add_parser("smc", help="Run SMC/ICT strategy backtest")
    smc_parser.add_argument("datafile", help="Path to OHLCV data file (5-minute bars recommended)")
    smc_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    smc_parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    smc_parser.add_argument("--equity", type=float, default=100000, help="Initial equity")
    smc_parser.add_argument("--risk", type=float, default=0.01, help="Risk per trade")
    smc_parser.add_argument("--output", help="Output file for results")
    smc_parser.add_argument("--report", action="store_true", help="Generate visualization report")

    if args.command == "smc":
        logger.info("SMC strategy: use 'uv run scripts/backtest_smc.py' for backtesting")
        print("The 'smc' command has been replaced by the dedicated SMC backtest CLI.")
        print("Usage: uv run scripts/backtest_smc.py --symbol BTC-USD --interval 1h")

        return

        return

    parser.print_help()


if __name__ == "__main__":
    main()
