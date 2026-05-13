"""
Notebook Helper Utilities

Centralized functions for common notebook operations including:
- Configuration validation and merging for dict-based configs
- Data loading and validation helpers
- Pattern analysis functions
- Visualization helpers
- Backtest utilities
- Reporting functions

Designed to support modular, configurable notebooks with clear separation
between parameters and implementation code.
"""

import json
import sys
import warnings
from copy import deepcopy

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

# Configure warnings
warnings.filterwarnings("ignore")

# ============================================================
# Configuration Classes and Defaults
# ============================================================


@dataclass
class DataConfig:
    """Data loading configuration with defaults."""

    file: str = "SPY_daily.csv"
    directory: str = "data/raw"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    columns: List[str] = field(default_factory=lambda: ["Open", "High", "Low", "Close", "Volume"])
    column_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )


@dataclass
class BacktestConfig:
    """Backtest runner configuration with defaults."""

    initial_equity: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005
    exclusive_orders: bool = True


@dataclass
class StrategyConfig:
    """Strategy parameters configuration with defaults."""

    min_confidence: float = 0.55
    risk_per_trade: float = 0.02
    max_open_positions: int = 3
    min_confluence_count: int = 2


@dataclass
class OutputConfig:
    """Output and display configuration with defaults."""

    directory: str = "reports"
    subdirectory: Optional[str] = None
    save_plots: bool = True
    show_plots: bool = True
    dpi: int = 150
    figsize_default: Tuple[int, int] = (14, 8)
    figsize_wide: Tuple[int, int] = (16, 8)
    figsize_square: Tuple[int, int] = (10, 10)


@dataclass
class PatternSelectionConfig:
    """Pattern selection framework configuration with defaults."""

    # File paths
    data_path: str = ""
    output_dir: str = "reports/pattern_selection"

    # Backtest configuration
    initial_equity: float = 1_000_000
    commission: float = 0.001
    min_confluence_count: int = 2

    # Phase 1 thresholds (Noise Filtering)
    min_trades: int = 30
    min_sharpe: float = 0.5
    min_profit_factor: float = 1.0
    min_win_rate: float = 0.40
    max_drawdown: float = 0.30

    # Phase 2 thresholds (Correlation Analysis)
    correlation_threshold: float = 0.8

    # Phase 3 thresholds (Ablation Testing)
    positive_contribution_threshold: float = 0.0
    noise_threshold: float = -0.05
    primary_signal_threshold: float = 0.1

    # Output configuration
    cache_results: bool = True

    # Quick test mode
    quick_test: bool = False
    quick_test_patterns: int = 5

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors: List[str] = []
        if self.min_trades < 1:
            errors.append("min_trades must be >= 1")
        if self.min_sharpe < 0:
            errors.append("min_sharpe must be >= 0")
        if self.min_profit_factor <= 0:
            errors.append("min_profit_factor must be > 0")
        if not 0 <= self.min_win_rate <= 1:
            errors.append("min_win_rate must be between 0 and 1")
        if not 0 <= self.max_drawdown <= 1:
            errors.append("max_drawdown must be between 0 and 1")
        return errors


@dataclass
class SMCConfig:
    """SMC/ICT strategy configuration with defaults."""

    session_start: str = "00:00"
    session_end: str = "08:00"
    atr_period: int = 14
    atr_buffer_mult: float = 0.5
    ifvg_atr_mult: float = 1.2
    ifvg_proximity_mult: float = 1.5
    risk_per_trade: float = 0.01
    slippage_buffer: float = 0.1
    target_1r: float = 1.0
    target_2r: float = 2.0
    target_final: float = 2.5
    daily_loss_limit: float = 0.03
    max_trades_per_day: int = 3
    require_volume_confirmation: bool = True
    require_mss_confirmation: bool = True


# Default configuration dictionaries
DEFAULT_DATA_CONFIG = asdict(DataConfig())
DEFAULT_BACKTEST_CONFIG = asdict(BacktestConfig())
DEFAULT_STRATEGY_CONFIG = asdict(StrategyConfig())
DEFAULT_OUTPUT_CONFIG = asdict(OutputConfig())
DEFAULT_PATTERN_SELECTION_CONFIG = asdict(PatternSelectionConfig())
DEFAULT_SMC_CONFIG = asdict(SMCConfig())


# ============================================================
# Configuration Validation and Merging
# ============================================================


def validate_config(config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration dictionary against defaults.

    Args:
        config: User-provided configuration dictionary
        defaults: Default configuration dictionary

    Returns:
        Validated and merged configuration
    """
    validated = deepcopy(defaults)

    for key, value in config.items():
        if key in defaults:
            if isinstance(value, dict) and isinstance(defaults[key], dict):
                # Recursively merge nested dicts
                validated[key] = {**defaults[key], **value}
            else:
                validated[key] = value
        else:
            # Allow additional keys with warning
            validated[key] = value

    return validated


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.

    Later configs override earlier ones.

    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        Merged configuration dictionary
    """
    result: Dict[str, Any] = {}
    for config in configs:
        result = deep_merge(result, config)
    return result


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Dictionary to merge into base

    Returns:
        Merged dictionary
    """
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def create_config(
    data: Optional[Dict[str, Any]] = None,
    backtest: Optional[Dict[str, Any]] = None,
    strategy: Optional[Dict[str, Any]] = None,
    output: Optional[Dict[str, Any]] = None,
    pattern_selection: Optional[Dict[str, Any]] = None,
    smc: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Create a complete configuration dictionary with defaults.

    Args:
        data: Data configuration overrides
        backtest: Backtest configuration overrides
        strategy: Strategy configuration overrides
        output: Output configuration overrides
        pattern_selection: Pattern selection configuration overrides
        smc: SMC strategy configuration overrides
        **kwargs: Additional configuration sections

    Returns:
        Complete configuration dictionary
    """
    config = {
        "data": validate_config(data or {}, DEFAULT_DATA_CONFIG),
        "backtest": validate_config(backtest or {}, DEFAULT_BACKTEST_CONFIG),
        "strategy": validate_config(strategy or {}, DEFAULT_STRATEGY_CONFIG),
        "output": validate_config(output or {}, DEFAULT_OUTPUT_CONFIG),
    }

    if pattern_selection:
        config["pattern_selection"] = validate_config(
            pattern_selection, DEFAULT_PATTERN_SELECTION_CONFIG
        )

    if smc:
        config["smc"] = validate_config(smc, DEFAULT_SMC_CONFIG)

    # Add any additional sections
    for key, value in kwargs.items():
        config[key] = value

    return config


def print_config(config: Dict[str, Any], title: str = "Configuration") -> None:
    """
    Print configuration in formatted manner.

    Args:
        config: Configuration dictionary to print
        title: Title for the configuration display
    """
    print("=" * 60)
    print(f"📋 {title}")
    print("=" * 60)

    for section, values in config.items():
        if isinstance(values, dict):
            print(f"\n{section.upper()}:")
            for key, value in values.items():
                print(f"  {key}: {value}")
        else:
            print(f"\n{section}: {values}")

    print("=" * 60)


# ============================================================
# Path and Environment Setup
# ============================================================


def setup_project_root() -> Path:
    """
    Add project root to sys.path and return Path object.

    Returns:
        Path object pointing to project root
    """
    # Try different possible project root locations
    candidates: list[Path] = [
        Path("..").resolve(),
        Path(".").resolve(),
    ]
    try:
        candidates.append(Path(__file__).parent.parent.parent)
    except NameError:
        pass

    for root in candidates:
        if (root / "src").exists():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root

    # Fallback to current directory
    project_root = Path(".").resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def get_data_path(config: Dict[str, Any], project_root: Path) -> Path:
    """
    Get full data path from configuration.

    Args:
        config: Configuration dictionary
        project_root: Project root path

    Returns:
        Full path to data file
    """
    data_config = config.get("data", {})
    directory: str = data_config.get("directory", "data/raw")
    file: str = data_config.get("file", "SPY_daily.csv")

    return project_root / directory / file


def get_output_path(
    config: Dict[str, Any], project_root: Path, filename: Optional[str] = None
) -> Path:
    """
    Get full output path from configuration.

    Creates the output directory if it doesn't exist.

    Args:
        config: Configuration dictionary
        project_root: Project root path
        filename: Optional filename to append

    Returns:
        Full output path
    """
    output_config = config.get("output", {})
    directory: str = output_config.get("directory", "reports")
    subdirectory: Optional[str] = output_config.get("subdirectory")

    path: Path = project_root / directory
    if subdirectory:
        path = path / subdirectory

    # Ensure directory exists before saving
    path.mkdir(parents=True, exist_ok=True)

    if filename:
        path = path / filename

    return path


# ============================================================
# Data Loading and Validation
# ============================================================


def load_price_data(config: Dict[str, Any], project_root: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and prepare OHLCV data with optional filtering.

    Args:
        config: Configuration dictionary with 'data' section
        project_root: Optional project root path

    Returns:
        DataFrame with OHLCV data
    """
    if project_root is None:
        project_root = setup_project_root()

    data_config = config.get("data", DEFAULT_DATA_CONFIG)

    # Build full path
    data_path = get_data_path(config, project_root)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Apply column mapping
    column_mapping = data_config.get("column_mapping", {})
    df = df.rename(columns=column_mapping)

    # Ensure proper column casing
    df.columns = [c.capitalize() for c in df.columns]

    # Select required columns
    columns = data_config.get("columns", ["Open", "High", "Low", "Close", "Volume"])
    available_cols = [c for c in columns if c in df.columns]
    df = df[available_cols]

    # Add Volume if missing
    if "Volume" not in df.columns:
        df["Volume"] = 0

    # Apply date filters
    start_date = data_config.get("start_date")
    end_date = data_config.get("end_date")

    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    return df


def validate_ohlcv_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame has required OHLCV columns.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid, raises ValueError otherwise
    """
    required = ["Open", "High", "Low", "Close"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    return True


# ============================================================
# Data Summary and Display
# ============================================================


def print_data_summary(
    df: pd.DataFrame, title: str = "Data Summary", show_sample: bool = True, show_stats: bool = True
) -> None:
    """
    Print standardized data summary.

    Args:
        df: DataFrame to summarize
        title: Title for the summary
        show_sample: Whether to show first rows
        show_stats: Whether to show price statistics
    """
    print("=" * 60)
    print(f"📊 {title}")
    print("=" * 60)

    print(f"Date Range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Total Trading Days: {len(df):,}")
    print(f"Years of Data: {(df.index.max() - df.index.min()).days / 365.25:.1f}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData shape: {df.shape}")

    if show_sample:
        print("\nFirst 5 rows:")
        print(df.head().to_string())

    if show_stats:
        price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
        if price_cols:
            print("\n📈 Price Statistics:")
            print(df[price_cols].describe().round(2).to_string())

    print("=" * 60)


def print_backtest_summary(stats: Dict[str, Any], title: str = "Backtest Results") -> None:
    """
    Print formatted backtest statistics.

    Args:
        stats: Dictionary of backtest statistics
        title: Title for the summary
    """
    print("=" * 60)
    print(f"📈 {title}")
    print("=" * 60)

    # Returns section
    print("\nReturns:")
    print(f"  Total Return: {stats.get('Return [%]', 0):.2f}%")
    print(f"  Buy & Hold Return: {stats.get('Buy & Hold Return [%]', 0):.2f}%")
    print(f"  Annual Return: {stats.get('Return (Ann.) [%]', 0):.2f}%")

    # Risk metrics section
    print("\nRisk Metrics:")
    print(f"  Max Drawdown: {stats.get('Max. Drawdown [%]', 0):.2f}%")
    print(f"  Sharpe Ratio: {stats.get('Sharpe Ratio', 0):.2f}")
    print(f"  Sortino Ratio: {stats.get('Sortino Ratio', 0):.2f}")
    print(f"  Calmar Ratio: {stats.get('Calmar Ratio', 0):.2f}")

    # Trade statistics section
    print("\nTrade Statistics:")
    print(f"  Total Trades: {stats.get('# Trades', 0)}")
    print(f"  Win Rate: {stats.get('Win Rate [%]', 0):.2f}%")
    print(f"  Best Trade: {stats.get('Best Trade [%]', 0):.2f}%")
    print(f"  Worst Trade: {stats.get('Worst Trade [%]', 0):.2f}%")
    print(f"  Avg Trade: {stats.get('Avg. Trade [%]', 0):.2f}%")
    print(f"  Max Trade Duration: {stats.get('Max. Trade Duration', 'N/A')}")
    print(f"  Avg Trade Duration: {stats.get('Avg. Trade Duration', 'N/A')}")

    print("=" * 60)


def print_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]], title: str = "Performance Comparison"
) -> None:
    """
    Print metrics comparison table.

    Args:
        metrics_dict: Dictionary of {strategy_name: metrics}
        title: Title for the comparison
    """
    print("\n" + "=" * 60)
    print(f"📊 {title}")
    print("=" * 60)

    # Convert to DataFrame for nice display
    df = pd.DataFrame(metrics_dict).T

    # Format numeric columns
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].round(2)

    print(df.to_string())
    print("=" * 60)


# ============================================================
# Backtest Runner Helpers
# ============================================================


def create_runner_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract backtest runner keyword arguments from config.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of kwargs for BacktestPyRunner
    """
    backtest_config = config.get("backtest", DEFAULT_BACKTEST_CONFIG)

    return {
        "cash": backtest_config.get("initial_equity", 100000),
        "commission": backtest_config.get("commission", 0.001),
        "exclusive_orders": backtest_config.get("exclusive_orders", True),
    }


def create_strategy_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract strategy keyword arguments from config.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of kwargs for strategy
    """
    strategy_config = config.get("strategy", DEFAULT_STRATEGY_CONFIG)

    return {
        "min_confidence": strategy_config.get("min_confidence", 0.55),
        "risk_per_trade": strategy_config.get("risk_per_trade", 0.02),
        "max_open_positions": strategy_config.get("max_open_positions", 3),
        "min_confluence_count": strategy_config.get("min_confluence_count", 2),
    }


# ============================================================
# Metrics Calculation
# ============================================================


def calculate_performance_metrics(
    equity_curve: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.

    Args:
        equity_curve: Series of equity values indexed by date
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily)

    Returns:
        Dictionary of performance metrics
    """
    returns = equity_curve.pct_change().dropna()

    # Total return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100

    # Annualized return
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    annual_return = (
        ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (365.25 / days) - 1) * 100
        if days > 0
        else 0
    )

    # Volatility
    volatility = returns.std() * np.sqrt(periods_per_year) * 100

    # Sharpe Ratio
    daily_rf = risk_free_rate / periods_per_year
    sharpe = (
        (returns.mean() - daily_rf) / returns.std() * np.sqrt(periods_per_year)
        if returns.std() > 0
        else 0
    )

    # Max Drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    # Calmar Ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    sortino = (
        (returns.mean() - daily_rf) / downside_std * np.sqrt(periods_per_year)
        if downside_std > 0
        else 0
    )

    return {
        "Total Return (%)": round(total_return, 2),
        "Annual Return (%)": round(annual_return, 2),
        "Volatility (%)": round(volatility, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Calmar Ratio": round(calmar, 2),
        "Max Drawdown (%)": round(max_drawdown, 2),
    }


def calculate_trade_statistics(trades_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate trade-level statistics.

    Args:
        trades_df: DataFrame with trade information

    Returns:
        Dictionary of trade statistics
    """
    if len(trades_df) == 0:
        return {
            "Total Trades": 0,
            "Win Rate (%)": 0,
            "Avg Win ($)": 0,
            "Avg Loss ($)": 0,
            "Profit Factor": 0,
        }

    pnl_col = "pnl" if "pnl" in trades_df.columns else "PnL"

    wins = trades_df[trades_df[pnl_col] > 0]
    losses = trades_df[trades_df[pnl_col] < 0]

    total_trades = len(trades_df)
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    avg_win = wins[pnl_col].mean() if len(wins) > 0 else 0
    avg_loss = losses[pnl_col].mean() if len(losses) > 0 else 0

    total_wins = wins[pnl_col].sum() if len(wins) > 0 else 0
    total_losses = abs(losses[pnl_col].sum()) if len(losses) > 0 else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    return {
        "Total Trades": total_trades,
        "Win Rate (%)": round(win_rate, 2),
        "Avg Win ($)": round(avg_win, 2),
        "Avg Loss ($)": round(avg_loss, 2),
        "Profit Factor": round(profit_factor, 2),
    }


# ============================================================
# Visualization Helpers
# ============================================================


def setup_plot_style(style: str = "seaborn-v0_8-darkgrid") -> None:
    """
    Setup matplotlib plot style.

    Args:
        style: Matplotlib style name
    """
    try:
        plt.style.use(style)
    except (ValueError, OSError):
        plt.style.use("seaborn-v0_8-whitegrid")

    sns.set_palette("husl")


def create_equity_curve_plot(
    equity_curve: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Equity Curve",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Create standardized equity curve plot.

    Args:
        equity_curve: Series of equity values
        benchmark: Optional benchmark equity curve
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(equity_curve.index.to_numpy(), equity_curve.to_numpy(), label="Strategy", linewidth=2)

    if benchmark is not None:
        ax.plot(
            benchmark.index.to_numpy(),
            benchmark.to_numpy(),
            label="Benchmark",
            linewidth=2,
            linestyle="--",
            alpha=0.7,
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_drawdown_plot(
    equity_curve: pd.Series, figsize: Tuple[int, int] = (14, 4), save_path: Optional[str] = None
) -> Figure:
    """
    Create drawdown visualization.

    Args:
        equity_curve: Series of equity values
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max * 100

    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)
    ax.set_title("Drawdown", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_returns_distribution_plot(
    returns: pd.Series,
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 50,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Create returns distribution histogram.

    Args:
        returns: Series of returns
        figsize: Figure size
        bins: Number of histogram bins
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(returns * 100, bins=bins, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="red", linestyle="--", label="Break-even")
    ax.set_title("Returns Distribution", fontsize=14)
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_rolling_sharpe_plot(
    returns: pd.Series,
    window: int = 252,
    figsize: Tuple[int, int] = (14, 4),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Create rolling Sharpe ratio plot.

    Args:
        returns: Series of returns
        window: Rolling window period
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    rolling_sharpe = (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(252)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
    ax.axhline(y=0, color="red", linestyle="-", alpha=0.3)
    ax.axhline(y=1, color="green", linestyle="--", alpha=0.3, label="Sharpe = 1")
    ax.set_title(f"Rolling {window}-day Sharpe Ratio", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_metrics_comparison_chart(
    metrics_dict: Dict[str, Dict[str, float]],
    metrics_to_compare: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Create bar chart comparing metrics across strategies.

    Args:
        metrics_dict: Dictionary of {strategy_name: metrics}
        metrics_to_compare: List of metric names to compare
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    if metrics_to_compare is None:
        metrics_to_compare = ["Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)"]

    df = pd.DataFrame(metrics_dict).T
    df = df[[m for m in metrics_to_compare if m in df.columns]]

    fig, ax = plt.subplots(figsize=figsize)

    df.plot(kind="bar", ax=ax)
    ax.set_title("Strategy Comparison", fontsize=14)
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Value")
    ax.legend(title="Metric")
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ============================================================
# Pattern Analysis Helpers
# ============================================================


def print_pattern_summary(pattern_name: str, stats: Dict[str, Any], passed: bool = True) -> None:
    """
    Print pattern analysis summary.

    Args:
        pattern_name: Name of the pattern
        stats: Dictionary of pattern statistics
        passed: Whether pattern passed filtering
    """
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {pattern_name}")
    print(f"    Trades: {stats.get('trades', 0)}")
    print(f"    Sharpe: {stats.get('sharpe', 0):.3f}")
    print(f"    Return: {stats.get('return', 0):.2%}")
    print(f"    Win Rate: {stats.get('win_rate', 0):.1%}")


def create_pattern_performance_chart(
    performance_df: pd.DataFrame,
    metric: str = "sharpe_ratio",
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Create horizontal bar chart of pattern performance.

    Args:
        performance_df: DataFrame with pattern performance metrics
        metric: Column name for sorting
        top_n: Number of top patterns to show
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    if metric not in performance_df.columns:
        metric = performance_df.columns[0]

    df_sorted = performance_df.nlargest(top_n, metric)

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["green" if v >= 0 else "red" for v in df_sorted[metric]]

    ax.barh(range(len(df_sorted)), df_sorted[metric], color=colors, alpha=0.7)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted.index)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Top {top_n} Patterns by {metric.replace('_', ' ').title()}")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ============================================================
# Reporting Functions
# ============================================================


def save_config(config: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(config, f, indent=2, default=str)


def load_config(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        filepath: Input file path

    Returns:
        Configuration dictionary
    """
    with open(filepath, "r") as f:
        result: Dict[str, Any] = json.load(f)
        return result


def create_report_header(title: str, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Create markdown report header.

    Args:
        title: Report title
        config: Optional configuration to include

    Returns:
        Markdown formatted header string
    """
    from datetime import datetime

    lines = [
        f"# {title}",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    if config:
        lines.extend(
            [
                "## Configuration",
                "",
                "```json",
                json.dumps(config, indent=2, default=str),
                "```",
                "",
            ]
        )

    return "\n".join(lines)


# ============================================================
# Convenience Exports
# ============================================================

# Default notebook configuration template
DEFAULT_NOTEBOOK_CONFIG = create_config()

# Commonly used imports for notebooks
NOTEBOOK_IMPORTS = """
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.notebook_helpers import (
    setup_project_root,
    create_config,
    load_price_data,
    print_data_summary,
    print_backtest_summary,
    print_config,
    calculate_performance_metrics,
    calculate_trade_statistics,
    create_equity_curve_plot,
    create_drawdown_plot,
    create_returns_distribution_plot,
)
"""
