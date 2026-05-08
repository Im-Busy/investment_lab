"""
Utils Module

Contains utility functions and validators.
"""

from .helpers import (
    load_data,
    save_results,
    format_signal,
    calculate_position_size,
    validate_ohlcv_data,
)
from .validators import validate_dataframe, validate_signal, validate_pattern_params
from .notebook_helpers import (
    # Configuration
    create_config,
    validate_config,
    merge_configs,
    print_config,
    # Path setup
    setup_project_root,
    get_data_path,
    get_output_path,
    # Data loading
    load_price_data,
    validate_ohlcv_dataframe,
    print_data_summary,
    # Backtest helpers
    print_backtest_summary,
    print_metrics_comparison,
    create_runner_kwargs,
    create_strategy_kwargs,
    # Metrics
    calculate_performance_metrics,
    calculate_trade_statistics,
    # Visualization
    setup_plot_style,
    create_equity_curve_plot,
    create_drawdown_plot,
    create_returns_distribution_plot,
    create_rolling_sharpe_plot,
    create_metrics_comparison_chart,
    # Pattern analysis
    print_pattern_summary,
    create_pattern_performance_chart,
    # Reporting
    save_config,
    load_config,
    create_report_header,
    # Default configs
    DEFAULT_DATA_CONFIG,
    DEFAULT_BACKTEST_CONFIG,
    DEFAULT_STRATEGY_CONFIG,
    DEFAULT_OUTPUT_CONFIG,
    DEFAULT_PATTERN_SELECTION_CONFIG,
    DEFAULT_SMC_CONFIG,
    DEFAULT_NOTEBOOK_CONFIG,
)

__all__ = [
    # Original helpers
    "load_data",
    "save_results",
    "format_signal",
    "calculate_position_size",
    "validate_ohlcv_data",
    # Validators
    "validate_dataframe",
    "validate_signal",
    "validate_pattern_params",
    # Notebook helpers - Configuration
    "create_config",
    "validate_config",
    "merge_configs",
    "print_config",
    # Notebook helpers - Path setup
    "setup_project_root",
    "get_data_path",
    "get_output_path",
    # Notebook helpers - Data loading
    "load_price_data",
    "validate_ohlcv_dataframe",
    "print_data_summary",
    # Notebook helpers - Backtest
    "print_backtest_summary",
    "print_metrics_comparison",
    "create_runner_kwargs",
    "create_strategy_kwargs",
    # Notebook helpers - Metrics
    "calculate_performance_metrics",
    "calculate_trade_statistics",
    # Notebook helpers - Visualization
    "setup_plot_style",
    "create_equity_curve_plot",
    "create_drawdown_plot",
    "create_returns_distribution_plot",
    "create_rolling_sharpe_plot",
    "create_metrics_comparison_chart",
    # Notebook helpers - Pattern analysis
    "print_pattern_summary",
    "create_pattern_performance_chart",
    # Notebook helpers - Reporting
    "save_config",
    "load_config",
    "create_report_header",
    # Default configs
    "DEFAULT_DATA_CONFIG",
    "DEFAULT_BACKTEST_CONFIG",
    "DEFAULT_STRATEGY_CONFIG",
    "DEFAULT_OUTPUT_CONFIG",
    "DEFAULT_PATTERN_SELECTION_CONFIG",
    "DEFAULT_SMC_CONFIG",
    "DEFAULT_NOTEBOOK_CONFIG",
]
