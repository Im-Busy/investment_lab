"""Per-instrument configuration, strategy dispatch, and performance tracking.

Phase 25 — Single source of truth for per-instrument tuned params,
timezone/session-aware strategy dispatch, and append-only performance logging.
"""

from src.per_instrument.instrument_config import (
    INSTRUMENT_CONFIG,
    INSTRUMENT_STRATEGY,
    PRODUCTION_BASKET,
    get_config_for,
    get_allocation,
)
from src.per_instrument.timezone_registry import (
    TIMEZONE_SESSIONS,
    get_session_for,
    is_in_trading_window,
)
from src.per_instrument.performance_tracker import (
    log_performance,
    query_log,
    get_latest_for,
    compare_across_runs,
    PerformanceRecord,
)
from src.per_instrument.strategy_selector import (
    get_strategy_class,
    get_backtest_command,
    should_trade_now,
)

__all__ = [
    "INSTRUMENT_CONFIG",
    "INSTRUMENT_STRATEGY",
    "PRODUCTION_BASKET",
    "get_config_for",
    "get_allocation",
    "TIMEZONE_SESSIONS",
    "get_session_for",
    "is_in_trading_window",
    "log_performance",
    "query_log",
    "get_latest_for",
    "compare_across_runs",
    "PerformanceRecord",
    "get_strategy_class",
    "get_backtest_command",
    "should_trade_now",
]
