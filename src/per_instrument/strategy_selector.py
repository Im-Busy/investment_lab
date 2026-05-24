"""Dynamic strategy dispatch per instrument.

Phase 25 — Routes each instrument to the appropriate strategy class
based on asset class, timezone, and OOS-validated performance.

Decision logic:
  - US equities / ETFs / stocks (daily)  → RulesFirstStrategy
  - Crypto (daily)                        → RulesFirstStrategy (needs recalibrated registry)
  - Crypto/Futures (hourly)               → SMCStrategy
  - China A-shares (daily)                → RulesFirstStrategy
  - Forex (hourly)                        → SKIP (confirmed broken, 0% WR OOS)
"""

from __future__ import annotations

from datetime import datetime

from src.per_instrument.instrument_config import (
    INSTRUMENT_CONFIG,
    INSTRUMENT_STRATEGY,
    get_config_for,
)


def get_strategy_class(instrument: str) -> type:
    """Return the best strategy class for an instrument.

    Returns the strategy class object (not an instance).
    Falls back to RulesFirstStrategy for unknown instruments.
    """
    strategy_type = INSTRUMENT_STRATEGY.get(instrument, "rules_first")

    if strategy_type == "smc":
        from src.strategies.smc_strategy import SMCStrategy  # noqa: PLC0415

        return SMCStrategy
    # Default: rules_first
    from src.strategies.rules_first_strategy import RulesFirstStrategy  # noqa: PLC0415

    return RulesFirstStrategy


def get_backtest_command(
    instrument: str, start: str = "2025-01-01", end: str | None = None
) -> str | None:
    """Return the recommended backtest CLI command for an instrument.

    Returns None for instruments that should not be traded (Tier F).
    """
    cfg = INSTRUMENT_CONFIG.get(instrument)
    if cfg and cfg.get("tier") == "F":
        return None

    config = get_config_for(instrument, use_best=True)
    strategy_type = INSTRUMENT_STRATEGY.get(instrument, "rules_first")

    et = config.get("entry_threshold", 0.55)
    mr = config.get("min_reliability", 0.70)
    tsa = config.get("trail_stop_atr", 4.0)
    cb = config.get("confluence_bonus", 0.10)

    if strategy_type == "smc":
        cmd = (
            f"uv run scripts/backtest_smc.py --symbol {instrument} --interval 1h "
            f"--start {start} --no-short --entry-threshold {et} "
            f"--trail-stop-atr {tsa} --use-multi-tp --tp1-atr 1.0"
        )
        if end:
            cmd += f" --end {end}"
        return cmd

    cmd = (
        f"uv run scripts/backtest_rules_first.py {instrument} "
        f"--start {start} --entry-threshold {et} --min-reliability {mr} "
        f"--trail-stop-atr {tsa} --confluence-bonus {cb}"
    )
    if end:
        cmd += f" --end {end}"
    return cmd


def should_trade_now(instrument: str, current_time: datetime | None = None) -> bool:
    """Check if an instrument should be traded at the current time.

    For daily strategies: always True (entry/exit at daily close).
    For intraday strategies: checks trading window via timezone_registry.

    Also checks the tier — Tier F instruments should never be traded.
    """
    cfg = INSTRUMENT_CONFIG.get(instrument)
    if cfg and cfg.get("tier") == "F":
        return False

    strategy_type = INSTRUMENT_STRATEGY.get(instrument, "rules_first")

    if strategy_type == "smc":
        from src.per_instrument.timezone_registry import is_in_trading_window  # noqa: PLC0415

        return is_in_trading_window(instrument, current_time)

    return True  # Daily strategies execute at bar close
