"""Single source of truth for per-instrument production configuration.

Phase 25 — Consolidates the duplicated PER_INSTRUMENT_BEST dicts from
backtest_rules_batch.py and backtest_all_comprehensive.py into one registry.

Tier assignments from comprehensive 108-instrument backtest, 2026-05-21:
  Tier S — Primary basket (Energy, ETFs, Indices, Commodities)
  Tier A — Secondary (Select single stocks, China industrial)
  Tier B — Selective (Turnaround plays with strong OOS)
  Tier C — Watch only (Utilities, consumer defensive)
  Tier F — EXCLUDED (Financials, HK, Airlines, REITs, MicroCap, Bonds, Forex)
"""

from __future__ import annotations

from typing import Any

# Per-instrument tuned params from tuning sweep (2026-05-20):
#   et = entry_threshold, mr = min_reliability, tsa = trail_stop_atr, cb = confluence_bonus
#   alloc = portfolio allocation weight (sums to 1.0 across production basket)
INSTRUMENT_CONFIG: dict[str, dict[str, Any]] = {
    # ── Tier S — Primary Basket ──────────────────────────────────────────
    "SPY": {"et": 0.50, "mr": 0.70, "tsa": 4.0, "cb": 0.05, "tier": "S", "alloc": 0.14},
    "XLK": {"et": 0.55, "mr": 0.70, "tsa": 4.0, "cb": 0.10, "tier": "S", "alloc": 0.10},
    "XLE": {"et": 0.60, "mr": 0.70, "tsa": 4.0, "cb": 0.05, "tier": "S", "alloc": 0.10},
    "GLD": {"et": 0.50, "mr": 0.70, "tsa": 2.0, "cb": 0.10, "tier": "S", "alloc": 0.10},
    "SLV": {"et": 0.50, "mr": 0.70, "tsa": 2.0, "cb": 0.10, "tier": "S", "alloc": 0.05},
    "EEM": {"et": 0.55, "mr": 0.70, "tsa": 4.0, "cb": 0.10, "tier": "S", "alloc": 0.05},
    "QQQ": {"et": 0.60, "mr": 0.40, "tsa": 4.0, "cb": 0.10, "tier": "S", "alloc": 0.08},
    "MPC": {"et": 0.55, "mr": 0.70, "tsa": 4.0, "cb": 0.10, "tier": "S", "alloc": 0.04},
    "HAL": {"et": 0.55, "mr": 0.70, "tsa": 4.0, "cb": 0.10, "tier": "S", "alloc": 0.04},
    "EOG": {"et": 0.55, "mr": 0.70, "tsa": 4.0, "cb": 0.10, "tier": "S", "alloc": 0.04},
    "PSX": {"et": 0.55, "mr": 0.70, "tsa": 4.0, "cb": 0.10, "tier": "S", "alloc": 0.03},
    "XOM": {"et": 0.70, "mr": 0.70, "tsa": 2.0, "cb": 0.05, "tier": "S", "alloc": 0.03},
    # ── Tier A — Secondary Basket ────────────────────────────────────────
    "JNJ": {"et": 0.55, "mr": 0.70, "tsa": 4.0, "cb": 0.10, "tier": "A", "alloc": 0.04},
    "NUE": {"et": 0.55, "mr": 0.70, "tsa": 4.0, "cb": 0.10, "tier": "A", "alloc": 0.03},
    "STLD": {"et": 0.55, "mr": 0.70, "tsa": 4.0, "cb": 0.10, "tier": "A", "alloc": 0.03},
    "CN_CATL": {"et": 0.55, "mr": 0.70, "tsa": 4.0, "cb": 0.10, "tier": "A", "alloc": 0.02},
    "XLV": {"et": 0.60, "mr": 0.70, "tsa": 4.0, "cb": 0.05, "tier": "A", "alloc": 0.02},
    # ── Tier B — Selective B ─────────────────────────────────────────────
    "INTC": {"et": 0.55, "mr": 0.70, "tsa": 4.0, "cb": 0.10, "tier": "B", "alloc": 0.02},
    "AMD": {"et": 0.55, "mr": 0.70, "tsa": 4.0, "cb": 0.10, "tier": "B", "alloc": 0.02},
    "LMT": {"et": 0.55, "mr": 0.70, "tsa": 4.0, "cb": 0.10, "tier": "B", "alloc": 0.02},
    "REGN": {"et": 0.55, "mr": 0.70, "tsa": 4.0, "cb": 0.10, "tier": "B", "alloc": 0.01},
    "MRK": {"et": 0.55, "mr": 0.70, "tsa": 4.0, "cb": 0.10, "tier": "B", "alloc": 0.01},
    # ── Tier F — EXCLUDED (documented for completeness) ─────────────────
    # "XLF":    {"et": 0.70, "mr": 0.70, "tsa": 2.0, "cb": 0.10, "tier": "F", "alloc": 0.0},
    # "JPM":    {"et": 0.40, "mr": 0.70, "tsa": 2.0, "cb": 0.10, "tier": "F", "alloc": 0.0},
    # "BAC":    {"et": 0.55, "mr": 0.70, "tsa": 4.0, "cb": 0.10, "tier": "F", "alloc": 0.0},
    # "GS":     {"et": 0.55, "mr": 0.70, "tsa": 4.0, "cb": 0.10, "tier": "F", "alloc": 0.0},
    # "WFC":    {"et": 0.55, "mr": 0.70, "tsa": 4.0, "cb": 0.10, "tier": "F", "alloc": 0.0},
    # "TLT":    {"et": 0.40, "mr": 0.70, "tsa": 2.0, "cb": 0.05, "tier": "F", "alloc": 0.0},
    # "EURUSD_X": {"et": 0.40, "mr": 0.70, "tsa": 2.0, "cb": 0.05, "tier": "F", "alloc": 0.0},
    # "BTC_USD": {"et": 0.40, "mr": 0.70, "tsa": 4.0, "cb": 0.05, "tier": "F", "alloc": 0.0},
    # "IWM":    {"et": 0.70, "mr": 0.55, "tsa": 2.0, "cb": 0.05, "tier": "F", "alloc": 0.0},
}

INSTRUMENT_STRATEGY: dict[str, str] = {
    # Default for all instruments is "rules_first".
    # Override for instruments requiring a different strategy.
    "BTC_USD": "smc",  # SMC hourly — needs recalibrated crypto registry
    "ETH_USD": "smc",  # SMC hourly
    "NQ=F": "smc",  # SMC hourly futures
    "GC=F": "smc",  # SMC hourly futures
}

PRODUCTION_BASKET: list[str] = [
    s for s, c in INSTRUMENT_CONFIG.items() if c.get("tier", "F") in ("S", "A", "B")
]

FALLBACK_CONFIG: dict[str, Any] = {
    "et": 0.60,
    "mr": 0.70,
    "tsa": 4.0,
    "cb": 0.10,
    "tier": "B",
    "alloc": 0.02,
}


def get_config_for(symbol: str, use_best: bool = False) -> dict[str, Any]:
    """Return the production config for an instrument.

    Merges fixed defaults (multi-TP, quality registry, gates OFF) with
    per-instrument tuned params. Falls back to FALLBACK_CONFIG if symbol
    is not in the registry.

    Args:
        symbol: Ticker symbol (e.g. "SPY", "XLK").
        use_best: If True, uses per-instrument tuned params.
                   If False, returns universal defaults (et=0.55, mr=0.70, tsa=3.0).

    Returns:
        Dict of kwargs suitable for passing to RulesFirstStrategy or run_single().
    """
    base: dict[str, Any] = dict(
        entry_threshold=0.55,
        exit_threshold=0.30,
        trail_stop_atr=3.0,
        min_reliability=0.70,
        confluence_bonus=0.10,
        volume_confirm=True,
        use_multi_tp=True,
        use_quality_registry=True,
        quality_registry_path="reports/pattern_gate/all_patterns.json",
        use_vix_gate=False,
        use_yield_curve_gate=False,
        use_short=False,
    )

    if use_best and symbol in INSTRUMENT_CONFIG:
        override = INSTRUMENT_CONFIG[symbol]
        if "et" in override:
            base["entry_threshold"] = override["et"]
        if "mr" in override:
            base["min_reliability"] = override["mr"]
        if "tsa" in override:
            base["trail_stop_atr"] = override["tsa"]
        if "cb" in override:
            base["confluence_bonus"] = override["cb"]

    return base


def get_allocation(symbol: str) -> float:
    """Return portfolio allocation weight for an instrument (0.0–1.0)."""
    if symbol in INSTRUMENT_CONFIG:
        return float(INSTRUMENT_CONFIG[symbol].get("alloc", 0.0))
    return 0.0
