"""P28-2: Hierarchical symbol filtering pipeline.

Chained narrowing: Country → Sector → Industry Group → Industry → Exchange →
Market. Wraps FinanceDatabase's .select() API with a fluent interface that
systematically narrows the universe without loading full datasets at each step.

Source: FinanceDatabase — GICS-categorized symbol universe with filtering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_FILTER_ORDER = [
    "country",
    "sector",
    "industry_group",
    "industry",
    "exchange",
    "market",
    "market_cap",
    "currency",
]


@dataclass
class FilterStep:
    """Single narrowing step with count before/after."""

    filter_name: str
    value: str
    count_before: int
    count_after: int

    @property
    def excluded(self) -> int:
        return self.count_before - self.count_after


@dataclass
class FilterResult:
    """Result of a hierarchical filter chain."""

    symbols: list[str] = field(default_factory=list)
    steps: list[FilterStep] = field(default_factory=list)
    total_count: int = 0

    @property
    def final_count(self) -> int:
        return len(self.symbols)

    @property
    def summary(self) -> str:
        parts = [f"{self.total_count} → {self.final_count} symbols"]
        for s in self.steps:
            if s.excluded > 0:
                parts.append(f"  {s.filter_name}={s.value}: -{s.excluded}")
        return "\n".join(parts)


def filter_hierarchical(
    equities: pd.DataFrame,
    filters: dict[str, str] | None = None,
) -> FilterResult:
    """Apply hierarchical filters to an equities DataFrame.

    Args:
        equities: DataFrame from FinanceDatabase with index=symbol.
        filters: Dict of {filter_name: value} pairs. Applied in order:
            country → sector → industry_group → industry → exchange → market.

    Returns:
        FilterResult with remaining symbols and step-by-step counts.

    Example:
        >>> import financedatabase as fd
        >>> eq = fd.Equities()
        >>> us_tech = filter_hierarchical(
        ...     eq.data,
        ...     {"country": "United States", "sector": "Information Technology"},
        ... )
        >>> print(us_tech.summary)
    """
    filters = filters or {}
    df = equities.copy()
    result = FilterResult(total_count=len(df))
    remaining = set(df.index)

    for key in _FILTER_ORDER:
        if key not in filters:
            continue
        value = filters[key]
        if key not in df.columns:
            logger.warning("Filter column %s not available, skipping", key)
            continue

        mask = df[key] == value
        before = len(remaining)
        remaining &= set(df[mask].index)
        after = len(remaining)
        result.steps.append(FilterStep(key, value, before, after))
        logger.debug("%s=%s: %d → %d", key, value, before, after)

    result.symbols = sorted(remaining)
    return result


def filter_pipeline(
    equities: pd.DataFrame,
    level: str = "country",
    columns: list[str] | None = None,
) -> dict[str, FilterResult]:
    """Precompute filter results for ALL values at a given level.

    Useful for multi-instrument backtests — pre-filter once, then iterate
    over groups without repeated `.select()` calls.

    Args:
        equities: DataFrame from FinanceDatabase.
        level: Column to group by (e.g., "sector", "country").
        columns: Columns to include in filtered subsets (default: all).

    Returns:
        Dict mapping level value → FilterResult.
    """
    if level not in equities.columns:
        raise ValueError(f"Level '{level}' not in columns: {equities.columns.tolist()}")
    columns = columns or equities.columns.tolist()
    results: dict[str, FilterResult] = {}
    for value in sorted(equities[level].dropna().unique()):
        mask = equities[level] == value
        symbols = sorted(equities[mask].index.tolist())
        results[value] = FilterResult(
            symbols=symbols,
            steps=[FilterStep(level, value, len(equities), len(symbols))],
            total_count=len(equities),
        )
    return results


def available_filters(equities: pd.DataFrame) -> dict[str, list[str]]:
    """Return available filter keys and their unique values.

    Args:
        equities: DataFrame from FinanceDatabase.

    Returns:
        Dict of {column_name: sorted_unique_values}.
    """
    out: dict[str, list[str]] = {}
    for col in equities.columns:
        vals = equities[col].dropna().unique()
        if 0 < len(vals) < 500:
            out[col] = sorted(vals.astype(str).tolist())
    return out
