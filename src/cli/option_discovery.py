"""P28-4: show_options() pre-query pattern for CLI parameter discovery.

Allows users to query available filter values (sectors, exchanges, countries)
before loading the full dataset. Speeds up tuning script completions by
exposing categorical parameter domains without requiring a full data load.

Source: FinanceDatabase usage pattern — query schema before loading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OptionDomain:
    """Discovered parameter domain for a CLI flag."""

    name: str
    description: str
    values: list[str] = field(default_factory=list)
    count: int = 0
    default_value: str | None = None

    def __post_init__(self) -> None:
        if self.count == 0 and self.values:
            self.count = len(self.values)

    def as_choices(self) -> str:
        """Format as argparse choices definition for code generation."""
        return f"choices={self.values}"


@dataclass
class OptionCatalog:
    """Discovered catalog of available CLI options."""

    domains: list[OptionDomain] = field(default_factory=list)
    source_file: str = ""

    def get(self, name: str) -> OptionDomain | None:
        for d in self.domains:
            if d.name == name:
                return d
        return None

    def summary(self) -> str:
        lines = [f"OptionCatalog ({self.source_file}):", f"  {len(self.domains)} domains"]
        for d in self.domains:
            preview = ", ".join(d.values[:5])
            if len(d.values) > 5:
                preview += f", ... (+{len(d.values) - 5})"
            lines.append(f"  --{d.name}: [{preview}] ({d.count} values)")
        return "\n".join(lines)

    def to_argparse(self) -> str:
        """Generate argparse add_argument() stubs."""
        lines = []
        for d in self.domains:
            lines.append(
                f'parser.add_argument("--{d.name}", type=str, '
                f"choices={d.values}, default={repr(d.default_value)}, "
                f'help="{d.description}")'
            )
        return "\n".join(lines)


def show_options(
    domains: list[tuple[str, str, list[str]]],
    title: str = "Available Options",
) -> OptionCatalog:
    """Display available filter values without loading full dataset.

    Pre-query pattern: present all categorical parameter domains so users
    (or shell completions) can select valid values before constructing
    expensive data queries.

    Args:
        domains: List of (name, description, values) tuples.
        title: Display title for the catalog.

    Returns:
        OptionCatalog with discovered domains.

    Example:
        >>> cat = show_options([
        ...     ("sector", "GICS sector filter", ["Technology", "Energy", "Financials"]),
        ...     ("exchange", "Exchange filter", ["NYSE", "NASDAQ", "AMEX"]),
        ... ])
        >>> print(cat.summary())
    """
    catalog = OptionCatalog(source_file=title)
    for name, desc, values in domains:
        domain = OptionDomain(
            name=name,
            description=desc,
            values=sorted(values),
            default_value=values[0] if values else None,
        )
        catalog.domains.append(domain)
    return catalog


def query_financedb_schema(
    country: str | None = None,
    sector: str | None = None,
    exchange: str | None = None,
) -> dict[str, Any]:
    """Query FinanceDatabase schema to discover available filter values.

    This builds the pre-query catalog from a FinanceDatabase instance.
    Requires: pip install FinanceDatabase

    Args:
        country: Filter by country (e.g., "United States").
        sector: Filter by GICS sector.
        exchange: Filter by exchange.

    Returns:
        Dict with available values for each filter dimension.
    """
    try:
        from financedatabase import (  # type: ignore[import-untyped]
            Equities,
            ETFs,
            Indices,
        )
    except ImportError:
        logger.warning("FinanceDatabase not installed. Install via: uv add financedatabase")
        return {
            "countries": ["FinanceDatabase not installed"],
            "sectors": [],
            "exchanges": [],
            "error": "FinanceDatabase not installed",
        }
    except Exception as e:
        logger.warning("FinanceDatabase import failed: %s", e)
        return {"error": str(e)}

    try:
        equities = Equities()
        result = {
            "countries": equities.options("country"),
            "sectors": equities.options("sector"),
            "industry_groups": equities.options("industry_group"),
            "exchanges": equities.options("exchange"),
        }
        try:
            etfs = ETFs()
            result["etf_categories"] = etfs.options("category")
        except Exception:
            pass
        try:
            indices = Indices()
            result["index_currencies"] = indices.options("currency")
        except Exception:
            pass
        return result
    except Exception as e:
        logger.warning("FinanceDatabase schema query failed: %s", e)
        return {"error": str(e)}


def resolve_ticker_domain(
    ticker: str,
    universe: OptionCatalog | None = None,
) -> dict[str, str]:
    """Resolve ticker metadata: exchange, sector, industry from FinanceDatabase.

    Args:
        ticker: Ticker symbol to resolve.
        universe: Optional pre-built OptionCatalog (from show_options).

    Returns:
        Dict with exchange, sector, industry, name for the ticker.
    """
    try:
        from financedatabase import Equities  # type: ignore[import-untyped]
    except ImportError:
        return {"ticker": ticker.upper(), "error": "FinanceDatabase not installed"}

    try:
        eq = Equities()
        result = eq.select(ticker=ticker.upper())
        if result.empty:
            return {"ticker": ticker.upper(), "found": False}
        row = result.iloc[0]
        return {
            "ticker": ticker.upper(),
            "name": str(row.get("name", "")),
            "exchange": str(row.get("exchange", "")),
            "sector": str(row.get("sector", "")),
            "industry": str(row.get("industry", "")),
            "currency": str(row.get("currency", "")),
            "found": True,
        }
    except Exception as e:
        return {"ticker": ticker.upper(), "error": str(e)}
