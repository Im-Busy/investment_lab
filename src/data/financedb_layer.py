"""P28-6: FinanceDatabase as symbol universe layer.

Cached wrapper around FinanceDatabase providing 300K+ symbols (equities/ETFs/
funds/indices/crypto/currencies) with GICS categorization. Serves as the
single source of truth for symbol universes across all backtests.

Source: FinanceDatabase (JerBouma, 7.5K stars) — 1 pip dep.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _load_equities() -> pd.DataFrame:
    import financedatabase as fd

    eq = fd.Equities()
    logger.info("Loaded %d equities from FinanceDatabase", len(eq.data))
    return eq.data


@functools.lru_cache(maxsize=1)
def _load_etfs() -> pd.DataFrame:
    import financedatabase as fd

    etf = fd.ETFs()
    logger.info("Loaded %d ETFs from FinanceDatabase", len(etf.data))
    return etf.data


@functools.lru_cache(maxsize=1)
def _load_indices() -> pd.DataFrame:
    import financedatabase as fd

    idx = fd.Indices()
    logger.info("Loaded %d indices from FinanceDatabase", len(idx.data))
    return idx.data


def get_universe(
    country: str | None = None,
    sector: str | None = None,
    industry: str | None = None,
    exchange: str | None = None,
    market_cap: str | None = None,
    asset_type: str = "equities",
) -> list[str]:
    """Get filtered symbol universe by criteria.

    Args:
        country: e.g., "United States", "Canada".
        sector: e.g., "Information Technology", "Financials".
        industry: e.g., "Pharmaceuticals", "Semiconductors & Semiconductor Equipment".
        exchange: e.g., "NYQ", "NMS", "TOR".
        market_cap: e.g., "Large Cap", "Mid Cap".
        asset_type: "equities" | "etfs" | "indices".

    Returns:
        List of symbol strings.

    Example:
        >>> us_large_tech = get_universe(
        ...     country="United States",
        ...     sector="Information Technology",
        ...     market_cap="Large Cap",
        ... )
    """
    loaders = {
        "equities": _load_equities,
        "etfs": _load_etfs,
        "indices": _load_indices,
    }
    if asset_type not in loaders:
        raise ValueError(f"Unknown asset_type: {asset_type}. Options: {list(loaders)}")

    df = loaders[asset_type]()
    mask = pd.Series(True, index=df.index)

    filters: dict[str, str | None] = {
        "country": country,
        "sector": sector,
        "industry": industry,
        "exchange": exchange,
        "market_cap": market_cap,
    }
    for col, val in filters.items():
        if val is not None and col in df.columns:
            mask &= df[col] == val

    symbols = sorted(df[mask].index.tolist())
    logger.info(
        "get_universe(%s): %d symbols after filters",
        asset_type,
        len(symbols),
    )
    return symbols


def available_filter_values(asset_type: str = "equities") -> dict[str, list[str]]:
    """Return available filter categories and their values.

    Useful for CLI auto-completion and parameter discovery.

    Args:
        asset_type: "equities" | "etfs" | "indices".

    Returns:
        Dict of {column: sorted_unique_values} for categorical columns.
    """
    loaders = {
        "equities": _load_equities,
        "etfs": _load_etfs,
        "indices": _load_indices,
    }
    if asset_type not in loaders:
        raise ValueError(f"Unknown asset_type: {asset_type}")

    df = loaders[asset_type]()
    out: dict[str, list[str]] = {}
    for col in df.columns:
        vals = df[col].dropna().unique()
        if 0 < len(vals) < 500:
            out[col] = sorted(vals.astype(str).tolist())
    return out


def search_symbols(query: str, asset_type: str = "equities") -> pd.DataFrame:
    """Search for symbols by name or ticker.

    Args:
        query: Name fragment or ticker to search for.
        asset_type: "equities" | "etfs" | "indices".

    Returns:
        DataFrame of matching symbols with metadata.
    """
    import financedatabase as fd

    if asset_type == "equities":
        db = fd.Equities()
    elif asset_type == "etfs":
        db = fd.ETFs()
    elif asset_type == "indices":
        db = fd.Indices()
    else:
        raise ValueError(f"Unknown asset_type: {asset_type}")

    query_lower = query.lower()
    mask = db.data.index.str.lower().str.startswith(query_lower)
    mask |= db.data["name"].str.lower().str.contains(query_lower, na=False)
    return db.data[mask]


def count_by_sector(country: str = "United States") -> pd.Series:
    """Count equities per GICS sector for a given country.

    Args:
        country: Country to filter.

    Returns:
        Series with sector names as index and symbol counts as values.
    """
    df = _load_equities()
    mask = df["country"] == country
    return df[mask]["sector"].value_counts().sort_values(ascending=False)
