"""P28-8: US stock symbols auto-update (weekly cron pattern).

Downloads symbol lists from rreichel3/US-Stock-Symbols GitHub repo, maintains
a local cache with timestamps, and provides diff-based syncing to detect new
and removed tickers between runs.

Source: FinanceDatabase FAQ — recommended symbol source for US equities.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import pandas as pd

logger = logging.getLogger(__name__)

SYMBOL_SOURCE_URLS: dict[str, str] = {
    "nyse": "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_full.json",
    "nasdaq": "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_full.json",
    "amex": "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/amex/amex_full.json",
}

DEFAULT_CACHE_DIR = Path("data/symbol_cache")
META_FILE = "sync_meta.json"


@dataclass
class SyncMeta:
    """Metadata for a symbol sync run."""

    last_sync: str = ""
    source_count: int = 0
    ticker_count: int = 0
    sources: dict[str, str] = field(default_factory=dict)
    """Exchange → url synced."""


@dataclass
class SyncResult:
    """Result of a symbol sync operation."""

    tickers: list[str]
    """All tickers merged and sorted."""

    by_exchange: dict[str, list[str]]
    """Exchange → ticker list."""

    meta: SyncMeta = field(default_factory=SyncMeta)


def _load_json_url(url: str) -> list[dict]:
    with urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array from {url}")
    return data


def fetch_all_us_symbols(
    exchanges: Optional[list[str]] = None,
) -> SyncResult:
    """Download US stock symbols from rreichel3/US-Stock-Symbols.

    Args:
        exchanges: Which exchanges to fetch (default: all — nyse, nasdaq, amex).

    Returns:
        SyncResult with merged ticker list and per-exchange breakdowns.
    """
    targets = {k: v for k, v in SYMBOL_SOURCE_URLS.items()}
    if exchanges:
        targets = {k: v for k, v in targets.items() if k in exchanges}

    by_exchange: dict[str, list[str]] = {}
    all_tickers: set[str] = set()
    source_meta: dict[str, str] = {}

    for exchange, url in targets.items():
        try:
            records = _load_json_url(url)
            tickers = sorted(
                {r.get("Symbol", r.get("symbol", "")).upper() for r in records if r.get("Symbol")}
            )
            tickers = [t for t in tickers if t]
            by_exchange[exchange] = tickers
            all_tickers.update(tickers)
            source_meta[exchange] = url
            logger.info("Fetched %d symbols from %s", len(tickers), exchange)
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", exchange, e)
            by_exchange[exchange] = []

    sorted_all = sorted(all_tickers)
    logger.info(
        "Fetched %d total unique symbols across %d exchanges", len(sorted_all), len(by_exchange)
    )
    return SyncResult(
        tickers=sorted_all,
        by_exchange=by_exchange,
        meta=SyncMeta(
            last_sync=datetime.now(timezone.utc).isoformat(),
            source_count=len(targets),
            ticker_count=len(sorted_all),
            sources=source_meta,
        ),
    )


def diff_symbols(
    current: list[str],
    previous: list[str],
) -> dict[str, list[str]]:
    """Compute added and removed symbols between two snapshots.

    Args:
        current: Latest symbol list.
        previous: Previous symbol list.

    Returns:
        Dict with 'added' and 'removed' keys.
    """
    cur_set = set(current)
    prev_set = set(previous)
    return {
        "added": sorted(cur_set - prev_set),
        "removed": sorted(prev_set - cur_set),
    }


def save_cache(result: SyncResult, cache_dir: Optional[Path] = None) -> Path:
    """Persist sync result to local cache directory.

    Args:
        result: SyncResult from fetch_all_us_symbols().
        cache_dir: Cache directory (default: data/symbol_cache/).

    Returns:
        Path to cache directory.
    """
    cdir = cache_dir or DEFAULT_CACHE_DIR
    cdir.mkdir(parents=True, exist_ok=True)

    all_file = cdir / "all_us_symbols.json"
    with open(all_file, "w") as f:
        json.dump(
            {
                "tickers": result.tickers,
                "count": len(result.tickers),
                "updated": result.meta.last_sync,
                "by_exchange": result.by_exchange,
            },
            f,
            indent=2,
        )

    meta_file = cdir / META_FILE
    with open(meta_file, "w") as f:
        json.dump(
            {
                "last_sync": result.meta.last_sync,
                "source_count": result.meta.source_count,
                "ticker_count": result.meta.ticker_count,
                "sources": result.meta.sources,
            },
            f,
            indent=2,
        )

    logger.info("Cached %d symbols to %s", len(result.tickers), cdir)
    return cdir


def load_cache(cache_dir: Optional[Path] = None) -> SyncResult | None:
    """Load previously cached symbol list.

    Args:
        cache_dir: Cache directory (default: data/symbol_cache/).

    Returns:
        SyncResult if cache exists, None otherwise.
    """
    cdir = cache_dir or DEFAULT_CACHE_DIR
    all_file = cdir / "all_us_symbols.json"
    if not all_file.exists():
        return None
    with open(all_file) as f:
        data = json.load(f)
    by_exchange = data.get("by_exchange", {})
    return SyncResult(
        tickers=data.get("tickers", []),
        by_exchange=by_exchange,
        meta=SyncMeta(
            last_sync=data.get("updated", ""),
            ticker_count=data.get("count", 0),
            sources={},
        ),
    )


def sync_with_diff(
    exchanges: Optional[list[str]] = None,
    cache_dir: Optional[Path] = None,
) -> tuple[SyncResult, dict[str, list[str]]]:
    """Fetch latest symbols, compare against cache, save, return diff.

    Args:
        exchanges: Exchanges to fetch.
        cache_dir: Cache directory.

    Returns:
        Tuple of (SyncResult, diff dict with 'added'/'removed').
    """
    prev = load_cache(cache_dir)
    current = fetch_all_us_symbols(exchanges)
    save_cache(current, cache_dir)
    if prev:
        diff = diff_symbols(current.tickers, prev.tickers)
        if diff["added"] or diff["removed"]:
            logger.info(
                "Symbol diff: +%d added, -%d removed",
                len(diff["added"]),
                len(diff["removed"]),
            )
    else:
        diff = {"added": [], "removed": []}
    return current, diff
