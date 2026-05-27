"""P28-5: Batch query with progress + concurrency control.

Provides batch data loading with progress callbacks and concurrent execution,
following the stock-sdk API pattern of `getAllQuotes({batchSize, concurrency,
onProgress})`. Wraps yfinance-based data loading for multi-instrument sweeps.

Source: stock-sdk (chengzuopeng) API pattern.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class BatchProgress:
    """Progress state emitted during batch loading."""

    total_symbols: int
    completed: int
    failed: int
    errors: list[str] = field(default_factory=list)
    current_symbol: str = ""
    elapsed_seconds: float = 0.0

    @property
    def progress_pct(self) -> float:
        return (self.completed + self.failed) / max(self.total_symbols, 1) * 100

    @property
    def success_pct(self) -> float:
        done = self.completed + self.failed
        return self.completed / max(done, 1) * 100 if done > 0 else 0.0


@dataclass
class BatchResult:
    """Result of a batch data load."""

    data: dict[str, Any]
    """Symbol → loaded data (DataFrame, etc.)."""

    failed: dict[str, str]
    """Symbol → error message for failures."""

    elapsed_seconds: float = 0.0

    @property
    def total(self) -> int:
        return len(self.data) + len(self.failed)


ProgressCallback = Callable[[BatchProgress], None]


def load_batch(
    symbols: list[str],
    loader_fn: Callable[[str], Any],
    *,
    concurrency: int = 5,
    on_progress: ProgressCallback | None = None,
    max_retries: int = 2,
    retry_delay: float = 1.0,
) -> BatchResult:
    """Load data for multiple symbols concurrently with progress tracking.

    Args:
        symbols: List of ticker symbols to load.
        loader_fn: Function that takes a symbol and returns loaded data.
        concurrency: Max concurrent workers.
        on_progress: Optional callback receiving BatchProgress updates.
        max_retries: Number of retry attempts per symbol on failure.
        retry_delay: Seconds between retry attempts.

    Returns:
        BatchResult with data dict and failed dict.

    Example:
        >>> from src.data.batch_loader import load_batch
        >>> import yfinance as yf
        >>> result = load_batch(
        ...     ["SPY", "QQQ", "XLK", "XLE"],
        ...     lambda sym: yf.download(sym, period="5y", progress=False),
        ...     concurrency=4,
        ... )
        >>> print(f"Loaded {len(result.data)}, failed {len(result.failed)}")
    """
    start = time.monotonic()

    def _load_with_retry(symbol: str) -> Any:
        for attempt in range(max_retries + 1):
            try:
                return loader_fn(symbol)
            except Exception as e:
                if attempt < max_retries:
                    logger.debug("Retry %d/%d for %s: %s", attempt + 1, max_retries, symbol, e)
                    time.sleep(retry_delay)
                else:
                    raise

    data: dict[str, Any] = {}
    failed: dict[str, str] = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map: dict[Future[Any], str] = {
            executor.submit(_load_with_retry, sym): sym for sym in symbols
        }

        for future in as_completed(future_map):
            sym = future_map[future]
            try:
                data[sym] = future.result()
                completed += 1
            except Exception as e:
                failed[sym] = str(e)
                logger.warning("Failed loading %s: %s", sym, e)

            elapsed = time.monotonic() - start
            progress = BatchProgress(
                total_symbols=len(symbols),
                completed=completed,
                failed=len(failed),
                errors=list(failed.values())[-5:],
                current_symbol=sym,
                elapsed_seconds=elapsed,
            )
            if on_progress:
                on_progress(progress)
            else:
                logger.info(
                    "%s: %d/%d loaded, %d failed (%.0f%%)",
                    sym,
                    completed,
                    len(symbols),
                    len(failed),
                    progress.progress_pct,
                )

    return BatchResult(
        data=data,
        failed=failed,
        elapsed_seconds=time.monotonic() - start,
    )
