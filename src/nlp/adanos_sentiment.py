"""P28-9: Adanos sentiment API integration — cross-source sentiment signals.

Adanos aggregates sentiment data from Reddit, X (Twitter), news headlines,
and Polymarket prediction markets for any ticker. REST API, no auth required,
returns structured sentiment scores and raw mentions.

Source: awesome-ai-in-finance — Adanos cross-source sentiment service.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError
import json

logger = logging.getLogger(__name__)

ADANOS_BASE_URL = "https://api.adanos.ai/v1/sentiment"
REQUEST_TIMEOUT = 10
CACHE_TTL_SECONDS = 300


@dataclass
class SentimentSource:
    """Per-source sentiment breakdown."""

    source: str
    score: float
    mentions: int
    bullish_pct: float
    bearish_pct: float


@dataclass
class AdanosSentimentResult:
    """Complete sentiment result for a ticker."""

    ticker: str
    composite_score: float
    sources: list[SentimentSource] = field(default_factory=list)
    timestamp: float = 0.0
    raw_response: dict | None = None

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    @property
    def signal(self) -> float:
        """Composite signal in [-1, +1] for strategy integration."""
        return max(-1.0, min(1.0, self.composite_score))

    @property
    def confidence(self) -> float:
        """Signal confidence based on total mention count."""
        total = sum(s.mentions for s in self.sources)
        return min(1.0, total / 50.0) if total > 0 else 0.0

    @property
    def source_count(self) -> int:
        """Number of contributing sources."""
        return len([s for s in self.sources if s.mentions > 0])

    def summary(self) -> str:
        lines = [f"{self.ticker}: composite={self.composite_score:.3f}"]
        for s in self.sources:
            lines.append(
                f"  {s.source}: {s.score:+.3f} ({s.mentions} mentions, "
                f"{s.bullish_pct:.0%}B/{s.bearish_pct:.0%}A)"
            )
        return "\n".join(lines)


_sentiment_cache: dict[str, tuple[float, AdanosSentimentResult]] = {}


def fetch_adanos_sentiment(
    ticker: str,
    timeout: int = REQUEST_TIMEOUT,
    use_cache: bool = True,
) -> AdanosSentimentResult:
    """Fetch cross-source sentiment for a ticker from Adanos API.

    Aggregates Reddit, X, news, and Polymarket data. Returns composite
    score and per-source breakdown. Cache TTL: 5 minutes.

    Args:
        ticker: Stock/crypto symbol (e.g., "SPY", "BTC").
        timeout: Request timeout in seconds.
        use_cache: Whether to use internal cache.

    Returns:
        AdanosSentimentResult with composite score and source breakdown.
    """
    if use_cache:
        cached = _sentiment_cache.get(ticker.upper())
        if cached is not None:
            ts, result = cached
            if time.time() - ts < CACHE_TTL_SECONDS:
                return result

    url = f"{ADANOS_BASE_URL}?symbol={ticker.upper()}"
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except URLError as e:
        logger.warning("Adanos API unreachable for %s: %s", ticker, e)
        return AdanosSentimentResult(ticker=ticker, composite_score=0.0)
    except json.JSONDecodeError:
        logger.warning("Adanos API returned invalid JSON for %s", ticker)
        return AdanosSentimentResult(ticker=ticker, composite_score=0.0)

    sources = [
        SentimentSource(
            source=s.get("source", "unknown"),
            score=float(s.get("score", 0.0)),
            mentions=int(s.get("mentions", 0)),
            bullish_pct=float(s.get("bullish_pct", 0.0)),
            bearish_pct=float(s.get("bearish_pct", 0.0)),
        )
        for s in data.get("sources", [])
    ]

    result = AdanosSentimentResult(
        ticker=ticker.upper(),
        composite_score=float(data.get("composite_score", 0.0)),
        sources=sources,
        timestamp=time.time(),
        raw_response=data,
    )

    if use_cache:
        _sentiment_cache[ticker.upper()] = (time.time(), result)
    return result


def fetch_adanos_batch(
    tickers: list[str],
    timeout: int = REQUEST_TIMEOUT,
    delay: float = 0.25,
) -> dict[str, AdanosSentimentResult]:
    """Fetch sentiment for multiple tickers with rate-limiting.

    Args:
        tickers: List of ticker symbols.
        timeout: Per-request timeout in seconds.
        delay: Delay between requests in seconds.

    Returns:
        Dict mapping ticker → AdanosSentimentResult.
    """
    results: dict[str, AdanosSentimentResult] = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(delay)
        results[ticker.upper()] = fetch_adanos_sentiment(ticker, timeout=timeout)
    return results


def clear_sentiment_cache() -> None:
    """Clear the internal sentiment cache."""
    _sentiment_cache.clear()
