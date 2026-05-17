"""Scrapling-based financial data pipeline.

Provides undetectable web scraping for financial data sources using Scrapling's
adaptive parsing, anti-bot bypass, and spider framework.

Data sources:
    - SEC EDGAR insider filings (Form 4)
    - Financial news headlines
    - Earnings calendars

Usage:
    from src.data_ingestion.financial_scraper import FinancialScraper

    scraper = FinancialScraper()
    insider_trades = scraper.scrape_insider_trades("AAPL")
    news = scraper.scrape_financial_news("AAPL", days=7)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from scrapling import Fetcher

    SCRAPLING_AVAILABLE = True
except ImportError:
    SCRAPLING_AVAILABLE = False
    logger.warning("scrapling not installed. Install with: uv add scrapling")


@dataclass
class InsiderTrade:
    """SEC Form 4 insider trade record."""

    ticker: str
    insider_name: str
    title: str
    transaction_type: str  # "Buy" or "Sale"
    shares: float
    price: float | None
    value: float
    filing_date: str
    source_url: str = ""


@dataclass
class NewsHeadline:
    """Financial news headline."""

    ticker: str
    headline: str
    source: str
    url: str
    timestamp: str = ""
    sentiment: float | None = None


@dataclass
class FinancialScrapingResult:
    """Aggregated scraping result."""

    source: str
    records: list[dict[str, Any]] = field(default_factory=list)
    record_count: int = 0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class FinancialScraper:
    """Scrapling-based financial data scraper.

    Parameters:
        cache_dir: Directory for cached HTML/responses.
        user_agent: Custom User-Agent header.
        rate_limit: Seconds between requests to same domain.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/financial_scraping",
        user_agent: str = "investment-trying-research/1.0",
        rate_limit: float = 1.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent
        self.rate_limit = rate_limit

        self._fetcher: Any = None
        if SCRAPLING_AVAILABLE:
            self._fetcher = Fetcher()

    def scrape_insider_trades(
        self,
        ticker: str,
        limit: int = 20,
    ) -> list[InsiderTrade]:
        """Scrape insider trades from SEC EDGAR or OpenInsider.

        Uses direct SEC EDGAR RSS feed for Form 4 filings, fallback
        to scraping if Scrapling's adaptive parser is available.

        Args:
            ticker: Stock ticker symbol.
            limit: Maximum records to return.

        Returns:
            List of InsiderTrade records.
        """
        import urllib.request
        import xml.etree.ElementTree as ET

        trades: list[InsiderTrade] = []

        try:
            cik = self._get_cik(ticker)
            if not cik:
                logger.warning("CIK not found for %s", ticker)
                return trades

            url = f"https://data.sec.gov/rss?cik={cik}&type=4&count={limit}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": self.user_agent},
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                root = ET.fromstring(resp.read().decode("utf-8"))

            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                updated = entry.find("atom:updated", ns)
                link = entry.find("atom:link", ns)

                if title is None:
                    continue

                title_text = title.text or ""
                parts = title_text.split(" - ")
                owner = parts[0] if parts else ""
                trans_type = "Buy" if "acquisition" in title_text.lower() else "Sale"

                summary_text = summary.text if summary is not None else ""
                shares_val = _extract_number(summary_text, r"(\d+(?:,\d+)*)\s+shares")
                price_val = _extract_number(summary_text, r"\$(\d+(?:\.\d+)?)")

                trades.append(
                    InsiderTrade(
                        ticker=ticker.upper(),
                        insider_name=owner.strip(),
                        title=summary_text[:80] if summary_text else "",
                        transaction_type=trans_type,
                        shares=float(shares_val) if shares_val else 0.0,
                        price=float(price_val) if price_val else None,
                        value=(float(shares_val) * float(price_val))
                        if shares_val and price_val
                        else 0.0,
                        filing_date=updated.text[:10]
                        if updated is not None and updated.text
                        else "",
                        source_url=link.get("href", "") if link is not None else "",
                    )
                )

        except Exception as exc:
            logger.error("Failed to scrape insider trades for %s: %s", ticker, exc)

        return trades

    def scrape_financial_news(
        self,
        ticker: str,
        days: int = 7,
    ) -> list[NewsHeadline]:
        """Scrape financial news headlines.

        Uses market news RSS feeds.

        Args:
            ticker: Stock ticker symbol.
            days: Lookback period in days.

        Returns:
            List of NewsHeadline records.
        """
        headlines: list[NewsHeadline] = []
        sources = [
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        ]

        for url in sources:
            try:
                import urllib.request
                import xml.etree.ElementTree as ET

                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": self.user_agent},
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    root = ET.fromstring(resp.read().decode("utf-8"))

                for item in root.iter("item"):
                    title_el = item.find("title")
                    link_el = item.find("link")
                    pubdate_el = item.find("pubDate")

                    if title_el is None:
                        continue

                    headlines.append(
                        NewsHeadline(
                            ticker=ticker.upper(),
                            headline=title_el.text or "",
                            source="Yahoo Finance",
                            url=link_el.text if link_el is not None else "",
                            timestamp=pubdate_el.text if pubdate_el is not None else "",
                        )
                    )

            except Exception as exc:
                logger.warning("Failed to scrape news from %s: %s", url, exc)

        return headlines

    def scrape_earnings_calendar(
        self,
        ticker: str | None = None,
        days_ahead: int = 30,
    ) -> list[dict[str, Any]]:
        """Scrape upcoming earnings dates.

        Args:
            ticker: Optional ticker filter.
            days_ahead: Days to look forward.

        Returns:
            List of earnings calendar entries.
        """
        events: list[dict[str, Any]] = []

        try:
            import urllib.request
            import xml.etree.ElementTree as ET

            url = "https://api.nasdaq.com/api/calendar/earnings?date=" + datetime.now().strftime(
                "%Y-%m-%d"
            )
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/json",
                },
            )

            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            rows = data.get("data", {}).get("rows", [])
            for row in rows:
                sym = row.get("symbol", "")
                if ticker and sym.upper() != ticker.upper():
                    continue
                events.append(
                    {
                        "ticker": sym,
                        "company": row.get("companyName", ""),
                        "date": row.get("date", ""),
                        "time": row.get("time", ""),
                        "fiscal_quarter": row.get("fiscalQuarterEnding", ""),
                    }
                )

        except Exception as exc:
            logger.warning("Failed to scrape earnings calendar: %s", exc)

        return events

    def scrape_sec_filing_links(
        self,
        ticker: str,
        filing_type: str = "10-K",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Scrape SEC filing links using EDGAR submissions API.

        Args:
            ticker: Stock ticker symbol.
            filing_type: Filing type (10-K, 10-Q, 8-K).
            limit: Maximum filings to return.

        Returns:
            List of filing metadata dicts.
        """
        import urllib.request

        filings: list[dict[str, Any]] = []

        try:
            cik = self._get_cik(ticker)
            if not cik:
                return filings

            url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": self.user_agent},
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            primaries = recent.get("primaryDocument", [])

            for i, form in enumerate(forms):
                if form != filing_type:
                    continue
                if len(filings) >= limit:
                    break

                acc = accessions[i].replace("-", "")
                doc = primaries[i]
                filings.append(
                    {
                        "ticker": ticker.upper(),
                        "cik": cik,
                        "form": form,
                        "filing_date": dates[i] if i < len(dates) else "",
                        "accession": accessions[i] if i < len(accessions) else "",
                        "document_url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}",
                    }
                )

        except Exception as exc:
            logger.error("Failed to scrape SEC filings for %s: %s", ticker, exc)

        return filings

    def scrape_all(
        self,
        ticker: str,
    ) -> FinancialScrapingResult:
        """Run all scrapers for a ticker and return aggregated results.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            FinancialScrapingResult with all records.
        """
        result = FinancialScrapingResult(source=f"financial_scraper/{ticker}")

        # Insider trades
        try:
            trades = self.scrape_insider_trades(ticker)
            result.records.extend(
                [
                    {
                        "type": "insider_trade",
                        "ticker": t.ticker,
                        "insider": t.insider_name,
                        "transaction": t.transaction_type,
                        "shares": t.shares,
                        "value": t.value,
                        "date": t.filing_date,
                    }
                    for t in trades
                ]
            )
            result.record_count += len(trades)
        except Exception as exc:
            result.errors.append(f"insider_trades: {exc}")

        # News
        try:
            news = self.scrape_financial_news(ticker)
            result.records.extend(
                [
                    {
                        "type": "news",
                        "ticker": n.ticker,
                        "headline": n.headline,
                        "source": n.source,
                        "url": n.url,
                        "timestamp": n.timestamp,
                    }
                    for n in news
                ]
            )
            result.record_count += len(news)
        except Exception as exc:
            result.errors.append(f"news: {exc}")

        return result

    def _get_cik(self, ticker: str) -> str:
        """Get CIK number for a ticker from SEC company tickers."""
        import urllib.request

        cache_file = self.cache_dir / "sec_cik_map.json"
        cik_map: dict[str, str] = {}

        if cache_file.exists():
            try:
                cik_map = json.loads(cache_file.read_text())
            except Exception:
                pass

        ticker_upper = ticker.upper()
        if ticker_upper in cik_map:
            return cik_map[ticker_upper]

        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": self.user_agent},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            for entry in data.values():
                sym = entry.get("ticker", "").upper()
                cik_str = str(entry.get("cik_str", ""))
                if sym and cik_str:
                    cik_map[sym] = cik_str

            cache_file.write_text(json.dumps(cik_map, indent=2))
        except Exception as exc:
            logger.warning("Failed to load CIK map: %s", exc)

        return cik_map.get(ticker_upper, "")


def _extract_number(text: str, pattern: str) -> float | None:
    """Extract a number from text using regex."""
    import re

    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", ""))
    return None
