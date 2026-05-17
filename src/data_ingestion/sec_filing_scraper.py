"""SEC EDGAR filing scraper.

Downloads 10-K/10-Q filings for any ticker via the SEC EDGAR public API.
Extracts the MD&A (Management Discussion & Analysis) section for NLP analysis.

No API key required — EDGAR is publicly accessible.
Rate limit: 10 requests/second. Must include User-Agent header.

Usage:
    scraper = SECFilingScraper()
    filings = scraper.get_filings("AAPL", filing_type="10-K", limit=5)
    for f in filings:
        text = scraper.download_filing_text(f)
        mda = scraper.extract_mda(text)
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

SEC_HEADERS = {
    "User-Agent": "investment-trying-research/1.0 (your.email@example.com)",
    "Accept": "application/json",
}
CIK_LOOKUP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc_name}"

FILING_TYPES = ["10-K", "10-Q", "8-K", "S-1", "DEF 14A"]


class SECFilingScraper:
    """Download SEC EDGAR filings for NLP analysis.

    Attributes:
        cache_dir: Directory to cache downloaded filings.
        delay: Seconds to wait between SEC requests (rate limit compliance).
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/sec_filings",
        delay: float = 0.12,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self._cik_map: dict[str, int] | None = None

    def _load_cik_map(self) -> dict[str, int]:
        if self._cik_map is not None:
            return self._cik_map
        resp = requests.get(CIK_LOOKUP_URL, headers=SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        self._cik_map = {item["ticker"].upper(): item["cik_str"] for item in data.values()}
        return self._cik_map

    def _cik_padded(self, cik: int) -> str:
        return str(cik).zfill(10)

    def get_cik(self, ticker: str) -> int | None:
        cik_map = self._load_cik_map()
        return cik_map.get(ticker.upper())

    def get_submissions(self, cik: int) -> dict[str, Any]:
        url = SUBMISSIONS_URL.format(cik_padded=self._cik_padded(cik))
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        time.sleep(self.delay)
        return resp.json()

    def get_filings(
        self,
        ticker: str,
        filing_type: str = "10-K",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get recent filings list for a ticker.

        Args:
            ticker: Ticker symbol (e.g., "AAPL").
            filing_type: Filing type filter ("10-K", "10-Q", etc.).
            limit: Maximum number of filings to return.

        Returns:
            List of filing dicts with keys: accession_number, filing_date,
            report_date, primary_document, form, cik, ticker.
        """
        cik = self.get_cik(ticker)
        if cik is None:
            logger.warning("No CIK found for ticker %s", ticker)
            return []

        submissions = self.get_submissions(cik)
        recent = submissions.get("filings", {}).get("recent", {})
        if not recent:
            return []

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])
        report_dates = recent.get("reportDate", [])

        filings = []
        for i in range(min(len(forms), len(dates), len(accessions))):
            if forms[i] == filing_type:
                filings.append(
                    {
                        "accession_number": accessions[i],
                        "filing_date": dates[i],
                        "report_date": report_dates[i] if i < len(report_dates) else None,
                        "primary_document": docs[i] if i < len(docs) else None,
                        "form": forms[i],
                        "cik": cik,
                        "ticker": ticker.upper(),
                    }
                )
            if len(filings) >= limit:
                break

        return filings

    def download_filing_text(self, filing: dict[str, Any]) -> str:
        """Download and extract text from a filing.

        Args:
            filing: Filing dict from get_filings().

        Returns:
            Raw text content of the filing.
        """
        cik = filing["cik"]
        accession = filing["accession_number"].replace("-", "")
        doc_name = filing.get("primary_document", "")

        cache_key = f"{filing['ticker']}_{filing['form']}_{filing['filing_date']}.txt"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8", errors="replace")

        url = ARCHIVE_URL.format(cik=cik, accession=accession, doc_name=doc_name)
        resp = requests.get(url, headers=SEC_HEADERS, timeout=60)
        resp.raise_for_status()
        time.sleep(self.delay)

        text = self._strip_html(resp.text)
        cache_path.write_text(text, encoding="utf-8", errors="replace")
        return text

    @staticmethod
    def _strip_html(html_text: str) -> str:
        """Strip HTML tags and decode entities for plain text."""
        import html as html_mod

        text = re.sub(r"<[^>]+>", " ", html_text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"\s+", " ", text)
        return html_mod.unescape(text).strip()

    def extract_mda(self, filing_text: str) -> str:
        """Extract MD&A section from 10-K/10-Q text.

        Looks for common MD&A section headers:
        - "Management's Discussion and Analysis"
        - "Item 7." or "Item 2." (10-K vs 10-Q)
        - Terminates at next major section header.

        Args:
            filing_text: Raw text of a 10-K or 10-Q.

        Returns:
            MD&A section text, or empty string if not found.
        """
        # Normalize
        text = filing_text

        mda_start_patterns = [
            r"Item\s*7[\.\s].*Management.*Discussion",
            r"Item\s*2[\.\s].*Management.*Discussion",
            r"Management['\u2019]?s\s+Discussion\s+and\s+Analysis",
            r"MANAGEMENT['\u2019]?S\s+DISCUSSION\s+AND\s+ANALYSIS",
        ]

        mda_end_patterns = [
            r"Item\s*7A[\.\s]",
            r"Item\s*8[\.\s]",
            r"Item\s*3[\.\s]",
            r"QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES",
        ]

        best_start = None
        for pattern in mda_start_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                best_start = match.start()
                break

        if best_start is None:
            return ""

        remaining = text[best_start:]
        best_end = len(remaining)
        for pattern in mda_end_patterns:
            match = re.search(pattern, remaining[100:], re.IGNORECASE)
            if match and match.start() < best_end:
                best_end = match.start() + 100

        mda_text = remaining[:best_end].strip()
        # Clean: remove excessive whitespace
        mda_text = re.sub(r"\s+", " ", mda_text)
        return mda_text


def download_filings_cli(
    ticker: str,
    filing_types: list[str] | None = None,
    limit: int = 3,
) -> dict[str, list[str]]:
    """CLI-friendly function to download filings for analysis.

    Returns:
        Dict mapping filing desc to list of MD&A text strings.
    """
    if filing_types is None:
        filing_types = ["10-K", "10-Q"]

    scraper = SECFilingScraper()
    results: dict[str, list[str]] = {}

    for ft in filing_types:
        filings = scraper.get_filings(ticker, filing_type=ft, limit=limit)
        texts = []
        for f in filings:
            try:
                raw = scraper.download_filing_text(f)
                mda = scraper.extract_mda(raw)
                texts.append(mda)
                logger.info(
                    "Downloaded %s %s (%s): %d chars of MD&A",
                    ticker,
                    ft,
                    f["filing_date"],
                    len(mda),
                )
            except Exception as e:
                logger.warning("Failed to download %s %s: %s", ticker, ft, e)
        results[ft] = texts

    return results
