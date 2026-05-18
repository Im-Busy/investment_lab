#!/usr/bin/env python
"""Financial data scraping CLI using Scrapling.

Scrapes insider trades, news, SEC filings, and earnings calendars.

Usage:
    uv run scripts/scrape_financial_data.py AAPL
    uv run scripts/scrape_financial_data.py AAPL --insider --news --sec-filings
    uv run scripts/scrape_financial_data.py AAPL --earnings --days-ahead 30
    uv run scripts/scrape_financial_data.py --batch AAPL MSFT GOOGL --all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_ingestion.financial_scraper import FinancialScraper

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("scrape_financial")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape financial data using Scrapling",
    )
    parser.add_argument("tickers", nargs="*", default=[], help="Ticker symbols")
    parser.add_argument("--insider", action="store_true", help="Scrape insider trades")
    parser.add_argument("--news", action="store_true", help="Scrape financial news")
    parser.add_argument("--sec-filings", action="store_true", help="List SEC filings")
    parser.add_argument("--earnings", action="store_true", help="Scrape earnings calendar")
    parser.add_argument("--all", action="store_true", help="Run all scrapers")
    parser.add_argument("--batch", nargs="*", default=None, help="Batch ticker list")
    parser.add_argument("--days", type=int, default=7, help="Days lookback for news")
    parser.add_argument("--days-ahead", type=int, default=30, help="Days ahead for earnings")
    parser.add_argument("--limit", type=int, default=10, help="Max records per source")
    parser.add_argument("--json-output", type=Path, default=None, help="Save results to JSON")
    parser.add_argument("--cache-dir", default="data/financial_scraping", help="Cache directory")

    args = parser.parse_args()

    tickers = args.tickers
    if args.batch:
        tickers = args.batch

    if not tickers:
        parser.error("No tickers specified")
        return

    scraper = FinancialScraper(cache_dir=args.cache_dir)

    all_results: dict[str, dict[str, list[dict[str, object]]]] = {}

    run_all = args.all or (
        not args.insider and not args.news and not args.sec_filings and not args.earnings
    )
    if run_all:
        args.insider = args.news = args.sec_filings = args.earnings = True

    for ticker in tickers:
        ticker_results: dict[str, list[dict[str, object]]] = {}

        if args.insider:
            trades = scraper.scrape_insider_trades(ticker, limit=args.limit)
            ticker_results["insider_trades"] = [
                {
                    "insider": t.insider_name,
                    "transaction": t.transaction_type,
                    "shares": t.shares,
                    "value": t.value,
                    "date": t.filing_date,
                }
                for t in trades
            ]
            print(f"\n  {ticker} Insider Trades ({len(trades)}):")
            for t in trades[:5]:
                print(
                    f"    {t.insider_name:30s} {t.transaction_type:4s}  {t.shares:>10,.0f} shares  ${t.value:>12,.2f}  {t.filing_date}"
                )

        if args.news:
            headlines = scraper.scrape_financial_news(ticker, days=args.days)
            ticker_results["news"] = [
                {
                    "headline": h.headline,
                    "source": h.source,
                    "timestamp": h.timestamp,
                }
                for h in headlines[: args.limit]
            ]
            print(f"\n  {ticker} News ({len(headlines)}):")
            for h in headlines[:5]:
                print(f"    [{h.source:15s}] {h.headline[:80]}")

        if args.sec_filings:
            filings = scraper.scrape_sec_filing_links(ticker, filing_type="10-K", limit=args.limit)
            ticker_results["sec_filings"] = filings
            print(f"\n  {ticker} SEC Filings ({len(filings)}):")
            for f in filings[:5]:
                print(f"    {f['form']:6s}  {f['filing_date']}  {f['document_url']}")

        if args.earnings:
            events = scraper.scrape_earnings_calendar(ticker, days_ahead=args.days_ahead)
            ticker_results["earnings"] = events
            print(f"\n  {ticker} Earnings ({len(events)}):")
            for e in events[:5]:
                print(
                    f"    {e['ticker']:6s}  {e['date']}  {e['time']:6s}  {e.get('company', '')[:40]}"
                )

        all_results[ticker] = ticker_results

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(all_results, indent=2, default=str))
        logger.info("Results saved to %s", args.json_output)


if __name__ == "__main__":
    main()
