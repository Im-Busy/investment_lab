"""Backtest all remaining tickers not in the original 20. Appends results."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.batch_individual_backtest import (
    run_ticker,
    produce_analysis,
    CATEGORY_MAP,
    OUTPUT_DIR,
)

COMPLETED = {
    "SPY",
    "QQQ",
    "XLK",
    "XLE",
    "GLD",
    "SLV",
    "XLF",
    "XLV",
    "NUE",
    "STLD",
    "HAL",
    "MPC",
    "EOG",
    "INTC",
    "AMD",
    "LMT",
    "JNJ",
    "MRK",
    "NEM",
    "CN_CATL",
}


def auto_category(symbol: str) -> str:
    if symbol in CATEGORY_MAP:
        return CATEGORY_MAP[symbol]
    if symbol.startswith("CN_"):
        return "China"
    if symbol.startswith("HK_"):
        return "Hong Kong"
    if symbol.endswith("_USD"):
        return "Crypto"
    return "US Stock"


def find_remaining() -> list[str]:
    data_dir = Path("data/raw")
    tickers = []
    for f in sorted(data_dir.glob("*_daily.csv")):
        t = f.stem.replace("_daily", "")
        if t not in COMPLETED:
            tickers.append(t)
    return tickers


def main():
    tickers = find_remaining()
    print(f"Found {len(tickers)} remaining tickers")

    # Load existing results
    existing = []
    for f in sorted(OUTPUT_DIR.glob("*.json")):
        existing.append(json.loads(f.read_text()))
    print(f"Loaded {len(existing)} existing results")

    all_results = existing

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
        result = run_ticker(ticker)
        if result:
            if ticker not in CATEGORY_MAP:
                CATEGORY_MAP[ticker] = auto_category(ticker)
                result["category"] = CATEGORY_MAP[ticker]
            all_results.append(result)

    print(f"\n\nTotal results: {len(all_results)}")

    # Regenerate full analysis
    produce_analysis(
        all_results,
        Path("reports/individual_backtest_analysis.md"),
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
