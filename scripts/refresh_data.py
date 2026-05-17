"""
Refresh daily data files with latest data from Yahoo Finance.

Appends new bars to existing CSV files without re-downloading historical data.
Safe for daily use — fetches only the gap between last date in file and today.

Usage:
    # Refresh all instruments in data/raw/
    uv run scripts/refresh_data.py

    # Refresh specific instruments
    uv run scripts/refresh_data.py --symbols QQQ,GLD,TLT,BTC_USD

    # Fetch last N days (override auto-detect)
    uv run scripts/refresh_data.py --days 7

    # Dry run (show what would be fetched, don't save)
    uv run scripts/refresh_data.py --dry-run
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).parent.parent

# Map data file symbols to yfinance tickers
SYMBOL_TO_YF: dict[str, str] = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",
    "XLK": "XLK",
    "XLF": "XLF",
    "XLE": "XLE",
    "XLV": "XLV",
    "GLD": "GLD",
    "TLT": "TLT",
    "KO": "KO",
    "JPM": "JPM",
    "XOM": "XOM",
    "JNJ": "JNJ",
    "SO": "SO",
    "BTC_USD": "BTC-USD",
    "EURUSD_X": "EURUSD=X",
}

DATA_DIR = PROJECT_ROOT / "data" / "raw"


def _last_date_in_file(symbol: str) -> date | None:
    path = DATA_DIR / f"{symbol}_daily.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=[0])
    last = pd.Timestamp(df.iloc[-1, 0])
    return last.date()


def _expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Handle MultiIndex columns (yfinance sometimes returns tuples)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    cols = [str(c).lower() for c in df.columns]
    col_map = {}
    for i, c in enumerate(cols):
        if "date" in c:
            col_map[df.columns[i]] = "Date"
        elif c == "open":
            col_map[df.columns[i]] = "Open"
        elif c == "high":
            col_map[df.columns[i]] = "High"
        elif c == "low":
            col_map[df.columns[i]] = "Low"
        elif c == "close":
            col_map[df.columns[i]] = "Close"
        elif c == "volume":
            col_map[df.columns[i]] = "Volume"
    df = df.rename(columns=col_map)
    desired = ["Date", "Close", "High", "Low", "Open", "Volume"]
    available = [c for c in desired if c in df.columns]
    return df[available]


def refresh_symbol(
    symbol: str,
    end_date: date | None = None,
    days: int | None = None,
    dry_run: bool = False,
) -> dict:
    yf_ticker = SYMBOL_TO_YF.get(symbol)
    if yf_ticker is None:
        return {"symbol": symbol, "status": "unknown_symbol"}

    existing_path = DATA_DIR / f"{symbol}_daily.csv"
    existing_last = _last_date_in_file(symbol)

    if end_date is None:
        end_date = date.today() + timedelta(days=1)

    if days:
        start_date = end_date - timedelta(days=days)
        if existing_last and existing_last >= start_date:
            start_date = existing_last + timedelta(days=1)
    elif existing_last:
        start_date = existing_last + timedelta(days=1)
    else:
        return {"symbol": symbol, "status": "no_existing_file"}

    if start_date >= end_date:
        return {
            "symbol": symbol,
            "status": "up_to_date",
            "last_date": str(existing_last),
        }

    if dry_run:
        return {
            "symbol": symbol,
            "status": "would_fetch",
            "yf_ticker": yf_ticker,
            "from_date": str(start_date),
            "to_date": str(end_date),
            "existing_last": str(existing_last) if existing_last else None,
        }

    try:
        new_df = yf.download(
            yf_ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
        )
    except Exception as e:
        return {"symbol": symbol, "status": "download_error", "error": str(e)}

    if new_df.empty:
        return {
            "symbol": symbol,
            "status": "no_new_data",
            "from_date": str(start_date),
        }

    new_df = new_df.reset_index()
    new_df = _expected_columns(new_df)

    if existing_path.exists() and existing_last:
        old_df = pd.read_csv(existing_path, parse_dates=[0])
        old_last = pd.Timestamp(old_df.iloc[-1, 0])
        new_first = pd.Timestamp(new_df.iloc[0, 0])
        if new_first <= old_last:
            new_df = new_df[new_df["Date"] > str(old_last.date())]

    if new_df.empty:
        return {
            "symbol": symbol,
            "status": "no_new_data_after_dedup",
            "from_date": str(start_date),
        }

    if existing_path.exists() and existing_last:
        old_df = pd.read_csv(existing_path, parse_dates=[0])
        combined = pd.concat([old_df, new_df], ignore_index=True)
        # Convert Date column to string format matching existing files
        combined["Date"] = combined["Date"].dt.strftime("%Y-%m-%d")
        combined.to_csv(existing_path, index=False)
    else:
        new_df["Date"] = new_df["Date"].dt.strftime("%Y-%m-%d")
        new_df.to_csv(existing_path, index=False)

    return {
        "symbol": symbol,
        "status": "updated",
        "new_rows": len(new_df),
        "from_date": str(new_df.iloc[0]["Date"]),
        "to_date": str(new_df.iloc[-1]["Date"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh daily data with latest bars from Yahoo Finance"
    )
    parser.add_argument(
        "--symbols",
        help="Comma-separated symbols (default: all in SYMBOL_TO_YF)",
    )
    parser.add_argument(
        "--days",
        type=int,
        help="Fetch last N days (overrides auto-detect from file end date)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without downloading",
    )
    args = parser.parse_args()

    symbols = (
        [s.strip() for s in args.symbols.split(",")]
        if args.symbols
        else sorted(SYMBOL_TO_YF.keys())
    )

    end_date = date.today() + timedelta(days=1)

    print(f"\n{'=' * 60}")
    print(f" DATA REFRESH — {date.today()}")
    print(f"{'=' * 60}")
    if args.dry_run:
        print(" DRY RUN — no data will be downloaded\n")
    else:
        print()

    results = []
    for symbol in symbols:
        r = refresh_symbol(symbol, end_date=end_date, days=args.days, dry_run=args.dry_run)
        results.append(r)
        if r["status"] == "updated":
            print(f"  {symbol:10s} +{r['new_rows']} bars  ({r['from_date']} -> {r['to_date']})")
        elif r["status"] == "up_to_date":
            print(f"  {symbol:10s} up to date (last: {r['last_date']})")
        elif r["status"] == "would_fetch":
            print(
                f"  {symbol:10s} WOULD FETCH {r['yf_ticker']}  ({r['from_date']} -> {r['to_date']})"
            )
        elif r["status"] == "no_new_data":
            print(f"  {symbol:10s} no new data available ({r['from_date']} onward)")
        elif r["status"] == "no_existing_file":
            print(f"  {symbol:10s} SKIP — no existing file")
        else:
            print(f"  {symbol:10s} {r['status']}: {r.get('error', '')}")

    updated = [r for r in results if r["status"] == "updated"]
    would = [r for r in results if r["status"] == "would_fetch"]
    uptodate = [r for r in results if r["status"] == "up_to_date"]

    print(f"\n{'=' * 60}")
    if args.dry_run:
        print(f" Would fetch: {len(would)}  Up to date: {len(uptodate)}")
    else:
        print(f" Updated: {len(updated)}  Up to date: {len(uptodate)}")
        if updated:
            total_rows = sum(r.get("new_rows", 0) for r in updated)
            print(f" Total new rows: {total_rows}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
