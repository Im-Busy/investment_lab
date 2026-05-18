"""
Chinese Market Data Ingestion via AKShare

Fetches A-share stocks, indexes, futures, and macro data from Chinese markets.
AKShare is free and requires no API token. Data saved to data/raw/ with CN_ prefix.

Chinese financial terms are mapped to English column names for project consistency.
Company names are preserved in Chinese.

Usage:
    uv run src/data_ingestion/fetch_china.py            # Fetch all instruments
    uv run src/data_ingestion/fetch_china.py --indexes  # Indexes only
    uv run src/data_ingestion/fetch_china.py --stocks   # Stocks only
    uv run src/data_ingestion/fetch_china.py --futures  # Futures only
"""

from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

import akshare as ak
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# ── Column name mapping: Chinese → English ─────────────────────────────────
COLUMN_MAP: dict[str, str] = {
    "日期": "Date",
    "开盘": "Open",
    "收盘": "Close",
    "最高": "High",
    "最低": "Low",
    "成交量": "Volume",
    "成交额": "Amount",
    "振幅": "Amplitude",
    "涨跌幅": "ChangePct",
    "涨跌额": "Change",
    "换手率": "Turnover",
    "商品": "Indicator",
    "今值": "Value",
    "预测值": "Forecast",
    "前值": "Previous",
}


def _save(symbol: str, df: pd.DataFrame) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = df.rename(columns=COLUMN_MAP)
    keep = [c for c in ["Date", "Close", "High", "Low", "Open", "Volume"] if c in df.columns]
    out = df[keep].copy()
    path = DATA_DIR / f"CN_{symbol}_daily.csv"
    out.to_csv(path, index=False)
    return path


# ── Indexes ─────────────────────────────────────────────────────────────────

INDEX_MAP: dict[str, str] = {
    "CSI300": "sh000300",  # CSI 300 (large cap, most tracked)
    "CSI500": "sh000905",  # CSI 500 (mid/small cap)
    "SSE50": "sh000016",  # SSE 50 (super large cap)
    "SHCOMP": "sh000001",  # Shanghai Composite (broadest)
    "SZCOMP": "sz399001",  # Shenzhen Component (tech-heavy)
    "CHINEXT": "sz399006",  # ChiNext (China's NASDAQ)
    "STAR50": "sh000688",  # STAR 50 (science & tech innovation)
}


def fetch_indexes() -> dict[str, Path]:
    results: dict[str, Path] = {}
    for name, symbol in INDEX_MAP.items():
        try:
            df = ak.stock_zh_index_daily(symbol=symbol)
            if df is not None and not df.empty:
                path = _save(name, df)
                results[name] = path
                print(f"  {name:10s} {len(df):>6d} bars  -> {path.name}")
            else:
                print(f"  {name:10s} no data")
        except Exception as e:
            print(f"  {name:10s} ERROR: {e}")
        time.sleep(0.5)
    return results


# ── Major Stocks ────────────────────────────────────────────────────────────

STOCK_LIST: dict[str, str] = {
    "600519": "Moutai",  # Kweichow Moutai (白酒)
    "000858": "Wuliangye",  # Wuliangye Yibin (白酒)
    "601318": "PingAn",  # Ping An Insurance
    "600036": "CMB",  # China Merchants Bank
    "000333": "Midea",  # Midea Group
    "300750": "CATL",  # CATL (battery)
    "002594": "BYD",  # BYD (EV)
    "601857": "PetroChina",  # PetroChina
    "600900": "YangtzePower",  # Yangtze Power
    "601088": "ChinaShenhua",  # China Shenhua Energy
    "688981": "SMIC",  # SMIC (chip foundry)
    "600276": "Hengrui",  # Hengrui Medicine
}


def fetch_stocks() -> dict[str, Path]:
    results: dict[str, Path] = {}
    sid_map: dict[str, str] = {}
    for code, name in STOCK_LIST.items():
        if code.startswith("6"):
            sid_map[name] = f"sh{code}"
        else:
            sid_map[name] = f"sz{code}"

    for name, sid in sid_map.items():
        code = [c for c, n in STOCK_LIST.items() if n == name][0]
        try:
            df = ak.stock_zh_a_daily(symbol=sid, adjust="qfq")
            if df is not None and not df.empty:
                path = _save(name, df)
                results[name] = path
                print(f"  {name:20s} ({code}) {len(df):>6d} bars  -> {path.name}")
            else:
                print(f"  {name:20s} ({code}) no data")
        except Exception as e:
            print(f"  {name:20s} ({code}) ERROR: {e}")
        time.sleep(0.5)
    return results


# ── Futures ─────────────────────────────────────────────────────────────────

FUTURES_LIST: dict[str, str] = {
    "IF": "CSI300_Futures",  # CSI 300 Index Futures
    "IC": "CSI500_Futures",  # CSI 500 Index Futures
    "IH": "SSE50_Futures",  # SSE 50 Index Futures
    "IM": "CSI1000_Futures",  # CSI 1000 Index Futures
    "RB": "Rebar_Futures",  # Rebar (螺纹钢)
    "I": "IronOre_Futures",  # Iron Ore (铁矿石)
    "CU": "Copper_Futures",  # Copper (沪铜)
    "AU": "Gold_Futures_CNY",  # Gold (沪金) — CNY-denominated
    "AG": "Silver_Futures_CNY",  # Silver (沪银)
    "SC": "CrudeOil_Futures_CNY",  # Crude Oil (上海原油)
}


def fetch_futures() -> dict[str, Path]:
    results: dict[str, Path] = {}
    for code, name in FUTURES_LIST.items():
        try:
            df = ak.futures_main_sina(symbol=code)
            if df is not None and not df.empty:
                path = _save(name, df)
                results[name] = path
                print(f"  {name:25s} ({code}) {len(df):>6d} bars  -> {path.name}")
            else:
                print(f"  {name:25s} ({code}) no data")
        except Exception as e:
            print(f"  {name:25s} ({code}) ERROR: {e}")
        time.sleep(0.5)
    return results


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Chinese market data via AKShare")
    parser.add_argument("--indexes", action="store_true", help="Fetch only indexes")
    parser.add_argument("--stocks", action="store_true", help="Fetch only stocks")
    parser.add_argument("--futures", action="store_true", help="Fetch only futures")
    parser.add_argument("--symbol", help="Fetch single stock by code (e.g. 600519)")
    args = parser.parse_args()

    all_mode = not (args.indexes or args.stocks or args.futures or args.symbol)

    print(f"\n{'=' * 60}")
    print(f" CHINESE MARKET DATA INGESTION — {date.today()}")
    print(" Source: AKShare (free, no token)")
    print(f"{'=' * 60}\n")

    if args.symbol:
        code = args.symbol
        name = STOCK_LIST.get(code, code)
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date="20150101",
                end_date=date.today().strftime("%Y%m%d"),
                adjust="qfq",
            )
            if df is not None and not df.empty:
                path = _save(name, df)
                print(f"  {name} ({code}): {len(df)} bars -> {path}")
        except Exception as e:
            print(f"  ERROR: {e}")
        return

    if all_mode or args.indexes:
        print("[Indexes]")
        fetch_indexes()
        print()

    if all_mode or args.stocks:
        print("[Stocks]")
        fetch_stocks()
        print()

    if all_mode or args.futures:
        print("[Futures]")
        fetch_futures()
        print()

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
