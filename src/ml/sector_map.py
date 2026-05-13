"""
Per-sector model mapping for 33 tickers across 7 sectors.

Used by B10 per-sector training (train_ml_pipeline_v3.py --sector) and
per-sector inference (MLStrategy ticker param → sector → model path).
"""

from __future__ import annotations

SECTOR_MAP: dict[str, str] = {
    # Tech
    "XLK": "tech",
    "QQQ": "tech",
    "AAPL": "tech",
    "MSFT": "tech",
    "NVDA": "tech",
    "AVGO": "tech",
    "AMD": "tech",
    "ADBE": "tech",
    "CRM": "tech",
    "CSCO": "tech",
    "INTC": "tech",
    "IBM": "tech",
    # Financials
    "XLF": "financials",
    "JPM": "financials",
    "BAC": "financials",
    "WFC": "financials",
    "GS": "financials",
    "MS": "financials",
    "C": "financials",
    # Energy
    "XLE": "energy",
    "XOM": "energy",
    "CVX": "energy",
    "COP": "energy",
    "SLB": "energy",
    # Healthcare
    "XLV": "healthcare",
    "JNJ": "healthcare",
    "UNH": "healthcare",
    "PFE": "healthcare",
    "ABBV": "healthcare",
    # Consumer
    "XLP": "consumer",
    "PG": "consumer",
    "KO": "consumer",
    "PEP": "consumer",
    "WMT": "consumer",
    "COST": "consumer",
    # Industrials
    "XLI": "industrials",
    "UNP": "industrials",
    "CAT": "industrials",
    "GE": "industrials",
    "BA": "industrials",
    # Utilities / REITs
    "XLU": "utilities_reits",
    "AVB": "utilities_reits",
    "D": "utilities_reits",
    "SO": "utilities_reits",
}

SECTOR_NAMES: list[str] = [
    "tech",
    "financials",
    "energy",
    "healthcare",
    "consumer",
    "industrials",
    "utilities_reits",
]
