"""Instrument → market session → timezone mapping.

Phase 25 — Provides timezone-aware trading window checks per instrument.
Used by strategy_selector.py for gating trades to appropriate market hours.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytz


TIMEZONE_SESSIONS: dict[str, dict[str, Any]] = {
    "US_EQUITY": {
        "timezone": "America/New_York",
        "market_open": "09:30",
        "market_close": "16:00",
        "entry_window_start": "09:45",
        "entry_window_end": "15:30",
        "description": "US equities — NYSE/NASDAQ regular trading hours",
        "instruments": [
            "SPY",
            "QQQ",
            "D",
            "EEM",
            "IWM",
            "XLK",
            "XLF",
            "XLE",
            "XLV",
            "AAPL",
            "MSFT",
            "NVDA",
            "GOOGL",
            "META",
            "AMD",
            "INTC",
            "CRM",
            "COST",
            "JPM",
            "BAC",
            "GS",
            "MS",
            "WFC",
            "C",
            "BLK",
            "SCHW",
            "JNJ",
            "UNH",
            "ABBV",
            "MRK",
            "AMGN",
            "PFE",
            "GILD",
            "VRTX",
            "REGN",
            "BIIB",
            "LLY",
            "TMO",
            "BMY",
            "XOM",
            "CVX",
            "COP",
            "EOG",
            "SLB",
            "HAL",
            "MPC",
            "PSX",
            "OXY",
            "KO",
            "PEP",
            "PG",
            "WMT",
            "HD",
            "MO",
            "KMB",
            "PM",
            "SO",
            "DUK",
            "NEE",
            "AEP",
            "EXC",
            "SRE",
            "ED",
            "PEG",
            "CAT",
            "LMT",
            "FDX",
            "UPS",
            "UNP",
            "NSC",
            "DAL",
            "UAL",
            "LUV",
            "AAL",
            "CSX",
            "O",
            "PLD",
            "SPG",
            "PSA",
            "AVB",
            "AMT",
            "WELL",
            "EQR",
            "DHR",
            "GOLD",
            "FCX",
            "NEM",
            "AEM",
            "AA",
            "STLD",
            "NUE",
            "VLO",
            "CAR",
            "KODK",
            "JOE",
            "HIFS",
            "CRVL",
        ],
    },
    "US_FUTURES": {
        "timezone": "America/Chicago",
        "market_open": None,  # Nearly 24h Sun-Fri
        "market_close": None,
        "description": "CME futures — extended hours, killzone-based entry",
        "instruments": ["NQ=F", "GC=F", "CL"],
    },
    "CRYPTO": {
        "timezone": "UTC",
        "market_open": None,  # 24/7
        "market_close": None,
        "description": "Cryptocurrency — 24/7 trading, no market hours gating",
        "instruments": ["BTC_USD", "ETH_USD"],
    },
    "CHINA": {
        "timezone": "Asia/Shanghai",
        "market_open": "09:30",
        "market_close": "15:00",
        "entry_window_start": "09:45",
        "entry_window_end": "14:45",
        "lunch_break": "11:30-13:00",
        "description": "China A-shares — Shanghai/Shenzhen exchanges",
        "instruments": [
            "CN_SHCOMP",
            "CN_CSI300",
            "CN_CSI500",
            "CN_SSE50",
            "CN_CHINEXT",
            "CN_STAR50",
            "CN_SZCOMP",
            "CN_Moutai",
            "CN_Wuliangye",
            "CN_PingAn",
            "CN_CMB",
            "CN_PetroChina",
            "CN_ChinaShenhua",
            "CN_BYD",
            "CN_CATL",
            "CN_Midea",
            "CN_Hengrui",
            "CN_YangtzePower",
            "CN_SMIC",
        ],
    },
    "HK": {
        "timezone": "Asia/Hong_Kong",
        "market_open": "09:30",
        "market_close": "16:00",
        "entry_window_start": "09:45",
        "entry_window_end": "15:45",
        "lunch_break": "12:00-13:00",
        "description": "Hong Kong Stock Exchange",
        "instruments": [
            "HK_Tencent",
            "HK_Alibaba_HK",
            "HK_Meituan",
            "HK_Xiaomi",
            "HK_BYD_HK",
            "HK_JD_HK",
            "HK_Kuaishou",
            "HK_LiAuto",
            "HK_AIA",
            "HK_HSBC",
            "HK_HKEX",
            "HK_Mixue",
        ],
    },
    "FOREX": {
        "timezone": "UTC",
        "market_open": None,  # 24h Mon-Fri
        "market_close": None,
        "description": "Forex — 24/5, killzone-based entry for SMC strategies",
        "instruments": ["EURUSD_X", "GBPJPY_X"],
    },
    "COMMODITY_ETF": {
        "timezone": "America/New_York",
        "market_open": "09:30",
        "market_close": "16:00",
        "description": "Commodity ETFs — trade on NYSE Arca",
        "instruments": ["GLD", "SLV", "IAU"],
    },
    "BOND_ETF": {
        "timezone": "America/New_York",
        "market_open": "09:30",
        "market_close": "16:00",
        "description": "Bond ETFs — trade on NYSE Arca",
        "instruments": ["TLT"],
    },
}


_SESSION_CACHE: dict[str, dict[str, Any]] = {}


def get_session_for(instrument: str) -> dict[str, Any] | None:
    """Return the timezone session config for an instrument.

    Caches results for fast repeated lookups.
    """
    if instrument in _SESSION_CACHE:
        return _SESSION_CACHE[instrument]
    for _name, cfg in TIMEZONE_SESSIONS.items():
        if instrument in cfg.get("instruments", []):
            _SESSION_CACHE[instrument] = cfg
            return cfg
    _SESSION_CACHE[instrument] = None
    return None


def is_in_trading_window(instrument: str, current_time: datetime | None = None) -> bool:
    """Check if an instrument is in its active trading window.

    For 24/7 instruments (crypto, futures, forex), always returns True.
    For session-based instruments, checks against market hours.

    Args:
        instrument: Ticker symbol.
        current_time: UTC datetime to check. Defaults to datetime.now(UTC).

    Returns:
        True if the instrument can be traded now.
    """
    session = get_session_for(instrument)
    if session is None:
        return True  # Unknown instruments — don't gate

    if session.get("market_open") is None:
        return True  # 24/7 markets

    if current_time is None:
        current_time = datetime.now(pytz.UTC)

    tz = pytz.timezone(session["timezone"])
    local_time = current_time.astimezone(tz)

    try:
        open_h, open_m = map(int, session["market_open"].split(":"))
        close_h, close_m = map(int, session["market_close"].split(":"))
    except (ValueError, KeyError):
        return True

    now_minutes = local_time.hour * 60 + local_time.minute
    open_minutes = open_h * 60 + open_m
    close_minutes = close_h * 60 + close_m

    return open_minutes <= now_minutes <= close_minutes


def get_trading_sessions_summary() -> list[dict[str, Any]]:
    """Return a human-readable summary of all trading sessions."""
    result: list[dict[str, Any]] = []
    for name, cfg in TIMEZONE_SESSIONS.items():
        result.append(
            {
                "session": name,
                "timezone": cfg["timezone"],
                "hours": f"{cfg.get('market_open', '24/7')} → {cfg.get('market_close', '24/7')}",
                "instruments": len(cfg.get("instruments", [])),
                "description": cfg.get("description", ""),
            }
        )
    return result
