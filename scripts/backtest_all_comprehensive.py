"""
Comprehensive batch backtest — all tickers, IS + OOS, production config.

Runs the Phase 07 production paper-trading config (mr=0.70, multi-TP, quality registry)
across ALL available tickers in IS (2016-2024) and OOS (2025-2026) periods.
Produces detailed per-batch JSON + comprehensive markdown summary.

Usage:
    # Run a specific batch
    uv run scripts/backtest_all_comprehensive.py --batch 1
    uv run scripts/backtest_all_comprehensive.py --batch all
    uv run scripts/backtest_all_comprehensive.py --tickers SPY,QQQ,GLD
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.backtest_rules_first import run_single

PRODUCTION_CONFIG = dict(
    entry_threshold=0.55,
    exit_threshold=0.30,
    trail_stop_atr=3.0,
    min_reliability=0.70,
    confluence_bonus=0.10,
    volume_confirm=True,
    use_multi_tp=True,
    use_quality_registry=True,
    quality_registry_path="reports/pattern_gate/all_patterns.json",
    use_vix_gate=False,
    use_yield_curve_gate=False,
)

# Phase 25: Per-instrument best params from single source of truth.
from src.per_instrument.instrument_config import (
    get_config_for as _cfg_get,
    INSTRUMENT_CONFIG as _INSTR_CFG,
)

PER_INSTRUMENT_BEST: dict[str, dict] = {
    sym: {
        "entry_threshold": c.get("et", cfg.get("entry_threshold", 0.55)),
        "min_reliability": c.get("mr", cfg.get("min_reliability", 0.70)),
        "trail_stop_atr": c.get("tsa", cfg.get("trail_stop_atr", 3.0)),
        "confluence_bonus": c.get("cb", cfg.get("confluence_bonus", 0.10)),
    }
    for sym, c in _INSTR_CFG.items()
    for cfg in [_cfg_get(sym, use_best=True)]
    if c.get("tier") in ("S", "A", "B")
}


def _get_config_for(symbol: str, use_best: bool = False) -> dict:
    if use_best and symbol in PER_INSTRUMENT_BEST:
        cfg = dict(PRODUCTION_CONFIG)
        cfg.update(PER_INSTRUMENT_BEST[symbol])
        return cfg
    return dict(PRODUCTION_CONFIG)


IS_PERIOD = ("2016-01-01", "2024-12-31")
OOS_PERIOD = ("2025-01-01", None)

CATEGORY_MAP: dict[str, str] = {
    # Indices
    "SPY": "Index-LargeCap",
    "QQQ": "Index-NASDAQ",
    "IWM": "Index-SmallCap",
    "D": "Index-Dow",
    "EEM": "Index-EM",
    # Sector ETFs
    "XLK": "Sector-Tech",
    "XLF": "Sector-Fin",
    "XLE": "Sector-Energy",
    "XLV": "Sector-Health",
    # Commodities
    "GLD": "Commodity-Gold",
    "SLV": "Commodity-Silver",
    "IAU": "Commodity-Gold",
    "CL": "Commodity-Oil",
    # Bonds
    "TLT": "Bond-Treasury",
    # Crypto
    "BTC_USD": "Crypto-BTC",
    "ETH_USD": "Crypto-ETH",
    # Forex
    "EURUSD_X": "Forex-EURUSD",
    # US Large Cap
    "AAPL": "Stock-Tech",
    "MSFT": "Stock-Tech",
    "NVDA": "Stock-Tech",
    "GOOGL": "Stock-Tech",
    "META": "Stock-Tech",
    "AMD": "Stock-Tech",
    "INTC": "Stock-Tech",
    "CRM": "Stock-Tech",
    "JPM": "Stock-Fin",
    "BAC": "Stock-Fin",
    "GS": "Stock-Fin",
    "MS": "Stock-Fin",
    "WFC": "Stock-Fin",
    "C": "Stock-Fin",
    "BLK": "Stock-Fin",
    "SCHW": "Stock-Fin",
    "JNJ": "Stock-Health",
    "UNH": "Stock-Health",
    "ABBV": "Stock-Health",
    "MRK": "Stock-Health",
    "AMGN": "Stock-Health",
    "PFE": "Stock-Health",
    "GILD": "Stock-Health",
    "VRTX": "Stock-Health",
    "REGN": "Stock-Health",
    "BIIB": "Stock-Health",
    "LLY": "Stock-Health",
    "TMO": "Stock-Health",
    "BMY": "Stock-Health",
    "XOM": "Stock-Energy",
    "CVX": "Stock-Energy",
    "COP": "Stock-Energy",
    "EOG": "Stock-Energy",
    "SLB": "Stock-Energy",
    "HAL": "Stock-Energy",
    "MPC": "Stock-Energy",
    "PSX": "Stock-Energy",
    "OXY": "Stock-Energy",
    "KO": "Stock-Cons",
    "PEP": "Stock-Cons",
    "PG": "Stock-Cons",
    "WMT": "Stock-Cons",
    "COST": "Stock-Cons",
    "HD": "Stock-Cons",
    "MO": "Stock-Cons",
    "KMB": "Stock-Cons",
    "SO": "Stock-Util",
    "DUK": "Stock-Util",
    "NEE": "Stock-Util",
    "AEP": "Stock-Util",
    "EXC": "Stock-Util",
    "SRE": "Stock-Util",
    "ED": "Stock-Util",
    "PEG": "Stock-Util",
    "CAT": "Stock-Ind",
    "LMT": "Stock-Ind",
    "FDX": "Stock-Ind",
    "UPS": "Stock-Ind",
    "UNP": "Stock-Ind",
    "NSC": "Stock-Ind",
    "DAL": "Stock-Ind",
    "UAL": "Stock-Ind",
    "LUV": "Stock-Ind",
    "AAL": "Stock-Ind",
    "CSX": "Stock-Ind",
    "FCX": "Stock-Material",
    "NEM": "Stock-Material",
    "O": "Stock-REIT",
    "PLD": "Stock-REIT",
    "SPG": "Stock-REIT",
    "PSA": "Stock-REIT",
    "AVB": "Stock-REIT",
    "AMT": "Stock-REIT",
    "WELL": "Stock-REIT",
    "EQR": "Stock-REIT",
    "DHR": "Stock-Conglom",
    "GOLD": "Stock-Material",
    "PM": "Stock-Cons",
    # Mid/Small
    "STLD": "MidCap-Steel",
    "NUE": "MidCap-Steel",
    "VLO": "MidCap-Refinery",
    "CAR": "MidCap-Consumer",
    "KODK": "MicroCap-Special",
    "JOE": "SmallCap-RE",
    "HIFS": "MicroCap-Bank",
    "CRVL": "MicroCap-Service",
    "AEM": "MidCap-Gold",
    # China
    "CN_SHCOMP": "China-Index",
    "CN_CSI300": "China-Index",
    "CN_CSI500": "China-Index",
    "CN_SSE50": "China-Index",
    "CN_CHINEXT": "China-Index",
    "CN_STAR50": "China-Index",
    "CN_SZCOMP": "China-Index",
    "CN_Moutai": "China-Stock",
    "CN_Wuliangye": "China-Stock",
    "CN_PingAn": "China-Stock",
    "CN_CMB": "China-Stock",
    "CN_PetroChina": "China-Stock",
    "CN_ChinaShenhua": "China-Stock",
    "CN_BYD": "China-Stock",
    "CN_CATL": "China-Stock",
    "CN_Midea": "China-Stock",
    "CN_Hengrui": "China-Stock",
    "CN_YangtzePower": "China-Stock",
    "CN_SMIC": "China-Stock",
    # Hong Kong
    "HK_Tencent": "HK-Stock",
    "HK_Alibaba_HK": "HK-Stock",
    "HK_Meituan": "HK-Stock",
    "HK_Xiaomi": "HK-Stock",
    "HK_BYD_HK": "HK-Stock",
    "HK_JD_HK": "HK-Stock",
    "HK_Kuaishou": "HK-Stock",
    "HK_LiAuto": "HK-Stock",
    "HK_AIA": "HK-Stock",
    "HK_HSBC": "HK-Stock",
    "HK_HKEX": "HK-Stock",
    "HK_Mixue": "HK-Stock",
}

BATCHES: dict[str, list[str]] = {
    "1_indices_etfs": [
        "SPY",
        "QQQ",
        "D",
        "EEM",
        "XLK",
        "XLF",
        "XLE",
        "XLV",
        "GLD",
        "SLV",
        "IAU",
        "CL",
        "BTC_USD",
        "ETH_USD",
    ],
    "2_largecap_tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD", "INTC", "CRM", "COST"],
    "3_largecap_fin": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW"],
    "4_largecap_health": [
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
    ],
    "5_largecap_energy": ["XOM", "CVX", "COP", "EOG", "SLB", "HAL", "MPC", "PSX", "OXY"],
    "6_largecap_consumer": ["KO", "PEP", "PG", "WMT", "HD", "MO", "KMB", "PM"],
    "7_largecap_util_ind": [
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
    ],
    "8_largecap_reit_material": [
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
    ],
    "9_midcap_special": ["STLD", "NUE", "VLO", "CAR", "KODK", "JOE", "HIFS", "CRVL"],
    "10_china": [
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
    "11_hk": [
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
    "12_forex_extra": [],
}


@dataclass
class TickerResult:
    symbol: str
    category: str
    is_return: float
    is_sharpe: float
    is_sortino: float
    is_calmar: float
    is_maxdd: float
    is_trades: int
    is_winrate: float
    is_pf: float
    is_exp: float
    is_annual: float
    is_start: str
    is_end: str
    oos_return: float
    oos_sharpe: float
    oos_sortino: float
    oos_calmar: float
    oos_maxdd: float
    oos_trades: int
    oos_winrate: float
    oos_pf: float
    oos_exp: float
    oos_annual: float
    oos_start: str
    oos_end: str
    oos_delta_sharpe: float = 0.0
    status: str = "ok"


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except (ValueError, TypeError):
        return default


def run_ticker(symbol: str, cash: float = 100_000, use_best: bool = False) -> TickerResult | None:
    """Run IS + OOS for a single ticker using production config."""
    cat = CATEGORY_MAP.get(symbol, "Other")
    config = _get_config_for(symbol, use_best)

    # IS
    try:
        is_stats = run_single(
            symbol,
            cash=cash,
            start=IS_PERIOD[0],
            end=IS_PERIOD[1],
            **config,
        )
    except Exception as e:
        print(f"    IS FAIL: {e}")
        return None

    # OOS
    try:
        oos_stats = run_single(
            symbol,
            cash=cash,
            start=OOS_PERIOD[0],
            end=OOS_PERIOD[1],
            **config,
        )
    except Exception as e:
        print(f"    OOS FAIL: {e}")
        oos_stats = {}

    is_sharpe = safe_float(is_stats.get("sharpe"))
    oos_sharpe = safe_float(oos_stats.get("sharpe"))

    r = TickerResult(
        symbol=symbol,
        category=cat,
        is_return=safe_float(is_stats.get("return_pct")),
        is_sharpe=is_sharpe,
        is_sortino=safe_float(is_stats.get("sortino")),
        is_calmar=safe_float(is_stats.get("calmar")),
        is_maxdd=safe_float(is_stats.get("max_dd_pct")),
        is_trades=int(is_stats.get("trades", 0)),
        is_winrate=safe_float(is_stats.get("win_rate_pct")),
        is_pf=safe_float(is_stats.get("profit_factor")),
        is_exp=safe_float(is_stats.get("exposure_pct")),
        is_annual=safe_float(
            is_stats.get("annual_return_pct") if "annual_return_pct" in is_stats else 0
        ),
        is_start=str(is_stats.get("start", "")),
        is_end=str(is_stats.get("end", "")),
        oos_return=safe_float(oos_stats.get("return_pct")),
        oos_sharpe=oos_sharpe,
        oos_sortino=safe_float(oos_stats.get("sortino")),
        oos_calmar=safe_float(oos_stats.get("calmar")),
        oos_maxdd=safe_float(oos_stats.get("max_dd_pct")),
        oos_trades=int(oos_stats.get("trades", 0)),
        oos_winrate=safe_float(oos_stats.get("win_rate_pct")),
        oos_pf=safe_float(oos_stats.get("profit_factor")),
        oos_exp=safe_float(oos_stats.get("exposure_pct")),
        oos_annual=safe_float(
            oos_stats.get("annual_return_pct") if "annual_return_pct" in oos_stats else 0
        ),
        oos_start=str(oos_stats.get("start", "")),
        oos_end=str(oos_stats.get("end", "")),
        oos_delta_sharpe=oos_sharpe - is_sharpe,
        status="ok",
    )
    # Phase 25: Log performance to cumulative ledger
    try:
        from src.per_instrument.performance_tracker import log_performance

        def _verdict(sh: float) -> str:
            if sh > 0.10:
                return "PASS"
            if sh < -0.05:
                return "FAIL"
            return "MARGINAL"

        for period, stats in [("IS", is_stats), ("OOS", oos_stats)]:
            if not stats:
                continue
            s = safe_float(stats.get("sharpe"))
            date_range = (
                f"{stats.get('start', '?')}:{stats.get('end', '?')}"
                if stats.get("start")
                else f"{IS_PERIOD[0]}:{IS_PERIOD[1]}"
                if period == "IS"
                else f"{OOS_PERIOD[0]}:today"
            )
            log_performance(
                instrument=symbol,
                period=period,
                date_range=date_range,
                strategy="rules_first",
                config={k: v for k, v in config.items()},
                metrics={
                    "sharpe": s,
                    "return_pct": safe_float(stats.get("return_pct")),
                    "trades": int(stats.get("trades", 0)),
                    "win_rate": safe_float(stats.get("win_rate_pct")),
                    "profit_factor": safe_float(stats.get("profit_factor")),
                    "max_dd_pct": safe_float(stats.get("max_dd_pct")),
                    "exposure_pct": safe_float(stats.get("exposure_pct")),
                    "sortino": safe_float(stats.get("sortino")),
                    "calmar": safe_float(stats.get("calmar")),
                },
                benchmark={"buyhold_return_pct": 0, "buyhold_sharpe": 0},
                verdict=_verdict(s),
            )
    except Exception:
        pass  # Logging is best-effort

    return r


def result_to_dict(r: TickerResult) -> dict:
    return {k: v for k, v in r.__dict__.items()}


def print_batch_header(batch_name: str, batch_tickers: list[str], use_best: bool = False) -> None:
    print(f"\n{'=' * 140}")
    print(f" BATCH: {batch_name}  ({len(batch_tickers)} tickers)")
    if use_best:
        print(" Config: per-instrument best (from BESTS.md tuning sweep)")
    else:
        print(
            f" Config: mr={PRODUCTION_CONFIG['min_reliability']}  et={PRODUCTION_CONFIG['entry_threshold']}  "
            f"trail={PRODUCTION_CONFIG['trail_stop_atr']}  multi_tp={PRODUCTION_CONFIG['use_multi_tp']}  "
            f"quality_registry={PRODUCTION_CONFIG['use_quality_registry']}  "
            f"vix_gate={PRODUCTION_CONFIG['use_vix_gate']}  yield_gate={PRODUCTION_CONFIG['use_yield_curve_gate']}"
        )
    print(f" IS: {IS_PERIOD[0]} -> {IS_PERIOD[1]}   OOS: {OOS_PERIOD[0]} -> today")
    print(f"{'=' * 140}\n")


def print_comprehensive_table(results: list[TickerResult], period: str = "IS") -> None:
    """Print comprehensive detailed table for a period."""
    if not results:
        print("  No results.")
        return

    if period == "IS":
        header = [
            "Rank",
            "Symbol",
            "Category",
            "Return%",
            "Sharpe",
            "Sortino",
            "Calmar",
            "MaxDD%",
            "Trades",
            "Win%",
            "PF",
            "Exp%",
            "Ann%",
        ]
    else:
        header = [
            "Rank",
            "Symbol",
            "Category",
            "Return%",
            "Sharpe",
            "Sortino",
            "Calmar",
            "MaxDD%",
            "Trades",
            "Win%",
            "PF",
            "Exp%",
            "d Sharpe",
        ]

    # Sort by Sharpe descending
    if period == "IS":
        sorted_results = sorted(results, key=lambda r: r.is_sharpe, reverse=True)
    else:
        sorted_results = sorted(results, key=lambda r: r.oos_sharpe, reverse=True)

    widths = [6, 12, 18] + [9] * 10
    fmt_parts = [f"{{:<{w}}}" for w in widths]
    fmt_line = "  ".join(fmt_parts)

    print(f"\n{'=' * 135}")
    print(f"  {period} RESULTS -- Sorted by Sharpe")
    print(f"{'=' * 135}")
    print(fmt_line.format(*header))
    print("  ".join("-" * w for w in widths))

    for rank, r in enumerate(sorted_results, 1):
        if period == "IS":
            sharpe = r.is_sharpe
            if r.is_trades == 0 and sharpe == 0:
                sharpe_str = "NaN"
            else:
                sharpe_str = f"{sharpe:.3f}"
            row = [
                str(rank),
                r.symbol,
                r.category[:17],
                f"{r.is_return:+.2f}",
                sharpe_str,
                f"{r.is_sortino:.3f}",
                f"{r.is_calmar:.3f}",
                f"{r.is_maxdd:.2f}",
                str(r.is_trades),
                f"{r.is_winrate:.1f}",
                f"{r.is_pf:.2f}",
                f"{r.is_exp:.1f}",
                f"{r.is_annual:+.2f}",
            ]
        else:
            sharpe = r.oos_sharpe
            if r.oos_trades == 0 and sharpe == 0:
                sharpe_str = "NaN"
            else:
                sharpe_str = f"{sharpe:.3f}"
            row = [
                str(rank),
                r.symbol,
                r.category[:17],
                f"{r.oos_return:+.2f}",
                sharpe_str,
                f"{r.oos_sortino:.3f}",
                f"{r.oos_calmar:.3f}",
                f"{r.oos_maxdd:.2f}",
                str(r.oos_trades),
                f"{r.oos_winrate:.1f}",
                f"{r.oos_pf:.2f}",
                f"{r.oos_exp:.1f}",
                f"{r.oos_delta_sharpe:+.3f}",
            ]
        print(fmt_line.format(*row))

    print("  ".join("-" * w for w in widths))
    print()


def print_summary_insights(results: list[TickerResult]) -> None:
    """Print comprehensive summary insights."""
    print(f"\n{'=' * 140}")
    print(" BATCH SUMMARY & INSIGHTS")
    print(f"{'=' * 140}")

    n = len(results)
    is_pos = [r for r in results if r.is_sharpe > 0.01]
    oos_pos = [r for r in results if r.oos_sharpe > 0.01]
    is_profit = [r for r in results if r.is_return > 0]
    oos_profit = [r for r in results if r.oos_return > 0]
    is_traded = [r for r in results if r.is_trades > 0]
    oos_traded = [r for r in results if r.oos_trades > 0]
    oos_better = [r for r in results if r.oos_delta_sharpe > 0.01]

    print(f"\n  Total tickers: {n}")
    print("\n  --- IS (2016-2024) ---")
    print(f"  Positive Sharpe (>0.01):  {len(is_pos):>3}/{n} ({100 * len(is_pos) / n:.0f}%)")
    print(
        f"  Profitable (Return > 0):    {len(is_profit):>3}/{n} ({100 * len(is_profit) / n:.0f}%)"
    )
    print(
        f"  Traded (>0 trades):         {len(is_traded):>3}/{n} ({100 * len(is_traded) / n:.0f}%)"
    )
    is_sharpes = [r.is_sharpe for r in results if r.is_trades > 0]
    if is_sharpes:
        print(f"  Mean Sharpe (traded only):  {np.mean(is_sharpes):.3f}")
        print(f"  Median Sharpe:              {np.median(is_sharpes):.3f}")

    print("\n  --- OOS (2025-2026) ---")
    print(f"  Positive Sharpe (>0.01):  {len(oos_pos):>3}/{n} ({100 * len(oos_pos) / n:.0f}%)")
    print(
        f"  Profitable (Return > 0):    {len(oos_profit):>3}/{n} ({100 * len(oos_profit) / n:.0f}%)"
    )
    print(
        f"  Traded (>0 trades):         {len(oos_traded):>3}/{n} ({100 * len(oos_traded) / n:.0f}%)"
    )
    print(
        f"  OOS > IS Sharpe:            {len(oos_better):>3}/{n} ({100 * len(oos_better) / n:.0f}%)"
    )
    oos_sharpes = [r.oos_sharpe for r in results if r.oos_trades > 0]
    if oos_sharpes:
        print(f"  Mean Sharpe (traded only):  {np.mean(oos_sharpes):.3f}")
        print(f"  Median Sharpe:              {np.median(oos_sharpes):.3f}")

    # Top 10 IS
    print("\n  --- Top 10 IS Sharpe ---")
    top_is = sorted(results, key=lambda r: r.is_sharpe, reverse=True)[:10]
    for r in top_is:
        print(
            f"  {r.symbol:>12s}  {r.category:<20s}  Sharpe={r.is_sharpe:+.3f}  "
            f"Ret={r.is_return:+.1f}%  Trades={r.is_trades}  Win={r.is_winrate:.1f}%  PF={r.is_pf:.2f}"
        )

    # Top 10 OOS
    print("\n  --- Top 10 OOS Sharpe ---")
    top_oos = sorted(results, key=lambda r: r.oos_sharpe, reverse=True)[:10]
    for r in top_oos:
        print(
            f"  {r.symbol:>12s}  {r.category:<20s}  Sharpe={r.oos_sharpe:+.3f}  "
            f"Ret={r.oos_return:+.1f}%  Trades={r.oos_trades}  Win={r.oos_winrate:.1f}%  "
            f"d={r.oos_delta_sharpe:+.3f}"
        )

    # Bottom 5 IS
    print("\n  --- Bottom 5 IS Sharpe ---")
    bot_is = sorted(results, key=lambda r: r.is_sharpe)[:5]
    for r in bot_is:
        print(
            f"  {r.symbol:>12s}  {r.category:<20s}  Sharpe={r.is_sharpe:+.3f}  "
            f"Ret={r.is_return:+.1f}%  Trades={r.is_trades}"
        )

    # Bottom 5 OOS
    print("\n  --- Bottom 5 OOS Sharpe ---")
    bot_oos = sorted(results, key=lambda r: r.oos_sharpe)[:5]
    for r in bot_oos:
        print(
            f"  {r.symbol:>12s}  {r.category:<20s}  Sharpe={r.oos_sharpe:+.3f}  "
            f"Ret={r.oos_return:+.1f}%  Trades={r.oos_trades}"
        )

    # Category summary
    print("\n  --- By Category ---")
    cats: dict[str, list[TickerResult]] = {}
    for r in results:
        cat_key = r.category.split("-")[0] if "-" in r.category else r.category
        cats.setdefault(cat_key, []).append(r)
    print(
        f"  {'Category':<20s} {'N':>3s}  {'IS Sharpe':>9s}  {'OOS Sharpe':>9s}  {'IS Ret%':>8s}  "
        f"{'OOS Ret%':>8s}  {'dSharpe':>8s}"
    )
    print(f"  {'-' * 20} {'-' * 3} {'-' * 9} {'-' * 9} {'-' * 8} {'-' * 8} {'-' * 8}")
    for cat_name, items in sorted(cats.items()):
        is_s = (
            np.mean([r.is_sharpe for r in items if r.is_trades > 0])
            if any(r.is_trades > 0 for r in items)
            else 0
        )
        oos_s = (
            np.mean([r.oos_sharpe for r in items if r.oos_trades > 0])
            if any(r.oos_trades > 0 for r in items)
            else 0
        )
        is_r = np.mean([r.is_return for r in items])
        oos_r = np.mean([r.oos_return for r in items])
        delta_s = oos_s - is_s
        print(
            f"  {cat_name:<20s} {len(items):>3d}  {is_s:>+9.3f}  {oos_s:>+9.3f}  "
            f"{is_r:>+8.1f}  {oos_r:>+8.1f}  {delta_s:>+8.3f}"
        )

    # IS->OOS correlation
    if len(results) >= 5:
        is_v = [r.is_sharpe for r in results if r.is_trades > 0]
        oos_v = [r.oos_sharpe for r in results if r.oos_trades > 0]
        if len(is_v) >= 5 and len(oos_v) >= 5:
            # Simple Pearson correlation
            common = [
                (r.is_sharpe, r.oos_sharpe) for r in results if r.is_trades > 0 and r.oos_trades > 0
            ]
            if len(common) >= 5:
                is_arr = np.array([c[0] for c in common])
                oos_arr = np.array([c[1] for c in common])
                corr = np.corrcoef(is_arr, oos_arr)[0, 1]
                print(f"\n  IS -> OOS Sharpe correlation: {corr:.3f} (n={len(common)})")

    print(f"\n{'=' * 140}\n")


def save_batch_json(results: list[TickerResult], batch_name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"batch_{batch_name}.json"
    data = {
        "batch": batch_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            **{k: v for k, v in PRODUCTION_CONFIG.items()},
            "is_period": list(IS_PERIOD),
            "oos_period": list(OOS_PERIOD),
        },
        "n_tickers": len(results),
        "results": [result_to_dict(r) for r in results],
    }
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"  Results saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive all-ticker backtest")
    parser.add_argument("--batch", default=None, help="Batch name or 'all'")
    parser.add_argument("--tickers", default=None, help="Comma-separated ticker list")
    parser.add_argument("--batch-index", type=int, default=None, help="Run batch by index (1-N)")
    parser.add_argument("--list-batches", action="store_true", help="List available batches")
    parser.add_argument(
        "--output-dir", default="reports/comprehensive_batch", help="Output directory"
    )
    parser.add_argument("--cash", type=float, default=100_000, help="Initial cash")
    parser.add_argument("--period", default="both", choices=["is", "oos", "both"])
    parser.add_argument(
        "--use-best",
        action="store_true",
        help="Use per-instrument best tuned params from BESTS.md",
    )
    args = parser.parse_args()

    if args.list_batches:
        for i, (name, tickers) in enumerate(BATCHES.items(), 1):
            print(f"  Batch {i}: {name} ({len(tickers)} tickers): {', '.join(tickers[:5])}...")
        return

    output_dir = project_root / args.output_dir

    # Determine which batches/tickers to run
    batch_list: list[tuple[str, list[str]]] = []
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
        batch_list = [("custom", tickers)]
    elif args.batch_index:
        batch_names = list(BATCHES.keys())
        if 1 <= args.batch_index <= len(batch_names):
            name = batch_names[args.batch_index - 1]
            batch_list = [(name.replace("_", " "), BATCHES[name])]
    elif args.batch and args.batch != "all":
        if args.batch in BATCHES:
            batch_list = [(args.batch.replace("_", " "), BATCHES[args.batch])]
        else:
            print(f"Unknown batch: {args.batch}")
            print(f"Available: {', '.join(BATCHES.keys())}")
            return
    else:
        # All batches
        batch_list = [(name.replace("_", " "), tickers) for name, tickers in BATCHES.items()]

    all_batch_results: dict[str, list[TickerResult]] = {}

    start_time = time.time()
    total_tickers = sum(len(tickers) for _, tickers in batch_list)
    processed = 0

    for batch_name, tickers in batch_list:
        print_batch_header(batch_name, tickers, use_best=args.use_best)
        batch_results: list[TickerResult] = []

        for i, symbol in enumerate(tickers):
            processed += 1
            t0 = time.time()
            print(f"  [{processed}/{total_tickers}] {symbol:>14s} ... ", end="", flush=True)
            r = run_ticker(symbol, cash=args.cash, use_best=args.use_best)
            elapsed = time.time() - t0
            if r:
                batch_results.append(r)
                if r.is_trades > 0:
                    print(
                        f"IS Sharpe={r.is_sharpe:+.3f} Ret={r.is_return:+.1f}% "
                        f"T={r.is_trades} OOS Sharpe={r.oos_sharpe:+.3f} Ret={r.oos_return:+.1f}% "
                        f"T={r.oos_trades}  ({elapsed:.1f}s)"
                    )
                else:
                    print(f"NO TRADES ({elapsed:.1f}s)")
            else:
                print(f"SKIPPED ({elapsed:.1f}s)")

        if batch_results:
            print_comprehensive_table(batch_results, "IS")
            print_comprehensive_table(batch_results, "OOS")
            print_summary_insights(batch_results)
            save_batch_json(batch_results, batch_name, output_dir)
            all_batch_results[batch_name] = batch_results
        else:
            print(f"  No valid results in batch {batch_name}.")

    total_elapsed = time.time() - start_time
    print(f"\n{'=' * 140}")
    print(
        f" ALL BATCHES COMPLETE — {processed} tickers processed in {total_elapsed:.1f}s ({total_elapsed / processed:.1f}s avg)"
    )
    print(f"{'=' * 140}")


if __name__ == "__main__":
    main()
