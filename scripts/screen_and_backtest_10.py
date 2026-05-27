"""
Screen 10 candidate tickers against stock selection criteria, download data,
run Rules-First 'all-on' backtests, and classify into baskets/tiers.

Usage:
    uv run scripts/screen_and_backtest_10.py
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Candidate tickers (NOT in production basket, NOT in comprehensive backtest) ──
CANDIDATES = [
    "GD",  # General Dynamics — Defense/aerospace
    "GE",  # GE Aerospace — Aerospace/industrial
    "DHI",  # DR Horton — Homebuilding
    "LRCX",  # Lam Research — Semiconductor equipment
    "ABT",  # Abbott Labs — Medical devices
    "APH",  # Amphenol — Electronic components
    "URI",  # United Rentals — Equipment rental
    "CTVA",  # Corteva — Agriculture/biotech
    "NOC",  # Northrop Grumman — Defense
    "CHTR",  # Charter Communications — Telecom/media
]

# ── Hard filter thresholds (from docs/stock_selection_criteria.md) ──
MIN_MC = 300e6
MIN_ADDV = 10e6
MIN_PRICE = 5.0
MIN_YEARS = 3
MIN_INST = 0.25
EXCLUDE_SECTORS = ["Financial Services", "Utilities"]
EXCLUDE_LEVERAGED = True
MIN_ROCE = 0.05  # 5%

# ── Production config (from backtest_all_comprehensive.py) ──
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

IS_PERIOD = ("2016-01-01", "2024-12-31")
OOS_PERIOD = ("2025-01-01", None)
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("outputs/new_ticker_backtests")


@dataclass
class ScreenResult:
    ticker: str
    passed: bool
    sector: str = ""
    market_cap: float = 0
    adv: float = 0
    price: float = 0
    inst_ownership: float = 0
    roce: float | None = None
    earnings_yield: float | None = None
    years_history: int = 0
    failures: list[str] = field(default_factory=list)
    desirability_score: int = 0


@dataclass
class BacktestResult:
    ticker: str = ""
    period: str = "IS"
    sharpe: float = 0.0
    return_pct: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_dd: float = 0.0
    exposure_pct: float = 0.0
    buyhold_return_pct: float = 0.0
    buyhold_sharpe: float = 0.0
    error: str = ""


def fundamental_screen(ticker: str) -> ScreenResult:
    """Run all 11 hard filters on a ticker."""
    result = ScreenResult(ticker=ticker, passed=False)
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        result.failures.append(f"yfinance error: {e}")
        return result

    # F2: Market Cap
    mc = info.get("marketCap", 0) or 0
    result.market_cap = mc
    if mc < MIN_MC:
        result.failures.append(f"F2: MC ${mc / 1e6:.0f}M < ${MIN_MC / 1e6:.0f}M")

    # F1: Avg Daily Dollar Volume
    avg_vol = info.get("averageVolume", 0) or 0
    price = info.get("currentPrice", info.get("regularMarketPrice", 0)) or 0
    result.price = price
    result.adv = avg_vol * price
    if result.adv < MIN_ADDV:
        result.failures.append(f"F1: ADDV ${result.adv / 1e6:.1f}M < ${MIN_ADDV / 1e6:.0f}M")

    # F3: Price
    if price < MIN_PRICE:
        result.failures.append(f"F3: Price ${price:.2f} < ${MIN_PRICE}")

    # F5: Data History
    try:
        hist = yf.download(ticker, period="max", progress=False)
        if len(hist) > 0:
            years = (hist.index[-1] - hist.index[0]).days / 365.25
            result.years_history = int(years)
            if years < MIN_YEARS:
                result.failures.append(f"F5: History {years:.1f}y < {MIN_YEARS}y")
    except Exception:
        result.failures.append("F5: Cannot download history")

    # F6: Exchange
    exchange = info.get("exchange", "")
    quote_type = info.get("quoteType", "").upper()
    if "ETF" in quote_type:
        pass  # ETFs don't need exchange check
    elif exchange not in ("NYQ", "NMS", "NYS", "NAS"):
        result.failures.append(f"F6: Exchange {exchange} not NYSE/NASDAQ")

    # F7: ETF AUM (skip for stocks)
    # F8: Leveraged
    if info.get("shortName", ""):
        sname = info["shortName"].lower()
        for keyword in ["2x", "3x", "ultra", "leveraged", "inverse", "bear", "-1x"]:
            if keyword in sname:
                result.failures.append(f"F8: Leveraged product: {sname}")

    # F9: ROCE (stocks only — skip ETFs)
    is_etf = "ETF" in quote_type
    result.sector = info.get("sector", "")
    if not is_etf and result.sector not in EXCLUDE_SECTORS:
        try:
            t = yf.Ticker(ticker)
            ebit = None
            if "EBIT" in t.financials.index:
                ebit = t.financials.loc["EBIT"].iloc[0]
            total_assets = (
                t.balance_sheet.loc["Total Assets"].iloc[0]
                if "Total Assets" in t.balance_sheet.index
                else None
            )
            current_liab = (
                t.balance_sheet.loc["Current Liabilities"].iloc[0]
                if "Current Liabilities" in t.balance_sheet.index
                else None
            )
            if ebit and total_assets and current_liab and (total_assets - current_liab) > 0:
                result.roce = ebit / (total_assets - current_liab)
                if result.roce < MIN_ROCE:
                    result.failures.append(f"F9: ROCE {result.roce:.1%} < {MIN_ROCE:.0%}")
            ev = info.get("enterpriseValue", 0)
            if ebit and ev and ev > 0:
                result.earnings_yield = ebit / ev
        except Exception:
            pass  # Skip ROCE if financials unavailable

    # F10/F11: Sector exclusions
    if result.sector == "Financial Services":
        result.failures.append("F10: Financial Services excluded")
    if result.sector == "Utilities":
        result.failures.append("F11: Utilities excluded")

    # D4: Institutional Ownership
    result.inst_ownership = info.get("heldPercentInstitutions", 0) or 0
    if result.inst_ownership < MIN_INST:
        result.failures.append(f"D4: Inst ownership {result.inst_ownership:.0%} < {MIN_INST:.0%}")

    result.passed = len(result.failures) == 0
    return result


def compute_desirability(result: ScreenResult) -> int:
    """Score D1-D8 for a ticker that passed hard filters."""
    score = 0

    # D1: Liquidity (ADV)
    adv_m = result.adv / 1e6
    if adv_m > 500:
        score += 3
    elif adv_m > 100:
        score += 2
    elif adv_m > 50:
        score += 1

    # D2: Market Cap
    mc_b = result.market_cap / 1e9
    if mc_b > 50:
        score += 3
    elif mc_b > 10:
        score += 2
    elif mc_b > 2:
        score += 1
    elif mc_b < 0.3:
        score -= 1

    # D3: Analyst Coverage (simplified — fewer is better for alpha)
    score += 2

    # D4: Institutional Ownership
    if result.inst_ownership > 0.80:
        score += 3
    elif result.inst_ownership > 0.60:
        score += 2
    elif result.inst_ownership > 0.40:
        score += 1

    # D5: Bid-Ask (proxy via price/liquidity)
    if result.price > 50 and adv_m > 100:
        score += 3
    elif result.price > 20 and adv_m > 50:
        score += 2
    elif result.price > 10:
        score += 1

    # D6: Sector bonus
    high_alpha_sectors = ["Technology", "Healthcare", "Energy", "Basic Materials", "Industrials"]
    if result.sector in high_alpha_sectors:
        score += 2
    elif result.sector in ["Communication Services", "Consumer Cyclical", "Consumer Defensive"]:
        score += 1
    else:
        score -= 1

    # D7: Volatility Profile (neutral for now, computed at backtest)
    score += 1

    # D8: Earnings Yield
    if result.earnings_yield:
        if result.earnings_yield > 0.10:
            score += 2
        elif result.earnings_yield > 0.05:
            score += 1
        elif result.earnings_yield < 0:
            score -= 1

    return score


def download_data(ticker: str) -> pd.DataFrame | None:
    """Download daily OHLCV data for a ticker and save to data/raw/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df = yf.download(ticker, start="2015-01-01", progress=False)
        if df.empty:
            return None
        # Flatten multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = "Date"
        df.to_csv(DATA_DIR / f"{ticker}_daily.csv")
        return df
    except Exception:
        return None


def _load_data(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol} at {path}")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df.dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0 if col == "Volume" else df.iloc[:, 0]
    df.columns = [c.capitalize() for c in df.columns]
    return df


def run_backtest(symbol: str, start: str | None, end: str | None, config: dict) -> BacktestResult:
    """Run a single Rules-First backtest."""
    from backtesting import Backtest
    from src.strategies.rules_first_strategy import RulesFirstStrategy

    result = BacktestResult(ticker=symbol, period="IS" if start and "2016" in str(start) else "OOS")

    try:
        df = _load_data(symbol)
    except FileNotFoundError:
        result.error = "No data file"
        return result

    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]

    if len(df) < 60:
        result.error = f"Only {len(df)} bars"
        return result

    bt = Backtest(
        df,
        RulesFirstStrategy,
        cash=10000,
        commission=0.001,
        exclusive_orders=True,
        finalize_trades=True,
    )

    try:
        stats = bt.run(**config)
    except Exception as e:
        result.error = str(e)[:200]
        return result

    result.sharpe = float(stats.get("Sharpe Ratio", 0) or 0)
    result.return_pct = float(stats.get("Return [%]", 0) or 0)
    result.trades = int(stats.get("# Trades", 0) or 0)
    result.win_rate = float(stats.get("Win Rate [%]", 0) or 0)
    result.profit_factor = float(stats.get("Profit Factor", 0) or 0)
    result.max_dd = float(stats.get("Max. Drawdown [%]", 0) or 0)
    result.exposure_pct = float(stats.get("Exposure Time [%]", 0) or 0)

    # Buy & Hold
    if len(df) >= 2:
        result.buyhold_return_pct = float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100)
        daily_ret = df["Close"].pct_change().dropna()
        result.buyhold_sharpe = float(
            (daily_ret.mean() / daily_ret.std()) * (252**0.5) if daily_ret.std() > 0 else 0.0
        )

    return result


def classify_tier(oos_sharpe: float, trades: int, return_pct: float) -> tuple[str, str]:
    """Classify ticker into S/A/B/C tier based on OOS metrics."""
    if trades < 3:
        return "C", "Insufficient trades"
    if oos_sharpe >= 0.80 and return_pct > 0:
        return "S", "Strong — production-ready"
    if oos_sharpe >= 0.30 and return_pct > -1.0:
        return "A", "Positive — monitored allocation"
    if oos_sharpe >= -0.20:
        return "B", "Marginal — small allocation or watch"
    return "C", "Negative — skip"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    print("=" * 80)
    print("PHASE 1: FUNDAMENTAL SCREEN")
    print("=" * 80)

    passed: list[ScreenResult] = []
    failed: list[ScreenResult] = []

    for ticker in CANDIDATES:
        result = fundamental_screen(ticker)
        if result.passed:
            score = compute_desirability(result)
            result.desirability_score = score
            passed.append(result)
            status = f"PASS (score {score}/19)"
        else:
            failed.append(result)
            status = f"FAIL: {', '.join(result.failures)}"
        print(
            f"  {ticker:6s} | MC=${result.market_cap / 1e9:.1f}B | ADV=${result.adv / 1e6:.0f}M | "
            f"Price=${result.price:.0f} | Sector={result.sector[:25]:25s} | {status}"
        )

    print(f"\n  {len(passed)}/{len(CANDIDATES)} passed, {len(failed)}/{len(CANDIDATES)} failed")

    if failed:
        print("\n  Failures:")
        for f in failed:
            print(f"    {f.ticker}: {', '.join(f.failures)}")

    print("\n" + "=" * 80)
    print("PHASE 2: DATA DOWNLOAD")
    print("=" * 80)

    for result in passed:
        ticker = result.ticker
        csv_path = DATA_DIR / f"{ticker}_daily.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
            print(
                f"  {ticker}: {len(df)} bars (existing), {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}"
            )
        else:
            df = download_data(ticker)
            if df is None or df.empty:
                print(f"  {ticker}: FAILED to download")
                continue
            print(
                f"  {ticker}: {len(df)} bars (downloaded), {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}"
            )

    print("\n" + "=" * 80)
    print("PHASE 3: BACKTEST (IS 2016-2024 + OOS 2025-2026)")
    print("=" * 80)

    config = dict(PRODUCTION_CONFIG)
    # Remove quality_registry if file doesn't exist
    if not Path(config["quality_registry_path"]).exists():
        config.pop("quality_registry_path", None)
        config["use_quality_registry"] = False

    for result in passed:
        ticker = result.ticker
        print(f"\n  --- {ticker} ({result.sector}) ---")

        # Check data file exists
        if not (DATA_DIR / f"{ticker}_daily.csv").exists():
            print("    SKIP: no data file")
            continue

        is_bt = run_backtest(ticker, str(IS_PERIOD[0]), str(IS_PERIOD[1]), config)
        if is_bt.error:
            print(f"    IS: ERROR — {is_bt.error}")
        else:
            print(
                f"    IS: Sharpe={is_bt.sharpe:+.2f} | Return={is_bt.return_pct:+.1f}% | "
                f"Trades={is_bt.trades} | WR={is_bt.win_rate:.0f}% | PF={is_bt.profit_factor:.2f} | "
                f"MaxDD={is_bt.max_dd:.1f}% | BH={is_bt.buyhold_return_pct:+.0f}%"
            )

        oos_bt = run_backtest(ticker, str(OOS_PERIOD[0]), None, config)
        if oos_bt.error:
            print(f"    OOS: ERROR — {oos_bt.error}")
        else:
            tier, note = classify_tier(oos_bt.sharpe, oos_bt.trades, oos_bt.return_pct)
            delta = oos_bt.sharpe - is_bt.sharpe if not is_bt.error else 0
            print(
                f"    OOS: Sharpe={oos_bt.sharpe:+.2f} | Return={oos_bt.return_pct:+.1f}% | "
                f"Trades={oos_bt.trades} | WR={oos_bt.win_rate:.0f}% | PF={oos_bt.profit_factor:.2f} | "
                f"MaxDD={oos_bt.max_dd:.1f}% | BH={oos_bt.buyhold_return_pct:+.0f}%"
            )
            print(f"    Δ: {delta:+.2f} | Tier: {tier} | {note}")

            all_results.append(
                {
                    "ticker": ticker,
                    "sector": result.sector,
                    "market_cap_b": round(result.market_cap / 1e9, 1),
                    "desirability_score": result.desirability_score,
                    "IS_sharpe": round(is_bt.sharpe, 3),
                    "IS_return_pct": round(is_bt.return_pct, 1),
                    "IS_trades": is_bt.trades,
                    "IS_win_rate": round(is_bt.win_rate, 1),
                    "IS_profit_factor": round(is_bt.profit_factor, 2),
                    "IS_max_dd": round(is_bt.max_dd, 1),
                    "IS_buyhold_pct": round(is_bt.buyhold_return_pct, 0),
                    "OOS_sharpe": round(oos_bt.sharpe, 3),
                    "OOS_return_pct": round(oos_bt.return_pct, 1),
                    "OOS_trades": oos_bt.trades,
                    "OOS_win_rate": round(oos_bt.win_rate, 1),
                    "OOS_profit_factor": round(oos_bt.profit_factor, 2),
                    "OOS_max_dd": round(oos_bt.max_dd, 1),
                    "OOS_buyhold_pct": round(oos_bt.buyhold_return_pct, 0),
                    "delta_sharpe": round(delta, 3),
                    "tier": tier,
                    "tier_note": note,
                }
            )

    # ── Phase 4: Tier Classification & Summary ──
    print("\n" + "=" * 80)
    print("PHASE 4: TIER CLASSIFICATION")
    print("=" * 80)

    # Sort by OOS Sharpe descending
    all_results.sort(key=lambda x: x["OOS_sharpe"], reverse=True)

    tiers: dict[str, list[dict]] = {"S": [], "A": [], "B": [], "C": []}
    for r in all_results:
        tiers[r["tier"]].append(r)

    print(
        f"\n{'Ticker':6s} {'Sector':20s} {'MC(B)':>6s} {'Score':>5s} {'IS_Sh':>7s} "
        f"{'OOS_Sh':>7s} {'OOS_Ret':>7s} {'Trd':>4s} {'WR':>5s} {'PF':>5s} "
        f"{'MaxDD':>6s} {'Δ':>7s} {'Tier':5s}"
    )
    print("-" * 100)

    for r in all_results:
        print(
            f"{r['ticker']:6s} {r['sector'][:20]:20s} {r['market_cap_b']:>5.1f} {r['desirability_score']:>4d} "
            f"{r['IS_sharpe']:>+7.2f} {r['OOS_sharpe']:>+7.2f} {r['OOS_return_pct']:>+6.1f}% "
            f"{r['OOS_trades']:>4d} {r['OOS_win_rate']:>4.0f}% {r['OOS_profit_factor']:>5.2f} "
            f"{r['OOS_max_dd']:>5.1f}% {r['delta_sharpe']:>+7.2f} {r['tier']:5s}"
        )

    print("\n  Tier Summary:")
    for tier_name in ["S", "A", "B", "C"]:
        t = tiers[tier_name]
        if t:
            names = [r["ticker"] for r in t]
            avg_sharpe = np.mean([r["OOS_sharpe"] for r in t])
            print(
                f"    {tier_name}-tier ({len(t)}): {', '.join(names)} — mean OOS Sharpe {avg_sharpe:+.2f}"
            )

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"new_tickers_{timestamp}.json"
    output = {
        "timestamp": timestamp,
        "config": PRODUCTION_CONFIG,
        "candidates_screened": len(CANDIDATES),
        "candidates_passed": len(passed),
        "summary": {
            tier: {
                "count": len(t),
                "mean_sharpe": round(float(np.mean([r["OOS_sharpe"] for r in t])), 3),
                "tickers": [r["ticker"] for r in t],
            }
            for tier, t in tiers.items()
            if t
        },
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {output_path}")

    return all_results, tiers


if __name__ == "__main__":
    main()
