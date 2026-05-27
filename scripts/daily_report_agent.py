"""P28-22: Automated daily backtest reports agent.

Cron-based daily summary script for the production basket. Generates
personalized email-format reports with signal status, portfolio health,
and regime conditions. Patterned after OpenStock's Inngest + Gemini
daily workflow.

Source: OpenStock Inngest cron + Gemini AI patterns.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "config_files" / "production_basket.yaml"
_OUTPUT_DIR = _PROJECT_ROOT / "reports" / "daily"

try:
    import yaml
except ImportError:
    yaml = None

_DEFAULT_BASKET = {
    "S": ["XLK", "XLE", "GLD", "SPY", "SLV", "QQQ"],
    "A": ["NUE", "STLD", "HAL", "MPC", "EOG"],
    "B": ["INTC", "AMD", "LMT", "JNJ", "MRK", "NEM"],
}

_SENTIMENT_MAP: dict[str, str] = {
    "bullish": "\U0001f7e2",
    "bearish": "\U0001f534",
    "neutral": "\U000026ab",
}


@dataclass
class TickerStatus:
    """Daily status for a single ticker."""

    ticker: str
    tier: str
    price: float
    change_pct: float
    rsi: float | None
    above_ma50: bool | None
    above_ma200: bool | None
    atr_pct: float | None
    signal_count: int
    sentiment: str


def load_basket(config_path: Optional[Path] = None) -> dict[str, list[str]]:
    """Load the production basket from YAML config.

    Args:
        config_path: Path to basket YAML. Defaults to config_files/production_basket.yaml.

    Returns:
        Dict mapping tier name to list of tickers.
    """
    config_path = config_path or _CONFIG_PATH
    if yaml and config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
            if config and "basket" in config:
                return config["basket"]
    if config_path.exists():
        with open(config_path) as f:
            content = f.read()
    return _DEFAULT_BASKET


def fetch_ticker_status(ticker: str, tier: str) -> TickerStatus:
    """Fetch current market status for a single ticker.

    Args:
        ticker: Yahoo Finance ticker symbol.
        tier: Basket tier label (S/A/B).

    Returns:
        TickerStatus with price, change, RSI, MA status, ATR.
    """
    end = datetime.now()
    start = end - timedelta(days=365)

    try:
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            return TickerStatus(
                ticker=ticker,
                tier=tier,
                price=0.0,
                change_pct=0.0,
                rsi=None,
                above_ma50=None,
                above_ma200=None,
                atr_pct=None,
                signal_count=0,
                sentiment="unknown",
            )

        close = df["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        price = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else price
        change_pct = (price - prev_close) / prev_close * 100 if prev_close > 0 else 0.0

        ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        rsi = _compute_rsi(close)

        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]

        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else None
        atr_pct = (atr / price * 100) if atr and price > 0 else None

        sentiment = "neutral"
        if rsi is not None:
            if rsi > 60 and (ma50 and price > ma50):
                sentiment = "bullish"
            elif rsi < 40 and (ma50 and price < ma50):
                sentiment = "bearish"

        return TickerStatus(
            ticker=ticker,
            tier=tier,
            price=price,
            change_pct=round(change_pct, 2),
            rsi=round(rsi, 1) if rsi else None,
            above_ma50=(price > ma50) if ma50 else None,
            above_ma200=(price > ma200) if ma200 else None,
            atr_pct=round(atr_pct, 2) if atr_pct else None,
            signal_count=0,
            sentiment=sentiment,
        )

    except Exception as e:
        logger.warning(f"Failed to fetch {ticker}: {e}")
        return TickerStatus(
            ticker=ticker,
            tier=tier,
            price=0.0,
            change_pct=0.0,
            rsi=None,
            above_ma50=None,
            above_ma200=None,
            atr_pct=None,
            signal_count=0,
            sentiment="error",
        )


def generate_daily_report(
    basket: Optional[dict[str, list[str]]] = None,
    output_path: Optional[Path] = None,
) -> str:
    """Generate a daily report for the production basket.

    Args:
        basket: Dict of tier→tickers. Defaults to production_basket.yaml.
        output_path: Path to write report. Defaults to reports/daily/YYYY-MM-DD.md.

    Returns:
        Markdown report string.
    """
    basket = basket or load_basket()
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")

    all_tickers = []
    for tier in ["S", "A", "B"]:
        for ticker in basket.get(tier, []):
            all_tickers.append(fetch_ticker_status(ticker, tier))

    lines = [
        f"# Daily Production Basket Report — {date_str}",
        "",
        f"**Generated:** {now.strftime('%Y-%m-%d %H:%M')}",
        f"**Basket size:** {len(all_tickers)} instruments (S:{len(basket.get('S', []))} A:{len(basket.get('A', []))} B:{len(basket.get('B', []))})",
        "",
        "---",
        "",
        "## Market Overview",
        "",
    ]

    bullish = sum(1 for s in all_tickers if s.sentiment == "bullish")
    bearish = sum(1 for s in all_tickers if s.sentiment == "bearish")
    neutral = sum(1 for s in all_tickers if s.sentiment == "neutral")

    lines.append(f"- Bullish: {bullish} | Bearish: {bearish} | Neutral: {neutral}")
    lines.append("")

    for tier in ["S", "A", "B"]:
        tier_tickers = [t for t in all_tickers if t.tier == tier]
        if not tier_tickers:
            continue

        lines.append(f"## Tier {tier}")
        lines.append("")
        lines.append("| Ticker | Price | Chg% | RSI | >MA50 | >MA200 | ATR% | Sentiment |")
        lines.append("|--------|-------|------|-----|-------|--------|------|-----------|")

        for t in tier_tickers:
            emoji = _SENTIMENT_MAP.get(t.sentiment, "⚪")
            ma50 = "✓" if t.above_ma50 else ("✗" if t.above_ma50 is not None else "?")
            ma200 = "✓" if t.above_ma200 else ("✗" if t.above_ma200 is not None else "?")
            lines.append(
                f"| {t.ticker} | {t.price:.2f} | {t.change_pct:+.2f}% | "
                f"{t.rsi if t.rsi else '?'} | {ma50} | {ma200} | "
                f"{t.atr_pct if t.atr_pct else '?'} | {emoji} {t.sentiment} |"
            )
        lines.append("")

    tier_returns = {}
    for tier in ["S", "A", "B"]:
        tier_tickers = [t for t in all_tickers if t.tier == tier]
        if tier_tickers:
            valid = [t.change_pct for t in tier_tickers if t.price > 0]
            tier_returns[tier] = round(sum(valid) / len(valid), 2) if valid else 0.0

    lines.append("## Tier Summary")
    lines.append("")
    for tier, ret in tier_returns.items():
        lines.append(f"- **Tier {tier}:** Avg change {ret:+.2f}%")
    lines.append("")

    regime_note = _regime_check(all_tickers)
    lines.append("## Regime Check")
    lines.append("")
    lines.append(regime_note)
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Report generated by `scripts/daily_report_agent.py`")
    lines.append(f"- Data source: Yahoo Finance, as of {now.strftime('%Y-%m-%d %H:%M')}")
    lines.append("- This is an automated report — verify signals before trading.")
    lines.append("")

    report = "\n".join(lines)

    if output_path:
        output_path = output_path or (_OUTPUT_DIR / f"{date_str}.md")
    else:
        output_path = _OUTPUT_DIR / f"{date_str}.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    return report


def _compute_rsi(close: pd.Series, period: int = 14) -> float | None:
    """Compute RSI for a series."""
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None


def _regime_check(statuses: list[TickerStatus]) -> str:
    """Detect current market regime from basket status."""
    valid = [s for s in statuses if s.rsi is not None and s.price > 0]
    if not valid:
        return "Insufficient data for regime check."

    avg_rsi = sum(s.rsi for s in valid) / len(valid)  # type: ignore[operator]
    above_ma50_pct = sum(1 for s in valid if s.above_ma50) / len(valid) * 100

    lines = [f"- Average RSI(14) across basket: {avg_rsi:.1f}"]
    lines.append(f"- Above MA(50): {above_ma50_pct:.0f}% of instruments")

    if above_ma50_pct >= 70:
        lines.append("- **Regime: BULLISH** — strong trend, favor momentum entries.")
    elif above_ma50_pct <= 30:
        lines.append("- **Regime: BEARISH** — weak trend, favor defensive/reversal entries.")
    else:
        lines.append(
            "- **Regime: MIXED/TRANSITIONAL** — favor quality patterns over direction bets."
        )

    if avg_rsi > 70:
        lines.append("- **Warning:** RSI overbought across basket — expect mean reversion.")
    elif avg_rsi < 30:
        lines.append(
            "- **Opportunity:** RSI oversold across basket — expect mean reversion bounce."
        )

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = generate_daily_report()
    print(report)
