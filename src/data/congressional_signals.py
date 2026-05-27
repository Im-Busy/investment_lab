"""P28-10: Congressional trade signal data feed.

Fetches congressional stock trade disclosures and generates alpha signals
based on committee membership weighting, trade recency decay, and
insider cluster detection. Members of finance/banking/armed-services
committees get higher alpha-weight due to sector-specific information access.

Source: awesome-ai-in-finance — congressional trade tracking tools.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

import pandas as pd

logger = logging.getLogger(__name__)

HOUSE_API_URL = (
    "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
)
SENATE_API_URL = (
    "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
)

DEFAULT_CACHE_DIR = Path("data/congress_cache")
CACHE_TTL_SECONDS = 86400

COMMITTEE_WEIGHTS: dict[str, float] = {
    "financial services": 1.5,
    "banking": 1.4,
    "finance": 1.4,
    "ways and means": 1.3,
    "armed services": 1.2,
    "energy and commerce": 1.1,
    "health": 1.1,
    "intelligence": 1.3,
    "appropriations": 1.1,
    "budget": 1.1,
    "judiciary": 0.9,
    "agriculture": 1.0,
}
DEFAULT_WEIGHT = 1.0
SIGNAL_DECAY_DAYS = 90


@dataclass
class CongressTrade:
    """Single congressional trade disclosure."""

    ticker: str
    transaction_date: str
    disclosure_date: str
    type: str  # "buy", "sell", "exchange"
    amount: str  # "$1,001 - $15,000" or ranges
    member_name: str
    party: str  # "Democrat", "Republican", "Independent"
    chamber: str  # "House", "Senate"
    committees: list[str] = field(default_factory=list)
    comment: str = ""

    @property
    def amount_range(self) -> tuple[float, float]:
        parts = self.amount.replace("$", "").replace(",", "").split()
        values = [float(p) for p in parts if p.replace(".", "").isdigit()]
        if len(values) >= 2:
            return (values[0], values[-1])
        if len(values) == 1:
            return (values[0], values[0])
        return (0.0, 0.0)

    @property
    def amount_mid(self) -> float:
        lo, hi = self.amount_range
        return (lo + hi) / 2

    @property
    def direction(self) -> int:
        return 1 if self.type.lower() in ("buy", "purchase") else -1

    @property
    def committee_weight(self) -> float:
        if not self.committees:
            return DEFAULT_WEIGHT
        max_w = DEFAULT_WEIGHT
        for c in self.committees:
            c_lower = c.lower()
            for key, weight in COMMITTEE_WEIGHTS.items():
                if key in c_lower:
                    max_w = max(max_w, weight)
        return max_w

    @property
    def days_since_disclosure(self) -> float:
        if not self.disclosure_date:
            return float("inf")
        try:
            d = datetime.fromisoformat(self.disclosure_date.replace("Z", "+00:00"))
            return (datetime.now(timezone.utc) - d).total_seconds() / 86400
        except (ValueError, TypeError):
            return float("inf")


@dataclass
class TickerSignal:
    """Aggregated signal for a single ticker."""

    ticker: str
    buy_count: int = 0
    sell_count: int = 0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    weighted_buys: float = 0.0
    weighted_sells: float = 0.0
    unique_members: int = 0
    unique_parties: int = 0
    latest_trade_date: str = ""

    @property
    def net_signal(self) -> float:
        raw = self.weighted_buys - self.weighted_sells
        total = self.weighted_buys + self.weighted_sells
        return raw / max(total, 1)

    @property
    def signal_strength(self) -> float:
        count_strength = min((self.buy_count + self.sell_count) / 10, 1.0)
        diversity_bonus = 0.1 if self.unique_parties >= 2 else 0.0
        return min(count_strength * (1 + diversity_bonus), 1.0)


def _fetch_json_api(url: str) -> list[dict]:
    req = Request(url, headers={"User-Agent": "investment_trying/1.0"})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_congress_trades(
    chambers: Optional[list[str]] = None,
    max_results: int = 2000,
) -> list[CongressTrade]:
    """Fetch recent congressional trades from public disclosures.

    Args:
        chambers: Which chambers to fetch (default: both House and Senate).
        max_results: Max trades to return.

    Returns:
        List of CongressTrade objects, sorted by disclosure_date desc.
    """
    targets: dict[str, str] = {}
    if chambers is None or "house" in [c.lower() for c in chambers]:
        targets["House"] = HOUSE_API_URL
    if chambers is None or "senate" in [c.lower() for c in chambers]:
        targets["Senate"] = SENATE_API_URL

    trades: list[CongressTrade] = []
    for chamber, url in targets.items():
        try:
            data = _fetch_json_api(url)
            for item in data[-max_results:]:
                if not item.get("ticker"):
                    continue
                trades.append(
                    CongressTrade(
                        ticker=str(item["ticker"]).upper(),
                        transaction_date=str(item.get("transaction_date", "")),
                        disclosure_date=str(item.get("disclosure_date", "")),
                        type=str(item.get("type", "").lower()),
                        amount=str(item.get("amount", "$0")),
                        member_name=str(item.get("representative", item.get("senator", ""))),
                        party=str(item.get("party", "")),
                        chamber=chamber,
                        committees=_parse_committees(item),
                        comment=str(item.get("comment", "")),
                    )
                )
            logger.info("Fetched %d trades from %s", len(trades), chamber)
        except Exception as e:
            logger.warning("Failed to fetch %s trades: %s", chamber, e)

    trades.sort(key=lambda t: t.disclosure_date, reverse=True)
    trades = trades[:max_results]
    logger.info("Total: %d congressional trades loaded", len(trades))
    return trades


def _parse_committees(item: dict) -> list[str]:
    raw = item.get("committees", [])
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        return [c.strip() for c in raw.split(",") if c.strip()]
    return []


def aggregate_trades_by_ticker(
    trades: list[CongressTrade],
    *,
    max_days_old: int = SIGNAL_DECAY_DAYS,
) -> dict[str, TickerSignal]:
    """Aggregate congressional trades into per-ticker signals.

    Trades are time-weighted with exponential decay: trades from yesterday
    get weight ~1.0, trades from 90 days ago get weight ~0.0.

    Args:
        trades: List of CongressTrade objects.
        max_days_old: Maximum age of trades to consider.

    Returns:
        Dict of ticker → TickerSignal.
    """
    signals: dict[str, TickerSignal] = {}

    for trade in trades:
        age = trade.days_since_disclosure
        if age > max_days_old:
            continue

        decay = max(0.0, 1.0 - (age / max_days_old))
        ticker = trade.ticker
        if ticker not in signals:
            signals[ticker] = TickerSignal(ticker=ticker)

        sig = signals[ticker]
        amt = trade.amount_mid * decay

        if trade.direction > 0:
            sig.buy_count += 1
            sig.buy_volume += amt
            sig.weighted_buys += amt * trade.committee_weight
        else:
            sig.sell_count += 1
            sig.sell_volume += amt
            sig.weighted_sells += amt * trade.committee_weight

        if trade.latest_trade_date < sig.latest_trade_date or not sig.latest_trade_date:
            sig.latest_trade_date = trade.disclosure_date

        sig.unique_members = len({t.member_name for t in trades if t.ticker == ticker})
        sig.unique_parties = len({t.party for t in trades if t.ticker == ticker})

    return signals


def get_congress_signals(
    chambers: Optional[list[str]] = None,
    max_days_old: int = SIGNAL_DECAY_DAYS,
    min_trades: int = 3,
) -> pd.DataFrame:
    """End-to-end: fetch trades → aggregate → filter → DataFrame.

    Args:
        chambers: Chambers to query.
        max_days_old: Maximum trade age in days.
        min_trades: Minimum number of trades for a ticker to be included.

    Returns:
        DataFrame indexed by ticker with signal columns.
    """
    trades = fetch_congress_trades(chambers)
    signals = aggregate_trades_by_ticker(trades, max_days_old=max_days_old)

    rows: list[dict] = []
    for ticker, sig in sorted(signals.items()):
        if sig.buy_count + sig.sell_count < min_trades:
            continue
        rows.append(
            {
                "ticker": ticker,
                "buy_count": sig.buy_count,
                "sell_count": sig.sell_count,
                "net_signal": round(sig.net_signal, 4),
                "signal_strength": round(sig.signal_strength, 4),
                "weighted_buys": round(sig.weighted_buys, 0),
                "weighted_sells": round(sig.weighted_sells, 0),
                "unique_members": sig.unique_members,
                "unique_parties": sig.unique_parties,
                "latest_trade": sig.latest_trade_date[:10] if sig.latest_trade_date else "",
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("ticker").sort_values("net_signal", ascending=False)

    logger.info("Congress signals: %d tickers with ≥%d trades", len(df), min_trades)
    return df
