"""P24-20: Event-type specific trading strategies.

Different event types have different optimal holding periods:
  - Long-term events (Lawsuits, Products, Acquisitions): hold 20-40 bars
  - Short-term events (Dividends, Earnings): hold 3-10 bars
  - News events: hold 1-5 bars
  - Recurring events (FOMC, NFP): avoid entirely on event day
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class EventCategory(Enum):
    NEWS = auto()
    EARNINGS = auto()
    DIVIDEND = auto()
    FOMC = auto()
    NFP = auto()
    PRODUCT = auto()
    LAWSUIT = auto()
    ACQUISITION = auto()
    INSIDER = auto()
    OTHER = auto()


@dataclass
class EventConfig:
    """Trading parameters for a specific event category."""

    category: EventCategory
    min_hold_bars: int
    max_hold_bars: int
    tp_atr: float
    sl_atr: float
    signal_multiplier: float
    description: str = ""


EVENT_CONFIGS: Dict[EventCategory, EventConfig] = {
    EventCategory.EARNINGS: EventConfig(
        EventCategory.EARNINGS,
        min_hold_bars=3,
        max_hold_bars=10,
        tp_atr=1.5,
        sl_atr=2.0,
        signal_multiplier=1.2,
        description="Earnings — short-term, high-vol, tight stops",
    ),
    EventCategory.DIVIDEND: EventConfig(
        EventCategory.DIVIDEND,
        min_hold_bars=5,
        max_hold_bars=15,
        tp_atr=1.0,
        sl_atr=1.5,
        signal_multiplier=0.8,
        description="Dividend — modest drift, lower conviction",
    ),
    EventCategory.FOMC: EventConfig(
        EventCategory.FOMC,
        min_hold_bars=0,
        max_hold_bars=0,
        tp_atr=0.0,
        sl_atr=0.0,
        signal_multiplier=0.0,
        description="FOMC — NO TRADE on event day",
    ),
    EventCategory.NFP: EventConfig(
        EventCategory.NFP,
        min_hold_bars=0,
        max_hold_bars=0,
        tp_atr=0.0,
        sl_atr=0.0,
        signal_multiplier=0.0,
        description="NFP — NO TRADE on event day",
    ),
    EventCategory.PRODUCT: EventConfig(
        EventCategory.PRODUCT,
        min_hold_bars=10,
        max_hold_bars=40,
        tp_atr=3.0,
        sl_atr=2.0,
        signal_multiplier=1.0,
        description="Product launch — multi-week trend drift",
    ),
    EventCategory.LAWSUIT: EventConfig(
        EventCategory.LAWSUIT,
        min_hold_bars=10,
        max_hold_bars=30,
        tp_atr=3.0,
        sl_atr=2.5,
        signal_multiplier=0.9,
        description="Lawsuit — slow resolution, higher uncertainty",
    ),
    EventCategory.ACQUISITION: EventConfig(
        EventCategory.ACQUISITION,
        min_hold_bars=15,
        max_hold_bars=40,
        tp_atr=2.5,
        sl_atr=1.5,
        signal_multiplier=1.1,
        description="Acquisition — spreads compress over weeks",
    ),
    EventCategory.INSIDER: EventConfig(
        EventCategory.INSIDER,
        min_hold_bars=5,
        max_hold_bars=20,
        tp_atr=2.0,
        sl_atr=2.0,
        signal_multiplier=1.1,
        description="Insider activity — modest drift, elevated confidence",
    ),
    EventCategory.NEWS: EventConfig(
        EventCategory.NEWS,
        min_hold_bars=1,
        max_hold_bars=5,
        tp_atr=1.5,
        sl_atr=2.0,
        signal_multiplier=1.3,
        description="News — fade after 5 bars",
    ),
    EventCategory.OTHER: EventConfig(
        EventCategory.OTHER,
        min_hold_bars=3,
        max_hold_bars=10,
        tp_atr=1.5,
        sl_atr=2.0,
        signal_multiplier=1.0,
        description="Other — generic short-term config",
    ),
}


def get_event_config(category: EventCategory) -> EventConfig:
    """Get trading configuration for an event category."""
    return EVENT_CONFIGS.get(category, EVENT_CONFIGS[EventCategory.OTHER])


def event_aware_entry(
    score: float,
    category: EventCategory,
    bars_held: int,
) -> Tuple[bool, float]:
    """Determine whether to enter/exit based on event type.

    Returns:
        (should_exit, adjusted_signal_multiplier)
    """
    config = get_event_config(category)

    if config.signal_multiplier <= 0:
        return False, 0.0

    if bars_held > config.min_hold_bars:
        return True, config.signal_multiplier

    if bars_held >= config.max_hold_bars:
        return True, 0.0

    return False, score * config.signal_multiplier


@dataclass
class EventTradingResult:
    """Trade result annotated with event context."""

    event_category: EventCategory
    entry_bar: int
    exit_bar: int
    hold_bars: int
    return_pct: float
    signal_multiplier: float
    notes: str = ""


def classify_event(
    event_name: str,
    description: str = "",
) -> EventCategory:
    """Classify an event name into a category.

    Args:
        event_name: Raw event name/title.
        description: Optional additional context.

    Returns:
        EventCategory enum.
    """
    text = (event_name + " " + description).lower()

    if any(kw in text for kw in ("earnings", "eps", "revenue report", "guidance")):
        return EventCategory.EARNINGS
    if any(kw in text for kw in ("dividend", "distribution", "yield increase")):
        return EventCategory.DIVIDEND
    if any(kw in text for kw in ("fomc", "fed decision", "fed meeting", "federal reserve")):
        return EventCategory.FOMC
    if any(kw in text for kw in ("nfp", "nonfarm", "employment report", "jobs report")):
        return EventCategory.NFP
    if any(kw in text for kw in ("product launch", "product release", "new product", "announced")):
        return EventCategory.PRODUCT
    if any(kw in text for kw in ("lawsuit", "litigation", "court", "settlement", "class action")):
        return EventCategory.LAWSUIT
    if any(kw in text for kw in ("acquisition", "merger", "takeover", "buyout", "acquired")):
        return EventCategory.ACQUISITION
    if any(kw in text for kw in ("insider", "form 4", "insider purchase", "insider sale")):
        return EventCategory.INSIDER
    if any(kw in text for kw in ("news", "headline", "press release")):
        return EventCategory.NEWS
    return EventCategory.OTHER
