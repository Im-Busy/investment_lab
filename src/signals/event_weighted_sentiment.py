"""P24-21: Event-time-weighted sentiment decay.

Sum sentiment × exp(−days/10) — 10-day half-life for event relevance.
Recent events contribute more to sentiment signal than stale ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class WeightedSentimentEvent:
    """Time-weighted sentiment signal."""

    timestamp: pd.Timestamp
    sentiment: float  # [-1, 1]
    decay_factor: float
    weighted_sentiment: float
    days_ago: int


def compute_time_weighted_sentiment(
    events: List[WeightedSentimentEvent],
    half_life_days: float = 10.0,
) -> float:
    """Aggregate weighted sentiment from events with exponential decay.

    Args:
        events: List of sentiment events with timestamps.
        half_life_days: Days for weight to halve (default 10).

    Returns:
        Weighted aggregate sentiment in [-1, 1].
    """
    if not events:
        return 0.0

    now = pd.Timestamp.now()
    weights = []
    sentiments = []

    for e in sorted(events, key=lambda x: x.timestamp):
        days = (now - e.timestamp).total_seconds() / 86400.0
        w = np.exp(-np.log(2.0) * days / half_life_days)
        weights.append(w)
        sentiments.append(e.sentiment)

    if sum(weights) < 1e-10:
        return 0.0

    weighted = float(np.average(sentiments, weights=weights))
    return float(np.clip(weighted, -1.0, 1.0))


def decay_fn(days: float, half_life: float = 10.0) -> float:
    """Exponential decay weight for sentiment."""
    if days < 0:
        return 1.0
    return float(np.exp(-np.log(2.0) * days / half_life))


def batch_weight_events(
    sentiment_series: pd.Series,
    half_life_days: float = 10.0,
) -> pd.Series:
    """Apply exponential decay weights to a time series of sentiment values.

    Args:
        sentiment_series: Series with datetime index and sentiment values.
        half_life_days: Half-life in calendar days.

    Returns:
        Series with same index, weighted values.
    """
    if len(sentiment_series) == 0:
        return sentiment_series

    latest_ts = sentiment_series.index[-1]
    days = (latest_ts - sentiment_series.index).total_seconds() / 86400.0
    weights = decay_fn(np.abs(days.values), half_life_days)

    return sentiment_series * weights
