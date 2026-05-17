"""Sentiment signal scoring from research papers (4+ sentiment papers).

Design: SentimentScorer is a pluggable interface that accepts sentiment data
from any source (FinBERT, VADER, Polygon.io API, pre-computed CSV).
It modifies existing signals by applying a sentiment weight multiplier.

Paper findings:
- Ensemble classifiers (LR+SVM+RF+XGBoost) achieve >80% accuracy on financial sentiment
- VADER and FinBERT are standard baselines
- Sentiment polarity correlates with next-day returns (direction, not magnitude)
- Best results when sentiment is used as signal MODIFIER (weight adjuster),
  not standalone signal
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


class SentimentProvider(Protocol):
    """Protocol for sentiment data sources."""

    def get_sentiment(self, dates: pd.DatetimeIndex, symbol: str) -> pd.Series:
        """Return sentiment scores in [-1, 1] for given dates.

        -1 = extremely negative, 0 = neutral, +1 = extremely positive.
        """
        ...


class CSVSentimentProvider:
    """Load pre-computed sentiment scores from CSV.

    CSV format: date,symbol,sentiment_score
    """

    def __init__(self, csv_path: Path, default_score: float = 0.0) -> None:
        self.csv_path = Path(csv_path)
        self.default_score = default_score
        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.csv_path, parse_dates=["date"])
            self._df.set_index(["date", "symbol"], inplace=True)
        return self._df

    def get_sentiment(self, dates: pd.DatetimeIndex, symbol: str) -> pd.Series:
        df = self._load()
        scores = pd.Series(self.default_score, index=dates, dtype=float)
        for i, dt in enumerate(dates):
            try:
                scores.iloc[i] = float(df.loc[(dt, symbol), "sentiment_score"])
            except (KeyError, TypeError):
                pass
        return scores


class SyntheticSentimentProvider:
    """Generate synthetic sentiment for testing (no real data needed).

    Uses an AR(1) process with mean reversion to 0.
    Useful for smoke-testing the sentiment pipeline before connecting real data.
    """

    def __init__(
        self,
        seed: int = 42,
        autocorr: float = 0.3,
        volatility: float = 0.15,
    ) -> None:
        self.rng = np.random.RandomState(seed)
        self.autocorr = autocorr
        self.volatility = volatility

    def get_sentiment(self, dates: pd.DatetimeIndex, symbol: str) -> pd.Series:
        n = len(dates)
        scores = np.zeros(n)
        for i in range(1, n):
            scores[i] = self.autocorr * scores[i - 1] + self.rng.normal(0, self.volatility)
        return pd.Series(np.clip(scores, -1, 1), index=dates)


class SentimentSignalModifier:
    """Modify ML trading signals with sentiment scores.

    Paper insight: Sentiment works best as a SIGNAL MODIFIER, not standalone signal.
    Multiplies ML probability with sentiment score to produce adjusted signal.

    Formula: adjusted_signal = base_signal * (1 + sentiment_weight * sentiment_score)
    where sentiment_weight controls how much sentiment influences the signal.
    """

    def __init__(
        self,
        sentiment_provider: SentimentProvider,
        sentiment_weight: float = 0.15,
        min_sentiment_abs: float = 0.1,
    ) -> None:
        self.provider = sentiment_provider
        self.sentiment_weight = sentiment_weight
        self.min_sentiment_abs = min_sentiment_abs

    def adjust_signals(
        self,
        base_signals: pd.Series,
        dates: pd.DatetimeIndex,
        symbol: str,
    ) -> pd.Series:
        """Apply sentiment adjustment to base trading signals.

        Args:
            base_signals: Raw signals in [0, 1] (ML probabilities)
            dates: DatetimeIndex aligned with signals
            symbol: Ticker symbol for sentiment lookup

        Returns:
            Adjusted signals in [0, 1]
        """
        sentiment = self.provider.get_sentiment(dates, symbol)

        sentiment_masked = sentiment.where(
            sentiment.abs() >= self.min_sentiment_abs,
            0.0,
        )

        adjustment = 1.0 + self.sentiment_weight * sentiment_masked.values
        adjusted = base_signals.values * adjustment

        return pd.Series(np.clip(adjusted, 0, 1), index=base_signals.index)

    def adjust_single(
        self,
        base_signal: float,
        date: pd.Timestamp,
        symbol: str,
    ) -> float:
        """Adjust a single signal value with sentiment at a specific date.

        Args:
            base_signal: ML probability in [0, 1]
            date: The date of the signal
            symbol: Ticker symbol

        Returns:
            Adjusted probability in [0, 1]
        """
        sentiment_series = self.provider.get_sentiment(pd.DatetimeIndex([date]), symbol)
        sentiment_score = float(sentiment_series.iloc[0])

        if abs(sentiment_score) < self.min_sentiment_abs:
            return base_signal

        adjustment = 1.0 + self.sentiment_weight * sentiment_score
        adjusted = base_signal * adjustment
        return float(np.clip(adjusted, 0, 1))
