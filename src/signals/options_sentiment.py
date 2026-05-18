"""
Q4: Put/Call Ratio + Gamma Exposure (GEX) Sentiment Signals.

Options-derived sentiment signals that reveal institutional hedging flow
and dealer positioning — leading indicators for equity directional moves.

Put/Call Ratio:
  - PC ratio = Put volume / Call volume (equity-only or total)
  - PC > 1.2 → bearish (excessive hedging), sentiment < 0
  - PC < 0.7 → bullish (complacent), sentiment > 0
  - The single best options-derived equity signal per literature

Gamma Exposure (GEX) Proxy:
  - Dealer gamma positioning predicts intraday range and reversals
  - Positive GEX → dealers dampen moves (range-bound, fade extremes)
  - Negative GEX → dealers amplify moves (trend days, gamma squeezes)
  - Proxy via VIX/SPX correlation and PC ratio extremes

Data sources:
  - FMP free tier: put/call ratio endpoint
  - VIX as GEX proxy (GEX inversely correlated with VIX level)

Integration: Implements SentimentProvider protocol for plug-in to
SentimentSignalModifier and RulesFirstStrategy.

Usage:
    >>> provider = OptionsSentimentProvider()
    >>> scores = provider.get_sentiment(dates, "SPY")
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.signals.sentiment_scorer import SentimentProvider

logger = logging.getLogger(__name__)

FMP_BASE = "https://financialmodelingprep.com/stable"
FMP_API_KEY = "cv5v6VVC1p6ZAunPjWwwfMXQ0cvslEYI"

PC_EXTREME_BEARISH = 1.20
PC_BEARISH = 0.85
PC_NEUTRAL_HIGH = 0.75
PC_NEUTRAL_LOW = 0.65
PC_BULLISH = 0.55
PC_EXTREME_BULLISH = 0.40


class OptionsSentimentProvider:
    """Options-derived sentiment provider implementing SentimentProvider protocol.

    Fetches put/call ratio from FMP free tier and computes GEX proxy from
    VIX level. Normalizes to [-1, 1] sentiment scores.

    Args:
        use_fmp: Try FMP API first for PC ratio data.
        gex_weight: Weight of GEX proxy in composite score (0-1).
        pc_weight: Weight of PC ratio in composite score (0-1).
    """

    def __init__(
        self,
        use_fmp: bool = True,
        gex_weight: float = 0.3,
        pc_weight: float = 0.7,
    ):
        self.use_fmp = use_fmp
        self.gex_weight = gex_weight
        self.pc_weight = pc_weight
        self._pc_data: Optional[pd.DataFrame] = None
        self._gex_data: Optional[pd.Series] = None
        self._fitted: bool = False

    def fit(self, start: str = "2015-01-01") -> OptionsSentimentProvider:
        """Fetch options sentiment data for the given date range.

        Args:
            start: Start date for data.

        Returns:
            Self for chaining.
        """
        if self.use_fmp:
            self._fetch_fmp_pc_ratio(start)
        self._compute_gex_proxy(start)
        self._fitted = True
        return self

    def _fetch_fmp_pc_ratio(self, start: str) -> None:
        """Fetch put/call ratio from FMP free tier."""
        try:
            import urllib.request
            import json as _json

            url = f"{FMP_BASE}/v4/put-call-ratio?apikey={FMP_API_KEY}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = _json.loads(resp.read().decode())

            if not raw:
                logger.warning("FMP returned no PC ratio data")
                return

            records = []
            for item in raw:
                date = item.get("date")
                ratio = item.get("ratio", item.get("putCallRatio"))
                if date and ratio is not None:
                    d = pd.Timestamp(date)
                    if str(d.date()) >= start:
                        records.append({"date": d, "pc_ratio": float(ratio)})

            if records:
                self._pc_data = pd.DataFrame(records).set_index("date").sort_index()
                logger.info("FMP PC ratio: %d records", len(self._pc_data))
            else:
                logger.warning("FMP PC ratio: 0 records in range")
        except Exception as e:
            logger.warning("FMP PC ratio fetch failed: %s, using VIX proxy", e)
            self._vix_pc_proxy(start)

    def _vix_pc_proxy(self, start: str) -> None:
        """Use VIX level as inverse PC ratio proxy when real data unavailable.

        PC ratio typically rises with VIX:
        - VIX < 15 → PC ~ 0.5 (bullish)
        - VIX = 20 → PC ~ 0.7 (neutral)
        - VIX > 30 → PC ~ 1.2 (bearish)
        """
        try:
            import yfinance as yf

            vix = yf.download("^VIX", start=start, progress=False, auto_adjust=True)
            if vix.empty:
                return
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            vix_close = vix["Close"] if "Close" in vix.columns else vix.iloc[:, 0]
            pc = pd.DataFrame({"pc_ratio": 0.5 + vix_close * 0.02}, index=vix_close.index)
            pc = pc[pc.index >= start]
            self._pc_data = pc
            logger.info("VIX-based PC proxy: %d records", len(self._pc_data))
        except Exception as e:
            logger.warning("VIX PC proxy failed: %s", e)

    def _compute_gex_proxy(self, start: str) -> None:
        """Compute GEX proxy from VIX level.

        GEX is inversely related to VIX:
        - Low VIX → dealers long gamma → stabilizing (GEX high)
        - High VIX → dealers short gamma → amplifying (GEX low/negative)

        Proxy: GEX = -tanh((VIX - 20) / 10)
        Maps VIX=10 → +0.76, VIX=20 → 0, VIX=30 → -0.76
        """
        try:
            import yfinance as yf

            vix = yf.download("^VIX", start=start, progress=False, auto_adjust=True)
            if vix.empty:
                return
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            vix_close = vix["Close"] if "Close" in vix.columns else vix.iloc[:, 0]
            self._gex_data = -np.tanh((vix_close.astype(float) - 20.0) / 10.0)
            self._gex_data.name = "gex_proxy"
        except Exception as e:
            logger.warning("GEX proxy computation failed: %s", e)

    def get_sentiment(self, dates: pd.DatetimeIndex, symbol: str = "SPY") -> pd.Series:
        """Return sentiment scores in [-1, 1] for given dates.

        Sentiment composite = pc_weight * pc_sentiment + gex_weight * gex_sentiment

        Args:
            dates: DatetimeIndex of dates to look up.
            symbol: Symbol (unused for market-level sentiment).

        Returns:
            Series of sentiment scores in [-1, 1].
        """
        if not self._fitted:
            return pd.Series(0.0, index=dates)

        pc_sent = self._pc_sentiment(dates)
        gex_sent = self._gex_sentiment(dates)

        composite = self.pc_weight * pc_sent + self.gex_weight * gex_sent
        composite = composite.fillna(0.0).clip(-1.0, 1.0)
        return pd.Series(composite, index=dates)

    def _pc_sentiment(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Convert PC ratio to [-1, 1] sentiment score."""
        scores = pd.Series(0.0, index=dates)
        if self._pc_data is None or self._pc_data.empty:
            return scores

        for i, d in enumerate(dates):
            idx = self._pc_data.index
            nearby = idx[idx <= d]
            if len(nearby) == 0:
                continue
            pc = float(self._pc_data.loc[nearby[-1], "pc_ratio"])
            scores.iloc[i] = self._pc_to_sentiment(pc)
        return scores

    def _pc_to_sentiment(self, pc_ratio: float) -> float:
        """Map PC ratio to sentiment score via piecewise linear function."""
        if pc_ratio >= PC_EXTREME_BEARISH:
            return -1.0
        elif pc_ratio >= PC_BEARISH:
            return -0.5 - (pc_ratio - PC_BEARISH) / (PC_EXTREME_BEARISH - PC_BEARISH) * 0.5
        elif pc_ratio >= PC_NEUTRAL_HIGH:
            return -0.25 - (pc_ratio - PC_NEUTRAL_HIGH) / (PC_BEARISH - PC_NEUTRAL_HIGH) * 0.25
        elif pc_ratio >= PC_NEUTRAL_LOW:
            return 0.0
        elif pc_ratio >= PC_BULLISH:
            return 0.25 - (pc_ratio - PC_BULLISH) / (PC_NEUTRAL_LOW - PC_BULLISH) * 0.25
        elif pc_ratio >= PC_EXTREME_BULLISH:
            return 0.5 - (pc_ratio - PC_EXTREME_BULLISH) / (PC_BULLISH - PC_EXTREME_BULLISH) * 0.5
        return 1.0

    def _gex_sentiment(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Get GEX proxy sentiment for dates."""
        scores = pd.Series(0.0, index=dates)
        if self._gex_data is None or self._gex_data.empty:
            return scores

        for i, d in enumerate(dates):
            idx = self._gex_data.index
            nearby = idx[idx <= d]
            if len(nearby) == 0:
                continue
            scores.iloc[i] = float(self._gex_data.loc[nearby[-1]])
        return scores

    def pc_ratio(self, date=None) -> Optional[float]:
        """Get put/call ratio for a specific date."""
        if self._pc_data is None or self._pc_data.empty:
            return None
        if date is None:
            return float(self._pc_data["pc_ratio"].iloc[-1])
        if date in self._pc_data.index:
            return float(self._pc_data.loc[date, "pc_ratio"])
        idx = self._pc_data.index
        nearby = idx[idx <= pd.Timestamp(date)]
        return float(self._pc_data.loc[nearby[-1], "pc_ratio"]) if len(nearby) > 0 else None

    @property
    def fitted(self) -> bool:
        return self._fitted
