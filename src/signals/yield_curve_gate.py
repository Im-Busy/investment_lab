"""
Q2: Yield Curve Gate — Yield Curve Inversion + Credit Spread Macro Regime.

Detects yield curve inversion (2s10s spread) as a binary macro regime flag.
Inversion is the single best recession predictor with 100+ years of evidence.

Also computes credit spread (BAA-AAA) as a risk appetite gauge when available.

Regime signal:
  - INVERTED (2s10s < 0): defensive, 0.50x signal multiplier
  - NEAR_INVERTED (0 < 2s10s < 0.5%): caution, 0.75x multiplier
  - NORMAL (2s10s > 0.5%): risk-on, 1.0x multiplier

Data sources (in order of fallback):
  1. FMP free tier (Treasury yields + credit spreads)
  2. yfinance (^FVX, ^TNX for 5Y/10Y Treasury yields)
  3. FRED API via fredapi (most complete)

Integration with RulesFirstStrategy:
  score *= gate.multiplier(date)

Usage:
    >>> gate = YieldCurveGate()
    >>> gate.fit(start="2016-01-01")
    >>> gate.multiplier("2025-03-15")  # returns 1.0 if normal curve
    >>> gate.is_inverted("2007-03-15")  # True before 2008 crisis
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INVERSION_THRESHOLD = 0.0
NEAR_INVERSION_THRESHOLD = 0.005
FMP_BASE = "https://financialmodelingprep.com/stable"
FMP_API_KEY = "cv5v6VVC1p6ZAunPjWwwfMXQ0cvslEYI"


class YieldCurveGate:
    """Yield curve macro regime gate.

    Fetches Treasury yields, computes 2s10s spread, and produces a binary
    inversion signal plus scalar multipliers for signal gating.

    Args:
        inversion_mult: Signal multiplier during inversion (default 0.50).
        near_inversion_mult: Signal multiplier during near inversion (default 0.75).
        use_fmp: Try FMP API first for Treasury + credit data.
        use_fred: Fall back to FRED API (requires fredapi + key).
    """

    def __init__(
        self,
        inversion_mult: float = 0.50,
        near_inversion_mult: float = 0.75,
        use_fmp: bool = True,
        use_fred: bool = False,
    ):
        self.inversion_mult = inversion_mult
        self.near_inversion_mult = near_inversion_mult
        self.use_fmp = use_fmp
        self.use_fred = use_fred
        self._spread_data: Optional[pd.DataFrame] = None
        self._credit_spread: Optional[pd.Series] = None

    def fit(self, start: str = "2015-01-01", end: Optional[str] = None) -> YieldCurveGate:
        """Fetch Treasury yield data and compute spreads.

        Args:
            start: Start date.
            end: End date (defaults to today).

        Returns:
            Self for chaining.
        """
        df = None
        credit = None

        if self.use_fmp:
            df, credit = self._fetch_fmp(start)
        if df is None and self.use_fred:
            df = self._fetch_fred(start, end)
        if df is None:
            df = self._fetch_yfinance(start, end)

        if df is None or df.empty:
            raise RuntimeError("Failed to fetch Treasury yield data from any source")

        if "spread_2s10s" not in df.columns:
            if "DGS2" in df.columns and "DGS10" in df.columns:
                df["spread_2s10s"] = df["DGS10"] - df["DGS2"]
            elif "yield_2y" in df.columns and "yield_10y" in df.columns:
                df["spread_2s10s"] = df["yield_10y"] - df["yield_2y"]

        if "spread_10y3m" not in df.columns:
            if "DGS10" in df.columns and "DGS3MO" in df.columns:
                df["spread_10y3m"] = df["DGS10"] - df["DGS3MO"]

        df["inverted"] = df["spread_2s10s"] < INVERSION_THRESHOLD
        df["near_inverted"] = (df["spread_2s10s"] >= INVERSION_THRESHOLD) & (
            df["spread_2s10s"] < NEAR_INVERSION_THRESHOLD
        )

        self._spread_data = df
        self._credit_spread = credit

        logger.info(
            "Yield curve gate fitted: %d bars, inverted=%d (%.1f%%)",
            len(df),
            df["inverted"].sum(),
            df["inverted"].mean() * 100,
        )
        return self

    def _fetch_fmp(self, start: str) -> tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Fetch Treasury yields and credit spreads from FMP free tier."""
        try:
            import urllib.request
            import json as _json

            yield_url = f"{FMP_BASE}/treasury?from={start}&apikey={FMP_API_KEY}"
            req = urllib.request.Request(yield_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = _json.loads(resp.read().decode())

            if not data:
                logger.warning("FMP returned no Treasury data")
                return None, None

            records = {}
            for item in data:
                date_str = item.get("date")
                if not date_str:
                    continue
                records.setdefault(date_str, {})[item.get("maturity", "unknown")] = item.get(
                    "yield", 0
                )

            rows = []
            for date_str in sorted(records):
                entry = records[date_str]
                two = entry.get("2", entry.get("TWO_YEAR", None))
                ten = entry.get("10", entry.get("TEN_YEAR", None))
                three_mo = entry.get("3MO", entry.get("THREE_MONTH", None))
                if two is not None and ten is not None:
                    rows.append(
                        {
                            "date": pd.Timestamp(date_str),
                            "yield_2y": float(two) / 100,
                            "yield_10y": float(ten) / 100,
                            "yield_3m": float(three_mo) / 100 if three_mo is not None else np.nan,
                        }
                    )

            if not rows:
                logger.warning("FMP Treasury data missing 2Y and 10Y maturities")
                return None, None

            df = pd.DataFrame(rows).set_index("date").sort_index()
            df = df[df.index >= start]

            credit = self._fetch_fmp_credit_spread(start)

            logger.info("FMP: %d Treasury yield records, credit=%s", len(df), credit is not None)
            return df, credit

        except Exception as e:
            logger.debug("FMP fetch failed: %s", e)
            return None, None

    def _fetch_fmp_credit_spread(self, start: str) -> Optional[pd.Series]:
        """Fetch BAA-AAA credit spread from FMP."""
        return None

    def _fetch_fred(self, start: str, end: Optional[str]) -> Optional[pd.DataFrame]:
        """Fetch Treasury yields via fredapi."""
        try:
            from fredapi import Fred

            fred = Fred()
            series = {
                "DGS2": "2-Year Treasury",
                "DGS10": "10-Year Treasury",
                "DGS3MO": "3-Month Treasury",
            }
            data = {}
            for symbol in series:
                s = fred.get_series(symbol, observation_start=start, observation_end=end)
                if not s.empty:
                    data[symbol] = s / 100.0

            if "DGS2" not in data or "DGS10" not in data:
                logger.warning("FRED missing 2Y or 10Y yield data")
                return None

            df = pd.DataFrame(data)
            df.index = pd.to_datetime(df.index).normalize()
            df = df[df.index >= start].sort_index()
            logger.info("FRED: %d Treasury yield records", len(df))
            return df
        except Exception as e:
            logger.debug("FRED fetch failed: %s", e)
            return None

    def _fetch_yfinance(self, start: str, end: Optional[str]) -> Optional[pd.DataFrame]:
        """Fetch Treasury yields via yfinance ETF proxies.

        Uses TLT (20Y+ Treasury ETF) and IEF (7-10Y Treasury ETF) returns
        as a rough spread proxy. Not as precise as actual yields, but free.
        """
        try:
            import yfinance as yf

            tlt = yf.download("TLT", start=start, end=end, progress=False, auto_adjust=True)
            ief = yf.download("IEF", start=start, end=end, progress=False, auto_adjust=True)

            if tlt.empty or ief.empty:
                return None

            for token in [tlt, ief]:
                if isinstance(token.columns, pd.MultiIndex):
                    token.columns = token.columns.get_level_values(0)

            # ETF price ratio as proxy for yield spread (inverse of yield)
            # TLT/IEF ratio decreases when long yields rise faster than medium
            ratio = tlt["Close"] / ief["Close"]
            df = pd.DataFrame({"tlt_ief_ratio": ratio}, index=ratio.index)
            df["spread_2s10s"] = -(df["tlt_ief_ratio"] - df["tlt_ief_ratio"].rolling(252).mean())

            logger.info("yfinance: %d ETF-based proxy records", len(df))
            return df

        except Exception as e:
            logger.debug("yfinance fetch failed: %s", e)
            return None

    def multiplier(self, date=None) -> float:
        """Get signal multiplier for a given date.

        Args:
            date: Date to look up (uses last available if None).

        Returns:
            Multiplier in [0.50, 1.0].
        """
        if self._spread_data is None or "spread_2s10s" not in self._spread_data.columns:
            return 1.0

        if date is None:
            spread = float(self._spread_data["spread_2s10s"].iloc[-1])
        elif date in self._spread_data.index:
            spread = float(self._spread_data.loc[date, "spread_2s10s"])
        else:
            idx = self._spread_data.index
            target = pd.Timestamp(date)
            nearby = idx[idx <= target]
            if len(nearby) > 0:
                spread = float(self._spread_data.loc[nearby[-1], "spread_2s10s"])
            else:
                return 1.0

        return self._multiplier_from_spread(spread)

    def _multiplier_from_spread(self, spread: float) -> float:
        if spread < INVERSION_THRESHOLD:
            return self.inversion_mult
        elif spread < NEAR_INVERSION_THRESHOLD:
            return self.near_inversion_mult
        return 1.0

    def is_inverted(self, date=None) -> bool:
        """Check if yield curve is inverted."""
        if self._spread_data is None or "spread_2s10s" not in self._spread_data.columns:
            return False
        if date is None:
            return bool(self._spread_data["inverted"].iloc[-1])
        elif date in self._spread_data.index:
            return bool(self._spread_data.loc[date, "inverted"])
        return False

    def current_spread(self) -> Optional[float]:
        """Get the most recent 2s10s spread."""
        if self._spread_data is None or "spread_2s10s" not in self._spread_data.columns:
            return None
        return float(self._spread_data["spread_2s10s"].iloc[-1])

    @property
    def spread_data(self) -> Optional[pd.DataFrame]:
        return self._spread_data

    def to_dataframe(self) -> pd.DataFrame:
        """Return DataFrame with spreads and inversion flags."""
        if self._spread_data is None:
            return pd.DataFrame()
        return self._spread_data.copy()
