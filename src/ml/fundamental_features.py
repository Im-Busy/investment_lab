"""Multi-factor fundamental factor extraction for ML pipeline integration.

Phase Q: Extract fundamental factors (Value, Quality, Size, Growth, Income,
Risk, Sentiment) from yfinance for integration into the CatBoost feature
pipeline. Complements the 88 technical features with a new data dimension.

Usage:
    extractor = FundamentalFeatureExtractor()
    factors = extractor.extract(["SPY", "QQQ", "AAPL"])
    composites = extractor.composite_score(factors)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FUNDAMENTAL_FACTOR_CONFIG: dict[str, dict[str, str]] = {
    "pe_ratio": {"field": "trailingPE", "category": "Value", "direction": "lower"},
    "pb_ratio": {"field": "priceToBook", "category": "Value", "direction": "lower"},
    "ps_ratio": {
        "field": "priceToSalesTrailing12Months",
        "category": "Value",
        "direction": "lower",
    },
    "ev_ebitda": {"field": "enterpriseToEbitda", "category": "Value", "direction": "lower"},
    "roe": {"field": "returnOnEquity", "category": "Quality", "direction": "higher"},
    "roa": {"field": "returnOnAssets", "category": "Quality", "direction": "higher"},
    "profit_margin": {"field": "profitMargins", "category": "Quality", "direction": "higher"},
    "debt_equity": {"field": "debtToEquity", "category": "Quality", "direction": "lower"},
    "market_cap": {"field": "marketCap", "category": "Size", "direction": "higher"},
    "revenue_growth": {"field": "revenueGrowth", "category": "Growth", "direction": "higher"},
    "earnings_growth": {"field": "earningsGrowth", "category": "Growth", "direction": "higher"},
    "dividend_yield": {"field": "dividendYield", "category": "Income", "direction": "higher"},
    "beta": {"field": "beta", "category": "Risk", "direction": "lower"},
    "short_pct_float": {
        "field": "shortPercentOfFloat",
        "category": "Sentiment",
        "direction": "lower",
    },
}

FACTOR_NAMES: list[str] = list(FUNDAMENTAL_FACTOR_CONFIG.keys())
FACTOR_DIRECTIONS: dict[str, str] = {
    k: v["direction"] for k, v in FUNDAMENTAL_FACTOR_CONFIG.items()
}
FACTOR_FIELDS: dict[str, str] = {k: v["field"] for k, v in FUNDAMENTAL_FACTOR_CONFIG.items()}


class FundamentalFeatureExtractor:
    """Extract fundamental factors from yfinance.

    Fetches per-ticker fundamental data via yfinance.ticker.info and
    returns a cross-sectionally z-scored DataFrame for ML integration.

    Fundamental data is point-in-time (quarterly/annual filings). The
    extractor returns the current snapshot. For backtesting, use
    QuarterlyFundamentalProvider to align with price data.

    Attributes:
        source: Data source identifier (currently only "yfinance").
        factor_names: List of extracted factor column names.
    """

    def __init__(self, source: str = "yfinance") -> None:
        if source != "yfinance":
            raise ValueError(f"Unsupported source: {source}. Use 'yfinance'.")
        self.source = source
        self.factor_names = FACTOR_NAMES

    def extract(
        self,
        tickers: list[str],
        delay: float = 0.1,
        retries: int = 2,
    ) -> pd.DataFrame:
        """Extract fundamental factors for a list of tickers.

        Args:
            tickers: List of ticker symbols.
            delay: Seconds to wait between yfinance calls (to avoid rate-limiting).
            retries: Number of retries per ticker on failure.

        Returns:
            DataFrame indexed by ticker, columns = factor names.
            Missing values are NaN. Call z_score_factors() to normalize.
        """
        import time
        import yfinance as yf

        rows: dict[str, dict[str, float | None]] = {}
        fetch_count = 0

        for ticker in tickers:
            ticker_data: dict[str, float | None] = {}
            for attempt in range(retries + 1):
                try:
                    t = yf.Ticker(ticker)
                    info = t.info
                    for factor_name, field in FACTOR_FIELDS.items():
                        raw = info.get(field)
                        ticker_data[factor_name] = float(raw) if raw is not None else None
                    fetch_count += 1
                    break
                except Exception as e:
                    if attempt < retries:
                        logger.debug(
                            "yfinance fetch for %s attempt %d failed: %s",
                            ticker,
                            attempt + 1,
                            e,
                        )
                        time.sleep(delay * 2)
                    else:
                        logger.warning(
                            "Failed to fetch fundamentals for %s after %d attempts",
                            ticker,
                            retries + 1,
                        )
                        ticker_data = {fn: None for fn in FACTOR_NAMES}

            rows[ticker] = ticker_data
            if delay > 0 and ticker != tickers[-1]:
                time.sleep(delay)

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "ticker"

        n_fetched = int((df.notna().sum(axis=1) > 0).sum())
        logger.info(
            "Fundamental factors extracted: %d/%d tickers with data, %d API calls",
            n_fetched,
            len(tickers),
            fetch_count,
        )
        return df

    def z_score_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectionally z-score all fundamental factors.

        Z-scoring is done per-column (factor) across all tickers.
        This makes factors comparable and suitable for composite scoring.
        Direction signs are adjusted so that higher z-score always means
        "better" (e.g., lower P/E -> higher z-score).

        Args:
            df: Raw factor DataFrame from extract().

        Returns:
            Z-scored DataFrame with direction-adjusted signs.
        """
        z = pd.DataFrame(index=df.index, dtype=float)
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) < 2 or col_data.std(ddof=0) == 0:
                z[col] = 0.0
            else:
                z[col] = (df[col] - col_data.mean()) / col_data.std(ddof=0)

        for factor, direction in FACTOR_DIRECTIONS.items():
            if factor not in z.columns:
                continue
            if direction == "lower":
                z[factor] = -z[factor]

        return z

    def composite_score(
        self,
        df: pd.DataFrame,
        weights: dict[str, float] | None = None,
    ) -> pd.Series:
        """Compute weighted multi-factor composite score.

        Z-scores the raw factors, then computes a weighted sum. Higher
        composite score = more attractive on fundamentals.

        Args:
            df: Raw factor DataFrame from extract().
            weights: Dict of factor_name -> weight. If None, uses equal
                weights for all available factors.

        Returns:
            pd.Series of composite scores, indexed by ticker.
        """
        z = self.z_score_factors(df)

        if weights is None:
            available = [c for c in z.columns if z[c].notna().any()]
            if not available:
                return pd.Series(np.nan, index=df.index, dtype=float)
            weights = {c: 1.0 / len(available) for c in available}

        weighted_cols = [c for c in weights if c in z.columns]
        if not weighted_cols:
            return pd.Series(np.nan, index=df.index, dtype=float)

        composite = pd.Series(0.0, index=df.index, dtype=float)
        for col in weighted_cols:
            col_data = z[col].fillna(0.0)
            composite += col_data * weights[col]

        return composite

    def extract_all(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """Extract both raw and z-scored factors plus composite.

        Returns:
            dict with keys: 'raw', 'zscored', 'composite'.
        """
        raw = self.extract(tickers)
        zscored = self.z_score_factors(raw)
        composite = self.composite_score(raw)
        return {
            "raw": raw,
            "zscored": zscored,
            "composite": composite,
        }

    def factor_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate per-factor summary statistics.

        Returns DataFrame with: count, mean, std, min, 25%, 50%, 75%, max.
        """
        return df.describe().T

    def to_feature_columns(
        self,
        df: pd.DataFrame,
        z_score: bool = True,
    ) -> pd.DataFrame:
        """Convert fundamental factors to ML feature columns.

        Takes per-ticker factor DataFrame and returns a wide-format
        DataFrame suitable for joining with technical features.
        Each factor becomes a column prefixed with 'fund_'.

        Args:
            df: Raw factor DataFrame from extract().
            z_score: If True, z-score factors first.

        Returns:
            DataFrame with columns like 'fund_pe_ratio', 'fund_roe', etc.
        """
        if z_score and len(df) > 1:
            df = self.z_score_factors(df)

        renamed = df.rename(columns=lambda c: f"fund_{c}")
        return renamed


class QuarterlyFundamentalProvider:
    """Provide quarterly fundamental data aligned with price data for backtesting.

    Since fundamentals update infrequently (quarterly/annual), this class
    creates a time-series of fundamental features by forward-filling the
    latest available fundamental values through each trading day.

    For production use, this should be replaced with actual point-in-time
    fundamental data (Compustat, Bloomberg, etc.).
    """

    def __init__(
        self,
        extractor: FundamentalFeatureExtractor | None = None,
    ) -> None:
        self.extractor = extractor or FundamentalFeatureExtractor()

    def build_time_series(
        self,
        ticker: str,
        price_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Build fundamental feature time series for a single ticker.

        Extracts current fundamentals and forward-fills through the price
        index. This is a simplification - real implementation would use
        quarterly-snapshot data aligned with filing dates.

        Args:
            ticker: Ticker symbol.
            price_index: DatetimeIndex of trading days.

        Returns:
            DataFrame with fundamental factor columns, indexed by price_index.
            Values are forward-filled constants (current snapshot).
        """
        df = self.extractor.extract([ticker])
        if df.empty or df.isna().all(axis=1).iloc[0]:
            logger.warning("No fundamental data for %s", ticker)
            result = pd.DataFrame(
                {f"fund_{fn}": np.nan for fn in FACTOR_NAMES},
                index=price_index,
            )
            return result

        row = df.iloc[0]
        result = pd.DataFrame(index=price_index)
        for fn in FACTOR_NAMES:
            result[f"fund_{fn}"] = row.get(fn, np.nan)

        return result
