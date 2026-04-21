"""
Cross-Asset Feature Engineering

Generates features using cross-asset data (VIX, bonds, gold, oil, sector ETFs)
to improve regime detection and signal quality.

Feature Categories:
1. VIX & Volatility Index features
2. Yield Curve / Bond features (using TLT/IEF as proxies)
3. Safe-haven assets (gold, USD)
4. Oil / commodity features
5. Sector ETF correlation features
6. Cross-asset momentum divergence
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class CrossAssetFeatures:
    """
    Generate cross-asset features for ML regime detection.

    Example:
        >>> ca = CrossAssetFeatures(
        ...     sector_tickers=["XLK", "XLF", "XLE", "XLV"],
        ... )
        >>> features = ca.generate_features(spy_df, cross_asset_data)
    """

    def __init__(
        self,
        sector_tickers: Optional[List[str]] = None,
    ):
        self.sector_tickers = sector_tickers or [
            "XLK",
            "XLF",
            "XLE",
            "XLV",
            "XLI",
            "XLP",
            "XLU",
            "XLY",
            "XLB",
        ]

    def generate_features(
        self,
        df: pd.DataFrame,
        cross_asset_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Generate cross-asset features.

        Args:
            df: Primary OHLCV DataFrame (SPY)
            cross_asset_data: Dict of ticker -> OHLCV DataFrame with Close column

        Returns:
            DataFrame with cross-asset feature columns
        """
        features = pd.DataFrame(index=df.index)

        # VIX features
        features = self._add_vix_features(features, cross_asset_data)

        # Bond / yield curve features
        features = self._add_bond_features(features, cross_asset_data)

        # Safe-haven features
        features = self._add_safe_haven_features(features, cross_asset_data)

        # Oil features
        features = self._add_oil_features(features, cross_asset_data)

        # Sector correlation features
        features = self._add_sector_features(features, df, cross_asset_data)

        # Cross-asset momentum divergence
        features = self._add_momentum_divergence(features, df, cross_asset_data)

        return features

    def _add_vix_features(
        self,
        features: pd.DataFrame,
        data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """VIX-related features: level, changes, term structure proxy."""
        if "^VIX" not in data and "VIX" not in data and "VIXY" not in data:
            return features

        # Try different VIX ticker formats
        vix_df = data.get("^VIX")
        if vix_df is None:
            vix_df = data.get("VIX")
        if vix_df is None:
            vix_df = data.get("VIXY")
        vix_close = vix_df["Close"].reindex(features.index)
        vix_close = vix_close.ffill().bfill()

        # VIX level
        features["vix_level"] = vix_close

        # VIX return
        features["vix_return"] = vix_close.pct_change()

        # VIX moving averages
        features["vix_ma5"] = vix_close.rolling(5).mean()
        features["vix_ma20"] = vix_close.rolling(20).mean()

        # VIX term structure proxy (short vs long MA)
        features["vix_slope"] = vix_close.rolling(5).mean() / vix_close.rolling(20).mean() - 1

        # VIX percentile (rolling 252-day)
        features["vix_percentile"] = vix_close.rolling(252, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )

        # VIX change over windows
        for w in [5, 20]:
            features[f"vix_return_{w}"] = vix_close.pct_change(w)

        return features

    def _add_bond_features(
        self,
        features: pd.DataFrame,
        data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Bond proxy features using TLT (long-term) / IEF (medium-term)."""
        tlt_df = data.get("TLT")
        ief_df = data.get("IEF")

        if tlt_df is None or ief_df is None:
            return features

        tlt_close = tlt_df["Close"].reindex(features.index).ffill().bfill()
        ief_close = ief_df["Close"].reindex(features.index).ffill().bfill()

        # Yield curve proxy (bond price ratio — inversely related to yield spread)
        features["tlt_ief_ratio"] = tlt_close / ief_close

        # Bond returns
        features["tlt_return"] = tlt_close.pct_change()
        features["ief_return"] = ief_close.pct_change()

        # Bond momentum
        features["tlt_momentum_20"] = tlt_close.pct_change(20)
        features["ief_momentum_20"] = ief_close.pct_change(20)

        # Bond momentum divergence
        features["bond_momentum_div"] = tlt_close.pct_change(20) - ief_close.pct_change(20)

        # TLT volatility
        features["tlt_volatility"] = tlt_close.pct_change().rolling(20).std()
        features["ief_volatility"] = ief_close.pct_change().rolling(20).std()

        # TLT/SPY rolling correlation
        spy_close = None
        if "Close" in features.index.names:
            pass
        return features

    def _add_safe_haven_features(
        self,
        features: pd.DataFrame,
        data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Gold (GLD) and USD (UUP) features."""
        gld_df = data.get("GLD")
        uup_df = data.get("UUP")

        if gld_df is not None:
            gld_close = gld_df["Close"].reindex(features.index).ffill().bfill()
            features["gold_return"] = gld_close.pct_change()
            features["gold_momentum_20"] = gld_close.pct_change(20)
            features["gold_volatility"] = gld_close.pct_change().rolling(20).std()

        if uup_df is not None:
            uup_close = uup_df["Close"].reindex(features.index).ffill().bfill()
            features["usd_return"] = uup_close.pct_change()
            features["usd_momentum_20"] = uup_close.pct_change(20)

        return features

    def _add_oil_features(
        self,
        features: pd.DataFrame,
        data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Oil (USO) features."""
        uso_df = data.get("USO")

        if uso_df is None:
            return features

        uso_close = uso_df["Close"].reindex(features.index).ffill().bfill()
        features["oil_return"] = uso_close.pct_change()
        features["oil_momentum_20"] = uso_close.pct_change(20)
        features["oil_volatility"] = uso_close.pct_change().rolling(20).std()

        return features

    def _add_sector_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Sector ETF correlation and relative strength features."""
        spy_close = df["Close"]

        for ticker in self.sector_tickers:
            sector_df = data.get(ticker)
            if sector_df is None:
                continue

            sector_close = sector_df["Close"].reindex(features.index).ffill().bfill()

            # Rolling correlation with SPY
            corr = sector_close.rolling(60, min_periods=20).corr(spy_close)
            features[f"{ticker.lower()}_corr_spy"] = corr

            # Relative strength (sector / SPY)
            rs = sector_close / spy_close
            features[f"{ticker.lower()}_rs"] = rs

            # Sector momentum
            for w in [20, 60]:
                features[f"{ticker.lower()}_momentum_{w}"] = sector_close.pct_change(w)

        # Cross-sector dispersion (std of sector returns)
        sector_returns = []
        for ticker in self.sector_tickers:
            sector_df = data.get(ticker)
            if sector_df is not None:
                ret = sector_df["Close"].reindex(features.index).ffill().bfill().pct_change()
                sector_returns.append(ret)

        if sector_returns:
            ret_df = pd.DataFrame({t: r for t, r in zip(self.sector_tickers, sector_returns)})
            features["sector_dispersion"] = ret_df.std(axis=1)
            features["sector_breadth"] = (ret_df > 0).sum(axis=1) / len(sector_returns)

        return features

    def _add_momentum_divergence(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Cross-asset momentum divergence features."""
        spy_close = df["Close"]
        spy_ret = spy_close.pct_change(20)

        # SPY vs bond momentum divergence (risk-on vs risk-off)
        tlt_df = data.get("TLT")
        if tlt_df is not None:
            tlt_ret = tlt_df["Close"].reindex(features.index).ffill().bfill().pct_change(20)
            features["spy_tlt_momentum_div"] = spy_ret - tlt_ret

        # SPY vs gold momentum
        gld_df = data.get("GLD")
        if gld_df is not None:
            gld_ret = gld_df["Close"].reindex(features.index).ffill().bfill().pct_change(20)
            features["spy_gld_momentum_div"] = spy_ret - gld_ret

        # SPY vs VIX momentum (inverse relationship)
        vix_df = data.get("^VIX")
        if vix_df is None:
            vix_df = data.get("VIX")
        if vix_df is None:
            vix_df = data.get("VIXY")
        if vix_df is not None:
            vix_ret = vix_df["Close"].reindex(features.index).ffill().bfill().pct_change(20)
            features["spy_vix_momentum_div"] = spy_ret - vix_ret

        return features


def prepare_cross_asset_data(
    spy_df: pd.DataFrame,
    tickers: Optional[List[str]] = None,
    start: str = "2014-01-01",
    end: str = "2024-12-31",
) -> Dict[str, pd.DataFrame]:
    """
    Download and prepare cross-asset data aligned to SPY index.

    Args:
        spy_df: SPY OHLCV DataFrame
        tickers: List of ticker symbols to download
        start: Start date for data download
        end: End date for data download

    Returns:
        Dict of ticker -> DataFrame with Close column
    """
    import yfinance as yf

    if tickers is None:
        tickers = [
            "^VIX",
            "TLT",
            "IEF",
            "GLD",
            "USO",
            "UUP",
            "XLK",
            "XLF",
            "XLE",
            "XLV",
            "XLI",
            "XLP",
            "XLU",
            "XLY",
            "XLB",
        ]

    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Close" in df.columns:
                df.index.name = "date"
                df = df.reindex(spy_df.index)
                df["Close"] = df["Close"].ffill().bfill()
                data[t] = df
        except Exception:
            pass

    return data
