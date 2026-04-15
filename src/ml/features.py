"""
ML Feature Engineering for Trading Signals

Generates features for machine learning models from OHLCV data and indicator outputs.

Feature Categories:
1. Price Features - returns, gaps, ranges
2. Momentum Features - RSI, MACD, rate of change
3. Volatility Features - ATR, Bollinger bandwidth, standard deviation
4. Volume Features - volume ratio, OBV
5. Pattern Features - pattern detection flags, confluence scores
6. Regime Features - ADX, trend strength
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Generate ML features from OHLCV data.

    Example:
        >>> eng = FeatureEngineer()
        >>> features = eng.generate_features(df)
        >>> features.head()
    """

    def __init__(
        self,
        price_windows: Optional[List[int]] = None,
        momentum_windows: Optional[List[int]] = None,
        volatility_windows: Optional[List[int]] = None,
    ):
        self.price_windows = price_windows or [5, 10, 20, 50]
        self.momentum_windows = momentum_windows or [5, 10, 14, 20]
        self.volatility_windows = volatility_windows or [10, 20, 50]

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from OHLCV data.

        Args:
            df: OHLCV DataFrame with columns: High, Low, Close, Open, Volume

        Returns:
            DataFrame with feature columns
        """
        features = pd.DataFrame(index=df.index)

        # Price-based features
        features = self._add_price_features(features, df)

        # Momentum features
        features = self._add_momentum_features(features, df)

        # Volatility features
        features = self._add_volatility_features(features, df)

        # Volume features
        features = self._add_volume_features(features, df)

        # Pattern-like features (simplified)
        features = self._add_pattern_features(features, df)

        # Regime features
        features = self._add_regime_features(features, df)

        return features

    def _add_price_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        close = df["Close"]

        # Returns at multiple windows
        for w in self.price_windows:
            features[f"return_{w}"] = close.pct_change(w)
            features[f"log_return_{w}"] = np.log(close / close.shift(w))

        # Price position relative to moving averages
        for w in self.price_windows:
            ma = close.rolling(w).mean()
            features[f"price_to_ma_{w}"] = (close - ma) / ma

        # High-low range
        features["hl_range"] = (df["High"] - df["Low"]) / close

        # Open-close range
        features["oc_range"] = (close - df["Open"]) / df["Open"]

        # Gap from previous close
        features["gap"] = (df["Open"] - close.shift(1)) / close.shift(1)

        # Distance to N-bar high/low
        for w in self.price_windows:
            high_n = df["High"].rolling(w).max()
            low_n = df["Low"].rolling(w).min()
            features[f"dist_to_high_{w}"] = (close - high_n) / high_n
            features[f"dist_to_low_{w}"] = (close - low_n) / low_n

        return features

    def _add_momentum_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features."""
        close = df["Close"]

        # RSI at multiple windows
        for w in [5, 10, 14, 20]:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.rolling(w).mean()
            avg_loss = loss.rolling(w).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            features[f"rsi_{w}"] = 100 - (100 / (1 + rs))

        # Rate of change
        for w in self.momentum_windows:
            features[f"roc_{w}"] = close.pct_change(w) * 100

        # MACD
        ema_fast = close.ewm(span=12).mean()
        ema_slow = close.ewm(span=26).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9).mean()
        features["macd"] = macd_line
        features["macd_signal"] = signal_line
        features["macd_histogram"] = macd_line - signal_line
        features["macd_cross"] = (macd_line - signal_line).diff() > 0

        # Stochastic
        for w in [14, 21]:
            low_n = df["Low"].rolling(w).min()
            high_n = df["High"].rolling(w).max()
            stoch_k = 100 * (close - low_n) / (high_n - low_n).replace(0, np.nan)
            features[f"stoch_k_{w}"] = stoch_k
            features[f"stoch_d_{w}"] = stoch_k.rolling(3).mean()

        # Momentum (simple price difference)
        for w in self.momentum_windows:
            features[f"momentum_{w}"] = close - close.shift(w)

        return features

    def _add_volatility_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        close = df["Close"]

        # ATR
        high = df["High"]
        low = df["Low"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)

        for w in self.volatility_windows:
            features[f"atr_{w}"] = tr.rolling(w).mean()
            # ATR as % of price
            features[f"atr_pct_{w}"] = features[f"atr_{w}"] / close * 100

        # Historical volatility
        returns = close.pct_change()
        for w in self.volatility_windows:
            features[f"volatility_{w}"] = returns.rolling(w).std() * np.sqrt(252)

        # Bollinger Bands
        for w in [10, 20]:
            ma = close.rolling(w).mean()
            std = close.rolling(w).std()
            bb_upper = ma + 2 * std
            bb_lower = ma - 2 * std
            features[f"bb_pct_{w}"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
            features[f"bb_width_{w}"] = (bb_upper - bb_lower) / ma

        # Standard deviation of returns
        for w in [10, 20, 50]:
            features[f"std_return_{w}"] = returns.rolling(w).std()

        return features

    def _add_volume_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        volume = df["Volume"]

        # Volume ratio
        for w in [5, 10, 20]:
            vol_ma = volume.rolling(w).mean()
            features[f"volume_ratio_{w}"] = volume / vol_ma.replace(0, np.nan)

        # On-Balance Volume (OBV)
        obv = (np.sign(df["Close"].diff()) * volume).cumsum()
        features["obv"] = obv
        features["obv_change"] = obv.diff(5) / obv.replace(0, np.nan)

        # Volume-weighted average price
        for w in [10, 20]:
            vwap = (df["Close"] * volume).rolling(w).sum() / volume.rolling(w).sum()
            features[f"vwap_{w}"] = vwap
            features[f"close_to_vwap_{w}"] = (df["Close"] - vwap) / vwap.replace(0, np.nan)

        return features

    def _add_pattern_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add simplified pattern-like features."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # Doji-like (small body)
        body = (close - df["Open"]).abs()
        hl_range = high - low
        features["doji"] = (body / hl_range.replace(0, np.nan)).clip(0, 1)

        # Engulfing-like
        prev_body = body.shift(1) / hl_range.shift(1).replace(0, np.nan)
        current_body = body / hl_range.replace(0, np.nan)
        features["engulfing"] = (current_body > prev_body).astype(float)

        # Higher high / Lower low
        features["higher_high"] = (high > high.shift(1)).astype(float)
        features["lower_low"] = (low < low.shift(1)).astype(float)

        # Consecutive up/down days
        direction = close.diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        features["consecutive_dir"] = direction.rolling(5).sum() / 5

        # NR7-like (narrowest range in 7 days)
        for w in [7, 14]:
            rank = hl_range.rolling(w).rank()
            features[f"nr{w}"] = (rank == 1).astype(float)

        return features

    def _add_regime_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime-related features (ADX, trend strength)."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # ADX
        period = 14
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        atr = tr.rolling(period).mean()

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0),
            index=close.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0),
            index=close.index,
        )

        plus_smooth = plus_dm.rolling(period).mean()
        minus_smooth = minus_dm.rolling(period).mean()
        plus_di = 100 * plus_smooth / atr.replace(0, np.nan)
        minus_di = 100 * minus_smooth / atr.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(period).mean()

        features["adx"] = adx
        features["plus_di"] = plus_di
        features["minus_di"] = minus_di
        features["di_diff"] = plus_di - minus_di

        # Trend strength (slope of moving average)
        for w in [20, 50]:
            ma = close.rolling(w).mean()
            slope = ma.diff(5) / ma.shift(5)
            features[f"slope_{w}"] = slope

        # Volatility regime
        atr_20 = tr.rolling(20).mean()
        atr_100 = tr.rolling(100).mean()
        features["vol_regime"] = atr_20 / atr_100.replace(0, np.nan)

        return features

    def get_feature_names(
        self,
        exclude_nan: bool = True,
        exclude_cols: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Get list of feature names.

        Args:
            exclude_nan: Whether to exclude features with all NaN
            exclude_cols: Columns to exclude

        Returns:
            List of feature names
        """
        exclude = exclude_cols or []
        exclude_set = set(exclude)

        names = []
        for col in self.generate_features(
            pd.DataFrame(
                {
                    "Open": [100.0] * 200,
                    "High": [101.0] * 200,
                    "Low": [99.0] * 200,
                    "Close": [100.0] * 200,
                    "Volume": [1000000.0] * 200,
                }
            )
        ).columns:
            if col not in exclude_set:
                names.append(col)

        return names
