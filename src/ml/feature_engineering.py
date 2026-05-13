"""
Enhanced Feature Engineering for ML Pattern Detection

Extracts comprehensive features from OHLCV data for ML model training.
Includes technical indicators, pattern-specific metrics, and regime features.

Feature Categories:
1. Technical Features - RSI, MACD, volatility, volume
2. Pattern-Specific Features - shape, magnitude, context
3. Regime Features - trend state, volatility regime
4. Forward Returns - for label generation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    price_windows: List[int] = None
    momentum_windows: List[int] = None
    volatility_windows: List[int] = None
    volume_windows: List[int] = None

    def __post_init__(self):
        if self.price_windows is None:
            self.price_windows = [5, 10, 20, 50, 100]
        if self.momentum_windows is None:
            self.momentum_windows = [5, 10, 14, 20, 50]
        if self.volatility_windows is None:
            self.volatility_windows = [10, 20, 50, 100]
        if self.volume_windows is None:
            self.volume_windows = [5, 10, 20, 50]


class FeatureExtractor:
    """
    Comprehensive feature extractor for trading patterns.

    Extracts 50-100 features per signal across multiple categories:
    - Technical indicators (RSI, MACD, Bollinger Bands)
    - Pattern-specific metrics (shape, magnitude, context)
    - Regime features (trend, volatility state)
    - Volume and liquidity features

    Example:
        >>> extractor = FeatureExtractor()
        >>> features = extractor.extract_all_features(df)
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature extractor.

        Args:
            config: Feature extraction configuration
        """
        self.config = config or FeatureConfig()

    def extract_all_features(
        self,
        df: pd.DataFrame,
        include_forward_returns: bool = False,
    ) -> pd.DataFrame:
        """
        Extract all feature categories from OHLCV data.

        Args:
            df: OHLCV DataFrame with columns: Open, High, Low, Close, Volume
            include_forward_returns: If True, include forward return columns
                for label generation. Default False to prevent data leakage
                (forward returns are labels, not features).

        Returns:
            DataFrame with all features
        """
        col_map: dict[str, pd.Series] = {}

        self._add_price_features(col_map, df)
        self._add_momentum_features(col_map, df)
        self._add_volatility_features(col_map, df)
        self._add_volume_features(col_map, df)
        self._add_pattern_shape_features(col_map, df)
        self._add_regime_features(col_map, df)

        if include_forward_returns:
            self._add_forward_returns(col_map, df)

        features = pd.DataFrame(col_map, index=df.index)
        features = features.dropna(axis=1, how="all")

        return features

    def _add_price_features(self, col_map: dict[str, pd.Series], df: pd.DataFrame) -> None:
        """Add price-based features."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        open_price = df["Open"]

        for w in self.config.price_windows:
            col_map[f"return_{w}"] = close.pct_change(w)
            col_map[f"log_return_{w}"] = np.log(close / close.shift(w))
            col_map[f"highest_{w}"] = high.rolling(w).max()
            col_map[f"lowest_{w}"] = low.rolling(w).min()

        for w in self.config.price_windows:
            ma = close.rolling(w).mean()
            col_map[f"price_to_ma_{w}"] = (close - ma) / ma
            col_map[f"ma_slope_{w}"] = ma.diff(w) / ma.shift(w)

        ema_8 = close.ewm(span=8).mean()
        ema_21 = close.ewm(span=21).mean()
        ema_55 = close.ewm(span=55).mean()
        col_map["ema_8"] = ema_8
        col_map["ema_21"] = ema_21
        col_map["ema_55"] = ema_55
        col_map["ema_8_21_spread"] = ema_8 - ema_21
        col_map["ema_21_55_spread"] = ema_21 - ema_55

        col_map["hl_range"] = (high - low) / close
        col_map["oc_range"] = (close - open_price) / open_price
        col_map["gap"] = (open_price - close.shift(1)) / close.shift(1)

        for w in [10, 20, 50]:
            high_n = high.rolling(w).max()
            low_n = low.rolling(w).min()
            col_map[f"price_position_{w}"] = (close - low_n) / (high_n - low_n + 1e-10)
            col_map[f"dist_to_high_{w}"] = (close - high_n) / high_n
            col_map[f"dist_to_low_{w}"] = (close - low_n) / low_n

    def _add_momentum_features(self, col_map: dict[str, pd.Series], df: pd.DataFrame) -> None:
        """Add momentum-based features."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        for w in [5, 10, 14, 20, 50]:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.rolling(w).mean()
            avg_loss = loss.rolling(w).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            col_map[f"rsi_{w}"] = 100 - (100 / (1 + rs))

        for w in self.config.momentum_windows:
            col_map[f"roc_{w}"] = close.pct_change(w) * 100
            col_map[f"momentum_{w}"] = close.diff(w)

        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        col_map["macd"] = macd_line
        col_map["macd_signal"] = signal_line
        col_map["macd_histogram"] = macd_line - signal_line
        col_map["macd_cross"] = (macd_line - signal_line).diff() > 0

        for w in [14, 21]:
            low_n = low.rolling(w).min()
            high_n = high.rolling(w).max()
            stoch_k = 100 * (close - low_n) / (high_n - low_n + 1e-10)
            col_map[f"stoch_k_{w}"] = stoch_k
            col_map[f"stoch_d_{w}"] = stoch_k.rolling(3).mean()

        close_pos = df["Close"].replace(0, np.nan)
        col_map["atr_14"] = self._compute_atr(df, 14) / close_pos
        col_map["atr_20"] = self._compute_atr(df, 20) / close_pos

    def _add_volatility_features(self, col_map: dict[str, pd.Series], df: pd.DataFrame) -> None:
        """Add volatility-based features."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        returns = close.pct_change()

        for w in self.config.volatility_windows:
            col_map[f"volatility_{w}"] = returns.rolling(w).std() * np.sqrt(252)
            col_map[f"std_return_{w}"] = returns.rolling(w).std()

        for w in [10, 20]:
            ma = close.rolling(w).mean()
            std = close.rolling(w).std()
            bb_upper = ma + 2 * std
            bb_lower = ma - 2 * std
            col_map[f"bb_pct_{w}"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
            col_map[f"bb_width_{w}"] = (bb_upper - bb_lower) / ma
            col_map[f"bb_squeeze_{w}"] = False

        # Compute bb_squeeze after bb_width is available in col_map
        for w in [10, 20]:
            bb_width_key = f"bb_width_{w}"
            col_map[f"bb_squeeze_{w}"] = (
                col_map[bb_width_key] < col_map[bb_width_key].rolling(20).mean()
            )

        col_map["volatility_regime"] = col_map["volatility_20"] / col_map["volatility_100"].replace(
            0, np.nan
        )

        col_map["volatility_zscore"] = (returns - rolling_mean(returns, 252)) / (
            rolling_std(returns, 252) + 1e-10
        )

    def _add_volume_features(self, col_map: dict[str, pd.Series], df: pd.DataFrame) -> None:
        """Add volume-based features."""
        volume = df["Volume"]
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        for w in self.config.volume_windows:
            vol_ma = volume.rolling(w).mean()
            col_map[f"volume_ratio_{w}"] = volume / (vol_ma + 1e-10)
            col_map[f"volume_zscore_{w}"] = (volume - vol_ma) / (volume.rolling(w).std() + 1e-10)

        obv = (np.sign(close.diff()) * volume).cumsum()
        col_map["obv"] = obv
        col_map["obv_change"] = obv.diff(5) / (obv.abs() + 1e-10)

        for w in [10, 20]:
            vwap = (close * volume).rolling(w).sum() / (volume.rolling(w).sum() + 1e-10)
            col_map[f"vwap_{w}"] = vwap
            col_map[f"close_to_vwap_{w}"] = (close - vwap) / (vwap + 1e-10)

        cmf = self._compute_chaikin_money_flow(df, window=20)
        col_map["cmf"] = cmf

        col_map["volume_trend"] = volume.rolling(10).mean() / volume.rolling(50).mean()

    def _add_pattern_shape_features(self, col_map: dict[str, pd.Series], df: pd.DataFrame) -> None:
        """Add pattern shape and magnitude features."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        open_price = df["Open"]

        body = (close - open_price).abs()
        hl_range = high - low
        col_map["body_ratio"] = body / (hl_range + 1e-10)
        col_map["upper_shadow"] = (high - close.clip(lower=open_price)) / (hl_range + 1e-10)
        col_map["lower_shadow"] = (close.clip(lower=open_price) - low) / (hl_range + 1e-10)

        col_map["doji"] = (body / (hl_range + 1e-10) < 0.1).astype(float)
        col_map["long_body"] = (body / hl_range > 0.7).astype(float)

        prev_body = body.shift(1)
        col_map["engulfing"] = ((body > prev_body) & (close.diff() != 0)).astype(float)

        col_map["higher_high"] = (high > high.shift(1)).astype(float)
        col_map["lower_low"] = (low < low.shift(1)).astype(float)
        col_map["higher_close"] = (close > close.shift(1)).astype(float)

        direction = close.diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        col_map["consecutive_dir"] = direction.rolling(5).sum() / 5

        for w in [7, 14, 21]:
            rank = hl_range.rolling(w).rank()
            col_map[f"narrow_range_{w}"] = (rank == 1).astype(float)

        col_map["inside_bar"] = ((high < high.shift(1)) & (low > low.shift(1))).astype(float)
        col_map["outside_bar"] = ((high > high.shift(1)) & (low < low.shift(1))).astype(float)

        for w in [5, 10, 20]:
            col_map[f"close_percentile_{w}"] = close.rolling(w).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10)
            )

    def _add_regime_features(self, col_map: dict[str, pd.Series], df: pd.DataFrame) -> None:
        """Add regime and trend state features."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

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
        plus_di = 100 * plus_smooth / (atr + 1e-10)
        minus_di = 100 * minus_smooth / (atr + 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()

        col_map["adx"] = adx
        col_map["plus_di"] = plus_di
        col_map["minus_di"] = minus_di
        col_map["di_diff"] = plus_di - minus_di
        col_map["adx_trend"] = adx.diff()

        for w in [20, 50]:
            ma = close.rolling(w).mean()
            slope = ma.diff(5) / (ma.shift(5) + 1e-10)
            col_map[f"ma_slope_{w}"] = slope
            col_map[f"price_vs_ma_{w}"] = (close - ma) / (ma + 1e-10)

        _atr_20 = col_map.get("atr_20", tr.rolling(20).mean())
        _atr_100 = col_map.get("atr_100", tr.rolling(100).mean())
        col_map["vol_regime"] = _atr_20 / (_atr_100 + 1e-10)

        col_map["trend_strength"] = adx / 100

        vix = col_map.get("volatility_20", close.pct_change().rolling(20).std())
        vix = vix.dropna()

        if len(vix) > 0 and not vix.isna().all():
            q33 = vix.quantile(0.33)
            q67 = vix.quantile(0.67)
            if not np.isnan(q33) and not np.isnan(q67) and q33 < q67:
                bins = [-np.inf, q33, q67, np.inf]
                col_map["vol_regime_state"] = pd.cut(
                    vix,
                    bins=bins,
                    labels=[0, 1, 2],
                    include_lowest=True,
                ).astype(float)
            else:
                col_map["vol_regime_state"] = 1.0
        else:
            col_map["vol_regime_state"] = 1.0

    def extract_labels(
        self,
        df: pd.DataFrame,
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Extract forward-return labels for supervised learning.

        Args:
            df: OHLCV DataFrame with Close column.
            horizons: List of forward horizons (default [1, 3, 5, 10, 20]).

        Returns:
            DataFrame with forward_return_N and forward_binary_N columns.
        """
        if horizons is None:
            horizons = [1, 3, 5, 10, 20]
        labels = pd.DataFrame(index=df.index)
        return self._add_forward_returns(labels, df)

    def _add_forward_returns(self, col_map: dict[str, pd.Series], df: pd.DataFrame) -> None:
        """Add forward returns for label generation."""
        close = df["Close"]

        for horizon in [1, 3, 5, 10, 20]:
            future_return = close.shift(-horizon) / close - 1
            col_map[f"forward_return_{horizon}"] = future_return

            col_map[f"forward_binary_{horizon}"] = (future_return > 0).astype(int)

            max_dd = close.shift(-horizon).rolling(horizon).min() / close - 1
            col_map[f"max_drawdown_{horizon}"] = max_dd

            max_runup = close.shift(-horizon).rolling(horizon).max() / close - 1
            col_map[f"max_runup_{horizon}"] = max_runup

    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Average True Range."""
        high = df["High"]
        low = df["Low"]
        prev_close = df["Close"].shift(1)

        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)

        return tr.rolling(period).mean()

    def _compute_chaikin_money_flow(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Compute Chaikin Money Flow."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        volume = df["Volume"]

        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
        mfv = mfm * volume

        cmf = mfv.rolling(window).sum() / volume.rolling(window).sum()

        return cmf

    def get_feature_names(self, exclude_nan: bool = True) -> List[str]:
        """Get list of feature names by generating sample features."""
        sample_df = pd.DataFrame(
            {
                "Open": np.random.randn(200).cumsum() + 100,
                "High": np.random.randn(200).cumsum() + 101,
                "Low": np.random.randn(200).cumsum() + 99,
                "Close": np.random.randn(200).cumsum() + 100,
                "Volume": np.random.randint(1000000, 2000000, 200),
            }
        )

        sample_features = self.extract_all_features(sample_df)

        if exclude_nan:
            non_nan_mask = sample_features.notna().all(axis=0)
            return list(sample_features.columns[non_nan_mask])

        return list(sample_features.columns)


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling mean."""
    return series.rolling(window).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling standard deviation."""
    return series.rolling(window).std()


def extract_features_for_signals(
    df: pd.DataFrame,
    signal_timestamps: Optional[pd.DatetimeIndex] = None,
    feature_extractor: Optional[FeatureExtractor] = None,
) -> pd.DataFrame:
    """
    Extract features for specific signal timestamps.

    Args:
        df: OHLCV DataFrame
        signal_timestamps: Timestamps of detected signals
        feature_extractor: FeatureExtractor instance

    Returns:
        Feature matrix for signal timestamps
    """
    if feature_extractor is None:
        feature_extractor = FeatureExtractor()

    all_features = feature_extractor.extract_all_features(df)

    if signal_timestamps is not None:
        signal_features = all_features.reindex(signal_timestamps)
        return signal_features.dropna(axis=0, how="all")

    return all_features.dropna(axis=0, how="all")
