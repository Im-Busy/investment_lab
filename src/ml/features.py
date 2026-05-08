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
7. Alpha Factors - formulaic alphas from WorldQuant 101

Includes IC-based feature evaluation and filtering against forward returns.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

FORWARD_RETURN_HORIZONS: tuple[int, ...] = (1, 5, 20)
DEFAULT_IC_THRESHOLD: float = 0.02


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

    def generate_features(
        self,
        df: pd.DataFrame,
        include_alphas: bool = True,
        drop_na: bool = False,
    ) -> pd.DataFrame:
        """
        Generate all features from OHLCV data.

        Args:
            df: OHLCV DataFrame with columns: High, Low, Close, Open, Volume
            include_alphas: Whether to include formulaic alpha factors.
            drop_na: If True, drop rows with any NaN after filling.
                     If False, forward/backward fill NaN (suitable for
                     testing or when caller handles NaN).

        Returns:
            DataFrame with feature columns
        """
        features = pd.DataFrame(index=df.index)

        features = self._add_price_features(features, df)
        features = self._add_momentum_features(features, df)
        features = self._add_volatility_features(features, df)
        features = self._add_volume_features(features, df)
        features = self._add_pattern_features(features, df)
        features = self._add_regime_features(features, df)

        if include_alphas:
            features = self._add_alpha_factors(features, df)

        # Replace inf/-inf with NaN before any NaN handling
        features = features.replace([np.inf, -np.inf], np.nan)
        # Drop columns that are entirely NaN (insufficient data for rolling windows)
        features = features.dropna(axis=1, how="all")

        if drop_na:
            # Forward-fill then drop any remaining NaN rows
            features = features.ffill()
            features = features.dropna()
        else:
            # Fill NaN gracefully (suitable for testing or preprocessing)
            features = features.ffill().bfill()

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

    def add_forward_returns(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        horizons: tuple[int, ...] = FORWARD_RETURN_HORIZONS,
    ) -> pd.DataFrame:
        """Add forward return columns as labels for supervised learning.

        Args:
            features: Existing features DataFrame.
            df: OHLCV DataFrame with Close column.
            horizons: Forward return horizons in days (default: 1, 5, 20).

        Returns:
            Features DataFrame with added forward_return_{h}d columns.
        """
        close = df["Close"]
        for h in horizons:
            features[f"forward_return_{h}d"] = close.pct_change(h).shift(-h)
        return features

    def generate_features_with_labels(
        self,
        df: pd.DataFrame,
        horizons: tuple[int, ...] = FORWARD_RETURN_HORIZONS,
    ) -> pd.DataFrame:
        """Generate features including forward return labels.

        Args:
            df: OHLCV DataFrame.
            horizons: Forward return horizons.

        Returns:
            Features DataFrame with forward_return_Xd columns.
        """
        features = self.generate_features(df)
        features = self.add_forward_returns(features, df, horizons)
        return features

    def compute_ic_summary(
        self,
        df: pd.DataFrame,
        forward_returns: pd.Series | str = 0,
    ) -> pd.DataFrame:
        """Compute IC summary for all feature columns against forward returns.

        Args:
            df: Feature DataFrame.
            forward_returns: Forward return series or name of label column in df.

        Returns:
            DataFrame with IC, rank_IC, hit_rate per feature, sorted by abs_rank_ic desc.
        """
        from src.ml.metrics import ic_summary

        if isinstance(forward_returns, str):
            y = df[forward_returns]
        elif isinstance(forward_returns, pd.Series):
            y = forward_returns
        else:
            raise TypeError("forward_returns must be str (column name) or pd.Series")

        feature_cols = [
            c
            for c in df.columns
            if c not in ("forward_return_1d", "forward_return_5d", "forward_return_20d")
            and not c.startswith("forward_return_")
        ]
        X = df[feature_cols]

        return ic_summary(X, y)

    def filter_low_ic_features(
        self,
        df: pd.DataFrame,
        forward_returns: pd.Series | str = "forward_return_5d",
        min_abs_ic: float = DEFAULT_IC_THRESHOLD,
        min_abs_rank_ic: float = DEFAULT_IC_THRESHOLD,
    ) -> pd.DataFrame:
        """Filter feature columns keeping only those with sufficient IC.

        Args:
            df: Feature DataFrame.
            forward_returns: Forward return series or column name.
            min_abs_ic: Minimum absolute Pearson IC.
            min_abs_rank_ic: Minimum absolute Spearman rank IC.

        Returns:
            DataFrame with only high-IC feature columns (plus label column if present).
        """
        from src.ml.metrics import filter_features_by_ic

        if isinstance(forward_returns, str):
            y = df[forward_returns]
        else:
            y = forward_returns

        feature_cols = [c for c in df.columns if not c.startswith("forward_return_")]
        X = df[feature_cols].dropna()

        idx = X.index.intersection(y.dropna().index)
        X_clean = X.loc[idx]
        y_clean = y.loc[idx]

        selected = filter_features_by_ic(X_clean, y_clean, min_abs_ic, min_abs_rank_ic)

        if isinstance(forward_returns, str):
            selected_cols = (
                selected + [forward_returns] if forward_returns in df.columns else selected
            )
        else:
            selected_cols = selected

        return df[selected_cols]

    def _add_alpha_factors(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add formulaic alpha factors (subset of WorldQuant 101).

        Each alpha is a composite formula over price, volume, and returns.
        """
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        open_price = df["Open"]
        volume = df["Volume"]
        returns = close.pct_change()

        ret_5 = returns.rolling(5).sum()
        ret_20 = returns.rolling(20).sum()

        # Alpha 1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
        # Simplified: sign of returns over recent volatility
        vol_20 = returns.rolling(20).std()
        features["alpha_1"] = np.tanh(returns / (vol_20 + 1e-10)).rolling(5).mean()

        # Alpha 2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
        # Simplified: volume-return relationship
        log_vol = np.log(volume + 1e-10)
        features["alpha_2"] = -log_vol.diff(2).rolling(6).corr(returns)

        # Alpha 3: (-1 * correlation(rank(open), rank(volume), 10))
        features["alpha_3"] = -open_price.rolling(10).corr(volume)

        # Alpha 4: (-1 * Ts_Rank(rank(low), 9))
        features["alpha_4"] = -low.rolling(9).apply(
            lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1)
        )

        # Alpha 6: (-1 * correlation(open, volume, 10))
        features["alpha_6"] = -open_price.rolling(10).corr(volume)

        # Alpha 7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
        # Simplified: momentum with volume confirmation
        vol_ma_20 = volume.rolling(20).mean()
        close_delta_7 = close.diff(7)
        sign_mom = np.sign(close_delta_7)
        ts_rank = (
            close_delta_7.abs()
            .rolling(60)
            .apply(lambda x: (x.rank().iloc[-1] - 1) / (max(len(x) - 1, 1)))
        )
        features["alpha_7"] = np.where(volume > vol_ma_20, -ts_rank * sign_mom, -1.0)

        # Alpha 8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
        sum_open_5 = open_price.rolling(5).sum()
        sum_return_5 = returns.rolling(5).sum()
        product = sum_open_5 * sum_return_5
        features["alpha_8"] = -(product - product.shift(10))

        # Alpha 9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
        close_delta = close.diff()
        ts_min_delta = close_delta.rolling(5).min()
        ts_max_delta = close_delta.rolling(5).max()
        features["alpha_9"] = np.where(
            ts_min_delta > 0, close_delta, np.where(ts_max_delta < 0, close_delta, -close_delta)
        )

        # Alpha 10: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
        ts_min_d4 = close_delta.rolling(4).min()
        ts_max_d4 = close_delta.rolling(4).max()
        raw_alpha_10 = np.where(
            ts_min_d4 > 0, close_delta, np.where(ts_max_d4 < 0, close_delta, -close_delta)
        )
        features["alpha_10"] = pd.Series(raw_alpha_10, index=close.index).rolling(20).rank(pct=True)

        # Alpha 12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
        features["alpha_12"] = np.sign(volume.diff()) * (-close.diff())

        # Alpha 20: (-1 * ts_rank(open - delay(high, 1), 15))
        features["alpha_20"] = -(open_price - high.shift(1)).rolling(15).rank(pct=True)

        # Alpha 23: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
        high_ma_20 = high.rolling(20).mean()
        features["alpha_23"] = np.where(high_ma_20 < high, -high.diff(2), 0.0)

        # Alpha 24: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) <= 0.05) |
        #             ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ?
        #            (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))
        ma_100 = close.rolling(100).mean()
        delay_100 = close.shift(100)
        cond_val = ma_100.diff(100) / (delay_100 + 1e-10)
        ts_min_100 = close.rolling(100).min()
        features["alpha_24"] = np.where(
            cond_val.abs() <= 0.05,
            -(close - ts_min_100),
            -close.diff(3),
        )

        # Alpha 32: ((scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))
        # Simplified: mean reversion with correlation
        ma_7 = close.rolling(7).mean()
        features["alpha_32"] = (ma_7 - close) / (close + 1e-10)

        # Alpha 38: ((((-1 * ts_rank(close, 10))) * (delta(close) / delta(close).std())) * correlation((high - low), volume, 10))
        # Simplified: mean-reverting rank with volume-weighted range
        close_rank_10 = close.rolling(10).rank(pct=True)
        close_delta_norm = close_delta / (close_delta.rolling(20).std() + 1e-10)
        hl = high - low
        hl_vol_corr = hl.rolling(10).corr(volume)
        features["alpha_38"] = (-(close_rank_10 - 0.5)) * close_delta_norm * hl_vol_corr

        # Alpha 49: ((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < -1 * 0.1 * volume)
        # Simplified: momentum deceleration indicator
        close_10 = close.shift(10)
        close_20 = close.shift(20)
        mom_decay = ((close_20 - close_10) / 10) - ((close_10 - close) / 10)
        features["alpha_49"] = (mom_decay < -0.1 * volume).astype(float)

        # Alpha 54: (-1 * ((low - close) * power(open, 5)) / ((low - high) * power(close, 5)))
        features["alpha_54"] = (
            -(low - close) * (open_price * 0.01) / ((low - high) * (close * 0.01) + 1e-10)
        )

        # Alpha 101: ((close - open) / ((high - low) + .001))
        features["alpha_101"] = (close - open_price) / (high - low + 0.001)

        features = self._add_alpha_factors_extended(
            features, df, volume, open_price, close, high, low, returns
        )

        return features

    def _add_alpha_factors_extended(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        volume: pd.Series,
        open_price: pd.Series,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        returns: pd.Series,
    ) -> pd.DataFrame:
        """Add extended WorldQuant 101 alpha factors (additional ~35)."""
        adv20 = volume.rolling(20).mean()
        vwap_approx = ((high + low + close) / 3 * volume).rolling(1).sum() / (
            volume.rolling(1).sum() + 1e-10
        )

        # Alpha 5: volume-high rank correlation decay
        vol_rank5 = volume.rolling(5).rank(pct=True)
        high_rank5 = high.rolling(5).rank(pct=True)
        features["alpha_5"] = -vol_rank5.rolling(5).corr(high_rank5).rolling(3).max()

        # Alpha 11: vwap-close spread with volume delta
        vc_diff = vwap_approx - close
        features["alpha_11"] = (
            (vc_diff.rolling(3).max().rank(pct=True) + vc_diff.rolling(3).min().rank(pct=True))
            * volume.diff(3).rank(pct=True)
            * 0.01
        )

        # Alpha 22: high-volume correlation change * close volatility rank
        hv_corr_diff = high.rolling(5).corr(volume).diff(5)
        features["alpha_22"] = -hv_corr_diff * close.rolling(20).std().rank(pct=True)

        # Alpha 25: rank of return * volume * vwap * high-close spread
        features["alpha_25"] = (-returns * adv20 * vwap_approx * (high - close)).rank(pct=True)

        # Alpha 28: adv20-low correlation + midpoint vs close
        features["alpha_28"] = adv20.rolling(5).corr(low) + (high + low) / 2 - close

        # Alpha 30: close position rank diff * sign(close delta)
        c_min5 = (close - close.rolling(5).min()).rank(pct=True)
        c_min15 = (close - close.rolling(15).min()).rank(pct=True)
        features["alpha_30"] = (c_min5 - c_min15) * np.sign(close.diff())

        # Alpha 33: rank of inverse open-to-close ratio
        features["alpha_33"] = (-(1 - open_price / close.replace(0, np.nan))).rank(pct=True)

        # Alpha 34: composite std ratio + close delta rank
        sr2 = returns.rolling(2).std()
        sr5 = returns.rolling(5).std()
        features["alpha_34"] = (
            (1 - (sr2 / (sr5 + 1e-10)).rank(pct=True)) + (1 - close.diff().rank(pct=True))
        ).rank(pct=True)

        # Alpha 41: geometric mean vs vwap
        features["alpha_41"] = np.sqrt(high * low) - vwap_approx

        # Alpha 42: high volatility rank * high-volume correlation
        features["alpha_42"] = -high.rolling(10).std().rank(pct=True) * high.rolling(10).corr(
            volume
        )

        # Alpha 46: volume-weighted mid-price momentum
        mid_price = close * 0.5 + vwap_approx * 0.5
        features["alpha_46"] = ((mid_price.diff(3) / close.shift(3)) - 1) * 20

        # Alpha 48: close-vwap ratio rank * close-volume correlation rank
        cvr = (close - vwap_approx) / close.replace(0, np.nan)
        features["alpha_48"] = -cvr.rank(pct=True) * close.rolling(10).corr(volume).rank(pct=True)

        # Alpha 52: low min rebound * long-short return rank * volume rank
        lmin5 = low.rolling(5).min()
        rs240 = returns.rolling(240).sum()
        rs20 = returns.rolling(20).sum()
        features["alpha_52"] = (
            (-lmin5 + lmin5.shift(5))
            * ((rs240 - rs20) / 220).rank(pct=True)
            * volume.rolling(5).rank(pct=True)
        )

        # Alpha 53: candle efficiency delta
        c_eff = ((close - low) - (high - close)) / (close - low + 1e-10)
        features["alpha_53"] = -c_eff.diff(9)

        # Alpha 55: -corr(close_position_rank, volume_rank, 6)
        rel_pos = (close - low.rolling(12).min()) / (
            high.rolling(12).max() - low.rolling(12).min() + 1e-10
        )
        features["alpha_55"] = -rel_pos.rank(pct=True).rolling(6).corr(volume.rank(pct=True))

        # Alpha 56: inverse close-vwap correlation rank
        cvc10 = close.rolling(10).corr(vwap_approx)
        features["alpha_56"] = 0.001 / (cvc10.rank(pct=True) + 0.001)

        # Alpha 57: mean reversion away from vwap
        decay_r = (
            vwap_approx.diff()
            .rolling(2)
            .apply(lambda x: (x.rank().iloc[-1] - 1) / max(len(x) - 1, 1))
        )
        features["alpha_57"] = -(close - vwap_approx) / (decay_r.abs() + 1e-10) * 0.01

        # Alpha 65: ranked abs close delta * sign
        abs_cd7 = close.diff(7).abs()
        ts_r = abs_cd7.rolling(60).apply(lambda x: (x.rank().iloc[-1] - 1) / max(len(x) - 1, 1))
        features["alpha_65"] = -ts_r * np.sign(close.diff(7))

        # Alpha 68: high-adv15 correlation threshold
        adv15 = volume.rolling(15).mean()
        features["alpha_68"] = np.where(high.rolling(9).corr(adv15) < close.diff(3), -1.0, 1.0)

        # Alpha 83: hl ratio rank composite
        cma5 = close.rolling(5).mean()
        hl_r = (high - low) / (cma5 + 1e-10)
        num = (hl_r.shift(1).shift(1) * (close.shift(1) / close)).rank(pct=True)
        denom = hl_r / (vwap_approx - close).abs().replace(0, 1e-10)
        features["alpha_83"] = (
            num * volume.rank(pct=True).rank(pct=True) / (denom.abs() + 1e-10)
        ).clip(-10, 10)

        # Alpha 85: squared vwap/close ratio rank
        features["alpha_85"] = (
            ((vwap_approx / close.replace(0, np.nan)) ** 2).rolling(10).rank(pct=True)
        )

        # Alpha 88: 20-day price momentum percentage
        features["alpha_88"] = (close - close.shift(20)) / close.shift(20).replace(0, np.nan) * 100

        # Alpha 96: short-term reversal with volume confirmation
        r_pos = returns.clip(lower=0)
        r_neg = returns.clip(upper=0).abs()
        features["alpha_96"] = (
            r_pos.rolling(10).sum()
            / (r_neg.rolling(10).sum() + 1e-10)
            * volume.rolling(10).mean()
            / (volume.rolling(50).mean() + 1e-10)
        )

        # Alpha 98: vwap-adv5 vs open-adv15 correlation rank diff
        adv5s26 = volume.rolling(5).mean().rolling(26).sum()
        features["alpha_98"] = vwap_approx.rolling(5).corr(adv5s26).rank(
            pct=True
        ) - open_price.rank(pct=True).rolling(21).corr(adv15.rank(pct=True)).rank(pct=True)

        # Alpha 99: hl-adv60 vs low-volume correlation threshold
        hl_avg = (high + low) / 2
        adv60 = volume.rolling(60).mean()
        hl_corr = hl_avg.rolling(20).sum().rolling(9).corr(adv60.rolling(20).sum()).rank(pct=True)
        lv_corr = low.rolling(6).corr(volume).rank(pct=True)
        features["alpha_99"] = np.where(hl_corr < lv_corr, -1.0, 1.0)

        # Alpha 100: -high-vol corr rank * close-vol corr rank
        features["alpha_100"] = -high.rolling(5).corr(volume).rank(pct=True) * close.rolling(
            5
        ).corr(volume).rank(pct=True)

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
