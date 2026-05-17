"""
Q8: Order Book Dynamics Features — Bid-Ask Imbalance & Microstructure Signals.

Leading indicators for short-term directional moves derived from order
book dynamics. Extends the existing `order_flow.py` golden-ratio alpha
with additional microstructure features.

Features:
  Bid-Ask Imbalance (BAI):
    BAI = (BidVol - AskVol) / (BidVol + AskVol)
    Range [-1, 1]. Positive → buying pressure, negative → selling pressure.

  Queue Position:
    Relative position in the order queue. Closer to front → more likely to fill.

  Order Flow Toxicity (VPIN):
    Volume-synchronized probability of informed trading.
    High VPIN → toxic order flow, adverse selection risk.

  Trade Size Imbalance:
    Large trades vs small trades. Large buy trades → institutional accumulation.

Usage:
    >>> from src.signals.order_book_features import OrderBookFeatures
    >>> ob = OrderBookFeatures()
    >>> features = ob.compute(trades_df, quotes_df)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VPIN_BUCKETS = 50
VPIN_THRESHOLD = 0.80


class OrderBookFeatures:
    """Order book dynamics feature generator.

    Computes microstructure features from trade and quote data.
    Designed to work with OHLCV (proxy) or real tick data.

    Args:
        vpin_buckets: Number of volume buckets for VPIN.
        volume_window: Rolling window for volume-based features.
    """

    def __init__(self, vpin_buckets: int = VPIN_BUCKETS, volume_window: int = 20):
        self.vpin_buckets = vpin_buckets
        self.volume_window = volume_window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute order book features from OHLCV data.

        Uses OHLCV as proxy for order book dynamics:
        - High-Low range → intraday volatility (spread proxy)
        - Close relative to range → closing pressure (queue proxy)
        - Volume surges → informed trading (toxicity proxy)

        Args:
            df: DataFrame with Open, High, Low, Close, Volume.

        Returns:
            DataFrame of order book features with same index.
        """
        features = pd.DataFrame(index=df.index)

        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        close = df["Close"].astype(float)
        volume = (
            df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(1, index=df.index)
        )

        # Bid-ask imbalance proxy: close position within day's range
        day_range = high - low
        close_position = (close - low) / day_range.replace(0, np.nan)
        features["bai_proxy"] = 2 * close_position - 1

        # Spread proxy: normalized high-low range
        features["spread_proxy"] = day_range / close

        # Volume-synchronized probability of informed trading (VPIN proxy)
        features["vpin_proxy"] = self._vpin_proxy(volume)

        # Trade size imbalance proxy: volume surge vs recent average
        vol_ma = volume.rolling(self.volume_window).mean()
        features["volume_imbalance"] = volume / vol_ma.replace(0, np.nan) - 1

        # Order flow momentum: consecutive bars with same directional close
        up_bars = (close > close.shift(1)).astype(int)
        down_bars = (close < close.shift(1)).astype(int)
        features["up_streak"] = up_bars.groupby((up_bars != up_bars.shift(1)).cumsum()).cumsum()
        features["down_streak"] = down_bars.groupby(
            (down_bars != down_bars.shift(1)).cumsum()
        ).cumsum()

        # Absorption ratio: volume required to move price by 1 ATR
        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
        ).max(axis=1)
        atr = tr.rolling(14).mean()
        price_move = close.diff().abs()
        features["absorption"] = volume / (price_move / atr.replace(1e-10, np.nan)).replace(
            np.inf, np.nan
        )

        return features

    def _vpin_proxy(self, volume: pd.Series) -> pd.Series:
        """VPIN proxy: rolling volume concentration.

        High VPIN → informed trading, adverse selection risk.
        Uses volume skew as proxy for trade clustering.
        """
        vol_log = np.log(volume.replace(0, np.nan) + 1)
        vol_ma = vol_log.rolling(self.volume_window).mean()
        vol_std = vol_log.rolling(self.volume_window).std()
        z_vol = (vol_log - vol_ma) / vol_std.replace(0, np.nan)
        vpin = z_vol.rolling(self.volume_window).apply(lambda x: (np.abs(x) > 1.5).mean(), raw=True)
        return vpin.fillna(0.0)

    def is_toxic(self, df: pd.DataFrame) -> pd.Series:
        """Identify periods of toxic order flow.

        Returns:
            Boolean Series indicating toxic bars.
        """
        features = self.compute(df)
        return features.get("vpin_proxy", pd.Series(False, index=df.index)) > VPIN_THRESHOLD

    def on_balance_volume(self, df: pd.DataFrame) -> pd.Series:
        """Compute On-Balance Volume (OBV) as order flow proxy.

        OBV(t) = OBV(t-1) + sign(Close - PrevClose) * Volume
        """
        close = df["Close"].astype(float)
        volume = df.get("Volume", pd.Series(1, index=df.index))
        direction = np.sign(close.diff().fillna(0))
        obv = (direction * volume).cumsum()
        obv.name = "obv"
        return obv


class OrderBookSignal:
    """Order book dynamics → trading signal [-1, 1].

    Combines multiple microstructure features into a directional signal.
    Beta feature — designed for blend with existing pattern-based signals.

    Args:
        bai_weight: Bid-ask imbalance weight.
        vpin_weight: VPIN (inverse) weight.
        volume_weight: Volume imbalance weight.
    """

    def __init__(
        self,
        bai_weight: float = 0.40,
        vpin_weight: float = 0.30,
        volume_weight: float = 0.30,
    ):
        self.bai_weight = bai_weight
        self.vpin_weight = vpin_weight
        self.volume_weight = volume_weight
        self._features = OrderBookFeatures()

    def compute_signal(self, df: pd.DataFrame) -> pd.Series:
        """Compute composite order book signal.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Series of signals in [-1, 1].
        """
        features = self._features.compute(df)

        bai = features.get("bai_proxy", pd.Series(0, index=df.index)).fillna(0).clip(-1, 1)
        vpin = features.get("vpin_proxy", pd.Series(0, index=df.index)).fillna(0)
        vpin_signal = -np.tanh(vpin * 3)
        vol_imb = features.get("volume_imbalance", pd.Series(0, index=df.index)).fillna(0)
        vol_signal = np.tanh(vol_imb)

        signal = (
            self.bai_weight * bai + self.vpin_weight * vpin_signal + self.volume_weight * vol_signal
        )
        return signal.clip(-1, 1)
