"""
Regime Detector - ADX/ATR-based Market Regime Classification

Classifies market regimes into:
- Trending: ADX(14) > 25 → strong trend
- Ranging: ADX(14) < 20 AND ATR(14) < median ATR → low volatility, no trend
- Volatile: ATR(14) > 80th percentile of trailing ATR → high volatility
- Transition: ADX(14) between 20-25 → regime ambiguity, default to previous

Usage:
    from src.indicators.regime_detector import RegimeDetector, RegimeState

    detector = RegimeDetector()
    regimes = detector.classify(data)  # returns array of RegimeState
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class RegimeState(Enum):
    """Market regime states."""

    TRENDING = "Trending"
    RANGING = "Ranging"
    VOLATILE = "Volatile"
    TRANSITION = "Transition"


@dataclass
class RegimeDetectorConfig:
    """Configuration for regime detection."""

    # ADX thresholds
    adx_period: int = 14
    adx_trending_threshold: float = 25.0
    adx_ranging_threshold: float = 20.0

    # ATR thresholds
    atr_period: int = 14
    atr_volatile_percentile: float = 0.80  # 80th percentile
    atr_median_period: int = 100  # Bars for median ATR calculation

    # Minimum bars needed for classification
    warmup_bars: int = 100


class RegimeDetector:
    """
    Market regime classifier using ADX and ATR.

    Detection rules:
    - TRENDING: ADX(14) > 25
    - RANGING: ADX(14) < 20 AND ATR(14) < median(trailing ATR)
    - VOLATILE: ATR(14) > 80th percentile of trailing 100-bar ATR
    - TRANSITION: ADX(14) between 20-25 (use previous regime)
    """

    def __init__(self, config: Optional[RegimeDetectorConfig] = None):
        self.config = config or RegimeDetectorConfig()

    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate ADX(14) using pandas-ta or manual calculation."""
        try:
            import pandas_ta as ta

            adx = ta.adx(high, low, close, length=self.config.adx_period)
            return adx[f"ADX_{self.config.adx_period}"]
        except ImportError:
            return self._calculate_adx_manual(high, low, close)

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate ATR(14)."""
        try:
            import pandas_ta as ta

            atr = ta.atr(high, low, close, length=self.config.atr_period)
            return atr
        except ImportError:
            return self._calculate_atr_manual(high, low, close)

    def _calculate_adx_manual(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Manual ADX calculation."""
        period = self.config.adx_period

        # True Range
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed values
        atr = pd.Series(0.0, index=close.index)
        plus_di = pd.Series(0.0, index=close.index)
        minus_di = pd.Series(0.0, index=close.index)

        # First value
        atr.iloc[period - 1] = tr.iloc[:period].sum()
        plus_di.iloc[period - 1] = np.sum(plus_dm[:period])
        minus_di.iloc[period - 1] = np.sum(minus_dm[:period])

        # Subsequent values (Wilders smoothing)
        for i in range(period, len(close)):
            atr.iloc[i] = atr.iloc[i - 1] - atr.iloc[i - 1] / period + tr.iloc[i]
            plus_di.iloc[i] = plus_di.iloc[i - 1] - plus_di.iloc[i - 1] / period + plus_dm[i]
            minus_di.iloc[i] = minus_di.iloc[i - 1] - minus_di.iloc[i - 1] / period + minus_dm[i]

        # Calculate +DI and -DI
        plus_di = plus_di / atr * 100
        minus_di = minus_di / atr * 100

        # Calculate DX
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
        dx = dx.replace([np.inf, -np.inf], 0).fillna(0)

        # Calculate ADX (smoothed DX)
        adx = pd.Series(0.0, index=close.index)
        for i in range(2 * period - 1, len(close)):
            if i == 2 * period - 1:
                adx.iloc[i] = dx.iloc[period : i + 1].mean()
            else:
                adx.iloc[i] = (adx.iloc[i - 1] * (period - 1) + dx.iloc[i]) / period

        return adx

    def _calculate_atr_manual(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Manual ATR calculation."""
        period = self.config.atr_period

        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)

        atr = pd.Series(0.0, index=close.index)
        atr.iloc[period - 1] = tr.iloc[:period].sum()

        for i in range(period, len(close)):
            atr.iloc[i] = atr.iloc[i - 1] - atr.iloc[i - 1] / period + tr.iloc[i]

        return atr

    def classify(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:
        """
        Classify market regime for each bar.

        Args:
            data: OHLCV DataFrame with 'High', 'Low', 'Close' columns

        Returns:
            Series of RegimeState values
        """
        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        adx = self._calculate_adx(high, low, close)
        atr = self._calculate_atr(high, low, close)

        # Calculate thresholds
        atr_median = atr.rolling(self.config.atr_median_period, min_periods=20).median()
        atr_80th = atr.rolling(self.config.atr_median_period, min_periods=20).quantile(
            self.config.atr_volatile_percentile
        )

        # Generate regime labels
        regimes = []
        prev_regime = RegimeState.TRANSITION
        warmup = max(self.config.adx_period * 2, self.config.atr_median_period)

        for i in range(len(close)):
            if i < warmup:
                regimes.append(RegimeState.TRANSITION)
                continue

            current_adx = adx.iloc[i] if not pd.isna(adx.iloc[i]) else 0
            current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0
            current_atr_median = (
                atr_median.iloc[i] if not pd.isna(atr_median.iloc[i]) else current_atr
            )
            current_atr_80th = (
                atr_80th.iloc[i] if not pd.isna(atr_80th.iloc[i]) else current_atr * 1.2
            )

            # Determine regime
            if current_adx > self.config.adx_trending_threshold:
                regime = RegimeState.TRENDING
            elif current_adx < self.config.adx_ranging_threshold:
                if current_atr < current_atr_median:
                    regime = RegimeState.RANGING
                else:
                    regime = RegimeState.TRANSITION
            elif current_atr > current_atr_80th:
                regime = RegimeState.VOLATILE
            else:
                regime = RegimeState.TRANSITION

            # If transition, use previous regime
            if regime == RegimeState.TRANSITION:
                regime = prev_regime

            regimes.append(regime)
            prev_regime = regime

        return pd.Series(regimes, index=close.index, name="regime")

    def get_regime_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get regime classification with supporting indicators.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with regime, ADX, ATR, and threshold columns
        """
        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        adx = self._calculate_adx(high, low, close)
        atr = self._calculate_atr(high, low, close)
        atr_median = atr.rolling(self.config.atr_median_period, min_periods=20).median()
        atr_80th = atr.rolling(self.config.atr_median_period, min_periods=20).quantile(
            self.config.atr_volatile_percentile
        )

        return pd.DataFrame(
            {
                "regime": self.classify(data),
                "adx": adx,
                "atr": atr,
                "atr_median": atr_median,
                "atr_80th_percentile": atr_80th,
            },
            index=data.index,
        )
