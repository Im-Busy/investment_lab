"""
Simple rule-based regime detectors implementing RegimeDetectorBase.

Fallback from unsupervised detectors (HMM/GMM/PCAKMeans) which produce
noisy 10-12 bar average regimes on 132-dim feature space.

Provides two regime axes:
  1. TrendRegime: Bull (Close > SMA200) / Bear (Close < SMA200)
  2. VolRegime: HighVol (ATR14 > 80th pctile) / LowVol (else)

These are simple, interpretable, and historically validated (200MA is the
most common institutional trend filter).

Usage:
    from src.ml.simple_regime import TrendRegimeDetector, VolRegimeDetector

    trend = TrendRegimeDetector(ma_period=200)
    trend.fit(data)  # computes SMA
    labels = trend.predict(data)

    vol = VolRegimeDetector(atr_period=14, vol_pctile=80)
    vol.fit(data)
    labels = vol.predict(data)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.ml.regime_base import RegimeDetectorBase, RegimeSummary


class SimpleTrendRegimeDetector(RegimeDetectorBase):
    """Binary trend regime: Bull (Close > SMA) or Bear (Close < SMA).

    Args:
        ma_period: Moving average window (default 200).
    """

    def __init__(self, ma_period: int = 200, min_samples: int = 250):
        super().__init__(name="SimpleTrendRegimeDetector")
        self.ma_period = ma_period
        self.min_samples = min_samples
        self._sma: Optional[pd.Series] = None
        self._fitted_close_idx: Optional[pd.DatetimeIndex] = None

    def fit(self, data: pd.DataFrame) -> SimpleTrendRegimeDetector:
        self._validate_features(data, require_fitted=False)
        self._sma = data["Close"].rolling(self.ma_period).mean()
        self._fitted_close_idx = data.index
        self.is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")

        if "Close" in data.columns:
            price = data["Close"]
        else:
            # Fallback: use any price-like column
            close_cols = [c for c in data.columns if c.lower() == "close"]
            if close_cols:
                price = data[close_cols[0]]
            else:
                price = data.iloc[:, 0]

        sma = price.rolling(self.ma_period).mean()
        regime = pd.Series("Bear", index=data.index, name="trend_regime")
        regime[price > sma] = "Bull"
        regime[sma.isna()] = "Unknown"
        return regime

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        labels = self.predict(data)
        n = len(labels)
        probs = pd.DataFrame(0.0, index=labels.index, columns=["Bull", "Bear", "Unknown"])
        for regime in ["Bull", "Bear", "Unknown"]:
            probs.loc[labels == regime, regime] = 1.0
        return probs

    def get_regime_summary(self) -> RegimeSummary:
        if not self.is_fitted:
            raise ValueError("Detector not fitted.")
        return RegimeSummary(
            name=self.name,
            n_regimes=2,
            regime_labels=["Bull", "Bear"],
            label_distribution={"Bull": 0, "Bear": 0},
            label_proportions={"Bull": 0.5, "Bear": 0.5},
            metadata={"ma_period": self.ma_period},
        )


class SimpleVolRegimeDetector(RegimeDetectorBase):
    """Binary volatility regime: HighVol (ATR > pctile) or LowVol (else).

    Args:
        atr_period: ATR window (default 14).
        vol_pctile: Percentile threshold for high volatility (default 80).
    """

    def __init__(self, atr_period: int = 14, vol_pctile: float = 80.0, min_samples: int = 250):
        super().__init__(name="SimpleVolRegimeDetector")
        self.atr_period = atr_period
        self.vol_pctile = vol_pctile
        self.min_samples = min_samples
        self._threshold: float = 0.0
        self._fitted_idx: Optional[pd.DatetimeIndex] = None

    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        high, low, close = df["High"], df["Low"], df["Close"]
        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        return tr.rolling(self.atr_period).mean()

    def fit(self, data: pd.DataFrame) -> SimpleVolRegimeDetector:
        self._validate_features(data, require_fitted=False)
        atr = self._compute_atr(data)
        self._threshold = float(np.percentile(atr.dropna(), self.vol_pctile))
        self._fitted_idx = data.index
        self.is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        atr = self._compute_atr(data)
        regime = pd.Series("LowVol", index=data.index, name="vol_regime")
        regime[atr > self._threshold] = "HighVol"
        regime[atr.isna()] = "Unknown"
        return regime

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        labels = self.predict(data)
        probs = pd.DataFrame(0.0, index=labels.index, columns=["HighVol", "LowVol", "Unknown"])
        for regime in ["HighVol", "LowVol", "Unknown"]:
            probs.loc[labels == regime, regime] = 1.0
        return probs

    def get_regime_summary(self) -> RegimeSummary:
        if not self.is_fitted:
            raise ValueError("Detector not fitted.")
        return RegimeSummary(
            name=self.name,
            n_regimes=2,
            regime_labels=["HighVol", "LowVol"],
            label_distribution={"HighVol": 0, "LowVol": 0},
            label_proportions={"HighVol": 0.2, "LowVol": 0.8},
            metadata={
                "atr_period": self.atr_period,
                "vol_pctile": self.vol_pctile,
                "atr_threshold": round(self._threshold, 4),
            },
        )


class CombinedSimpleRegimeDetector(RegimeDetectorBase):
    """Combine TrendRegime + VolRegime into a single multi-class detector.

    Produces 4 combined regime labels:
        Bull_LowVol, Bull_HighVol, Bear_LowVol, Bear_HighVol

    This is the simplest actionable regime split — enough for per-regime
    model training without the noise of unsupervised clustering.
    """

    def __init__(
        self,
        ma_period: int = 200,
        atr_period: int = 14,
        vol_pctile: float = 80.0,
        min_samples: int = 250,
    ):
        super().__init__(name="CombinedSimpleRegimeDetector")
        self.trend_detector = SimpleTrendRegimeDetector(
            ma_period=ma_period, min_samples=min_samples
        )
        self.vol_detector = SimpleVolRegimeDetector(
            atr_period=atr_period, vol_pctile=vol_pctile, min_samples=min_samples
        )

    def fit(self, data: pd.DataFrame) -> CombinedSimpleRegimeDetector:
        self.trend_detector.fit(data)
        self.vol_detector.fit(data)
        self.is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        trend = self.trend_detector.predict(data)
        vol = self.vol_detector.predict(data)
        return (trend.astype(str) + "_" + vol.astype(str).str.replace("LowVol", "Low")).str.replace(
            "HighVol", "High"
        )

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        trend_probs = self.trend_detector.predict_proba(data)
        vol_probs = self.vol_detector.predict_proba(data)
        labels = self.predict(data)
        probs = pd.DataFrame(0.0, index=labels.index, columns=list(labels.unique()))
        for col in probs.columns:
            parts = col.split("_")
            trend_label = parts[0]
            vol_label = "HighVol" if "High" in col else "LowVol"
            probs[col] = trend_probs.get(trend_label, 0.0) * vol_probs.get(vol_label, 0.0)
        return probs.div(probs.sum(axis=1), axis=0).fillna(0.0)

    def get_regime_summary(self) -> RegimeSummary:
        if not self.is_fitted:
            raise ValueError("Detector not fitted.")
        return RegimeSummary(
            name=self.name,
            n_regimes=4,
            regime_labels=[
                "Bull_Low",
                "Bull_High",
                "Bear_Low",
                "Bear_High",
            ],
            label_distribution={},
            label_proportions={},
            metadata={"type": "combined_trend_vol"},
        )
