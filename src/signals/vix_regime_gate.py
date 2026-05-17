"""
Q1: VIX Regime Gate — VIX Term Structure Regime Detection for Signal Gating.

Classifies market environment based on VIX level and term structure slope,
producing a regime multiplier that can gate or scale trade signals.

Regimes:
  COMPLACENT (VIX < 15): risk-on, 1.0x signal multiplier
  NORMAL (15 <= VIX < 25): neutral, 1.0x multiplier
  ELEVATED (25 <= VIX < 30): caution, 0.75x multiplier
  STRESS (VIX >= 30): risk-off, 0.30x multiplier

VIX slope (short-term vs long-term VIX MA) provides additional refinement:
  - Contango (slope > 0): future vol higher, slight positive bias
  - Backwardation (slope < -0.05): future vol lower, stress signal

Integration with RulesFirstStrategy:
  score *= gate.multiplier(vix_level)

Data: ^VIX spot via yfinance (free, no API key).

Usage:
    >>> gate = VixRegimeGate()
    >>> gate.fit(start="2016-01-01")
    >>> mult = gate.multiplier("2025-03-15")  # returns 0.75 for elevated VIX
    >>> gate.is_stress("2008-10-15")  # True for VIX > 30
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VIX_STRESS_THRESHOLD = 30.0
VIX_ELEVATED_THRESHOLD = 25.0
VIX_NORMAL_THRESHOLD = 15.0
SLOPE_SHORT_WINDOW = 5
SLOPE_LONG_WINDOW = 20
BACKWARDATION_THRESHOLD = -0.05
CONTANGO_THRESHOLD = 0.0


class VixRegimeGate:
    """VIX-based regime gate for trade signal gating.

    Fetches VIX data via yfinance, computes regime classification and
    signal multipliers. VIX > 30 is a universal stress indicator.

    Args:
        stress_mult: Signal multiplier for STRESS regime (default 0.30).
        elevated_mult: Signal multiplier for ELEVATED regime (default 0.75).
        vix_ticker: Yahoo Finance ticker for VIX (default "^VIX").
    """

    def __init__(
        self,
        stress_mult: float = 0.30,
        elevated_mult: float = 0.75,
        vix_ticker: str = "^VIX",
    ):
        self.stress_mult = stress_mult
        self.elevated_mult = elevated_mult
        self.vix_ticker = vix_ticker
        self._vix_data: Optional[pd.DataFrame] = None
        self._vix_slope: Optional[pd.Series] = None
        self._regime: Optional[pd.Series] = None

    def fit(self, start: str = "2015-01-01", end: Optional[str] = None) -> VixRegimeGate:
        """Fetch VIX data and compute regime classifications.

        Args:
            start: Start date for VIX data.
            end: End date (defaults to today).

        Returns:
            Self for chaining.
        """
        import yfinance as yf

        df = yf.download(self.vix_ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            raise RuntimeError(f"No VIX data from {start} to {end}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close_col = "Close" if "Close" in df.columns else df.columns[0]
        self._vix_data = df[[close_col]].rename(columns={close_col: "vix"}).copy()

        vix = self._vix_data["vix"]

        # VIX slope: (short VIX MA / long VIX MA - 1) as proxy for term structure
        vix_ma_short = vix.rolling(SLOPE_SHORT_WINDOW).mean()
        vix_ma_long = vix.rolling(SLOPE_LONG_WINDOW).mean()
        self._vix_slope = (vix_ma_short / vix_ma_long - 1).rename("vix_slope")

        # Regime classification
        regimes = []
        for i in range(len(vix)):
            vix_val = float(vix.iloc[i])
            slope_val = float(self._vix_slope.iloc[i]) if i >= SLOPE_LONG_WINDOW else 0.0

            if np.isnan(vix_val):
                regimes.append("UNKNOWN")
            elif vix_val >= VIX_STRESS_THRESHOLD:
                regimes.append("STRESS")
            elif vix_val >= VIX_ELEVATED_THRESHOLD:
                if slope_val < BACKWARDATION_THRESHOLD:
                    regimes.append("STRESS")
                else:
                    regimes.append("ELEVATED")
            elif vix_val >= VIX_NORMAL_THRESHOLD:
                regimes.append("NORMAL")
            else:
                regimes.append("COMPLACENT")

        self._regime = pd.Series(regimes, index=vix.index, name="vix_regime")
        self._vix_data["regime"] = self._regime
        self._vix_data["vix_slope"] = self._vix_slope

        logger.info(
            "VIX regime gate fitted: %d bars, regimes=%s",
            len(self._regime),
            self._regime.value_counts().to_dict(),
        )
        return self

    def multiplier(self, date=None, vix_level: Optional[float] = None) -> float:
        """Get signal multiplier for a given date or VIX level.

        Args:
            date: Date or index label (uses last available if None).
            vix_level: Direct VIX level (overrides date lookup).

        Returns:
            Multiplier in [0.30, 1.0].
        """
        if vix_level is not None:
            return self._multiplier_from_level(vix_level)

        if self._vix_data is None:
            logger.warning("VIX gate not fitted, returning 1.0")
            return 1.0

        if date is not None:
            if date in self._vix_data.index:
                vix = float(self._vix_data.loc[date, "vix"])
                return self._multiplier_from_level(vix)
            idx = self._vix_data.index
            target = pd.Timestamp(date)
            nearby = idx[idx <= target]
            if len(nearby) > 0:
                vix = float(self._vix_data.loc[nearby[-1], "vix"])
                return self._multiplier_from_level(vix)

        vix = float(self._vix_data["vix"].iloc[-1])
        return self._multiplier_from_level(vix)

    def _multiplier_from_level(self, vix: float) -> float:
        if vix >= VIX_STRESS_THRESHOLD:
            return self.stress_mult
        elif vix >= VIX_ELEVATED_THRESHOLD:
            return self.elevated_mult
        return 1.0

    def regime(self, date=None, vix_level: Optional[float] = None) -> str:
        """Get regime label for a given date or VIX level.

        Args:
            date: Date to look up.
            vix_level: Direct VIX level (overrides date).

        Returns:
            One of: COMPLACENT, NORMAL, ELEVATED, STRESS, UNKNOWN.
        """
        if vix_level is not None:
            return self._regime_from_level(vix_level)

        if self._regime is None:
            return "UNKNOWN"

        if date is not None:
            if date in self._regime.index:
                return str(self._regime[date])
            idx = self._regime.index
            target = pd.Timestamp(date)
            nearby = idx[idx <= target]
            if len(nearby) > 0:
                return str(self._regime[nearby[-1]])

        return str(self._regime.iloc[-1])

    def _regime_from_level(self, vix: float) -> str:
        if vix >= VIX_STRESS_THRESHOLD:
            return "STRESS"
        elif vix >= VIX_ELEVATED_THRESHOLD:
            return "ELEVATED"
        elif vix >= VIX_NORMAL_THRESHOLD:
            return "NORMAL"
        return "COMPLACENT"

    def is_stress(self, date=None) -> bool:
        """Check if market is in stress regime."""
        return self.regime(date=date) == "STRESS"

    def is_elevated(self, date=None) -> bool:
        """Check if market is in elevated regime."""
        return self.regime(date=date) in ("ELEVATED", "STRESS")

    @property
    def vix_data(self) -> Optional[pd.DataFrame]:
        return self._vix_data

    @property
    def regime_series(self) -> Optional[pd.Series]:
        return self._regime

    def to_dataframe(self) -> pd.DataFrame:
        """Return DataFrame with VIX, regime, and slope columns."""
        if self._vix_data is None:
            return pd.DataFrame()
        return self._vix_data.copy()
