"""
B10: Real Yield Analysis (TIPS) — Inflation-Adjusted Yield for Macro Regime.

Computes real yields from Treasury Inflation-Protected Securities (TIPS)
and nominal Treasury yields. The breakeven inflation rate (nominal - real)
is a powerful macro regime indicator.

Key signals:
- Real yield (TIPS yield): True risk-free rate after inflation.
- Breakeven inflation: Market-implied inflation expectations.
- Real yield curve slope: Growth expectations (steep = expansion).
- Real yield vs nominal regime classification.

Reference: Campbell & Shiller (1996), Fed TIPS/t-note data.

Usage:
    >>> ry = RealYieldAnalyzer()
    >>> ry.real_yield(nominal=0.045, inflation=0.025)
    >>> features = ry.real_yield_features(nominal_df, tips_df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard TIPS maturities and their nominal counterparts
TIPS_MATURITIES = {
    "5Y": {"tips": "DFII5", "nominal": "DGS5"},
    "10Y": {"tips": "DFII10", "nominal": "DGS10"},
    "20Y": {"tips": "DFII20", "nominal": None},  # No exact nominal match
    "30Y": {"tips": "DFII30", "nominal": "DGS30"},
}


@dataclass
class RealYieldSnapshot:
    date: date
    nominal_5y: float
    nominal_10y: float
    nominal_30y: float
    real_5y: float
    real_10y: float
    real_30y: float

    @property
    def be_5y(self) -> float:
        return self.nominal_5y - self.real_5y

    @property
    def be_10y(self) -> float:
        return self.nominal_10y - self.real_10y

    @property
    def be_30y(self) -> float:
        return self.nominal_30y - self.real_30y

    @property
    def real_curve_slope(self) -> float:
        return self.real_30y - self.real_5y

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "nominal_5y": round(self.nominal_5y, 4),
            "nominal_10y": round(self.nominal_10y, 4),
            "nominal_30y": round(self.nominal_30y, 4),
            "real_5y": round(self.real_5y, 4),
            "real_10y": round(self.real_10y, 4),
            "real_30y": round(self.real_30y, 4),
            "be_5y": round(self.be_5y, 4),
            "be_10y": round(self.be_10y, 4),
            "be_30y": round(self.be_30y, 4),
            "real_curve_slope": round(self.real_curve_slope, 4),
        }


class RealYieldAnalyzer:
    """Real yield and breakeven inflation analysis for macro regime detection.

    Models the relationship between nominal yields, real yields, and
    market-implied inflation expectations. Produces features for regime
    classification and allocation decisions.

    Args:
        rfr: Default risk-free rate proxy if external data unavailable.
    """

    def __init__(self, rfr: float = 0.05):
        self.rfr = rfr

    @staticmethod
    def real_yield(nominal: float, inflation: float) -> float:
        r"""Fisher equation: real rate = (1+nominal)/(1+inflation) - 1.

        For small rates, this approximates nominal - inflation.

        Args:
            nominal: Nominal interest rate (decimal).
            inflation: Expected inflation rate (decimal).

        Returns:
            Real yield (decimal).
        """
        return (1.0 + nominal) / (1.0 + inflation) - 1.0

    @staticmethod
    def breakeven_inflation(nominal: float, real: float) -> float:
        """Market-implied inflation expectation from TIPS spread.

        Args:
            nominal: Nominal Treasury yield.
            real: TIPS real yield.

        Returns:
            Breakeven inflation rate.
        """
        return ((1.0 + nominal) / (1.0 + real)) - 1.0

    def classify_real_yield_regime(self, real_10y: float, be_10y: float) -> str:
        """Classify macro regime from real yield and breakeven inflation.

        Args:
            real_10y: 10-year real yield.
            be_10y: 10-year breakeven inflation.

        Returns:
            Regime label.
        """
        if real_10y > 0.02 and be_10y > 0.025:
            return "growth_inflation"  # Strong growth, rising inflation
        elif real_10y > 0.02 and be_10y <= 0.025:
            return "growth_disflation"  # Strong growth, contained inflation
        elif real_10y <= 0.02 and be_10y > 0.025:
            return "stagflation_risk"  # Weak growth, rising inflation
        else:
            return "disinflation_slowdown"  # Weak growth, low inflation

    def real_yield_features(
        self,
        nominal_df: pd.DataFrame,
        tips_df: Optional[pd.DataFrame] = None,
        date_col: str = "Date",
    ) -> pd.DataFrame:
        """Generate real yield features from nominal and TIPS data.

        If TIPS data is unavailable, estimates real yields using a
        constant inflation proxy (CPI YoY approximation).

        Args:
            nominal_df: DataFrame with nominal yield columns (DGS5, DGS10, DGS30).
            tips_df: Optional DataFrame with TIPS yield columns.
            date_col: Date column name.

        Returns:
            DataFrame with real yield and breakeven inflation features.
        """
        df = nominal_df.copy()
        df["_date"] = pd.to_datetime(df[date_col])

        # Nominal yields (map to standard names)
        nom_10y = df.get("DGS10", df.get("nominal_10y", None))
        nom_5y = df.get("DGS5", df.get("nominal_5y", None))
        nom_30y = df.get("DGS30", df.get("nominal_30y", None))

        if tips_df is not None:
            tips_df = tips_df.copy()
            tips_df["_date"] = pd.to_datetime(tips_df[date_col])
            merged = df.merge(tips_df, on="_date", how="left", suffixes=("", "_tips"))
            real_5y = merged.get("DFII5", merged.get("real_5y", None))
            real_10y = merged.get("DFII10", merged.get("real_10y", None))
            real_30y = merged.get("DFII30", merged.get("real_30y", None))
        else:
            real_5y = real_10y = real_30y = None

        features = []

        for i in range(len(df)):
            row = {}
            row["date"] = df["_date"].iloc[i]

            n5 = (
                float(nom_5y.iloc[i])
                if nom_5y is not None and not pd.isna(nom_5y.iloc[i])
                else np.nan
            )
            n10 = (
                float(nom_10y.iloc[i])
                if nom_10y is not None and not pd.isna(nom_10y.iloc[i])
                else np.nan
            )
            n30 = (
                float(nom_30y.iloc[i])
                if nom_30y is not None and not pd.isna(nom_30y.iloc[i])
                else np.nan
            )

            if not np.isnan(n10):
                n5 = n10 - 0.005 if np.isnan(n5) else n5
                n30 = n10 + 0.003 if np.isnan(n30) else n30
            else:
                n5 = self.rfr - 0.005
                n10 = self.rfr
                n30 = self.rfr + 0.005

            if real_10y is not None and not pd.isna(real_10y.iloc[i]):
                r10 = float(real_10y.iloc[i])
                r5 = float(real_5y.iloc[i]) if real_5y is not None else r10 - 0.002
                r30 = float(real_30y.iloc[i]) if real_30y is not None else r10 + 0.002
            else:
                # Estimate real = nominal - 2.5% inflation proxy
                r5 = n5 - 0.025
                r10 = n10 - 0.025
                r30 = n30 - 0.025

            be5 = self.breakeven_inflation(n5, r5)
            be10 = self.breakeven_inflation(n10, r10)
            be30 = self.breakeven_inflation(n30, r30)

            row["real_yield_5y"] = r5
            row["real_yield_10y"] = r10
            row["real_yield_30y"] = r30
            row["be_inflation_5y"] = be5
            row["be_inflation_10y"] = be10
            row["be_inflation_30y"] = be30
            row["real_curve_slope"] = r30 - r5
            row["be_curve_slope"] = be30 - be5
            row["real_regime"] = self.classify_real_yield_regime(r10, be10)

            features.append(row)

        return pd.DataFrame(features)

    def regime_signal(self, real_10y: float, be_10y: float) -> float:
        """Trading signal from real yield regime.

        High real yields + moderate inflation = risk-on (bullish equities).
        0-2% real yields = neutral.
        Negative real yields or >3% inflation = risk-off.

        Returns signal in [-1, 1].
        """
        regime = self.classify_real_yield_regime(real_10y, be_10y)

        if regime == "growth_disflation":
            return 0.7
        elif regime == "growth_inflation":
            return 0.3
        elif regime == "disinflation_slowdown":
            return -0.3
        elif regime == "stagflation_risk":
            return -0.7
        return 0.0

    def snapshots_from_df(
        self,
        df: pd.DataFrame,
        date_col: str = "Date",
        nominal_10y_col: str = "DGS10",
        nominal_5y_col: str = "DGS5",
        nominal_30y_col: str = "DGS30",
        real_10y_col: str = "DFII10",
        real_5y_col: str = "DFII5",
        real_30y_col: str = "DFII30",
    ) -> List[RealYieldSnapshot]:
        """Extract RealYieldSnapshot list from a combined DataFrame.

        Args:
            df: DataFrame with nominal and real yield columns.
            date_col, nominal_*_col, real_*_col: Column names.

        Returns:
            List of RealYieldSnapshot objects.
        """
        from datetime import date as dt_date

        snapshots: List[RealYieldSnapshot] = []
        for i in range(len(df)):
            d = df[date_col].iloc[i]
            if isinstance(d, pd.Timestamp):
                d = d.date()
            snapshots.append(
                RealYieldSnapshot(
                    date=d,
                    nominal_5y=float(df[nominal_5y_col].iloc[i]),
                    nominal_10y=float(df[nominal_10y_col].iloc[i]),
                    nominal_30y=float(df[nominal_30y_col].iloc[i]),
                    real_5y=float(df[real_5y_col].iloc[i]),
                    real_10y=float(df[real_10y_col].iloc[i]),
                    real_30y=float(df[real_30y_col].iloc[i]),
                )
            )
        return snapshots
