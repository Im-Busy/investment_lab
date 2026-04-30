"""
Macroeconomic Regime Detection

Regime detection using FRED-MD (125 macroeconomic indicators).

Approach:
1. Load monthly macroeconomic data from FRED-MD
2. Compute factor scores (growth, inflation, credit conditions)
3. Cluster macro states into regimes:
   - Expansion: High growth, low unemployment, stable inflation
   - Recession: Negative growth, rising unemployment, credit stress
   - Recovery: Improving growth, falling unemployment
   - Overheating: High growth, rising inflation, tight labor market

Regimes:
- Expansion
- Recession
- Recovery
- Overheating

Data Sources:
- FRED-MD via fredapi or pandas-datareader
- Key indicators: IP, UNRATE, Yield Curve, Credit Spreads, CPI

Example:
    >>> from src.ml.macro_regime import MacroRegimeDetector
    >>> detector = MacroRegimeDetector()
    >>> detector.fit(macro_data)
    >>> regimes = detector.predict(macro_data)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.ml.regime_base import RegimeDetectorBase, RegimeSummary


class MacroRegimeDetector(RegimeDetectorBase):
    """
    Macroeconomic regime detector using FRED-MD indicators.

    Detects economic regimes based on:
    - Industrial production growth (economic activity)
    - Unemployment rate change (labor market)
    - Yield curve slope (10Y-3M, forward indicator)
    - Credit spreads (BAA-AAA, financial stress)
    - Inflation (CPI YoY)

    Args:
        n_regimes: Number of macro regimes (default 4)
        factors: List of factor columns to use
        lookback_months: Months for YoY calculations
        random_state: Random seed for clustering

    Attributes:
        factor_columns_: Columns used for factor analysis
        regime_labels_: Detected regime for each observation
        regime_mapping_: Mapping from cluster to regime name
    """

    def __init__(
        self,
        n_regimes: int = 4,
        factors: Optional[List[str]] = None,
        lookback_months: int = 12,
        random_state: int = 42,
        min_samples: int = 24,
    ):
        super().__init__(name="MacroRegimeDetector")

        self.n_regimes = n_regimes
        self.factors = factors or [
            "indprod_yoy",
            "unrate_diff",
            "yield_curve",
            "credit_spread",
            "cpi_yoy",
        ]
        self.lookback_months = lookback_months
        self.random_state = random_state
        self.min_samples = min_samples

        # Fitted attributes
        self.kmeans_ = None
        self.scaler_ = StandardScaler()
        self.regime_labels_: Optional[pd.Series] = None
        self.features_: Optional[pd.DataFrame] = None
        self.regime_mapping_: Dict[int, str] = {}
        self.cluster_centers_: Optional[np.ndarray] = None

    def fit(self, data: pd.DataFrame) -> MacroRegimeDetector:
        """
        Fit macro regime detector on macroeconomic data.

        Args:
            data: DataFrame with macroeconomic indicators

        Returns:
            Self for method chaining
        """
        self._validate_data(data)

        # Compute/validate factors
        self._compute_factors(data)

        # Standardize features
        X = self.scaler_.fit_transform(self.features_)

        # Fit k-means clustering
        self.kmeans_ = KMeans(
            n_clusters=self.n_regimes,
            random_state=self.random_state,
            n_init=10,
            max_iter=300,
        )
        cluster_labels = self.kmeans_.fit_predict(X)
        self.cluster_centers_ = self.kmeans_.cluster_centers_

        # Create interpretable regime mapping
        self._create_regime_mapping(X, cluster_labels)

        self.is_fitted = True

        return self

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        if data.empty:
            raise ValueError("Input data cannot be empty")

        if len(data) < self.min_samples:
            raise ValueError(
                f"Insufficient samples: {len(data)} < {self.min_samples} (minimum required)"
            )

    def _compute_factors(self, data: pd.DataFrame) -> None:
        """
        Compute macroeconomic factors from raw data.

        Factors computed:
        - indprod_yoy: Industrial production YoY growth
        - unrate_diff: Unemployment rate change
        - yield_curve: 10Y-3M Treasury spread
        - credit_spread: BAA-AAA corporate spread
        - cpi_yoy: CPI inflation YoY
        """
        self.features_ = pd.DataFrame(index=data.index)

        # Copy pre-computed factors if available
        for factor in self.factors:
            if factor in data.columns:
                self.features_[factor] = data[factor]

        # If raw data provided, compute factors
        if "INDPRO" in data.columns or "indpro" in data.columns:
            col = "INDPRO" if "INDPRO" in data.columns else "indpro"
            self.features_["indprod_yoy"] = data[col].pct_change(self.lookback_months) * 100

        if "UNRATE" in data.columns or "unrate" in data.columns:
            col = "UNRATE" if "UNRATE" in data.columns else "unrate"
            self.features_["unrate_diff"] = data[col].diff(self.lookback_months)

        if "yield_curve" in data.columns or "YC" in data.columns:
            col = "yield_curve" if "yield_curve" in data.columns else "YC"
            self.features_["yield_curve"] = data[col]

        if "credit_spread" in data.columns or "CREDIT" in data.columns:
            col = "credit_spread" if "credit_spread" in data.columns else "CREDIT"
            self.features_["credit_spread"] = data[col]

        if "CPI" in data.columns or "cpi" in data.columns:
            col = "CPI" if "CPI" in data.columns else "cpi"
            self.features_["cpi_yoy"] = data[col].pct_change(self.lookback_months) * 100

        # Drop rows with NaN from differencing
        self.features_ = self.features_.dropna()

        # Validate we have enough features
        missing = [f for f in self.factors if f not in self.features_.columns]
        if missing:
            raise ValueError(f"Missing required factors: {missing}")

    def _create_regime_mapping(self, X: np.ndarray, labels: np.ndarray) -> None:
        """
        Create interpretable regime labels from cluster assignments.

        Maps clusters to economic regime names based on factor values.
        """
        centers = self.cluster_centers_
        n_factors = len(self.factors)

        # Score each cluster on key dimensions
        cluster_scores = []
        for i in range(self.n_regimes):
            center = centers[i]

            # Growth score (indprod + employment)
            growth_idx = self.factors.index("indprod_yoy") if "indprod_yoy" in self.factors else 0
            unrate_idx = self.factors.index("unrate_diff") if "unrate_diff" in self.factors else 1
            growth_score = center[growth_idx] - center[unrate_idx]

            # Stress score (credit spread, yield curve)
            credit_idx = (
                self.factors.index("credit_spread") if "credit_spread" in self.factors else 3
            )
            yc_idx = self.factors.index("yield_curve") if "yield_curve" in self.factors else 2
            stress_score = center[credit_idx] - center[yc_idx]

            cluster_scores.append(
                {
                    "cluster": i,
                    "growth": growth_score,
                    "stress": stress_score,
                }
            )

        # Assign regime names based on growth/stress quadrants
        self.regime_mapping_ = {}
        assigned = set()

        # Recession: low growth, high stress
        recession_cluster = max(
            [c for c in cluster_scores if c["cluster"] not in assigned],
            key=lambda x: -x["growth"] + x["stress"],
        )["cluster"]
        self.regime_mapping_[recession_cluster] = "Recession"
        assigned.add(recession_cluster)

        # Expansion: high growth, low stress
        expansion_candidates = [c for c in cluster_scores if c["cluster"] not in assigned]
        if expansion_candidates:
            expansion_cluster = max(expansion_candidates, key=lambda x: x["growth"] - x["stress"])[
                "cluster"
            ]
            self.regime_mapping_[expansion_cluster] = "Expansion"
            assigned.add(expansion_cluster)

        # Recovery: improving growth, moderate stress
        recovery_candidates = [c for c in cluster_scores if c["cluster"] not in assigned]
        if recovery_candidates:
            recovery_cluster = max(recovery_candidates, key=lambda x: x["growth"])["cluster"]
            self.regime_mapping_[recovery_cluster] = "Recovery"
            assigned.add(recovery_cluster)

        # Overheating: high growth, potentially high inflation/stress
        remaining = [c for c in cluster_scores if c["cluster"] not in assigned]
        for c in remaining:
            self.regime_mapping_[c["cluster"]] = "Overheating"

        # Map cluster labels to regime names
        regime_names = [self.regime_mapping_.get(l, f"Cluster_{l}") for l in labels]
        self.regime_labels_ = pd.Series(
            regime_names, index=self.features_.index, name="macro_regime"
        )

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict macro regime for new data.

        Args:
            data: DataFrame with macroeconomic indicators

        Returns:
            Series of regime labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        self._validate_data(data)
        self._compute_factors(data)

        # Align features with training
        X = self.features_[self.factors].values
        X_scaled = self.scaler_.transform(X)

        # Predict cluster
        cluster_labels = self.kmeans_.predict(X_scaled)

        # Map to regime names
        regime_names = [self.regime_mapping_.get(c, f"Cluster_{c}") for c in cluster_labels]

        return pd.Series(regime_names, index=self.features_.index, name="macro_regime")

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime probabilities based on distance to centers.

        Args:
            data: DataFrame with macro indicators

        Returns:
            DataFrame with probability for each regime
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        self._validate_data(data)
        self._compute_factors(data)

        X = self.features_[self.factors].values
        X_scaled = self.scaler_.transform(X)

        # Distance to each center
        distances = np.zeros((len(X_scaled), self.n_regimes))
        for i in range(self.n_regimes):
            distances[:, i] = np.sqrt(((X_scaled - self.cluster_centers_[i]) ** 2).sum(axis=1))

        # Inverse distance → probability
        epsilon = 1e-6
        inverse_dist = 1 / (distances + epsilon)
        probs = inverse_dist / inverse_dist.sum(axis=1, keepdims=True)

        # Column names from regime mapping
        columns = [self.regime_mapping_.get(i, f"Cluster_{i}") for i in range(self.n_regimes)]

        return pd.DataFrame(probs, columns=columns, index=self.features_.index)

    def get_regime_summary(self) -> RegimeSummary:
        """
        Get summary information about detected regimes.

        Returns:
            RegimeSummary with label distribution and metadata
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.regime_labels_ is None:
            return RegimeSummary(
                name=self.name,
                n_regimes=0,
                regime_labels=[],
                label_distribution={},
                label_proportions={},
            )

        label_counts = self.regime_labels_.value_counts().to_dict()
        total = len(self.regime_labels_)
        proportions = {k: v / total for k, v in label_counts.items()}

        return RegimeSummary(
            name=self.name,
            n_regimes=self.n_regimes,
            regime_labels=list(label_counts.keys()),
            label_distribution=label_counts,
            label_proportions=proportions,
            metadata={
                "n_factors": len(self.factors),
                "factor_columns": self.factors,
                "cluster_centers": self.cluster_centers_.tolist()
                if self.cluster_centers_ is not None
                else None,
            },
        )

    def get_regime_characteristics(self) -> pd.DataFrame:
        """
        Get factor characteristics for each regime.

        Returns:
            DataFrame with mean factor values per regime
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.regime_labels_ is None or self.features_ is None:
            return pd.DataFrame()

        result = self.features_.copy()
        result["regime"] = self.regime_labels_

        summary = result.groupby("regime")[self.factors].mean()

        return summary

    def get_current_regime(self) -> str:
        """
        Get the most recent regime classification.

        Returns:
            Current regime label
        """
        if not self.is_fitted or self.regime_labels_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.regime_labels_.iloc[-1]

    @staticmethod
    def load_fred_md(api_key: Optional[str] = None) -> pd.DataFrame:
        """
        Load FRED-MD dataset.

        Args:
            api_key: FRED API key (optional, required for newest data)

        Returns:
            DataFrame with FRED-MD indicators
        """
        try:
            from fredapi import Fred

            fred = Fred(api_key=api_key)

            # Core FRED-MD series
            series_map = {
                "INDPRO": "Industrial Production",
                "UNRATE": "Unemployment Rate",
                "DGS10": "10-Year Treasury",
                "DGS3MO": "3-Month Treasury",
                "BAA": "BAA Corporate Yield",
                "AAA": "AAA Corporate Yield",
                "CPIAUCSL": "CPI All Items",
            }

            data = {}
            for symbol in series_map.keys():
                try:
                    series = fred.get_series(symbol)
                    data[symbol] = series
                except Exception as e:
                    print(f"Warning: Could not load {symbol}: {e}")

            if not data:
                raise ValueError("No FRED series loaded. Check API key.")

            df = pd.DataFrame(data)

            # Compute derived indicators
            if "DGS10" in df and "DGS3MO" in df:
                df["yield_curve"] = df["DGS10"] - df["DGS3MO"]

            if "BAA" in df and "AAA" in df:
                df["credit_spread"] = df["BAA"] - df["AAA"]

            return df

        except ImportError:
            raise ImportError("fredapi package required. Run: uv add fredapi")
