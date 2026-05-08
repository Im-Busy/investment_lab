"""
Historical Analog Matching (FS2)

"When did the market look like this before?" — k-NN search over historical
feature windows to find the most similar past market conditions and show
what happened next.

Uses sklearn NearestNeighbors for exact k-NN. For large datasets (>100k
points), consider switching to faiss with the faiss skill.

Key concept:
1. Fit on historical feature windows (training set)
2. Query with current/latest feature window
3. Return k most similar historical periods
4. Show forward returns from each analog (what happened next)

Example:
    >>> from src.ml.historical_analog import HistoricalAnalogMatcher
    >>> matcher = HistoricalAnalogMatcher(k=10)
    >>> matcher.fit(features)
    >>> analogs = matcher.find_analogs(query_features)
    >>> forward_returns = matcher.get_analog_forward_returns(prices, horizon=20)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class AnalogResult:
    """Single historical analog match result."""

    analog_date: pd.Timestamp
    distance: float
    rank: int
    forward_returns: Dict[int, float] = field(default_factory=dict)
    analog_index: int = -1


@dataclass
class AnalogMatchSet:
    """Set of k historical analogs for a query point."""

    query_date: Optional[pd.Timestamp]
    analogs: List[AnalogResult]
    avg_forward_return: Dict[int, float] = field(default_factory=dict)
    median_forward_return: Dict[int, float] = field(default_factory=dict)
    win_rate: Dict[int, float] = field(default_factory=dict)


class HistoricalAnalogMatcher:
    """
    k-NN historical analog search engine.

    Finds the k most similar historical market windows based on feature
    vector similarity. Answers: "when did the market look like this before,
    and what happened next?"

    Args:
        k: Number of nearest neighbors to find
        metric: Distance metric ("euclidean", "cosine", "manhattan", "chebyshev")
        n_jobs: Number of parallel jobs for NearestNeighbors
        min_distance: Minimum distance threshold (analogs closer than this
            are considered "too similar", e.g., duplicates)
        lookback_window: Number of bars to include in feature vector
            (if features are multi-bar windows, set to 1)

    Attributes:
        knn_: Fitted NearestNeighbors model
        scaler_: StandardScaler for feature normalization
        feature_index_: Index (dates) of the fitted feature matrix
        analog_indices_: Indices of found analogs
        analog_distances_: Distances of found analogs

    Example:
        >>> matcher = HistoricalAnalogMatcher(k=10)
        >>> matcher.fit(features)
        >>> analogs = matcher.find_analogs(query)
        >>> print(analogs.avg_forward_return[20])  # avg 20-bar forward return
    """

    def __init__(
        self,
        k: int = 10,
        metric: str = "euclidean",
        n_jobs: int = -1,
        min_distance: float = 0.0,
        lookback_window: int = 1,
    ):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        self.k = k
        self.metric = metric
        self.n_jobs = n_jobs
        self.min_distance = min_distance
        self.lookback_window = lookback_window

        self.knn_: Optional[NearestNeighbors] = None
        self.scaler_ = StandardScaler()
        self.feature_index_: Optional[pd.Index] = None
        self.feature_columns_: Optional[List[str]] = None
        self.analog_indices_: Optional[np.ndarray] = None
        self.analog_distances_: Optional[np.ndarray] = None
        self._forward_rets_: Optional[pd.DataFrame] = None
        self.fitted_n_: int = 0
        self.is_fitted: bool = False

    def fit(self, data: pd.DataFrame) -> HistoricalAnalogMatcher:
        """
        Fit the k-NN model on historical feature data.

        Args:
            data: DataFrame with features (rows = time, columns = features)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data is empty or has insufficient rows
        """
        if data.empty:
            raise ValueError("Input data cannot be empty")

        if len(data) < self.k:
            raise ValueError(f"Insufficient samples: {len(data)} < {self.k} (k required)")

        if data.isna().any().any():
            raise ValueError("Input data contains NaN values. Handle missing values first.")

        self.feature_index_ = data.index.copy()
        self.feature_columns_ = list(data.columns)

        X_scaled = self.scaler_.fit_transform(data)

        self.knn_ = NearestNeighbors(
            n_neighbors=min(self.k, len(data)),
            metric=self.metric,
            n_jobs=self.n_jobs,
        )
        self.knn_.fit(X_scaled)

        self.fitted_n_ = len(data)
        self.is_fitted = True

        return self

    def find_analogs(
        self,
        query: pd.DataFrame,
        exclude_recent: int = 20,
    ) -> AnalogMatchSet:
        """
        Find k nearest historical analogs for query feature vectors.

        Args:
            query: DataFrame with features (same columns as training)
            exclude_recent: Exclude analogs within this many bars from query date
                to prevent near-duplicate matches

        Returns:
            AnalogMatchSet with k closest historical analogs

        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if query.empty:
            raise ValueError("Query data cannot be empty")

        if query.isna().any().any():
            raise ValueError("Query data contains NaN values")

        query_date = query.index[-1] if len(query) > 0 else None

        X_query = self.scaler_.transform(query)
        distances, indices = self.knn_.kneighbors(X_query)

        last_distances = distances[-1]
        last_indices = indices[-1]

        self.analog_distances_ = last_distances
        self.analog_indices_ = last_indices

        analogs = []
        for rank, (dist, idx) in enumerate(zip(last_distances, last_indices)):
            analog_date = self.feature_index_[idx]

            if dist < self.min_distance:
                continue

            # Skip near-duplicate if query date is within exclude_recent
            if query_date is not None:
                if isinstance(analog_date, pd.Timestamp) and isinstance(query_date, pd.Timestamp):
                    bar_diff = abs((query_date - analog_date).days)
                    if bar_diff <= exclude_recent:
                        continue

            analogs.append(
                AnalogResult(
                    analog_date=analog_date,
                    distance=float(dist),
                    rank=rank + 1,
                    analog_index=int(idx),
                )
            )

        return AnalogMatchSet(
            query_date=query_date,
            analogs=analogs[: self.k],
        )

    def get_analog_forward_returns(
        self,
        prices: pd.Series,
        horizons: Optional[List[int]] = None,
    ) -> AnalogMatchSet:
        """
        Compute forward returns for the most recent analog query.

        For each analog date, computes what happened over the next N bars.
        This answers "what happened next when the market looked like this?"

        Args:
            prices: Price series with same index as training data
            horizons: Forward horizons in bars (default: [1, 5, 10, 20, 50])

        Returns:
            AnalogMatchSet with forward returns attached to each analog
        """
        if self.analog_indices_ is None:
            raise ValueError("No analogs found. Call find_analogs() first.")

        if horizons is None:
            horizons = [1, 5, 10, 20, 50]

        analogs = []
        forward_returns = {h: [] for h in horizons}

        for rank, (dist, idx) in enumerate(zip(self.analog_distances_, self.analog_indices_)):
            analog_date = self.feature_index_[idx]

            if dist < self.min_distance:
                continue

            fwd_ret = {}
            for h in horizons:
                if idx + h < len(prices):
                    ret = (prices.iloc[idx + h] / prices.iloc[idx]) - 1
                    fwd_ret[h] = float(ret)
                    forward_returns[h].append(float(ret))
                else:
                    fwd_ret[h] = np.nan

            analogs.append(
                AnalogResult(
                    analog_date=analog_date,
                    distance=float(dist),
                    rank=rank + 1,
                    analog_index=int(idx),
                    forward_returns=fwd_ret,
                )
            )

        avg_forward = {}
        median_forward = {}
        win_rate = {}

        for h in horizons:
            rets = [r for r in forward_returns[h] if not np.isnan(r)]
            if rets:
                avg_forward[h] = float(np.mean(rets))
                median_forward[h] = float(np.median(rets))
                win_rate[h] = float(sum(1 for r in rets if r > 0) / len(rets))
            else:
                avg_forward[h] = np.nan
                median_forward[h] = np.nan
                win_rate[h] = np.nan

        return AnalogMatchSet(
            query_date=None,
            analogs=analogs[: self.k],
            avg_forward_return=avg_forward,
            median_forward_return=median_forward,
            win_rate=win_rate,
        )

    def find_sequential_analogs(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        window_size: int = 10,
        step: int = 5,
        k: int = 5,
    ) -> pd.DataFrame:
        """
        Slide through time finding the k best historical analogs for each window.

        This is useful for: "throughout history, what are the 5 most similar
        past windows at each time step?" Results show how analog-based strategies
        would have performed.

        Args:
            prices: Price series aligned with features
            features: Feature DataFrame aligned with prices
            window_size: Number of bars in each query window
            step: Step size between windows
            k: Number of analogs per window

        Returns:
            DataFrame with columns: query_date, analog_date, distance, rank,
            forward_return_{h1}, forward_return_{h2}, ...
        """
        results = []

        for start in range(0, len(features) - window_size, step):
            query = features.iloc[start : start + window_size]
            query_date = query.index[-1] if len(query) > 0 else None

            analogs = self.find_analogs(query, exclude_recent=window_size)

            for analog in analogs.analogs[:k]:
                row = {
                    "query_date": query_date,
                    "analog_date": analog.analog_date,
                    "distance": analog.distance,
                    "rank": analog.rank,
                    "forward_return_5": self._compute_forward_return(
                        prices, analog.analog_index, 5
                    ),
                    "forward_return_10": self._compute_forward_return(
                        prices, analog.analog_index, 10
                    ),
                    "forward_return_20": self._compute_forward_return(
                        prices, analog.analog_index, 20
                    ),
                }
                results.append(row)

        return pd.DataFrame(results)

    def get_analog_distribution(self) -> pd.DataFrame:
        """
        Get statistical summary of analog distances.

        Returns:
            DataFrame with distance statistics for the last query
        """
        if self.analog_distances_ is None:
            raise ValueError("No analogs found. Call find_analogs() first.")

        return pd.DataFrame(
            {
                "distance": self.analog_distances_,
                "analog_date": [self.feature_index_[int(i)] for i in self.analog_indices_],
                "rank": range(1, len(self.analog_distances_) + 1),
            }
        ).set_index("rank")

    def get_analog_consensus(
        self,
        prices: pd.Series,
        horizons: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Get consensus forecast from all analogs.

        Aggregates forward returns across all k analogs to produce a
        consensus view of what "typically" happens next.

        Args:
            prices: Price series aligned with training data
            horizons: Forward horizons to compute

        Returns:
            DataFrame with columns: horizon, avg_return, median_return,
            win_rate, n_analogs
        """
        if horizons is None:
            horizons = [1, 5, 10, 20, 50]

        if self.analog_indices_ is None:
            raise ValueError("No analogs found. Call find_analogs() first.")

        rows = []
        for h in horizons:
            rets = []
            for idx in self.analog_indices_:
                ret = self._compute_forward_return(prices, int(idx), h)
                if not np.isnan(ret):
                    rets.append(ret)

            if rets:
                rows.append(
                    {
                        "horizon": h,
                        "avg_return": float(np.mean(rets)),
                        "median_return": float(np.median(rets)),
                        "std_return": float(np.std(rets)),
                        "win_rate": float(sum(1 for r in rets if r > 0) / len(rets)),
                        "min_return": float(min(rets)),
                        "max_return": float(max(rets)),
                        "n_analogs": len(rets),
                    }
                )

        return pd.DataFrame(rows).set_index("horizon")

    def _compute_forward_return(
        self,
        prices: pd.Series,
        idx: int,
        horizon: int,
    ) -> float:
        """Compute forward return from a given index."""
        if idx + horizon < len(prices):
            return float((prices.iloc[idx + horizon] / prices.iloc[idx]) - 1)
        return np.nan
