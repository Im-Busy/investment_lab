"""
Cross-Symbol Dynamic Clustering (FS12)

Extends CrossAssetFeatures with dynamic clustering of symbols based on
rolling correlation, co-movement, and lead-lag relationships.

Identifies which symbols are moving together NOW vs historically,
and detects correlation regime shifts. Useful for:
- Portfolio diversification (avoid over-concentration in correlated assets)
- Pair trading opportunities (symbols diverging from their usual cluster)
- Regime-aware allocation (shift weights between correlated and uncorrelated)
- Lead-lag detection (which symbol leads the group)

Features:
- Rolling correlation matrix → hierarchical clustering of symbols
- Dynamic cluster membership over time (which symbols cluster together)
- Cluster stability score (how often symbols stay in the same cluster)
- Lead-lag relationship detection via cross-correlation
- Correlation regime shift detection (sudden breaks in correlation structure)
- Co-movement strength index per cluster

Example:
    >>> from src.ml.cross_symbol_cluster import CrossSymbolCluster
    >>> dsc = CrossSymbolCluster(correlation_window=60, n_clusters=3)
    >>> dsc.fit(symbol_data_dict)
    >>> clusters = dsc.predict(symbol_data_dict)
    >>> leads = dsc.detect_lead_lag("SPY", "XLF")
    >>> shifts = dsc.detect_correlation_shifts(symbol_data_dict)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans


class CrossSymbolCluster:
    """
    Dynamic cross-symbol clustering based on rolling correlations.

    Args:
        correlation_window: Rolling window size for correlation matrix
        n_clusters: Target number of symbol clusters (0 = auto-detect)
        method: Clustering method ("hierarchical" or "kmeans")
        linkage_method: Linkage criterion for hierarchical clustering
        min_correlation: Minimum absolute correlation to consider as co-moving
        regime_shift_threshold: Correlation change threshold for regime shift detection
        lead_lag_max_lag: Maximum lag for lead-lag cross-correlation
        stability_window: Rolling window for cluster stability score

    Attributes:
        correlation_matrix_: Latest full-sample correlation matrix
        correlation_history_: Rolling correlation for key pairs over time
        cluster_labels_: Cluster labels for each symbol (current snapshot)
        cluster_history_: Cluster membership over time for each symbol
        stability_scores_: Per-symbol cluster stability scores
        lead_lag_pairs_: Detected lead-lag pairs with lag values
        regime_shifts_: Detected correlation regime shift dates
    """

    def __init__(
        self,
        correlation_window: int = 60,
        n_clusters: int = 3,
        method: str = "hierarchical",
        linkage_method: str = "ward",
        min_correlation: float = 0.3,
        regime_shift_threshold: float = 0.3,
        lead_lag_max_lag: int = 20,
        stability_window: int = 120,
    ):
        if correlation_window < 5:
            raise ValueError("correlation_window must be >= 5")
        if not 0 < min_correlation < 1:
            raise ValueError("min_correlation must be in (0, 1)")

        self.correlation_window = correlation_window
        self.n_clusters = n_clusters
        self.method = method
        self.linkage_method = linkage_method
        self.min_correlation = min_correlation
        self.regime_shift_threshold = regime_shift_threshold
        self.lead_lag_max_lag = lead_lag_max_lag
        self.stability_window = stability_window

        self.correlation_matrix_: Optional[pd.DataFrame] = None
        self.correlation_history_: Dict[Tuple[str, str], pd.Series] = {}
        self.cluster_labels_: Dict[str, int] = {}
        self.cluster_history_: Optional[pd.DataFrame] = None
        self.stability_scores_: Optional[pd.Series] = None
        self.lead_lag_pairs_: Dict[Tuple[str, str], Dict[str, float]] = {}
        self.regime_shifts_: List[pd.Timestamp] = []
        self.symbols_: List[str] = []
        self._distance_matrix_: Optional[np.ndarray] = None

    def _compute_returns(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compute aligned returns matrix for all symbols."""
        returns = {}
        for sym, df in symbol_data.items():
            if "Close" not in df.columns:
                continue
            ret = df["Close"].pct_change()
            returns[sym] = ret

        if not returns:
            return pd.DataFrame()

        result = pd.DataFrame(returns)
        result = result.dropna(how="all")
        return result

    def _compute_rolling_correlation(
        self,
        returns: pd.DataFrame,
        sym_a: str,
        sym_b: str,
    ) -> pd.Series:
        """Compute rolling correlation between two symbols."""
        if sym_a not in returns.columns or sym_b not in returns.columns:
            return pd.Series(dtype=float)

        common_idx = returns[[sym_a, sym_b]].dropna().index
        if len(common_idx) < self.correlation_window:
            return pd.Series(dtype=float)

        a = returns[sym_a].reindex(common_idx)
        b = returns[sym_b].reindex(common_idx)
        return a.rolling(
            self.correlation_window, min_periods=max(10, self.correlation_window // 3)
        ).corr(b)

    def fit(self, symbol_data: Dict[str, pd.DataFrame]) -> CrossSymbolCluster:
        """
        Fit cross-symbol clustering on historical data.

        Args:
            symbol_data: Dict of symbol -> OHLCV DataFrame with Close column

        Returns:
            Self for method chaining
        """
        self.symbols_ = [s for s in symbol_data if "Close" in symbol_data[s].columns]
        if len(self.symbols_) < 2:
            raise ValueError(f"Need at least 2 symbols with Close data, got {len(self.symbols_)}")

        returns = self._compute_returns(symbol_data)
        if returns.empty or len(returns.columns) < 2:
            raise ValueError("Insufficient return data for clustering")

        self.correlation_matrix_ = returns.corr()

        self._compute_cluster_labels(self.correlation_matrix_)

        self._compute_correlation_history(returns)

        self._compute_cluster_history(returns)

        self._compute_stability_scores()

        self._detect_regime_shifts(returns)

        return self

    def _compute_cluster_labels(self, corr_matrix: pd.DataFrame) -> None:
        """Assign cluster labels from correlation matrix."""
        valid_cols = corr_matrix.columns[
            corr_matrix.notna().all() & (corr_matrix.columns.isin(self.symbols_))
        ]
        if len(valid_cols) < 1:
            return

        sub = corr_matrix.loc[valid_cols, valid_cols]
        distance = 1 - sub.abs()

        if self._distance_matrix_ is not None and self._distance_matrix_.values is not None:
            self._distance_matrix_ = distance.values

        self._distance_matrix_ = distance.values

        if self.method == "hierarchical":
            if len(distance) >= 2:
                condensed = squareform(distance.values)
                Z = linkage(condensed, method=self.linkage_method)
                n_c = min(self.n_clusters or 3, len(distance))
                labels = fcluster(Z, n_c, criterion="maxclust")
                self.cluster_labels_ = dict(zip(valid_cols, labels))
            else:
                self.cluster_labels_ = {s: 0 for s in self.symbols_}
        elif self.method == "kmeans":
            n_c = min(self.n_clusters or 3, len(distance))
            km = KMeans(n_clusters=n_c, random_state=42, n_init=10)
            km_labels = km.fit_predict(distance.values)
            self.cluster_labels_ = dict(zip(valid_cols, km_labels))
        else:
            self.cluster_labels_ = {s: 0 for s in self.symbols_}

    def _compute_correlation_history(self, returns: pd.DataFrame) -> None:
        """Compute rolling correlation for key symbol pairs."""
        syms = list(returns.columns)
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                corr_series = self._compute_rolling_correlation(returns, syms[i], syms[j])
                if not corr_series.empty:
                    self.correlation_history_[(syms[i], syms[j])] = corr_series

    def _compute_cluster_history(self, returns: pd.DataFrame) -> None:
        """Compute cluster membership for each time point."""
        syms = list(returns.columns)
        if len(syms) < 2:
            return

        history_records = []

        for t in range(self.correlation_window - 1, len(returns)):
            window = returns.iloc[t - self.correlation_window + 1 : t + 1]
            corr = window.corr()
            valid = corr.columns[corr.notna().all()]

            if len(valid) < 2:
                continue

            sub = corr.loc[valid, valid]
            dist = 1 - sub.abs()

            if len(dist) < 2:
                continue

            condensed = squareform(dist.values)
            Z = linkage(condensed, method=self.linkage_method)
            n_c = max(1, min(self.n_clusters or 3, len(dist)))
            labels = fcluster(Z, n_c, criterion="maxclust")

            record = {"date": returns.index[t]}
            for sym, lbl in zip(valid, labels):
                record[f"cluster_{sym}"] = lbl
            history_records.append(record)

        if history_records:
            self.cluster_history_ = pd.DataFrame(history_records).set_index("date")
            if len(self.cluster_history_) > 1:
                self.cluster_history_ = self.cluster_history_.astype(float)

    def _compute_stability_scores(self) -> None:
        """Compute per-symbol cluster stability scores."""
        if self.cluster_history_ is None or self.cluster_history_.empty:
            return

        stability = {}
        for col in self.cluster_history_.columns:
            if not col.startswith("cluster_"):
                continue
            sym = col.replace("cluster_", "")
            series = self.cluster_history_[col].dropna()
            if len(series) < 2:
                continue
            changes = (series.diff().abs() > 0).sum()
            stability[sym] = 1 - changes / max(len(series) - 1, 1)

        self.stability_scores_ = pd.Series(stability).sort_values(ascending=False)

    def _detect_regime_shifts(self, returns: pd.DataFrame) -> None:
        """Detect dates where correlation structure significantly changes."""
        if self.cluster_history_ is None or self.cluster_history_.empty:
            return

        n_dims = self.cluster_history_.shape[1]
        if n_dims < 2:
            return

        changes = self.cluster_history_.diff().abs().sum(axis=1)
        threshold = self.regime_shift_threshold * n_dims

        for idx, val in changes.items():
            if val > threshold:
                self.regime_shifts_.append(idx)

    def predict(self, symbol_data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """
        Predict current cluster assignments for symbols.

        Args:
            symbol_data: Dict of symbol -> OHLCV DataFrame

        Returns:
            Dict of symbol -> cluster_id
        """
        returns = self._compute_returns(symbol_data)
        if returns.empty or len(returns.columns) < 2:
            return {}

        corr = returns.corr()
        self._compute_cluster_labels(corr)
        return dict(self.cluster_labels_)

    def detect_lead_lag(
        self,
        symbol_a: str,
        symbol_b: str,
        symbol_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, float]:
        """
        Detect lead-lag relationship between two symbols via cross-correlation.

        Positive lag means symbol_a leads symbol_b.

        Args:
            symbol_a: Leading symbol candidate
            symbol_b: Lagging symbol candidate
            symbol_data: Optional symbol data (uses fitted data if None)

        Returns:
            Dict with {"lag": int, "correlation": float, "leader": str}
        """
        data = symbol_data if symbol_data is not None else {}
        returns = (
            self._compute_returns(data)
            if data
            else (
                self._compute_returns({s: None for s in self.symbols_})
                if self.symbols_
                else pd.DataFrame()
            )
        )

        if returns.empty or symbol_a not in returns.columns or symbol_b not in returns.columns:
            return {"lag": 0, "correlation": 0.0, "leader": "unknown"}

        common = returns[[symbol_a, symbol_b]].dropna()
        if len(common) < self.correlation_window:
            return {"lag": 0, "correlation": 0.0, "leader": "unknown"}

        best_corr = 0.0
        best_lag = 0
        for lag in range(-self.lead_lag_max_lag, self.lead_lag_max_lag + 1):
            if lag < 0:
                corr_val = common[symbol_a].iloc[-lag:].corr(common[symbol_b].iloc[:lag])
            elif lag > 0:
                corr_val = common[symbol_a].iloc[:-lag].corr(common[symbol_b].iloc[lag:])
            else:
                corr_val = common[symbol_a].corr(common[symbol_b])

            if abs(corr_val) > abs(best_corr):
                best_corr = corr_val
                best_lag = lag

        result = {
            "lag": best_lag,
            "correlation": float(best_corr),
            "leader": symbol_a if best_lag > 0 else (symbol_b if best_lag < 0 else "none"),
        }
        self.lead_lag_pairs_[(symbol_a, symbol_b)] = result
        return result

    def get_correlation(self, symbol_a: str, symbol_b: str) -> Optional[float]:
        """Get current correlation between two symbols."""
        if self.correlation_matrix_ is None:
            return None
        if symbol_a not in self.correlation_matrix_.columns:
            return None
        if symbol_b not in self.correlation_matrix_.index:
            return None
        return float(self.correlation_matrix_.loc[symbol_a, symbol_b])

    def get_correlation_history(self, symbol_a: str, symbol_b: str) -> Optional[pd.Series]:
        """Get rolling correlation history for a symbol pair."""
        return self.correlation_history_.get((symbol_a, symbol_b)) or self.correlation_history_.get(
            (symbol_b, symbol_a)
        )

    def get_cluster_members(self, cluster_id: int) -> List[str]:
        """Get symbols belonging to a specific cluster."""
        return [s for s, c in self.cluster_labels_.items() if c == cluster_id]

    def get_clusters(self) -> Dict[int, List[str]]:
        """Get all clusters and their member symbols."""
        clusters: Dict[int, List[str]] = {}
        for sym, c_id in self.cluster_labels_.items():
            clusters.setdefault(c_id, []).append(sym)
        return clusters

    def get_co_movement_strength(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute co-movement strength index for each symbol.

        Measures how strongly each symbol moves with its cluster peers.
        Higher values = stronger co-movement.

        Args:
            symbol_data: Dict of symbol -> OHLCV DataFrame

        Returns:
            DataFrame with symbol, cluster, co_movement_strength columns
        """
        returns = self._compute_returns(symbol_data)
        if returns.empty:
            return pd.DataFrame()

        clusters = self.get_clusters()
        rows = []

        for cluster_id, symbols in clusters.items():
            if len(symbols) < 2:
                for s in symbols:
                    rows.append({"symbol": s, "cluster": cluster_id, "co_movement_strength": 1.0})
                continue

            cluster_returns = returns[symbols].dropna()
            if cluster_returns.empty:
                continue

            corr_matrix = cluster_returns.corr()
            for sym in symbols:
                peers = [p for p in symbols if p != sym]
                peer_corrs = [corr_matrix.loc[sym, p] for p in peers if p in corr_matrix.columns]
                strength = float(np.mean(peer_corrs)) if peer_corrs else 0.0
                rows.append(
                    {
                        "symbol": sym,
                        "cluster": cluster_id,
                        "co_movement_strength": strength,
                    }
                )

        return pd.DataFrame(rows)

    def detect_correlation_shifts(
        self, symbol_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Detect correlation regime shift dates and magnitudes.

        Args:
            symbol_data: Optional symbol data for re-detection

        Returns:
            DataFrame with shift dates and magnitude
        """
        if symbol_data is not None:
            returns = self._compute_returns(symbol_data)
            self._detect_regime_shifts(returns)

        if not self.regime_shifts_:
            return pd.DataFrame(columns=["date", "magnitude"])

        rows = []
        for shift_date in self.regime_shifts_:
            if self.cluster_history_ is not None and shift_date in self.cluster_history_.index:
                pos = self.cluster_history_.index.get_loc(shift_date)
                if pos > 0:
                    prev = self.cluster_history_.iloc[pos - 1]
                    curr = self.cluster_history_.iloc[pos]
                    magnitude = (prev - curr).abs().sum()
                    rows.append({"date": shift_date, "magnitude": float(magnitude)})

        return pd.DataFrame(rows).sort_values("magnitude", ascending=False)

    def get_divergence_alerts(
        self, symbol_data: Dict[str, pd.DataFrame], threshold: float = 2.0
    ) -> pd.DataFrame:
        """
        Find symbols that are diverging from their historical cluster.

        A symbol is diverging when its current pairwise correlation
        with cluster peers has dropped significantly vs historical norm.

        Args:
            symbol_data: Dict of symbol -> OHLCV DataFrame
            threshold: Number of standard deviations below mean to flag

        Returns:
            DataFrame with diverging symbols, current correlation, and z-score
        """
        returns = self._compute_returns(symbol_data)
        if returns.empty:
            return pd.DataFrame()

        clusters = self.get_clusters()
        alerts = []

        for cluster_id, symbols in clusters.items():
            if len(symbols) < 2:
                continue

            for i, sym_a in enumerate(symbols):
                for sym_b in symbols[i + 1 :]:
                    hist = self.get_correlation_history(sym_a, sym_b)
                    if hist is None or hist.empty:
                        continue

                    latest = hist.iloc[-1]
                    mean_corr = hist.mean()
                    std_corr = hist.std() or 1e-10
                    z_score = (latest - mean_corr) / std_corr

                    if abs(z_score) > threshold:
                        alerts.append(
                            {
                                "symbol_a": sym_a,
                                "symbol_b": sym_b,
                                "current_correlation": float(latest),
                                "historical_mean": float(mean_corr),
                                "z_score": float(z_score),
                                "direction": "decreasing" if z_score < 0 else "increasing",
                                "cluster": cluster_id,
                            }
                        )

        return pd.DataFrame(alerts).sort_values("z_score")

    def get_cluster_summary(self) -> Dict[str, object]:
        """
        Get a comprehensive summary of cross-symbol clustering.

        Returns:
            Dict with clusters, stability scores, regime shifts, and lead-lag pairs
        """
        return {
            "n_symbols": len(self.symbols_),
            "n_clusters": len(set(self.cluster_labels_.values())),
            "method": self.method,
            "correlation_window": self.correlation_window,
            "clusters": self.get_clusters(),
            "stability_scores": (
                self.stability_scores_.to_dict() if self.stability_scores_ is not None else {}
            ),
            "n_regime_shifts": len(self.regime_shifts_),
            "n_lead_lag_pairs": len(self.lead_lag_pairs_),
            "lead_lag_pairs": {f"{a}_{b}": v for (a, b), v in self.lead_lag_pairs_.items()},
        }
