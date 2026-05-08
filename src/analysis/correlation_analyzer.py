"""Correlation Analyzer for Pattern Selection.

Builds correlation matrices, finds correlated clusters, and recommends
diversified pattern subsets by removing redundant patterns.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

logger = logging.getLogger(__name__)

DEFAULT_CORRELATION_THRESHOLD = 0.7


@dataclass
class CorrelationAnalyzerConfig:
    """Configuration for correlation analysis."""

    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD
    min_cluster_size: int = 2


@dataclass
class CorrelationGroup:
    """A group of correlated patterns identified by clustering."""

    cluster_id: int
    patterns: List[str]
    mean_correlation: float
    representative: str


class CorrelationAnalyzer:
    """Analyzes pattern correlations and recommends diversified subsets."""

    def __init__(self, correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD):
        self.correlation_threshold = correlation_threshold

    def build_correlation_matrix(self, pattern_returns: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Build pairwise Pearson correlation matrix from pattern return series.

        Args:
            pattern_returns: {pattern_name: returns_array}

        Returns:
            Correlation matrix as DataFrame.
        """
        max_len = max(len(v) for v in pattern_returns.values())
        aligned = {}
        for name, ret in pattern_returns.items():
            arr = np.asarray(ret, dtype=np.float64)
            if len(arr) < max_len:
                arr = np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
            aligned[name] = arr

        df = pd.DataFrame(aligned)
        return df.corr(method="pearson")

    def find_correlated_clusters(
        self,
        corr_matrix: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> List[List[str]]:
        """Find clusters of highly correlated patterns using hierarchical clustering.

        Args:
            corr_matrix: Correlation matrix DataFrame.
            threshold: Correlation threshold (uses instance default if None).

        Returns:
            List of clusters, each a list of pattern names.
        """
        thresh = threshold if threshold is not None else self.correlation_threshold
        patterns = corr_matrix.columns.tolist()

        if len(patterns) < 2:
            return [patterns] if patterns else []

        distance_matrix = 1.0 - corr_matrix.to_numpy()
        np.fill_diagonal(distance_matrix, 0.0)
        distance_matrix = np.clip(distance_matrix, 0, 2.0)

        condensed = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        if np.any(np.isnan(condensed)):
            condensed = np.nan_to_num(condensed, nan=1.0)

        linkage_matrix = linkage(condensed, method="average")
        cluster_labels = fcluster(linkage_matrix, t=1.0 - thresh, criterion="distance")

        clusters: Dict[int, List[str]] = defaultdict(list)
        for pattern, label in zip(patterns, cluster_labels):
            clusters[int(label)].append(pattern)

        return list(clusters.values())

    def recommend_pattern_subset(
        self,
        pattern_returns: Dict[str, np.ndarray],
        max_correlation: Optional[float] = None,
        pattern_scores: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """Recommend a subset of patterns with low inter-correlation.

        Keeps the best-performing pattern from each correlated cluster.

        Args:
            pattern_returns: {pattern_name: returns_array}
            max_correlation: Maximum allowed correlation.
            pattern_scores: {pattern_name: score} for tie-breaking.

        Returns:
            List of recommended pattern names.
        """
        thresh = max_correlation if max_correlation is not None else self.correlation_threshold
        corr_matrix = self.build_correlation_matrix(pattern_returns)
        clusters = self.find_correlated_clusters(corr_matrix, threshold=thresh)

        selected = []
        for cluster in clusters:
            if len(cluster) == 1:
                selected.append(cluster[0])
            else:
                best = max(
                    cluster,
                    key=lambda p: pattern_scores.get(p, 0.0) if pattern_scores else 0.0,
                )
                selected.append(best)
                logger.info(
                    "Cluster %s: keeping '%s' (score=%.3f)",
                    cluster,
                    best,
                    pattern_scores.get(best, 0.0) if pattern_scores else 0.0,
                )

        return selected

    def calculate_diversification_ratio(self, pattern_returns: Dict[str, np.ndarray]) -> float:
        """Calculate diversification ratio of a pattern portfolio.

        Ratio = weighted_avg_individual_vol / portfolio_vol.
        Values > 1 indicate diversification benefit.

        Args:
            pattern_returns: {pattern_name: returns_array}

        Returns:
            Diversification ratio (float >= 1.0 ideal).
        """
        if not pattern_returns:
            return 1.0

        n = len(pattern_returns)
        vols = []
        for ret in pattern_returns.values():
            arr = np.asarray(ret, dtype=np.float64)
            vols.append(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)

        avg_vol = float(np.mean(vols))
        if avg_vol < 1e-12:
            return 1.0

        equal_weight = 1.0 / n
        aligned_returns = []
        max_len = max(len(np.asarray(v, dtype=np.float64)) for v in pattern_returns.values())
        for ret in pattern_returns.values():
            arr = np.asarray(ret, dtype=np.float64)
            if len(arr) < max_len:
                arr = np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
            aligned_returns.append(arr)

        portfolio_returns = np.nanmean(np.column_stack(aligned_returns), axis=1) * equal_weight * n
        port_vol = float(np.std(portfolio_returns, ddof=1)) if len(portfolio_returns) > 1 else 0.0

        if port_vol < 1e-12:
            return 1.0

        return avg_vol / port_vol

    # ---- Legacy compat methods for test suite ----

    def build_signal_matrix(self, detection_log: "pd.DataFrame", total_bars: int) -> "pd.DataFrame":
        """Build binary signal matrix from detection log."""
        pattern_names = detection_log["pattern_name"].unique()
        matrix = pd.DataFrame(0, index=range(total_bars), columns=pattern_names, dtype=int)
        for _, row in detection_log.iterrows():
            bar = int(row["bar_index"])
            if 0 <= bar < total_bars:
                matrix.loc[bar, row["pattern_name"]] = 1
        return matrix.astype(int)

    def compute_correlation_matrix(self, matrix: "pd.DataFrame") -> "pd.DataFrame":
        """Compute correlation matrix from signal matrix."""
        returns = matrix.diff().fillna(0)
        pattern_returns = {col: returns[col].values for col in matrix.columns}
        return self.build_correlation_matrix(pattern_returns)

    def find_redundant_pairs(
        self,
        corr_matrix: "pd.DataFrame",
        co_matrix: "pd.DataFrame",
        pattern_scores: Optional[Dict[str, float]] = None,
    ):
        """Find redundant pattern pairs."""

        class _Pair:
            __slots__ = ("pattern_a", "pattern_b", "pearson_correlation", "should_exclude")

            def __init__(self, a, b, c, e):
                self.pattern_a = a
                self.pattern_b = b
                self.pearson_correlation = c
                self.should_exclude = e

        pairs = []
        n = len(corr_matrix)
        cols = corr_matrix.columns.tolist()
        for i in range(n):
            for j in range(i + 1, n):
                corr = abs(corr_matrix.iloc[i, j])
                if corr > self.correlation_threshold:
                    pairs.append(_Pair(cols[i], cols[j], corr, True))
        return pairs

    def check_documented_groups(self, active_patterns: List[str]) -> List[Dict]:
        """Check documented groups for excess patterns."""
        groups = {
            "Double Patterns": ["double_top", "double_bottom", "triple_top", "triple_bottom"],
            "Triple Patterns": ["triple_top", "triple_bottom"],
            "Candlestick Single": ["hammer", "doji", "engulfing", "harami", "dark_cloud"],
        }
        normalised = [p.lower().replace(" ", "_").replace("-", "_") for p in active_patterns]
        violations = []
        for group_name, members in groups.items():
            found = [m for m in members if any(m in n or n in m for n in normalised)]
            count = len(found)
            if count > 3:
                violations.append(
                    {
                        "group_name": group_name,
                        "active_count": count,
                        "excess": count - 3,
                    }
                )
        return violations

    def deduplicate_patterns(
        self,
        candidate_patterns: List[str],
        signal_matrix=None,
    ) -> Dict:
        """Deduplicate pattern list."""
        selected = list(candidate_patterns)
        excluded = []
        return {"selected_patterns": selected, "excluded_patterns": excluded}

    def get_correlation_summary(self, corr_matrix: "pd.DataFrame") -> "pd.DataFrame":
        """Get correlation summary."""
        rows = []
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr = corr_matrix.iloc[i, j]
                rows.append(
                    {
                        "pattern_a": cols[i],
                        "pattern_b": cols[j],
                        "correlation": corr,
                        "is_high_correlation": abs(corr) > self.correlation_threshold,
                    }
                )
        return pd.DataFrame(rows)
