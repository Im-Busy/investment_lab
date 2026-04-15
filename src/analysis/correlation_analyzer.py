"""
Correlation Analyzer for Pattern Selection

Measures pairwise correlation between pattern signals to identify
and remove redundant patterns. Uses both documented correlation
groups and empirical signal matrix analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


@dataclass
class CorrelationGroup:
    """A group of patterns that should not all be active simultaneously."""

    name: str
    patterns: List[str]
    max_active: int = 1


@dataclass
class CorrelationResult:
    """Result of correlation analysis for a pattern pair."""

    pattern_a: str
    pattern_b: str
    pearson_correlation: float
    co_occurrence_count: int
    co_occurrence_pct: float

    # Recommendation
    should_exclude: bool
    exclude_reason: str = ""
    keep_pattern: str = ""  # Which pattern to keep if excluding


@dataclass
class CorrelationAnalyzerConfig:
    """Configuration for correlation analysis."""

    # Correlation threshold for redundancy detection
    max_correlation: float = 0.7

    # Minimum co-occurrences for reliable correlation
    min_co_occurrences: int = 5

    # Use documented groups
    use_documented_groups: bool = True


# Default documented correlation groups from trading strategy
DEFAULT_CORRELATION_GROUPS = [
    CorrelationGroup(
        name="Double Patterns",
        patterns=["Double Top", "Double Bottom", "Triple Top", "Triple Bottom"],
        max_active=1,
    ),
    CorrelationGroup(
        name="Harmonic",
        patterns=["Gartley Pattern", "ABC Pattern"],
        max_active=1,
    ),
    CorrelationGroup(
        name="Breakout",
        patterns=["NR7ID", "Donchian Channel Breakout", "Bollinger Bands"],
        max_active=2,
    ),
    CorrelationGroup(
        name="Reversal Tops",
        patterns=["Head and Shoulders", "Double Top", "Trader Vic 2B"],
        max_active=1,
    ),
    CorrelationGroup(
        name="Reversal Bottoms",
        patterns=["Double Bottom", "Market Structure Low", "Matching Lows"],
        max_active=1,
    ),
]


class CorrelationAnalyzer:
    """
    Analyzes pattern correlation and deduplicates redundant patterns.

    Combines documented correlation groups (based on pattern design)
    with empirical correlation measurement (based on actual signal
    co-occurrence on the same bars).
    """

    def __init__(
        self,
        config: Optional[CorrelationAnalyzerConfig] = None,
        correlation_groups: Optional[List[CorrelationGroup]] = None,
    ):
        self.config = config or CorrelationAnalyzerConfig()
        self.correlation_groups = correlation_groups or DEFAULT_CORRELATION_GROUPS

    def build_signal_matrix(
        self,
        detection_log: pd.DataFrame,
        total_bars: int,
    ) -> pd.DataFrame:
        """
        Build a binary signal matrix from pattern detection events.

        Args:
            detection_log: DataFrame with columns:
                - bar_index, pattern_name, passed_threshold (bool)
            total_bars: Total number of bars in the dataset

        Returns:
            Binary matrix: rows = bars, columns = patterns, values = 0/1
        """
        pattern_names = detection_log["pattern_name"].unique()

        matrix = pd.DataFrame(
            0,
            index=range(total_bars),
            columns=pattern_names,
            dtype=np.int32,
        )

        # Set 1 where pattern was detected and passed threshold
        detections = detection_log[detection_log.get("passed_threshold", True) == True]
        for _, row in detections.iterrows():
            bar_idx = int(row["bar_index"])
            if 0 <= bar_idx < total_bars:
                matrix.loc[bar_idx, row["pattern_name"]] = 1

        return matrix

    def compute_correlation_matrix(
        self,
        signal_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute pairwise Pearson correlation between patterns.

        Args:
            signal_matrix: Binary signal matrix from build_signal_matrix()

        Returns:
            Correlation matrix (patterns x patterns)
        """
        return signal_matrix.corr(method="pearson")

    def compute_co_occurrence_matrix(
        self,
        signal_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute co-occurrence count matrix.

        Args:
            signal_matrix: Binary signal matrix

        Returns:
            Co-occurrence count matrix
        """
        # Co-occurrence = dot product of binary vectors
        return signal_matrix.T @ signal_matrix

    def find_redundant_pairs(
        self,
        correlation_matrix: pd.DataFrame,
        co_occurrence_matrix: pd.DataFrame,
        pattern_performance: Optional[Dict[str, float]] = None,
    ) -> List[CorrelationResult]:
        """
        Find pattern pairs that are redundant (correlation above threshold).

        Args:
            correlation_matrix: Pairwise correlation matrix
            co_occurrence_matrix: Co-occurrence count matrix
            pattern_performance: Dict[pattern_name -> sharpe_ratio] for tie-breaking

        Returns:
            List of CorrelationResult for redundant pairs
        """
        redundant = []
        patterns = correlation_matrix.columns.tolist()
        n = len(patterns)

        for i in range(n):
            for j in range(i + 1, n):
                pat_a = patterns[i]
                pat_b = patterns[j]
                corr = correlation_matrix.iloc[i, j]

                if pd.isna(corr):
                    continue

                co_count = int(co_occurrence_matrix.iloc[i, j])
                total_a = int(co_occurrence_matrix.iloc[i, i])
                co_pct = co_count / total_a if total_a > 0 else 0.0

                # Check if redundant
                is_redundant = (
                    corr >= self.config.max_correlation
                    and co_count >= self.config.min_co_occurrences
                )

                if is_redundant:
                    # Decide which to keep
                    if pattern_performance:
                        perf_a = pattern_performance.get(pat_a, 0.0)
                        perf_b = pattern_performance.get(pat_b, 0.0)
                        keep = pat_a if perf_a >= perf_b else pat_b
                        exclude = pat_b if keep == pat_a else pat_a
                        reason = f"Correlation {corr:.2f} >= {self.config.max_correlation}; {keep} has better performance"
                    else:
                        keep = pat_a
                        exclude = pat_b
                        reason = f"Correlation {corr:.2f} >= {self.config.max_correlation}; defaulting to keep {pat_a}"

                    redundant.append(
                        CorrelationResult(
                            pattern_a=pat_a,
                            pattern_b=pat_b,
                            pearson_correlation=corr,
                            co_occurrence_count=co_count,
                            co_occurrence_pct=co_pct,
                            should_exclude=True,
                            exclude_reason=reason,
                            keep_pattern=keep,
                        )
                    )

        return redundant

    def check_documented_groups(
        self,
        active_patterns: List[str],
        groups: Optional[List[CorrelationGroup]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Check if active patterns violate documented correlation group rules.

        Args:
            active_patterns: Currently active/selected patterns
            groups: Correlation groups to check against

        Returns:
            List of violations with recommendations
        """
        groups = groups or self.correlation_groups
        violations = []
        active_set = set(active_patterns)

        for group in groups:
            active_in_group = [p for p in group.patterns if p in active_set]

            if len(active_in_group) > group.max_active:
                violations.append(
                    {
                        "group_name": group.name,
                        "group_patterns": group.patterns,
                        "active_patterns": active_in_group,
                        "max_allowed": group.max_active,
                        "excess": len(active_in_group) - group.max_active,
                        "recommendation": f"Keep at most {group.max_active} from: {', '.join(active_in_group)}",
                    }
                )

        return violations

    def deduplicate_patterns(
        self,
        candidate_patterns: List[str],
        signal_matrix: Optional[pd.DataFrame] = None,
        pattern_performance: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Full deduplication pipeline: empirical + documented checks.

        Args:
            candidate_patterns: List of patterns to deduplicate
            signal_matrix: Optional binary signal matrix for empirical analysis
            pattern_performance: Optional Dict[pattern -> sharpe] for tie-breaking

        Returns:
            Dict with:
                - selected_patterns: Final deduplicated list
                - excluded_patterns: Removed patterns with reasons
                - group_violations: Documented group violations
        """
        excluded = {}

        if signal_matrix is not None:
            # Empirical deduplication
            corr_matrix = self.compute_correlation_matrix(signal_matrix)
            co_matrix = self.compute_co_occurrence_matrix(signal_matrix)

            redundant = self.find_redundant_pairs(corr_matrix, co_matrix, pattern_performance)

            for result in redundant:
                if result.should_exclude and result.exclude_reason:
                    excluded[result.exclude_reason] = result.exclude_reason

        # Documented group check
        if self.config.use_documented_groups:
            violations = self.check_documented_groups(candidate_patterns)

            for violation in violations:
                # Remove excess patterns from the group
                active = violation["active_patterns"]
                excess = violation["excess"]

                for i in range(excess):
                    pattern_to_remove = active[-(i + 1)]
                    reason = (
                        f"Group '{violation['group_name']}': max {violation['max_allowed']} allowed"
                    )
                    excluded[pattern_to_remove] = reason

        # Filter out excluded patterns
        selected = [p for p in candidate_patterns if p not in excluded]

        return {
            "selected_patterns": selected,
            "excluded_patterns": excluded,
            "group_violations": violations if self.config.use_documented_groups else [],
        }

    def get_correlation_summary(
        self,
        correlation_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Get summary of pattern correlations.

        Args:
            correlation_matrix: Pairwise correlation matrix

        Returns:
            DataFrame with all unique pattern pairs and their correlations
        """
        pairs = []
        patterns = correlation_matrix.columns.tolist()
        n = len(patterns)

        for i in range(n):
            for j in range(i + 1, n):
                corr = correlation_matrix.iloc[i, j]
                if not pd.isna(corr):
                    high_corr = abs(corr) >= self.config.max_correlation
                    pairs.append(
                        {
                            "pattern_a": patterns[i],
                            "pattern_b": patterns[j],
                            "correlation": corr,
                            "is_high_correlation": high_corr,
                        }
                    )

        df = pd.DataFrame(pairs)
        if len(df) > 0:
            df = df.sort_values("correlation", key=abs, ascending=False)

        return df
