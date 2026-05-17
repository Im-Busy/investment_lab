"""
R5: Collinearity Analysis for Pattern Overlap.

Computes VIF (Variance Inflation Factor) matrix across all pattern signal
series. Flags redundant patterns (VIF > 5) and provides synthesis/discard
recommendations.

Rules:
- Within-category correlation → recommend IR-weight synthesis
- Across-category correlation → recommend discard weaker (lower IC)
- VIF = 1 / (1 - R²_i) where R²_i is regression of pattern_i on all others

Source: 华泰多因子 §2.1-2.2

Architecture: consumed by pattern pruning analysis, RulesFirstStrategy init.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)

# VIF threshold for flagging redundancy
DEFAULT_VIF_THRESHOLD = 5.0
# Minimum bars required for VIF computation
MIN_BARS_FOR_VIF = 20


@dataclass
class RedundancyPair:
    """A pair of redundant patterns identified by collinearity analysis.

    Attributes:
        pattern_a: Name of first pattern.
        pattern_b: Name of second pattern.
        correlation: Pearson correlation between the two patterns.
        vif_a: VIF of pattern_a when both are present.
        vif_b: VIF of pattern_b when both are present.
        category_a: Category of pattern_a (if known).
        category_b: Category of pattern_b (if known).
        recommendation: 'synthesize' (same category) or 'discard' (different).
        discard_candidate: Which pattern to discard (lower IC / higher VIF).
        reason: Human-readable explanation.
    """

    pattern_a: str
    pattern_b: str
    correlation: float
    vif_a: float
    vif_b: float
    category_a: str = ""
    category_b: str = ""
    recommendation: str = ""
    discard_candidate: str = ""
    reason: str = ""


@dataclass
class CollinearityReport:
    """Full collinearity analysis result.

    Attributes:
        vif_df: DataFrame of VIF values per pattern.
        corr_matrix: Pearson correlation matrix (DataFrame).
        redundant_pairs: List of RedundancyPair flagged as redundant.
        n_patterns: Total number of patterns analyzed.
        n_redundant: Number of patterns with VIF > threshold.
        n_discard: Number of patterns recommended for discard.
        n_synthesize: Number of patterns recommended for synthesis.
    """

    vif_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    corr_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    redundant_pairs: List[RedundancyPair] = field(default_factory=list)
    n_patterns: int = 0
    n_redundant: int = 0
    n_discard: int = 0
    n_synthesize: int = 0

    def summary(self) -> str:
        """One-paragraph text summary."""
        lines = [
            f"Collinearity Analysis: {self.n_patterns} patterns, "
            f"{self.n_redundant} redundant (VIF > {DEFAULT_VIF_THRESHOLD})",
            f"  Recommended: discard {self.n_discard}, synthesize {self.n_synthesize}",
        ]
        if self.redundant_pairs:
            for pair in self.redundant_pairs[:10]:
                lines.append(
                    f"  {pair.pattern_a} ↔ {pair.pattern_b} "
                    f"(r={pair.correlation:.3f}, VIFa={pair.vif_a:.1f}, VIFb={pair.vif_b:.1f}) "
                    f"→ {pair.recommendation}"
                )
        return "\n".join(lines)


def analyze_collinearity(
    signals_df: pd.DataFrame,
    categories: Optional[Dict[str, str]] = None,
    ic_scores: Optional[Dict[str, float]] = None,
    vif_threshold: float = DEFAULT_VIF_THRESHOLD,
) -> CollinearityReport:
    """Compute VIF matrix and identify redundant patterns.

    Args:
        signals_df: DataFrame with M columns (pattern names) and N rows (bars).
            Values should be signal outputs (-1/0/1 or continuous).
            NaN values are dropped row-wise.
        categories: Dict mapping pattern_name → category (e.g., 'trend', 'reversal').
            Used for synthesis vs discard recommendation.
        ic_scores: Dict mapping pattern_name → IC score.
            Lower IC patterns are preferred for discard.
        vif_threshold: VIF above which a pattern is considered redundant.

    Returns:
        CollinearityReport with VIF matrix, correlation matrix, and recommendations.
    """
    if categories is None:
        categories = {}
    if ic_scores is None:
        ic_scores = {}

    # Drop rows with NaN
    clean = signals_df.dropna()
    if len(clean) < MIN_BARS_FOR_VIF:
        logger.warning(
            "Insufficient data for VIF: %d clean bars < %d required",
            len(clean),
            MIN_BARS_FOR_VIF,
        )
        return CollinearityReport(n_patterns=len(signals_df.columns))

    patterns = list(clean.columns)
    n_patterns = len(patterns)
    n_bars = len(clean)

    # Correlation matrix
    corr_matrix = clean.corr()

    # Add constant for VIF (statsmodels requires constant)
    X = clean.values
    X_with_const = np.column_stack([np.ones(n_bars), X])

    # Compute VIF for each pattern (columns 1..M in X_with_const)
    vif_values = []
    valid_patterns = []
    for j in range(n_patterns):
        try:
            vif = variance_inflation_factor(X_with_const, j + 1)  # +1 for const
            if np.isfinite(vif) and vif > 0:
                vif_values.append(vif)
                valid_patterns.append(patterns[j])
            else:
                logger.debug("Non-finite VIF for %s, skipping", patterns[j])
        except Exception:
            logger.debug("VIF computation failed for %s", patterns[j])

    if not vif_values:
        return CollinearityReport(
            vif_df=pd.DataFrame(),
            corr_matrix=corr_matrix,
            n_patterns=n_patterns,
        )

    vif_df = pd.DataFrame({"pattern": valid_patterns, "vif": vif_values}).sort_values(
        "vif", ascending=False
    )
    vif_df = vif_df.set_index("pattern")

    # Identify redundant pairs
    redundant_pairs: List[RedundancyPair] = []
    redundant_patterns = set(vif_df[vif_df["vif"] > vif_threshold].index.tolist())

    # Build pairs from high-VIF patterns
    corr_values = corr_matrix.values
    pattern_idx = {p: i for i, p in enumerate(patterns)}

    for i, pa in enumerate(patterns):
        if pa not in redundant_patterns and pa not in valid_patterns:
            continue
        vif_a = vif_df.loc[pa, "vif"] if pa in vif_df.index else 0.0

        for j, pb in enumerate(patterns):
            if j <= i:
                continue
            vif_b = vif_df.loc[pb, "vif"] if pb in vif_df.index else 0.0

            # Flag if either has high VIF and correlation is substantial
            if (pa in redundant_patterns or pb in redundant_patterns) and abs(
                corr_values[i, j]
            ) > 0.5:
                pair = _build_redundancy_pair(
                    pa,
                    pb,
                    corr_values[i, j],
                    vif_a,
                    vif_b,
                    categories,
                    ic_scores,
                )
                redundant_pairs.append(pair)

    # Count recommendations
    n_discard = sum(1 for p in redundant_pairs if p.recommendation == "discard")
    n_synthesize = sum(1 for p in redundant_pairs if p.recommendation == "synthesize")

    report = CollinearityReport(
        vif_df=vif_df,
        corr_matrix=corr_matrix,
        redundant_pairs=redundant_pairs,
        n_patterns=n_patterns,
        n_redundant=len(redundant_patterns),
        n_discard=n_discard,
        n_synthesize=n_synthesize,
    )

    return report


def _build_redundancy_pair(
    pa: str,
    pb: str,
    corr: float,
    vif_a: float,
    vif_b: float,
    categories: Dict[str, str],
    ic_scores: Dict[str, float],
) -> RedundancyPair:
    """Build a RedundancyPair with recommendation logic."""
    cat_a = categories.get(pa, "")
    cat_b = categories.get(pb, "")
    same_category = bool(cat_a and cat_b and cat_a == cat_b)

    if same_category:
        recommendation = "synthesize"
        discard_candidate = ""
        reason = f"Same category ({cat_a}) — recommend IR-weight synthesis"
    else:
        recommendation = "discard"
        ic_a = ic_scores.get(pa, 0.0)
        ic_b = ic_scores.get(pb, 0.0)
        if ic_a >= ic_b:
            discard_candidate = pb
            reason = (
                f"Different categories ({cat_a} vs {cat_b}) — "
                f"discard {pb} (IC={ic_b:.4f}) < {pa} (IC={ic_a:.4f})"
            )
        else:
            discard_candidate = pa
            reason = (
                f"Different categories ({cat_a} vs {cat_b}) — "
                f"discard {pa} (IC={ic_a:.4f}) < {pb} (IC={ic_b:.4f})"
            )

    return RedundancyPair(
        pattern_a=pa,
        pattern_b=pb,
        correlation=float(corr),
        vif_a=float(vif_a),
        vif_b=float(vif_b),
        category_a=cat_a,
        category_b=cat_b,
        recommendation=recommendation,
        discard_candidate=discard_candidate,
        reason=reason,
    )


def analyze_pattern_overlap(
    signals: Dict[str, np.ndarray],
    categories: Optional[Dict[str, str]] = None,
    ic_scores: Optional[Dict[str, float]] = None,
    vif_threshold: float = DEFAULT_VIF_THRESHOLD,
) -> CollinearityReport:
    """Convenience: analyze collinearity from dict of signal arrays.

    Args:
        signals: Dict of pattern_name → (N,) int8 signal array.
        categories: Pattern → category mapping.
        ic_scores: Pattern → IC score for discard ranking.
        vif_threshold: VIF threshold for redundancy.

    Returns:
        CollinearityReport.
    """
    if not signals:
        return CollinearityReport()

    min_len = min(len(arr) for arr in signals.values())
    data: Dict[str, np.ndarray] = {}
    for name, arr in signals.items():
        data[name] = np.asarray(arr[:min_len], dtype=float)

    df = pd.DataFrame(data)
    return analyze_collinearity(df, categories, ic_scores, vif_threshold)


def get_discard_recommendations(
    report: CollinearityReport,
) -> List[str]:
    """Extract list of pattern names recommended for discard.

    Args:
        report: CollinearityReport from analyze_collinearity().

    Returns:
        Sorted list of pattern names to discard.
    """
    discards = set()
    for pair in report.redundant_pairs:
        if pair.recommendation == "discard" and pair.discard_candidate:
            discards.add(pair.discard_candidate)
    return sorted(discards)


def get_synthesize_recommendations(
    report: CollinearityReport,
) -> List[Tuple[str, str]]:
    """Extract list of pattern pairs recommended for IR-weight synthesis.

    Args:
        report: CollinearityReport from analyze_collinearity().

    Returns:
        List of (pattern_a, pattern_b) tuples to synthesize.
    """
    synthesizes = []
    for pair in report.redundant_pairs:
        if pair.recommendation == "synthesize":
            synthesizes.append((pair.pattern_a, pair.pattern_b))
    return synthesizes
