"""Multi-source sentiment fusion.

Combines multiple sentiment sources into a single trading signal:
- LM dictionary (Phase N): keyword-counting financial-domain sentiment
- FinBERT (Phase R1): contextual deep-learning financial sentiment
- SEC filing tone (Phase R2): MD&A text analysis trends
- Social media sentiment (Phase O): only if gate passed

Weights are initialized from the plan and can be rebalanced by rolling
predictive power (IC-based). Only trades when >= 2 sources agree on direction.

Usage:
    fusion = MultiSourceFusion()
    signal = fusion.fuse(
        lm_score=0.15,
        finbert_score=0.30,
        filing_tone=-0.05,
    )
    # signal in [-1, 1]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "lm_dictionary": 0.30,
    "finbert": 0.30,
    "sec_filing_tone": 0.20,
    "social_media": 0.20,
}

SOURCE_NAMES = list(DEFAULT_WEIGHTS.keys())


@dataclass
class SentimentSource:
    """Single sentiment source with current value and metadata."""

    name: str
    score: float
    weight: float
    confidence: float = 1.0
    active: bool = True


@dataclass
class FusionResult:
    """Result of multi-source sentiment fusion."""

    fused_score: float
    direction: str  # "bullish", "bearish", "neutral"
    agreement_count: int
    total_active: int
    sources: dict[str, float]
    weights: dict[str, float]
    trade_signal: bool  # True if >= 2 sources agree on non-neutral direction


class MultiSourceFusion:
    """Fuse multiple sentiment sources into a single trading signal.

    Only generates a trade signal when >= min_agreement sources agree
    on direction (positive or negative). This reduces noise and ensures
    the signal has multi-source confirmation.

    Attributes:
        weights: Dict mapping source name to weight.
        min_agreement: Minimum number of sources that must agree on direction.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        min_agreement: int = 2,
    ) -> None:
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.min_agreement = min_agreement

    def fuse(
        self,
        lm_score: float = 0.0,
        finbert_score: float = 0.0,
        filing_tone: float = 0.0,
        social_media_score: float = 0.0,
        **kwargs: float,
    ) -> FusionResult:
        """Fuse multiple sentiment scores into one signal.

        Args:
            lm_score: LM dictionary sentiment in [-1, 1].
            finbert_score: FinBERT sentiment in [-1, 1].
            filing_tone: SEC filing tone change score (normalized to [-1, 1]).
            social_media_score: Social media sentiment in [-1, 1].

        Returns:
            FusionResult with fused score, direction, agreement count, etc.
        """
        scores = {
            "lm_dictionary": lm_score,
            "finbert": finbert_score,
            "sec_filing_tone": filing_tone,
            "social_media": social_media_score,
        }

        sources = []
        weighted_sum = 0.0
        active_weight = 0.0
        bullish = 0
        bearish = 0

        for name in SOURCE_NAMES:
            score = scores.get(name, 0.0)
            weight = self.weights.get(name, 0.0)

            if weight <= 0 or np.isnan(score):
                sources.append(
                    SentimentSource(
                        name=name,
                        score=score,
                        weight=weight,
                        active=False,
                    )
                )
                continue

            sources.append(
                SentimentSource(
                    name=name,
                    score=score,
                    weight=weight,
                    active=True,
                )
            )
            weighted_sum += score * weight
            active_weight += weight

            if score > 0.01:
                bullish += 1
            elif score < -0.01:
                bearish += 1

        if active_weight > 0:
            fused_score = weighted_sum / active_weight
        else:
            fused_score = 0.0

        # Direction
        if fused_score > 0.02:
            direction = "bullish"
        elif fused_score < -0.02:
            direction = "bearish"
        else:
            direction = "neutral"

        agreement_count = max(bullish, bearish)
        active_sources = sum(1 for s in sources if s.active)
        trade_signal = direction != "neutral" and agreement_count >= self.min_agreement

        return FusionResult(
            fused_score=round(fused_score, 4),
            direction=direction,
            agreement_count=agreement_count,
            total_active=active_sources,
            sources={s.name: s.score for s in sources if s.active},
            weights={s.name: s.weight for s in sources if s.active},
            trade_signal=trade_signal,
        )

    def fuse_series(
        self,
        lm_series: pd.Series | None = None,
        finbert_series: pd.Series | None = None,
        filing_series: pd.Series | None = None,
        social_series: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Fuse time-series of sentiment scores.

        Args:
            lm_series: Time-series of LM dictionary scores.
            finbert_series: Time-series of FinBERT scores.
            filing_series: Time-series of filing tone scores.
            social_series: Time-series of social media scores.

        Returns:
            DataFrame with columns: fused_score, direction, trade_signal,
            agreement_count, per-source scores.
        """
        # Find common index
        all_indices = []
        for s in [lm_series, finbert_series, filing_series, social_series]:
            if s is not None:
                all_indices.extend(s.index.tolist())
        if not all_indices:
            return pd.DataFrame()

        common_idx = pd.DatetimeIndex(sorted(set(all_indices)))
        results = []

        for dt in common_idx:
            r = self.fuse(
                lm_score=float(lm_series.loc[dt])
                if lm_series is not None and dt in lm_series.index
                else 0.0,
                finbert_score=float(finbert_series.loc[dt])
                if finbert_series is not None and dt in finbert_series.index
                else 0.0,
                filing_tone=float(filing_series.loc[dt])
                if filing_series is not None and dt in filing_series.index
                else 0.0,
                social_media_score=float(social_series.loc[dt])
                if social_series is not None and dt in social_series.index
                else 0.0,
            )
            results.append(
                {
                    "date": dt,
                    "fused_score": r.fused_score,
                    "direction": r.direction,
                    "trade_signal": r.trade_signal,
                    "agreement_count": r.agreement_count,
                    "total_active": r.total_active,
                    **r.sources,
                }
            )

        df = pd.DataFrame(results).set_index("date")
        return df

    def rebalance_weights(
        self,
        ic_by_source: dict[str, float],
        min_weight: float = 0.05,
    ) -> dict[str, float]:
        """Rebalance source weights based on Information Coefficient.

        Sources with positive IC get higher weight. Negative IC sources
        get minimum weight. Sources with zero/NaN IC keep current weight.

        Args:
            ic_by_source: Dict mapping source name to IC value.
            min_weight: Minimum weight for any source.

        Returns:
            Updated weights dict.
        """
        new_weights = {}
        positive_ics = {}

        for name, ic in ic_by_source.items():
            if np.isnan(ic):
                new_weights[name] = self.weights.get(name, min_weight)
            elif ic > 0:
                positive_ics[name] = ic
            else:
                new_weights[name] = min_weight

        if positive_ics:
            total_ic = sum(positive_ics.values())
            remaining = 1.0 - sum(new_weights.values())
            for name, ic in positive_ics.items():
                new_weights[name] = remaining * (ic / total_ic)

        self.weights = new_weights
        return new_weights

    def summarize(self, result: FusionResult) -> str:
        """Human-readable summary of fusion result."""
        source_str = ", ".join(
            f"{name}={score:+.3f}(w={self.weights.get(name, 0):.1f})"
            for name, score in result.sources.items()
        )
        return (
            f"Fusion: {result.fused_score:+.3f} [{result.direction}] "
            f"agreement={result.agreement_count}/{result.total_active} "
            f"signal={'YES' if result.trade_signal else 'NO'} "
            f"| {source_str}"
        )
