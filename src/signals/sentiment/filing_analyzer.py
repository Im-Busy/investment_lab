"""SEC filing sentiment and text analysis.

Analyzes 10-K/10-Q MD&A sections for:
- Year-over-year text similarity (cosine similarity using TF-IDF)
- Tone change detection (sentiment shift from FinBERT or LM dictionary)
- Uncertainty ratio trends (is management getting more vague?)
- Emerging keyword detection (TF-IDF: new words vs. corpus)

Usage:
    analyzer = FilingAnalyzer()
    results = analyzer.analyze_filings(["10-K text 2023", "10-K text 2024"])
    print(results["tone_change"], results["yoy_similarity"])
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class FilingAnalyzer:
    """Analyze SEC filings for NLP-derived features."""

    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None

    def yoy_text_similarity(
        self,
        texts: list[str],
        years: list[int] | None = None,
    ) -> dict[str, float]:
        """Compute year-over-year cosine similarity of filing text.

        High similarity = management language unchanged ("boilerplate").
        Low similarity = management language shifted (potential signal).

        Args:
            texts: List of filing texts in chronological order.
            years: Optional list of years (for display).

        Returns:
            Dict with: mean_similarity, min_similarity, latest_similarity,
            and per-pair similarity values.
        """
        if len(texts) < 2:
            return {
                "mean_similarity": 1.0,
                "min_similarity": 1.0,
                "latest_similarity": 1.0,
                "pairs": {},
            }

        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarities = []
        pairs = {}

        for i in range(1, len(texts)):
            sim = float(cosine_similarity(tfidf_matrix[i - 1 : i], tfidf_matrix[i : i + 1])[0, 0])
            similarities.append(sim)
            label = f"{years[i - 1]}_vs_{years[i]}" if years else f"pair_{i}"
            pairs[label] = round(sim, 4)

        return {
            "mean_similarity": round(np.mean(similarities), 4),
            "min_similarity": round(np.min(similarities), 4),
            "latest_similarity": round(similarities[-1], 4),
            "pairs": pairs,
        }

    def tone_change(
        self,
        texts: list[str],
        scorer: Any | None = None,
    ) -> pd.DataFrame:
        """Detect sentiment tone changes across filings.

        Uses either LM dictionary or FinBERT for sentiment scoring.
        Returns per-filing sentiment metrics.

        Args:
            texts: List of filing texts in chronological order.
            scorer: Optional pre-loaded sentiment scorer. If None,
                uses LM dictionary.

        Returns:
            DataFrame with columns: sentiment, uncertainty_ratio,
            litigious_ratio, positive_count, negative_count, total_words.
        """
        if scorer is None:
            from src.signals.sentiment.dictionary import LMSentimentScorer

            scorer = LMSentimentScorer()

        rows = []
        for i, text in enumerate(texts):
            try:
                scores = scorer.score_text(text)
                rows.append({**scores, "filing_idx": i})
            except Exception as e:
                logger.debug("Scoring failed for filing %d: %s", i, e)
                rows.append(
                    {
                        "sentiment": 0.0,
                        "uncertainty_ratio": 0.0,
                        "litigious_ratio": 0.0,
                        "positive_count": 0,
                        "negative_count": 0,
                        "uncertainty_count": 0,
                        "litigious_count": 0,
                        "total_words": 0,
                        "filing_idx": i,
                    }
                )

        df = pd.DataFrame(rows)

        if len(df) > 1:
            df["sentiment_change"] = df["sentiment"].diff().round(4)
            df["uncertainty_change"] = df["uncertainty_ratio"].diff().round(4)

        return df

    def uncertainty_trend(
        self,
        tone_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Analyze uncertainty ratio trends across filings.

        Increasing uncertainty = management becoming more vague/uncertain.
        This is a well-documented signal in accounting research.

        Args:
            tone_df: DataFrame from tone_change().

        Returns:
            Dict with: trend_direction, slope, mean, growth_pct.
        """
        if len(tone_df) < 2:
            return {"trend_direction": "insufficient_data", "slope": 0.0}

        ratios = tone_df["uncertainty_ratio"].values
        x = np.arange(len(ratios))

        # Simple linear trend
        slope = float(np.polyfit(x, ratios, 1)[0])
        growth = (
            float((ratios[-1] - ratios[0]) / max(abs(ratios[0]), 1e-10) * 100)
            if len(ratios) > 1
            else 0.0
        )

        if slope > 0.001:
            direction = "increasing"
        elif slope < -0.001:
            direction = "decreasing"
        else:
            direction = "stable"

        return {
            "trend_direction": direction,
            "slope": round(slope, 6),
            "mean_uncertainty": round(float(np.mean(ratios)), 4),
            "latest_uncertainty": round(float(ratios[-1]), 4),
            "growth_pct": round(growth, 2),
        }

    def emerging_keywords(
        self,
        texts: list[str],
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """Detect keywords newly emerging in latest filing vs. historical corpus.

        Uses TF-IDF to find words that are distinctive in the latest filing
        compared to previous filings. These may signal new risks or focus areas.

        Args:
            texts: List of filing texts. Last element is the latest.
            top_n: Number of keywords to return.

        Returns:
            List of dicts with keys: word, tfidf_score, prevalence_ratio.
        """
        if len(texts) < 2:
            return []

        vectorizer = TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # TF-IDF scores for latest filing vs average of prior filings
        latest_tfidf = tfidf_matrix[-1].toarray().flatten()
        prior_tfidf = tfidf_matrix[:-1].mean(axis=0)
        if hasattr(prior_tfidf, "A1"):
            prior_tfidf = prior_tfidf.A1

        # Score = difference weighted by magnitude in latest filing
        scores = latest_tfidf * (latest_tfidf - prior_tfidf)

        top_indices = np.argsort(scores)[-top_n:][::-1]

        keywords = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            prevalence = (
                float(latest_tfidf[idx] / max(prior_tfidf[idx], 1e-10))
                if prior_tfidf[idx] > 0
                else float("inf")
            )
            keywords.append(
                {
                    "word": str(feature_names[idx]),
                    "tfidf_score": round(float(latest_tfidf[idx]), 4),
                    "prevalence_ratio": (
                        round(prevalence, 2) if prevalence != float("inf") else None
                    ),
                }
            )

        return keywords

    def filing_features(
        self,
        texts: list[str],
        years: list[int] | None = None,
    ) -> dict[str, Any]:
        """Extract all NLP features from a series of filings.

        Returns a single dict of features for CatBoost integration:
            - sentiment_last
            - sentiment_change
            - uncertainty_last
            - uncertainty_trend (increasing/decreasing/stable)
            - yoy_similarity_last
            - emerging_keyword_count
            - text_drift_score (1 - yoy_similarity)
        """
        if not texts:
            return {}

        tone = self.tone_change(texts)
        similarity = self.yoy_text_similarity(texts, years)
        uncertainty = self.uncertainty_trend(tone)
        keywords = self.emerging_keywords(texts)

        return {
            "sentiment_last": round(float(tone["sentiment"].iloc[-1]), 4) if len(tone) > 0 else 0.0,
            "sentiment_change": round(float(tone["sentiment_change"].iloc[-1]), 4)
            if len(tone) > 1 and "sentiment_change" in tone
            else 0.0,
            "uncertainty_last": uncertainty.get("latest_uncertainty", 0.0),
            "uncertainty_trend": uncertainty.get("trend_direction", "insufficient_data"),
            "uncertainty_slope": uncertainty.get("slope", 0.0),
            "yoy_similarity_last": similarity.get("latest_similarity", 1.0),
            "text_drift_score": round(1.0 - similarity.get("latest_similarity", 1.0), 4),
            "emerging_keyword_count": len(keywords),
            "top_emerging_keywords": [k["word"] for k in keywords[:3]],
        }
