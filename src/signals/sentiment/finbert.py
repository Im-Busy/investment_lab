"""FinBERT financial sentiment model (ProsusAI/finbert).

ProsusAI/finbert: BERT fine-tuned on financial text (TRC2-financial + FiQA).
Outputs: positive, negative, neutral probabilities per sentence.

The FinBERT model provides deeper financial-domain understanding than the
dictionary-based LM scorer. While LM counts keywords, FinBERT understands
context (e.g., "profit declined" vs "decline in losses").

Implements the SentimentProvider Protocol from sentiment_scorer.py so it
can be plugged into the existing sentiment pipeline.

Reference:
    Huang, Wang, Yang (2023). "FinBERT: A Large Language Model for Extracting
    Information from Financial Text." Contemporary Accounting Research.

Usage:
    model = FinBERTSentiment(model_name="ProsusAI/finbert")
    scores = model.predict(["Revenue grew 20% YoY", "Company faces headwinds"])
    # Returns: [[0.85, 0.05, 0.10], [0.02, 0.92, 0.06]]  # [pos, neg, neutral]
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FINBERT_MODEL_NAME = "ProsusAI/finbert"
FINBERT_CACHE_DIR = os.environ.get(
    "FINBERT_CACHE_DIR",
    str(Path("models/finbert_cache")),
)


class FinBERTSentiment:
    """FinBERT financial-domain sentiment analysis.

    Uses ProsusAI/finbert for professional financial sentiment classification.
    The model is loaded lazily on first use to avoid startup overhead.

    Attributes:
        model_name: HuggingFace model identifier.
        device: "cpu" or "cuda" — auto-detected.
    """

    def __init__(
        self,
        model_name: str = FINBERT_MODEL_NAME,
        device: str | None = None,
        cache_dir: str = FINBERT_CACHE_DIR,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self.cache_dir = cache_dir
        self._pipeline: Any = None
        self._loaded: bool = False

    def _load(self) -> None:
        """Lazy-load the FinBERT pipeline on first use."""
        if self._loaded:
            return

        import torch
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            "Loading FinBERT model %s on %s",
            self.model_name,
            self._device,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )

        self._pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1 if self._device == "cpu" else 0,
            return_all_scores=True,
        )
        self._loaded = True
        logger.info("FinBERT model loaded successfully")

    @property
    def device(self) -> str:
        return self._device or "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, texts: list[str]) -> pd.DataFrame:
        """Predict sentiment probabilities for one or more texts.

        Args:
            texts: List of text strings to analyze.

        Returns:
            DataFrame with columns: positive, negative, neutral, sentiment_score.
            sentiment_score = positive - negative in [-1, 1].
            Index matches input order.
        """
        self._load()
        results = self._pipeline(texts, truncation=True, max_length=512)

        positive = []
        negative = []
        neutral = []
        for output in results:
            scores = {item["label"].lower(): item["score"] for item in output}
            positive.append(scores.get("positive", 0.0))
            negative.append(scores.get("negative", 0.0))
            neutral.append(scores.get("neutral", 0.0))

        df = pd.DataFrame(
            {
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "sentiment_score": [p - n for p, n in zip(positive, negative)],
            }
        )
        return df

    def predict_single(self, text: str) -> dict[str, float]:
        """Predict sentiment for a single text string.

        Returns:
            dict with keys: positive, negative, neutral, sentiment_score.
        """
        df = self.predict([text])
        return df.iloc[0].to_dict()

    def get_sentiment(
        self,
        dates: pd.DatetimeIndex,
        symbol: str,
        headlines: pd.Series | None = None,
    ) -> pd.Series:
        """Implement SentimentProvider protocol for pipeline integration.

        Without real text data, returns zero-sentiment. Override with
        headlines= series of text aligned with dates.

        Args:
            dates: Trading day DatetimeIndex.
            symbol: Ticker symbol.
            headlines: Optional pd.Series of news headlines indexed by date.

        Returns:
            pd.Series of sentiment scores in [-1, 1] indexed by dates.
        """
        if headlines is None:
            return pd.Series(0.0, index=dates, dtype=float)

        texts = headlines.reindex(dates).fillna("").tolist()
        scores = []
        for i in range(0, len(texts), 32):
            batch = texts[i : i + 32]
            batch = [t if t else "neutral" for t in batch]
            df = self.predict(batch)
            scores.extend(df["sentiment_score"].tolist())

        return pd.Series(scores, index=dates, dtype=float)

    def batch_get_sentiment(
        self,
        headlines: dict[str, pd.Series],
        dates: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        """Get sentiment for multiple symbols at once.

        Args:
            headlines: Dict of symbol -> pd.Series of headlines indexed by date.
            dates: Optional common DatetimeIndex.

        Returns:
            DataFrame with columns = symbols, rows = dates.
        """
        if dates is None:
            all_indices = []
            for s in headlines.values():
                all_indices.extend(s.index.tolist())
            dates = pd.DatetimeIndex(sorted(set(all_indices)))

        result = pd.DataFrame(index=dates)
        for symbol, hl in headlines.items():
            result[symbol] = self.get_sentiment(dates, symbol, headlines=hl)
        return result


class FinBERTVsLMComparator:
    """Compare FinBERT vs LM Dictionary sentiment on the same texts.

    Measures agreement (correlation, sign agreement, disagreement cases)
    between deep-learning FinBERT and rule-based LM dictionary sentiment.
    Useful for understanding when the simpler LM scorer is sufficient vs
    when FinBERT's contextual understanding adds value.
    """

    def __init__(
        self,
        finbert: FinBERTSentiment | None = None,
    ) -> None:
        self.finbert = finbert or FinBERTSentiment()

    def compare(
        self,
        texts: list[str],
    ) -> pd.DataFrame:
        """Score texts with both FinBERT and LM dictionary.

        Args:
            texts: List of text strings.

        Returns:
            DataFrame with columns: text, finbert_sentiment, lm_sentiment,
            finbert_positive, finbert_negative, finbert_neutral, agreement,
            lm_pos_count, lm_neg_count, word_count.
        """
        from src.signals.sentiment.dictionary import LMSentimentScorer

        lm_scorer = LMSentimentScorer()
        finbert_results = self.finbert.predict(texts)

        results = []
        for i, text in enumerate(texts):
            lm = lm_scorer.score_text(text)
            fb = finbert_results.iloc[i]
            fb_sign = np.sign(fb["sentiment_score"])
            lm_sign = np.sign(lm["sentiment"])
            results.append(
                {
                    "text": text[:120],
                    "finbert_sentiment": round(fb["sentiment_score"], 4),
                    "lm_sentiment": round(lm["sentiment"], 4),
                    "finbert_positive": round(fb["positive"], 4),
                    "finbert_negative": round(fb["negative"], 4),
                    "finbert_neutral": round(fb["neutral"], 4),
                    "sign_agreement": fb_sign == lm_sign,
                    "lm_pos_count": lm["positive_count"],
                    "lm_neg_count": lm["negative_count"],
                    "word_count": lm["total_words"],
                }
            )

        df = pd.DataFrame(results)
        return df

    def agreement_stats(self, comparisons: pd.DataFrame) -> dict[str, float]:
        """Compute agreement statistics between FinBERT and LM.

        Returns:
            dict with: sign_agreement_pct, correlation_pearson,
            correlation_spearman, mean_finbert, mean_lm, n.
        """
        from scipy import stats as sc_stats

        fb = comparisons["finbert_sentiment"]
        lm = comparisons["lm_sentiment"]
        valid = fb.notna() & lm.notna()

        sign_agree = comparisons.loc[valid, "sign_agreement"].mean()
        pearson = sc_stats.pearsonr(fb[valid], lm[valid])
        spearman = sc_stats.spearmanr(fb[valid], lm[valid])

        return {
            "sign_agreement_pct": round(float(sign_agree * 100), 1),
            "correlation_pearson": round(float(pearson.statistic), 4),
            "correlation_pearson_p": round(float(pearson.pvalue), 4),
            "correlation_spearman": round(float(spearman.statistic), 4),
            "correlation_spearman_p": round(float(spearman.pvalue), 4),
            "mean_finbert": round(float(fb[valid].mean()), 4),
            "mean_lm": round(float(lm[valid].mean()), 4),
            "n": int(valid.sum()),
        }
