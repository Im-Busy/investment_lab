"""NLP module — sentiment analysis and text processing for trading signals.

Provides:
- SVM + TF-IDF sentiment (82-94% accuracy baseline)
- BiLSTM + LR sentiment (82.4% accuracy)
- Distant supervision labeling
- Ensemble sentiment (93.4% accuracy)
"""

from src.nlp.sentiment_pipeline import (
    SVMTfidfSentiment,
    SVMSentimentResult,
    BiLSTMSentiment,
    SentimentEnsemble,
    DistantSupervisionResult,
    strip_emoticons,
    normalize_text,
    preprocess_for_training,
    preprocess_for_inference,
    label_via_emoticons,
    apply_distant_supervision,
    load_finance_lexicon,
)

__all__ = [
    "SVMTfidfSentiment",
    "SVMSentimentResult",
    "BiLSTMSentiment",
    "SentimentEnsemble",
    "DistantSupervisionResult",
    "strip_emoticons",
    "normalize_text",
    "preprocess_for_training",
    "preprocess_for_inference",
    "label_via_emoticons",
    "apply_distant_supervision",
    "load_finance_lexicon",
]
