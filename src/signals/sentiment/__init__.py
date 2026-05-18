"""Sentiment subpackage -- NLP-based signal sources and modifiers."""

from .dictionary import LMDictionary, LMSentimentScorer, LMTradingSignalModifier
from .finbert import FinBERTSentiment, FinBERTVsLMComparator
from .multi_source_fusion import MultiSourceFusion, FusionResult
from .filing_analyzer import FilingAnalyzer

__all__ = [
    "LMDictionary",
    "LMSentimentScorer",
    "LMTradingSignalModifier",
    "FinBERTSentiment",
    "FinBERTVsLMComparator",
    "MultiSourceFusion",
    "FusionResult",
    "FilingAnalyzer",
]
