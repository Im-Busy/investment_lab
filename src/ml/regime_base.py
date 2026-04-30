"""
Regime Detection Base Classes

Abstract base class defining the common interface for all regime detectors.
All regime detectors must implement: fit(), predict(), predict_proba(), get_regime_summary()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class RegimeType(Enum):
    """Common regime states across all detectors."""

    TRENDING = "Trending"
    RANGING = "Ranging"
    VOLATILE = "Volatile"
    TRANSITION = "Transition"
    BULL_BEAR = "Bull"
    BEAR_BULL = "Bear"
    HIGH_VOL = "High_Vol"
    LOW_VOL = "Low_Vol"


@dataclass
class RegimeSummary:
    """Summary statistics for a regime detector."""

    name: str
    n_regimes: int
    regime_labels: List[str]
    label_distribution: Dict[str, int]
    label_proportions: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class RegimeDetectorBase(ABC):
    """
    Abstract base class for all regime detectors.

    All regime detectors must implement:
    - fit(data: pd.DataFrame) -> Self
    - predict(data: pd.DataFrame) -> pd.Series
    - predict_proba(data: pd.DataFrame) -> pd.DataFrame
    - get_regime_summary() -> RegimeSummary

    Example:
        >>> detector = HMMRegimeDetector(n_states=4)
        >>> detector.fit(features)
        >>> regimes = detector.predict(features)
        >>> probs = detector.predict_proba(features)
        >>> summary = detector.get_regime_summary()
    """

    def __init__(self, name: str = "BaseDetectors"):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> RegimeDetectorBase:
        """
        Fit the regime detector on input data.

        Args:
            data: DataFrame with features (columns) and observations (rows)

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict regime labels for input data.

        Args:
            data: DataFrame with features (same columns as training)

        Returns:
            Series of regime labels (strings) with same index as input
        """
        pass

    @abstractmethod
    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime probabilities for input data.

        Args:
            data: DataFrame with features (same columns as training)

        Returns:
            DataFrame with probability for each regime (columns sum to 1)
        """
        pass

    @abstractmethod
    def get_regime_summary(self) -> RegimeSummary:
        """
        Get summary information about the detected regimes.

        Returns:
            RegimeSummary with label distribution and metadata
        """
        pass

    def _validate_features(self, data: pd.DataFrame, require_fitted: bool = True) -> None:
        """
        Validate input features.

        Args:
            data: DataFrame to validate
            require_fitted: Whether to check if detector is fitted

        Raises:
            ValueError: If data is empty or detector not fitted
        """
        if data.empty:
            raise ValueError("Input data cannot be empty")

        if require_fitted and not self.is_fitted:
            raise ValueError(f"Detector not fitted. Call fit() first.")

    def _get_label_distribution(self, labels: pd.Series) -> tuple[Dict[str, int], Dict[str, float]]:
        """
        Calculate label distribution statistics.

        Args:
            labels: Series of regime labels

        Returns:
            Tuple of (count_dict, proportion_dict)
        """
        counts = labels.value_counts().to_dict()
        total = len(labels)
        proportions = {k: v / total for k, v in counts.items()}
        return counts, proportions

    def _map_to_standard_regimes(
        self,
        labels: pd.Series,
        mapping: Optional[Dict[str, str]] = None,
    ) -> pd.Series:
        """
        Map internal labels to standard regime names.

        Args:
            labels: Series of internal labels (integers or strings)
            mapping: Optional custom mapping dict

        Returns:
            Series of mapped regime names
        """
        if mapping is None:
            unique_labels = sorted(labels.unique())
            n_regimes = len(unique_labels)
            mapping = {label: f"Regime_{i}" for i, label in enumerate(unique_labels)}

        return labels.map(mapping)
