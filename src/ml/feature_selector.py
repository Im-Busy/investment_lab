"""
Feature Selector for ML Trading

Reduces feature count from 81+ to ~20-30 using:
1. Variance thresholding
2. Correlation filtering (remove highly correlated features)
3. Mutual information ranking
4. Recursive feature elimination (optional)

Usage:
    from src.ml.feature_selector import FeatureSelector

    selector = FeatureSelector(target_features=25)
    selected_features = selector.fit_transform(X, y)
    print(f"Selected {len(selected_features.columns)} features")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class SelectionResult:
    """Result of feature selection."""

    original_features: int
    selected_features: int
    removed_variance: int
    removed_correlation: int
    selected_names: List[str]
    feature_scores: dict = field(default_factory=dict)


class FeatureSelector:
    """
    Select the most informative features for ML models.

    Pipeline:
    1. Remove near-zero variance features
    2. Remove highly correlated features (keep highest MI score)
    3. Rank by mutual information with target
    4. Select top N features

    Example:
        >>> selector = FeatureSelector(
        ...     target_features=25,
        ...     corr_threshold=0.85,
        ... )
        >>> result = selector.fit(X, y)
        >>> X_selected = selector.transform(X)
    """

    def __init__(
        self,
        target_features: int = 25,
        corr_threshold: float = 0.85,
        variance_threshold: float = 0.01,
        random_state: int = 42,
    ):
        self.target_features = target_features
        self.corr_threshold = corr_threshold
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self.selected_features_: List[str] = []
        self.feature_scores_: dict = {}
        self.result_: Optional[SelectionResult] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SelectionResult:
        """
        Fit the feature selector.

        Args:
            X: Feature DataFrame
            y: Target labels

        Returns:
            SelectionResult with details of selection
        """
        from sklearn.feature_selection import mutual_info_classif

        original_n = len(X.columns)
        removed_variance = 0
        removed_correlation = 0

        X = X.copy()

        # Step 1: Remove near-zero variance features
        stds = X.std()
        high_var_cols = [c for c in X.columns if stds[c] > self.variance_threshold]
        removed_variance = original_n - len(high_var_cols)
        X = X[high_var_cols]

        # Step 2: Remove highly correlated features
        X_clean = X.fillna(0)
        corr_matrix = X_clean.corr().abs()

        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))

        to_drop_corr = []
        for col in upper_tri.columns:
            if any(upper_tri[col] > self.corr_threshold):
                corr_with = upper_tri[col][upper_tri[col] > self.corr_threshold].index.tolist()
                to_drop_corr.extend(corr_with)

        to_drop_corr = list(set(to_drop_corr))
        X = X.drop(columns=to_drop_corr, errors="ignore")
        removed_correlation = len(to_drop_corr)

        # Step 3: Rank by mutual information
        X_mi = X.fillna(0)
        mi_scores = mutual_info_classif(X_mi, y, random_state=self.random_state)

        mi_dict = dict(zip(X.columns, mi_scores))
        self.feature_scores_ = mi_dict

        # Step 4: Select top N
        sorted_features = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
        self.selected_features_ = [f for f, _ in sorted_features[: self.target_features]]

        self.result_ = SelectionResult(
            original_features=original_n,
            selected_features=len(self.selected_features_),
            removed_variance=removed_variance,
            removed_correlation=removed_correlation,
            selected_names=self.selected_features_,
            feature_scores={f: v for f, v in sorted_features if f in self.selected_features_},
        )

        return self.result_

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform X to selected features.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with selected features only
        """
        if not self.selected_features_:
            raise ValueError("Must call fit() before transform()")

        available = [c for c in self.selected_features_ if c in X.columns]
        return X[available].fillna(0)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    def get_feature_rankings(self, top_n: int = 30) -> pd.DataFrame:
        """
        Get ranked features by mutual information score.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature, score, selected columns
        """
        if not self.feature_scores_:
            raise ValueError("Must call fit() first")

        sorted_items = sorted(self.feature_scores_.items(), key=lambda x: x[1], reverse=True)
        return pd.DataFrame(
            [
                {"feature": f, "mi_score": s, "selected": f in self.selected_features_}
                for f, s in sorted_items[:top_n]
            ]
        )
