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
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

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
        y_vals = np.asarray(y, dtype=np.float64)
        y_clean = y_vals[~np.isnan(y_vals)]
        n_unique = len(np.unique(y_clean))
        is_discrete = np.issubdtype(y.dtype, np.integer) or np.issubdtype(y.dtype, np.bool_)
        is_categorical = is_discrete and (n_unique < 0.1 * len(y_clean) or n_unique <= 10)

        if is_categorical:
            mi_scores = mutual_info_classif(X_mi, y, random_state=self.random_state)
        else:
            mi_scores = mutual_info_regression(X_mi, y_vals, random_state=self.random_state)

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


@dataclass
class SFIResult:
    """Result of Sequential Feature Importance selection."""

    selected_features: List[str]
    feature_performance: list[dict]  # [{feature, ic, cumulative_ic, kept}, ...]
    n_original: int
    n_selected: int
    target_metric: str = "rank_ic"


class SFISelector:
    """Sequential Feature Importance (Lopez de Prado method).

    Iteratively builds the optimal feature subset by:
    1. Start with the single best feature by IC
    2. Try adding each remaining feature, train a model
    3. Keep the feature if it improves OOS IC
    4. Repeat until no feature improves performance

    This is superior to simple IC filtering because it accounts for
    feature interactions and redundancy.

    Example:
        >>> sfi = SFISelector(min_ic=0.02, max_features=40)
        >>> result = sfi.fit(X, y)
        >>> print(f"Selected {result.n_selected}/{result.n_original} features")
    """

    def __init__(
        self,
        min_ic: float = 0.02,
        max_features: int = 40,
        min_ic_improvement: float = 0.005,
        random_state: int = 42,
    ):
        self.min_ic = min_ic
        self.max_features = max_features
        self.min_ic_improvement = min_ic_improvement
        self.random_state = random_state
        self.selected_features_: List[str] = []
        self.result_: Optional[SFIResult] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 3,
    ) -> SFIResult:
        """Run Sequential Feature Importance.

        Args:
            X: Feature DataFrame.
            y: Forward returns (continuous).
            n_splits: Number of PurgedKFold splits (use small for speed).

        Returns:
            SFIResult with selected features and performance history.
        """
        from sklearn.ensemble import RandomForestRegressor

        from src.ml.metrics import compute_rank_ic
        from src.ml.purged_cv import PurgedKFold

        X_clean = X.fillna(0).copy()
        y_clean = y.fillna(0).copy()

        idx = X_clean.index.intersection(y_clean.index)
        X_arr = X_clean.loc[idx].values.astype(np.float64)
        y_arr = y_clean.loc[idx].values.astype(np.float64)
        col_names = list(X_clean.columns)

        purged_cv = PurgedKFold(n_splits=n_splits, pct_embargo=0.01)

        def _eval_features(feature_indices: list[int]) -> float:
            """Evaluate a set of features using mean OOS rank IC across CV folds."""
            if not feature_indices:
                return 0.0
            X_sub = X_arr[:, feature_indices]
            ic_values: list[float] = []
            for train_idx, test_idx in purged_cv.split(X_sub, y_arr):
                if len(train_idx) < 50 or len(test_idx) < 20:
                    continue
                model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=3,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                model.fit(X_sub[train_idx], y_arr[train_idx])
                y_pred = model.predict(X_sub[test_idx])
                ic_df = compute_rank_ic(
                    pd.DataFrame({"pred": y_pred}),
                    pd.Series(y_arr[test_idx]),
                )
                if not ic_df.empty:
                    ic_values.append(abs(ic_df["rank_ic"].iloc[0]))
            return float(np.mean(ic_values)) if ic_values else 0.0

        # Step 1: Evaluate each feature individually
        feature_performance: list[dict] = []
        single_ics = {}
        for i, col in enumerate(col_names):
            ic = _eval_features([i])
            single_ics[i] = ic

        # Step 2: Sort by single-feature IC and start with best
        sorted_indices = sorted(single_ics.items(), key=lambda x: x[1], reverse=True)

        if not sorted_indices or sorted_indices[0][1] < self.min_ic:
            self.result_ = SFIResult(
                selected_features=[],
                feature_performance=[],
                n_original=len(col_names),
                n_selected=0,
            )
            return self.result_

        selected_indices = [sorted_indices[0][0]]
        best_ic = sorted_indices[0][1]
        feature_performance.append({
            "feature": col_names[selected_indices[0]],
            "ic": best_ic,
            "cumulative_ic": best_ic,
            "kept": True,
            "step": 0,
        })

        remaining_indices = [idx for idx, _ in sorted_indices[1:]]
        step = 1

        while len(selected_indices) < self.max_features and remaining_indices:
            best_new_ic = best_ic
            best_new_idx = -1

            for idx in remaining_indices:
                candidate_indices = selected_indices + [idx]
                ic = _eval_features(candidate_indices)
                if ic > best_new_ic:
                    best_new_ic = ic
                    best_new_idx = idx

            if best_new_idx >= 0 and (best_new_ic - best_ic) >= self.min_ic_improvement:
                selected_indices.append(best_new_idx)
                remaining_indices.remove(best_new_idx)
                feature_performance.append({
                    "feature": col_names[best_new_idx],
                    "ic": best_new_ic - best_ic,
                    "cumulative_ic": best_new_ic,
                    "kept": True,
                    "step": step,
                })
                best_ic = best_new_ic
                step += 1
            else:
                break

        self.selected_features_ = [col_names[i] for i in selected_indices]

        self.result_ = SFIResult(
            selected_features=self.selected_features_,
            feature_performance=feature_performance,
            n_original=len(col_names),
            n_selected=len(self.selected_features_),
        )

        return self.result_

    def get_selected_features(self) -> list[str]:
        """Get list of selected feature names."""
        return self.selected_features_

    def to_dataframe(self) -> pd.DataFrame:
        """Get SFI results as a DataFrame."""
        if self.result_ is None:
            raise ValueError("Must call fit() first")
        return pd.DataFrame(self.result_.feature_performance)
