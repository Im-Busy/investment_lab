"""
CatBoost Wrapper for Financial Forecasting

Gradient boosting with native categorical feature support and ordered boosting
for time-series applications.
"""

from __future__ import annotations

from typing import Any, List, Optional
import pandas as pd
import numpy as np


class CatBoostForecaster:
    """
    CatBoost wrapper optimized for financial time series.

    Advantages over LightGBM/XGBoost:
    - Ordered boosting reduces overfitting on time series
    - Native categorical feature handling
    - Better calibration out-of-box
    - Robust to hyperparameter choices
    """

    def __init__(
        self,
        n_estimators: int = 500,
        depth: int = 6,
        learning_rate: float = 0.03,
        l2_leaf_reg: float = 3.0,
        random_seed: int = 42,
        verbose: bool = False,
    ):
        """
        Initialize CatBoost.

        Args:
            n_estimators: Number of trees
            depth: Tree depth
            learning_rate: Learning rate
            l2_leaf_reg: L2 regularization
            random_seed: Random seed
            verbose: Logging verbosity
        """
        self.n_estimators = n_estimators
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.verbose = verbose

        self.model_reg = None
        self.model_clf = None

    def _has_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _create_regressor(self) -> Any:
        """Create CatBoost regressor."""
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=self.n_estimators,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            verbose=self.verbose,
            loss_function="Huber:delta=1.0",
            task_type="GPU" if self._has_gpu() else "CPU",
        )

    def _create_classifier(self) -> Any:
        """Create CatBoost classifier."""
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=self.n_estimators,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            verbose=self.verbose,
            loss_function="Logloss",
            task_type="GPU" if self._has_gpu() else "CPU",
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        categorical_features: Optional[List[str]] = None,
    ) -> "CatBoostForecaster":
        """
        Train CatBoost model.

        Args:
            X: Feature matrix
            y: Target variable
            categorical_features: List of categorical feature names

        Returns:
            Self for chaining
        """
        if y.dtype in ["int64", "int32", "bool"]:
            self.model_clf = self._create_classifier()
            self.model_clf.fit(
                X,
                y,
                cat_features=categorical_features or [],
            )
        else:
            self.model_reg = self._create_regressor()
            self.model_reg.fit(
                X,
                y,
                cat_features=categorical_features or [],
            )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if self.model_clf is not None:
            return self.model_clf.predict(X)
        elif self.model_reg is not None:
            return self.model_reg.predict(X)
        else:
            raise ValueError("Model not trained. Call fit() first.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions (classification only)."""
        if self.model_clf is None:
            raise ValueError("Classifier not trained. Call fit() first.")
        return self.model_clf.predict_proba(X)

    def get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get feature importance scores."""
        model = self.model_clf or self.model_reg
        if model is None:
            raise ValueError("Model not trained.")

        importance = model.get_feature_importance()
        return pd.DataFrame({"feature": X.columns, "importance": importance}).sort_values(
            "importance", ascending=False
        )
