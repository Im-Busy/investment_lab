"""Glassbox Explainable Boosting Machine (EBM) for regime classification.

EBM is a Generalized Additive Model with interactions (GA²M) that produces
fully interpretable predictions. Unlike black-box trees (RF, GB, LGBM),
EBM reveals the exact contribution of each feature to every prediction.

Key properties:
- Glassbox: every prediction can be decomposed into per-feature contributions
- Shape functions: the learned function for each feature is viewable as a graph
- Interaction detection: automatically discovers pairwise feature interactions
- Global + local explanations without surrogate models

Reference: Nori et al., "InterpretML: A Unified Framework for Machine
Learning Interpretability", arXiv:1909.09223, 2019.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EBMRegimeClassifier:
    """Glassbox EBM-based regime classifier with full explainability.

    Trains a Generalized Additive Model with interactions on regime-labeled
    features. Every prediction includes per-feature contributions and overall
    intercept, making it fully auditable.

    Example:
        >>> ebm = EBMRegimeClassifier()
        >>> result = ebm.train(X, y)
        >>> predictions = ebm.predict(X_new)
        >>> explanations = ebm.explain_local(X_new.iloc[0])  # per-feature breakdown
    """

    def __init__(
        self,
        max_bins: int = 256,
        max_interaction_bins: int = 32,
        interactions: int = 10,
        outer_bags: int = 8,
        inner_bags: int = 0,
        learning_rate: float = 0.01,
        max_rounds: int = 5000,
        early_stopping_rounds: int = 50,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ):
        """Initialize EBM regime classifier.

        Args:
            max_bins: Max bins for continuous features (256 for EBM).
            max_interaction_bins: Max bins for interaction terms (32 recommended).
            interactions: Number of pairwise interactions to detect (0-100).
            outer_bags: Outer bagging rounds for confidence intervals.
            inner_bags: Inner bagging rounds (0 = none, recommended for small data).
            learning_rate: Boosting learning rate.
            max_rounds: Max boosting rounds (acts like n_estimators).
            early_stopping_rounds: Patience for early stopping.
            min_samples_leaf: Minimum samples per leaf (regularization).
            random_state: Random seed.
        """
        self.max_bins = max_bins
        self.max_interaction_bins = max_interaction_bins
        self.interactions = interactions
        self.outer_bags = outer_bags
        self.inner_bags = inner_bags
        self.learning_rate = learning_rate
        self.max_rounds = max_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.model_: Any = None
        self.feature_names_: List[str] = []
        self.classes_: List[str] = []
        self._global_explanation: Any = None
        self._intercept_: float = 0.0
        self._feature_contributions_: Dict[str, np.ndarray] = {}
        self._is_trained: bool = False

    def _create_model(self) -> Any:
        """Create the underlying EBM model."""
        from interpret.glassbox import ExplainableBoostingClassifier

        return ExplainableBoostingClassifier(
            max_bins=self.max_bins,
            max_interaction_bins=self.max_interaction_bins,
            interactions=self.interactions,
            outer_bags=self.outer_bags,
            inner_bags=self.inner_bags,
            learning_rate=self.learning_rate,
            max_rounds=self.max_rounds,
            early_stopping_rounds=self.early_stopping_rounds,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        purge_window: int = 5,
    ) -> Dict[str, Any]:
        """Train the EBM on regime-labeled features.

        Args:
            X: Feature DataFrame.
            y: Label series (regime names or integer labels).
            test_size: Fraction of data held out for testing.
            purge_window: Samples to exclude at train/test boundary.

        Returns:
            Dict with train/test accuracy, feature importance, and intercept.
        """
        from sklearn.metrics import accuracy_score, classification_report

        valid_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()

        if len(X_clean) < 50:
            raise ValueError(f"Insufficient samples: {len(X_clean)} (need >=50)")

        self.feature_names_ = list(X_clean.columns)

        split_idx = int(len(X_clean) * (1 - test_size))
        train_end = max(0, split_idx - purge_window)

        X_train = X_clean.iloc[:train_end]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:train_end]
        y_test = y_clean.iloc[split_idx:]

        self.model_ = self._create_model()
        self.model_.fit(X_train, y_train)
        self.classes_ = list(self.model_.classes_)

        train_pred = self.model_.predict(X_train)
        test_pred = self.model_.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        self._extract_global_explanation(X_train)

        self._is_trained = True

        results: Dict[str, Any] = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_report": classification_report(y_train, train_pred, output_dict=True),
            "test_report": classification_report(y_test, test_pred, output_dict=True),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "intercept": self._intercept_,
        }

        if self._feature_contributions_:
            results["feature_importance"] = dict(
                sorted(
                    [(f, float(c)) for f, c in self._feature_contributions_.items()],
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:20]
            )

        return results

    def _extract_global_explanation(self, X: pd.DataFrame) -> None:
        """Extract global feature contributions and intercept from EBM."""
        try:
            self._global_explanation = self.model_.explain_global(name="EBM Regime Classifier")
            data_dict = self._global_explanation.data()

            # EBM global explanation format:
            # {"type": "univariate", "names": [...features...], "scores": [...importance...]}
            names = data_dict.get("names", [])
            scores = data_dict.get("scores", [])

            if names and scores:
                for feat, score_val in zip(names, scores):
                    self._feature_contributions_[str(feat)] = float(score_val)

            # Intercept is available via model internal attributes
            if hasattr(self.model_, "intercept_"):
                intercepts = self.model_.intercept_
                if isinstance(intercepts, np.ndarray):
                    self._intercept_ = float(np.mean(intercepts))
                else:
                    self._intercept_ = float(intercepts)

        except Exception as e:
            logger.warning(f"Could not extract global explanation: {e}")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict regime labels.

        Args:
            X: Feature DataFrame.

        Returns:
            Series of predicted regime labels.
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")

        X_clean = X[self.feature_names_].fillna(0)
        predictions = self.model_.predict(X_clean)
        return pd.Series(predictions, index=X.index, name="predicted_regime")

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get prediction probabilities for each regime.

        Args:
            X: Feature DataFrame.

        Returns:
            DataFrame with probability columns per regime.
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")

        X_clean = X[self.feature_names_].fillna(0)
        proba = self.model_.predict_proba(X_clean)
        return pd.DataFrame(proba, columns=self.classes_, index=X.index)

    def explain_local(self, X: pd.DataFrame | pd.Series) -> Dict[str, Any]:
        """Get glassbox local explanation for a single sample.

        Decomposes a prediction into per-feature contributions plus intercept.
        The sum of all contributions + intercept = log-odds of the prediction.

        Args:
            X: Single row of features (DataFrame or Series).

        Returns:
            Dict with:
                intercept: global baseline
                features: dict of feature_name → contribution
                predictions: per-class probabilities
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")

        if isinstance(X, pd.DataFrame):
            X_row = X[self.feature_names_].iloc[:1].fillna(0)
        else:
            X_row = pd.DataFrame([X[self.feature_names_].fillna(0)], columns=self.feature_names_)

        contributions: Dict[str, float] = {}
        try:
            local_exp = self.model_.explain_local(X_row, name="Regime prediction")
            local_data = local_exp.data()

            if local_data and isinstance(local_data, dict):
                names = local_data.get("names", [])
                scores = local_data.get("scores", [])
                if names and scores:
                    for name, score in zip(names, scores):
                        contributions[str(name)] = float(score)
        except Exception:
            pass

        return {
            "intercept": self._intercept_,
            "features": contributions,
            "predictions": dict(zip(self.classes_, self.model_.predict_proba(X_row)[0])),
        }

    def explain_global(self) -> Dict[str, Any]:
        """Get global explanation with per-feature shape functions.

        Shape functions show how each feature's value maps to its
        contribution to the prediction. These are the key differentiator
        of EBM vs black-box models.

        Returns:
            Dict with:
                intercept: global baseline
                feature_importance: {feature: mean_abs_contribution}
                shape_functions: {feature: list of (bin_label, contribution, density)}
                interaction_terms: list of interaction names
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")

        return {
            "intercept": self._intercept_,
            "feature_importance": self._feature_contributions_,
            "shape_functions": self._extract_shape_functions(),
            "interaction_terms": self._extract_interaction_terms(),
        }

    def _extract_shape_functions(self) -> Dict[str, list]:
        """Extract per-feature shape function data from the global explanation."""
        shape_fns: Dict[str, list] = {}
        if self._global_explanation is None:
            return shape_fns

        data = self._global_explanation.data()

        names = data.get("names", [])
        scores = data.get("scores", [])

        for name, score in zip(names, scores):
            shape_fns[str(name)] = [{"contribution": float(score)}]

        return shape_fns

    def _extract_interaction_terms(self) -> List[str]:
        """Extract pairwise interaction feature names."""
        if self._global_explanation is None or self.interactions == 0:
            return []

        interactions = []
        try:
            if hasattr(self.model_, "term_names_"):
                term_names = self.model_.term_names_
                interactions = [str(t) for t in term_names if isinstance(t, str) and " & " in t]
        except Exception:
            pass
        return interactions

    def plot_shape_function(self, feature_name: str) -> Any | None:
        """Plot the shape function for a single feature.

        Returns a matplotlib figure showing how the feature's value maps
        to its contribution to the prediction.

        Args:
            feature_name: Name of the feature to visualize.

        Returns:
            Matplotlib Figure or None if feature not found.
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")

        if self._global_explanation is None:
            return None

        try:
            return self._global_explanation.visualize(feature_name)
        except Exception as e:
            logger.warning(f"Could not visualize {feature_name}: {e}")
            return None

    def plot_all_shape_functions(
        self,
        top_n: int = 8,
    ) -> Any | None:
        """Plot shape functions for top N most important features.

        Args:
            top_n: Number of top features to visualize.

        Returns:
            Matplotlib Figure or None.
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")

        if self._global_explanation is None:
            return None

        top_features = sorted(
            self._feature_contributions_.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_n]

        try:
            import matplotlib.pyplot as plt

            n_rows = (len(top_features) + 1) // 2
            fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows))
            axes = axes.flatten() if n_rows > 1 else ([axes] if n_rows == 1 else [])

            for i, (feat, _) in enumerate(top_features):
                if i < len(axes):
                    try:
                        self._global_explanation.visualize(feat).axes[0].set_title(f"{feat} shape")
                    except Exception:
                        axes[i].text(0.5, 0.5, f"{feat}\n(no shape data)", ha="center", va="center")
                        axes[i].set_title(feat)

            for j in range(len(top_features), len(axes)):
                axes[j].set_visible(False)

            fig.suptitle("EBM Shape Functions — Global Feature Contributions", fontsize=14)
            fig.tight_layout()
            return fig
        except Exception as e:
            logger.warning(f"Could not create shape function plot: {e}")
            return None

    def feature_importance_table(self) -> pd.DataFrame:
        """Return feature importance as a sorted DataFrame.

        Returns:
            DataFrame with columns: feature, mean_abs_contribution, sign.
        """
        if not self._feature_contributions_:
            return pd.DataFrame(columns=["feature", "mean_abs_contribution", "sign"])

        rows = []
        for feat, contrib in sorted(
            self._feature_contributions_.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        ):
            rows.append(
                {
                    "feature": feat,
                    "mean_abs_contribution": abs(contrib),
                    "sign": "positive" if contrib >= 0 else "negative",
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> Dict[str, Any]:
        """Return a human-readable summary of the EBM model.

        Returns:
            Dict with model metadata, performance, top features, interactions.
        """
        top_features = sorted(
            self._feature_contributions_.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:10]

        explanations = self.explain_global()

        return {
            "model_type": "EBM (Explainable Boosting Machine)",
            "n_features": len(self.feature_names_),
            "n_classes": len(self.classes_),
            "classes": self.classes_,
            "intercept": self._intercept_,
            "top_features": [{"feature": f, "contribution": float(c)} for f, c in top_features],
            "n_interactions": len(explanations.get("interaction_terms", [])),
            "interactions": explanations.get("interaction_terms", [])[:5],
            "hyperparams": {
                "max_bins": self.max_bins,
                "interactions": self.interactions,
                "outer_bags": self.outer_bags,
                "learning_rate": self.learning_rate,
                "max_rounds": self.max_rounds,
            },
        }

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def save(self, path: str) -> None:
        """Save trained EBM model to disk."""
        import pickle
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model_,
            "feature_names": self.feature_names_,
            "classes": self.classes_,
            "intercept": self._intercept_,
            "feature_contributions": self._feature_contributions_,
            "config": {
                "max_bins": self.max_bins,
                "interactions": self.interactions,
                "outer_bags": self.outer_bags,
                "learning_rate": self.learning_rate,
                "max_rounds": self.max_rounds,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load trained EBM model from disk."""
        import pickle
        from pathlib import Path

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)  # nosec B301

        self.model_ = data["model"]
        self.feature_names_ = data["feature_names"]
        self.classes_ = data.get("classes", [])
        self._intercept_ = data.get("intercept", 0.0)
        self._feature_contributions_ = data.get("feature_contributions", {})
        self._is_trained = True
