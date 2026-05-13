"""
ML Regime Classifier

Uses machine learning to enhance the rule-based ADX/ATR regime detector.
Supports multiple classifiers (Random Forest, Gradient Boosting, LightGBM, Logistic Regression).

Key capabilities:
1. Train on historical data with regime labels from rule-based detector
2. PurgedKFold + embargo cross-validation via Phase A purged_cv.py (ML4T Ch6:04)
3. MDI + permutation + SHAP feature importance
4. IC-based evaluation (Spearman rank IC, not just accuracy)
5. Automated experiment logging via ExperimentLogger
6. Multi-model comparison (RF vs GB vs LightGBM)
7. Walk-forward validation with purged splits
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MLRegimeState(Enum):
    """ML-predicted regime states (aligned with rule-based detector)."""

    TRENDING = "Trending"
    RANGING = "Ranging"
    VOLATILE = "Volatile"
    TRANSITION = "Transition"


class RegimeClassifier:
    """
    ML-based regime classifier.

    WARNING — DO NOT USE AS PREDICTION TARGET:
    This class trains on labels from the rule-based RegimeDetector, which are
    deterministic functions of ADX/ATR thresholds. Using these labels as ML
    targets causes the model to memorize the rule formula rather than learn
    market behavior, producing fake 0.95+ ICs.

    Rule-based regimes (Trending/Ranging/Volatile/Transition) are valid as ML
    FEATURES but NOT as prediction targets. Use triple-barrier labels from
    ``src/ml/triple_barrier.py`` for training targets instead.

    Uses PurgedKFold from Phase A for proper temporal cross-validation with embargo.

    Example:
        >>> from src.indicators.regime_detector import RegimeDetector
        >>> detector = RegimeDetector()
        >>> regimes = detector.detect(df)  # valid as features
        >>> # Do NOT train like this:
        >>> # classifier.train(features, regimes)  # BUG PATTERN
    """

    SUPPORTED_MODELS = [
        "random_forest",
        "gradient_boosting",
        "logistic_regression",
        "lightgbm",
        "catboost",
    ]

    def __init__(
        self,
        model_type: str = "catboost",
        n_estimators: int = 100,
        max_depth: Optional[int] = 3,
        random_state: int = 42,
        min_samples_leaf: int = 10,
        max_features: str = "sqrt",
    ):
        """
        Initialize regime classifier.

        Args:
            model_type: Type of classifier ("random_forest", "gradient_boosting", "logistic_regression", "lightgbm", "catboost")
            n_estimators: Number of trees (for tree-based models)
            max_depth: Maximum tree depth (reduced from 5 to 3 to prevent overfitting)
            random_state: Random seed for reproducibility
            min_samples_leaf: Minimum samples per leaf (regularization)
            max_features: Number of features per split (regularization)
        """
        import warnings

        warnings.warn(
            "RegimeClassifier is DEPRECATED as a prediction target. "
            "Rule-based regime labels (ADX/ATR thresholds) are deterministic formulas, "
            "not market behavior. Training ML models on these synthetic labels produces "
            "falsely high ICs (~0.95) — the model learns the rule, not the market. "
            "Use triple-barrier labels (src/ml/triple_barrier.py) as training targets instead. "
            "RegimeDetector outputs can still be used as ML features.",
            DeprecationWarning,
            stacklevel=2,
        )

        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model type: {model_type}. Supported: {self.SUPPORTED_MODELS}"
            )

        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.model = None
        self.classes_ = None
        self.feature_names_ = None

    def _create_model(self):
        """Create the underlying ML model with regularization."""
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        if self.model_type == "catboost":
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                iterations=self.n_estimators,
                depth=self.max_depth if self.max_depth else 6,
                learning_rate=0.03,
                l2_leaf_reg=3.0,
                random_seed=self.random_state,
                verbose=False,
                loss_function="MultiClass",
                auto_class_weights="Balanced",
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                class_weight="balanced",
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
            )
        elif self.model_type == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth if self.max_depth else -1,
                num_leaves=min(2 ** (self.max_depth or 3), 31),
                min_child_samples=self.min_samples_leaf,
                feature_fraction=0.8 if self.max_features == "sqrt" else 1.0,
                random_state=self.random_state,
                class_weight="balanced",
                verbose=-1,
            )
        else:
            return LogisticRegression(
                max_iter=1000,
                C=0.1,
                class_weight="balanced",
                random_state=self.random_state,
            )

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        purge_window: int = 5,
    ) -> Dict[str, Any]:
        """
        Train the classifier on features and labels.

        Args:
            X: Feature DataFrame
            y: Labels series (regime names)
            test_size: Fraction of data for testing
            purge_window: Number of training samples at the split boundary
                to exclude from training (prevents label overlap leakage).
                Set to match the forward-return horizon used for labels.

        Returns:
            Dict with train/test scores and feature importance
        """
        from sklearn.metrics import classification_report, accuracy_score

        valid_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        split_idx = int(len(X_clean) * (1 - test_size))
        train_end = max(0, split_idx - purge_window)
        X_train = X_clean.iloc[:train_end]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:train_end]
        y_test = y_clean.iloc[split_idx:]

        self.model = self._create_model()
        self.feature_names_ = list(X_train.columns)
        self.model.fit(X_train, y_train)
        self.classes_ = list(self.model.classes_)

        train_pred = self.model.predict(X_train)

        if len(X_test) > 0:
            test_pred = self.model.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_pred)
            test_report = classification_report(y_test, test_pred, output_dict=True)
        else:
            test_pred = None
            test_accuracy = None
            test_report = {}

        results = {
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": test_accuracy,
            "train_report": classification_report(y_train, train_pred, output_dict=True),
            "test_report": test_report,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        if hasattr(self.model, "feature_importances_"):
            results["feature_importance"] = dict(
                sorted(
                    zip(self.feature_names_, self.model.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True,
                )[:20]
            )

        return results

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict regime for new data.

        Args:
            X: Feature DataFrame

        Returns:
            Series of predicted regime names
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        predictions = self.model.predict(X.fillna(0))
        return pd.Series(predictions, index=X.index, name="predicted_regime")

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get prediction probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with probability for each regime
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        probs = self.model.predict_proba(X.fillna(0))
        return pd.DataFrame(probs, columns=self.model.classes_, index=X.index)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get MDI feature importance scores.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if not hasattr(self.model, "feature_importances_"):
            if hasattr(self.model, "coef_"):
                importance = np.abs(self.model.coef_).mean(axis=0)
                return (
                    pd.DataFrame({"feature": self.feature_names_, "importance": importance})
                    .sort_values("importance", ascending=False)
                    .head(top_n)
                )
            return pd.DataFrame()

        return (
            pd.DataFrame(
                {"feature": self.feature_names_, "importance": self.model.feature_importances_}
            )
            .sort_values("importance", ascending=False)
            .head(top_n)
        )

    def compute_permutation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 5,
        scoring: str = "accuracy",
    ) -> pd.DataFrame:
        """Compute permutation feature importance.

        Permutation importance measures the drop in model performance when
        a feature's values are randomly shuffled, breaking its relationship
        with the target. More reliable than MDI for measuring true feature
        contribution (ML4T Ch12:07).

        Args:
            X: Feature DataFrame.
            y: Label series.
            n_repeats: Number of times to permute each feature.
            scoring: Metric to evaluate (default: "accuracy").

        Returns:
            DataFrame with feature, importance_mean, importance_std.
        """
        from sklearn.inspection import permutation_importance

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_clean = X.fillna(0)
        y_clean = y.dropna()
        idx = X_clean.index.intersection(y_clean.index)
        X_clean = X_clean.loc[idx]
        y_clean = y_clean.loc[idx]

        result = permutation_importance(
            self.model,
            X_clean,
            y_clean,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring=scoring,
        )

        return pd.DataFrame(
            {
                "feature": self.feature_names_,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)

    def compute_shap_values(
        self,
        X: pd.DataFrame,
        max_samples: int = 200,
    ) -> tuple[pd.DataFrame, Any]:
        """Compute SHAP values for model interpretability.

        Uses TreeExplainer for tree-based models or KernelExplainer as fallback.
        SHAP provides additive feature attribution that satisfies consistency
        and fairness axioms (Lundberg & Lee, 2017).

        Args:
            X: Feature DataFrame.
            max_samples: Max samples for kernel explainer (tree models use all).

        Returns:
            Tuple of (shap_importance DataFrame, shap_values array).
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_clean = X.fillna(0).head(max_samples)

        try:
            import shap

            if self.model_type in ("random_forest", "gradient_boosting", "lightgbm"):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_clean)

                if isinstance(shap_values, list):
                    shap_vals = shap_values[0]
                else:
                    shap_vals = shap_values
            else:
                explainer = shap.KernelExplainer(
                    self.model.predict_proba, shap.sample(X_clean, min(50, len(X_clean)))
                )
                shap_values = explainer.shap_values(X_clean)
                shap_vals = shap_values if not isinstance(shap_values, list) else shap_values[0]

            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            if mean_abs_shap.ndim > 1:
                mean_abs_shap = mean_abs_shap.sum(axis=1)

            importance_df = pd.DataFrame(
                {
                    "feature": self.feature_names_[: len(mean_abs_shap)],
                    "mean_abs_shap": mean_abs_shap,
                }
            ).sort_values("mean_abs_shap", ascending=False)

            return importance_df, shap_vals

        except ImportError:
            logger.warning("shap not installed. Run: uv add shap")
            return pd.DataFrame(), None
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return pd.DataFrame(), None

    def evaluate_with_ic(
        self,
        X: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> pd.DataFrame:
        """Evaluate model predictions using IC metrics.

        Converts predicted regime probabilities to a continuous score
        (probability of "Trending" regime) and computes Spearman rank IC
        against forward returns.

        Args:
            X: Feature DataFrame.
            forward_returns: Forward return series (aligned by index).

        Returns:
            DataFrame with ic, rank_ic, hit_rate per regime class.
        """
        from src.ml.metrics import compute_ic, compute_rank_ic, compute_hit_rate

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_clean = X.fillna(0)
        proba = self.model.predict_proba(X_clean)
        classes = self.model.classes_

        results = []
        for i, cls_name in enumerate(classes):
            pred = pd.Series(proba[:, i], index=X_clean.index, name=f"proba_{cls_name}")
            ic_df = compute_rank_ic(pred, forward_returns)
            pearson_df = compute_ic(pred, forward_returns)

            hit = compute_hit_rate(pred.values, forward_returns.values)

            results.append(
                {
                    "regime": cls_name,
                    "rank_ic": ic_df["rank_ic"].iloc[0] if not ic_df.empty else np.nan,
                    "ic": pearson_df["ic"].iloc[0] if not pearson_df.empty else np.nan,
                    "hit_rate": hit,
                }
            )

        return pd.DataFrame(results).sort_values("rank_ic", key=abs, ascending=False)

    def train_with_purged_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        forward_returns: pd.Series | None = None,
        n_splits: int = 5,
        pct_embargo: float = 0.01,
        label_span: int = 5,
        experiment_logger: Any = None,
    ) -> Dict[str, Any]:
        """Train and evaluate using PurgedKFold cross-validation.

        Uses Phase A PurgedKFold from purged_cv.py for proper temporal
        cross-validation with embargo. Evaluates using both classification
        accuracy and IC metrics.

        Args:
            X: Feature DataFrame with datetime index.
            y: Label series (regime names).
            forward_returns: Forward returns for IC evaluation (optional).
            n_splits: Number of CV folds.
            pct_embargo: Fraction of test span to embargo.
            label_span: Forward label horizon for purging.
            experiment_logger: Optional ExperimentLogger instance.

        Returns:
            Dict with mean/std metrics per fold, overfit gap, feature importance.
        """
        from sklearn.metrics import accuracy_score
        from src.ml.purged_cv import PurgedKFold
        from src.ml.metrics import compute_rank_ic, compute_hit_rate

        X_clean = X.fillna(0).copy()
        y_clean = y.dropna().copy()
        idx = X_clean.index.intersection(y_clean.index)
        X_clean = X_clean.loc[idx]
        y_clean = y_clean.loc[idx]

        cv = PurgedKFold(n_splits=n_splits, pct_embargo=pct_embargo, label_span=label_span)

        fold_results: list[dict] = []
        train_scores: list[float] = []
        test_scores: list[float] = []
        train_ic: list[float] = []
        test_ic: list[float] = []

        self.feature_names_ = list(X_clean.columns)

        for fold_idx, (train_idx_arr, test_idx_arr) in enumerate(cv.split(X_clean), 1):
            train_idx = X_clean.index[train_idx_arr]
            test_idx = X_clean.index[test_idx_arr]

            X_train = X_clean.loc[train_idx]
            y_train = y_clean.loc[train_idx]
            X_test = X_clean.loc[test_idx]
            y_test = y_clean.loc[test_idx]

            if len(y_train) < 50 or len(y_test) < 10:
                continue

            model = self._create_model()
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            test_proba = model.predict_proba(X_test)

            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            train_scores.append(train_acc)
            test_scores.append(test_acc)

            fold_record: dict = {
                "fold": fold_idx,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "train_start": str(train_idx[0])
                if hasattr(train_idx[0], "strftime")
                else str(train_idx[0]),
                "train_end": str(train_idx[-1])
                if hasattr(train_idx[-1], "strftime")
                else str(train_idx[-1]),
                "test_start": str(test_idx[0])
                if hasattr(test_idx[0], "strftime")
                else str(test_idx[0]),
                "test_end": str(test_idx[-1])
                if hasattr(test_idx[-1], "strftime")
                else str(test_idx[-1]),
            }

            if forward_returns is not None and len(test_idx) > 0:
                trending_idx = (
                    list(model.classes_).index("Trending") if "Trending" in model.classes_ else 0
                )
                test_proba_trending = pd.Series(test_proba[:, trending_idx], index=X_test.index)
                ic_df = compute_rank_ic(test_proba_trending, forward_returns)
                if not ic_df.empty:
                    fold_record["test_rank_ic"] = ic_df["rank_ic"].iloc[0]
                    test_ic.append(ic_df["rank_ic"].iloc[0])

                hit = compute_hit_rate(
                    test_proba_trending.values, forward_returns.loc[X_test.index].values
                )
                fold_record["test_hit_rate"] = hit

            fold_results.append(fold_record)

            if experiment_logger is not None:
                experiment_logger.log_fold_metrics(
                    fold=fold_idx,
                    train_metrics={"accuracy": train_acc},
                    test_metrics={
                        "accuracy": test_acc,
                        **(
                            {"rank_ic": fold_record.get("test_rank_ic", 0)}
                            if "test_rank_ic" in fold_record
                            else {}
                        ),
                    },
                    n_train=len(y_train),
                    n_test=len(y_test),
                )

        if not fold_results:
            return {"error": "No valid folds created"}

        # Train final model on all data
        self.model = self._create_model()
        self.model.fit(X_clean, y_clean)
        self.classes_ = list(self.model.classes_)

        results: Dict[str, Any] = {
            "mean_train_accuracy": float(np.mean(train_scores)) if train_scores else 0,
            "std_train_accuracy": float(np.std(train_scores)) if train_scores else 0,
            "mean_test_accuracy": float(np.mean(test_scores)) if test_scores else 0,
            "std_test_accuracy": float(np.std(test_scores)) if test_scores else 0,
            "overfit_gap": float(np.mean(train_scores) - np.mean(test_scores))
            if train_scores and test_scores
            else 0,
            "n_folds": len(fold_results),
            "fold_details": fold_results,
        }

        if test_ic:
            results["mean_test_rank_ic"] = float(np.mean(test_ic))
            results["std_test_rank_ic"] = float(np.std(test_ic))

        if hasattr(self.model, "feature_importances_"):
            results["feature_importance"] = dict(
                sorted(
                    zip(self.feature_names_, self.model.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True,
                )[:20]
            )

        if experiment_logger is not None:
            experiment_logger.log_config(
                hyperparams={
                    "model_type": self.model_type,
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                },
                features=self.feature_names_,
                cv_params={
                    "n_splits": n_splits,
                    "pct_embargo": pct_embargo,
                    "label_span": label_span,
                },
            )
            experiment_logger.log_feature_importance(
                mdi=results.get("feature_importance", {}),
            )
            experiment_logger.log_summary_verdict(
                mean_oos_metrics={
                    "accuracy": results["mean_test_accuracy"],
                    "rank_ic": results.get("mean_test_rank_ic", 0),
                    "overfit_gap": {"accuracy": results["overfit_gap"]},
                },
                n_folds=results["n_folds"],
            )
            experiment_logger.log_model(self.model)

        return results

    def walk_forward_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_size: int = 500,
        step_size: int = 100,
    ) -> Dict[str, List[float]]:
        """
        Walk-forward validation to detect overfitting.

        Args:
            X: Feature DataFrame
            y: Labels series
            train_size: Initial training window size
            step_size: Step size for expanding window

        Returns:
            Dict with train_score and test_score lists
        """
        from sklearn.metrics import accuracy_score

        results = {"train_score": [], "test_score": []}
        start = 0

        while start + train_size < len(X):
            end = min(start + train_size + step_size, len(X))

            X_train = X.iloc[start : start + train_size]
            y_train = y.iloc[start : start + train_size]
            X_test = X.iloc[start + train_size : end]
            y_test = y.iloc[start + train_size : end]

            if len(X_test) == 0:
                break

            valid_train = X_train.notna().all(axis=1) & y_train.notna()
            valid_test = X_test.notna().all(axis=1) & y_test.notna()

            if valid_train.sum() == 0 or valid_test.sum() == 0:
                start += step_size
                continue

            model = self._create_model()
            model.fit(X_train[valid_train], y_train[valid_train])

            train_pred = model.predict(X_train[valid_train])
            test_pred = model.predict(X_test[valid_test])

            results["train_score"].append(accuracy_score(y_train[valid_train], train_pred))
            results["test_score"].append(accuracy_score(y_test[valid_test], test_pred))

            start += step_size

        return results

    # Alias for backward compatibility
    purged_kfold_validation = train_with_purged_cv

    @staticmethod
    def compare_models(
        X: pd.DataFrame,
        y: pd.Series,
        forward_returns: pd.Series | None = None,
        model_types: List[str] | None = None,
        n_splits: int = 5,
    ) -> pd.DataFrame:
        """Compare multiple model types using PurgedKFold CV.

        Trains RF, GB, LightGBM, and Logistic Regression with identical
        data and CV splits. Reports accuracy, rank IC, and overfit gap
        for each model.

        Args:
            X: Feature DataFrame.
            y: Label series.
            forward_returns: Forward returns for IC evaluation.
            model_types: List of model types to compare (default: all).
            n_splits: Number of CV folds.

        Returns:
            DataFrame with comparison metrics per model.
        """
        if model_types is None:
            model_types = ["random_forest", "gradient_boosting", "lightgbm", "logistic_regression"]

        results = []
        for mt in model_types:
            if mt not in RegimeClassifier.SUPPORTED_MODELS:
                continue

            clf = RegimeClassifier(model_type=mt)
            result = clf.train_with_purged_cv(
                X, y, forward_returns=forward_returns, n_splits=n_splits
            )

            if "error" not in result:
                results.append(
                    {
                        "model": mt,
                        "mean_test_accuracy": result.get("mean_test_accuracy", 0),
                        "std_test_accuracy": result.get("std_test_accuracy", 0),
                        "mean_test_rank_ic": result.get("mean_test_rank_ic", 0),
                        "overfit_gap": result.get("overfit_gap", 0),
                        "n_folds": result.get("n_folds", 0),
                    }
                )

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).sort_values("mean_test_rank_ic", ascending=False)
