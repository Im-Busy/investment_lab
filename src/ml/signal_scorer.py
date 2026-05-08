"""
ML Signal Scorer — Regression-based signal quality prediction.

Predicts forward returns (continuous) instead of binary profit/loss,
evaluated using Spearman rank IC as the primary metric.

Implements ML4T Ch12:05 methodology:
- Regression models (RF, GB, LightGBM, CatBoost)
- Spearman rank IC as primary evaluation metric
- PurgedKFold CV with embargo
- Walk-forward validation with expanding window
- Automated experiment logging

Architecture:
    SignalRegressor  →  predict forward returns (regression)
    SignalScorer     →  legacy classifier (kept for backward compat)

Usage:
    from src.ml.signal_scorer import SignalRegressor

    reg = SignalRegressor(model_type="gradient_boosting")
    result = reg.train(X, forward_returns)
    predictions = reg.predict(X_new)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScoredSignal:
    """A signal with ML-enhanced quality score."""

    timestamp: pd.Timestamp
    pattern_name: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    original_confidence: float
    ml_score: float
    ml_expected_return: float
    is_recommended: bool
    top_features: Dict[str, float] = field(default_factory=dict)


class SignalRegressor:
    """ML-based signal scorer using regression on forward returns.

    Predicts expected forward returns (continuous) rather than binary
    profit/loss. Evaluated using Spearman rank IC — the industry-standard
    metric for signal quality.

    Supports Random Forest, Gradient Boosting, LightGBM, and CatBoost
    regressors with PurgedKFold cross-validation.

    Example:
        >>> from src.ml.signal_scorer import SignalRegressor
        >>>
        >>> reg = SignalRegressor(model_type="gradient_boosting")
        >>> result = reg.train(X, forward_returns)
        >>> predictions = reg.predict(X_new)
        >>> ic = reg.evaluate_ic(X_test, forward_returns_test)
    """

    SUPPORTED_MODELS = ["gradient_boosting", "random_forest", "lightgbm", "catboost"]

    def __init__(
        self,
        model_type: str = "catboost",
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        random_state: int = 42,
        min_samples_leaf: int = 10,
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model type: {model_type}. Supported: {self.SUPPORTED_MODELS}"
            )

        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf
        self.model = None
        self.feature_names_: list[str] = []
        self._threshold = 0.001

    def _create_model(self):
        """Create regression model with regularization to prevent overfitting."""
        from sklearn.ensemble import (
            GradientBoostingRegressor,
            RandomForestRegressor,
        )

        if self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                subsample=0.8,
            )
        elif self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features="sqrt",
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth if self.max_depth else -1,
                num_leaves=min(2 ** (self.max_depth or 3), 31),
                learning_rate=self.learning_rate,
                min_child_samples=self.min_samples_leaf,
                feature_fraction=0.8,
                random_state=self.random_state,
                verbose=-1,
            )
        elif self.model_type == "catboost":
            try:
                from catboost import CatBoostRegressor

                return CatBoostRegressor(
                    iterations=self.n_estimators,
                    depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    l2_leaf_reg=3.0,
                    random_state=self.random_state,
                    verbose=False,
                    task_type="CPU",
                )
            except ImportError:
                raise ImportError("catboost not installed. Run: uv add catboost")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        purge_window: int = 5,
    ) -> Dict[str, Any]:
        """Train the regressor on features and forward returns.

        Args:
            X: Feature DataFrame.
            y: Forward return series (continuous labels).
            test_size: Fraction of data for testing.
            purge_window: Number of training samples at the split boundary
                to exclude from training (prevents label overlap leakage).
                Set to match the forward-return horizon used for labels.

        Returns:
            Dict with train/test IC, R², and feature importance.
        """
        from sklearn.metrics import r2_score
        from src.ml.metrics import compute_rank_ic, compute_ic, compute_hit_rate

        valid_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X.loc[valid_mask]
        y_clean = y.loc[valid_mask]

        if len(X_clean) < 50:
            return {"error": f"Only {len(X_clean)} valid samples after NaN filtering"}

        split_idx = int(len(X_clean) * (1 - test_size))
        train_end = max(0, split_idx - purge_window)
        X_train = X_clean.iloc[:train_end]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:train_end]
        y_test = y_clean.iloc[split_idx:]

        self.model = self._create_model()
        self.feature_names_ = list(X_train.columns)
        self.model.fit(X_train, y_train)

        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        train_pred_series = pd.Series(train_pred, index=y_train.index)
        test_pred_series = pd.Series(test_pred, index=y_test.index)

        train_ic_df = compute_rank_ic(train_pred_series, y_train)
        test_ic_df = compute_rank_ic(test_pred_series, y_test)
        test_pearson_df = compute_ic(test_pred_series, y_test)

        train_hit = compute_hit_rate(train_pred, y_train.values)
        test_hit = compute_hit_rate(test_pred, y_test.values)

        results: Dict[str, Any] = {
            "train_rank_ic": float(train_ic_df["rank_ic"].iloc[0]) if not train_ic_df.empty else 0,
            "test_rank_ic": float(test_ic_df["rank_ic"].iloc[0]) if not test_ic_df.empty else 0,
            "test_ic": float(test_pearson_df["ic"].iloc[0]) if not test_pearson_df.empty else 0,
            "train_r2": r2_score(y_train, train_pred),
            "test_r2": r2_score(y_test, test_pred),
            "train_hit_rate": train_hit,
            "test_hit_rate": test_hit,
            "overfit_gap_ic": float(
                abs(train_ic_df["rank_ic"].iloc[0]) - abs(test_ic_df["rank_ic"].iloc[0])
            )
            if not train_ic_df.empty and not test_ic_df.empty
            else 0,
            "n_train": len(y_train),
            "n_test": len(y_test),
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
        """Predict expected forward returns.

        Args:
            X: Feature DataFrame.

        Returns:
            Series of predicted returns.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        preds = self.model.predict(X.fillna(0))
        return pd.Series(preds, index=X.index, name="predicted_return")

    def evaluate_ic(self, X: pd.DataFrame, forward_returns: pd.Series) -> pd.DataFrame:
        """Evaluate predictions using IC metrics.

        Args:
            X: Feature DataFrame.
            forward_returns: Actual forward returns.

        Returns:
            DataFrame with rank_ic, ic, hit_rate.
        """
        from src.ml.metrics import compute_rank_ic, compute_ic, compute_hit_rate

        preds = self.predict(X)
        idx = preds.index.intersection(forward_returns.dropna().index)
        preds = preds.loc[idx]
        actuals = forward_returns.loc[idx]

        rank_ic_df = compute_rank_ic(preds, actuals)
        pearson_df = compute_ic(preds, actuals)
        hit = compute_hit_rate(preds.values, actuals.values)

        return pd.DataFrame(
            [
                {
                    "rank_ic": rank_ic_df["rank_ic"].iloc[0] if not rank_ic_df.empty else np.nan,
                    "ic": pearson_df["ic"].iloc[0] if not pearson_df.empty else np.nan,
                    "hit_rate": hit,
                    "n_samples": len(preds),
                }
            ]
        )

    def train_with_purged_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        pct_embargo: float = 0.01,
        label_span: int = 5,
        experiment_logger: Any = None,
    ) -> Dict[str, Any]:
        """Train and evaluate using PurgedKFold cross-validation.

        Uses Phase A PurgedKFold for temporal CV with embargo.
        Evaluates using rank IC as primary metric.

        Args:
            X: Feature DataFrame with datetime index.
            y: Forward returns (continuous).
            n_splits: Number of CV folds.
            pct_embargo: Fraction of test span to embargo.
            label_span: Forward label horizon for purging.
            experiment_logger: Optional ExperimentLogger instance.

        Returns:
            Dict with per-fold IC, overfit gap, feature importance.
        """
        from sklearn.metrics import r2_score
        from src.ml.purged_cv import PurgedKFold
        from src.ml.metrics import compute_rank_ic, compute_hit_rate

        X_clean = X.fillna(0).copy()
        y_clean = y.dropna().copy()
        idx = X_clean.index.intersection(y_clean.index)
        X_clean = X_clean.loc[idx]
        y_clean = y_clean.loc[idx]

        cv = PurgedKFold(n_splits=n_splits, pct_embargo=pct_embargo, label_span=label_span)

        fold_results: list[dict] = []
        test_ic_values: list[float] = []
        test_hit_rates: list[float] = []

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

            train_ic_df = compute_rank_ic(pd.Series(train_pred, index=y_train.index), y_train)
            test_ic_df = compute_rank_ic(pd.Series(test_pred, index=y_test.index), y_test)
            hit = compute_hit_rate(test_pred, y_test.values)

            train_ic_val = float(train_ic_df["rank_ic"].iloc[0]) if not train_ic_df.empty else 0
            test_ic_val = float(test_ic_df["rank_ic"].iloc[0]) if not test_ic_df.empty else 0

            test_ic_values.append(test_ic_val)
            test_hit_rates.append(hit)

            fold_record = {
                "fold": fold_idx,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "train_rank_ic": train_ic_val,
                "test_rank_ic": test_ic_val,
                "test_r2": r2_score(y_test, test_pred),
                "test_hit_rate": hit,
            }
            fold_results.append(fold_record)

            if experiment_logger is not None:
                experiment_logger.log_fold_metrics(
                    fold=fold_idx,
                    train_metrics={"rank_ic": train_ic_val},
                    test_metrics={"rank_ic": test_ic_val, "hit_rate": hit},
                    n_train=len(y_train),
                    n_test=len(y_test),
                    embargo_days=int(len(test_idx) * pct_embargo),
                )

        if not fold_results:
            return {"error": "No valid folds created"}

        self.model = self._create_model()
        self.model.fit(X_clean, y_clean)

        results: Dict[str, Any] = {
            "mean_test_rank_ic": float(np.mean(test_ic_values)) if test_ic_values else 0,
            "std_test_rank_ic": float(np.std(test_ic_values)) if test_ic_values else 0,
            "mean_test_hit_rate": float(np.mean(test_hit_rates)) if test_hit_rates else 0,
            "n_folds": len(fold_results),
            "fold_details": fold_results,
        }

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
                    "learning_rate": self.learning_rate,
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
                    "rank_ic": results["mean_test_rank_ic"],
                    "hit_rate": results["mean_test_hit_rate"],
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
    ) -> Dict[str, list]:
        """Walk-forward validation with IC tracking.

        Uses expanding window: train on [0:train_size], predict next step_size,
        expand window, repeat. Tracks rank IC at each step.

        Args:
            X: Feature DataFrame.
            y: Forward returns.
            train_size: Initial training window.
            step_size: Steps between re-fits.

        Returns:
            Dict with train_rank_ic, test_rank_ic, test_hit_rate lists.
        """
        from src.ml.metrics import compute_rank_ic, compute_hit_rate

        results: Dict[str, list] = {
            "train_rank_ic": [],
            "test_rank_ic": [],
            "test_hit_rate": [],
            "dates": [],
        }
        start = 0

        while start + train_size + step_size <= len(X):
            end = start + train_size + step_size
            window_start = X.index[start + train_size]

            X_train = X.iloc[start : start + train_size]
            y_train = y.iloc[start : start + train_size]
            X_test = X.iloc[start + train_size : end]
            y_test = y.iloc[start + train_size : end]

            valid_train = X_train.notna().all(axis=1) & y_train.notna()
            valid_test = X_test.notna().all(axis=1) & y_test.notna()

            if valid_train.sum() < 50 or valid_test.sum() < 10:
                start += step_size
                continue

            model = self._create_model()
            model.fit(X_train[valid_train], y_train[valid_train])

            train_pred = model.predict(X_train[valid_train])
            test_pred = model.predict(X_test[valid_test])

            train_ic = compute_rank_ic(
                pd.Series(train_pred, index=y_train[valid_train].index), y_train[valid_train]
            )
            test_ic = compute_rank_ic(
                pd.Series(test_pred, index=y_test[valid_test].index), y_test[valid_test]
            )
            hit = compute_hit_rate(test_pred, y_test[valid_test].values)

            results["train_rank_ic"].append(
                float(train_ic["rank_ic"].iloc[0]) if not train_ic.empty else 0
            )
            results["test_rank_ic"].append(
                float(test_ic["rank_ic"].iloc[0]) if not test_ic.empty else 0
            )
            results["test_hit_rate"].append(hit)
            results["dates"].append(str(window_start))

            start += step_size

        return results

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance scores.

        Args:
            top_n: Number of top features.

        Returns:
            DataFrame with feature and importance columns.
        """
        if not hasattr(self.model, "feature_importances_"):
            if hasattr(self.model, "coef_"):
                importance = np.abs(self.model.coef_)
                return (
                    pd.DataFrame({"feature": self.feature_names_, "importance": importance})
                    .sort_values("importance", ascending=False)
                    .head(top_n)
                )
            return pd.DataFrame()

        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "importance": self.model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(top_n)
        )

    def compute_permutation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 5,
    ) -> pd.DataFrame:
        """Compute permutation feature importance.

        Args:
            X: Feature DataFrame.
            y: Forward returns.
            n_repeats: Number of permutations per feature.

        Returns:
            DataFrame with feature, importance_mean, importance_std.
        """
        from sklearn.inspection import permutation_importance

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_clean = X.fillna(0)
        idx = X_clean.index.intersection(y.dropna().index)
        X_clean = X_clean.loc[idx]
        y_clean = y.loc[idx]

        result = permutation_importance(
            self.model,
            X_clean,
            y_clean,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring="neg_mean_squared_error",
        )

        return pd.DataFrame(
            {
                "feature": self.feature_names_,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)

    @staticmethod
    def compare_models(
        X: pd.DataFrame,
        y: pd.Series,
        model_types: List[str] | None = None,
        n_splits: int = 5,
    ) -> pd.DataFrame:
        """Compare regression models using PurgedKFold CV.

        Args:
            X: Feature DataFrame.
            y: Forward returns.
            model_types: List of model types to compare.
            n_splits: Number of CV folds.

        Returns:
            DataFrame with rank_ic and hit_rate per model.
        """
        if model_types is None:
            model_types = ["gradient_boosting", "random_forest", "lightgbm", "catboost"]

        results = []
        for mt in model_types:
            if mt not in SignalRegressor.SUPPORTED_MODELS and mt != "catboost":
                continue

            try:
                reg = SignalRegressor(model_type=mt)
                result = reg.train_with_purged_cv(X, y, n_splits=n_splits)
                if "error" not in result:
                    results.append(
                        {
                            "model": mt,
                            "mean_test_rank_ic": result.get("mean_test_rank_ic", 0),
                            "std_test_rank_ic": result.get("std_test_rank_ic", 0),
                            "mean_test_hit_rate": result.get("mean_test_hit_rate", 0),
                            "n_folds": result.get("n_folds", 0),
                        }
                    )
            except ImportError:
                logger.warning(f"Skipping {mt}: not installed")
            except Exception as e:
                logger.warning(f"Skipping {mt}: {e}")

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).sort_values("mean_test_rank_ic", ascending=False)


# ── Legacy classifier (kept for backward compatibility) ──────────────────


class SignalScorer:
    """Legacy ML-based signal quality scorer (classification).

    Predicts whether a signal will be profitable (binary classification).
    For new work, use SignalRegressor which predicts forward returns
    (regression) evaluated via Spearman rank IC.

    Example:
        >>> scorer = SignalScorer(model_type="gradient_boosting")
        >>> scorer.train(X_train, y_train)
        >>> scores = scorer.score(X_new)
    """

    SUPPORTED_MODELS = [
        "catboost",
        "gradient_boosting",
        "random_forest",
        "logistic_regression",
    ]

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        n_estimators: int = 100,
        max_depth: int = 4,
        random_state: int = 42,
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model type: {model_type}. Supported: {self.SUPPORTED_MODELS}"
            )

        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.feature_names_ = None
        self.threshold_ = 0.5

    def _create_model(self):
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            RandomForestClassifier,
        )
        from sklearn.linear_model import LogisticRegression

        if self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                class_weight="balanced",
            )
        elif self.model_type == "logistic_regression":
            return LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state,
            )
        elif self.model_type == "catboost":
            try:
                from catboost import CatBoostClassifier

                return CatBoostClassifier(
                    iterations=self.n_estimators,
                    depth=self.max_depth,
                    learning_rate=0.03,
                    l2_leaf_reg=3.0,
                    random_state=self.random_state,
                    verbose=False,
                    loss_function="Logloss",
                    task_type="CPU",
                )
            except ImportError:
                raise ImportError("catboost not installed. Run: uv add catboost")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            precision_recall_curve,
            roc_auc_score,
        )

        valid_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        if len(X_clean) < 2:
            raise ValueError(
                f"Need at least 2 samples for training, got {len(X_clean)}. "
                "The strategy may not have generated enough trades."
            )

        split_idx = int(len(X_clean) * 0.7)
        X_train = X_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:split_idx]
        y_test = y_clean.iloc[split_idx:]

        self.model = self._create_model()
        self.feature_names_ = list(X_train.columns)
        self.threshold_ = threshold
        self.model.fit(X_train, y_train)

        train_pred = self.model.predict(X_train)

        if len(X_test) > 0:
            test_pred = self.model.predict(X_test)
            test_proba = self.model.predict_proba(X_test)[:, 1]
            test_accuracy = accuracy_score(y_test, test_pred)
            test_auc_roc = roc_auc_score(y_test, test_proba)
            if threshold is None:
                precision, recall, thresholds = precision_recall_curve(y_test, test_proba)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                optimal_idx = np.argmax(f1_scores)
                self.threshold_ = thresholds[min(optimal_idx, len(thresholds) - 1)]
            test_precision = (
                float((y_test[test_pred == 1] == 1).mean()) if (test_pred == 1).any() else 0
            )
            test_report = classification_report(y_test, test_pred, output_dict=True)
        else:
            test_pred = None
            test_proba = None
            test_accuracy = None
            test_auc_roc = None
            test_precision = None
            test_report = {}

        results = {
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": test_accuracy,
            "test_auc_roc": test_auc_roc,
            "test_precision": test_precision,
            "test_report": test_report,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "optimal_threshold": self.threshold_,
            "n_profitable_train": int(y_train.sum()),
            "n_profitable_test": int(y_test.sum()),
        }

        if hasattr(self.model, "feature_importances_"):
            results["feature_importance"] = dict(
                sorted(
                    zip(self.feature_names_, self.model.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True,
                )[:15]
            )

        return results

    def score(
        self,
        X: pd.DataFrame,
        signals: Optional[List[Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_clean = X.fillna(0)
        proba = self.model.predict_proba(X_clean)[:, 1]

        scored = X.copy()
        scored["ml_score"] = proba
        scored["is_recommended"] = proba >= self.threshold_

        return scored

    def walk_forward_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_size: int = 500,
        step_size: int = 100,
    ) -> Dict[str, List[float]]:
        from sklearn.metrics import accuracy_score, roc_auc_score

        results = {"train_accuracy": [], "test_accuracy": [], "test_auc": []}
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

            if valid_train.sum() < 50 or valid_test.sum() < 10:
                start += step_size
                continue

            model = self._create_model()
            model.fit(X_train[valid_train], y_train[valid_train])

            train_pred = model.predict(X_train[valid_train])
            test_pred = model.predict(X_test[valid_test])

            try:
                test_proba = model.predict_proba(X_test[valid_test])[:, 1]
                test_auc = roc_auc_score(y_test[valid_test], test_proba)
            except Exception:
                test_auc = 0.5

            results["train_accuracy"].append(accuracy_score(y_train[valid_train], train_pred))
            results["test_accuracy"].append(accuracy_score(y_test[valid_test], test_pred))
            results["test_auc"].append(test_auc)

            start += step_size

        return results

    def get_top_features_by_importance(self, top_n: int = 10) -> pd.DataFrame:
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
