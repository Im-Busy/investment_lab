"""
Dynamic Ensemble Learning (DEL) for regime-adaptive prediction.

When a single model degrades under regime shift (18/57 features shift OOS,
KS=0.62 for ATR, p=10^-109), no single architecture dominates across regimes.
DEL maintains an ensemble of models and dynamically weights them.

Based on OneNet (NeurIPS 2023) and the IJIMAI 2023 survey of 223 papers
on ML for financial prediction under regime change.

FIX (Phase 12c P2-3, 2026-05-16):
  - Added stacking_method config: "stacking" (default), "egd", or "voting"
  - Stacking meta-model (LogisticRegression) outperforms EGD averaging
    because EGD washed out sparse signals and couldn't accumulate meaningful
    weights in 60-bar windows.
  - EGD reinit_every increased to 252 (1yr) for less aggressive reset.
  - Added predict_proba_from_base() to extract per-model probabilities.
  - Added weight_history tracking for diagnostics.

Usage:
    from src.ml.dynamic_ensemble import DynamicEnsemble

    del_model = DynamicEnsemble()
    del_model.fit(X_train, y_train)
    prob_t = del_model.predict(X_test.iloc[[i]])

    # After outcome known:
    del_model.update_weights(actual_return_t)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DynamicEnsembleConfig:
    """Configuration for the dynamic ensemble."""

    stacking_method: str = "stacking"
    eta: float = 0.1
    reinit_every: int = 252
    n_variants: int = 5
    random_state: int = 42
    model_type: str = "catboost"
    n_estimators: int = 100
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    variant_params: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.variant_params:
            self.variant_params = [
                {
                    "max_depth": 3,
                    "l2_leaf_reg": 10.0,
                    "random_strength": 3.0,
                    "min_data_in_leaf": 50,
                },
                {
                    "max_depth": 2,
                    "l2_leaf_reg": 10.0,
                    "random_strength": 3.0,
                    "min_data_in_leaf": 30,
                },
                {
                    "max_depth": 4,
                    "l2_leaf_reg": 10.0,
                    "random_strength": 3.0,
                    "min_data_in_leaf": 50,
                },
                {
                    "max_depth": 3,
                    "l2_leaf_reg": 20.0,
                    "random_strength": 5.0,
                    "min_data_in_leaf": 80,
                },
                {
                    "max_depth": 3,
                    "l2_leaf_reg": 10.0,
                    "random_strength": 3.0,
                    "min_data_in_leaf": 50,
                    "learning_rate": 0.01,
                },
            ]


class DynamicEnsemble:
    """Ensemble of CatBoost models with stacking, EGD, or voting.

    Three modes (controlled by config.stacking_method):
      - "stacking" (default): LogisticRegression meta-model on base predictions.
        Best for sparse financial signals - avoids the averaging washout problem.
      - "egd": Exponential Gradient Descent online weight updates.
      - "voting": Equal-weight averaging of base model probabilities.

    Attributes:
        config: Ensemble configuration.
        models: List of trained PatternClassifier instances.
        weights: Current weight vector (sums to 1). Used for EGD/voting.
        meta_model: LogisticRegression for stacking mode.
        last_prediction: Per-model probabilities from most recent predict().
        bar_count: Bars processed since last re-initialization.
        is_fitted: Whether fit() has been called.
        feature_names: Feature column names from training data.
        weight_history: List of weight snapshots for diagnostics.
    """

    def __init__(self, config: DynamicEnsembleConfig | None = None) -> None:
        self.config = config or DynamicEnsembleConfig()
        self.models: list[Any] = []
        self.weights: np.ndarray = np.ones(self.config.n_variants) / self.config.n_variants
        self.meta_model: Any = None
        self.last_prediction: np.ndarray | None = None
        self.bar_count: int = 0
        self.is_fitted: bool = False
        self.feature_names: list[str] = []
        self.weight_history: list[dict[str, Any]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DynamicEnsemble":
        """Train all variant models and optionally fit stacking meta-model.

        Args:
            X: Feature DataFrame.
            y: Binary labels (1 = profitable, 0 = unprofitable).

        Returns:
            Self for chaining.
        """
        from src.ml.pattern_classifier import PatternClassifier

        X_clean = X.fillna(0)
        self.feature_names = list(X_clean.columns)

        split_idx = int(len(X_clean) * 0.7)
        X_train, X_cal = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
        y_train, y_cal = y.iloc[:split_idx], y.iloc[split_idx:]

        self.models = []
        for i, params in enumerate(self.config.variant_params):
            clf = PatternClassifier(
                model_type=self.config.model_type,
                n_estimators=self.config.n_estimators,
                learning_rate=params.get("learning_rate", self.config.learning_rate),
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                random_state=self.config.random_state + i,
                max_depth=params.get("max_depth", 3),
                l2_leaf_reg=params.get("l2_leaf_reg", 10.0),
                random_strength=params.get("random_strength", 3.0),
                min_data_in_leaf=params.get("min_data_in_leaf", 50),
            )
            clf.train(X_train, y_train, calibration_data=(X_cal, y_cal))
            self.models.append(clf)

        if self.config.stacking_method == "stacking":
            self._fit_meta_model(X_cal, y_cal)

        self.weights = np.ones(self.config.n_variants) / self.config.n_variants
        self.bar_count = 0
        self.weight_history = []
        self.is_fitted = True
        return self

    def _fit_meta_model(self, X_cal: pd.DataFrame, y_cal: pd.Series) -> None:
        """Fit LogisticRegression meta-model on base model predictions."""
        from sklearn.linear_model import LogisticRegression

        if len(X_cal) < 20:
            logger.debug("Insufficient calibration data for stacking; falling back to voting")
            self.config.stacking_method = "voting"
            return

        base_preds = np.column_stack(
            [
                m.predict(X_cal[self.feature_names].fillna(0))["probability_profitable"].values
                for m in self.models
            ]
        )

        self.meta_model = LogisticRegression(
            max_iter=1000,
            random_state=self.config.random_state,
            class_weight="balanced",
        )
        self.meta_model.fit(base_preds, y_cal.values)
        logger.info(
            f"Stacking meta-model fitted on {len(X_cal)} samples. "
            f"Coefs: {np.round(self.meta_model.coef_[0], 3)}"
        )

    def _get_base_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get per-model probability predictions.

        Returns:
            Array of shape (n_samples, n_models).
        """
        X_clean = X[self.feature_names].fillna(0)
        return np.column_stack(
            [m.predict(X_clean)["probability_profitable"].values for m in self.models]
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return ensemble probability for each row in X.

        Args:
            X: Feature DataFrame (must have same columns as training data).

        Returns:
            Array of ensemble probabilities (shape: n_samples,).
        """
        if not self.is_fitted:
            raise RuntimeError("DynamicEnsemble not fitted. Call fit() first.")

        base_preds = self._get_base_predictions(X)
        self.last_prediction = base_preds

        if self.config.stacking_method == "stacking" and self.meta_model is not None:
            ensemble_prob = self.meta_model.predict_proba(base_preds)[:, 1]
        elif self.config.stacking_method == "egd":
            ensemble_prob = (base_preds @ self.weights) / self.weights.sum()
        else:
            ensemble_prob = base_preds.mean(axis=1)

        return ensemble_prob

    def predict_proba_from_base(self, X: pd.DataFrame) -> np.ndarray:
        """Return per-model probabilities without ensembling.

        Use for diagnostics and stacking meta-features.

        Returns:
            Array of shape (n_samples, n_models).
        """
        return self._get_base_predictions(X)

    def update_weights(self, actual_return: float) -> None:
        """Update model weights via EGD on prediction error.

        Uses squared error loss: (pred - target)^2 where target = sign(return).
        For stacking mode, this is a no-op as weights are controlled by meta-model.

        Args:
            actual_return: Realized return for the bar whose prediction
                is stored in last_prediction.
        """
        if self.last_prediction is None or self.config.stacking_method != "egd":
            return

        target = 1.0 if actual_return > 0 else 0.0
        losses = (self.last_prediction[-1] - target) ** 2

        self.weights *= np.exp(-self.config.eta * losses)
        weight_sum = self.weights.sum()
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.config.n_variants) / self.config.n_variants

        self.bar_count += 1

        self.weight_history.append(
            {
                "bar": self.bar_count,
                **{f"w_{i}": float(self.weights[i]) for i in range(len(self.weights))},
            }
        )

        # Re-initialize weights every K bars (OneNet, NeurIPS 2023)
        if self.bar_count >= self.config.reinit_every:
            self.weights = np.ones(self.config.n_variants) / self.config.n_variants
            self.bar_count = 0

    def save(self, path: str | Path) -> None:
        """Save ensemble models and state to disk.

        Args:
            path: Directory or .pkl file path for the ensemble.
        """
        path = Path(path)
        if path.suffix == ".pkl":
            path.parent.mkdir(parents=True, exist_ok=True)
            ensemble_dir = path.parent / path.stem
        else:
            ensemble_dir = path

        ensemble_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "config": self.config,
            "weights": self.weights.tolist(),
            "bar_count": self.bar_count,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "model_paths": [str(ensemble_dir / f"model_{i}.pkl") for i in range(len(self.models))],
            "weight_history": self.weight_history,
        }

        for i, model in enumerate(self.models):
            model.save(data["model_paths"][i])

        meta_path = ensemble_dir / "meta_model.pkl"
        if self.meta_model is not None:
            with open(meta_path, "wb") as f:
                pickle.dump(self.meta_model, f)
            data["meta_model_path"] = str(meta_path)
        else:
            data["meta_model_path"] = ""

        with open(ensemble_dir / "ensemble.pkl", "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str | Path) -> None:
        """Load ensemble models and state from disk.

        Args:
            path: Directory or .pkl file path for the ensemble.
        """
        from src.ml.pattern_classifier import PatternClassifier

        path = Path(path)
        if path.suffix == ".pkl":
            ensemble_dir = path.parent / path.stem
        else:
            ensemble_dir = path

        with open(ensemble_dir / "ensemble.pkl", "rb") as f:
            data = pickle.load(f)

        self.config = data["config"]
        self.weights = np.array(data["weights"])
        self.bar_count = data["bar_count"]
        self.is_fitted = data["is_fitted"]
        self.feature_names = data["feature_names"]
        self.weight_history = data.get("weight_history", [])

        self.models = []
        for mp in data["model_paths"]:
            model = PatternClassifier()
            model.load(mp)
            self.models.append(model)

        meta_path = data.get("meta_model_path", "")
        if meta_path and Path(meta_path).exists():
            with open(meta_path, "rb") as f:
                self.meta_model = pickle.load(f)
        else:
            self.meta_model = None

        self.last_prediction = None

    def get_weight_history(self) -> dict[str, list[float]]:
        """Return current weights and historical weight snapshots.

        Returns:
            Dict with per-variant current weights and weight_history list.
        """
        current = {f"variant_{i}": float(w) for i, w in enumerate(self.weights)}
        current["weight_history"] = self.weight_history
        return current

    def get_method_info(self) -> dict[str, Any]:
        """Return information about the active ensembling method.

        Returns:
            Dict with method, n_models, meta_coefficients (if stacking).
        """
        info: dict[str, Any] = {
            "method": self.config.stacking_method,
            "n_models": len(self.models),
            "reinit_every": self.config.reinit_every,
            "eta": self.config.eta,
        }
        if self.meta_model is not None:
            info["meta_coefficients"] = self.meta_model.coef_.tolist()
            info["meta_intercept"] = float(self.meta_model.intercept_[0])
        return info
