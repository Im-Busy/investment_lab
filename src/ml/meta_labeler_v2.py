"""
Meta-Labeler V2 — CatBoost Secondary Filter (B13).

Trains a CatBoost classifier to answer: "Given that the primary model generates
a signal, will this trade be profitable?" Uses features DIFFERENT from the
primary model: regime, volatility, signal-clustering, and recency — no
price/momentum/cross-asset features the primary model already captures.

Architecture:
    PrimaryModel.probability + MetaLabelContextFeatures
      -> CatBoostClassifier (PurgedKFold CV)
      -> MetaLabelerV2.filter(primary_signal) -> {take_trade: bool, confidence: float}

Reference: López de Prado / JFDS 2022, hudson-and-thames/meta-labeling
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .purged_cv import PurgedKFold
from .simple_meta_labeler import MetaLabelContextFeatures

logger = logging.getLogger(__name__)


@dataclass
class MetaLabelV2Result:
    """Result of MetaLabelerV2 training.

    Attributes:
        auc_mean: Mean AUC across PurgedKFold folds.
        auc_std: Std dev of AUC across folds.
        threshold: Optimal probability threshold (Youden's J).
        fold_metrics: Per-fold metrics.
        n_signals: Total signals labeled.
        n_profitable: Number of profitable signals.
        baseline_win_rate: Win rate without meta-labeling.
        feature_importance: Top features by importance.
    """

    auc_mean: float
    auc_std: float
    threshold: float
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)
    n_signals: int = 0
    n_profitable: int = 0
    baseline_win_rate: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "auc_mean": round(self.auc_mean, 4),
            "auc_std": round(self.auc_std, 4),
            "threshold": round(self.threshold, 4),
            "n_signals": self.n_signals,
            "n_profitable": self.n_profitable,
            "baseline_win_rate": round(self.baseline_win_rate, 4),
            "fold_metrics": self.fold_metrics,
            "top_features": dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        }


@dataclass
class MetaLabelV2Prediction:
    """Prediction for a single signal.

    Attributes:
        take_trade: Whether the meta-labeler recommends taking the trade.
        probability: Probability of trade being profitable.
        confidence: Confidence score (distance from 0.5, scaled to [0, 1]).
    """

    take_trade: bool
    probability: float
    confidence: float
    threshold: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "take_trade": self.take_trade,
            "probability": round(self.probability, 4),
            "confidence": round(self.confidence, 4),
            "threshold": self.threshold,
        }


class MetaLabelerV2:
    """CatBoost meta-labeler using regime/volatility/clustering/recency features.

    Unlike primary model features (price/momentum/returns/cross-asset), the
    meta-labeler uses features the primary model cannot access:
      1. Primary model probability (the signal being evaluated)
      2. ADX trend strength (regime context)
      3. vol_regime_ratio (current vol vs historical)
      4. signal_density (how many signals in recent window)
      5. days_since_signal (recency)
      6. prob_rolling_mean/std (signal distribution characteristics)

    Attributes:
        model: Trained CatBoostClassifier.
        feature_names_: List of feature column names.
        threshold_: Youden's J optimal probability threshold.
        context_gen: MetaLabelContextFeatures generator.
    """

    CATBOOST_PARAMS = {
        "iterations": 300,
        "depth": 4,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3,
        "random_state": 42,
        "verbose": False,
        "task_type": "CPU",
        "allow_writing_files": False,
    }

    def __init__(
        self,
        profit_horizon: int = 5,
        n_splits: int = 5,
        pct_embargo: float = 0.05,
        probability_threshold: float = 0.5,
    ) -> None:
        self.profit_horizon = profit_horizon
        self.n_splits = n_splits
        self.pct_embargo = pct_embargo
        self.threshold_ = probability_threshold

        self.model: Any = None
        self.feature_names_: List[str] = []
        self.context_gen = MetaLabelContextFeatures(
            adx_window=14,
            vol_window=20,
            vol_lookback=60,
            cluster_window=20,
        )

    def _compute_target(self, df: pd.DataFrame, signal_indices: pd.DatetimeIndex) -> pd.Series:
        """Compute binary target: was next trade profitable within horizon?

        Target = 1 if max(Close[next N bars]) > entry_price, else 0.
        """
        close = df["Close"]
        targets = []
        for ts in signal_indices:
            if ts not in df.index:
                targets.append(0)
                continue
            idx = df.index.get_loc(ts)
            entry = close.iloc[idx]
            future_end = min(idx + self.profit_horizon + 1, len(close))
            future_prices = close.iloc[idx + 1 : future_end].values
            if len(future_prices) == 0:
                targets.append(0)
                continue
            profitable = int(any(p > entry for p in future_prices))
            targets.append(profitable)
        return pd.Series(targets, index=signal_indices)

    def _build_features(
        self,
        df: pd.DataFrame,
        signal_indices: pd.DatetimeIndex,
        primary_probs: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Build meta-labeler feature matrix at signal bars.

        Features (all at signal bar, no forward-looking):
          1. primary_prob — primary model's probability score
          2. adx — ADX value (trend strength)
          3. adx_trend — 1 if ADX >= 25
          4. vol_regime_ratio — current vol / 60-bar avg vol
          5. vol_regime_high — 1 if vol_regime_ratio >= 1.5
          6. vol_regime_low — 1 if vol_regime_ratio <= 0.5
          7. recent_return_5d — 5-day cumulative return
          8. recent_return_20d — 20-day cumulative return
          9. up_days_ratio_20d — fraction of up days in last 20
          10. vol_skew — upside vol / downside vol
          11. signal_density — signals in last 20 bars / 20
          12. days_since_signal — bars since last signal
          13. prob_rolling_mean — 10-bar rolling mean of primary prob
          14. prob_rolling_std — 10-bar rolling std of primary prob
          15. prob_above_ma — 1 if prob >= rolling mean
        """
        # Generate base context features using shared generator
        ctx = self.context_gen.generate(
            df,
            signal_dates=signal_indices,
            primary_probs=primary_probs,
        )

        # Add primary probability as a feature
        if primary_probs is not None:
            aligned_prob = primary_probs.reindex(ctx.index)
            ctx["primary_prob"] = aligned_prob.fillna(0.5)
        else:
            ctx["primary_prob"] = 0.5

        return ctx

    def fit(
        self,
        df: pd.DataFrame,
        signal_indices: pd.DatetimeIndex,
        primary_probs: pd.Series | None = None,
    ) -> MetaLabelV2Result:
        """Train CatBoost meta-labeler on historical signals.

        Args:
            df: OHLCV DataFrame with columns [Open, High, Low, Close, Volume]
                and DatetimeIndex.
            signal_indices: DatetimeIndex of signal bars.
            primary_probs: Primary model probability scores (index-aligned).
                If None, default to 0.5 for all bars.

        Returns:
            MetaLabelV2Result with training metrics.
        """
        if len(signal_indices) < 50:
            logger.warning(
                "MetaLabelerV2: fewer than 50 signals (%d); skipping training",
                len(signal_indices),
            )
            return MetaLabelV2Result(
                auc_mean=0.5,
                auc_std=0.0,
                threshold=0.5,
                n_signals=len(signal_indices),
                n_profitable=0,
            )

        # Build features
        X = self._build_features(df, signal_indices, primary_probs)
        y = self._compute_target(df, signal_indices)

        # Align and drop NaN
        common = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[common]
        y = y.loc[common]

        if len(X) < 3 * self.n_splits:
            logger.warning(
                "MetaLabelerV2: insufficient data (%d samples) after cleaning",
                len(X),
            )
            return MetaLabelV2Result(
                auc_mean=0.5,
                auc_std=0.0,
                threshold=0.5,
                n_signals=len(X),
                n_profitable=int(y.sum()),
            )

        self.feature_names_ = list(X.columns)

        # PurgedKFold CV
        cv = PurgedKFold(
            n_splits=min(self.n_splits, len(X) // 3),
            pct_embargo=self.pct_embargo,
            label_span=self.profit_horizon,
        )

        try:
            from catboost import CatBoostClassifier
        except ImportError:
            logger.error("catboost not installed; cannot train MetaLabelerV2")
            return MetaLabelV2Result(
                auc_mean=0.5,
                auc_std=0.0,
                threshold=0.5,
                n_signals=len(X),
                n_profitable=int(y.sum()),
            )

        fold_metrics = []
        fold_aucs = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X), 1):
            if len(test_idx) < 5 or len(train_idx) < 10:
                continue
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if len(set(y_train)) < 2 or len(set(y_test)) < 2:
                continue

            model = CatBoostClassifier(**self.CATBOOST_PARAMS)
            model.fit(X_train, y_train)

            proba = model.predict_proba(X_test)[:, 1]
            from sklearn.metrics import roc_auc_score

            try:
                auc = float(roc_auc_score(y_test, proba))
            except ValueError:
                auc = 0.5

            fold_aucs.append(auc)
            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "auc": round(auc, 4),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                }
            )

        if not fold_aucs:
            return MetaLabelV2Result(
                auc_mean=0.5,
                auc_std=0.0,
                threshold=0.5,
                n_signals=len(X),
                n_profitable=int(y.sum()),
                baseline_win_rate=round(float(y.mean()), 4),
            )

        # Train final model on all data
        self.model = CatBoostClassifier(**self.CATBOOST_PARAMS)
        self.model.fit(X, y)

        # Find optimal threshold
        self.threshold_ = self._find_optimal_threshold(X, y)

        # Feature importance
        importance = dict(zip(self.feature_names_, self.model.feature_importances_))

        auc_mean = float(np.mean(fold_aucs))
        auc_std = float(np.std(fold_aucs))

        logger.info(
            "MetaLabelerV2: CatBoost AUC=%.4f ± %.4f, threshold=%.3f, n=%d",
            auc_mean,
            auc_std,
            self.threshold_,
            len(X),
        )

        return MetaLabelV2Result(
            auc_mean=round(auc_mean, 4),
            auc_std=round(auc_std, 4),
            threshold=round(self.threshold_, 4),
            fold_metrics=fold_metrics,
            n_signals=len(X),
            n_profitable=int(y.sum()),
            baseline_win_rate=round(float(y.mean()), 4),
            feature_importance=importance,
        )

    def predict(self, context: pd.DataFrame | Dict[str, Any]) -> MetaLabelV2Prediction:
        """Predict whether to take a signal given its context.

        Args:
            context: DataFrame row or dict with feature values matching
                the feature_names_ from training.

        Returns:
            MetaLabelV2Prediction with take_trade decision and probability.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if isinstance(context, dict):
            X = pd.DataFrame([context], columns=self.feature_names_)
        else:
            X = context[self.feature_names_]

        X = X.fillna(0)
        proba = float(self.model.predict_proba(X)[:, 1][0])
        take_trade = proba >= self.threshold_
        confidence = abs(proba - 0.5) * 2.0

        return MetaLabelV2Prediction(
            take_trade=take_trade,
            probability=proba,
            confidence=confidence,
            threshold=self.threshold_,
        )

    def predict_batch(self, contexts: pd.DataFrame) -> pd.DataFrame:
        """Predict take_trade for a batch of signal contexts.

        Args:
            contexts: DataFrame with context features.

        Returns:
            DataFrame with columns [meta_probability, meta_take_trade, meta_confidence].
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = contexts[self.feature_names_].fillna(0)
        proba = self.model.predict_proba(X)[:, 1]

        result = contexts.copy()
        result["meta_probability"] = proba
        result["meta_take_trade"] = proba >= self.threshold_
        result["meta_confidence"] = np.abs(proba - 0.5) * 2.0
        return result

    def _find_optimal_threshold(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Find optimal probability threshold using Youden's J statistic."""
        if self.model is None:
            return 0.5

        try:
            proba = self.model.predict_proba(X)[:, 1]
            from sklearn.metrics import roc_curve

            fpr, tpr, thresholds = roc_curve(y, proba)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            if best_idx < len(thresholds):
                return float(thresholds[best_idx])
            return 0.5
        except Exception:
            return 0.5

    def save(self, path: str | Path) -> None:
        """Save trained model to disk."""
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "feature_names": self.feature_names_,
            "threshold": self.threshold_,
            "config": {
                "profit_horizon": self.profit_horizon,
                "n_splits": self.n_splits,
                "pct_embargo": self.pct_embargo,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    def load(self, path: str | Path) -> None:
        """Load trained model from disk."""
        import pickle

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            model_data = pickle.load(f)  # nosec B301

        self.model = model_data["model"]
        self.feature_names_ = model_data["feature_names"]
        self.threshold_ = model_data.get("threshold", 0.5)
