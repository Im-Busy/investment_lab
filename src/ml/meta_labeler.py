"""
Signal Meta-Labeling (T9): "Should I take this signal?"

Separate model from "which direction?" — predicts whether a trade will hit
take-profit before stop-loss within a time limit, given the market context
at signal generation time.

Uses TripleBarrierLabeler for target generation and PurgedKFold for
temporal cross-validation. Default model is LightGBM; CatBoost is
substituted only when it demonstrates >=10% AUC improvement.

Architecture:
    TradeHistory + TripleBarrierLabeler
      -> labeled dataset (context features, hit_tp_or_not)
      -> LGBMClassifier (PurgedKFold CV)
      -> MetaLabeler.predict(signal) -> {take_trade: bool, confidence: float}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .triple_barrier import TripleBarrierLabeler
from .purged_cv import PurgedKFold

logger = logging.getLogger(__name__)


@dataclass
class MetaLabelResult:
    """Result of meta-labeler training.

    Attributes:
        lgbm_auc: LightGBM mean AUC across PurgedKFold folds.
        catboost_auc: CatBoost mean AUC (None if not trained).
        model_used: Which model was selected ('lgbm' | 'catboost').
        fold_metrics: Per-fold metrics for the selected model.
        n_signals: Total signals labeled.
        n_profitable: Number of profitable signals.
        baseline_win_rate: Win rate without meta-labeling.
        feature_importance: Top features by importance.
    """

    lgbm_auc: float
    catboost_auc: Optional[float]
    model_used: str
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)
    n_signals: int = 0
    n_profitable: int = 0
    baseline_win_rate: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lgbm_auc": round(self.lgbm_auc, 4),
            "catboost_auc": round(self.catboost_auc, 4) if self.catboost_auc else None,
            "model_used": self.model_used,
            "n_signals": self.n_signals,
            "n_profitable": self.n_profitable,
            "baseline_win_rate": round(self.baseline_win_rate, 4),
            "fold_metrics": self.fold_metrics,
            "top_features": dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        }


@dataclass
class MetaLabelPrediction:
    """Prediction for a single signal.

    Attributes:
        take_trade: Whether the meta-labeler recommends taking the trade.
        probability: Probability of hitting TP before SL.
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


class MetaLabeler:
    """Train a model to predict whether a signal will lead to a profitable trade.

    Given historical signals (pattern detections), labels each with the
    TripleBarrierLabeler (did TP hit before SL within time_limit?), builds
    context features describing the market state at signal time, and trains
    a binary classifier to predict which signals are worth taking.

    Attributes:
        tp_atr_mult: ATR multiplier for take-profit barrier.
        sl_atr_mult: ATR multiplier for stop-loss barrier.
        time_limit: Maximum bars until vertical barrier.
        n_splits: PurgedKFold splits.
        pct_embargo: Embargo percentage.
        model: Trained model (LGBMClassifier or CatBoostClassifier).
        model_used: 'lgbm' or 'catboost'.
        feature_names_: List of feature column names.
        probability_threshold_: Threshold for take_trade decision (fitted).
    """

    LGBM_PARAMS = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    }

    CATBOOST_PARAMS = {
        "iterations": 200,
        "depth": 5,
        "learning_rate": 0.05,
        "min_data_in_leaf": 20,
        "random_state": 42,
        "verbose": False,
        "task_type": "CPU",
    }

    def __init__(
        self,
        tp_atr_mult: float = 2.0,
        sl_atr_mult: float = 1.5,
        time_limit: int = 20,
        n_splits: int = 5,
        pct_embargo: float = 0.05,
        probability_threshold: float = 0.5,
    ) -> None:
        self.tp_atr_mult = tp_atr_mult
        self.sl_atr_mult = sl_atr_mult
        self.time_limit = time_limit
        self.n_splits = n_splits
        self.pct_embargo = pct_embargo
        self.probability_threshold_ = probability_threshold

        self.model: Any = None
        self.model_used: Optional[str] = None
        self.feature_names_: List[str] = []
        self._labeler = TripleBarrierLabeler(atr_mult_tp=tp_atr_mult, atr_mult_sl=sl_atr_mult)

    def fit(
        self,
        df: pd.DataFrame,
        signals_df: pd.DataFrame,
    ) -> MetaLabelResult:
        """Train the meta-labeler on historical signals.

        Args:
            df: OHLCV DataFrame with columns [Open, High, Low, Close, Volume]
                and DatetimeIndex.
            signals_df: DataFrame of historical signals with columns:
                [timestamp, pattern_name, direction, entry_price, stop_loss,
                 take_profit_1, confidence].

        Returns:
            MetaLabelResult with training metrics and model selection.
        """
        labeled = self._label_signals(df, signals_df)
        X, y = self._build_features(df, labeled)

        if len(X) < 50:
            logger.warning("Fewer than 50 labeled signals; meta-labeler may be unreliable")
        if len(X) < 20:
            return MetaLabelResult(
                lgbm_auc=0.5,
                catboost_auc=None,
                model_used="none",
                n_signals=len(X),
                n_profitable=int(y.sum()) if len(y) > 0 else 0,
            )

        self.feature_names_ = list(X.columns)

        lgbm_auc, lgbm_folds = self._train_lgbm_cv(X, y)
        catboost_auc, catboost_folds = self._train_catboost_cv(X, y)

        if catboost_auc is not None and lgbm_auc > 0 and catboost_auc >= 1.10 * lgbm_auc:
            self.model_used = "catboost"
            self.model = self._train_catboost_final(X, y)
            fold_metrics = catboost_folds
            logger.info(
                "MetaLabeler: CatBoost selected (AUC %.4f >= 1.10 * %.4f)", catboost_auc, lgbm_auc
            )
        else:
            self.model_used = "lgbm"
            self.model = self._train_lgbm_final(X, y)
            fold_metrics = lgbm_folds
            logger.info("MetaLabeler: LightGBM selected (AUC %.4f)", lgbm_auc)

        self.probability_threshold_ = self._find_optimal_threshold(X, y)
        feature_importance = self._extract_importance()

        return MetaLabelResult(
            lgbm_auc=round(lgbm_auc, 4),
            catboost_auc=round(catboost_auc, 4) if catboost_auc else None,
            model_used=self.model_used,
            fold_metrics=fold_metrics,
            n_signals=len(X),
            n_profitable=int(y.sum()),
            baseline_win_rate=round(float(y.mean()), 4),
            feature_importance=feature_importance,
        )

    def predict(self, context: pd.DataFrame | Dict[str, Any]) -> MetaLabelPrediction:
        """Predict whether to take a signal given its context.

        Args:
            context: DataFrame row or dict with feature values matching
                the feature_names_ from training.

        Returns:
            MetaLabelPrediction with take_trade decision and probability.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if isinstance(context, dict):
            X = pd.DataFrame([context], columns=self.feature_names_)
        else:
            X = context[self.feature_names_]

        X = X.fillna(0)
        proba = float(self.model.predict_proba(X)[:, 1][0])
        take_trade = proba >= self.probability_threshold_
        confidence = abs(proba - 0.5) * 2.0

        return MetaLabelPrediction(
            take_trade=take_trade,
            probability=proba,
            confidence=confidence,
            threshold=self.probability_threshold_,
        )

    def predict_batch(self, contexts: pd.DataFrame) -> pd.DataFrame:
        """Predict take_trade for a batch of signal contexts.

        Args:
            contexts: DataFrame with context features.

        Returns:
            DataFrame with columns [probability, take_trade, confidence].
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = contexts[self.feature_names_].fillna(0)
        proba = self.model.predict_proba(X)[:, 1]

        result = contexts.copy()
        result["meta_probability"] = proba
        result["meta_take_trade"] = proba >= self.probability_threshold_
        result["meta_confidence"] = np.abs(proba - 0.5) * 2.0
        return result

    def _label_signals(self, df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Label historical signals using TripleBarrierLabeler.

        Returns a DataFrame with columns: timestamp, pattern_name, direction,
        label (+1/-1/0), return_pct, barrier.
        """
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        labels = []
        for _, signal in signals_df.iterrows():
            ts = signal.get("timestamp")
            if ts is None or ts not in df.index:
                continue

            idx = df.index.get_loc(ts)
            entry = signal.get("entry_price", close.iloc[idx])

            future_close = close.iloc[idx + 1 : idx + 1 + self.time_limit].values
            future_high = high.iloc[idx + 1 : idx + 1 + self.time_limit].values
            future_low = low.iloc[idx + 1 : idx + 1 + self.time_limit].values

            tp = entry * (1 + self.tp_atr_mult * 0.02)
            sl = entry * (1 - self.sl_atr_mult * 0.02)

            barrier_label = self._labeler.fit_single(
                entry_price=entry,
                close_arr=future_close,
                high_arr=future_high,
                low_arr=future_low,
                take_profit=tp,
                stop_loss=sl,
                time_limit=self.time_limit,
            )

            labels.append(
                {
                    "timestamp": ts,
                    "pattern_name": signal.get("pattern_name", "unknown"),
                    "direction": str(signal.get("direction", "long")),
                    "entry_price": entry,
                    "confidence": float(signal.get("confidence", 0.5)),
                    "label": barrier_label.label,
                    "return_pct": barrier_label.return_pct,
                    "barrier": barrier_label.barrier,
                    "bars_to_exit": barrier_label.bars_to_exit,
                }
            )

        if not labels:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "pattern_name",
                    "direction",
                    "entry_price",
                    "confidence",
                    "label",
                    "return_pct",
                    "barrier",
                    "bars_to_exit",
                ]
            )

        return pd.DataFrame(labels)

    @staticmethod
    def _compute_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Compute ATR indicator."""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI indicator."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    def _build_features(
        self, df: pd.DataFrame, labeled: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build context features at each signal timestamp.

        Features:
            - pattern_confidence: Original signal confidence
            - atr_ratio: ATR / Close
            - rsi_value: 14-bar RSI
            - trend_20: 20-bar return (%)
            - trend_50: 50-bar return (%)
            - volatility_20: 20-bar std dev of returns
            - volume_ratio: Volume / 20-bar avg volume
            - relative_position: (Close - 50-bar_low) / (50-bar_high - 50-bar_low)
            - day_of_week: 0=Monday, 4=Friday
            - hour_of_day: Hour of signal timestamp
            - is_reversal: 1 if pattern_name contains reversal keywords
            - is_breakout: 1 if pattern_name contains breakout keywords

        Returns:
            (X, y) where X is feature DataFrame and y is binary label
            (1 if barrier_label.label == +1, 0 otherwise).
        """
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df.get("Volume", pd.Series(1, index=df.index))

        atr = self._compute_atr(high, low, close, period=14)
        rsi = self._compute_rsi(close, period=14)

        returns = close.pct_change()
        volatility_20 = returns.rolling(20).std()
        volume_avg_20 = volume.rolling(20).mean()
        high_50 = high.rolling(50).max()
        low_50 = low.rolling(50).min()

        rows = []
        for _, row in labeled.iterrows():
            ts = row["timestamp"]
            if ts not in df.index:
                continue
            idx = df.index.get_loc(ts)

            if idx < 50:
                continue

            c = close.iloc[idx]
            rows.append(
                {
                    "timestamp": ts,
                    "pattern_confidence": float(row.get("confidence", 0.5)),
                    "atr_ratio": float(atr.iloc[idx] / c)
                    if c > 0 and pd.notna(atr.iloc[idx])
                    else 0.0,
                    "rsi_value": float(rsi.iloc[idx]) if pd.notna(rsi.iloc[idx]) else 50.0,
                    "trend_20": float((c / close.iloc[max(0, idx - 20)] - 1))
                    if close.iloc[max(0, idx - 20)] > 0
                    else 0.0,
                    "trend_50": float((c / close.iloc[max(0, idx - 50)] - 1))
                    if close.iloc[max(0, idx - 50)] > 0
                    else 0.0,
                    "volatility_20": float(volatility_20.iloc[idx])
                    if pd.notna(volatility_20.iloc[idx])
                    else 0.0,
                    "volume_ratio": float(volume.iloc[idx] / volume_avg_20.iloc[idx])
                    if pd.notna(volume_avg_20.iloc[idx]) and volume_avg_20.iloc[idx] > 0
                    else 1.0,
                    "relative_position": float(
                        (c - low_50.iloc[idx]) / (high_50.iloc[idx] - low_50.iloc[idx])
                    )
                    if pd.notna(high_50.iloc[idx])
                    and pd.notna(low_50.iloc[idx])
                    and high_50.iloc[idx] != low_50.iloc[idx]
                    else 0.5,
                    "day_of_week": float(ts.dayofweek) if hasattr(ts, "dayofweek") else 0.0,
                    "hour_of_day": float(ts.hour) if hasattr(ts, "hour") else 0.0,
                    "is_reversal": 1.0
                    if any(
                        kw in str(row.get("pattern_name", "")).lower()
                        for kw in [
                            "double",
                            "triple",
                            "head",
                            "reversal",
                            "hammer",
                            "doji",
                            "engulfing",
                            "harami",
                        ]
                    )
                    else 0.0,
                    "is_breakout": 1.0
                    if any(
                        kw in str(row.get("pattern_name", "")).lower()
                        for kw in ["breakout", "donchian", "gap", "flag", "pennant"]
                    )
                    else 0.0,
                }
            )

        if not rows:
            return pd.DataFrame(), pd.Series(dtype=float)

        X = pd.DataFrame(rows).set_index("timestamp")
        feature_cols = [c for c in X.columns]
        y = (
            labeled.set_index("timestamp")
            .loc[X.index, "label"]
            .map(lambda v: 1 if v == 1 else 0)
            .astype(float)
        )

        X = X[feature_cols]
        return X, y

    def _train_lgbm_cv(self, X: pd.DataFrame, y: pd.Series) -> tuple[float, list]:
        """Train LightGBM with PurgedKFold CV."""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("lightgbm not installed; skipping LGBM")
            return 0.5, []

        cv = PurgedKFold(
            n_splits=min(self.n_splits, len(X) // 3),
            pct_embargo=self.pct_embargo,
            label_span=self.time_limit,
        )

        fold_metrics = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X), 1):
            if len(test_idx) < 5:
                continue
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if len(set(y_train)) < 2 or len(set(y_test)) < 2:
                continue

            model = lgb.LGBMClassifier(**self.LGBM_PARAMS)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]

            from sklearn.metrics import roc_auc_score

            try:
                auc = float(roc_auc_score(y_test, proba))
            except ValueError:
                auc = 0.5

            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "auc": round(auc, 4),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                }
            )

        mean_auc = float(np.mean([m["auc"] for m in fold_metrics])) if fold_metrics else 0.5
        return mean_auc, fold_metrics

    def _train_catboost_cv(self, X: pd.DataFrame, y: pd.Series) -> tuple[Optional[float], list]:
        """Train CatBoost with PurgedKFold CV."""
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            logger.warning("catboost not installed; skipping CatBoost")
            return None, []

        cv = PurgedKFold(
            n_splits=min(self.n_splits, len(X) // 3),
            pct_embargo=self.pct_embargo,
            label_span=self.time_limit,
        )

        fold_metrics = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X), 1):
            if len(test_idx) < 5:
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

            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "auc": round(auc, 4),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                }
            )

        mean_auc = float(np.mean([m["auc"] for m in fold_metrics])) if fold_metrics else 0.5
        return mean_auc, fold_metrics

    def _train_lgbm_final(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Train final LightGBM model on all data."""
        import lightgbm as lgb

        model = lgb.LGBMClassifier(**self.LGBM_PARAMS)
        model.fit(X, y)
        return model

    def _train_catboost_final(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Train final CatBoost model on all data."""
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(**self.CATBOOST_PARAMS)
        model.fit(X, y)
        return model

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
            return float(thresholds[best_idx])
        except Exception:
            return 0.5

    def _extract_importance(self) -> Dict[str, float]:
        """Extract feature importance from trained model."""
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return {}
        importance = self.model.feature_importances_
        return dict(
            sorted(
                zip(self.feature_names_, importance),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    def save(self, path: str | Path) -> None:
        """Save trained model to disk."""
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "feature_names": self.feature_names_,
            "model_used": self.model_used,
            "probability_threshold": self.probability_threshold_,
            "config": {
                "tp_atr_mult": self.tp_atr_mult,
                "sl_atr_mult": self.sl_atr_mult,
                "time_limit": self.time_limit,
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
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.feature_names_ = model_data["feature_names"]
        self.model_used = model_data.get("model_used")
        self.probability_threshold_ = model_data.get("probability_threshold", 0.5)
