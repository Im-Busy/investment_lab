"""
Breakout Probability Classifier.

ML-based scoring for Donchian Channel breakouts: predicts whether a breakout
will result in a profitable trade (true breakout) or fail (false breakout).

Uses CatBoost binary classification with features derived from the breakout
context: channel characteristics, volume profile, ATR dynamics, price action,
and market regime indicators.

The target label is generated using TripleBarrierLabeler on breakout signals:
  1 = profitable breakout (TP hit before SL or positive return at timeout)
  0 = failed breakout (SL hit or negative return at timeout)

Usage:
    from src.ml.breakout_classifier import BreakoutClassifier

    clf = BreakoutClassifier()
    result = clf.fit(df, breakout_signals)
    probs = clf.predict_proba(df_test)
    scored_breakouts = clf.score_breakouts(df_test, breakout_signals)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .triple_barrier import TripleBarrierLabeler


@dataclass
class BreakoutResult:
    """Results from breakout classifier training.

    Attributes:
        auc_roc: Area under ROC curve.
        accuracy: Classification accuracy.
        precision: Precision of profitable breakout prediction.
        recall: Recall of profitable breakout detection.
        f1: F1 score.
        feature_importance: Dict of feature name -> importance.
        n_train: Training samples.
        n_breakouts: Total breakout signals.
        n_profitable: Profitable breakouts (label=1).
        win_rate: Baseline win rate without ML.
    """

    auc_roc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    feature_importance: Dict[str, float]
    n_train: int
    n_breakouts: int
    n_profitable: int
    win_rate: float
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "auc_roc": round(self.auc_roc, 4),
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "n_train": self.n_train,
            "n_breakouts": self.n_breakouts,
            "n_profitable": self.n_profitable,
            "baseline_win_rate": round(self.win_rate, 4),
        }

    def __repr__(self) -> str:
        return (
            f"BreakoutResult(auc={self.auc_roc:.4f}, acc={self.accuracy:.4f}, "
            f"f1={self.f1:.4f}, win_rate={self.win_rate:.3f})"
        )


class BreakoutClassifier:
    """Predict breakout profitability using CatBoost.

    Features are extracted from the breakout context: Donchian channel stats,
    volume surge, ATR expansion, price position relative to bands, and market
    regime indicators.

    Labels are generated automatically using triple-barrier labeling on
    breakout signals. A breakout is "profitable" (label=1) if TP hits before
    SL within the time limit, or if the return at timeout is positive.

    Example:
        >>> clf = BreakoutClassifier()
        >>> result = clf.fit(df, breakout_signals)
        >>> probs = clf.predict_proba(df_test)
    """

    def __init__(
        self,
        channel_period: int = 20,
        time_limit: int = 20,
        atr_mult_tp: float = 2.0,
        atr_mult_sl: float = 1.0,
        n_estimators: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        bagging_temperature: float = 1.0,
        random_seed: int = 42,
        verbose: bool = False,
    ):
        """Initialize breakout classifier.

        Args:
            channel_period: Donchian channel lookback period (default 20).
            time_limit: Max bars for triple-barrier labeling.
            atr_mult_tp: ATR multiplier for take-profit.
            atr_mult_sl: ATR multiplier for stop-loss.
            n_estimators: Number of CatBoost trees.
            depth: Tree depth.
            learning_rate: Learning rate.
            l2_leaf_reg: L2 regularization.
            random_strength: Random score strength (CatBoost).
            bagging_temperature: Bayesian bootstrap temperature.
            random_seed: Random seed.
            verbose: Print training progress.
        """
        self.channel_period = channel_period
        self.time_limit = time_limit
        self.atr_mult_tp = atr_mult_tp
        self.atr_mult_sl = atr_mult_sl
        self.n_estimators = n_estimators
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.random_seed = random_seed
        self.verbose = verbose

        self._model = None
        self._feature_names: Optional[List[str]] = None
        self._labeler = TripleBarrierLabeler(atr_mult_tp=atr_mult_tp, atr_mult_sl=atr_mult_sl)
        self._threshold: float = 0.5

    @staticmethod
    def extract_breakout_signals(
        df: pd.DataFrame,
        channel_period: int = 20,
    ) -> pd.DataFrame:
        """Extract breakout signals and features from price data.

        Generates signals where price breaks the Donchian channel,
        along with contextual features for ML scoring.

        Args:
            df: DataFrame with OHLCV columns.
            channel_period: Donchian channel lookback.

        Returns:
            DataFrame indexed by signal bars with features and signal metadata.
        """
        n = len(df)
        if n < channel_period + 5:
            return pd.DataFrame()

        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values
        volume = df.get("Volume", pd.Series(1, index=df.index)).values
        o = df["Open"].values

        signals = []

        for i in range(channel_period, n):
            upper = np.max(high[i - channel_period : i])
            lower = np.min(low[i - channel_period : i])
            middle = (upper + lower) / 2
            channel_width = upper - lower
            channel_width_pct = channel_width / middle if middle > 0 else 0

            prev_upper = np.max(high[i - channel_period : i])
            prev_lower = np.min(low[i - channel_period : i])

            breakout_up = close[i] > prev_upper
            breakout_down = close[i] < prev_lower

            if not (breakout_up or breakout_down):
                continue

            direction = 1 if breakout_up else -1

            # Feature extraction
            features: Dict[str, float] = {}

            # Channel characteristics
            features["channel_width_pct"] = channel_width_pct
            features["price_position"] = (close[i] - lower) / max(channel_width, 1e-8)
            features["breakout_magnitude"] = (
                (close[i] - upper) / max(channel_width, 1e-8)
                if breakout_up
                else (lower - close[i]) / max(channel_width, 1e-8)
            )

            # Candle characteristics
            candle_range = float(high[i] - low[i])
            candle_body = abs(close[i] - o[i])
            features["candle_body_pct"] = candle_body / max(candle_range, 1e-8)
            features["candle_range_pct"] = candle_range / close[i]

            # Volume surge
            if i >= 30:
                vol_sma = np.mean(volume[i - 20 : i])
                features["volume_ratio"] = volume[i] / max(vol_sma, 1e-8)
            else:
                features["volume_ratio"] = 1.0

            # ATR and volatility
            if i >= 15:
                tr = np.maximum(
                    high[i - 14 : i + 1] - low[i - 14 : i + 1],
                    np.abs(high[i - 14 : i + 1] - np.roll(close[i - 14 : i + 1], 1)),
                )
                tr[0] = high[i - 14] - low[i - 14]
                atr_val = np.mean(tr)
                features["atr"] = atr_val
                features["atr_pct"] = atr_val / close[i]

                if i >= 16:
                    prev_tr = np.maximum(
                        high[i - 15 : i] - low[i - 15 : i],
                        np.abs(high[i - 15 : i] - np.roll(close[i - 15 : i], 1)),
                    )
                    prev_tr[0] = high[i - 15] - low[i - 15]
                    prev_atr = np.mean(prev_tr)
                    features["atr_change"] = (atr_val - prev_atr) / max(prev_atr, 1e-8)

            # Prior consolidation
            if i >= channel_period + 10:
                prev_upper_10 = np.max(high[i - 10 - channel_period : i - 10])
                prev_lower_10 = np.min(low[i - 10 - channel_period : i - 10])
                prev_width = prev_upper_10 - prev_lower_10
                features["channel_squeeze"] = channel_width / max(prev_width, 1e-8)

            # Return momentum before breakout
            if i >= 10:
                returns_5 = (close[i] - close[i - 5]) / max(close[i - 5], 1e-8)
                features["ret_5"] = returns_5
            if i >= 20:
                returns_20 = (close[i] - close[i - 20]) / max(close[i - 20], 1e-8)
                features["ret_20"] = returns_20
                vol_20 = np.std(np.diff(np.log(close[i - 20 : i + 1]))) * np.sqrt(252)
                features["vol_20"] = vol_20

            # RSI-like momentum
            if i >= 14:
                gains = np.maximum(np.diff(close[i - 14 : i + 1]), 0)
                losses = np.abs(np.minimum(np.diff(close[i - 14 : i + 1]), 0))
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                rs = avg_gain / max(avg_loss, 1e-8)
                features["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

            # Distance to recent high/low
            if i >= 50:
                hh_50 = np.max(high[i - 50 : i])
                ll_50 = np.min(low[i - 50 : i])
                features["dist_hh_50"] = (close[i] - hh_50) / max(hh_50, 1e-8)
                features["dist_ll_50"] = (close[i] - ll_50) / max(ll_50, 1e-8)

            row = {
                "index": i,
                "timestamp": df.index[i] if hasattr(df, "index") else i,
                "direction": direction,
                "entry_price": close[i],
                "upper": upper,
                "lower": lower,
                "middle": middle,
            }
            row.update(features)
            signals.append(row)

        return pd.DataFrame(signals).set_index("index") if signals else pd.DataFrame()

    def _generate_labels(
        self,
        df: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> pd.Series:
        """Generate profitability labels for breakout signals.

        Uses TripleBarrierLabeler to determine if each breakout signal
        resulted in a profitable trade.

        Args:
            df: DataFrame with OHLCV data.
            signals: DataFrame of breakout signals (from extract_breakout_signals).

        Returns:
            Series of binary labels: 1=profitable, 0=failed, indexed by signal index.
        """
        n = len(df)
        close = df["Close"].values.astype(np.float64)
        high = df["High"].values.astype(np.float64)
        low = df["Low"].values.astype(np.float64)

        labels = {}

        for idx, row in signals.iterrows():
            i = int(idx)
            entry = row["entry_price"]
            direction = int(row["direction"])

            # Compute TP/SL based on ATR or fixed percentages
            atr_val = row.get("atr", np.nan)
            if not np.isnan(atr_val):
                tp_dist = atr_val * self.atr_mult_tp
                sl_dist = atr_val * self.atr_mult_sl
            else:
                tp_dist = entry * 0.03
                sl_dist = entry * 0.015

            if direction == 1:
                tp = entry + tp_dist
                sl = entry - sl_dist
            else:
                tp = entry - tp_dist
                sl = entry + sl_dist

            # Forward window
            end = min(i + self.time_limit + 1, n)
            is_profitable = False

            for j in range(i + 1, end):
                if direction == 1:
                    if high[j] >= tp:
                        is_profitable = True
                        break
                    elif low[j] <= sl:
                        is_profitable = False
                        break
                else:
                    if low[j] <= tp:
                        is_profitable = True
                        break
                    elif high[j] >= sl:
                        is_profitable = False
                        break
            else:
                # Timeout — check final return
                final_close = close[end - 1]
                ret = (final_close - entry) / entry
                if direction == 1:
                    is_profitable = ret > 0
                else:
                    is_profitable = ret < 0

            labels[i] = int(is_profitable)

        return pd.Series(labels)

    def fit(
        self,
        df: pd.DataFrame,
        signals: Optional[pd.DataFrame] = None,
        cat_features: Optional[List[str]] = None,
    ) -> BreakoutClassifier:
        """Train breakout classifier.

        If signals are not provided, they are extracted automatically using
        extract_breakout_signals(). Labels are generated via triple-barrier.

        Args:
            df: DataFrame with OHLCV columns.
            signals: Optional pre-extracted breakout signals.
            cat_features: Categorical feature names.

        Returns:
            Self for chaining.
        """
        if signals is None:
            signals = self.extract_breakout_signals(df, self.channel_period)

        if signals.empty:
            raise ValueError("No breakout signals found in data.")

        y = self._generate_labels(df, signals)

        # Intersect valid rows
        valid = signals.index.intersection(y.index)
        X = signals.loc[valid]
        y = y.loc[valid]

        if len(X) < 20:
            raise ValueError(f"Insufficient labeled breakouts: {len(X)}. Need at least 20.")

        # Select feature columns (exclude metadata)
        exclude_cols = {"timestamp", "direction", "entry_price", "upper", "lower", "middle"}
        feature_cols = [c for c in X.columns if c not in exclude_cols]
        self._feature_names = feature_cols

        X_arr = X[feature_cols].values.astype(np.float64)
        y_arr = y.values.astype(np.int32)

        has_negative = y_arr.min() < 0
        if has_negative:
            y_binary = (y_arr > 0).astype(np.int32)
        else:
            y_binary = y_arr

        from catboost import CatBoostClassifier

        self._model = CatBoostClassifier(
            iterations=self.n_estimators,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_strength=self.random_strength,
            bagging_temperature=self.bagging_temperature,
            random_seed=self.random_seed,
            verbose=self.verbose,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            task_type="CPU",
        )
        self._model.fit(X_arr, y_binary, cat_features=cat_features or [])

        probs = self._model.predict_proba(X_arr)[:, 1]
        best_f1 = 0.0
        best_thresh = 0.5
        for thresh in np.arange(0.3, 0.7, 0.05):
            pred = (probs >= thresh).astype(int)
            tp = np.sum((pred == 1) & (y_binary == 1))
            fp = np.sum((pred == 1) & (y_binary == 0))
            fn = np.sum((pred == 0) & (y_binary == 1))
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        self._threshold = best_thresh

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict breakout profitability (0 or 1)."""
        self._check_fitted()
        probs = self._model.predict_proba(X[self._feature_names].values.astype(np.float64))[:, 1]
        return (probs >= self._threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict breakout profitability probabilities.

        Args:
            X: Features DataFrame.

        Returns:
            Array of probabilities that breakout is profitable.
        """
        self._check_fitted()
        return self._model.predict_proba(X[self._feature_names].values.astype(np.float64))[:, 1]

    def score_breakouts(
        self,
        df: pd.DataFrame,
        signals: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Score breakout signals with ML probability.

        Returns the signals DataFrame with added columns:
          - ml_prob: ML-predicted probability of profitability
          - ml_prediction: Binary prediction (1=predicted profitable)
          - ml_confidence: Confidence score

        Args:
            df: DataFrame with OHLCV data.
            signals: Optional pre-extracted signals.

        Returns:
            Signals DataFrame with ML scores appended.
        """
        self._check_fitted()

        if signals is None:
            signals = self.extract_breakout_signals(df, self.channel_period)

        if signals.empty:
            return signals

        X = signals[self._feature_names].values.astype(np.float64)
        probs = self._model.predict_proba(X)[:, 1]

        result = signals.copy()
        result["ml_prob"] = probs
        result["ml_prediction"] = (probs >= self._threshold).astype(int)
        result["ml_confidence"] = np.abs(probs - 0.5) * 2.0
        return result

    def evaluate(
        self,
        df: pd.DataFrame,
        signals: Optional[pd.DataFrame] = None,
    ) -> BreakoutResult:
        """Evaluate classifier performance.

        Args:
            df: DataFrame with OHLCV data.
            signals: Optional pre-extracted signals.

        Returns:
            BreakoutResult with metrics.
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        self._check_fitted()

        if signals is None:
            signals = self.extract_breakout_signals(df, self.channel_period)

        y_true = self._generate_labels(df, signals)
        valid = signals.index.intersection(y_true.index)
        X = signals.loc[valid]
        y = y_true.loc[valid].values

        has_negative = y.min() < 0
        if has_negative:
            y_binary = (y > 0).astype(int)
        else:
            y_binary = y

        X_arr = X[self._feature_names].values.astype(np.float64)
        probs = self._model.predict_proba(X_arr)[:, 1]
        preds = (probs >= self._threshold).astype(int)

        auc = float(roc_auc_score(y_binary, probs))
        acc = float(accuracy_score(y_binary, preds))
        prec = float(precision_score(y_binary, preds, zero_division=0))
        rec = float(recall_score(y_binary, preds, zero_division=0))
        f1 = float(f1_score(y_binary, preds, zero_division=0))

        win_rate = float(np.mean(y_binary))
        feat_imp = self._get_feature_importance()

        return BreakoutResult(
            auc_roc=auc,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            feature_importance=feat_imp,
            n_train=len(X),
            n_breakouts=len(signals),
            n_profitable=int(np.sum(y_binary)),
            win_rate=win_rate,
        )

    def cross_validate(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        pct_embargo: float = 0.05,
        signals: Optional[pd.DataFrame] = None,
    ) -> BreakoutResult:
        """Purged time-series cross-validation.

        Args:
            df: DataFrame with OHLCV data.
            n_splits: Number of folds.
            pct_embargo: Embargo fraction.
            signals: Optional pre-extracted signals.

        Returns:
            BreakoutResult with mean fold metrics.
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        if signals is None:
            signals = self.extract_breakout_signals(df, self.channel_period)

        if signals.empty:
            raise ValueError("No breakout signals found.")

        y_all = self._generate_labels(df, signals)
        valid = signals.index.intersection(y_all.index)
        X = signals.loc[valid]
        y = y_all.loc[valid]

        exclude_cols = {"timestamp", "direction", "entry_price", "upper", "lower", "middle"}
        feature_cols = [c for c in X.columns if c not in exclude_cols]
        self._feature_names = feature_cols

        X_arr = X[feature_cols].values.astype(np.float64)
        y_arr = y.values.astype(np.int32)

        has_negative = y_arr.min() < 0
        if has_negative:
            y_binary = (y_arr > 0).astype(np.int32)
        else:
            y_binary = y_arr

        from .purged_cv import PurgedKFold
        from catboost import CatBoostClassifier

        cv = PurgedKFold(
            n_splits=n_splits,
            pct_embargo=pct_embargo,
            label_span=self.time_limit,
        )

        fold_metrics = []
        all_probs = np.zeros(len(X_arr), dtype=np.float64)

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_arr), 1):
            model = CatBoostClassifier(
                iterations=self.n_estimators,
                depth=self.depth,
                learning_rate=self.learning_rate,
                l2_leaf_reg=self.l2_leaf_reg,
                random_seed=self.random_seed,
                verbose=False,
                loss_function="Logloss",
                auto_class_weights="Balanced",
                task_type="CPU",
            )
            model.fit(X_arr[train_idx], y_binary[train_idx])
            probs = model.predict_proba(X_arr[test_idx])[:, 1]
            all_probs[test_idx] = probs

            thresh = 0.5
            best_f1 = 0.0
            for t in np.arange(0.3, 0.7, 0.05):
                p = (probs >= t).astype(int)
                f1_v = f1_score(y_binary[test_idx], p, zero_division=0)
                if f1_v > best_f1:
                    best_f1 = f1_v
                    thresh = t

            preds = (probs >= thresh).astype(int)
            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "auc_roc": round(roc_auc_score(y_binary[test_idx], probs), 4),
                    "accuracy": round(accuracy_score(y_binary[test_idx], preds), 4),
                    "precision": round(
                        precision_score(y_binary[test_idx], preds, zero_division=0), 4
                    ),
                    "recall": round(recall_score(y_binary[test_idx], preds, zero_division=0), 4),
                    "f1": round(f1_score(y_binary[test_idx], preds, zero_division=0), 4),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                }
            )

        # Train final model on all data
        from catboost import CatBoostClassifier

        self._model = CatBoostClassifier(
            iterations=self.n_estimators,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_strength=self.random_strength,
            bagging_temperature=self.bagging_temperature,
            random_seed=self.random_seed,
            verbose=False,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            task_type="CPU",
        )
        self._model.fit(X_arr, y_binary)

        best_thresh = 0.5
        best_f1 = 0.0
        for t in np.arange(0.3, 0.7, 0.05):
            p = (all_probs >= t).astype(int)
            f1_v = f1_score(y_binary, p, zero_division=0)
            if f1_v > best_f1:
                best_f1 = f1_v
                best_thresh = t
        self._threshold = best_thresh

        mean_auc = float(np.mean([m["auc_roc"] for m in fold_metrics]))
        mean_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))
        mean_prec = float(np.mean([m["precision"] for m in fold_metrics]))
        mean_rec = float(np.mean([m["recall"] for m in fold_metrics]))
        mean_f1 = float(np.mean([m["f1"] for m in fold_metrics]))

        win_rate = float(np.mean(y_binary))
        feat_imp = self._get_feature_importance()

        return BreakoutResult(
            auc_roc=mean_auc,
            accuracy=mean_acc,
            precision=mean_prec,
            recall=mean_rec,
            f1=mean_f1,
            feature_importance=feat_imp,
            n_train=len(X_arr),
            n_breakouts=len(signals),
            n_profitable=int(np.sum(y_binary)),
            win_rate=win_rate,
            fold_metrics=fold_metrics,
        )

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if self._model is None or self._feature_names is None:
            return {}
        importance = self._model.get_feature_importance()
        return dict(
            sorted(
                zip(self._feature_names, importance),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "feature_names": self._feature_names,
                "threshold": self._threshold,
                "config": {
                    "channel_period": self.channel_period,
                    "time_limit": self.time_limit,
                    "atr_mult_tp": self.atr_mult_tp,
                    "atr_mult_sl": self.atr_mult_sl,
                    "n_estimators": self.n_estimators,
                    "depth": self.depth,
                    "learning_rate": self.learning_rate,
                    "l2_leaf_reg": self.l2_leaf_reg,
                    "random_strength": self.random_strength,
                    "bagging_temperature": self.bagging_temperature,
                    "random_seed": self.random_seed,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "BreakoutClassifier":
        """Load model from disk."""
        import joblib

        data = joblib.load(path)
        config = data["config"]
        instance = cls(**config)
        instance._model = data["model"]
        instance._feature_names = data.get("feature_names")
        instance._threshold = data.get("threshold", 0.5)
        return instance

    def _check_fitted(self) -> None:
        if self._model is None:
            raise ValueError("Model not trained. Call fit() or cross_validate() first.")

    def __repr__(self) -> str:
        status = "fitted" if self._model is not None else "untrained"
        return f"BreakoutClassifier(status={status}, threshold={self._threshold:.2f})"
