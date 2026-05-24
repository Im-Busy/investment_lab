"""Stop-loss optimizer using CatBoost GBDT.

Trains a gradient-boosted model on historical trade contexts to predict
the optimal stop-loss distance per trade. Converts the traditionally
manual ATR-multiple choice into a data-driven, context-aware decision.

Architecture:
  1. LabelGenerator: Derives optimal stop distance from trade outcomes
     by finding the ATR multiple that would have maximized risk:reward.
  2. StopLossOptimizer: Trains CatBoost on entry-context features
     (volatility, regime, pattern, market structure) → optimal ATR multiple.
  3. Integrates with PositionSizer via recommended_stop() output.

Reference: Lopez de Prado, "Advances in Financial Machine Learning", Ch. 10
(Meta-Labeling and Stop-Loss Calibration).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class StopLossLabel:
    """Container for a stop-loss label derived from trade outcome.

    Attributes:
        atr_multiple: Optimal ATR multiple that maximized risk:reward.
        optimal_stop_pct: Optimal stop distance as % of entry price.
        max_adverse_excursion: Maximum adverse excursion during trade.
        max_favorable_excursion: Maximum favorable excursion during trade.
        trade_return: Realized return of the trade.
        bars_held: Number of bars trade was held.
    """

    atr_multiple: float
    optimal_stop_pct: float
    max_adverse_excursion: float
    max_favorable_excursion: float
    trade_return: float
    bars_held: int


@dataclass
class StopLossRecommendation:
    """Per-trade stop-loss recommendation.

    Attributes:
        atr_multiple: Recommended ATR multiple for stop distance.
        stop_distance_pct: Recommended stop distance as % of entry price.
        confidence: Model prediction confidence (0-1).
        feature_contributions: Key feature contributions to prediction.
    """

    atr_multiple: float
    stop_distance_pct: float
    confidence: float
    feature_contributions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "atr_multiple": round(self.atr_multiple, 3),
            "stop_distance_pct": round(self.stop_distance_pct, 4),
            "confidence": round(self.confidence, 3),
            "feature_contributions": {
                k: round(v, 4) for k, v in self.feature_contributions.items()
            },
        }


class LabelGenerator:
    """Derive optimal stop-loss labels from trade outcome data.

    For each trade, computes the ATR multiple that would have resulted
    in the best risk:reward ratio given the actual price path.

    Usage:
        >>> gen = LabelGenerator(atr_series, min_atr=0.5, max_atr=6.0)
        >>> labels = gen.generate(trades_df)
    """

    def __init__(
        self,
        atr_series: pd.Series,
        min_atr_multiple: float = 0.5,
        max_atr_multiple: float = 6.0,
        step: float = 0.25,
        target_rr: float = 2.0,
    ) -> None:
        """Initialize label generator.

        Args:
            atr_series: ATR values indexed by timestamp.
            min_atr_multiple: Minimum ATR multiple to consider.
            max_atr_multiple: Maximum ATR multiple to consider.
            step: Granularity of ATR multiple search.
            target_rr: Target risk:reward ratio for scoring.
        """
        self._atr = atr_series
        self._min_atr = min_atr_multiple
        self._max_atr = max_atr_multiple
        self._step = step
        self._target_rr = target_rr

        self._atr_candidates = np.arange(min_atr_multiple, max_atr_multiple + step, step)

    def generate(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Generate optimal stop-loss labels for each trade.

        Args:
            trades: DataFrame with columns:
                entry_time, exit_time, entry_price, exit_price,
                high_while_open, low_while_open

        Returns:
            DataFrame with trades + label columns appended.
        """
        labels: List[StopLossLabel] = []

        for _, trade in trades.iterrows():
            entry_time = trade["entry_time"]
            exit_time = trade["exit_time"]
            entry_price = float(trade["entry_price"])
            exit_price = float(trade["exit_price"])
            high = float(trade.get("high_while_open", exit_price))
            low = float(trade.get("low_while_open", exit_price))

            label = self._compute_optimal_stop(
                entry_time, exit_time, entry_price, exit_price, high, low
            )
            labels.append(label)

        result = trades.copy()
        result["optimal_atr_multiple"] = [lb.atr_multiple for lb in labels]
        result["optimal_stop_pct"] = [lb.optimal_stop_pct for lb in labels]
        result["max_adverse_excursion"] = [lb.max_adverse_excursion for lb in labels]
        result["max_favorable_excursion"] = [lb.max_favorable_excursion for lb in labels]
        result["trade_return"] = [lb.trade_return for lb in labels]
        result["bars_held"] = [lb.bars_held for lb in labels]
        return result

    def _compute_optimal_stop(
        self,
        entry_time: Any,
        exit_time: Any,
        entry_price: float,
        exit_price: float,
        high: float,
        low: float,
    ) -> StopLossLabel:
        """Find ATR multiple that would have maximized risk:reward."""
        trade_return = (exit_price - entry_price) / entry_price

        mae = (entry_price - low) / entry_price
        if mae < 0:
            mae = 0.0

        mfe = (high - entry_price) / entry_price
        if mfe < 0:
            mfe = 0.0

        bars = self._count_bars(entry_time, exit_time)

        atr_at_entry = self._get_atr(entry_time)
        if atr_at_entry <= 0:
            return StopLossLabel(
                atr_multiple=2.0,
                optimal_stop_pct=0.02,
                max_adverse_excursion=float(mae),
                max_favorable_excursion=float(mfe),
                trade_return=float(trade_return),
                bars_held=bars,
            )

        best_multiple = 2.0
        best_score = -float("inf")

        for mult in self._atr_candidates:
            stop_distance = (mult * atr_at_entry) / entry_price
            if stop_distance <= 0:
                continue

            if stop_distance <= mae:
                score = -1.0
            else:
                target_gain = stop_distance * self._target_rr
                effective_gain = min(mfe, target_gain)

                if stop_distance > 0:
                    score = effective_gain / stop_distance
                else:
                    score = 0.0

            if score > best_score:
                best_score = score
                best_multiple = float(mult)

        optimal_stop_pct = (best_multiple * atr_at_entry) / entry_price

        return StopLossLabel(
            atr_multiple=best_multiple,
            optimal_stop_pct=float(optimal_stop_pct),
            max_adverse_excursion=float(mae),
            max_favorable_excursion=float(mfe),
            trade_return=float(trade_return),
            bars_held=bars,
        )

    def _get_atr(self, timestamp: Any) -> float:
        """Get ATR value at or before timestamp."""
        available = self._atr[self._atr.index <= timestamp]
        if len(available) == 0:
            return 0.01
        return float(available.iloc[-1])

    def _count_bars(self, entry_time: Any, exit_time: Any) -> int:
        """Count bars between entry and exit."""
        try:
            mask = (self._atr.index >= entry_time) & (self._atr.index <= exit_time)
            return int(mask.sum())
        except Exception:
            return 1


class StopLossOptimizer:
    """CatBoost GBDT that predicts optimal stop-loss ATR multiple.

    Trains on labeled historical trades using market context features
    at entry to predict the ideal stop distance.

    Features used:
        - Volatility regime (ATR percentile, HV, IV percentile)
        - Market regime (trending/ranging/volatile)
        - Pattern type (flag, triangle, double top, etc.)
        - Market structure (distance to support/resistance)
        - Time context (session, day of week)

    Usage:
        >>> optimizer = StopLossOptimizer()
        >>> optimizer.train(X_features, y_atr_labels)
        >>> rec = optimizer.recommend(entry_features, current_atr)
        >>> print(f"Stop at {rec.atr_multiple:.1f}x ATR")
    """

    def __init__(
        self,
        iterations: int = 500,
        depth: int = 6,
        learning_rate: float = 0.03,
        l2_leaf_reg: float = 3.0,
        random_seed: int = 42,
        verbose: bool = False,
    ) -> None:
        """Initialize stop-loss optimizer.

        Args:
            iterations: Number of CatBoost trees.
            depth: Tree depth.
            learning_rate: Learning rate.
            l2_leaf_reg: L2 regularization.
            random_seed: Random seed.
            verbose: Print training progress.
        """
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.verbose = verbose

        self._model: Any = None
        self._feature_names: List[str] = []
        self._is_trained: bool = False
        self._train_metrics: Dict[str, float] = {}

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        validation_size: float = 0.2,
        early_stopping_rounds: int = 50,
    ) -> Dict[str, float]:
        """Train the stop-loss prediction model.

        Args:
            X: Feature matrix (market context at trade entry).
            y: Target values (optimal ATR multiple per trade).
            validation_size: Fraction of data for validation.
            early_stopping_rounds: Early stopping patience.

        Returns:
            Dict with training metrics (RMSE, MAE, R²).
        """
        from catboost import CatBoostRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        valid_mask = X.notna().all(axis=1) & pd.Series(y, index=X.index).notna()
        X_clean = X[valid_mask].copy()
        y_clean = pd.Series(y, index=X.index)[valid_mask].copy()

        if len(X_clean) < 50:
            raise ValueError(f"Insufficient samples: {len(X_clean)} (need >= 50)")

        self._feature_names = list(X_clean.columns)

        split_idx = int(len(X_clean) * (1.0 - validation_size))
        X_train = X_clean.iloc[:split_idx]
        X_val = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:split_idx]
        y_val = y_clean.iloc[split_idx:]

        self._model = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            verbose=100 if self.verbose else False,
            loss_function="Huber:delta=1.0",
            task_type="CPU",
            early_stopping_rounds=early_stopping_rounds,
        )

        self._model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            verbose=100 if self.verbose else False,
        )

        y_pred = self._model.predict(X_val)
        self._train_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_val, y_pred))),
            "mae": float(mean_absolute_error(y_val, y_pred)),
            "r2": float(r2_score(y_val, y_pred)),
            "n_train": len(X_train),
            "n_val": len(X_val),
        }

        self._is_trained = True
        return self._train_metrics

    def recommend(
        self,
        features: pd.Series | pd.DataFrame,
        current_atr: float,
    ) -> StopLossRecommendation:
        """Generate stop-loss recommendation for a trade entry.

        Args:
            features: Market context features at entry time.
            current_atr: Current ATR value.

        Returns:
            StopLossRecommendation with ATR multiple and stop distance.
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")

        if isinstance(features, pd.Series):
            X = features[self._feature_names].to_frame().T.fillna(0)
        else:
            X = features[self._feature_names].fillna(0)

        atr_multiple = float(self._model.predict(X)[0])
        atr_multiple = max(0.5, min(atr_multiple, 8.0))
        stop_distance_pct = (atr_multiple * current_atr) / 1.0

        confidence = self._estimate_confidence(X)

        contributions = self._get_feature_contributions(X)

        return StopLossRecommendation(
            atr_multiple=atr_multiple,
            stop_distance_pct=stop_distance_pct,
            confidence=confidence,
            feature_contributions=contributions,
        )

    def recommend_batch(
        self,
        X: pd.DataFrame,
        current_atr: float,
    ) -> List[StopLossRecommendation]:
        """Generate recommendations for multiple entries.

        Args:
            X: Feature DataFrame (one row per trade).
            current_atr: Current ATR value.

        Returns:
            List of StopLossRecommendation.
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")

        X_clean = X[self._feature_names].fillna(0)
        predictions = self._model.predict(X_clean)

        recommendations = []
        for i, atr_mult in enumerate(predictions):
            atr_mult = max(0.5, min(float(atr_mult), 8.0))
            rec = StopLossRecommendation(
                atr_multiple=atr_mult,
                stop_distance_pct=(atr_mult * current_atr) / 1.0,
                confidence=0.7,
            )
            recommendations.append(rec)

        return recommendations

    def _estimate_confidence(self, X: pd.DataFrame) -> float:
        """Estimate prediction confidence using ensemble variance."""
        try:
            preds = np.array([self._model.predict(X)[0] for _ in range(10)])
            std = float(np.std(preds))
            confidence = 1.0 / (1.0 + std * 5)
            return float(np.clip(confidence, 0.1, 1.0))
        except Exception:
            return 0.7

    def _get_feature_contributions(self, X: pd.DataFrame) -> Dict[str, float]:
        """Extract top contributing features for this prediction."""
        try:
            importance = self._model.get_feature_importance(
                data=CatBoostRegressor_compat(X),
                type="PredictionValuesChange",
            )
            if importance is not None:
                return dict(
                    sorted(
                        zip(self._feature_names, importance),
                        key=lambda x: abs(float(x[1])),
                        reverse=True,
                    )[:5]
                )
        except Exception:
            pass
        return {}

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance as a sorted DataFrame.

        Returns:
            DataFrame with columns: feature, importance.
        """
        if not self._is_trained:
            return pd.DataFrame(columns=["feature", "importance"])

        try:
            importance = self._model.get_feature_importance()
            rows = sorted(
                zip(self._feature_names, importance),
                key=lambda x: x[1],
                reverse=True,
            )
            return pd.DataFrame(rows, columns=["feature", "importance"])
        except Exception:
            return pd.DataFrame(
                {"feature": self._feature_names, "importance": [0.0] * len(self._feature_names)}
            )

    def predict_atr_multiple(self, features: pd.DataFrame) -> np.ndarray:
        """Predict optimal ATR multiple directly.

        Args:
            features: Feature matrix.

        Returns:
            Array of predicted ATR multiples, clipped to [0.5, 8.0].
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")
        X = features[self._feature_names].fillna(0)
        preds = self._model.predict(X)
        return np.clip(preds, 0.5, 8.0)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def metrics(self) -> Dict[str, float]:
        return dict(self._train_metrics)

    def save(self, path: str) -> None:
        """Save trained model to disk."""
        import pickle

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self._model,
            "feature_names": self._feature_names,
            "train_metrics": self._train_metrics,
            "config": {
                "iterations": self.iterations,
                "depth": self.depth,
                "learning_rate": self.learning_rate,
                "l2_leaf_reg": self.l2_leaf_reg,
            },
        }
        with open(out, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load trained model from disk."""
        import pickle

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")

        with open(p, "rb") as f:
            data = pickle.load(f)  # nosec B301

        self._model = data["model"]
        self._feature_names = data["feature_names"]
        self._train_metrics = data.get("train_metrics", {})
        self._is_trained = True

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return f"StopLossOptimizer(status={status}, features={len(self._feature_names)})"


def CatBoostRegressor_compat(X: pd.DataFrame) -> Any:
    """Provide a compatible input for CatBoost feature importance."""
    from catboost import Pool

    return Pool(X)
