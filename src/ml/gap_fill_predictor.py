"""
Gap-Fill Predictor (FS19): Predict whether an overnight/weekend gap will fill.

Uses LightGBM binary classification to predict whether the price will
retrace to fill a gap within N bars. Gap detection identifies overnight
and weekend gaps (close[t] vs open[t+1]), then builds context features
around the gap for the ML model.

Architecture:
    OHLCV data -> gap detection -> context features -> LGBMClassifier
    -> GapFillPredictor.predict(gap_context) -> {fill_prob, fill_bars_est}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .purged_cv import PurgedKFold

logger = logging.getLogger(__name__)


@dataclass
class GapFillTrainingResult:
    """Results from training the gap-fill predictor.

    Attributes:
        auc_roc: Area under ROC curve (PurgedKFold mean).
        accuracy: Classification accuracy.
        n_gaps: Total gaps detected.
        n_filled: Number of gaps that filled within N bars.
        baseline_fill_rate: Fill rate without ML (naive baseline).
        f1_score: F1 score.
        feature_importance: Top features by importance.
        fold_metrics: Per-fold PurgedKFold metrics.
    """

    auc_roc: float
    accuracy: float
    n_gaps: int
    n_filled: int
    baseline_fill_rate: float
    f1_score: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "auc_roc": round(self.auc_roc, 4),
            "accuracy": round(self.accuracy, 4),
            "n_gaps": self.n_gaps,
            "n_filled": self.n_filled,
            "baseline_fill_rate": round(self.baseline_fill_rate, 4),
            "f1_score": round(self.f1_score, 4),
            "top_features": dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        }


@dataclass
class GapFillPrediction:
    """Prediction for a single gap.

    Attributes:
        fill_probability: Probability gap will fill within N bars.
        estimated_bars: Estimated bars to fill (mode of similar gaps).
        gap_pct: Gap size as percentage.
        bar_index: Index in the DataFrame.
    """

    fill_probability: float
    estimated_bars: int
    gap_pct: float
    bar_index: int


@dataclass
class GapDetection:
    """Detected gap in price data.

    Attributes:
        bar_index: Index of the gap bar (open of the gap bar).
        gap_pct: Gap size as percentage (positive = gap up, negative = gap down).
        gap_type: 'up' or 'down'.
        pre_close: Previous bar close price.
        gap_open: Current bar open price.
        fill_level: Price level that must be touched to consider gap filled.
        filled: Whether gap was filled within look_forward bars.
        fill_bar_idx: Bar index where gap filled (or -1 if not filled).
        day_of_week: Day of week (0=Monday).
    """

    bar_index: int
    gap_pct: float
    gap_type: str
    pre_close: float
    gap_open: float
    fill_level: float
    filled: bool
    fill_bar_idx: int
    day_of_week: int


class GapFillPredictor:
    """Detect gaps and predict whether they will fill.

    Uses LightGBM binary classifier with PurgedKFold cross-validation.
    Features include gap size, pre-gap trend, volatility, volume profile,
    and time-based features.

    Attributes:
        look_forward: Bars to look forward for fill.
        gap_threshold_pct: Minimum gap size to consider (percentage).
        n_splits: PurgedKFold splits.
        model: Trained LGBMClassifier.
        feature_names_: List of feature column names.
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

    def __init__(
        self,
        look_forward: int = 5,
        gap_threshold_pct: float = 0.1,
        n_splits: int = 5,
        pct_embargo: float = 0.05,
    ) -> None:
        self.look_forward = look_forward
        self.gap_threshold_pct = gap_threshold_pct
        self.n_splits = n_splits
        self.pct_embargo = pct_embargo

        self.model: Any = None
        self.feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame) -> GapFillTrainingResult:
        """Detect gaps and train the fill predictor.

        Args:
            df: OHLCV DataFrame with columns [Open, High, Low, Close, Volume]
                and DatetimeIndex.

        Returns:
            GapFillTrainingResult with training metrics.
        """
        gaps = self.detect_gaps(df)
        X, y = self._build_features(df, gaps)

        if len(X) < 50:
            logger.warning("Fewer than 50 gaps; predictor may be unreliable")

        if len(X) < 20:
            return GapFillTrainingResult(
                auc_roc=0.5,
                accuracy=0.5,
                n_gaps=len(gaps),
                n_filled=sum(1 for g in gaps if g.filled),
                baseline_fill_rate=0.0,
                f1_score=0.0,
            )

        self.feature_names_ = list(X.columns)
        metrics = self._train_lgbm_cv(X, y)
        self.model = self._train_lgbm_final(X, y)

        return GapFillTrainingResult(
            auc_roc=metrics["auc"],
            accuracy=metrics["accuracy"],
            n_gaps=len(gaps),
            n_filled=sum(1 for g in gaps if g.filled),
            baseline_fill_rate=float(metrics["baseline"]),
            f1_score=metrics["f1"],
            feature_importance=self._extract_importance(),
            fold_metrics=metrics.get("fold_metrics", []),
        )

    def predict(self, df: pd.DataFrame, bar_index: int) -> GapFillPrediction:
        """Predict whether the gap at bar_index will fill.

        Args:
            df: OHLCV DataFrame.
            bar_index: Index of the gap bar.

        Returns:
            GapFillPrediction with fill probability and estimated bars.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        gap_info = self._detect_single_gap(df, bar_index)
        if gap_info is None:
            return GapFillPrediction(
                fill_probability=0.0,
                estimated_bars=0,
                gap_pct=0.0,
                bar_index=bar_index,
            )

        X, _ = self._build_features(df, [gap_info])
        if len(X) == 0:
            return GapFillPrediction(
                fill_probability=0.0,
                estimated_bars=0,
                gap_pct=gap_info.gap_pct,
                bar_index=bar_index,
            )

        proba = float(self.model.predict_proba(X)[:, 1][0])
        return GapFillPrediction(
            fill_probability=proba,
            estimated_bars=self._estimate_fill_bars(proba, gap_info.gap_pct),
            gap_pct=gap_info.gap_pct,
            bar_index=bar_index,
        )

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict fill probability for all gaps in DataFrame.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with gap details and fill predictions.
        """
        if self.model is None:
            return pd.DataFrame()

        gaps = self.detect_gaps(df)
        if not gaps:
            return pd.DataFrame()

        X, _, gaps_passed = self._build_features(df, gaps, return_gaps=True)
        if len(X) == 0:
            return pd.DataFrame()

        proba = self.model.predict_proba(X)[:, 1]
        results = []
        for j, gap in enumerate(gaps_passed):
            results.append(
                {
                    "bar_index": gap.bar_index,
                    "gap_pct": round(gap.gap_pct, 4),
                    "gap_type": gap.gap_type,
                    "day_of_week": gap.day_of_week,
                    "fill_probability": round(float(proba[j]), 4),
                    "estimated_bars": self._estimate_fill_bars(float(proba[j]), gap.gap_pct),
                }
            )
        return pd.DataFrame(results)

    def detect_gaps(self, df: pd.DataFrame) -> List[GapDetection]:
        """Detect all gaps in the data.

        Gap is defined as the difference between previous close and current
        open exceeding the threshold.

        Args:
            df: OHLCV DataFrame with columns [Open, High, Low, Close, Volume].

        Returns:
            List of GapDetection objects.
        """
        close = df["Close"].values
        open_ = df["Open"].values
        high = df["High"].values
        low = df["Low"].values
        n = len(df)

        if not hasattr(df.index, "dayofweek"):
            day_of_week_arr = np.zeros(n, dtype=int)
        else:
            day_of_week_arr = df.index.dayofweek.values

        gaps = []
        for i in range(1, n):
            prev_close = close[i - 1]
            curr_open = open_[i]

            is_gap_up = curr_open > prev_close

            gap_pct = (curr_open - prev_close) / prev_close * 100

            if abs(gap_pct) < self.gap_threshold_pct:
                continue

            fill_level = prev_close
            gap_type = "up" if is_gap_up else "down"

            # Check if gap fills within look_forward bars
            filled = False
            fill_bar_idx = -1
            end = min(i + self.look_forward, n)
            for j in range(i, end):
                if gap_type == "up" and low[j] <= fill_level:
                    filled = True
                    fill_bar_idx = j
                    break
                elif gap_type == "down" and high[j] >= fill_level:
                    filled = True
                    fill_bar_idx = j
                    break

            gaps.append(
                GapDetection(
                    bar_index=i,
                    gap_pct=gap_pct,
                    gap_type=gap_type,
                    pre_close=prev_close,
                    gap_open=curr_open,
                    fill_level=fill_level,
                    filled=filled,
                    fill_bar_idx=fill_bar_idx,
                    day_of_week=int(day_of_week_arr[i]),
                )
            )

        return gaps

    def _detect_single_gap(self, df: pd.DataFrame, bar_index: int) -> Optional[GapDetection]:
        """Detect a gap at a single bar index.

        Args:
            df: OHLCV DataFrame.
            bar_index: Index to check for gap.

        Returns:
            GapDetection or None if no gap at this bar.
        """
        n = len(df)
        if bar_index < 1 or bar_index >= n:
            return None

        prev_close = float(df["Close"].iloc[bar_index - 1])
        curr_open = float(df["Open"].iloc[bar_index])
        gap_pct = (curr_open - prev_close) / prev_close * 100

        if abs(gap_pct) < self.gap_threshold_pct:
            return None

        is_gap_up = curr_open > prev_close
        fill_level = prev_close
        gap_type = "up" if is_gap_up else "down"

        low_arr = df["Low"].values
        high_arr = df["High"].values

        filled = False
        fill_bar_idx = -1
        end = min(bar_index + self.look_forward, n)
        for j in range(bar_index, end):
            if gap_type == "up" and low_arr[j] <= fill_level:
                filled = True
                fill_bar_idx = j
                break
            elif gap_type == "down" and high_arr[j] >= fill_level:
                filled = True
                fill_bar_idx = j
                break

        day_of_week = int(df.index[bar_index].dayofweek) if hasattr(df.index, "dayofweek") else 0

        return GapDetection(
            bar_index=bar_index,
            gap_pct=gap_pct,
            gap_type=gap_type,
            pre_close=prev_close,
            gap_open=curr_open,
            fill_level=fill_level,
            filled=filled,
            fill_bar_idx=fill_bar_idx,
            day_of_week=day_of_week,
        )

    @staticmethod
    def _compute_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _build_features(
        self,
        df: pd.DataFrame,
        gaps: List[GapDetection],
        return_gaps: bool = False,
    ) -> Tuple[pd.DataFrame, pd.Series] | Tuple[pd.DataFrame, pd.Series, List[GapDetection]]:
        """Build feature matrix and labels from gap detections.

        Features:
            - gap_pct: Gap size as percentage (abs value)
            - gap_direction: 1=up, -1=down
            - pre_trend_5: 5-bar return before gap
            - pre_trend_20: 20-bar return before gap
            - volatility_20: 20-bar std dev of returns
            - volume_ratio: Volume / 20-bar avg volume at gap bar
            - atr_ratio: ATR(14) / Close
            - relative_position: (Close - 50-bar Low) / (50-bar High - 50-bar Low)
            - day_of_week: 0=Monday through 4=Friday
            - is_monday: 1 if Monday (weekend gap)
            - pre_gap_range: Pre-gap bar range as % of close
            - consecutive_gaps: Number of gaps in recent 10 bars

        Args:
            df: OHLCV DataFrame.
            gaps: List of GapDetection objects.
            return_gaps: If True, also return the subset of gaps that passed filtering.

        Returns:
            (X, y) or (X, y, gaps_passed) if return_gaps=True.
        """
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df.get("Volume", pd.Series(1, index=df.index))

        atr = self._compute_atr(high, low, close, period=14)
        returns = close.pct_change()
        volatility_20 = returns.rolling(20).std()
        volume_avg_20 = volume.rolling(20).mean()
        high_50 = high.rolling(50).max()
        low_50 = low.rolling(50).min()

        gap_indices_set = {g.bar_index for g in gaps}

        rows = []
        labels = []
        gaps_passed = []
        for gap in gaps:
            i = gap.bar_index
            if i < 50:
                continue

            prev_c = close.iloc[i - 1]

            # Count gaps in recent 10 bars (excluding current bar)
            consecutive_gaps = sum(1 for j in range(max(0, i - 10), i) if j in gap_indices_set)

            rows.append(
                {
                    "bar_index": i,
                    "gap_pct_abs": abs(gap.gap_pct),
                    "gap_direction": 1.0 if gap.gap_type == "up" else -1.0,
                    "pre_trend_5": float(close.iloc[max(0, i - 5)] / prev_c - 1)
                    if i >= 5 and prev_c > 0
                    else 0.0,
                    "pre_trend_20": float(close.iloc[max(0, i - 20)] / prev_c - 1)
                    if i >= 20 and prev_c > 0
                    else 0.0,
                    "volatility_20": float(volatility_20.iloc[i - 1])
                    if pd.notna(volatility_20.iloc[i - 1])
                    else 0.0,
                    "volume_ratio": float(volume.iloc[i] / volume_avg_20.iloc[i])
                    if pd.notna(volume_avg_20.iloc[i]) and volume_avg_20.iloc[i] > 0
                    else 1.0,
                    "atr_ratio": float(atr.iloc[i - 1] / prev_c)
                    if prev_c > 0 and pd.notna(atr.iloc[i - 1])
                    else 0.0,
                    "relative_position": float(
                        (prev_c - low_50.iloc[i - 1]) / (high_50.iloc[i - 1] - low_50.iloc[i - 1])
                    )
                    if pd.notna(high_50.iloc[i - 1])
                    and pd.notna(low_50.iloc[i - 1])
                    and high_50.iloc[i - 1] != low_50.iloc[i - 1]
                    else 0.5,
                    "day_of_week": float(gap.day_of_week),
                    "is_monday": 1.0 if gap.day_of_week == 0 else 0.0,
                    "pre_gap_range": float((high.iloc[i - 1] - low.iloc[i - 1]) / prev_c)
                    if prev_c > 0
                    else 0.0,
                    "consecutive_gaps": float(consecutive_gaps),
                }
            )
            labels.append(1.0 if gap.filled else 0.0)
            gaps_passed.append(gap)

        if not rows:
            if return_gaps:
                return pd.DataFrame(), pd.Series(dtype=float), []
            return pd.DataFrame(), pd.Series(dtype=float)

        X = pd.DataFrame(rows)
        y = pd.Series(labels, dtype=float)

        feature_cols = [c for c in X.columns if c != "bar_index"]
        to_return_X = X[feature_cols].copy()
        if return_gaps:
            return to_return_X, y, gaps_passed
        return to_return_X, y

    def _train_lgbm_cv(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train LightGBM with PurgedKFold CV."""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("lightgbm not installed")
            return {
                "auc": 0.5,
                "accuracy": 0.5,
                "f1": 0.0,
                "baseline": float(y.mean()),
                "fold_metrics": [],
            }

        cv = PurgedKFold(
            n_splits=min(self.n_splits, len(X) // 3, 5),
            pct_embargo=self.pct_embargo,
            label_span=self.look_forward,
        )

        fold_aucs = []
        fold_accs = []
        fold_f1s = []
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
            preds = (proba >= 0.5).astype(int)

            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

            try:
                auc = float(roc_auc_score(y_test, proba))
            except ValueError:
                auc = 0.5
            acc = float(accuracy_score(y_test, preds))
            try:
                f1 = float(f1_score(y_test, preds, zero_division=0))
            except ValueError:
                f1 = 0.0

            fold_aucs.append(auc)
            fold_accs.append(acc)
            fold_f1s.append(f1)
            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "auc": round(auc, 4),
                    "accuracy": round(acc, 4),
                    "f1": round(f1, 4),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                }
            )

        return {
            "auc": float(np.mean(fold_aucs)) if fold_aucs else 0.5,
            "accuracy": float(np.mean(fold_accs)) if fold_accs else 0.5,
            "f1": float(np.mean(fold_f1s)) if fold_f1s else 0.0,
            "baseline": float(y.mean()) if len(y) > 0 else 0.0,
            "fold_metrics": fold_metrics,
        }

    def _train_lgbm_final(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Train final LightGBM model on all data."""
        import lightgbm as lgb

        model = lgb.LGBMClassifier(**self.LGBM_PARAMS)
        model.fit(X, y)
        return model

    def _estimate_fill_bars(self, probability: float, gap_pct: float) -> int:
        """Estimate bars to fill based on probability and gap size.

        Simple heuristic: higher probability and smaller gaps fill faster.
        """
        base = max(1, self.look_forward // 2)
        if probability < 0.5:
            return self.look_forward + 1
        # Scale: higher prob -> fewer bars; smaller gap -> fewer bars
        adjustment = int((probability - 0.5) * self.look_forward * 0.5)
        gap_adjustment = int(min(self.look_forward * 0.3, abs(gap_pct) * 2))
        return max(1, base + adjustment - gap_adjustment)

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
            "config": {
                "look_forward": self.look_forward,
                "gap_threshold_pct": self.gap_threshold_pct,
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
