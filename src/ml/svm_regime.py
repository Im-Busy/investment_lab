"""P25: SVM market classifier — raw price sequences as features for regime detection.

Uses raw price sequences (not technical indicators) as features for regime
classification. Paper result: 82% precision vs 64% with technical indicators.

Reference: Paiva et al. (2016), Sadeghi et al. (2021), Fisichella et al. (2021).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

RegimeLabel = Literal["up", "down", "side"]
DEFAULT_SEQUENCE_LENGTH = 100


@dataclass
class SVMRegimeResult:
    """Result of SVM regime classification."""

    regime: RegimeLabel
    confidence: float  # probability estimate or distance
    prediction_time: pd.Timestamp | None = None
    regime_probabilities: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"SVMRegime({self.regime}, conf={self.confidence:.3f})"


class SVMRegimeClassifier:
    """SVM-based market regime classifier using raw price sequences.

    Trains an SVM on sliding windows of raw OHLCV data to classify market
    as uptrend, downtrend, or sideways. Outperforms indicator-based methods
    by 18 percentage points (82% vs 64% precision per Paiva 2016).

    Key insight: raw price sequence patterns capture regime information
    that hand-crafted indicators miss.

    Usage:
        from src.ml.svm_regime import SVMRegimeClassifier

        svm = SVMRegimeClassifier(window=100)
        svm.fit(X_prices, y_regime_labels)
        result = svm.predict(current_window)
    """

    def __init__(
        self,
        window: int = DEFAULT_SEQUENCE_LENGTH,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: float | str = "scale",
        class_weight: dict | str = "balanced",
        probability: bool = True,
        random_state: int | None = 42,
        n_components: int = 20,
    ) -> None:
        self.window = window
        self.kernel = kernel
        self.C = C
        self.gamma_val = gamma
        self.class_weight = class_weight
        self.probability = probability
        self.random_state = random_state
        self.n_components = n_components

        self._model: SVC | None = None
        self._scaler = StandardScaler()
        self._is_fitted = False
        self._feature_names: list[str] = []
        self._label_map: dict[str, int] = {"up": 1, "down": -1, "side": 0}

    @staticmethod
    def _extract_sequences(
        df: pd.DataFrame,
        window: int,
    ) -> pd.DataFrame:
        """Extract raw price sequences as features.

        Creates flat feature vectors from price windows. Each row = one
        window of raw returns, normalized.

        Args:
            df: OHLCV DataFrame with columns ['close', 'high', 'low', 'open', 'volume'].
            window: Lookback window length.

        Returns:
            DataFrame of shape (n_windows, window * n_prices) with flattened sequences.
        """
        columns = ["close", "high", "low", "open", "volume"]
        available = [c for c in columns if c in df.columns]
        if "close" not in available:
            available = ["close"]

        features = []
        feature_names = []

        for col in available:
            series = df[col].values
            for i in range(window):
                shifted = pd.Series(np.roll(series, i + 1), index=df.index)
                shifted.iloc[: i + 1] = np.nan
                name = f"{col}_lag{i + 1}"
                if len(features) == 0:
                    features.append(shifted.to_frame(name))
                else:
                    features.append(shifted.to_frame(name))
                feature_names.append(name)

        result = pd.concat(features, axis=1)
        result = result.iloc[window:]  # drop NaN rows at start
        return result

    @staticmethod
    def _build_regime_labels(
        df: pd.DataFrame,
        window: int,
        forward_window: int = 5,
        threshold: float = 0.01,
    ) -> pd.Series:
        """Build up/down/side labels from forward returns.

        Args:
            df: OHLCV DataFrame with 'close' column.
            window: Lookback window for features.
            forward_window: Forward window for return calculation.
            threshold: Return threshold for up/down classification.

        Returns:
            Series of regime labels aligned with sequence features.
        """
        close = df["close"].values
        n = len(close)
        labels = np.full(n, np.nan)

        for i in range(window, n - forward_window):
            fwd_return = (close[i + forward_window] - close[i]) / close[i]
            if fwd_return > threshold:
                labels[i] = 1
            elif fwd_return < -threshold:
                labels[i] = -1
            else:
                labels[i] = 0

        result = pd.Series(labels, index=df.index, name="regime_label")
        return result.dropna()

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | None = None,
        forward_window: int = 5,
        threshold: float = 0.01,
    ) -> SVMRegimeClassifier:
        """Train SVM on raw price sequences.

        If y is None, auto-generates regime labels from forward returns.

        Args:
            df: OHLCV DataFrame.
            y: Optional pre-computed regime labels (1=up, -1=down, 0=side).
            forward_window: Forward window for auto-labeling (if y is None).
            threshold: Return threshold for auto-labeling.
        """
        X = self._extract_sequences(df, self.window)
        self._feature_names = list(X.columns)

        if y is None:
            y = self._build_regime_labels(df, self.window, forward_window, threshold)

        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        if len(X) < 20:
            raise ValueError(f"Only {len(X)} samples after alignment. Need >= 20.")

        X_scaled = self._scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

        self._model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma_val,
            class_weight=self.class_weight,
            probability=self.probability,
            random_state=self.random_state,
        )
        self._model.fit(X_scaled, y)
        self._is_fitted = True

        n_classes = len(np.unique(y))
        support_per_class = (
            dict(
                zip(
                    ["up", "down", "side"][:n_classes],
                    [int(s) for s in self._model.n_support_],
                )
            )
            if hasattr(self._model, "n_support_")
            else {}
        )
        logger.info(
            "SVMRegimeClassifier fitted: n_samples=%d, n_features=%d, kernel=%s, C=%.2f, support=%s",
            len(X),
            len(self._feature_names),
            self.kernel,
            self.C,
            support_per_class,
        )
        return self

    def predict(self, df_row: pd.Series | np.ndarray) -> SVMRegimeResult:
        """Predict regime for a single price window.

        Args:
            df_row: Raw price values (flattened window) as Series or array.

        Returns:
            SVMRegimeResult with regime label and confidence.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if isinstance(df_row, pd.Series):
            values = df_row.values.reshape(1, -1)
        else:
            values = np.array(df_row).reshape(1, -1)

        values_scaled = self._scaler.transform(values)
        pred = int(self._model.predict(values_scaled)[0])
        prob = float(np.max(self._model.predict_proba(values_scaled))) if self.probability else 1.0

        regime_map: dict[int, RegimeLabel] = {1: "up", -1: "down", 0: "side"}
        regime = regime_map.get(pred, "side")

        probs = {}
        if self.probability:
            raw_probs = self._model.predict_proba(values_scaled)[0]
            classes = self._model.classes_
            for cls, p in zip(classes, raw_probs):
                label = regime_map.get(int(cls), "side")
                probs[label] = float(p)

        return SVMRegimeResult(
            regime=regime,
            confidence=prob,
            regime_probabilities=probs,
        )

    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict regime for a batch of windows.

        Args:
            X: DataFrame of sequence features (from _extract_sequences).

        Returns:
            DataFrame with 'regime' and 'confidence' columns.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model not fitted.")

        X_scaled = self._scaler.transform(X)
        preds = self._model.predict(X_scaled)
        probs = self._model.predict_proba(X_scaled) if self.probability else None

        regime_map = {1: "up", -1: "down", 0: "side"}
        results = pd.DataFrame(index=X.index)
        results["regime"] = [regime_map.get(int(p), "side") for p in preds]
        results["confidence"] = np.max(probs, axis=1) if probs is not None else 1.0

        if probs is not None:
            classes = self._model.classes_
            for cls, idx in zip(classes, range(len(classes))):
                label = regime_map.get(int(cls), "side")
                results[f"prob_{label}"] = probs[:, idx]

        return results

    def get_support_vectors(self) -> np.ndarray:
        """Return support vectors (in original scale)."""
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._scaler.inverse_transform(self._model.support_vectors_)

    def get_decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Return decision function values for batch."""
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        X_scaled = self._scaler.transform(X)
        return self._model.decision_function(X_scaled)  # type: ignore[return-value]
