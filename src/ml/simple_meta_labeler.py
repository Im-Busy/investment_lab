"""
Simple Meta-Labeler — post-filter for ML signal quality.

Trains a secondary model that predicts: "given the primary model's confidence
AND current market context, will this signal actually be profitable?"

Unlike the full MetaLabeler (meta_labeler.py) designed for pattern-detector
trade signals, this lightweight version works directly with per-bar probability
scores from any classifier. It uses:

1. Primary model probability score
2. Top 5 SHAP features as context
3. LogisticRegression with Youden threshold

Usage:
    from src.ml.simple_meta_labeler import SimpleMetaLabeler, train_meta_labeler

    meta = train_meta_labeler(primary_probs, labels, context_features)
    filtered = meta.filter(primary_probs, context_features)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MetaLabelResult:
    """Result of meta-labeler training."""

    train_auc: float
    test_auc: float
    threshold: float
    n_train: int
    n_test: int
    signal_reduction_pct: float
    train_win_rate: float
    test_win_rate: float
    feature_names: list[str] = None

    def __post_init__(self):
        if self.feature_names is None:
            self.feature_names = []

    def summary(self) -> dict[str, Any]:
        return {
            "train_auc": round(self.train_auc, 4),
            "test_auc": round(self.test_auc, 4),
            "threshold": round(self.threshold, 4),
            "signal_reduction_pct": round(self.signal_reduction_pct, 1),
            "train_win_rate": round(self.train_win_rate, 3),
            "test_win_rate": round(self.test_win_rate, 3),
        }


class SimpleMetaLabeler:
    """Lightweight meta-labeler for filtering ML signals.

    Trains LogisticRegression on: [primary_prob] + [top_context_features].
    Target: 1 if signal was profitable, 0 otherwise.

    Attributes:
        model: Trained sklearn LogisticRegression.
        threshold: Youden's J optimal probability threshold.
        feature_names: Feature column names.
    """

    def __init__(self):
        self.model: Any = None
        self.threshold: float = 0.5
        self.feature_names: list[str] = []

    def train(
        self,
        primary_probs: pd.Series,
        context_features: pd.DataFrame,
        labels: pd.Series,
    ) -> MetaLabelResult:
        """Train meta-labeler on primary model outputs + context.

        Args:
            primary_probs: Primary model probability scores.
            context_features: Context features at prediction time (top SHAP features).
            labels: Actual binary outcomes (1 = profitable, 0 = not).

        Returns:
            MetaLabelResult with train/test metrics and threshold.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        # Build feature matrix: primary_prob + context
        meta_features = context_features.copy()
        meta_features["_primary_prob"] = primary_probs.values
        self.feature_names = list(meta_features.columns)

        common = meta_features.dropna().index.intersection(labels.dropna().index)
        X = meta_features.loc[common]
        y = labels.loc[common]

        if len(X) < 100:
            logger.warning(f"Only {len(X)} samples for meta-labeler — skipping")
            return MetaLabelResult(
                train_auc=0.5,
                test_auc=0.5,
                threshold=0.5,
                n_train=0,
                n_test=0,
                signal_reduction_pct=0,
                train_win_rate=0,
                test_win_rate=0,
            )

        # Chronological 70/30 split
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model = LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced")
        self.model.fit(X_train, y_train)

        train_probs = self.model.predict_proba(X_train)[:, 1]
        test_probs = self.model.predict_proba(X_test)[:, 1]

        train_auc = float(roc_auc_score(y_train, train_probs))
        test_auc = float(roc_auc_score(y_test, test_probs))

        self.threshold = _find_youden_threshold(test_probs, y_test)

        # Signal reduction: what fraction of primary_model_positive signals
        # does the meta-labeler reject?
        primary_positive = (primary_probs.loc[X_test.index] >= 0.5).sum()
        meta_positive = (test_probs >= self.threshold).sum()
        signal_reduction = (
            (1 - meta_positive / max(primary_positive, 1)) * 100 if primary_positive > 0 else 0
        )

        # Win rates
        train_win_rate = float(y_train.mean())
        test_win_rate = float(y_test.mean())

        logger.info(
            f"Meta-labeler: Train AUC={train_auc:.4f}, Test AUC={test_auc:.4f}, "
            f"threshold={self.threshold:.3f}, "
            f"signal_reduction={signal_reduction:.1f}%"
        )

        return MetaLabelResult(
            train_auc=train_auc,
            test_auc=test_auc,
            threshold=float(self.threshold),
            n_train=len(y_train),
            n_test=len(y_test),
            signal_reduction_pct=float(signal_reduction),
            train_win_rate=train_win_rate,
            test_win_rate=test_win_rate,
            feature_names=self.feature_names,
        )

    def filter(
        self,
        primary_probs: pd.Series,
        context_features: pd.DataFrame,
    ) -> pd.Series:
        """Filter signals: return boolean mask of trades to take.

        Args:
            primary_probs: Primary model probability scores.
            context_features: Context features.

        Returns:
            Boolean Series, True = take trade.
        """
        if self.model is None:
            return pd.Series(True, index=primary_probs.index)

        meta_features = context_features.copy()
        meta_features["_primary_prob"] = primary_probs.values

        probs = self.model.predict_proba(meta_features[self.feature_names].fillna(0))[:, 1]

        return pd.Series(probs >= self.threshold, index=primary_probs.index, name="take_trade")

    def predict_meta_proba(
        self,
        primary_probs: pd.Series,
        context_features: pd.DataFrame,
    ) -> pd.Series:
        """Return meta-labeler probability scores."""
        if self.model is None:
            return pd.Series(0.5, index=primary_probs.index)

        meta_features = context_features.copy()
        meta_features["_primary_prob"] = primary_probs.values

        probs = self.model.predict_proba(meta_features[self.feature_names].fillna(0))[:, 1]

        return pd.Series(probs, index=primary_probs.index, name="meta_probability")


def _find_youden_threshold(probs: np.ndarray, labels: pd.Series) -> float:
    """Find optimal threshold using Youden's J statistic (sensitivity + specificity - 1)."""
    from sklearn.metrics import roc_curve

    if len(np.unique(labels)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)

    if best_idx < len(thresholds):
        return float(thresholds[best_idx])
    return 0.5


def train_meta_labeler(
    primary_probs: pd.Series,
    labels: pd.Series,
    context_features: pd.DataFrame,
    top_n_features: int = 5,
) -> SimpleMetaLabeler | None:
    """Convenience function: train meta-labeler with top-N context features.

    Args:
        primary_probs: Primary model probability scores (index-aligned).
        labels: Binary outcomes (index-aligned).
        context_features: Feature DataFrame. Uses top_n most important columns.
        top_n_features: Number of context features to use.

    Returns:
        Trained SimpleMetaLabeler or None if insufficient data.
    """
    # Select top-N features by variance (simple heuristic)
    if context_features.shape[1] > top_n_features:
        variances = context_features.var().sort_values(ascending=False)
        top_cols = variances.head(top_n_features).index.tolist()
        ctx = context_features[top_cols]
    else:
        ctx = context_features

    meta = SimpleMetaLabeler()
    result = meta.train(primary_probs, ctx, labels)

    if result.n_train < 100:
        logger.warning(
            f"Meta-labeler insufficient data ({result.n_train} samples). "
            f"Need 100+ for reliable training."
        )
        return None

    if result.test_auc < 0.52:
        logger.warning(
            f"Meta-labeler test AUC {result.test_auc:.4f} < 0.52 — "
            f"not adding value over random filter."
        )
        return None

    return meta


class MetaLabelContextFeatures:
    """Generate features that are DIFFERENT from primary model features.

    The primary model uses price/momentum/return/cross-asset features.
    The meta-labeler needs unique features that the primary model doesn't
    have access to:
    1. Regime gate: ADX, volatility state, market direction
    2. Volatility context: current vol vs recent average
    3. Signal clustering: signal density in recent window
    4. Recency: time since last trade signal
    """

    def __init__(
        self,
        adx_window: int = 14,
        vol_window: int = 20,
        vol_lookback: int = 60,
        cluster_window: int = 30,
    ):
        self.adx_window = adx_window
        self.vol_window = vol_window
        self.vol_lookback = vol_lookback
        self.cluster_window = cluster_window

    def generate(
        self,
        df: pd.DataFrame,
        signal_dates: pd.DatetimeIndex | None = None,
        primary_probs: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Generate meta-labeler context features at signal bars.

        Args:
            df: OHLCV DataFrame with 'High', 'Low', 'Close', 'Volume'.
            signal_dates: If provided, return features only at these dates.
                          If None, return for all bars.
            primary_probs: Primary model probability scores (used for signal clustering).

        Returns:
            DataFrame with context features, index-aligned to signal_dates.
        """
        features = pd.DataFrame(index=df.index)

        # --- Regime gate features ---
        adx = self._compute_adx(df, self.adx_window)
        features["adx"] = adx
        features["adx_trend"] = (adx >= 25).astype(float)

        # Volatility regime: current vol / recent average vol
        vol = self._compute_volatility(df, self.vol_window)
        mean_vol = vol.rolling(self.vol_lookback).mean()
        features["vol_regime_ratio"] = vol / mean_vol.replace(0, np.nan)
        features["vol_regime_high"] = (features["vol_regime_ratio"] >= 1.5).astype(float)
        features["vol_regime_low"] = (features["vol_regime_ratio"] <= 0.5).astype(float)

        # Market direction: recent return and trend strength
        ret = df["Close"].pct_change()
        features["recent_return_5d"] = ret.rolling(5).sum()
        features["recent_return_20d"] = ret.rolling(20).sum()
        features["up_days_ratio_20d"] = (ret > 0).rolling(20).mean()

        # Volatility skew: ratio of upside to downside volatility
        up_vol = ret.clip(lower=0).rolling(self.vol_window).std()
        down_vol = ret.clip(upper=0).rolling(self.vol_window).std()
        features["vol_skew"] = up_vol / down_vol.replace(0, np.nan)

        # --- Signal clustering features ---
        if signal_dates is not None:
            features["is_signal"] = 0.0
            features.loc[signal_dates, "is_signal"] = 1.0
            features["signals_past_N"] = features["is_signal"].rolling(self.cluster_window).sum()
            features["signal_density"] = features["signals_past_N"] / self.cluster_window
            features = features.drop(columns=["is_signal"])

        # --- Recent probability distribution ---
        if primary_probs is not None:
            aligned = primary_probs.reindex(df.index).fillna(0.5)
            features["prob_rolling_mean"] = aligned.rolling(10).mean()
            features["prob_rolling_std"] = aligned.rolling(10).std().fillna(0)
            features["prob_above_ma"] = (aligned >= features["prob_rolling_mean"]).astype(float)

        # --- Time since last trade signal ---
        if signal_dates is not None:
            features["days_since_signal"] = self._compute_days_since_signal(df.index, signal_dates)
        else:
            features["days_since_signal"] = 0.0

        # Drop rows with NaN from rolling windows
        features = features.dropna()

        if signal_dates is not None:
            return features.loc[features.index.intersection(signal_dates)]

        return features

    @staticmethod
    def _compute_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
        high, low, close = df["High"], df["Low"], df["Close"]
        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window).mean()
        up = high.diff()
        down = -low.diff()
        pdm = np.where((up > down) & (up > 0), up, 0)
        ndm = np.where((down > up) & (down > 0), down, 0)
        pdm_smooth = pd.Series(pdm, index=df.index).rolling(window).mean()
        ndm_smooth = pd.Series(ndm, index=df.index).rolling(window).mean()
        pdi = 100 * pdm_smooth / atr
        ndi = 100 * ndm_smooth / atr
        dx = 100 * abs(pdi - ndi) / (pdi + ndi + 1e-10)
        return dx.rolling(window).mean()

    @staticmethod
    def _compute_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
        return df["Close"].pct_change().rolling(window).std()

    @staticmethod
    def _compute_days_since_signal(
        index: pd.DatetimeIndex,
        signal_dates: pd.DatetimeIndex,
    ) -> pd.Series:
        """Compute trading days since the most recent signal for each bar."""
        signal_set = set(signal_dates)
        days_since: list[float] = []
        last_signal = -1
        for i, dt in enumerate(index):
            if dt in signal_set:
                last_signal = i
                days_since.append(0.0)
            elif last_signal >= 0:
                days_since.append(float(i - last_signal))
            else:
                days_since.append(float("nan"))
        result = pd.Series(days_since, index=index)
        return result.fillna(result.max())
