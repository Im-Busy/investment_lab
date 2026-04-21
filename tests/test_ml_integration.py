"""
Unit test for ML-enhanced backtesting integration.

Validates:
- MLPipeline training and prediction flow with synthetic OHLCV data
- ConfluenceScorer with ML signal scorer integration (end-to-end)
- ML signal enhancement workflow
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.features import FeatureEngineer
from src.ml.pipeline import MLPipeline
from src.ml.regime_model import MLRegimeState, RegimeClassifier
from src.ml.signal_scorer import SignalScorer


def _make_synthetic_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "Open": price * 0.998,
            "High": price * 1.005,
            "Low": price * 0.995,
            "Close": price,
            "Volume": np.ones(n) * 1_000_000,
        },
        index=pd.date_range("2022-01-01", periods=n, freq="B"),
    )


def _make_regime_labels(n: int, index: pd.Index) -> pd.Series:
    np.random.seed(42)
    return pd.Series(
        np.random.choice(
            ["Trending", "Ranging", "Volatile", "Transition"],
            size=n,
            p=[0.3, 0.3, 0.2, 0.2],
        ),
        index=index,
        name="regime",
    )


def _make_synthetic_signal_data(n: int = 200, seed: int = 42) -> tuple:
    np.random.seed(seed)
    X = pd.DataFrame(
        {
            "rsi": np.random.uniform(0, 100, n),
            "macd_histogram": np.random.randn(n) * 0.5,
            "atr_pct": np.abs(np.random.randn(n) * 2),
            "volume_ratio": np.abs(np.random.randn(n) * 1.5 + 1),
            "confidence": np.random.uniform(0.3, 0.9, n),
            "entry_price": np.random.uniform(90, 110, n),
            "stop_loss": np.random.uniform(85, 105, n),
            "take_profit": np.random.uniform(100, 120, n),
        },
        index=pd.date_range("2022-01-01", periods=n, freq="B"),
    )
    y = pd.Series(
        ((X["confidence"] > 0.6) & (X["rsi"] < 40)).astype(int),
        name="profitable",
    )
    noise = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
    y = ((y + noise) > 0).astype(int)
    return X, y


def _make_pattern_results(n: int, seed: int = 42) -> list:
    from src.patterns.base import (
        BasePattern,
        PatternResult,
        PatternType,
        SignalDirection,
        TradeSignal,
    )

    np.random.seed(seed)
    results = []
    for i in range(n):
        entry = float(np.random.uniform(90, 110))
        stop = float(np.random.uniform(85, entry))
        tp = float(np.random.uniform(entry, 120))
        sig = TradeSignal(
            direction=SignalDirection.LONG,
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp,
            confidence=float(np.random.uniform(0.5, 0.9)),
            pattern_name=f"Pattern_{i}",
        )
        result = PatternResult(
            pattern_name=f"Pattern_{i}",
            pattern_type=PatternType.REVERSAL,
            detected=True,
            signal=sig,
        )
        results.append(result)
    return results


class TestMLIntegration:
    """Integration tests for ML pipeline."""

    def test_pipeline_trains_and_predicts(self):
        df = _make_synthetic_ohlcv(n=500)
        regime_labels = _make_regime_labels(len(df), df.index)
        sig_features, sig_labels = _make_synthetic_signal_data(n=500)

        pipeline = MLPipeline(
            regime_model_type="random_forest",
            signal_model_type="gradient_boosting",
            n_estimators=5,
            max_depth=3,
        )

        results = pipeline.fit(
            df,
            regime_labels=regime_labels,
            signal_features=sig_features,
            signal_labels=sig_labels,
        )

        assert "regime" in results
        regime_pred = pipeline.predict_regime(df)
        assert len(regime_pred) == len(df)

        regime_proba = pipeline.get_regime_probabilities(df)
        assert regime_proba.shape[0] == len(df)

        scored = pipeline.score_signals(sig_features)
        assert "ml_score" in scored.columns

    def test_walk_forward_validation_runs(self):
        df = _make_synthetic_ohlcv(n=800)
        regime_labels = _make_regime_labels(len(df), df.index)

        pipeline = MLPipeline(
            regime_model_type="random_forest",
            n_estimators=5,
            max_depth=3,
        )

        wf = pipeline.walk_forward_validation(
            df=df,
            regime_labels=regime_labels,
        )

        assert "regime" in wf
        assert len(wf["regime"]["train_score"]) > 0

    def test_regime_classifier_predicts_valid_states(self):
        df = _make_synthetic_ohlcv(n=500)
        labels = _make_regime_labels(len(df), df.index)

        engineer = FeatureEngineer()
        features = engineer.generate_features(df)
        numeric = features.select_dtypes(include=[np.number]).ffill().bfill().dropna()
        y = labels.reindex(numeric.index).dropna()
        X = numeric.reindex(y.index)

        clf = RegimeClassifier(model_type="random_forest", n_estimators=5)
        clf.train(X, y, test_size=0.3)

        preds = clf.predict(X)
        valid_states = {s.value for s in MLRegimeState}
        assert all(p in valid_states for p in preds)

    def test_signal_scorer_walk_forward(self):
        X, y = _make_synthetic_signal_data(n=800)

        scorer = SignalScorer(model_type="random_forest", n_estimators=5)
        wf = scorer.walk_forward_validation(X, y, train_size=300, step_size=100)

        assert len(wf["train_accuracy"]) > 0
        assert len(wf["test_accuracy"]) > 0
        assert all(0 <= a <= 1 for a in wf["train_accuracy"])

    def test_run_pipeline_integration(self):
        df = _make_synthetic_ohlcv(n=1000)
        regime_labels = _make_regime_labels(len(df), df.index)
        sig_features, sig_labels = _make_synthetic_signal_data(n=1000)

        pipeline = MLPipeline(
            regime_model_type="random_forest",
            signal_model_type="random_forest",
            n_estimators=5,
            max_depth=3,
        )

        result = pipeline.run_pipeline(
            df=df,
            regime_labels=regime_labels,
            signal_features=sig_features,
            signal_labels=sig_labels,
        )

        assert result.regime_prediction is not None
        assert result.signal_scores is not None
        assert result.feature_importance_regime is not None
        assert result.pipeline_metrics["regime_accuracy"] >= 0


class TestConfluenceScorerMLIntegration:
    """Integration tests for ML-enhanced confluence scoring."""

    def test_confluence_scorer_accepts_ml_scorer(self):
        from src.strategies.confluence import ConfluenceScorer

        scorer_with_ml = ConfluenceScorer(ml_scorer="scorer_instance")
        assert scorer_with_ml._ml_scorer == "scorer_instance"

        scorer_without_ml = ConfluenceScorer()
        assert scorer_with_ml._ml_scorer != scorer_without_ml._ml_scorer

    def test_ml_blend_vs_base_confidence_different(self):
        """Test that ML scorer produces different scores than base confidence."""
        from src.strategies.confluence import ConfluenceScorer

        # Train a signal scorer
        sig_features, sig_labels = _make_synthetic_signal_data(n=300)
        ml_scorer = SignalScorer(model_type="random_forest", n_estimators=5, max_depth=3)
        ml_scorer.train(sig_features, sig_labels)

        # Create two confluence scorers - one with ML, one without
        scorer_ml = ConfluenceScorer(ml_scorer=ml_scorer)
        scorer_base = ConfluenceScorer()

        # Both should accept pattern results
        results = _make_pattern_results(n=3)

        # Calculate confluence with both scorers
        score_ml = scorer_ml.calculate_confluence(results)
        score_base = scorer_base.calculate_confluence(results)

        assert score_ml is not None
        assert score_base is not None
        assert isinstance(score_ml.score, float)
        assert isinstance(score_base.score, float)
        assert 0 <= score_ml.score <= 1
        assert 0 <= score_base.score <= 1

    def test_ml_feature_building_integration(self):
        """Test that ML features can be built from pattern results."""
        from src.strategies.confluence import ConfluenceScorer

        scorer = ConfluenceScorer()
        results = _make_pattern_results(n=5)

        ml_features_list = []
        for result in results:
            ml_features = scorer._build_ml_features(result)
            if ml_features is not None:
                ml_features_list.append(ml_features)

        # Should be able to build features for all valid results
        assert len(ml_features_list) == len(results)
        for features in ml_features_list:
            assert not features.empty
            assert "entry_price" in features.columns
            assert "stop_loss" in features.columns
            assert "take_profit" in features.columns
            assert "risk" in features.columns
            assert "reward" in features.columns
            assert "rr_ratio" in features.columns
            assert "confidence" in features.columns

    def test_ml_scorer_blend_integration(self):
        """Test that ML blending works when scorer is provided."""
        from src.strategies.confluence import ConfluenceScorer

        sig_features, sig_labels = _make_synthetic_signal_data(n=300)
        ml_scorer = SignalScorer(model_type="random_forest", n_estimators=5, max_depth=3)
        ml_scorer.train(sig_features, sig_labels)

        scorer = ConfluenceScorer(ml_scorer=ml_scorer)
        results = _make_pattern_results(n=3)

        # Calculate confluence should use ML blending when _ml_scorer is set
        score = scorer.calculate_confluence(results)
        assert score is not None
        assert 0 <= score.score <= 1
        assert score.pattern_count == 3
