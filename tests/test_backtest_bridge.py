"""Unit tests for BacktestBridge."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.ml.backtest_bridge import BacktestBridge
from src.ml.features import FeatureEngineer
from src.ml.registry import ModelRegistry


def _make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "Open": prices * 0.998,
            "High": prices * 1.005,
            "Low": prices * 0.995,
            "Close": prices,
            "Volume": np.ones(n) * 1_000_000,
        },
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


@pytest.fixture
def sample_data():
    return _make_ohlcv(500, seed=42)


@pytest.fixture
def trained_on_features(sample_data):
    """Train a classifier on actual FeatureEngineer output."""
    eng = FeatureEngineer()
    features = eng.generate_features(sample_data)
    X = features.replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = (features.iloc[:, 0] > features.iloc[:, 0].median()).astype(int).values
    model = RandomForestClassifier(n_estimators=5, random_state=42, max_depth=3)
    model.fit(X, y)
    return model


@pytest.fixture
def bridge_and_registry(tmp_path, sample_data, trained_on_features):
    """Create bridge and registry with a registered model on matching features."""
    reg = ModelRegistry(models_dir=tmp_path)
    reg.register(
        "regime_rf", trained_on_features, metrics={"ic": 0.08}, metadata={"task": "regime"}
    )
    bridge = BacktestBridge(registry=reg)
    return bridge, reg, trained_on_features


class TestBacktestBridge:
    """Tests for BacktestBridge."""

    def test_generate_signal_scores_classification(
        self, bridge_and_registry, sample_data, trained_on_features
    ):
        """Test signal score generation with classification model on matching features."""
        bridge, _, _ = bridge_and_registry
        scores = bridge.generate_signal_scores(trained_on_features, sample_data)
        assert "signal_score" in scores.columns
        assert "ml_confidence" in scores.columns
        assert scores["signal_score"].between(0, 1).all()
        assert len(scores) > 0

    def test_generate_signal_scores_with_threshold(
        self, bridge_and_registry, trained_on_features, sample_data
    ):
        """Test that higher threshold reduces signal count."""
        bridge, _, _ = bridge_and_registry
        low = bridge.generate_signal_scores(trained_on_features, sample_data, min_confidence=0.3)
        high = bridge.generate_signal_scores(trained_on_features, sample_data, min_confidence=0.8)
        assert (high["signal_score"] > 0).sum() <= (low["signal_score"] > 0).sum()

    def test_generate_directional_signals(
        self, bridge_and_registry, trained_on_features, sample_data
    ):
        """Test directional signal generation from regime model."""
        bridge, _, _ = bridge_and_registry
        signals = bridge.generate_directional_signals(trained_on_features, sample_data)
        assert "direction" in signals.columns
        assert "confidence" in signals.columns
        assert set(signals["direction"].unique()).issubset({"BUY", "SELL", "HOLD"})

    def test_compare_with_baseline(self, bridge_and_registry):
        """Test ML vs baseline comparison."""
        bridge, _, _ = bridge_and_registry
        ml = {"return": 0.15, "sharpe": 0.8, "max_drawdown": -0.12}
        bl = {"return": 0.10, "sharpe": 0.5, "max_drawdown": -0.15}

        comparison = bridge.compare_with_baseline("test_model", "1.0", ml, bl)
        assert len(comparison) == 3
        metric_vals = dict(zip(comparison["metric"], comparison["diff"]))
        assert pytest.approx(metric_vals["return"], abs=1e-9) == 0.05
        assert pytest.approx(metric_vals["sharpe"], abs=1e-9) == 0.3

    def test_load_and_predict_with_classifier(self, bridge_and_registry, sample_data):
        """Test loading model and generating predictions."""
        bridge, _, _ = bridge_and_registry
        model, pred_df = bridge.load_and_predict("regime_rf", ohlcv_data=sample_data)

        assert "prediction" in pred_df.columns
        assert len(pred_df) > 0
        assert model is not None

    def test_load_and_predict_empty_data(self, bridge_and_registry):
        """Test load_and_predict with empty OHLCV returns empty DF."""
        bridge, _, _ = bridge_and_registry
        model, pred_df = bridge.load_and_predict("regime_rf", ohlcv_data=pd.DataFrame())
        assert pred_df.empty

    def test_predict_proba_columns(self, bridge_and_registry, sample_data):
        """Test that predict_proba adds probability columns."""
        bridge, _, _ = bridge_and_registry
        _, pred_df = bridge.load_and_predict("regime_rf", ohlcv_data=sample_data)

        prob_cols = [c for c in pred_df.columns if c.startswith("prob_")]
        assert len(prob_cols) > 0
        for col in prob_cols:
            assert pred_df[col].between(0, 1).all()
