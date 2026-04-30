"""Tests for new ML model implementations."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 500
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    y = pd.Series(
        (np.random.rand(n_samples) > 0.5).astype(int),
        name="target",
    )

    return X, y


@pytest.fixture
def sample_series():
    """Create sample time series."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=500, freq="D")
    values = np.random.randn(500).cumsum() + 100
    return pd.Series(values, index=dates, name="price")


class TestCatBoost:
    """Test CatBoost wrapper."""

    def test_catboost_fit_predict(self, sample_data):
        """Test CatBoost training and prediction."""
        from src.ml.models.catboost_wrapper import CatBoostForecaster

        X, y = sample_data
        model = CatBoostForecaster(n_estimators=10, verbose=False)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_catboost_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        from src.ml.models.catboost_wrapper import CatBoostForecaster

        X, y = sample_data
        model = CatBoostForecaster(n_estimators=10, verbose=False)
        model.fit(X, y)

        importance = model.get_feature_importance(X)
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert len(importance) == X.shape[1]


class TestChronos:
    """Test Chronos-2 wrapper."""

    @pytest.mark.skip(reason="Requires GPU and large model download")
    def test_chronos_zero_shot(self, sample_series):
        """Test Chronos zero-shot forecasting."""
        from src.ml.models.chronos import ChronosForecaster

        model = ChronosForecaster(model_size="tiny")
        predictions = model.predict(sample_series)

        assert "mean" in predictions
        assert len(predictions["mean"]) == model.prediction_length


class TestFinCast:
    """Test FinCast wrapper."""

    def test_fincast_not_implemented(self, sample_series):
        """Test that FinCast raises NotImplementedError."""
        from src.ml.models.fincast import FinCastForecaster

        model = FinCastForecaster()

        with pytest.raises(NotImplementedError):
            model.predict(sample_series)


class TestxLSTM:
    """Test xLSTM wrapper."""

    @pytest.mark.skip(reason="Requires xLSTM installation")
    def test_xlstm_fit_predict(self, sample_series):
        """Test xLSTM training and prediction."""
        from src.ml.models.xlstm import xLSTMForecaster

        model = xLSTMForecaster(
            hidden_size=32,
            num_layers=2,
            sequence_length=64,
            forecast_horizon=10,
        )

        model.fit(sample_series, epochs=1)
        predictions = model.predict(sample_series)

        assert len(predictions) == model.forecast_horizon


class TestModelSelectorIntegration:
    """Test model selector with new models."""

    def test_catboost_model_config(self, sample_data):
        """Test CatBoost integration with ModelSelector."""
        from src.ml.model_selector import ModelSelector

        X, y = sample_data
        selector = ModelSelector()

        rec = selector.recommend(
            X,
            y,
            task="classification",
            model_type="catboost",
        )

        assert rec.model.name == "catboost"

        result = selector.train_and_evaluate(
            X,
            y,
            rec.model,
            rec.validation,
        )

        assert result.model_name == "CatBoost"
        assert result.train_score >= 0

    def test_chronos_model_config(self, sample_data):
        """Test Chronos configuration in ModelSelector."""
        from src.ml.model_selector import ModelSelector

        X, y = sample_data
        selector = ModelSelector()

        rec = selector.recommend(
            X,
            y,
            task="regression",
            model_type="chronos",
        )

        assert rec.model.name == "chronos"
        assert rec.model.display_name == "Chronos-2 (Foundation Model)"


class TestPatternClassifierNewModels:
    """Test pattern classifier with new models."""

    def test_catboost_pattern_classifier(self, sample_data):
        """Test CatBoost in pattern classifier."""
        from src.ml.pattern_classifier import PatternClassifier

        X, y = sample_data
        clf = PatternClassifier(
            model_type="catboost",
            n_estimators=10,
            random_state=42,
        )

        result = clf.train(X, y)

        assert result.train_auc > 0
        assert result.test_auc > 0

    def test_supported_models_updated(self):
        """Test that supported models list is updated."""
        from src.ml.pattern_classifier import PatternClassifier

        assert "lightgbm" not in PatternClassifier.SUPPORTED_MODELS
        assert "xgboost" not in PatternClassifier.SUPPORTED_MODELS
        assert "catboost" in PatternClassifier.SUPPORTED_MODELS
        assert "chronos" in PatternClassifier.SUPPORTED_MODELS
        assert "fincast" in PatternClassifier.SUPPORTED_MODELS
        assert "xlstm" in PatternClassifier.SUPPORTED_MODELS
