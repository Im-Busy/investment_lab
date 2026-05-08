"""Unit tests for ML components:
- FeatureEngineer
- RegimeClassifier
- SignalScorer
- MLPipeline
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.features import FeatureEngineer
from src.ml.pipeline import MLPipeline
from src.ml.regime_model import RegimeClassifier
from src.ml.signal_scorer import SignalScorer


def _make_ohlcv_data(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Create synthetic OHLCV data for testing."""
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
        index=pd.date_range("2022-01-01", periods=n, freq="B"),
    )


class TestFeatureEngineer:
    """Tests for FeatureEngineer."""

    def test_generate_features(self):
        """Test that features are generated without errors."""
        df = _make_ohlcv_data(seed=42)
        eng = FeatureEngineer()

        features = eng.generate_features(df)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(df)
        assert len(features.columns) > 20  # Should have many features

    def test_feature_names(self):
        """Test that feature names can be retrieved."""
        eng = FeatureEngineer()
        names = eng.get_feature_names()

        assert len(names) > 20

    def test_custom_windows(self):
        """Test custom window sizes."""
        eng = FeatureEngineer(
            price_windows=[10, 20],
            momentum_windows=[10, 20],
            volatility_windows=[10, 20],
        )
        df = _make_ohlcv_data(n=500, seed=42)
        features = eng.generate_features(df)

        # Verify specific windows are present
        assert "return_10" in features.columns
        assert "return_20" in features.columns
        assert "rsi_10" in features.columns
        assert "rsi_20" in features.columns

    def test_features_contain_no_inf(self):
        """Test that features don't contain infinity values."""
        df = _make_ohlcv_data(n=500, seed=42)
        # Add a zero volume row to test edge cases
        df.iloc[0, df.columns.get_loc("Volume")] = 0

        eng = FeatureEngineer()
        features = eng.generate_features(df)

        numeric_features = features.select_dtypes(include=[np.number])
        assert not np.isinf(numeric_features.values).any()

    def test_feature_columns_for_categories(self):
        """Test that all feature categories are represented."""
        df = _make_ohlcv_data(seed=42)
        eng = FeatureEngineer()
        features = eng.generate_features(df)
        cols = set(features.columns)

        # Price features
        assert "hl_range" in cols or any("return_" in c for c in cols)

        # Momentum features
        assert any("rsi_" in c for c in cols)
        assert "macd" in cols

        # Volatility features
        assert any("atr_" in c for c in cols)
        assert any("volatility_" in c for c in cols)

        # Volume features
        assert any("volume_ratio_" in c for c in cols)

        # Regime features
        assert "adx" in cols


class TestRegimeClassifier:
    """Tests for RegimeClassifier."""

    def _make_synthetic_labels(self, index: pd.Index) -> pd.Series:
        """Create synthetic regime labels aligned with given index."""
        n = len(index)
        np.random.seed(42)
        regimes = np.random.choice(
            ["Trending", "Ranging", "Volatile", "Transition"],
            size=n,
            p=[0.3, 0.3, 0.2, 0.2],
        )
        return pd.Series(regimes, index=index, name="regime")

    def _prepare_data(self, df: pd.DataFrame) -> tuple:
        """Generate features and labels, cleaning NaN."""
        eng = FeatureEngineer()
        features = eng.generate_features(df)
        features = features.select_dtypes(include=[np.number])
        features = features.dropna(axis=1, how="all").ffill().bfill()
        valid = features.notna().all(axis=1)
        features = features[valid].reset_index(drop=True)
        labels = self._make_synthetic_labels(features.index)
        return features, labels

    def test_train_and_predict(self):
        """Test basic training and prediction flow."""
        df = _make_ohlcv_data(n=1500, seed=42)
        features, labels = self._prepare_data(df)

        classifier = RegimeClassifier(model_type="random_forest", n_estimators=10)
        results = classifier.train(features, labels, test_size=0.3)

        assert "train_accuracy" in results
        assert "test_accuracy" in results
        assert results["train_accuracy"] >= 0
        assert results["test_accuracy"] >= 0

        # Test prediction
        predictions = classifier.predict(features)
        assert len(predictions) == len(features)
        assert all(isinstance(p, str) for p in predictions)

    def test_predict_proba(self):
        """Test probability predictions."""
        df = _make_ohlcv_data(n=1500, seed=42)
        features, labels = self._prepare_data(df)

        classifier = RegimeClassifier(model_type="random_forest", n_estimators=10)
        classifier.train(features, labels, test_size=0.3)

        proba = classifier.predict_proba(features)

        assert proba.shape[1] > 0
        assert (proba.sum(axis=1).round(6) == 1.0).all()

    def test_unsupported_model_raises(self):
        """Test that unsupported model type raises an error."""
        with pytest.raises(ValueError):
            RegimeClassifier(model_type="unsupported")

    def test_feature_importance(self):
        """Test feature importance retrieval."""
        df = _make_ohlcv_data(n=1500, seed=42)
        features, labels = self._prepare_data(df)

        classifier = RegimeClassifier(model_type="random_forest", n_estimators=10)
        classifier.train(features, labels, test_size=0.3)

        importance = classifier.get_feature_importance(top_n=10)

        assert len(importance) > 0
        assert "feature" in importance.columns
        assert "importance" in importance.columns


class TestSignalScorer:
    """Tests for SignalScorer."""

    def _make_signal_data(self, n: int = 500, seed: int = 42) -> tuple:
        """Create synthetic signal features and labels."""
        np.random.seed(seed)
        X = pd.DataFrame(
            {
                "rsi": np.random.uniform(0, 100, n),
                "macd_histogram": np.random.randn(n) * 0.5,
                "atr_pct": np.abs(np.random.randn(n) * 2),
                "volume_ratio": np.abs(np.random.randn(n) * 1.5 + 1),
                "confidence": np.random.uniform(0.3, 0.9, n),
            },
            index=pd.date_range("2022-01-01", periods=n, freq="B"),
        )
        # Label: profitable if confidence > 0.6 and RSI < 40 (simplified)
        y = pd.Series(
            ((X["confidence"] > 0.6) & (X["rsi"] < 40)).astype(int),
            name="profitable",
        )
        # Add some noise
        noise = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
        y = ((y + noise) > 0).astype(int)

        return X, y

    def test_train_and_score(self):
        """Test basic training and scoring."""
        X, y = self._make_signal_data(n=500, seed=42)

        scorer = SignalScorer(model_type="gradient_boosting", n_estimators=10)
        results = scorer.train(X, y)

        assert "train_accuracy" in results
        assert "test_accuracy" in results
        assert 0 <= results["test_accuracy"] <= 1

        # Test scoring
        scored = scorer.score(X)
        assert "ml_score" in scored.columns
        assert "is_recommended" in scored.columns
        assert scored["ml_score"].between(0, 1).all()

    def test_walk_forward_validation(self):
        """Test walk-forward validation."""
        X, y = self._make_signal_data(n=1500, seed=42)

        scorer = SignalScorer(model_type="random_forest", n_estimators=10)
        results = scorer.walk_forward_validation(X, y, train_size=300, step_size=100)

        assert "train_accuracy" in results
        assert "test_accuracy" in results
        assert len(results["train_accuracy"]) > 0

    def test_unsupported_model_raises(self):
        """Test unsupported model type."""
        with pytest.raises(ValueError):
            SignalScorer(model_type="neural_network")

    def test_top_features(self):
        """Test feature importance retrieval."""
        X, y = self._make_signal_data(n=500, seed=42)

        scorer = SignalScorer(model_type="random_forest", n_estimators=10)
        scorer.train(X, y)

        importance = scorer.get_top_features_by_importance(top_n=5)

        assert len(importance) > 0
        assert "feature" in importance.columns
        assert "importance" in importance.columns


class TestMLPipeline:
    """Tests for MLPipeline."""

    def _make_signal_data(self, n: int = 500, seed: int = 42) -> tuple:
        np.random.seed(seed)
        X = pd.DataFrame(
            {
                "rsi": np.random.uniform(0, 100, n),
                "macd_histogram": np.random.randn(n) * 0.5,
                "atr_pct": np.abs(np.random.randn(n) * 2),
                "volume_ratio": np.abs(np.random.randn(n) * 1.5 + 1),
            },
            index=pd.date_range("2022-01-01", periods=n, freq="B"),
        )
        y = pd.Series((X["rsi"] < 40).astype(int), name="profitable")
        noise = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
        y = ((y + noise) > 0).astype(int)
        return X, y

    def _make_regime_labels(self, n: int, index: pd.Index) -> pd.Series:
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

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        eng = FeatureEngineer()
        features = eng.generate_features(df)
        features = features.select_dtypes(include=[np.number])
        features = features.dropna(axis=1, how="all").ffill().bfill()
        valid = features.notna().all(axis=1)
        return features[valid].reset_index(drop=True)

    def test_pipeline_not_fitted_raises(self):
        df = _make_ohlcv_data(n=500, seed=42)
        pipeline = MLPipeline()
        with pytest.raises(ValueError):
            pipeline.predict_regime(df)
        with pytest.raises(ValueError):
            pipeline.score_signals(pd.DataFrame({"a": [1]}))

    def test_pipeline_fit_and_predict(self):
        df = _make_ohlcv_data(n=1500, seed=42)
        features = self._prepare_features(df)
        sig_features, sig_labels = self._make_signal_data(n=len(features), seed=42)

        # Create regime labels with the same DatetimeIndex as df
        regime_labels = pd.Series(
            np.random.choice(
                ["Trending", "Ranging", "Volatile", "Transition"],
                size=len(df),
                p=[0.3, 0.3, 0.2, 0.2],
            ),
            index=df.index,
            name="regime",
        )

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
        assert "signal" in results

        regime_pred = pipeline.predict_regime(df)
        assert len(regime_pred) == len(df)

        scored = pipeline.score_signals(sig_features)
        assert "ml_score" in scored.columns
        assert len(scored) == len(sig_features)

    def test_pipeline_feature_importance(self):
        df = _make_ohlcv_data(n=1500, seed=42)
        features = self._prepare_features(df)
        sig_features, sig_labels = self._make_signal_data(n=len(features), seed=42)

        regime_labels = pd.Series(
            np.random.choice(
                ["Trending", "Ranging", "Volatile", "Transition"],
                size=len(df),
                p=[0.3, 0.3, 0.2, 0.2],
            ),
            index=df.index,
            name="regime",
        )

        pipeline = MLPipeline(
            regime_model_type="random_forest",
            signal_model_type="random_forest",
            n_estimators=5,
            max_depth=3,
        )
        pipeline.fit(
            df, regime_labels=regime_labels, signal_features=sig_features, signal_labels=sig_labels
        )

        regime_imp = pipeline.get_regime_feature_importance(top_n=5)
        signal_imp = pipeline.get_signal_feature_importance(top_n=5)

        assert len(regime_imp) > 0
        assert len(signal_imp) > 0
        assert "feature" in regime_imp.columns
        assert "feature" in signal_imp.columns

    def test_walk_forward_validation(self):
        df = _make_ohlcv_data(n=1500, seed=42)
        features = self._prepare_features(df)
        sig_features, sig_labels = self._make_signal_data(n=len(features), seed=42)

        regime_labels = pd.Series(
            np.random.choice(
                ["Trending", "Ranging", "Volatile", "Transition"],
                size=len(df),
                p=[0.3, 0.3, 0.2, 0.2],
            ),
            index=df.index,
            name="regime",
        )

        pipeline = MLPipeline(
            regime_model_type="random_forest",
            signal_model_type="random_forest",
            n_estimators=5,
            max_depth=3,
        )

        results = pipeline.walk_forward_validation(
            df=df,
            regime_labels=regime_labels,
            signal_features=sig_features,
            signal_labels=sig_labels,
        )

        assert "regime" in results
        assert len(results["regime"]["train_score"]) > 0

    def test_run_pipeline_integration(self):
        df = _make_ohlcv_data(n=1000, seed=42)
        features = self._prepare_features(df)
        sig_features, sig_labels = self._make_signal_data(n=len(features), seed=42)

        regime_labels = pd.Series(
            np.random.choice(
                ["Trending", "Ranging", "Volatile", "Transition"],
                size=len(df),
                p=[0.3, 0.3, 0.2, 0.2],
            ),
            index=df.index,
            name="regime",
        )

        pipeline = MLPipeline(
            regime_model_type="random_forest",
            signal_model_type="gradient_boosting",
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
        assert result.feature_importance_signals is not None
        assert "regime_accuracy" in result.pipeline_metrics
