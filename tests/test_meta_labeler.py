"""Unit tests for MetaLabeler (P1.1 / T9)."""

import numpy as np
import pandas as pd
import pytest

from src.ml.meta_labeler import MetaLabeler, MetaLabelResult, MetaLabelPrediction


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data with DatetimeIndex."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    base = 100.0 + np.cumsum(np.random.randn(n) * 0.3)
    close = pd.Series(base + np.random.randn(n) * 0.1, index=dates)
    high = pd.Series(close.values + np.abs(np.random.randn(n)) * 0.2, index=dates)
    low = pd.Series(close.values - np.abs(np.random.randn(n)) * 0.2, index=dates)
    high = pd.Series(np.maximum(high.values, close.values), index=dates)
    low = pd.Series(np.minimum(low.values, close.values), index=dates)
    volume = pd.Series(np.random.randint(100000, 1000000, n), index=dates)
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


@pytest.fixture
def sample_signals(sample_ohlcv):
    """Generate synthetic historical signals."""
    np.random.seed(43)
    n_signals = 60
    idx = np.random.choice(range(100, 400), n_signals, replace=False)
    idx = sorted(idx)
    timestamps = sample_ohlcv.index[idx]
    entries = sample_ohlcv["Close"].iloc[idx].values

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "pattern_name": np.random.choice(
                [
                    "Double Bottom",
                    "Double Top",
                    "Head and Shoulders",
                    "Flag Continuation",
                    "Donchian Channel Breakout",
                    "Doji",
                ],
                n_signals,
            ),
            "direction": np.random.choice(["Long", "Short"], n_signals),
            "entry_price": entries,
            "stop_loss": entries * 0.98,
            "take_profit_1": entries * 1.04,
            "confidence": np.random.uniform(0.3, 0.9, n_signals),
        }
    )


class TestMetaLabelerInit:
    """Test MetaLabeler initialization."""

    def test_default_init(self):
        ml = MetaLabeler()
        assert ml.tp_atr_mult == 2.0
        assert ml.sl_atr_mult == 1.5
        assert ml.time_limit == 20
        assert ml.n_splits == 5
        assert ml.model is None
        assert ml.model_used is None
        assert ml.feature_names_ == []

    def test_custom_init(self):
        ml = MetaLabeler(tp_atr_mult=3.0, sl_atr_mult=2.0, time_limit=10, n_splits=3)
        assert ml.tp_atr_mult == 3.0
        assert ml.sl_atr_mult == 2.0
        assert ml.time_limit == 10
        assert ml.n_splits == 3


class TestMetaLabelerFit:
    """Test MetaLabeler training."""

    def test_fit_returns_result(self, sample_ohlcv, sample_signals):
        ml = MetaLabeler()
        result = ml.fit(sample_ohlcv, sample_signals)
        assert isinstance(result, MetaLabelResult)
        assert result.n_signals > 0
        assert 0 <= result.baseline_win_rate <= 1
        assert result.model_used in ("lgbm", "catboost")

    def test_fit_sets_feature_names(self, sample_ohlcv, sample_signals):
        ml = MetaLabeler()
        ml.fit(sample_ohlcv, sample_signals)
        assert len(ml.feature_names_) > 0

    def test_fit_trains_model(self, sample_ohlcv, sample_signals):
        ml = MetaLabeler()
        ml.fit(sample_ohlcv, sample_signals)
        assert ml.model is not None

    def test_fit_with_few_signals(self, sample_ohlcv):
        """Should handle sparse signals gracefully."""
        np.random.seed(99)
        n_sig = 5
        idx = sorted(np.random.choice(range(100, 400), n_sig, replace=False))
        timestamps = sample_ohlcv.index[idx]
        signals = pd.DataFrame(
            {
                "timestamp": timestamps,
                "pattern_name": ["Doji"] * n_sig,
                "direction": ["Long"] * n_sig,
                "entry_price": sample_ohlcv["Close"].iloc[idx].values,
                "stop_loss": sample_ohlcv["Close"].iloc[idx].values * 0.97,
                "take_profit_1": sample_ohlcv["Close"].iloc[idx].values * 1.05,
                "confidence": [0.5] * n_sig,
            }
        )
        ml = MetaLabeler()
        result = ml.fit(sample_ohlcv, signals)
        assert isinstance(result, MetaLabelResult)

    def test_fit_with_empty_signals(self, sample_ohlcv):
        """Empty signals should return result with none model."""
        empty = pd.DataFrame(
            columns=[
                "timestamp",
                "pattern_name",
                "direction",
                "entry_price",
                "stop_loss",
                "take_profit_1",
                "confidence",
            ]
        )
        ml = MetaLabeler()
        result = ml.fit(sample_ohlcv, empty)
        assert result.model_used == "none"
        assert result.n_signals == 0


class TestMetaLabelerPredict:
    """Test MetaLabeler prediction."""

    def test_predict_returns_meta_label(self, sample_ohlcv, sample_signals):
        ml = MetaLabeler()
        ml.fit(sample_ohlcv, sample_signals)

        if ml.model is None:
            pytest.skip("Model not trained")

        context = {name: 0.0 for name in ml.feature_names_}
        pred = ml.predict(context)
        assert isinstance(pred, MetaLabelPrediction)
        assert isinstance(pred.take_trade, bool)
        assert 0.0 <= pred.probability <= 1.0
        assert 0.0 <= pred.confidence <= 1.0

    def test_predict_batch(self, sample_ohlcv, sample_signals):
        ml = MetaLabeler()
        ml.fit(sample_ohlcv, sample_signals)

        if ml.model is None:
            pytest.skip("Model not trained")

        X = pd.DataFrame([{name: 0.0 for name in ml.feature_names_} for _ in range(5)])
        result = ml.predict_batch(X)
        assert len(result) == 5
        assert "meta_probability" in result.columns
        assert "meta_take_trade" in result.columns
        assert "meta_confidence" in result.columns

    def test_predict_raises_before_fit(self):
        ml = MetaLabeler()
        with pytest.raises(ValueError, match="not trained"):
            ml.predict({"test": 0.0})


class TestMetaLabelerSaveLoad:
    """Test model persistence."""

    def test_save_and_load(self, sample_ohlcv, sample_signals, tmp_path):
        ml = MetaLabeler()
        ml.fit(sample_ohlcv, sample_signals)

        if ml.model is None:
            pytest.skip("Model not trained")

        path = tmp_path / "meta_labeler.pkl"
        ml.save(path)
        assert path.exists()

        ml2 = MetaLabeler()
        ml2.load(path)
        assert ml2.model is not None
        assert ml2.feature_names_ == ml.feature_names_
        assert ml2.model_used == ml.model_used

    def test_load_nonexistent_raises(self):
        from pathlib import Path

        ml = MetaLabeler()
        with pytest.raises(FileNotFoundError):
            ml.load(Path("nonexistent.pkl"))


class TestMetaLabelerFeatureBuilding:
    """Test context feature construction."""

    def test_features_no_nan(self, sample_ohlcv, sample_signals):
        ml = MetaLabeler()
        labeled = ml._label_signals(sample_ohlcv, sample_signals)
        X, y = ml._build_features(sample_ohlcv, labeled)
        assert not X.isna().any().any()
        assert len(X) == len(y)

    def test_features_have_expected_columns(self, sample_ohlcv, sample_signals):
        ml = MetaLabeler()
        labeled = ml._label_signals(sample_ohlcv, sample_signals)
        X, y = ml._build_features(sample_ohlcv, labeled)
        expected = [
            "pattern_confidence",
            "atr_ratio",
            "rsi_value",
            "trend_20",
            "trend_50",
            "volatility_20",
            "volume_ratio",
            "relative_position",
            "day_of_week",
            "hour_of_day",
            "is_reversal",
            "is_breakout",
        ]
        for col in expected:
            assert col in X.columns, f"Missing column: {col}"

    def test_labels_are_binary(self, sample_ohlcv, sample_signals):
        ml = MetaLabeler()
        labeled = ml._label_signals(sample_ohlcv, sample_signals)
        X, y = ml._build_features(sample_ohlcv, labeled)
        assert set(y.unique()).issubset({0, 1})

    def test_is_reversal_flag(self, sample_ohlcv, sample_signals):
        ml = MetaLabeler()
        labeled = ml._label_signals(sample_ohlcv, sample_signals)
        X, y = ml._build_features(sample_ohlcv, labeled)
        assert X["is_reversal"].max() <= 1.0
        assert X["is_reversal"].min() >= 0.0
        assert X["is_breakout"].max() <= 1.0
