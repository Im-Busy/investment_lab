"""Unit tests for FeatureStore."""

import json

import numpy as np
import pandas as pd
import pytest

from src.ml.feature_store import FeatureStore


def _make_ohlcv(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Create synthetic OHLCV data."""
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
def tmp_store(tmp_path):
    """Create a FeatureStore with temp directory."""
    return FeatureStore(store_dir=tmp_path)


@pytest.fixture
def sample_data():
    """Sample OHLCV data for two tickers."""
    return {
        "SPY": _make_ohlcv(1500, seed=42),
        "QQQ": _make_ohlcv(1500, seed=99),
    }


class TestFeatureStore:
    """Tests for FeatureStore."""

    def test_build_creates_features(self, tmp_store, sample_data):
        """Test that build creates feature DataFrame."""
        features = tmp_store.build(sample_data)
        assert "ticker" in features.columns
        assert "date" in features.columns
        assert len(features) > 0

    def test_build_multiple_tickers(self, tmp_store, sample_data):
        """Test that features from multiple tickers are concatenated."""
        features = tmp_store.build(sample_data)
        tickers = features["ticker"].unique()
        assert set(tickers) == {"SPY", "QQQ"}

    def test_build_creates_labels(self, tmp_store, sample_data):
        """Test that forward return labels are computed."""
        tmp_store.build(sample_data, compute_labels=True, label_horizons=[1, 5])
        labels = tmp_store.get_labels(horizon=5)
        assert "forward_return_5d" in labels.columns
        assert len(labels) > 0

    def test_build_saves_files(self, tmp_store, sample_data):
        """Test that parquet and metadata files are saved."""
        tmp_store.build(sample_data, compute_labels=True)
        assert tmp_store.feature_path.exists()
        assert tmp_store.metadata_path.exists()
        assert tmp_store.labels_path.exists()

    def test_load_roundtrip(self, tmp_store, sample_data):
        """Test that saved features can be loaded back."""
        tmp_store.build(sample_data)
        features_before = tmp_store._features.copy()

        fresh_store = FeatureStore(store_dir=tmp_store.store_dir)
        assert not fresh_store.is_loaded()
        features_after = fresh_store.load()

        pd.testing.assert_frame_equal(features_before, features_after)

    def test_load_raises_if_not_exists(self, tmp_store):
        """Test loading from non-existent store raises error."""
        with pytest.raises(FileNotFoundError):
            tmp_store.load()

    def test_get_features_filter_tickers(self, tmp_store, sample_data):
        """Test filtering by tickers."""
        tmp_store.build(sample_data)
        spy = tmp_store.get_features(tickers=["SPY"])
        assert (spy["ticker"] == "SPY").all()
        assert "QQQ" not in set(spy["ticker"])

    def test_get_features_filter_date_range(self, tmp_store, sample_data):
        """Test filtering by date range."""
        tmp_store.build(sample_data)
        filtered = tmp_store.get_features(start="2020-06-01", end="2020-09-01")
        for date in filtered["date"]:
            assert date >= "2020-06-01"
            assert date <= "2020-09-01"

    def test_get_features_exclude_columns(self, tmp_store, sample_data):
        """Test excluding specific feature columns."""
        tmp_store.build(sample_data)
        feature_names = tmp_store.get_feature_names()
        if feature_names:
            excluded = tmp_store.get_features(exclude_features=[feature_names[0]])
            assert feature_names[0] not in excluded.columns

    def test_get_labels_wrong_horizon(self, tmp_store, sample_data):
        """Test getting labels with non-existent horizon raises error."""
        tmp_store.build(sample_data, label_horizons=[5])
        with pytest.raises(ValueError, match="Horizon 50d not found"):
            tmp_store.get_labels(horizon=50)

    def test_get_labels_no_data(self, tmp_store):
        """Test getting labels before build raises error."""
        with pytest.raises(ValueError, match="Labels not computed"):
            tmp_store.get_labels(horizon=5)

    def test_get_feature_names(self, tmp_store, sample_data):
        """Test feature name retrieval."""
        tmp_store.build(sample_data)
        names = tmp_store.get_feature_names()
        assert len(names) > 20
        assert "ticker" not in names
        assert "date" not in names

    def test_get_tickers(self, tmp_store, sample_data):
        """Test ticker list retrieval."""
        tmp_store.build(sample_data)
        tickers = tmp_store.get_tickers()
        assert tickers == ["QQQ", "SPY"]

    def test_is_loaded(self, tmp_store, sample_data):
        """Test loaded state flag."""
        assert not tmp_store.is_loaded()
        tmp_store.build(sample_data)
        assert tmp_store.is_loaded()

    def test_metadata_saved(self, tmp_store, sample_data):
        """Test metadata content is correct."""
        tmp_store.build(sample_data, label_horizons=[1, 5])
        with open(tmp_store.metadata_path) as f:
            meta = json.load(f)
        assert meta["n_tickers"] == 2
        assert meta["n_features"] > 20
        assert meta["label_horizons"] == [1, 5]

    def test_build_empty_data_raises(self, tmp_store):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError):
            tmp_store.build({})

    def test_performance(self, tmp_store):
        """Test that loading is fast (<5 seconds for reasonable data)."""
        import time

        large_data = {f"TKR{i}": _make_ohlcv(500, seed=i) for i in range(10)}

        t0 = time.time()
        tmp_store.build(large_data)
        build_time = time.time() - t0

        fresh = FeatureStore(store_dir=tmp_store.store_dir)
        t0 = time.time()
        fresh.load()
        load_time = time.time() - t0

        assert load_time < 5, f"Load took {load_time:.2f}s"
        assert len(fresh.get_tickers()) == 10
