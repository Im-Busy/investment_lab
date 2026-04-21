"""
Unit tests for ModelRegistry."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.ml.registry import ModelRegistry


@pytest.fixture
def tmp_registry(tmp_path):
    """Create a ModelRegistry backed by a temp directory."""
    return ModelRegistry(models_dir=tmp_path)


@pytest.fixture
def trained_model():
    """Small trained RandomForest for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    return model


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_first_version(self, tmp_registry, trained_model):
        """Test registering first version creates artifact and metadata."""
        ver = tmp_registry.register(
            "regime_rf", trained_model, metrics={"ic": 0.08}, metadata={"task": "regime"}
        )
        assert ver == "1.0"

        artifact = tmp_registry.models_dir / "regime_rf_v1.0.pkl"
        assert artifact.exists()

        reg_json = tmp_registry.registry_path
        assert reg_json.exists()
        with open(reg_json) as f:
            data = json.load(f)
        assert "regime_rf" in data
        assert data["regime_rf"]["latest"] == "1.0"
        assert len(data["regime_rf"]["versions"]) == 1

    def test_register_auto_increments(self, tmp_registry, trained_model):
        """Test that repeated registrations auto-increment version."""
        tmp_registry.register("signal_gb", trained_model)
        v2 = tmp_registry.register("signal_gb", trained_model)
        v3 = tmp_registry.register("signal_gb", trained_model)
        assert v2 == "1.1"
        assert v3 == "1.2"

    def test_register_explicit_version(self, tmp_registry, trained_model):
        """Test explicit version string."""
        ver = tmp_registry.register("custom", trained_model, version="2.3")
        assert ver == "2.3"
        artifact_path = tmp_registry.models_dir / "custom_v2.3.pkl"
        assert artifact_path.exists()

    def test_load_latest(self, tmp_registry, trained_model):
        """Test loading the latest version."""
        m1 = tmp_registry.register("load_test", trained_model)
        m2 = tmp_registry.register("load_test", trained_model)

        loaded = tmp_registry.load("load_test")
        assert loaded is not None
        assert isinstance(loaded, RandomForestClassifier)

    def test_load_specific_version(self, tmp_registry, trained_model):
        """Test loading a specific version."""
        tmp_registry.register("ver_test", trained_model)
        tmp_registry.register("ver_test", trained_model)

        loaded = tmp_registry.load("ver_test", version="1.0")
        assert loaded is not None

    def test_load_nonexistent_model_raises(self, tmp_registry):
        """Test loading unregistered model raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            tmp_registry.load("nonexistent")

    def test_load_nonexistent_version_raises(self, tmp_registry, trained_model):
        """Test loading missing version raises KeyError."""
        tmp_registry.register("ver_test", trained_model)
        with pytest.raises(KeyError, match="Version"):
            tmp_registry.load("ver_test", version="9.9")

    def test_list_models(self, tmp_registry, trained_model):
        """Test listing all registered models."""
        tmp_registry.register("model_a", trained_model, metrics={"ic": 0.05})
        tmp_registry.register(
            "model_b", trained_model, metrics={"ic": 0.10}, metadata={"task": "signal"}
        )

        models = tmp_registry.list_models()
        assert len(models) == 2
        names = {m["name"] for m in models}
        assert names == {"model_a", "model_b"}
        metrics = {m["name"]: m["metrics"] for m in models}
        assert metrics["model_a"]["ic"] == 0.05
        assert metrics["model_b"]["ic"] == 0.10

    def test_get_versions(self, tmp_registry, trained_model):
        """Test retrieving all versions for a model."""
        tmp_registry.register("ver_list", trained_model)
        tmp_registry.register("ver_list", trained_model)
        tmp_registry.register("ver_list", trained_model)

        versions = tmp_registry.get_versions("ver_list")
        assert versions == ["1.0", "1.1", "1.2"]

    def test_get_versions_nonexistent_raises(self, tmp_registry):
        """Test get_versions for unregistered model raises KeyError."""
        with pytest.raises(KeyError):
            tmp_registry.get_versions("ghost")

    def test_get_history(self, tmp_registry, trained_model):
        """Test full version history retrieval."""
        tmp_registry.register("hist", trained_model, metrics={"ic": 0.05, "version": "1.0"})
        tmp_registry.register("hist", trained_model, metrics={"ic": 0.07, "version": "1.1"})

        history = tmp_registry.get_history("hist")
        assert len(history) == 2
        assert history[0]["version"] == "1.0"
        assert history[1]["version"] == "1.1"
        assert history[1]["metrics"]["ic"] == 0.07

    def test_register_preserves_model_predictions(self, tmp_registry):
        """Test that loaded model produces identical predictions."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=3, random_state=42)
        model.fit(X, y)

        original_preds = model.predict(X[:5])
        tmp_registry.register("pred_test", model)

        loaded = tmp_registry.load("pred_test")
        loaded_preds = loaded.predict(X[:5])

        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_metadata_stored(self, tmp_registry, trained_model):
        """Test that metadata is correctly stored."""
        tmp_registry.register(
            "meta_test",
            trained_model,
            metadata={"run_id": "abc123", "features": ["rsi", "macd"]},
        )
        models = tmp_registry.list_models()
        assert models[0]["metadata"]["run_id"] == "abc123"
        assert models[0]["metadata"]["features"] == ["rsi", "macd"]

    def test_default_values(self, tmp_registry, trained_model):
        """Test registration with minimal arguments."""
        ver = tmp_registry.register("minimal", trained_model)
        assert ver == "1.0"

        loaded = tmp_registry.load("minimal")
        assert loaded is not None

        models = tmp_registry.list_models()
        assert models[0]["metrics"] == {}
        assert models[0]["metadata"] == {}
