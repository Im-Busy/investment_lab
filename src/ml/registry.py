"""
Model Registry — model version management and artifact tracking.

Provides ModelRegistry for registering, versioning, loading, and listing
ML models with their metrics and metadata.

Storage structure:
    models/
    ├── registry.json            # Metadata for all models
    ├── regime_rf_v1.0.pkl        # Serialized model artifact
    └── signal_gb_v1.0.pkl

Usage:
    from src.ml.registry import ModelRegistry

    registry = ModelRegistry()
    registry.register(model, name="regime_rf", metrics={"ic": 0.08}, tags={"task": "classification"})
    model = registry.load("regime_rf", version="latest")
    history = registry.get_versions("regime_rf")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


class ModelRegistry:
    """Registry for ML model versions with metadata and artifact tracking.

    Manages model lifecycle: registration, versioning, loading,
    and listing. Uses JSON metadata file + joblib artifacts.

    Attributes:
        models_dir: Base directory for model storage.
        registry_path: Path to registry.json metadata file.
    """

    def __init__(self, models_dir: Path | str = MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.models_dir / "registry.json"
        self._registry: dict[str, dict[str, Any]] = self._load()

    def register(
        self,
        name: str,
        model: Any,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        version: str | None = None,
    ) -> str:
        """Register a trained model with version tracking.

        Args:
            name: Model identifier (e.g. 'regime_rf', 'signal_gb').
            model: Trained sklearn-compatible model object.
            metrics: Performance metrics dict (e.g. IC, rank IC, accuracy).
            metadata: Additional metadata (e.g. task, feature count, run_id).
            version: Explicit version string. If None, auto-increments.

        Returns:
            Version string assigned to this registration.
        """
        versions = self._registry.get(name, {"versions": []})["versions"]

        if version is None:
            version = self._next_version(versions)

        artifact_path = self.models_dir / f"{name}_v{version}.pkl"
        joblib.dump(model, artifact_path)

        entry = {
            "version": version,
            "artifact": str(artifact_path),
            "metrics": metrics or {},
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
        }
        versions.append(entry)
        self._registry[name] = {
            "versions": versions,
            "latest": version,
            "updated_at": datetime.now().isoformat(),
        }
        self._save()
        logger.info(f"Registered model '{name}' v{version} at {artifact_path}")
        return version

    def load(self, name: str, version: str = "latest") -> Any:
        """Load a registered model by name and version.

        Args:
            name: Model identifier.
            version: Version string or "latest".

        Returns:
            Loaded model object (joblib-deserialized).

        Raises:
            KeyError: If model name or version not found.
            FileNotFoundError: If artifact file is missing.
        """
        if name not in self._registry:
            raise KeyError(f"Model '{name}' not found in registry")

        reg = self._registry[name]
        if version == "latest":
            version = reg.get("latest", reg["versions"][-1]["version"])

        entry = next((v for v in reg["versions"] if v["version"] == version), None)
        if entry is None:
            raise KeyError(f"Version '{version}' not found for model '{name}'")

        artifact = Path(entry["artifact"])
        if not artifact.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact}")

        logger.info(f"Loaded model '{name}' v{version}")
        return joblib.load(artifact)

    def list_models(self) -> list[dict[str, Any]]:
        """List all registered models with their latest versions.

        Returns:
            List of dicts with keys: name, latest, version_count, metrics, registered_at.
        """
        results = []
        for name, reg in self._registry.items():
            versions = reg["versions"]
            latest_entry = next((v for v in versions if v["version"] == reg.get("latest")), None)
            results.append(
                {
                    "name": name,
                    "latest": reg["latest"],
                    "version_count": len(versions),
                    "metrics": latest_entry.get("metrics", {}) if latest_entry else {},
                    "metadata": latest_entry.get("metadata", {}) if latest_entry else {},
                    "registered_at": latest_entry.get("registered_at", "") if latest_entry else "",
                }
            )
        return results

    def get_versions(self, name: str) -> list[str]:
        """Get all registered version strings for a model.

        Args:
            name: Model identifier.

        Returns:
            Sorted list of version strings.

        Raises:
            KeyError: If model not found.
        """
        if name not in self._registry:
            raise KeyError(f"Model '{name}' not found")
        return [v["version"] for v in self._registry[name]["versions"]]

    def get_history(self, name: str) -> list[dict[str, Any]]:
        """Get full version history with metrics for a model.

        Args:
            name: Model identifier.

        Returns:
            List of version entries with metrics, metadata, and timestamps.
        """
        if name not in self._registry:
            raise KeyError(f"Model '{name}' not found")
        return list(self._registry[name]["versions"])

    def _next_version(self, versions: list[dict]) -> str:
        """Auto-increment version number."""
        if not versions:
            return "1.0"
        last = versions[-1]["version"]
        parts = last.split(".")
        try:
            major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            return f"{last}.1"
        return f"{major}.{minor + 1}"

    def _load(self) -> dict[str, Any]:
        """Load registry from JSON file."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        """Save registry to JSON file."""
        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)
