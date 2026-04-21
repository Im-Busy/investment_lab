"""Unit tests for ExperimentLogger."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.experiment_logger import ExperimentLogger


@pytest.fixture
def logger(tmp_path):
    """Create an ExperimentLogger instance with a temp directory."""
    return ExperimentLogger(
        run_id="test_run_001",
        base_dir=tmp_path,
        description="Test experiment",
        model_type="random_forest",
        task="regime_classification",
    )


class TestExperimentLogger:
    """Tests for ExperimentLogger."""

    def test_run_dir_created_on_log(self, logger, tmp_path):
        """Test that run directory is created when logging."""
        expected = tmp_path / "test_run_001"
        assert not expected.exists()

        logger.log_metadata(n_samples=100, n_features=10)
        assert expected.exists()
        assert (expected / "metadata.json").exists()

    def test_metadata_logged(self, logger):
        """Test metadata is correctly saved and queryable."""
        logger.log_metadata(
            description="RF regime test",
            model_type="random_forest",
            task="regime_classification",
            tickers=["SPY", "QQQ"],
            data_start="2020-01-01",
            data_end="2024-12-31",
            n_samples=1000,
            n_features=25,
            random_state=42,
            author="tester",
        )
        meta_path = logger.run_dir / "metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["run_id"] == "test_run_001"
        assert meta["model_type"] == "random_forest"
        assert meta["tickers"] == ["SPY", "QQQ"]
        assert meta["n_samples"] == 1000

    def test_config_logged(self, logger):
        """Test hyperparams and config are saved."""
        logger.log_config(
            hyperparams={"n_estimators": 100, "max_depth": 5},
            features=["rsi", "macd", "atr"],
            cv_params={"n_splits": 5, "embargo_days": 10},
        )
        config_path = logger.run_dir / "config.json"
        assert config_path.exists()
        with open(config_path) as f:
            config = json.load(f)
        assert config["hyperparams"]["n_estimators"] == 100
        assert config["features"] == ["rsi", "macd", "atr"]
        assert config["cv_params"]["embargo_days"] == 10

    def test_data_hash_computed(self, logger):
        """Test SHA256 hash of data is computed and saved."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        sha = logger.log_data_hash(data)
        assert len(sha) == 64  # SHA256 hex length

        hash_path = logger.run_dir / "data_hash.json"
        assert hash_path.exists()
        with open(hash_path) as f:
            saved = json.load(f)
        assert saved["sha256"] == sha

    def test_data_hash_dataframe(self, logger):
        """Test hash computation works with DataFrame input."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        sha = logger.log_data_hash(df)
        assert len(sha) == 64

    def test_fold_metrics_appended(self, logger):
        """Test per-fold metrics are appended to JSONL."""
        logger.log_fold_metrics(
            fold=1,
            train_metrics={"accuracy": 0.78, "spearman_ic": 0.12},
            test_metrics={"accuracy": 0.52, "spearman_ic": 0.03},
            train_start="2020-01-01",
            train_end="2022-06-30",
            test_start="2022-07-01",
            test_end="2022-12-31",
            n_train=500,
            n_test=120,
        )
        logger.log_fold_metrics(
            fold=2,
            train_metrics={"accuracy": 0.76, "spearman_ic": 0.10},
            test_metrics={"accuracy": 0.54, "spearman_ic": 0.04},
            n_train=520,
            n_test=100,
        )

        filepath = logger.run_dir / "fold_metrics.jsonl"
        assert filepath.exists()
        records = logger.load_fold_metrics()
        assert len(records) == 2
        assert records[0]["fold"] == 1
        assert records[1]["fold"] == 2
        assert records[0]["overfit_gap"]["accuracy"] == 0.26
        assert records[0]["overfit_gap"]["spearman_ic"] == 0.09

    def test_feature_importance_logged(self, logger):
        """Test feature importance from multiple methods saved."""
        logger.log_feature_importance(
            mdi={"rsi": 0.15, "macd": 0.12, "atr": 0.08},
            permutation={"atr": 0.05, "rsi": 0.03, "macd": 0.02},
            shap_top=[{"feature": "rsi", "mean_abs_shap": 0.18}],
        )
        imp_path = logger.run_dir / "feature_importance.json"
        assert imp_path.exists()
        with open(imp_path) as f:
            imp = json.load(f)
        assert imp["mdi"]["rsi"] == 0.15
        assert imp["permutation"]["atr"] == 0.05
        assert len(imp["shap_top10"]) == 1

    def test_predictions_logged(self, logger):
        """Test predictions are saved to CSV."""
        df = pd.DataFrame(
            {
                "timestamp": ["2024-01-15", "2024-01-16"],
                "ticker": ["SPY", "QQQ"],
                "pred": ["Trending", "Ranging"],
                "actual": ["Trending", "Volatile"],
            }
        )
        logger.log_predictions(df)
        csv_path = logger.run_dir / "predictions.csv"
        assert csv_path.exists()

        loaded = logger.load_predictions()
        assert len(loaded) == 2
        assert loaded.iloc[0]["ticker"] == "SPY"

    def test_predictions_empty_when_not_logged(self, logger):
        """Test load_predictions returns empty DF when no predictions logged."""
        df = logger.load_predictions()
        assert df.empty

    def test_model_saved(self, logger):
        """Test model is serialized and can be reloaded."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=2, random_state=42)
        path = logger.log_model(model)
        assert path.exists()
        assert logger._model_saved is True

        loaded = logger.load_model()
        assert isinstance(loaded, RandomForestClassifier)

    def test_model_not_found_raises(self, logger):
        """Test loading a non-existent model raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            logger.load_model()

    def test_summary_verdict_weak_signal(self, logger):
        """Test verdict is WEAK SIGNAL when OOS IC < 0.05."""
        logger.log_summary_verdict(
            mean_oos_metrics={
                "spearman_ic": 0.028,
                "accuracy": 0.51,
                "overfit_gap": {"accuracy": 0.10},
            },
            n_folds=5,
        )
        summary_path = logger.run_dir / "summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["verdict"] == "WEAK SIGNAL"
        assert "production" in summary["recommendation"].lower()

    def test_summary_verdict_overfit(self, logger):
        """Test verdict is OVERFIT when gap > 0.20."""
        logger.log_summary_verdict(
            mean_oos_metrics={
                "spearman_ic": 0.06,
                "accuracy": 0.55,
                "overfit_gap": {"accuracy": 0.28},
            },
            n_folds=5,
        )
        with open(logger.run_dir / "summary.json") as f:
            summary = json.load(f)
        assert summary["verdict"] == "OVERFIT"

    def test_summary_verdict_viable(self, logger):
        """Test verdict is VIABLE when metrics pass thresholds."""
        logger.log_summary_verdict(
            mean_oos_metrics={
                "spearman_ic": 0.08,
                "accuracy": 0.58,
                "overfit_gap": {"accuracy": 0.12},
            },
            n_folds=10,
            best_fold_metrics={"spearman_ic": 0.12},
            worst_fold_metrics={"spearman_ic": 0.04},
        )
        with open(logger.run_dir / "summary.json") as f:
            summary = json.load(f)
        assert summary["verdict"] == "VIABLE"

    def test_get_run_dir(self, logger):
        """Test run_dir path is correct."""
        expected = logger.base_dir / "test_run_001"
        assert logger.get_run_dir() == expected

    def test_auto_generated_run_id(self, tmp_path):
        """Test auto-generated run ID when not provided."""
        logger = ExperimentLogger(base_dir=tmp_path)
        assert logger.run_id.startswith("run_")
        assert len(logger.run_id) > 10

    def test_fold_metrics_empty_when_not_logged(self, logger):
        """Test load_fold_metrics returns empty list when nothing logged."""
        assert logger.load_fold_metrics() == []
