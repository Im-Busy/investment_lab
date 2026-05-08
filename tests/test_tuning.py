"""Tests for metaheuristic tuning modules (ARO + GWO)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.tuning.base import (
    BaseOptimizer,
    OptimizerResult,
    ParamSpec,
    SearchSpace,
)
from src.ml.tuning.aro_selector import AROFeatureSelector
from src.ml.tuning.gwo_tuner import CATBOOST_PARAM_SPACE, GWOTuner


class TestParamSpec:
    def test_float_param(self) -> None:
        spec = ParamSpec("lr", "float", 0.01, 0.3, log_scale=True)
        rng = np.random.default_rng(42)
        val = spec.sample(rng)
        assert 0.01 <= val <= 0.3
        assert isinstance(val, float)

    def test_int_param(self) -> None:
        spec = ParamSpec("depth", "int", 3, 10)
        rng = np.random.default_rng(42)
        val = spec.sample(rng)
        assert 3 <= val <= 10
        assert isinstance(val, (int, np.integer))

    def test_choice_param(self) -> None:
        spec = ParamSpec("model", "choice", choices=["catboost", "lightgbm"])
        rng = np.random.default_rng(42)
        val = spec.sample(rng)
        assert val in ["catboost", "lightgbm"]

    def test_clip_float(self) -> None:
        spec = ParamSpec("lr", "float", 0.01, 0.3)
        assert spec.clip(5.0) == 0.3
        assert spec.clip(-1.0) == 0.01
        assert spec.clip(0.1) == 0.1

    def test_clip_int(self) -> None:
        spec = ParamSpec("depth", "int", 3, 10)
        assert spec.clip(20) == 10
        assert spec.clip(0) == 3
        assert spec.clip(5) == 5

    def test_choice_requires_choices(self) -> None:
        with pytest.raises(ValueError, match="requires choices"):
            ParamSpec("bad", "choice")

    def test_float_requires_bounds(self) -> None:
        with pytest.raises(ValueError, match="requires low and high"):
            ParamSpec("bad", "float")


class TestSearchSpace:
    def test_default_space(self) -> None:
        space = SearchSpace(CATBOOST_PARAM_SPACE)
        assert space.n_dims == 7

    def test_random_position(self) -> None:
        space = SearchSpace(CATBOOST_PARAM_SPACE)
        pos = space.random_position()
        assert len(pos) == 7
        # All valid
        assert 0.01 <= pos[0] <= 0.3
        assert 3 <= pos[1] <= 10

    def test_clip_position(self) -> None:
        space = SearchSpace(CATBOOST_PARAM_SPACE)
        bad = np.array([10.0, 100, 50.0, 10.0, 5.0, 500, 100])
        clipped = space.clip_position(bad)
        assert clipped[0] == 0.3
        assert clipped[1] == 10

    def test_to_dict(self) -> None:
        space = SearchSpace(CATBOOST_PARAM_SPACE)
        pos = space.random_position()
        d = space.to_dict(pos)
        assert set(d.keys()) == {
            "learning_rate",
            "depth",
            "l2_leaf_reg",
            "random_strength",
            "bagging_temperature",
            "border_count",
            "min_data_in_leaf",
        }
        assert isinstance(d["depth"], (int, np.integer))


class TestOptimizerResult:
    def test_basic_result(self) -> None:
        result = OptimizerResult(
            best_params={"lr": 0.1},
            best_score=0.85,
            all_scores=[0.80, 0.82, 0.85],
            optimizer_name="test",
        )
        assert result.best_score == 0.85
        assert result.mean_score == pytest.approx(0.823, abs=0.01)
        assert result.score_std > 0


class MockGWOTuner(BaseOptimizer):
    """Concrete implementation of BaseOptimizer for testing."""

    def optimize(self, max_iter: int = 50, early_stop: int = 10) -> OptimizerResult:
        pop = self._initialize_population()
        best_score = -np.inf
        history = []
        for _ in range(10):
            for i in range(len(pop)):
                pop[i] += self._rng.uniform(-0.1, 0.1, self.search_space.n_dims)
                pop[i] = self.search_space.clip_position(pop[i])
            scores = [self._evaluate(p) for p in pop]
            best = max(scores)
            if best > best_score:
                best_score = best
            history.append(best_score)

        best_idx = np.argmax(scores)
        return OptimizerResult(
            best_params=self.search_space.to_dict(pop[best_idx]),
            best_score=best_score,
            all_scores=scores,
            n_iterations=10,
            convergence_history=history,
            optimizer_name="MockGWO",
        )


class TestBaseOptimizer:
    def test_initialize_population(self) -> None:
        space = SearchSpace(CATBOOST_PARAM_SPACE[:3])
        counter = {"count": 0}

        def fitness(params: dict) -> float:
            counter["count"] += 1
            return 0.5

        tuner = MockGWOTuner(space, fitness, population_size=10)
        result = tuner.optimize(max_iter=10)

        assert result.n_iterations == 10
        assert counter["count"] > 0
        assert len(result.convergence_history) == 10

    def test_maximize(self) -> None:
        space = SearchSpace([ParamSpec("x", "float", 0.0, 1.0)])

        def fitness(params: dict) -> float:
            return -params["x"]

        tuner = MockGWOTuner(space, fitness, population_size=5, maximize=True)
        result = tuner.optimize(max_iter=5)
        assert result.best_score <= 0.0

    def test_minimize(self) -> None:
        space = SearchSpace([ParamSpec("x", "float", 0.0, 1.0)])

        def fitness(params: dict) -> float:
            return params["x"]

        tuner = MockGWOTuner(space, fitness, population_size=5, maximize=False)
        result = tuner.optimize(max_iter=5)
        assert result.best_score >= 0.0


class TestGWOTuner:
    def test_optimize_quadratic(self) -> None:
        """GWO should find the minimum of (x-0.5)^2."""
        space = SearchSpace(
            [
                ParamSpec("x", "float", 0.0, 1.0),
                ParamSpec("y", "float", 0.0, 1.0),
            ]
        )

        def fitness(params: dict) -> float:
            return -((params["x"] - 0.5) ** 2 + (params["y"] - 0.5) ** 2)

        tuner = GWOTuner(space, fitness, n_wolves=15, maximize=True, seed=42)
        result = tuner.optimize(max_iter=30, early_stop=10)

        assert result.best_score >= -0.05, f"GWO didn't converge: {result.best_score}"
        assert abs(result.best_params["x"] - 0.5) < 0.3, f"x={result.best_params['x']}"
        assert abs(result.best_params["y"] - 0.5) < 0.3, f"y={result.best_params['y']}"

    def test_output_structure(self) -> None:
        space = SearchSpace(CATBOOST_PARAM_SPACE[:3])

        def fitness(params: dict) -> float:
            return 0.8

        tuner = GWOTuner(space, fitness, n_wolves=5, maximize=True, seed=42)
        result = tuner.optimize(max_iter=5)

        assert isinstance(result.best_params, dict)
        assert len(result.best_params) == 3
        assert result.best_score > 0
        assert result.n_iterations > 0
        assert len(result.convergence_history) > 0

    def test_convergence_monotonic(self) -> None:
        """Best score should never decrease (maximize mode)."""
        space = SearchSpace(
            [
                ParamSpec("x", "float", 0.0, 1.0),
            ]
        )

        def fitness(params: dict) -> float:
            return params["x"] + np.random.random() * 0.01

        tuner = GWOTuner(space, fitness, n_wolves=10, maximize=True, seed=123)
        result = tuner.optimize(max_iter=20, early_stop=20)

        for i in range(1, len(result.convergence_history)):
            assert result.convergence_history[i] >= result.convergence_history[i - 1] - 0.1


class TestAROFeatureSelector:
    def test_initialization(self) -> None:
        feature_names = [f"feat_{i}" for i in range(20)]
        X = pd.DataFrame(np.random.randn(100, 20), columns=feature_names)
        y = pd.Series(np.random.randint(0, 2, 100))

        from src.ml.pattern_classifier import PatternClassifier

        selector = AROFeatureSelector(
            feature_names=feature_names,
            X=X,
            y=y,
            model_class=PatternClassifier,
            n_rabbits=5,
            target_n_features=10,
        )
        assert selector.n_rabbits == 5

    def test_select_finds_features(self) -> None:
        """ARO should converge on a feature subset."""
        feature_names = [f"feat_{i}" for i in range(15)]
        X = pd.DataFrame(np.random.randn(200, 15), columns=feature_names)
        y = pd.Series(np.random.randint(0, 2, 200))

        from src.ml.pattern_classifier import PatternClassifier

        selector = AROFeatureSelector(
            feature_names=feature_names,
            X=X,
            y=y,
            model_class=PatternClassifier,
            n_rabbits=10,
            target_n_features=8,
            cv_splits=2,
            seed=42,
        )
        result = selector.select(max_iter=20, early_stop=10)

        assert isinstance(result.best_params["selected_features"], list)
        assert 3 <= len(result.best_params["selected_features"]) <= 8
        assert result.best_score >= 0.0
        assert result.n_iterations > 0


class TestIntegration:
    """Integration test: tune PatternClassifier on synthetic data."""

    def test_tune_on_synthetic_data(self) -> None:
        """Verify GWO runs end-to-end on synthetic OHLCV data."""
        from src.ml.pattern_classifier import PatternClassifier

        n_samples = 600
        dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
        df = pd.DataFrame(
            {
                "Open": 100 + np.random.randn(n_samples).cumsum() * 0.5,
                "High": np.zeros(n_samples),
                "Low": np.zeros(n_samples),
                "Close": np.zeros(n_samples),
                "Volume": np.random.randint(1000000, 5000000, n_samples),
            },
            index=dates,
        )
        df["High"] = df["Open"] + np.abs(np.random.randn(n_samples))
        df["Low"] = df["Open"] - np.abs(np.random.randn(n_samples))
        df["Close"] = (df["High"] + df["Low"]) / 2 + np.random.randn(n_samples) * 0.2

        from src.ml.feature_engineering import FeatureExtractor

        extractor = FeatureExtractor()
        features = extractor.extract_all_features(df)

        future_returns = df["Close"].shift(-5) / df["Close"] - 1
        y = (future_returns > 0).astype(int)
        y = y.dropna()
        features = features.loc[y.index].dropna()
        y = y.loc[features.index]

        X = features

        # Use only a subset of params for speed
        space = SearchSpace(CATBOOST_PARAM_SPACE[:3])

        from src.ml.tuning.base import map_params

        def fitness(params: dict) -> float:
            mapped = map_params(params, "pattern_classifier")
            model = PatternClassifier(
                model_type="catboost",
                n_estimators=50,
                random_state=42,
                **mapped,
            )
            result = model.train(X, y)
            return result.test_auc

        tuner = GWOTuner(space, fitness, n_wolves=5, maximize=True, seed=42)
        result = tuner.optimize(max_iter=5, early_stop=10)

        assert result.best_score > 0.3, f"AUC too low: {result.best_score}"
        assert "learning_rate" in result.best_params
        assert "depth" in result.best_params
        assert "l2_leaf_reg" in result.best_params


class TestGARegimeOptimizer:
    def test_optimize_finds_regimes(self) -> None:
        """GA should find an optimal n_regimes via silhouette score."""
        from sklearn.datasets import make_blobs
        from src.ml.tuning.ga_tuner import GARegimeOptimizer, silhouette_fitness, REGIME_PARAM_SPACE

        X, _ = make_blobs(n_samples=200, n_features=5, centers=3, random_state=42)
        space = SearchSpace(REGIME_PARAM_SPACE)

        def fitness(params: dict) -> float:
            return silhouette_fitness(
                X,
                n_regimes=int(params["n_regimes"]),
                n_init=int(params["n_init"]),
                max_iter=int(params["max_iter"]),
                tol=float(params["tol"]),
            )

        ga = GARegimeOptimizer(
            space, fitness, population_size=10, crossover_rate=0.8, mutation_rate=0.15
        )
        result = ga.optimize(max_iter=10, early_stop=8)

        assert result.best_score > -1.0
        assert 2 <= result.best_params["n_regimes"] <= 8
        assert result.n_iterations > 0
        assert len(result.convergence_history) > 0

    def test_output_structure(self) -> None:
        """GA result should have the expected structure."""
        from sklearn.datasets import make_blobs
        from src.ml.tuning.ga_tuner import GARegimeOptimizer, silhouette_fitness, REGIME_PARAM_SPACE

        X, _ = make_blobs(n_samples=100, n_features=4, centers=3, random_state=42)
        space = SearchSpace(REGIME_PARAM_SPACE)

        def fitness(params: dict) -> float:
            return silhouette_fitness(
                X,
                n_regimes=int(params["n_regimes"]),
                n_init=int(params["n_init"]),
                max_iter=int(params["max_iter"]),
                tol=float(params["tol"]),
            )

        ga = GARegimeOptimizer(space, fitness, population_size=5)
        result = ga.optimize(max_iter=5)

        assert isinstance(result.best_params, dict)
        assert "n_regimes" in result.best_params
        assert "n_init" in result.best_params
        assert result.best_score >= -1.0
        assert result.n_iterations > 0

    def test_elite_preservation(self) -> None:
        """Best fitness should not degrade (maximize mode)."""
        from sklearn.datasets import make_blobs
        from src.ml.tuning.ga_tuner import GARegimeOptimizer, silhouette_fitness, REGIME_PARAM_SPACE

        X, _ = make_blobs(n_samples=100, n_features=3, centers=3, random_state=42)
        space = SearchSpace(REGIME_PARAM_SPACE)

        def fitness(params: dict) -> float:
            return silhouette_fitness(
                X,
                n_regimes=int(params["n_regimes"]),
                n_init=int(params["n_init"]),
                max_iter=int(params["max_iter"]),
                tol=float(params["tol"]),
            )

        ga = GARegimeOptimizer(space, fitness, population_size=8, crossover_rate=0.8)
        result = ga.optimize(max_iter=15, early_stop=15)

        for i in range(1, len(result.convergence_history)):
            assert result.convergence_history[i] >= result.convergence_history[i - 1] - 0.15


class TestReimDiscovery:
    def test_optimize_and_labels(self) -> None:
        """RegimeDiscovery should produce labels and centroids."""
        from sklearn.datasets import make_blobs
        from src.ml.tuning.ga_tuner import RegimeDiscovery

        X_arr, _ = make_blobs(n_samples=150, n_features=5, centers=3, random_state=42)
        X = pd.DataFrame(X_arr, columns=[f"feat_{i}" for i in range(5)])

        discovery = RegimeDiscovery(X)
        result = discovery.optimize(population_size=10, max_generations=10)

        labels = discovery.get_regime_labels()
        centroids = discovery.get_centroid_features()

        assert len(labels) == len(X)
        assert len(np.unique(labels)) >= 2
        assert centroids.shape[0] == discovery.n_regimes_
        assert centroids.shape[1] == 5
        assert discovery.n_regimes_ >= 2
        assert result.best_score > -1.0


class TestWOATuner:
    def test_optimize_quadratic(self) -> None:
        """WOA should find the minimum of (x-0.5)^2."""
        from src.ml.tuning.woa_tuner import WOATuner

        space = SearchSpace(
            [
                ParamSpec("x", "float", 0.0, 1.0),
                ParamSpec("y", "float", 0.0, 1.0),
            ]
        )

        def fitness(params: dict) -> float:
            return -((params["x"] - 0.5) ** 2 + (params["y"] - 0.5) ** 2)

        tuner = WOATuner(space, fitness, n_whales=15, maximize=True, seed=42)
        result = tuner.optimize(max_iter=30, early_stop=10)

        assert result.best_score >= -0.08, f"WOA didn't converge: {result.best_score}"
        assert abs(result.best_params["x"] - 0.5) < 0.3
        assert abs(result.best_params["y"] - 0.5) < 0.3

    def test_output_structure(self) -> None:
        """WOA result should have expected structure."""
        from src.ml.tuning.woa_tuner import WOATuner

        space = SearchSpace(
            [
                ParamSpec("a", "float", 0.0, 1.0),
                ParamSpec("b", "float", 0.0, 1.0),
            ]
        )

        def fitness(params: dict) -> float:
            return params["a"] + params["b"]

        tuner = WOATuner(space, fitness, n_whales=5, maximize=True, seed=42)
        result = tuner.optimize(max_iter=5)

        assert isinstance(result.best_params, dict)
        assert len(result.best_params) == 2
        assert result.best_score >= 0
        assert result.n_iterations > 0

    def test_convergence_monotonic(self) -> None:
        """Best score should never decrease (maximize mode)."""
        from src.ml.tuning.woa_tuner import WOATuner

        space = SearchSpace(
            [
                ParamSpec("x", "float", 0.0, 1.0),
            ]
        )

        def fitness(params: dict) -> float:
            return params["x"] + np.random.random() * 0.01

        tuner = WOATuner(space, fitness, n_whales=10, maximize=True, seed=123)
        result = tuner.optimize(max_iter=20, early_stop=20)

        for i in range(1, len(result.convergence_history)):
            assert result.convergence_history[i] >= result.convergence_history[i - 1] - 0.1

    def test_build_threshold_search_space(self) -> None:
        """build_threshold_search_space should create correct params."""
        from src.ml.tuning.woa_tuner import build_threshold_search_space

        categories = ["bullish", "bearish", "neutral"]
        space = build_threshold_search_space(categories)

        assert space.n_dims == 3
        assert space.params[0].name == "bullish"
        assert space.params[1].name == "bearish"
        assert space.params[2].name == "neutral"
        for p in space.params:
            assert p.low == 0.10
            assert p.high == 0.95
            assert p.type == "float"


class TestPatternScorer:
    def test_train_and_score(self) -> None:
        """PatternScorer should train and produce scores."""
        from src.ml.pattern_scorer import PatternScorer

        X = pd.DataFrame(
            np.random.randn(500, 8),
            columns=[f"feat_{i}" for i in range(8)],
        )
        y = pd.Series(np.random.randint(0, 2, 500))

        scorer = PatternScorer(n_estimators=30, max_depth=3)
        result = scorer.train(X, y)

        assert scorer.is_trained
        assert "train_auc" in result
        assert "test_auc" in result
        assert result["n_train"] > 0
        assert result["n_test"] > 0

        scored = scorer.score_detections(X.head(10))
        assert "probability_profitable" in scored.columns
        assert "confidence" in scored.columns
        assert "is_recommended" in scored.columns
        assert scored["confidence"].between(0, 1).all()

    def test_score_single(self) -> None:
        """score_single should return a PatternScore."""
        from src.ml.pattern_scorer import PatternScorer, PatternScore

        X = pd.DataFrame(
            np.random.randn(200, 5),
            columns=[f"feat_{i}" for i in range(5)],
        )
        y = pd.Series(np.random.randint(0, 2, 200))

        scorer = PatternScorer(n_estimators=30, max_depth=3)
        scorer.train(X, y)

        score = scorer.score_single(X.iloc[0], pattern_name="doji", direction="bullish")

        assert isinstance(score, PatternScore)
        assert 0.0 <= score.probability_profitable <= 1.0
        assert 0.0 <= score.confidence <= 1.0
        assert isinstance(score.is_recommended, bool)
        assert score.pattern_name == "doji"
        assert score.direction == "bullish"

    def test_score_batch(self) -> None:
        """score_batch should return ScoringResult."""
        from src.ml.pattern_scorer import PatternScorer, ScoringResult

        X = pd.DataFrame(
            np.random.randn(200, 6),
            columns=[f"feat_{i}" for i in range(6)],
        )
        y = pd.Series(np.random.randint(0, 2, 200))

        scorer = PatternScorer(n_estimators=30, max_depth=3)
        scorer.train(X, y)

        result = scorer.score_batch(X.head(20))
        assert isinstance(result, ScoringResult)
        assert len(result.pattern_scores) == 20
        assert 0.0 <= result.acceptance_rate <= 1.0
        assert 0.0 <= result.mean_probability <= 1.0
        assert 0.0 <= result.mean_confidence <= 1.0

    def test_feature_importance(self) -> None:
        """get_feature_importance should return sorted DataFrame."""
        from src.ml.pattern_scorer import PatternScorer

        X = pd.DataFrame(
            np.random.randn(200, 10),
            columns=[f"feat_{i}" for i in range(10)],
        )
        y = pd.Series(np.random.randint(0, 2, 200))

        scorer = PatternScorer(n_estimators=30, max_depth=3)
        scorer.train(X, y)

        imp = scorer.get_feature_importance(top_n=5)
        assert len(imp) == 5
        assert list(imp.columns) == ["feature", "importance"]

    def test_threshold_controls_recommendation(self) -> None:
        """Lower threshold → more recommendations, higher → fewer."""
        from src.ml.pattern_scorer import PatternScorer

        X = pd.DataFrame(
            np.random.randn(200, 4),
            columns=[f"feat_{i}" for i in range(4)],
        )
        y = pd.Series(np.random.randint(0, 2, 200))

        scorer_low = PatternScorer(n_estimators=30, max_depth=3, threshold=0.30)
        scorer_low.train(X, y)
        low_result = scorer_low.score_detections(X.head(50))

        scorer_high = PatternScorer(n_estimators=30, max_depth=3, threshold=0.70)
        scorer_high.train(X, y)
        high_result = scorer_high.score_detections(X.head(50))

        assert low_result["is_recommended"].mean() >= high_result["is_recommended"].mean()

    def test_untrained_raises(self) -> None:
        """Scoring without training should raise ValueError."""
        from src.ml.pattern_scorer import PatternScorer

        scorer = PatternScorer()
        X = pd.DataFrame(np.random.randn(10, 4))

        with pytest.raises(ValueError, match="not trained"):
            scorer.score_detections(X)
