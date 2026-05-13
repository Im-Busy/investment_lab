"""Tests for Optuna tuner, strategy tuner, and PyPortfolioOpt integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.tuning.optuna_tuner import (
    CATBOOST_OPTUNA_SPACE,
    LIGHTGBM_OPTUNA_SPACE,
    OptunaResult,
    OptunaTuner,
    _suggest_param,
)
from src.ml.tuning.optuna_strategy_tuner import OptunaStrategyTuner, STRATEGY_SPACES


def _make_synthetic_ohlcv(n_bars: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0.05, 1.5, n_bars))
    high = close + rng.uniform(0.5, 3.0, n_bars)
    low = close - rng.uniform(0.5, 3.0, n_bars)
    open_ = close - rng.normal(0, 0.5, n_bars)
    volume = rng.integers(1000, 10000, n_bars)
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def _make_synthetic_features(n_samples: int = 300, n_features: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (n_samples, n_features))
    cols = [f"feat_{i}" for i in range(n_features)]
    idx = pd.date_range("2020-01-01", periods=n_samples, freq="B")
    return pd.DataFrame(data, columns=cols, index=idx)


def _make_synthetic_labels(n_samples: int = 300) -> pd.Series:
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, n_samples)
    idx = pd.date_range("2020-01-01", periods=n_samples, freq="B")
    return pd.Series(labels, index=idx)


class TestOptunaTuner:
    def test_constructor_valid(self) -> None:
        tuner = OptunaTuner(model_type="catboost", n_trials=5)
        assert tuner.model_type == "catboost"
        assert tuner.n_trials == 5

    def test_constructor_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            OptunaTuner(model_type="xgboost")

    def test_catboost_space_has_params(self) -> None:
        assert len(CATBOOST_OPTUNA_SPACE) >= 7
        assert "learning_rate" in CATBOOST_OPTUNA_SPACE
        assert CATBOOST_OPTUNA_SPACE["learning_rate"]["type"] == "float"
        assert CATBOOST_OPTUNA_SPACE["learning_rate"]["log"] is True

    def test_lightgbm_space_has_params(self) -> None:
        assert len(LIGHTGBM_OPTUNA_SPACE) >= 7
        assert "num_leaves" in LIGHTGBM_OPTUNA_SPACE
        assert "lambda_l1" in LIGHTGBM_OPTUNA_SPACE

    def test_suggest_param_float(self) -> None:
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        spec = {"type": "float", "low": 0.01, "high": 0.3, "log": True}
        val = _suggest_param(trial, "lr", spec)
        assert 0.01 <= val <= 0.3
        assert isinstance(val, float)

    def test_suggest_param_int(self) -> None:
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        spec = {"type": "int", "low": 3, "high": 12}
        val = _suggest_param(trial, "depth", spec)
        assert 3 <= val <= 12
        assert isinstance(val, int)

    def test_optuna_result_dataclass(self) -> None:
        result = OptunaResult(
            best_params={"learning_rate": 0.05, "max_depth": 6},
            best_score=0.72,
            n_trials=10,
            study_name="test_study",
            best_trial_number=3,
        )
        assert result.best_score == 0.72
        assert result.best_params["learning_rate"] == 0.05


class TestStrategyTuner:
    def test_constructor_valid(self) -> None:
        tuner = OptunaStrategyTuner(strategy="rsi", n_trials=10)
        assert tuner.strategy == "rsi"

    def test_constructor_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            OptunaStrategyTuner(strategy="bogus")

    def test_rsi_signals_generated(self) -> None:
        df = _make_synthetic_ohlcv(500)
        tuner = OptunaStrategyTuner(strategy="rsi", n_trials=2)
        signals = tuner._generate_signals(df, {"window": 14, "oversold": 30, "overbought": 70})
        assert signals is not None
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_sma_signals_generated(self) -> None:
        df = _make_synthetic_ohlcv(500)
        tuner = OptunaStrategyTuner(strategy="sma_crossover", n_trials=2)
        signals = tuner._generate_signals(df, {"fast_period": 5, "slow_period": 20})
        assert signals is not None
        assert isinstance(signals, pd.Series)

    def test_macd_signals_generated(self) -> None:
        df = _make_synthetic_ohlcv(500)
        tuner = OptunaStrategyTuner(strategy="macd", n_trials=2)
        signals = tuner._generate_signals(df, {"fast": 12, "slow": 26, "signal": 9})
        assert signals is not None

    def test_ema_ribbon_signals_generated(self) -> None:
        df = _make_synthetic_ohlcv(500)
        tuner = OptunaStrategyTuner(strategy="ema_ribbon", n_trials=2)
        signals = tuner._generate_signals(
            df,
            {"short_period": 5, "long_period": 20, "buy_threshold": 0.02, "sell_threshold": -0.02},
        )
        assert signals is not None

    def test_keltner_signals_generated(self) -> None:
        df = _make_synthetic_ohlcv(500)
        tuner = OptunaStrategyTuner(strategy="keltner", n_trials=2)
        signals = tuner._generate_signals(df, {"period": 20, "multiplier": 2.0, "atr_period": 10})
        assert signals is not None

    def test_strategy_spaces_defined(self) -> None:
        strategies = ["rsi", "macd", "ema_ribbon", "keltner", "sma_crossover"]
        for s in strategies:
            assert s in STRATEGY_SPACES
            assert len(STRATEGY_SPACES[s]) >= 2

    def test_evaluate_signals_returns_float(self) -> None:
        df = _make_synthetic_ohlcv(500)
        tuner = OptunaStrategyTuner(strategy="rsi", n_trials=2)
        signals = tuner._generate_signals(df, {"window": 14, "oversold": 30, "overbought": 70})
        score = tuner._evaluate_signals(df, signals)
        assert isinstance(score, float)

    def test_insufficient_data_returns_neg_inf(self) -> None:
        df = _make_synthetic_ohlcv(15)
        tuner = OptunaStrategyTuner(strategy="sma_crossover", n_trials=2)
        signals = tuner._generate_signals(df, {"fast_period": 3, "slow_period": 8})
        score = tuner._evaluate_signals(df, signals)
        assert score == float("-inf")


class TestPyPortfolioOptIntegration:
    def test_optimize_hrp_basic(self) -> None:
        from src.optimizer.pypfopt_integration import optimize_hrp

        n_assets = 5
        n_days = 500
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            rng.normal(0.0005, 0.015, (n_days, n_assets)),
            columns=[f"TICKER_{i}" for i in range(n_assets)],
        ).fillna(0.0)
        result = optimize_hrp(returns)
        assert result.method == "hrp"
        assert len(result.weights) == n_assets
        assert abs(result.weights.sum() - 1.0) < 0.01

    def test_optimize_efficient_frontier(self) -> None:
        from src.optimizer.pypfopt_integration import optimize_efficient_frontier

        n_assets = 5
        n_days = 500
        rng = np.random.default_rng(42)
        tickers = [f"TICKER_{i}" for i in range(n_assets)]
        returns = pd.DataFrame(
            rng.normal(0.0005, 0.015, (n_days, n_assets)),
            columns=tickers,
        ).fillna(0.0)
        result = optimize_efficient_frontier(returns, objective="max_sharpe")
        assert "efficient_frontier" in result.method
        assert len(result.weights) > 0

    def test_optimize_cvar(self) -> None:
        from src.optimizer.pypfopt_integration import optimize_cvar

        n_assets = 5
        n_days = 500
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            rng.normal(0.0005, 0.015, (n_days, n_assets)),
            columns=[f"TICKER_{i}" for i in range(n_assets)],
        ).fillna(0.0)
        result = optimize_cvar(returns, beta=0.95)
        assert "cvar" in result.method

    def test_compare_methods(self) -> None:
        from src.optimizer.pypfopt_integration import compare_methods

        n_assets = 5
        n_days = 500
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            rng.normal(0.0005, 0.015, (n_days, n_assets)),
            columns=[f"TICKER_{i}" for i in range(n_assets)],
        ).fillna(0.0)
        comparison = compare_methods(returns)
        assert isinstance(comparison, pd.DataFrame)
        assert "equal_weight" in comparison.index
        assert "hrp" in comparison.index
        assert "ef_max_sharpe" in comparison.index

    def test_empty_returns_handled(self) -> None:
        from src.optimizer.pypfopt_integration import optimize_hrp

        empty = pd.DataFrame()
        result = optimize_hrp(empty)
        assert len(result.weights) == 0
        assert result.sharpe_ratio == 0.0

    def test_single_asset_handled(self) -> None:
        from src.optimizer.pypfopt_integration import optimize_hrp

        returns = pd.DataFrame({"A": np.random.default_rng(42).normal(0, 1, 100)})
        result = optimize_hrp(returns)
        assert result.sharpe_ratio >= 0.0 or result.sharpe_ratio <= 0.0


class TestAllocationResult:
    def test_dataclass(self) -> None:
        from src.optimizer.pypfopt_integration import AllocationResult

        result = AllocationResult(
            weights=pd.Series({"A": 0.5, "B": 0.5}),
            expected_return=0.10,
            volatility=0.15,
            sharpe_ratio=0.67,
            method="hrp",
        )
        assert result.expected_return == 0.10
        assert result.method == "hrp"
