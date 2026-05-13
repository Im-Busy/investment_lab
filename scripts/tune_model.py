"""Hyperparameter Tuning for ML Models.

Tunes PatternClassifier, SignalRegressor, or RegimeClassifier using
Optuna (TPE Bayesian), Grey Wolf Optimizer (GWO), Genetic Algorithm (GA),
or Whale Optimization (WOA).

Usage:
    uv run scripts/tune_model.py --symbol SPY --algo optuna --trials 50
    uv run scripts/tune_model.py --symbol SPY --wolves 20 --iterations 50
    uv run scripts/tune_model.py --symbol SPY --target signal_regressor --algo optuna --model lgbm
    uv run scripts/tune_model.py --symbol SPY --target regime --algo optuna --trials 30
    uv run scripts/tune_model.py --symbol SPY --algo compare --trials 30
    uv run scripts/tune_model.py --symbol SPY --algo ga --target regime_discovery
    uv run scripts/tune_model.py --symbol SPY --algo woa --target pattern_threshold
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.feature_engineering import FeatureExtractor
from src.ml.pattern_classifier import PatternClassifier
from src.ml.signal_scorer import SignalRegressor
from src.ml.regime_model import RegimeClassifier
from src.ml.tuning.gwo_tuner import CATBOOST_PARAM_SPACE, GWOTuner
from src.ml.tuning.woa_tuner import WOATuner, build_threshold_search_space
from src.ml.tuning.base import SearchSpace, map_params

AVAILABLE_STRATEGIES = ["rsi", "macd", "ema_ribbon", "keltner", "sma_crossover"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_CONFIGS = {
    "pattern_classifier": {
        "class": PatternClassifier,
        "model_type": "catboost",
        "description": "Pattern Classifier (CatBoost)",
    },
    "signal_regressor": {
        "class": SignalRegressor,
        "model_type": "catboost",
        "description": "Signal Regressor (CatBoost)",
    },
    "regime": {
        "class": RegimeClassifier,
        "model_type": "lightgbm",
        "description": "Regime Classifier (LightGBM)",
    },
    "regime_discovery": {
        "class": None,
        "model_type": None,
        "description": "Unsupervised Regime Discovery (GA + KMeans)",
    },
    "pattern_threshold": {
        "class": None,
        "model_type": None,
        "description": "Pattern Confidence Threshold Tuning (WOA)",
    },
    "strategy_params": {
        "class": None,
        "model_type": None,
        "description": "Strategy Parameter Optimization (Optuna + backtesting)",
    },
}


def load_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Load OHLCV data from Yahoo Finance."""
    import yfinance as yf

    logger.info(f"Loading {symbol} from {start} to {end}")
    df = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    logger.info(f"Loaded {len(df)} bars")
    return df


def prepare_training_data(
    df: pd.DataFrame,
    horizon: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract features and generate labels."""
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(df)

    future_returns = df["Close"].shift(-horizon) / df["Close"] - 1
    y = (future_returns > 0).astype(int)

    aligned = features.dropna().index.intersection(y.dropna().index)
    X = features.loc[aligned]
    y = y.loc[aligned]

    logger.info(
        f"Training data: {len(X)} samples, {X.shape[1]} features, positive ratio {y.mean():.1%}"
    )
    return X, y


def make_fitness_fn(
    X: pd.DataFrame,
    y: pd.Series,
    model_class: type,
    model_type: str,
    target_name: str,
    cv_splits: int = 3,
) -> Any:
    """Create a fitness function for GWO to optimize."""
    from sklearn.model_selection import TimeSeriesSplit

    def fitness(params: Dict[str, Any]) -> float:
        try:
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            scores = []

            # Map CatBoost-native names → target model kwargs
            mapped = map_params(params, target_name)

            for train_idx, test_idx in tscv.split(X):
                X_tr = X.iloc[train_idx]
                X_te = X.iloc[test_idx]
                y_tr = y.iloc[train_idx]
                y_te = y.iloc[test_idx]

                if len(np.unique(y_tr)) < 2:
                    continue

                model = model_class(
                    model_type=model_type,
                    n_estimators=150,
                    random_state=42,
                    **mapped,
                )
                result = model.train(X_tr, y_tr)
                scores.append(result.test_auc)

            return float(np.mean(scores)) if scores else 0.0
        except Exception as e:
            logger.debug(f"Fitness evaluation failed: {e}")
            return 0.0

    return fitness


def run_tuning(
    symbol: str = "SPY",
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    target: str = "pattern_classifier",
    n_wolves: int = 20,
    max_iter: int = 50,
    horizon: int = 5,
    output_dir: str = "reports/ml_tuning",
) -> Dict[str, Any]:
    """Run GWO hyperparameter tuning pipeline.

    Returns:
        Dict with best_params, best_score, convergence_history.
    """
    config = TARGET_CONFIGS[target]
    logger.info(f"Tuning target: {config['description']}")
    logger.info(f"Wolves: {n_wolves}, Max iterations: {max_iter}")

    df = load_data(symbol, start, end)
    X, y = prepare_training_data(df, horizon=horizon)

    fitness_fn = make_fitness_fn(X, y, config["class"], config["model_type"], target, cv_splits=3)

    space = SearchSpace(CATBOOST_PARAM_SPACE)
    tuner = GWOTuner(
        search_space=space,
        fitness_fn=fitness_fn,
        n_wolves=n_wolves,
        maximize=True,
    )

    result = tuner.optimize(max_iter=max_iter)

    logger.info("=" * 60)
    logger.info("Tuning Results")
    logger.info("=" * 60)
    logger.info(f"Best AUC: {result.best_score:.4f}")
    logger.info(f"Mean AUC: {result.mean_score:.4f} ± {result.score_std:.4f}")
    logger.info(f"Iterations: {result.n_iterations}")
    logger.info("Best hyperparameters:")
    for k, v in result.best_params.items():
        logger.info(f"  {k}: {v}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{target}"
    results = {
        "run_id": run_id,
        "symbol": symbol,
        "target": target,
        "best_score": result.best_score,
        "mean_score": result.mean_score,
        "score_std": result.score_std,
        "n_iterations": result.n_iterations,
        "best_params": result.best_params,
        "n_wolves": n_wolves,
        "max_iter": max_iter,
        "convergence_history": result.convergence_history,
    }

    results_path = output_path / f"gwo_tuning_{run_id}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {results_path}")
    return results


def run_ga_regime(
    symbol: str = "SPY",
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    population_size: int = 20,
    max_generations: int = 30,
    horizon: int = 5,
    output_dir: str = "reports/ml_tuning",
) -> Dict[str, Any]:
    """Run GA regime discovery pipeline."""
    from src.ml.tuning.ga_tuner import RegimeDiscovery

    logger.info("GA Regime Discovery: optimizing n_regimes via silhouette score")
    logger.info(f"Population: {population_size}, Generations: {max_generations}")

    df = load_data(symbol, start, end)
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(df)
    X = features.dropna()

    discovery = RegimeDiscovery(X)
    result = discovery.optimize(
        population_size=population_size,
        max_generations=max_generations,
    )

    logger.info("=" * 60)
    logger.info("GA Regime Discovery Results")
    logger.info("=" * 60)
    logger.info(f"Optimal n_regimes: {discovery.n_regimes_}")
    logger.info(f"Silhouette score: {result.best_score:.4f}")
    logger.info(f"Generations: {result.n_iterations}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_regime_discovery"
    results = {
        "run_id": run_id,
        "symbol": symbol,
        "target": "regime_discovery",
        "best_score": result.best_score,
        "n_regimes": discovery.n_regimes_,
        "silhouette": discovery.silhouette_,
        "best_params": result.best_params,
        "n_generations": result.n_iterations,
        "convergence_history": result.convergence_history,
    }

    results_path = output_path / f"ga_regime_{run_id}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {results_path}")
    return results


def run_woa_threshold(
    symbol: str = "SPY",
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    n_whales: int = 30,
    max_iter: int = 50,
    horizon: int = 5,
    output_dir: str = "reports/ml_tuning",
) -> Dict[str, Any]:
    """Run WOA pattern threshold tuning pipeline."""
    from src.ml.tuning.woa_tuner import DEFAULT_PATTERN_CATEGORIES

    logger.info("WOA Pattern Threshold Tuning: optimizing per-pattern confidence thresholds")
    logger.info(f"Whales: {n_whales}, Max iterations: {max_iter}")

    categories = DEFAULT_PATTERN_CATEGORIES

    df = load_data(symbol, start, end)
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(df)
    X = features.dropna()

    future_returns = df["Close"].shift(-horizon) / df["Close"] - 1
    y = (future_returns > 0).astype(int)
    aligned = X.index.intersection(y.dropna().index)
    X = X.loc[aligned]
    y = y.loc[aligned]

    def fitness_fn(params: Dict[str, float]) -> float:
        mean_threshold = float(np.mean(list(params.values())))
        n_signals = max(1, int(len(X) * (1.0 - mean_threshold)))
        selected = np.random.default_rng(42).choice(
            len(y), size=min(n_signals, len(y)), replace=False
        )
        if len(selected) < 10:
            return 0.0
        return float(y.iloc[selected].mean()) - 0.5 + mean_threshold * 0.1

    space = build_threshold_search_space(categories)
    tuner = WOATuner(
        search_space=space,
        fitness_fn=fitness_fn,
        n_whales=n_whales,
        maximize=True,
    )
    result = tuner.optimize(max_iter=max_iter)

    logger.info("=" * 60)
    logger.info("WOA Pattern Threshold Results")
    logger.info("=" * 60)
    logger.info(f"Best score: {result.best_score:.4f}")
    logger.info(f"Iterations: {result.n_iterations}")
    logger.info("Best per-pattern thresholds:")
    for k, v in sorted(result.best_params.items()):
        logger.info(f"  {k}: {v:.3f}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_pattern_threshold"
    results = {
        "run_id": run_id,
        "symbol": symbol,
        "target": "pattern_threshold",
        "algo": "woa",
        "best_score": result.best_score,
        "best_params": result.best_params,
        "n_iterations": result.n_iterations,
        "convergence_history": result.convergence_history,
    }

    results_path = output_path / f"woa_threshold_{run_id}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {results_path}")
    return results


def run_optuna(
    symbol: str = "SPY",
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    target: str = "pattern_classifier",
    model_type: str = "catboost",
    n_trials: int = 50,
    n_splits: int = 5,
    label_span: int = 5,
    prune: bool = False,
    pruner_patience: int = 10,
    study_name: str = "",
    output_dir: str = "reports/ml_tuning",
) -> Dict[str, Any]:
    """Run Optuna TPE hyperparameter tuning."""
    from src.ml.tuning.optuna_tuner import OptunaTuner

    config = TARGET_CONFIGS[target]
    logger.info(f"Optuna tuning: {config['description']}")
    logger.info(f"Model: {model_type}, Trials: {n_trials}, CV folds: {n_splits}")
    logger.info(f"Pruner: {'MedianPruner' if prune else 'None'}, Patience: {pruner_patience}")

    df = load_data(symbol, start, end)
    X, y = prepare_training_data(df, horizon=label_span)

    if len(X) < 100:
        logger.error(f"Only {len(X)} training samples — need at least 100")
        sys.exit(1)

    tuner = OptunaTuner(
        model_type=model_type,
        n_trials=n_trials,
        n_splits=n_splits,
        label_span=label_span,
        pruner_patience=pruner_patience if prune else 0,
        study_name=study_name,
    )
    result = tuner.optimize(X, y, target=target, model_class=config["class"])

    logger.info("=" * 60)
    logger.info("Optuna Results")
    logger.info("=" * 60)
    logger.info(f"Best AUC: {result.best_score:.4f}")
    logger.info(f"Trials: {result.n_trials}")
    logger.info(f"Best trial: #{result.best_trial_number}")
    logger.info("Best hyperparameters:")
    for k, v in sorted(result.best_params.items()):
        logger.info(f"  {k}: {v}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{target}_optuna"
    results_data = {
        "run_id": run_id,
        "symbol": symbol,
        "target": target,
        "algo": "optuna",
        "model_type": model_type,
        "best_score": result.best_score,
        "best_params": result.best_params,
        "n_trials": result.n_trials,
        "best_trial": result.best_trial_number,
        "trial_scores": result.trial_scores,
        "study_name": result.study_name,
    }

    results_path = output_path / f"optuna_tuning_{run_id}.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    logger.info(f"Results saved to {results_path}")

    try:
        tuner.log_to_mlflow(result)
    except Exception:
        logger.debug("MLflow logging skipped")

    return results_data


def run_compare(
    symbol: str = "SPY",
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    target: str = "pattern_classifier",
    model_type: str = "catboost",
    n_trials: int = 30,
    label_span: int = 5,
    output_dir: str = "reports/ml_tuning",
) -> Dict[str, Any]:
    """Compare Optuna TPE vs GWO on same data."""
    from src.ml.tuning.optuna_tuner import compare_optimizers

    logger.info(f"Comparing Optuna TPE vs GWO on {target}")
    logger.info(f"Model: {model_type}, Trials/Iterations: {n_trials}")

    df = load_data(symbol, start, end)
    X, y = prepare_training_data(df, horizon=label_span)

    results_data = compare_optimizers(
        X,
        y,
        target=target,
        model_type=model_type,
        n_trials=n_trials,
        label_span=label_span,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{target}_compare"
    results_path = output_path / f"compare_{run_id}.json"
    with open(results_path, "w") as f:
        json.dump(
            {k: v for k, v in results_data.items() if k not in ("gwo", "optuna") or True},
            f,
            indent=2,
            default=str,
        )

    logger.info(f"Comparison saved to {results_path}")
    return results_data


def run_strategy_tuning(
    symbol: str = "SPY",
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    strategy: str = "rsi",
    n_trials: int = 50,
    metric: str = "sharpe_ratio",
    output_dir: str = "reports/ml_tuning",
) -> Dict[str, Any]:
    """Run Optuna strategy parameter tuning."""
    from src.ml.tuning.optuna_strategy_tuner import OptunaStrategyTuner

    logger.info(f"Optuna strategy tuning: {strategy} (metric={metric}, trials={n_trials})")

    df = load_data(symbol, start, end)
    if len(df) < 100:
        logger.error(f"Only {len(df)} bars — need at least 100")
        sys.exit(1)

    tuner = OptunaStrategyTuner(
        strategy=strategy,
        n_trials=n_trials,
        metric=metric,
    )
    result = tuner.optimize(df)

    logger.info("=" * 60)
    logger.info(f"Strategy Tuning Results — {strategy}")
    logger.info("=" * 60)
    logger.info(f"Best {metric}: {result.best_score:.4f}")
    logger.info(f"Trials: {result.n_trials}")
    logger.info("Best parameters:")
    for k, v in sorted(result.best_params.items()):
        logger.info(f"  {k}: {v}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_strategy_{strategy}"
    results_data = {
        "run_id": run_id,
        "symbol": symbol,
        "target": "strategy_params",
        "strategy": strategy,
        "algo": "optuna",
        "best_score": result.best_score,
        "metric": metric,
        "best_params": result.best_params,
        "n_trials": result.n_trials,
        "trial_scores": result.trial_scores,
    }

    results_path = output_path / f"strategy_tune_{run_id}.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    logger.info(f"Results saved to {results_path}")
    return results_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning (Optuna/GWO/GA/WOA)")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument(
        "--target",
        default="pattern_classifier",
        choices=list(TARGET_CONFIGS),
    )
    parser.add_argument(
        "--algo",
        default="gwo",
        choices=["gwo", "ga", "woa", "optuna", "compare"],
        help="Algorithm: optuna (TPE Bayesian), gwo (Grey Wolf), ga (Genetic), woa (Whale), compare (Optuna vs GWO)",
    )
    parser.add_argument("--wolves", type=int, default=20, help="Population size (GWO/WOA)")
    parser.add_argument("--population", type=int, default=20, help="Population size (GA)")
    parser.add_argument(
        "--iterations", type=int, default=50, help="Max iterations/generations (GWO/GA/WOA)"
    )
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument(
        "--horizon", type=int, default=5, help="Forward return horizon (label_span)"
    )
    parser.add_argument(
        "--model",
        default="catboost",
        choices=["catboost", "lightgbm"],
        help="Model type for Optuna",
    )
    parser.add_argument("--prune", action="store_true", help="Enable Optuna MedianPruner")
    parser.add_argument("--patience", type=int, default=10, help="Pruner patience steps")
    parser.add_argument("--study-name", default="", help="Resume or name Optuna study")
    parser.add_argument(
        "--strategy",
        default="rsi",
        choices=AVAILABLE_STRATEGIES,
        help="Strategy name for strategy_params target",
    )
    parser.add_argument("--output-dir", default="reports/ml_tuning")

    args = parser.parse_args()

    if args.algo == "optuna":
        if args.target == "strategy_params":
            run_strategy_tuning(
                symbol=args.symbol,
                start=args.start,
                end=args.end,
                strategy=args.strategy,
                n_trials=args.trials,
                output_dir=args.output_dir,
            )
        elif args.target in ("regime_discovery", "pattern_threshold"):
            parser.error(
                f"Target '{args.target}' requires --algo ga (regime_discovery) "
                f"or --algo woa (pattern_threshold)"
            )
        else:
            run_optuna(
                symbol=args.symbol,
                start=args.start,
                end=args.end,
                target=args.target,
                model_type=args.model,
                n_trials=args.trials,
                label_span=args.horizon,
                prune=args.prune,
                pruner_patience=args.patience,
                study_name=args.study_name,
                output_dir=args.output_dir,
            )
    elif args.algo == "compare":
        if args.target in ("regime_discovery", "pattern_threshold"):
            parser.error(
                f"Compare not supported for '{args.target}'. Use pattern_classifier, "
                f"signal_regressor, or regime."
            )
        run_compare(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            target=args.target,
            model_type=args.model,
            n_trials=args.trials,
            label_span=args.horizon,
            output_dir=args.output_dir,
        )
    elif args.algo == "ga" and args.target == "regime_discovery":
        run_ga_regime(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            population_size=args.population,
            max_generations=args.iterations,
            horizon=args.horizon,
            output_dir=args.output_dir,
        )
    elif args.algo == "woa" and args.target == "pattern_threshold":
        run_woa_threshold(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            n_whales=args.wolves,
            max_iter=args.iterations,
            horizon=args.horizon,
            output_dir=args.output_dir,
        )
    else:
        if args.target in ("regime_discovery", "pattern_threshold"):
            parser.error(
                f"Target '{args.target}' requires --algo ga (regime_discovery) "
                f"or --algo woa (pattern_threshold)"
            )
        run_tuning(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            target=args.target,
            n_wolves=args.wolves,
            max_iter=args.iterations,
            horizon=args.horizon,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
