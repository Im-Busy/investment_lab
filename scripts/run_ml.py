"""
ML Model Selector CLI

Command-line interface for automated model selection and evaluation.

Usage:
    uv run python scripts/run_ml.py --auto
    uv run python scripts/run_ml.py --recommend
    uv run python scripts/run_ml.py --compare --export json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.model_selector import (
    ModelSelector,
    Recommendation,
    ModelResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load data from file or generate sample data.

    Args:
        data_path: Path to data file or preset name (e.g., "SPY")

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    path = Path(data_path)

    if path.exists():
        logger.info(f"Loading data from {path}")

        if path.suffix == ".csv":
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        elif path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        if "target" in df.columns:
            y = df["target"]
            X = df.drop(columns=["target"])
        else:
            y = df.iloc[:, -1]
            X = df.iloc[:, :-1]

        return X, y

    elif data_path.upper() == "SPY":
        logger.info("Generating sample SPY-like data for demo")
        np.random.seed(42)
        n_samples = 1000

        X = pd.DataFrame(
            {
                "feature_1": np.random.randn(n_samples),
                "feature_2": np.random.randn(n_samples),
                "feature_3": np.random.randn(n_samples),
                "feature_4": np.random.randn(n_samples),
                "feature_5": np.random.randn(n_samples),
            },
            index=pd.date_range(start="2020-01-01", periods=n_samples, freq="D"),
        )

        y = pd.Series(
            np.random.randint(0, 2, n_samples),
            index=X.index,
            name="target",
        )

        return X, y

    else:
        raise ValueError(
            f"Data path not found: {data_path}. Use valid file path or preset like 'SPY'"
        )


def print_recommendation(recommendation: Recommendation) -> None:
    """Print recommendation in formatted table.

    Args:
        recommendation: Recommendation object
    """
    print("\n" + "=" * 80)
    print("MODEL RECOMMENDATION")
    print("=" * 80)
    print(f"\nRecommended Model: {recommendation.model.display_name}")
    print(f"Recommended Validation: {recommendation.validation.display_name}")
    print(f"Reasoning: {recommendation.reasoning}")
    print(f"Estimated Training Time: {recommendation.estimated_time_seconds:.1f}s")
    print("=" * 80 + "\n")


def print_result(result: ModelResult, verbose: bool = False) -> None:
    """Print training result in formatted table.

    Args:
        result: ModelResult object
        verbose: Whether to show detailed metrics
    """
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)
    print(f"\nModel: {result.model_name}")
    print(f"Validation Method: {result.validation_method}")
    print(f"\nTraining Time: {result.training_time:.2f}s")
    print(f"\nTrain Score: {result.train_score:.4f}")
    print(f"Test Score: {result.test_score:.4f}")
    print(f"Overfit Gap: {result.overfit_gap:.4f}")

    if verbose and result.metrics:
        print("\nAdditional Metrics:")
        for key, value in result.metrics.items():
            print(f"  {key}: {value}")

    if verbose and result.feature_importance:
        print("\nTop 10 Feature Importance:")
        for i, (feat, imp) in enumerate(list(result.feature_importance.items())[:10], 1):
            print(f"  {i:2d}. {feat}: {imp:.4f}")

    print("=" * 80 + "\n")


def print_comparison(results: list[ModelResult]) -> None:
    """Print model comparison table.

    Args:
        results: List of ModelResult objects
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    print(f"\n{'Model':<20} {'Validation':<20} {'Train':<10} {'Test':<10} {'Gap':<10} {'Time':<10}")
    print("-" * 80)

    for result in results:
        print(
            f"{result.model_name:<20} "
            f"{result.validation_method:<20} "
            f"{result.train_score:<10.4f} "
            f"{result.test_score:<10.4f} "
            f"{result.overfit_gap:<10.4f} "
            f"{result.training_time:<10.2f}s"
        )

    print("=" * 80 + "\n")


def export_results(
    result: ModelResult,
    format: str,
    output_path: Path,
) -> None:
    """Export results to specified format.

    Args:
        result: ModelResult object
        format: Export format (json, csv, pickle)
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        data = {
            "model_name": result.model_name,
            "model_type": result.model_type,
            "validation_method": result.validation_method,
            "train_score": result.train_score,
            "test_score": result.test_score,
            "overfit_gap": result.overfit_gap,
            "training_time": result.training_time,
            "metrics": result.metrics,
            "feature_importance": result.feature_importance,
            "exported_at": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    elif format == "csv":
        if result.feature_importance:
            df = pd.DataFrame(
                list(result.feature_importance.items()),
                columns=["feature", "importance"],
            )
            df.to_csv(output_path, index=False)
        else:
            logger.warning("No feature importance to export")

    elif format == "pickle":
        import pickle

        with open(output_path, "wb") as f:
            pickle.dump(result, f)

    else:
        raise ValueError(f"Unknown export format: {format}")

    logger.info(f"Results exported to {output_path}")


def save_to_registry(
    result: ModelResult,
    name: str,
    registry_path: Path = Path("models/registry.json"),
) -> None:
    """Save model to model registry.

    Args:
        result: ModelResult object
        name: Model name for registry
        registry_path: Path to registry file
    """
    try:
        from src.ml.registry import ModelRegistry

        registry = ModelRegistry(registry_path.parent)

        metrics = {
            "train_score": result.train_score,
            "test_score": result.test_score,
            "overfit_gap": result.overfit_gap,
        }

        metadata = {
            "model_type": result.model_type,
            "validation_method": result.validation_method,
            "training_time": result.training_time,
        }

        version = registry.register(
            name=name,
            model=result.model,
            metrics=metrics,
            metadata=metadata,
        )

        logger.info(f"Model saved to registry as '{name}' v{version}")

    except ImportError:
        logger.warning("ModelRegistry not available, skipping registry save")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ML Model Selector - Automated model selection and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/run_ml.py --auto
  uv run python scripts/run_ml.py --recommend
  uv run python scripts/run_ml.py --compare
  uv run python scripts/run_ml.py --auto --data SPY
  uv run python scripts/run_ml.py --auto --priority accurate
  uv run python scripts/run_ml.py --auto --export json
  uv run python scripts/run_ml.py --auto --registry
  uv run python scripts/run_ml.py --auto --model lightgbm --validation walk_forward
        """,
    )

    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-select and train best model",
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Get recommendation only (no training)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all suitable models",
    )
    parser.add_argument(
        "--data",
        default="SPY",
        help="Data source (CSV, parquet, or preset like 'SPY')",
    )
    parser.add_argument(
        "--priority",
        default="balanced",
        choices=["balanced", "fast", "accurate", "interpretable"],
        help="Selection priority mode",
    )
    parser.add_argument(
        "--model",
        choices=[
            "lightgbm",
            "xgboost",
            "random_forest",
            "gradient_boosting",
            "logistic_regression",
        ],
        help="Force specific model type",
    )
    parser.add_argument(
        "--validation",
        choices=["train_test", "walk_forward", "purged_kfold"],
        help="Force specific validation type",
    )
    parser.add_argument(
        "--export",
        choices=["json", "csv", "pickle"],
        help="Export results to specified format",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for exported results",
    )
    parser.add_argument(
        "--registry",
        action="store_true",
        help="Save model to model registry",
    )
    parser.add_argument(
        "--registry-name",
        default="ml_model",
        help="Model name for registry",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--task",
        default="classification",
        choices=["classification", "regression"],
        help="Task type (classification or regression)",
    )

    args = parser.parse_args()

    if not any([args.auto, args.recommend, args.compare]):
        parser.print_help()
        print("\nError: Please specify --auto, --recommend, or --compare")
        sys.exit(1)

    try:
        logger.info("=" * 80)
        logger.info("ML Model Selector")
        logger.info("=" * 80)

        X, y = load_data(args.data)
        logger.info(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Task type: {args.task}")

        selector = ModelSelector()

        if args.recommend or args.auto:
            recommendation = selector.recommend(
                X=X,
                y=y,
                task=args.task,
                priority=args.priority,
                model_type=args.model,
                validation_type=args.validation,
                is_time_series=X.index.is_monotonic_increasing,
            )

            print_recommendation(recommendation)

            if args.recommend:
                return

            logger.info("Training model...")
            result = selector.train_and_evaluate(
                X=X,
                y=y,
                model_config=recommendation.model,
                validation_config=recommendation.validation,
            )

            print_result(result, verbose=args.verbose)

            if args.export:
                output_path = (
                    Path(args.output)
                    if args.output
                    else Path(f"reports/ml_results_{datetime.now():%Y%m%d_%H%M%S}.{args.export}")
                )
                export_results(result, args.export, output_path)

            if args.registry:
                save_to_registry(result, args.registry_name)

        elif args.compare:
            logger.info("Comparing all suitable models...")
            results = selector.compare_models(
                X=X,
                y=y,
                task=args.task,
                validation_config=(
                    selector.VALIDATION_CONFIGS[args.validation] if args.validation else None
                ),
            )

            if not results:
                logger.error("No models were successfully trained")
                sys.exit(1)

            print_comparison(results)

            if args.export:
                output_path = (
                    Path(args.output)
                    if args.output
                    else Path(f"reports/ml_comparison_{datetime.now():%Y%m%d_%H%M%S}.{args.export}")
                )

                if args.export == "json":
                    comparison_data = [
                        {
                            "model_name": r.model_name,
                            "model_type": r.model_type,
                            "validation_method": r.validation_method,
                            "train_score": r.train_score,
                            "test_score": r.test_score,
                            "overfit_gap": r.overfit_gap,
                            "training_time": r.training_time,
                        }
                        for r in results
                    ]
                    with open(output_path, "w") as f:
                        json.dump(comparison_data, f, indent=2, default=str)

                logger.info(f"Comparison exported to {output_path}")

        logger.info("Done!")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
