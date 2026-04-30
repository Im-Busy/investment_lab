"""
ML Model Selector Gradio App

Interactive web UI using Gradio - 100% open-source, no subscription required.

Run: uv run python scripts/ml_selector_gradio.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.model_selector import (
    ModelSelector,
    Recommendation,
    ModelResult,
)


def load_sample_data() -> str:
    """Generate sample SPY-like data and return info string."""
    np.random.seed(42)
    n_samples = 1000

    X = pd.DataFrame(
        {
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
            "feature_4": np.random.randn(n_samples),
            "feature_5": np.random.randn(n_samples),
            "rsi_14": np.random.uniform(0, 100, n_samples),
            "macd_signal": np.random.randn(n_samples),
            "bb_width": np.random.uniform(0, 5, n_samples),
            "volume_ratio": np.random.exponential(1, n_samples),
            "price_momentum": np.random.randn(n_samples),
            "volatility": np.random.exponential(0.5, n_samples),
        },
        index=pd.date_range(start="2020-01-01", periods=n_samples, freq="D"),
    )

    y = pd.Series(
        np.random.randint(0, 2, n_samples),
        index=X.index,
        name="target",
    )

    return X, y, f"✅ Loaded {X.shape[0]} samples, {X.shape[1]} features"


def load_uploaded_file(file_path: str) -> str:
    """Load data from uploaded file."""
    if not file_path:
        return "⚠️ No file uploaded"

    path = Path(file_path)

    try:
        if path.suffix == ".csv":
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        elif path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            return f"⚠️ Unsupported file format: {path.suffix}"

        if "target" in df.columns:
            y = df["target"]
            X = df.drop(columns=["target"])
        else:
            y = df.iloc[:, -1]
            X = df.iloc[:, :-1]

        return X, y, f"✅ Loaded {X.shape[0]} samples, {X.shape[1]} features"

    except Exception as e:
        return f"⚠️ Error loading file: {str(e)}"


def get_recommendation_text(
    data_info: str,
    task: str,
    priority: str,
    model_type: str,
    validation_type: str,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
) -> str:
    """Get model and validation recommendation."""
    if X is None or y is None:
        return "⚠️ Please load data first"

    selector = ModelSelector()

    try:
        recommendation = selector.recommend(
            X=X,
            y=y,
            task=task,
            priority=priority,
            model_type=model_type if model_type != "auto" else None,
            validation_type=validation_type if validation_type != "auto" else None,
            is_time_series=X.index.is_monotonic_increasing,
        )

        text = f"""## 📋 Model Recommendation

**Recommended Model:** {recommendation.model.display_name}
**Model Type:** {recommendation.model.model_type}
**Validation Method:** {recommendation.validation.display_name}
**Reasoning:** {recommendation.reasoning}
**Estimated Training Time:** {recommendation.estimated_time_seconds:.1f}s

Model is ready to train! Click "🚀 Auto-Select Best Model" to train this model.
"""
        return recommendation, text

    except Exception as e:
        return None, f"⚠️ Error getting recommendation: {str(e)}"


def train_model_fn(
    recommendation: Recommendation | None,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
) -> str:
    """Train model with selected configuration."""
    if recommendation is None:
        return "⚠️ Please get a recommendation first"

    if X is None or y is None:
        return "⚠️ Data not loaded"

    selector = ModelSelector()

    try:
        result = selector.train_and_evaluate(
            X=X,
            y=y,
            model_config=recommendation.model,
            validation_config=recommendation.validation,
        )

        metrics_text = ""
        if result.metrics:
            metrics_text = "\n\n**Additional Metrics:**\n"
            for key, value in result.metrics.items():
                metrics_text += f"- {key}: {value}\n"

        text = f"""## 📊 Training Results

**Model:** {result.model_name}
**Validation Method:** {result.validation_method}

**Training Time:** {result.training_time:.2f}s

### Performance Metrics
- **Train Score:** {result.train_score:.4f}
- **Test Score:** {result.test_score:.4f}
- **Overfit Gap:** {result.overfit_gap:.4f}
{metrics_text}

{'✅ Model training completed successfully!' if result.test_score > 0.5 else '⚠️ Consider adjusting model parameters'}
"""

        if result.feature_importance:
            importance_text = "\n\n### Top 10 Feature Importance\n"
            for i, (feat, imp) in enumerate(
                list(result.feature_importance.items())[:10], 1
            ):
                importance_text += f"{i:2d}. {feat}: {imp:.4f}\n"
            text += importance_text

        return result, text

    except Exception as e:
        return None, f"⚠️ Error training model: {str(e)}"


def compare_models_fn(
    data_info: str,
    task: str,
    validation_type: str,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
) -> str:
    """Compare all suitable models."""
    if X is None or y is None:
        return None, "⚠️ Please load data first"

    selector = ModelSelector()

    try:
        validation_config = (
            selector.VALIDATION_CONFIGS[validation_type]
            if validation_type != "auto"
            else None
        )

        results = selector.compare_models(
            X=X,
            y=y,
            task=task,
            validation_config=validation_config,
        )

        if not results:
            return None, "⚠️ No models were successfully trained"

        comparison_text = "## 🔬 Model Comparison\n\n"

        comparison_text += "| Model | Validation | Train | Test | Gap | Time |\n"
        comparison_text += "|-------|------------|-------|------|-----|------|\n"

        for result in results:
            comparison_text += (
                f"| {result.model_name} | {result.validation_method} | "
                f"{result.train_score:.4f} | { {result.test_score:.4f}} | "
                f"{result.overfit_gap:.4f} | {result.training_time:.2f}s |\n"
            )

        if results:
            best = results[0]
            comparison_text += (
                f"\n🏆 **Best Model:** {best.model_name} "
                f"(Test Score: {best.test_score:.4f})"
            )

        return results, comparison_text

    except Exception as e:
        return None, f"⚠️ Error comparing models: {str(e)}"


def export_results_json(result: ModelResult | None) -> str:
    """Export results to JSON."""
    if result is None:
        return "⚠️ No results to export"

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

    return json.dumps(data, indent=2, default=str)


def export_comparison_json(
    results: list[ModelResult] | None,
) -> str:
    """Export comparison results to JSON."""
    if results is None:
        return "⚠️ No results to export"

    data = [
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

    return json.dumps(data, indent=2, default=str)


def create_ui() -> gr.Blocks:
    """Create Gradio UI."""
    with gr.Blocks(title="ML Model Selector - Gradio", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 ML Model Selector")
        gr.Markdown(
            "Automated model selection and evaluation for trading strategies "
            "- Powered by Gradio (100% open-source)"
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📁 Data Source")

        with gr.Row():
            with gr.Column(scale=1):
                load_sample_btn = gr.Button("Load Sample Data", variant="primary")
                data_info = gr.Textbox(
                    label="Data Status",
                    value="⚠️ No data loaded",
                    interactive=False,
                )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Or Upload Your Data")
                csv_upload = gr.File(label="Upload CSV", file_types=[".csv"])
                parquet_upload = gr.File(label="Upload Parquet", file_types=[".parquet"])

        gr.Divider()

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🎯 Configuration")

                task_dropdown = gr.Dropdown(
                    choices=["classification", "regression"],
                    value="classification",
                    label="Task Type",
                )

                priority_dropdown = gr.Dropdown(
                    choices=["balanced", "fast", "accurate", "interpretable"],
                    value="balanced",
                    label="Priority Mode",
                )

                with gr.Accordion("🔧 Advanced Options", open=False):
                    model_dropdown = gr.Dropdown(
                        choices=[
                            "auto",
                            "lightgbm",
                            "xgboost",
                            "random_forest",
                            "gradient_boosting",
                            "logistic_regression",
                        ],
                        value="auto",
                        label="Force Model Type",
                    )

                    validation_dropdown = gr.Dropdown(
                        choices=["auto", "train_test", "walk_forward", "purged_kfold"],
                        value="auto",
                        label="Force Validation Type",
                    )

        gr.Divider()

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🚀 Actions")

        with gr.Row():
            recommend_btn = gr.Button(
                "📋 Get Recommendation",
                variant="secondary",
                size="lg",
            )
            train_btn = gr.Button(
                "🚀 Auto-Select Best Model",
                variant="primary",
                size="lg",
            )
            compare_btn = gr.Button(
                "🔬 Compare All Models",
                variant="secondary",
                size="lg",
            )

        gr.Divider()

        with gr.Row():
            with gr.Column(scale=2):
                recommendation_output = gr.Markdown(label="Recommendation Results")

        with gr.Row():
            with gr.Column(scale=2):
                training_output = gr.Markdown(label="Training Results")

        with gr.Row():
            with gr.Column(scale=2):
                comparison_output = gr.Markdown(label="Comparison Results")

        gr.Divider()

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 💾 Export")

        with gr.Row():
            export_json_btn = gr.Button("Export Results as JSON")
            export_json_output = gr.Textbox(
                label="JSON Output",
                interactive=False,
                lines=10,
            )

        gr.Markdown(
            """
            ---
            **Tips:**
            - Load data first (sample or upload)
            - Choose task type and priority
            - Get recommendation to see suggested model
            - Train to evaluate the model
            - Compare all models to find best performer
            - Export results for further analysis
            """
        )

        demo.load(
            load_sample_data,
            inputs=[],
            outputs=[data_info],
        )

        load_sample_btn.click(
            load_sample_data,
            inputs=[],
            outputs=[data_info],
        )

        csv_upload.change(
            load_uploaded_file,
            inputs=[csv_upload],
            outputs=[data_info],
        )

        parquet_upload.change(
            load_uploaded_file,
            inputs=[parquet_upload],
            outputs=[data_info],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()

    demo.launch(
        server_name="0.0.0.0",
        share=False,
        show_error=True,
        show_api=True,
    )
